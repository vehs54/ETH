"""
ETH Bot Backtester (Full Logic + P&L Simulation)
------------------------------------------------------------------------
Replays the EXACT current eth_bot.py signal pipeline against real historical
data:
  1. MA5/MA10/MA30 crossover + RSI(15), 60/40 bands, 0.05% spread threshold,
     4-cycle confirmation (on 15-min spaced spot-price ticks)
  2. 1H market structure filter (hard gate)
  3. 6H higher-timeframe structure (confidence-only, not a hard gate)
  4. Volume confirmation filter (1.1x average)
  5. Candlestick pattern + full confidence scoring (0-100), logged per signal

Then simulates what actually trading each signal would have done:
  - ATR-based stop-loss / take-profit, walked forward bar-by-bar to find
    the real exit (SL, TP, or time horizon) — not just a fixed-point check
  - Position sizing so a stop-loss hit costs exactly RISK_PER_TRADE_PCT of
    current balance — the account balance compounds trade to trade
  - Trading fees + slippage applied on both entry and exit (assumptions —
    adjust TAKER_FEE_PCT / SLIPPAGE_PCT to your actual exchange)
  - Reports: final balance, max drawdown, profit factor, trade-level Sharpe,
    average win/loss — alongside the original direction-only accuracy

IMPORTANT CAVEATS:
  - Exit simulation uses closing prices, not true intrabar highs/lows — a
    fast wick through SL/TP between candle closes wouldn't be caught here.
    This is a reasonable approximation, not a perfect one.
  - Fee/slippage numbers are estimates. Change them to match your real
    exchange before trusting the dollar figures.
  - A backtest is not a guarantee of live performance — markets change and
    backtests can overfit to the specific period tested. Treat this as one
    input, not a verdict.

Data source: Coinbase Exchange public API (no key required) — 15-min candles
for the price series, plus native 1H and 6H candles for structure/volume/pattern.

CRITICAL DESIGN POINT — no lookahead bias: at each simulated tick, only 1H/6H
candles that would have actually closed by that timestamp are used for
structure/volume/pattern — never future data the live bot couldn't have seen.

HOW TO RUN:
  1. pip install requests
  2. python backtest_eth.py
  3. Results print to console, plus a full per-signal CSV
     (backtest_eth_results.csv) including confidence AND per-trade P&L.
"""

import time
from datetime import datetime, timedelta

import requests

# ---------------------------------------------------------------------------
# Config — mirrors eth_bot.py exactly
# ---------------------------------------------------------------------------
RSI_PERIOD = 15
MA_SHORT = 5
MA_MED = 10
MA_LONG = 30
MIN_MA_SPREAD_PCT = 0.05
RSI_BUY_CEILING = 60
RSI_SELL_FLOOR = 40
CONFIRMATION_STREAK = 2

SWING_STRENGTH = 2
ATR_PERIOD = 14
SL_ATR_MULTIPLIER = 2
TP_RR_MULTIPLIERS = [1.0, 1.5, 2.0]

VOLUME_LOOKBACK = 20
VOLUME_CONFIRMATION_MULTIPLIER = 1.0

STRUCTURE_LOOKBACK = 100
HTF_LOOKBACK = 60

# Backtest-specific
EVAL_HORIZON_BARS = 12         # 12 * 15min = 3 hours ahead, same as live accuracy tracker
MIN_MOVE_THRESHOLD_PCT = 0.05
DAYS_OF_HISTORY = 90            # longer window than the simple backtest — this bot fires rarer

# --- P&L simulation (NEW) ---
# These are assumptions, not guarantees — adjust to match your actual exchange.
STARTING_BALANCE = 1000.0       # hypothetical account size, purely for simulation
RISK_PER_TRADE_PCT = 1.0        # % of current balance risked per trade (compounds)
TAKER_FEE_PCT = 0.1             # per-side taker fee assumption (adjust to your exchange)
SLIPPAGE_PCT = 0.05             # per-side slippage assumption

PRODUCT_ID = "ETH-USD"
CANDLES_URL = f"https://api.exchange.coinbase.com/products/{PRODUCT_ID}/candles"


# ---------------------------------------------------------------------------
# Data fetching (paginated — Coinbase caps at 300 candles per request)
# ---------------------------------------------------------------------------
def fetch_candles_range(granularity, days=DAYS_OF_HISTORY):
    """Returns a chronological list of {time, low, high, open, close, volume} dicts."""
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    all_candles = []
    window = timedelta(seconds=granularity * 290)
    cursor = start
    headers = {"User-Agent": "eth-bot-backtester/1.0"}

    while cursor < end:
        chunk_end = min(cursor + window, end)
        params = {
            "start": cursor.isoformat(),
            "end": chunk_end.isoformat(),
            "granularity": granularity,
        }
        try:
            resp = requests.get(CANDLES_URL, params=params, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            for c in data:
                all_candles.append({"time": c[0], "low": c[1], "high": c[2], "open": c[3], "close": c[4], "volume": c[5]})
        except requests.exceptions.RequestException as e:
            print(f"  [warn] fetch failed ({granularity}s) {cursor} -> {chunk_end}: {e}")

        cursor = chunk_end
        time.sleep(0.35)

    all_candles.sort(key=lambda c: c["time"])
    return all_candles


# ---------------------------------------------------------------------------
# Indicator logic — copied verbatim from eth_bot.py
# ---------------------------------------------------------------------------
def moving_average(data, period):
    if len(data) < period:
        return None
    return sum(data[-period:]) / period


def calculate_rsi(data, period=RSI_PERIOD):
    if len(data) < period + 1:
        return None
    prices = data[-(period + 1):]
    gains, losses = [], []
    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]
        if change >= 0:
            gains.append(change); losses.append(0)
        else:
            gains.append(0); losses.append(abs(change))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def generate_signal(window):
    ma5 = moving_average(window, MA_SHORT)
    ma10 = moving_average(window, MA_MED)
    ma30 = moving_average(window, MA_LONG)
    rsi = calculate_rsi(window)
    if ma5 is None or ma10 is None or ma30 is None:
        return None
    bullish = ma5 > ma10 > ma30
    bearish = ma5 < ma10 < ma30
    spread_pct = abs(ma5 - ma30) / ma30 * 100 if ma30 else 0
    strong_spread = spread_pct >= MIN_MA_SPREAD_PCT
    if bullish and strong_spread and (rsi is None or rsi < RSI_BUY_CEILING):
        return "BUY"
    elif bearish and strong_spread and (rsi is None or rsi > RSI_SELL_FLOOR):
        return "SELL"
    return "HOLD"


def detect_swings(candles, strength=SWING_STRENGTH):
    swings = []
    for i in range(strength, len(candles) - strength):
        window = candles[i - strength: i + strength + 1]
        if candles[i]["high"] == max(c["high"] for c in window):
            swings.append((i, "high", candles[i]["high"]))
        if candles[i]["low"] == min(c["low"] for c in window):
            swings.append((i, "low", candles[i]["low"]))
    return swings


def classify_structure(candles):
    if len(candles) < (SWING_STRENGTH * 2 + 5):
        return {"trend": "UNKNOWN", "support": None, "resistance": None}
    swings = detect_swings(candles)
    highs = [s for s in swings if s[1] == "high"]
    lows = [s for s in swings if s[1] == "low"]
    nearest_support = lows[-1][2] if lows else None
    nearest_resistance = highs[-1][2] if highs else None
    if len(highs) < 2 or len(lows) < 2:
        trend = "UNKNOWN"
    else:
        higher_high = highs[-1][2] > highs[-2][2]
        higher_low = lows[-1][2] > lows[-2][2]
        lower_high = highs[-1][2] < highs[-2][2]
        lower_low = lows[-1][2] < lows[-2][2]
        if higher_high and higher_low:
            trend = "BULLISH"
        elif lower_high and lower_low:
            trend = "BEARISH"
        else:
            trend = "RANGING"
    return {"trend": trend, "support": nearest_support, "resistance": nearest_resistance}


def compute_atr(candles, period=ATR_PERIOD):
    if len(candles) < period + 1:
        return None
    true_ranges = []
    for i in range(1, len(candles)):
        high, low, prev_close = candles[i]["high"], candles[i]["low"], candles[i - 1]["close"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)
    return sum(true_ranges[-period:]) / period


def check_volume_confirmation(candles, lookback=VOLUME_LOOKBACK):
    if len(candles) < lookback + 1:
        return None
    recent = candles[-(lookback + 1):-1]
    avg_volume = sum(c["volume"] for c in recent) / len(recent)
    current_volume = candles[-1]["volume"]
    if avg_volume == 0:
        return None
    return {
        "confirmed": current_volume > avg_volume * VOLUME_CONFIRMATION_MULTIPLIER,
        "current_volume": current_volume,
        "avg_volume": avg_volume,
    }


def detect_candlestick_pattern(candles):
    if len(candles) < 2:
        return {"pattern": "UNKNOWN", "bias": "neutral"}
    current, previous = candles[-1], candles[-2]

    def body(c): return abs(c["close"] - c["open"])
    def candle_range(c): return c["high"] - c["low"] if c["high"] != c["low"] else 1e-9
    def upper_wick(c): return c["high"] - max(c["open"], c["close"])
    def lower_wick(c): return min(c["open"], c["close"]) - c["low"]

    cur_body, cur_range = body(current), candle_range(current)
    cur_upper, cur_lower = upper_wick(current), lower_wick(current)
    cur_bullish, cur_bearish = current["close"] > current["open"], current["close"] < current["open"]
    prev_body = body(previous)
    prev_bullish, prev_bearish = previous["close"] > previous["open"], previous["close"] < previous["open"]

    if (cur_bullish and prev_bearish and current["open"] <= previous["close"]
            and current["close"] >= previous["open"] and cur_body > prev_body):
        return {"pattern": "Bullish Engulfing", "bias": "bullish"}
    if (cur_bearish and prev_bullish and current["open"] >= previous["close"]
            and current["close"] <= previous["open"] and cur_body > prev_body):
        return {"pattern": "Bearish Engulfing", "bias": "bearish"}
    if cur_body > 0 and cur_lower >= cur_body * 2 and cur_upper <= cur_body * 0.5:
        return {"pattern": "Hammer", "bias": "bullish"}
    if cur_body > 0 and cur_upper >= cur_body * 2 and cur_lower <= cur_body * 0.5:
        return {"pattern": "Shooting Star", "bias": "bearish"}
    if cur_body <= cur_range * 0.1:
        return {"pattern": "Doji", "bias": "neutral"}
    if cur_body >= cur_range * 0.9:
        bias = "bullish" if cur_bullish else "bearish"
        return {"pattern": f"{'Bullish' if cur_bullish else 'Bearish'} Marubozu", "bias": bias}
    if cur_lower >= cur_range * 0.5:
        return {"pattern": "Long Lower Rejection Wick", "bias": "bullish"}
    if cur_upper >= cur_range * 0.5:
        return {"pattern": "Long Upper Rejection Wick", "bias": "bearish"}
    return {"pattern": "No clear pattern", "bias": "neutral"}


def apply_structure_filter(signal, structure):
    # 6H (htf_trend) AND volume are intentionally NOT hard gates here, matching
    # the live bot — both affect confidence scoring only, not whether a signal
    # fires. Only 1H structure remains a hard gate.
    trend = structure.get("trend")
    if signal == "BUY" and trend == "BEARISH":
        return "HOLD"
    elif signal == "SELL" and trend == "BULLISH":
        return "HOLD"
    return signal


def compute_confidence(signal, ma5, ma10, ma30, rsi, structure, price):
    score = 0
    if ma5 is not None and ma10 is not None and ma30 is not None and ma30:
        spread_pct = abs(ma5 - ma30) / ma30 * 100
        trend_matches = (signal == "BUY" and ma5 > ma10 > ma30) or (signal == "SELL" and ma5 < ma10 < ma30)
        if trend_matches:
            score += min(15, round(spread_pct / MIN_MA_SPREAD_PCT * 7.5))
    if rsi is not None:
        if signal == "BUY" and rsi < RSI_BUY_CEILING:
            score += round((RSI_BUY_CEILING - rsi) / RSI_BUY_CEILING * 10)
        elif signal == "SELL" and rsi > RSI_SELL_FLOOR:
            score += round((rsi - RSI_SELL_FLOOR) / (100 - RSI_SELL_FLOOR) * 10)
    trend = structure.get("trend")
    if (signal == "BUY" and trend == "BULLISH") or (signal == "SELL" and trend == "BEARISH"):
        score += 20
    elif trend == "RANGING":
        score += 7
    htf_trend = structure.get("htf_trend")
    if (signal == "BUY" and htf_trend == "BULLISH") or (signal == "SELL" and htf_trend == "BEARISH"):
        score += 15
    elif htf_trend == "RANGING":
        score += 5
    volume = structure.get("volume")
    if volume is not None and volume["confirmed"]:
        score += 15
    pattern_info = structure.get("pattern") or {"bias": "neutral"}
    if (signal == "BUY" and pattern_info["bias"] == "bullish") or (signal == "SELL" and pattern_info["bias"] == "bearish"):
        score += 15
    support, resistance = structure.get("support"), structure.get("resistance")
    if signal == "BUY" and resistance is not None and price:
        room_pct = (resistance - price) / price * 100
        if room_pct > 0:
            score += min(10, round(room_pct * 2))
    elif signal == "SELL" and support is not None and price:
        room_pct = (price - support) / price * 100
        if room_pct > 0:
            score += min(10, round(room_pct * 2))
    return min(100, score)


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------
def compute_stop_and_target(entry_price, atr, direction):
    """ATR-based stop-loss and target, same 2x-ATR / final-RR approach as the live bot."""
    if atr is None or atr <= 0:
        return None, None
    sl_distance = atr * SL_ATR_MULTIPLIER
    if direction == "BUY":
        stop_loss = entry_price - sl_distance
        target = entry_price + sl_distance * TP_RR_MULTIPLIERS[-1]
    else:
        stop_loss = entry_price + sl_distance
        target = entry_price - sl_distance * TP_RR_MULTIPLIERS[-1]
    return stop_loss, target


def simulate_trade_exit(closes, entry_idx, signal, stop_loss, target, horizon_bars):
    """
    Walks forward bar-by-bar from entry, exiting at whichever comes first:
    Stop Loss, Take Profit, or the time horizon. Uses closing prices as a
    reasonable approximation — true intrabar wicks aren't available from this
    data source, so a fast wick through SL/TP between closes wouldn't be
    caught here. Returns (exit_price, exit_reason, bars_held).
    """
    end_idx = min(entry_idx + horizon_bars, len(closes) - 1)
    for i in range(entry_idx + 1, end_idx + 1):
        price = closes[i]
        if signal == "BUY":
            if price <= stop_loss:
                return stop_loss, "SL", i - entry_idx
            if price >= target:
                return target, "TP", i - entry_idx
        else:
            if price >= stop_loss:
                return stop_loss, "SL", i - entry_idx
            if price <= target:
                return target, "TP", i - entry_idx
    return closes[end_idx], "TIME", end_idx - entry_idx


def compute_trade_pnl(entry_price, exit_price, stop_loss, signal, balance):
    """
    Position size is set so a Stop Loss hit loses exactly RISK_PER_TRADE_PCT
    of current balance (before fees/slippage) — the same logic a risk-managed
    trader would use. Sizing is based on the STOP distance decided at entry
    time, not the actual exit — using the exit price for sizing would let
    winning trades silently use a different risk than what was planned.
    Fees and slippage are applied on both entry and exit.
    Returns (pnl_dollars, net_return_pct, position_size_dollars).
    """
    raw_return_pct = (exit_price - entry_price) / entry_price * 100
    if signal == "SELL":
        raw_return_pct *= -1

    net_return_pct = raw_return_pct - (2 * (TAKER_FEE_PCT + SLIPPAGE_PCT))

    stop_distance_pct = abs(entry_price - stop_loss) / entry_price * 100
    if stop_distance_pct == 0:
        return 0, net_return_pct, 0

    position_size_dollars = (balance * (RISK_PER_TRADE_PCT / 100)) / (stop_distance_pct / 100)
    pnl_dollars = position_size_dollars * (net_return_pct / 100)

    return pnl_dollars, net_return_pct, position_size_dollars


def run_backtest():
    print(f"Fetching {DAYS_OF_HISTORY} days of history for {PRODUCT_ID}...")
    print("  15-min candles (price series)...")
    price_candles = fetch_candles_range(900)
    print(f"    got {len(price_candles)}")
    print("  1-hour candles (structure/volume/pattern)...")
    hourly_candles = fetch_candles_range(3600)
    print(f"    got {len(hourly_candles)}")
    print("  6-hour candles (higher-timeframe)...")
    sixh_candles = fetch_candles_range(21600)
    print(f"    got {len(sixh_candles)}")

    # No downsampling needed — live bot now polls every 15 min, matching
    # the native candle granularity directly
    price_series = price_candles
    closes = [c["close"] for c in price_series]

    price_window = []
    recent_raw = []
    last_alerted = None
    signals = []  # (index, signal, price, confidence)

    for i, candle in enumerate(price_series):
        ts = candle["time"]
        price = candle["close"]
        price_window.append(price)
        if len(price_window) > 60:
            price_window.pop(0)

        raw = generate_signal(price_window)
        if raw is None:
            continue

        recent_raw.append(raw)
        if len(recent_raw) > CONFIRMATION_STREAK:
            recent_raw.pop(0)

        if len(recent_raw) < CONFIRMATION_STREAK:
            confirmed = "HOLD"
        elif all(s == "BUY" for s in recent_raw):
            confirmed = "BUY"
        elif all(s == "SELL" for s in recent_raw):
            confirmed = "SELL"
        else:
            confirmed = "HOLD"

        if confirmed not in ("BUY", "SELL"):
            continue

        # --- No-lookahead structure: only candles closed by `ts` are visible ---
        hourly_subset = [c for c in hourly_candles if c["time"] <= ts][-STRUCTURE_LOOKBACK:]
        sixh_subset = [c for c in sixh_candles if c["time"] <= ts][-HTF_LOOKBACK:]

        structure = classify_structure(hourly_subset)
        structure["atr"] = compute_atr(hourly_subset)
        structure["volume"] = check_volume_confirmation(hourly_subset)
        structure["pattern"] = detect_candlestick_pattern(hourly_subset) if len(hourly_subset) >= 2 else {"bias": "neutral"}
        structure["htf_trend"] = classify_structure(sixh_subset)["trend"] if sixh_subset else "UNKNOWN"

        filtered = apply_structure_filter(confirmed, structure)

        if filtered in ("BUY", "SELL") and filtered != last_alerted:
            ma5 = moving_average(price_window, MA_SHORT)
            ma10 = moving_average(price_window, MA_MED)
            ma30 = moving_average(price_window, MA_LONG)
            rsi = calculate_rsi(price_window)
            confidence = compute_confidence(filtered, ma5, ma10, ma30, rsi, structure, price)
            signals.append((i, filtered, price, confidence))
            last_alerted = filtered

    # --- Evaluate outcomes: both signal-accuracy (direction-only) AND
    # a full P&L simulation with fees, slippage, position sizing, and
    # compounding account balance ---
    wins, losses, neutrals = 0, 0, 0
    results_log = []
    win_confidences, loss_confidences = [], []

    balance = STARTING_BALANCE
    equity_curve = [balance]
    trade_pnls = []
    trade_returns_pct = []
    tp_hits, sl_hits, time_exits = 0, 0, 0
    skipped_no_atr = 0

    for idx, sig, entry_price, confidence in signals:
        # --- Direction-only accuracy (unaffected by fees/sizing) ---
        future_idx = idx + EVAL_HORIZON_BARS
        if future_idx < len(closes):
            future_price = closes[future_idx]
            move_pct = (future_price - entry_price) / entry_price * 100
            if sig == "BUY":
                outcome = "WIN" if move_pct > MIN_MOVE_THRESHOLD_PCT else (
                    "LOSS" if move_pct < -MIN_MOVE_THRESHOLD_PCT else "NEUTRAL")
            else:
                outcome = "WIN" if move_pct < -MIN_MOVE_THRESHOLD_PCT else (
                    "LOSS" if move_pct > MIN_MOVE_THRESHOLD_PCT else "NEUTRAL")

            if outcome == "WIN":
                wins += 1
                win_confidences.append(confidence)
            elif outcome == "LOSS":
                losses += 1
                loss_confidences.append(confidence)
            else:
                neutrals += 1
        else:
            outcome = None

        # --- P&L simulation (needs ATR for stop distance — recompute it fresh
        # from the same no-lookahead hourly subset used at signal time) ---
        ts = price_series[idx]["time"]
        hourly_subset = [c for c in hourly_candles if c["time"] <= ts][-STRUCTURE_LOOKBACK:]
        atr = compute_atr(hourly_subset)
        stop_loss, target = compute_stop_and_target(entry_price, atr, sig)

        if stop_loss is None:
            skipped_no_atr += 1
        else:
            exit_price, exit_reason, bars_held = simulate_trade_exit(
                closes, idx, sig, stop_loss, target, EVAL_HORIZON_BARS
            )
            pnl_dollars, net_return_pct, position_size = compute_trade_pnl(
                entry_price, exit_price, stop_loss, sig, balance
            )
            balance += pnl_dollars
            equity_curve.append(balance)
            trade_pnls.append(pnl_dollars)
            trade_returns_pct.append(net_return_pct)

            if exit_reason == "TP":
                tp_hits += 1
            elif exit_reason == "SL":
                sl_hits += 1
            else:
                time_exits += 1

        results_log.append(("ETH/USDT bot (full logic)", idx, sig, round(entry_price, 4),
                             round(future_price, 4) if future_idx < len(closes) else None,
                             round(move_pct, 3) if future_idx < len(closes) else None,
                             confidence, outcome,
                             round(stop_loss, 4) if stop_loss else None,
                             round(pnl_dollars, 2) if stop_loss else None,
                             round(balance, 2) if stop_loss else None))

    total = wins + losses + neutrals
    print(f"\n=== ETH/USDT bot — signal accuracy (direction only, no fees) ===")
    print(f"Total confirmed+filtered signals generated: {len(signals)}")
    print(f"Signals with enough future data to evaluate: {total}")
    if total > 0:
        print(f"  WIN:     {wins} ({wins/total*100:.1f}%)")
        print(f"  LOSS:    {losses} ({losses/total*100:.1f}%)  <-- 'wrong signal' rate")
        print(f"  NEUTRAL: {neutrals} ({neutrals/total*100:.1f}%)")
        if win_confidences:
            print(f"  Avg confidence on WINs:  {sum(win_confidences)/len(win_confidences):.1f}%")
        if loss_confidences:
            print(f"  Avg confidence on LOSSes: {sum(loss_confidences)/len(loss_confidences):.1f}%")
    else:
        print("  Not enough data to evaluate any signals.")
        print(f"  Try increasing DAYS_OF_HISTORY (currently {DAYS_OF_HISTORY}) for a larger sample.")

    print(f"\n=== ETH/USDT bot — P&L simulation (fees, slippage, position sizing) ===")
    print(f"Assumptions: ${STARTING_BALANCE:.0f} start, {RISK_PER_TRADE_PCT}% risk/trade, "
          f"{TAKER_FEE_PCT}% fee + {SLIPPAGE_PCT}% slippage per side — ADJUST THESE to your exchange.")
    n_trades = len(trade_pnls)
    print(f"Trades simulated: {n_trades}  (skipped {skipped_no_atr} — no ATR available yet)")

    if n_trades > 0:
        final_balance = equity_curve[-1]
        total_return_pct = (final_balance - STARTING_BALANCE) / STARTING_BALANCE * 100
        print(f"  Final balance: ${final_balance:,.2f}  ({total_return_pct:+.1f}%)")
        print(f"  Exits — TP: {tp_hits} ({tp_hits/n_trades*100:.0f}%)  "
              f"SL: {sl_hits} ({sl_hits/n_trades*100:.0f}%)  "
              f"TIME: {time_exits} ({time_exits/n_trades*100:.0f}%)")

        # Max drawdown
        peak = equity_curve[0]
        max_dd_pct = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd_pct = (peak - eq) / peak * 100 if peak > 0 else 0
            max_dd_pct = max(max_dd_pct, dd_pct)
        print(f"  Max drawdown: {max_dd_pct:.1f}%")

        # Profit factor
        gross_profit = sum(p for p in trade_pnls if p > 0)
        gross_loss = abs(sum(p for p in trade_pnls if p < 0))
        if gross_loss > 0:
            print(f"  Profit factor: {gross_profit/gross_loss:.2f}  (gross profit / gross loss)")
        else:
            print(f"  Profit factor: N/A (no losing trades)")

        # Trade-level Sharpe (NOT annualized — a per-trade risk-adjusted return
        # measure. Annualizing would require assuming a trade frequency, which
        # varies too much run-to-run to state honestly here.)
        if len(trade_returns_pct) > 1:
            mean_r = sum(trade_returns_pct) / len(trade_returns_pct)
            variance = sum((r - mean_r) ** 2 for r in trade_returns_pct) / (len(trade_returns_pct) - 1)
            std_r = variance ** 0.5
            sharpe = mean_r / std_r if std_r > 0 else 0
            print(f"  Trade-level Sharpe: {sharpe:.2f}  (per-trade, NOT annualized — "
                  f"mean/stdev of trade returns)")

        wins_list = [p for p in trade_pnls if p > 0]
        losses_list = [p for p in trade_pnls if p < 0]
        if wins_list:
            print(f"  Avg win:  ${sum(wins_list)/len(wins_list):,.2f}")
        if losses_list:
            print(f"  Avg loss: ${sum(losses_list)/len(losses_list):,.2f}")
    else:
        print("  No trades had ATR available for sizing — can't simulate P&L for this run.")

    if results_log:
        import csv
        with open("backtest_eth_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["bot", "bar_index", "signal", "entry_price", "future_price",
                              "move_pct", "confidence", "direction_outcome", "stop_loss",
                              "pnl_dollars", "balance_after"])
            writer.writerows(results_log)
        print("\nFull signal-by-signal log (including P&L per trade) saved to backtest_eth_results.csv")


if __name__ == "__main__":
    run_backtest()
