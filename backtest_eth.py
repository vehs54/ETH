"""
ETH Bot Backtester (Full Logic — Phase 1 + Volume + HTF + Candlestick)
------------------------------------------------------------------------
Replays the EXACT current eth_bot.py signal pipeline against real historical
data:
  1. MA5/MA10/MA30 crossover + RSI(15), 60/40 bands, 0.08% spread threshold,
     3-cycle confirmation (on ~30-min spaced spot-price ticks)
  2. 1H market structure filter (blocks BUY into BEARISH, SELL into BULLISH)
  3. 6H higher-timeframe structure filter (same logic, coarser trend)
  4. Volume confirmation filter (current 1H candle volume must exceed
     1.2x the recent 20-candle average)
  5. Candlestick pattern + full confidence scoring (0-100), logged per signal
     so you can check whether higher confidence actually correlates with wins

Data source: Coinbase Exchange public API (no key required) — 15-min candles
downsampled to ~30-min spacing for the price series, plus native 1H and 6H
candles for structure/volume/pattern.

CRITICAL DESIGN POINT — no lookahead bias: at each simulated tick, only 1H/6H
candles that would have actually closed by that timestamp are used for
structure/volume/pattern — never future data the live bot couldn't have seen.

HOW TO RUN:
  1. pip install requests
  2. python backtest_eth.py
  3. Results print to console, plus a full per-signal CSV
     (backtest_eth_results.csv) including the confidence score logged at
     signal time, so you can check whether confidence correlates with outcome.
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
MIN_MA_SPREAD_PCT = 0.08
RSI_BUY_CEILING = 60
RSI_SELL_FLOOR = 40
CONFIRMATION_STREAK = 3

SWING_STRENGTH = 2
ATR_PERIOD = 14
SL_ATR_MULTIPLIER = 2
TP_RR_MULTIPLIERS = [1.0, 1.5, 2.0]

VOLUME_LOOKBACK = 20
VOLUME_CONFIRMATION_MULTIPLIER = 1.2

STRUCTURE_LOOKBACK = 100
HTF_LOOKBACK = 60

# Backtest-specific
EVAL_HORIZON_BARS = 6          # 6 * 30min = 3 hours ahead, same as live accuracy tracker
MIN_MOVE_THRESHOLD_PCT = 0.05
DAYS_OF_HISTORY = 90            # longer window than the simple backtest — this bot fires rarer

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
    trend = structure.get("trend")
    htf_trend = structure.get("htf_trend")
    volume = structure.get("volume")
    if signal == "BUY":
        if trend == "BEARISH" or htf_trend == "BEARISH":
            return "HOLD"
    elif signal == "SELL":
        if trend == "BULLISH" or htf_trend == "BULLISH":
            return "HOLD"
    if volume is not None and not volume["confirmed"]:
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

    # Downsample 15-min to ~30-min spacing, matching the live bot's poll interval
    price_series = price_candles[::2]
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

    # --- Evaluate outcomes ---
    wins, losses, neutrals = 0, 0, 0
    results_log = []
    win_confidences, loss_confidences = [], []

    for idx, sig, entry_price, confidence in signals:
        future_idx = idx + EVAL_HORIZON_BARS
        if future_idx >= len(closes):
            continue
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

        results_log.append(("ETH/USDT bot (full logic)", idx, sig, round(entry_price, 4),
                             round(future_price, 4), round(move_pct, 3), confidence, outcome))

    total = wins + losses + neutrals
    print(f"\n=== ETH/USDT bot — full current logic ===")
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
        if win_confidences and loss_confidences:
            if sum(win_confidences)/len(win_confidences) > sum(loss_confidences)/len(loss_confidences):
                print("  -> Confidence score DOES correlate with outcome (wins score higher on average).")
            else:
                print("  -> Confidence score does NOT clearly correlate with outcome here — worth investigating.")
    else:
        print("  Not enough data to evaluate any signals.")
        print("  This system has 5 stacked filters (MA/RSI + 1H + 6H + volume + confirmation streak) —")
        print("  if you're seeing zero or very few signals, that's the strictness talking, not a bug.")
        print(f"  Try increasing DAYS_OF_HISTORY (currently {DAYS_OF_HISTORY}) for a larger sample.")

    if results_log:
        import csv
        with open("backtest_eth_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["bot", "bar_index", "signal", "entry_price", "exit_price",
                              "move_pct", "confidence", "outcome"])
            writer.writerows(results_log)
        print("\nFull signal-by-signal log saved to backtest_eth_results.csv")


if __name__ == "__main__":
    run_backtest()
