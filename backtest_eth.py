"""
backtest_eth.py
----------------
Purpose: contains all trading intelligence and market analysis for the ETH
bot. eth_bot.py handles Telegram/scheduling/infra and calls analyze() here
to get a structured trading decision back. This module has no Telegram
dependency and no side effects on import — it can be unit tested or run
standalone (see `if __name__ == "__main__"` at the bottom).

Everything here is rule-based, deterministic math on OHLCV candles — no ML,
no external TA library (indicators are implemented from scratch below).
"""

import math
import logging
import requests
from datetime import datetime, timezone

logger = logging.getLogger("backtest_eth")

# ---------------------------------------------------------------------------
# Constants / Configuration
# ---------------------------------------------------------------------------

PRODUCT_ID = "ETH-USD"
CANDLES_URL = f"https://api.exchange.coinbase.com/products/{PRODUCT_ID}/candles"
SPOT_URL = f"https://api.coinbase.com/v2/prices/{PRODUCT_ID}/spot"

# Coinbase's public candle API only supports these granularities (seconds).
# There's no native 4H or weekly bucket, so those two are built locally by
# resampling 1H and daily candles respectively.
GRANULARITY_15M = 900
GRANULARITY_1H = 3600
GRANULARITY_1D = 86400

TIMEFRAMES = ["weekly", "daily", "4h", "1h", "30m", "15m"]

SWING_STRENGTH = 2                     # bars each side to confirm a swing pivot
ATR_PERIOD = 14
SL_ATR_MULTIPLIER = 2
TP_RR_MULTIPLIERS = [1.0, 1.5, 2.0]    # TP1 / TP2 / TP3 as multiples of SL distance

VOLUME_LOOKBACK = 20
VOLUME_SPIKE_MULTIPLIER = 1.5
VOLUME_CONFIRMATION_MULTIPLIER = 1.0

RSI_PERIOD = 14
STOCH_RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ADX_PERIOD = 14
BOLLINGER_PERIOD, BOLLINGER_STDDEV = 20, 2
CCI_PERIOD = 20
ROC_PERIOD = 12
MOMENTUM_PERIOD = 10

WEAK_ADX = 20          # below this, trend has little strength regardless of direction

# Data model note: a "candle" is a plain dict:
#   {"time": <unix_seconds>, "open": f, "high": f, "low": f, "close": f, "volume": f}
# All functions below operate on chronologically-sorted (oldest-first) lists of these.


# ---------------------------------------------------------------------------
# Market Data
# ---------------------------------------------------------------------------

def fetch_spot_price():
    """Latest ETH/USD spot price from Coinbase. Returns float or None."""
    try:
        resp = requests.get(SPOT_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", {})
        return float(data["amount"])
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching spot price: {e}")
        return None
    except (ValueError, KeyError, TypeError) as e:
        logger.error(f"Error parsing spot price: {e}")
        return None


def fetch_candles(granularity, limit=200):
    """Chronological list of OHLCV candle dicts from Coinbase. [] on failure."""
    try:
        resp = requests.get(
            CANDLES_URL,
            params={"granularity": granularity},
            headers={"User-Agent": "eth-analysis-engine/1.0"},
            timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json()
        raw.sort(key=lambda c: c[0])  # Coinbase returns newest-first
        candles = [
            {"time": c[0], "open": c[3], "high": c[2], "low": c[1], "close": c[4], "volume": c[5]}
            for c in raw
        ]
        return candles[-limit:]
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching candles (granularity={granularity}): {e}")
        return []
    except (ValueError, KeyError, TypeError, IndexError) as e:
        logger.error(f"Error parsing candle data (granularity={granularity}): {e}")
        return []


def resample_candles(candles, factor):
    """
    Aggregates every `factor` consecutive candles into one bigger candle
    (e.g. factor=4 on 1H candles approximates 4H; factor=7 on daily candles
    approximates weekly). Used because Coinbase's public API doesn't offer
    those granularities directly. Drops a trailing partial bucket.
    """
    out = []
    usable = len(candles) - (len(candles) % factor)
    for i in range(0, usable, factor):
        chunk = candles[i:i + factor]
        out.append({
            "time": chunk[0]["time"],
            "open": chunk[0]["open"],
            "high": max(c["high"] for c in chunk),
            "low": min(c["low"] for c in chunk),
            "close": chunk[-1]["close"],
            "volume": sum(c["volume"] for c in chunk),
        })
    return out


def fetch_multi_timeframe_candles():
    """
    Fetches candles for all six analysis timeframes in one call, using the
    minimum number of API requests (4H is derived from 1H, weekly from
    daily). Returns {"weekly": [...], "daily": [...], "4h": [...],
    "1h": [...], "30m": [...], "15m": [...]}. Any timeframe that couldn't be
    built (fetch failure or insufficient candles to resample) is [].
    """
    daily = fetch_candles(GRANULARITY_1D, limit=120)
    hourly = fetch_candles(GRANULARITY_1H, limit=200)
    fifteen = fetch_candles(GRANULARITY_15M, limit=200)

    return {
        "weekly": resample_candles(daily, 7),
        "daily": daily,
        "4h": resample_candles(hourly, 4),
        "1h": hourly,
        "30m": resample_candles(fifteen, 2),
        "15m": fifteen,
    }


# ---------------------------------------------------------------------------
# Indicators (implemented from scratch — no TA library dependency)
# ---------------------------------------------------------------------------

def sma(values, period):
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def ema_series(values, period):
    """Full EMA series (same length as `values`, first `period`-1 entries are
    None). Needed internally by MACD; use ema() for just the latest value."""
    if len(values) < period:
        return [None] * len(values)
    k = 2 / (period + 1)
    out = [None] * (period - 1)
    seed = sum(values[:period]) / period
    out.append(seed)
    prev = seed
    for v in values[period:]:
        cur = v * k + prev * (1 - k)
        out.append(cur)
        prev = cur
    return out


def ema(values, period):
    series = ema_series(values, period)
    return series[-1] if series else None


def rsi(values, period=RSI_PERIOD):
    if len(values) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def rsi_series(values, period=RSI_PERIOD):
    """Full RSI series, needed for Stochastic RSI. None-padded to align lengths."""
    if len(values) < period + 1:
        return [None] * len(values)
    out = [None] * period
    for end in range(period + 1, len(values) + 1):
        out.append(rsi(values[:end], period))
    return out


def stochastic_rsi(values, period=STOCH_RSI_PERIOD):
    """Stochastic RSI: normalizes RSI itself into a 0-100 range using a
    rolling min/max, making it more sensitive than plain RSI. Returns None
    if there isn't enough history."""
    series = [v for v in rsi_series(values, period) if v is not None]
    if len(series) < period:
        return None
    window = series[-period:]
    lo, hi = min(window), max(window)
    if hi == lo:
        return 50.0
    return round((series[-1] - lo) / (hi - lo) * 100, 2)


def atr(candles, period=ATR_PERIOD):
    if len(candles) < period + 1:
        return None
    trs = []
    for i in range(1, len(candles)):
        h, l, pc = candles[i]["high"], candles[i]["low"], candles[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs[-period:]) / period


def macd(values, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    """Returns (macd_line, signal_line, histogram) — any may be None if
    there isn't enough history yet."""
    if len(values) < slow + signal:
        return None, None, None
    fast_series = ema_series(values, fast)
    slow_series = ema_series(values, slow)
    macd_line_series = [
        (f - s) if (f is not None and s is not None) else None
        for f, s in zip(fast_series, slow_series)
    ]
    valid = [v for v in macd_line_series if v is not None]
    if len(valid) < signal:
        return macd_line_series[-1], None, None
    signal_series = ema_series(valid, signal)
    signal_line = signal_series[-1]
    macd_line = valid[-1]
    histogram = macd_line - signal_line if signal_line is not None else None
    return macd_line, signal_line, histogram


def adx(candles, period=ADX_PERIOD):
    """Average Directional Index — trend strength (not direction), 0-100.
    Standard Wilder smoothing. Returns None if insufficient history."""
    if len(candles) < period * 2:
        return None

    plus_dm, minus_dm, trs = [], [], []
    for i in range(1, len(candles)):
        up_move = candles[i]["high"] - candles[i - 1]["high"]
        down_move = candles[i - 1]["low"] - candles[i]["low"]
        plus_dm.append(up_move if (up_move > down_move and up_move > 0) else 0)
        minus_dm.append(down_move if (down_move > up_move and down_move > 0) else 0)
        h, l, pc = candles[i]["high"], candles[i]["low"], candles[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))

    def wilder_smooth(vals, period):
        smoothed = [sum(vals[:period])]
        for v in vals[period:]:
            smoothed.append(smoothed[-1] - (smoothed[-1] / period) + v)
        return smoothed

    smoothed_tr = wilder_smooth(trs, period)
    smoothed_plus = wilder_smooth(plus_dm, period)
    smoothed_minus = wilder_smooth(minus_dm, period)

    dx_values = []
    for tr_v, plus_v, minus_v in zip(smoothed_tr, smoothed_plus, smoothed_minus):
        if tr_v == 0:
            continue
        plus_di = 100 * plus_v / tr_v
        minus_di = 100 * minus_v / tr_v
        denom = plus_di + minus_di
        dx_values.append(0 if denom == 0 else 100 * abs(plus_di - minus_di) / denom)

    if len(dx_values) < period:
        return round(dx_values[-1], 2) if dx_values else None
    return round(sum(dx_values[-period:]) / period, 2)


def bollinger_bands(values, period=BOLLINGER_PERIOD, stddev_mult=BOLLINGER_STDDEV):
    if len(values) < period:
        return None
    window = values[-period:]
    mid = sum(window) / period
    variance = sum((v - mid) ** 2 for v in window) / period
    stddev = math.sqrt(variance)
    return {"upper": mid + stddev_mult * stddev, "mid": mid, "lower": mid - stddev_mult * stddev}


def vwap(candles):
    """Volume-weighted average price over the given candle window (treats
    the whole list as one session — callers pass a recent slice)."""
    if not candles:
        return None
    total_pv, total_vol = 0, 0
    for c in candles:
        typical = (c["high"] + c["low"] + c["close"]) / 3
        total_pv += typical * c["volume"]
        total_vol += c["volume"]
    return total_pv / total_vol if total_vol else None


def cci(candles, period=CCI_PERIOD):
    if len(candles) < period:
        return None
    typical_prices = [(c["high"] + c["low"] + c["close"]) / 3 for c in candles[-period:]]
    mean_tp = sum(typical_prices) / period
    mean_dev = sum(abs(tp - mean_tp) for tp in typical_prices) / period
    if mean_dev == 0:
        return 0.0
    current_tp = typical_prices[-1]
    return round((current_tp - mean_tp) / (0.015 * mean_dev), 2)


def roc(values, period=ROC_PERIOD):
    if len(values) < period + 1:
        return None
    prior = values[-period - 1]
    if prior == 0:
        return None
    return round((values[-1] - prior) / prior * 100, 2)


def momentum(values, period=MOMENTUM_PERIOD):
    if len(values) < period + 1:
        return None
    return round(values[-1] - values[-period - 1], 2)


def compute_indicators(candles):
    """Bundles every indicator above into one dict for a given candle list.
    This is the primary entry point other sections use, so indicator param
    tuning only has to happen in one place (the constants above)."""
    closes = [c["close"] for c in candles]
    macd_line, macd_signal, macd_hist = macd(closes)
    return {
        "sma20": sma(closes, 20),
        "sma50": sma(closes, 50),
        "ema20": ema(closes, 20),
        "ema50": ema(closes, 50),
        "rsi": rsi(closes),
        "stoch_rsi": stochastic_rsi(closes),
        "atr": atr(candles),
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "adx": adx(candles),
        "bollinger": bollinger_bands(closes),
        "vwap": vwap(candles[-VOLUME_LOOKBACK:]),
        "cci": cci(candles),
        "roc": roc(closes),
        "momentum": momentum(closes),
    }


# ---------------------------------------------------------------------------
# Market Structure
# ---------------------------------------------------------------------------

def detect_swings(candles, strength=SWING_STRENGTH):
    """Fractal swing detection: candle i is a swing high/low if its high/low
    is the extreme within `strength` bars on each side. Chronological list
    of (index, 'high'|'low', price)."""
    swings = []
    for i in range(strength, len(candles) - strength):
        window = candles[i - strength: i + strength + 1]
        if candles[i]["high"] == max(c["high"] for c in window):
            swings.append((i, "high", candles[i]["high"]))
        if candles[i]["low"] == min(c["low"] for c in window):
            swings.append((i, "low", candles[i]["low"]))
    return swings


def classify_structure(candles):
    """
    BULLISH = last two swing highs AND lows both rising (HH + HL)
    BEARISH = last two swing highs AND lows both falling (LH + LL)
    RANGING = mixed sequence: also reports break-of-structure (BOS, a swing
              breaking the same direction as the prevailing trend) and
              change-of-character (CHoCH, the first break against it — the
              earliest hint of a potential reversal).
    """
    if len(candles) < SWING_STRENGTH * 2 + 5:
        return {"trend": "UNKNOWN", "support": None, "resistance": None,
                "bos": False, "choch": False, "strength": "UNKNOWN"}

    swings = detect_swings(candles)
    highs = [s for s in swings if s[1] == "high"]
    lows = [s for s in swings if s[1] == "low"]
    support = lows[-1][2] if lows else None
    resistance = highs[-1][2] if highs else None

    if len(highs) < 2 or len(lows) < 2:
        trend = "UNKNOWN"
    else:
        hh, hl = highs[-1][2] > highs[-2][2], lows[-1][2] > lows[-2][2]
        lh, ll = highs[-1][2] < highs[-2][2], lows[-1][2] < lows[-2][2]
        if hh and hl:
            trend = "BULLISH"
        elif lh and ll:
            trend = "BEARISH"
        else:
            trend = "RANGING"

    # BOS/CHoCH: compare current close against the most recent opposite-side
    # swing. A close beyond the last swing high in an uptrend = continuation
    # (BOS). A close beyond the last swing low during an uptrend = the first
    # sign of reversal (CHoCH).
    bos, choch = False, False
    last_close = candles[-1]["close"]
    if trend == "BULLISH" and support is not None:
        bos = last_close > (resistance or last_close)
        choch = last_close < support
    elif trend == "BEARISH" and resistance is not None:
        bos = last_close < (support or last_close)
        choch = last_close > resistance

    adx_val = adx(candles)
    strength = "UNKNOWN" if adx_val is None else ("WEAK" if adx_val < WEAK_ADX else
                ("STRONG" if adx_val > 40 else "MODERATE"))

    return {"trend": trend, "support": support, "resistance": resistance,
            "bos": bos, "choch": choch, "strength": strength, "adx": adx_val}


# ---------------------------------------------------------------------------
# Smart Money Concepts
# ---------------------------------------------------------------------------

def detect_order_blocks(candles, lookback=30):
    """
    Simplified order-block detection: the last down-candle immediately
    before a strong up-move is a bullish order block (institutional buy
    zone); mirror for bearish. "Strong move" = a candle whose body is at
    least 1.5x the average body size in the window. Returns the most recent
    bullish and bearish OB (each a {"top","bottom","time"} dict or None).
    """
    window = candles[-lookback:]
    if len(window) < 5:
        return {"bullish": None, "bearish": None}
    bodies = [abs(c["close"] - c["open"]) for c in window]
    avg_body = sum(bodies) / len(bodies) or 1e-9

    bullish_ob, bearish_ob = None, None
    for i in range(1, len(window)):
        move_body = abs(window[i]["close"] - window[i]["open"])
        if move_body < avg_body * 1.5:
            continue
        prev = window[i - 1]
        if window[i]["close"] > window[i]["open"] and prev["close"] < prev["open"]:
            bullish_ob = {"top": prev["open"], "bottom": prev["close"], "time": prev["time"]}
        elif window[i]["close"] < window[i]["open"] and prev["close"] > prev["open"]:
            bearish_ob = {"top": prev["close"], "bottom": prev["open"], "time": prev["time"]}
    return {"bullish": bullish_ob, "bearish": bearish_ob}


def detect_fair_value_gaps(candles, lookback=30):
    """
    3-candle imbalance: bullish FVG when candle[i-1].high < candle[i+1].low
    (a gap price never traded through); bearish FVG is the mirror. Returns
    a list of {"type","top","bottom","time"} for every gap still open
    (i.e. price hasn't fully retraced through it since).
    """
    window = candles[-lookback:]
    gaps = []
    for i in range(1, len(window) - 1):
        left, right = window[i - 1], window[i + 1]
        if left["high"] < right["low"]:
            gaps.append({"type": "bullish", "top": right["low"], "bottom": left["high"], "time": window[i]["time"]})
        elif left["low"] > right["high"]:
            gaps.append({"type": "bearish", "top": left["low"], "bottom": right["high"], "time": window[i]["time"]})

    # Drop gaps price has already closed fully back through
    last_close = window[-1]["close"]
    open_gaps = [g for g in gaps if not (g["bottom"] < last_close < g["top"])]
    return open_gaps[-5:]  # most recent few are what matters


def detect_liquidity_sweep(candles, swings=None, lookback=30):
    """
    A liquidity sweep: a wick pokes beyond a recent swing high/low (grabbing
    stop-loss liquidity resting there) but the candle closes back inside —
    a classic stop-hunt-then-reverse signature. Returns
    {"swept": "high"|"low"|None, "level": price|None}.
    """
    window = candles[-lookback:]
    if swings is None:
        swings = detect_swings(window)
    highs = [s[2] for s in swings if s[1] == "high"]
    lows = [s[2] for s in swings if s[1] == "low"]
    last = window[-1]

    if highs and last["high"] > max(highs) and last["close"] < max(highs):
        return {"swept": "high", "level": max(highs)}
    if lows and last["low"] < min(lows) and last["close"] > min(lows):
        return {"swept": "low", "level": min(lows)}
    return {"swept": None, "level": None}


def classify_premium_discount(price, range_high, range_low):
    """Where price sits within the recent range: PREMIUM (top half, favors
    selling), DISCOUNT (bottom half, favors buying), or EQUILIBRIUM (~midpoint)."""
    if range_high is None or range_low is None or range_high == range_low:
        return "UNKNOWN"
    pct = (price - range_low) / (range_high - range_low)
    if pct > 0.6:
        return "PREMIUM"
    if pct < 0.4:
        return "DISCOUNT"
    return "EQUILIBRIUM"


def analyze_smart_money(candles):
    order_blocks = detect_order_blocks(candles)
    fvgs = detect_fair_value_gaps(candles)
    swings = detect_swings(candles)
    sweep = detect_liquidity_sweep(candles, swings)
    recent = candles[-30:]
    zone = classify_premium_discount(
        candles[-1]["close"],
        max(c["high"] for c in recent) if recent else None,
        min(c["low"] for c in recent) if recent else None,
    )
    return {"order_blocks": order_blocks, "fvgs": fvgs, "liquidity_sweep": sweep, "zone": zone}


# ---------------------------------------------------------------------------
# Support & Resistance
# ---------------------------------------------------------------------------

def find_support_resistance(candles, lookback=100):
    """
    Dynamic S/R from swing points in the lookback window, split into
    major (touched/approached 2+ times within 0.5% tolerance) vs minor
    (touched once). Also flags a breakout/breakdown/retest against the
    single closest major level.
    """
    window = candles[-lookback:]
    swings = detect_swings(window)
    highs = sorted([s[2] for s in swings if s[1] == "high"])
    lows = sorted([s[2] for s in swings if s[1] == "low"])

    def cluster(levels, tolerance_pct=0.5):
        if not levels:
            return [], []
        major, minor, used = [], [], [False] * len(levels)
        for i, lvl in enumerate(levels):
            if used[i]:
                continue
            touches = [lvl]
            for j in range(i + 1, len(levels)):
                if used[j]:
                    continue
                if abs(levels[j] - lvl) / lvl * 100 <= tolerance_pct:
                    touches.append(levels[j])
                    used[j] = True
            avg_level = sum(touches) / len(touches)
            (major if len(touches) >= 2 else minor).append(round(avg_level, 2))
        return major, minor

    major_res, minor_res = cluster(highs)
    major_sup, minor_sup = cluster(lows)

    last_close = window[-1]["close"]
    all_major = sorted(major_res + major_sup)
    closest_major = min(all_major, key=lambda lvl: abs(lvl - last_close)) if all_major else None

    status = "NONE"
    if closest_major is not None and len(window) >= 2:
        prev_close = window[-2]["close"]
        if prev_close < closest_major <= last_close:
            status = "BREAKOUT"
        elif prev_close > closest_major >= last_close:
            status = "BREAKDOWN"
        elif abs(last_close - closest_major) / closest_major * 100 <= 0.3:
            status = "RETEST"

    return {
        "major_resistance": major_res, "minor_resistance": minor_res,
        "major_support": major_sup, "minor_support": minor_sup,
        "closest_major_level": closest_major, "level_status": status,
    }


# ---------------------------------------------------------------------------
# Candlestick Recognition
# ---------------------------------------------------------------------------

def detect_candlestick_pattern(candles):
    """
    Classifies the most recent candle (plus prior 1-2 for multi-candle
    patterns) via plain OHLC math. Checked in priority order (multi-candle
    patterns first, since they're higher-conviction); first match wins.
    Returns {"pattern": name, "bias": "bullish"|"bearish"|"neutral"}.
    """
    if len(candles) < 3:
        return {"pattern": "UNKNOWN", "bias": "neutral"}

    c0, c1, c2 = candles[-3], candles[-2], candles[-1]  # oldest -> newest of the 3

    def body(c): return abs(c["close"] - c["open"])
    def rng(c): return c["high"] - c["low"] if c["high"] != c["low"] else 1e-9
    def upper(c): return c["high"] - max(c["open"], c["close"])
    def lower(c): return min(c["open"], c["close"]) - c["low"]
    def bullish(c): return c["close"] > c["open"]
    def bearish(c): return c["close"] < c["open"]

    # --- Three-candle patterns ---
    if bearish(c0) and body(c1) < body(c0) * 0.5 and bullish(c2) and c2["close"] > (c0["open"] + c0["close"]) / 2:
        return {"pattern": "Morning Star", "bias": "bullish"}
    if bullish(c0) and body(c1) < body(c0) * 0.5 and bearish(c2) and c2["close"] < (c0["open"] + c0["close"]) / 2:
        return {"pattern": "Evening Star", "bias": "bearish"}
    if bullish(c0) and bullish(c1) and bullish(c2) and c2["close"] > c1["close"] > c0["close"] \
            and all(body(c) > rng(c) * 0.6 for c in (c0, c1, c2)):
        return {"pattern": "Three White Soldiers", "bias": "bullish"}
    if bearish(c0) and bearish(c1) and bearish(c2) and c2["close"] < c1["close"] < c0["close"] \
            and all(body(c) > rng(c) * 0.6 for c in (c0, c1, c2)):
        return {"pattern": "Three Black Crows", "bias": "bearish"}

    # --- Two-candle patterns (prev = c1, current = c2) ---
    prev, cur = c1, c2
    cur_body, prev_body = body(cur), body(prev)

    if bullish(cur) and bearish(prev) and cur["open"] <= prev["close"] and cur["close"] >= prev["open"] and cur_body > prev_body:
        return {"pattern": "Bullish Engulfing", "bias": "bullish"}
    if bearish(cur) and bullish(prev) and cur["open"] >= prev["close"] and cur["close"] <= prev["open"] and cur_body > prev_body:
        return {"pattern": "Bearish Engulfing", "bias": "bearish"}
    if cur_body < prev_body * 0.6 and max(cur["open"], cur["close"]) < max(prev["open"], prev["close"]) \
            and min(cur["open"], cur["close"]) > min(prev["open"], prev["close"]):
        bias = "bullish" if bearish(prev) else "bearish"
        return {"pattern": "Harami", "bias": bias}
    if cur["high"] > prev["high"] and cur["low"] < prev["low"] and cur_body > prev_body:
        bias = "bullish" if bullish(cur) else "bearish"
        return {"pattern": "Outside Bar (Engulfing Range)", "bias": bias}
    if cur["high"] <= prev["high"] and cur["low"] >= prev["low"]:
        return {"pattern": "Inside Bar", "bias": "neutral"}

    # --- Single-candle patterns ---
    cb, cr, cu, cl = body(cur), rng(cur), upper(cur), lower(cur)

    if cb > 0 and cl >= cb * 2 and cu <= cb * 0.5:
        return {"pattern": "Hammer", "bias": "bullish"}
    if cb > 0 and cu >= cb * 2 and cl <= cb * 0.5:
        return {"pattern": "Shooting Star", "bias": "bearish"}
    if cb <= cr * 0.05 and cu >= cr * 0.4 and cl <= cr * 0.1:
        return {"pattern": "Gravestone Doji", "bias": "bearish"}
    if cb <= cr * 0.05 and cl >= cr * 0.4 and cu <= cr * 0.1:
        return {"pattern": "Dragonfly Doji", "bias": "bullish"}
    if cb <= cr * 0.1:
        return {"pattern": "Doji", "bias": "neutral"}
    if cb >= cr * 0.9:
        return {"pattern": "Bullish Marubozu" if bullish(cur) else "Bearish Marubozu",
                "bias": "bullish" if bullish(cur) else "bearish"}
    if cl >= cr * 0.5:
        return {"pattern": "Long Lower Wick", "bias": "bullish"}
    if cu >= cr * 0.5:
        return {"pattern": "Long Upper Wick", "bias": "bearish"}
    if cr < (sum(rng(c) for c in candles[-10:]) / min(10, len(candles))) * 0.5:
        return {"pattern": "Short Wick / Low Range", "bias": "neutral"}

    return {"pattern": "No clear pattern", "bias": "neutral"}


# ---------------------------------------------------------------------------
# Volume Analysis
# ---------------------------------------------------------------------------

def analyze_volume(candles, lookback=VOLUME_LOOKBACK):
    """Average volume, spike/confirmation flags, buy/sell pressure (based on
    where each candle closes within its own range), and a simple
    price-vs-volume divergence check over the last two swing highs."""
    if len(candles) < lookback + 1:
        return {"avg_volume": None, "spike": None, "confirmed": None,
                "pressure": "UNKNOWN", "divergence": None}

    recent = candles[-(lookback + 1):-1]
    avg_volume = sum(c["volume"] for c in recent) / len(recent)
    current = candles[-1]

    spike = avg_volume > 0 and current["volume"] > avg_volume * VOLUME_SPIKE_MULTIPLIER
    confirmed = avg_volume > 0 and current["volume"] > avg_volume * VOLUME_CONFIRMATION_MULTIPLIER

    c_range = current["high"] - current["low"] or 1e-9
    close_position = (current["close"] - current["low"]) / c_range
    pressure = "BUYING" if close_position > 0.6 else ("SELLING" if close_position < 0.4 else "NEUTRAL")

    # Divergence: compare the two most recent swing highs' price vs volume
    swings = detect_swings(candles[-lookback * 2:])
    highs = [s for s in swings if s[1] == "high"]
    divergence = None
    if len(highs) >= 2:
        idx_a, _, price_a = highs[-2]
        idx_b, _, price_b = highs[-1]
        vol_a = candles[-lookback * 2:][idx_a]["volume"]
        vol_b = candles[-lookback * 2:][idx_b]["volume"]
        if price_b > price_a and vol_b < vol_a:
            divergence = "BEARISH"  # higher high on weaker volume
        elif price_b < price_a and vol_b > vol_a:
            divergence = "BULLISH"  # lower high but volume picking up = absorption

    return {"avg_volume": round(avg_volume, 2), "current_volume": current["volume"],
            "spike": spike, "confirmed": confirmed, "pressure": pressure, "divergence": divergence}


# ---------------------------------------------------------------------------
# Multi-Timeframe Analysis
# ---------------------------------------------------------------------------

def trend_for_timeframe(candles):
    """Lightweight trend read for one timeframe: EMA20 vs EMA50 slope plus
    swing structure. Used for the MTF agreement/conflict check — cheaper
    than running full classify_structure() six times per cycle."""
    if len(candles) < 55:
        structure = classify_structure(candles)
        return structure["trend"]
    closes = [c["close"] for c in candles]
    fast, slow = ema(closes, 20), ema(closes, 50)
    if fast is None or slow is None:
        return "UNKNOWN"
    structure_trend = classify_structure(candles)["trend"]
    ema_trend = "BULLISH" if fast > slow else "BEARISH"
    # Require EMA slope and structure to agree for a confident read; otherwise RANGING
    if structure_trend in ("BULLISH", "BEARISH") and structure_trend == ema_trend:
        return structure_trend
    return "RANGING"


def analyze_multi_timeframe(tf_candles):
    """
    tf_candles: dict from fetch_multi_timeframe_candles(). Returns trend per
    timeframe plus an agreement score (0-100, weighted toward higher
    timeframes) and a conflict flag when weekly/daily disagree with 1h/15m.
    """
    weights = {"weekly": 25, "daily": 20, "4h": 20, "1h": 15, "30m": 10, "15m": 10}
    trends = {tf: trend_for_timeframe(candles) if candles else "UNKNOWN"
              for tf, candles in tf_candles.items()}

    bullish_weight = sum(weights[tf] for tf, t in trends.items() if t == "BULLISH")
    bearish_weight = sum(weights[tf] for tf, t in trends.items() if t == "BEARISH")
    total_weight = sum(w for tf, w in weights.items() if trends.get(tf) != "UNKNOWN") or 1

    agreement_pct = round(max(bullish_weight, bearish_weight) / total_weight * 100)
    dominant = "BULLISH" if bullish_weight > bearish_weight else (
        "BEARISH" if bearish_weight > bullish_weight else "MIXED")

    htf_trend = trends.get("weekly") if trends.get("weekly") != "UNKNOWN" else trends.get("daily")
    ltf_trend = trends.get("1h") if trends.get("1h") != "UNKNOWN" else trends.get("15m")
    conflict = bool(htf_trend and ltf_trend and htf_trend != "UNKNOWN" and ltf_trend != "UNKNOWN"
                     and htf_trend != ltf_trend and "RANGING" not in (htf_trend, ltf_trend))

    return {"trends": trends, "dominant": dominant, "agreement_pct": agreement_pct, "conflict": conflict}


# ---------------------------------------------------------------------------
# Pullback / Reversal Detection
# ---------------------------------------------------------------------------

def detect_pullback(candles, structure, indicators):
    """
    Measures the current retracement from the most recent swing extreme (in
    the direction of the prevailing trend) as a multiple of ATR:
      < 1x ATR   -> Healthy Pullback (normal noise within a trend)
      1x - 2.5x  -> Correction / Retracement (still trend-continuation territory)
      > 2.5x     -> Deep Pullback (trend may be exhausting)
    Returns None if trend is unclear or ATR isn't available.
    """
    trend = structure.get("trend")
    atr_val = indicators.get("atr")
    if trend not in ("BULLISH", "BEARISH") or not atr_val:
        return None

    price = candles[-1]["close"]
    extreme = structure.get("resistance") if trend == "BULLISH" else structure.get("support")
    if extreme is None:
        return None

    retracement = abs(extreme - price)
    ratio = retracement / atr_val

    if ratio < 1:
        depth = "Healthy Pullback"
    elif ratio < 2.5:
        depth = "Retracement / Correction"
    else:
        depth = "Deep Pullback"

    return {"depth": depth, "atr_multiple": round(ratio, 2), "continuation_likely": ratio < 2.5}


def detect_reversal(structure, pattern, volume_info):
    """
    Combines CHoCH (structure), a reversal-biased candlestick pattern, and
    volume divergence into one reversal read. Needs at least two of the
    three signals pointing the same direction to call it — a single signal
    alone is treated as "possible" rather than confirmed.
    """
    bullish_votes = sum([
        structure.get("choch") and structure.get("trend") == "BEARISH",
        pattern.get("bias") == "bullish",
        volume_info.get("divergence") == "BULLISH",
    ])
    bearish_votes = sum([
        structure.get("choch") and structure.get("trend") == "BULLISH",
        pattern.get("bias") == "bearish",
        volume_info.get("divergence") == "BEARISH",
    ])

    if bullish_votes >= 2:
        return {"reversal": "BULLISH", "confirmed": True, "votes": bullish_votes}
    if bearish_votes >= 2:
        return {"reversal": "BEARISH", "confirmed": True, "votes": bearish_votes}
    if bullish_votes == 1 or bearish_votes == 1:
        direction = "BULLISH" if bullish_votes else "BEARISH"
        return {"reversal": direction, "confirmed": False, "votes": 1}
    return {"reversal": None, "confirmed": False, "votes": 0}


# ---------------------------------------------------------------------------
# Risk Management
# ---------------------------------------------------------------------------

def compute_trade_levels(entry_price, atr_val, direction):
    """ATR-based stop-loss and 3-tier take-profit. None if ATR unavailable."""
    if not atr_val:
        return None
    sl_distance = atr_val * SL_ATR_MULTIPLIER
    if direction == "BUY":
        stop_loss = entry_price - sl_distance
        take_profits = [entry_price + sl_distance * m for m in TP_RR_MULTIPLIERS]
    else:
        stop_loss = entry_price + sl_distance
        take_profits = [entry_price - sl_distance * m for m in TP_RR_MULTIPLIERS]
    return {"stop_loss": stop_loss, "take_profits": take_profits, "risk_reward": TP_RR_MULTIPLIERS[-1]}


def compute_trailing_stop(direction, current_price, atr_val, multiplier=SL_ATR_MULTIPLIER):
    if not atr_val:
        return None
    distance = atr_val * multiplier
    return current_price - distance if direction == "BUY" else current_price + distance


def compute_position_size(account_balance, risk_pct, entry_price, stop_loss):
    """Position size in units of the asset, sized so a stop-out loses exactly
    `risk_pct` of account_balance. Returns None if any input is missing —
    the bot doesn't track account balance by default, so this is opt-in."""
    if not all([account_balance, risk_pct, entry_price, stop_loss]):
        return None
    risk_amount = account_balance * (risk_pct / 100)
    per_unit_risk = abs(entry_price - stop_loss)
    if per_unit_risk == 0:
        return None
    return round(risk_amount / per_unit_risk, 6)


def check_breakeven(direction, entry_price, current_price, atr_val):
    """True once price has moved 1x ATR in the trade's favor — the usual
    trigger for moving a stop to break-even."""
    if not atr_val:
        return False
    moved = (current_price - entry_price) if direction == "BUY" else (entry_price - current_price)
    return moved >= atr_val


# ---------------------------------------------------------------------------
# Confidence Engine
# ---------------------------------------------------------------------------

def compute_confidence(signal, indicators, structure, mtf, pattern, volume_info, sr):
    """
    Rule-based 0-100 confidence score, transparent and reproducible (not a
    model prediction). Weights: MTF agreement 20, 1H/local structure 20,
    momentum (RSI/MACD/ADX) 20, pattern 15, volume 15, S/R room 10.
    Returns (score, reasons) where reasons is a human-readable checklist.
    """
    score, reasons = 0, []

    # Multi-timeframe agreement (0-20)
    if mtf["dominant"] == signal_to_trend(signal) :
        tf_score = round(mtf["agreement_pct"] / 100 * 20)
        score += tf_score
        reasons.append(f"✅ Multi-timeframe agreement {mtf['agreement_pct']}% ({mtf['dominant']})")
    elif mtf["conflict"]:
        reasons.append("❌ Higher and lower timeframes conflict")
    else:
        reasons.append(f"⚠️ Timeframes mixed ({mtf['dominant']})")

    # Structure (0-20)
    trend = structure.get("trend")
    if trend == signal_to_trend(signal):
        struct_score = 20 if structure.get("strength") == "STRONG" else (14 if structure.get("strength") == "MODERATE" else 8)
        score += struct_score
        reasons.append(f"✅ Structure confirms ({trend}, {structure.get('strength')})")
    elif trend == "RANGING":
        score += 5
        reasons.append("⚠️ Structure ranging — weaker confirmation")
    else:
        reasons.append(f"❌ Structure disagrees ({trend})")

    # Momentum: RSI + MACD + ADX combined (0-20)
    momentum_score = 0
    rsi_val, macd_hist, adx_val = indicators.get("rsi"), indicators.get("macd_hist"), indicators.get("adx")
    if rsi_val is not None:
        if (signal == "BUY" and rsi_val < 70) or (signal == "SELL" and rsi_val > 30):
            momentum_score += 7
    if macd_hist is not None:
        if (signal == "BUY" and macd_hist > 0) or (signal == "SELL" and macd_hist < 0):
            momentum_score += 7
    if adx_val is not None and adx_val > WEAK_ADX:
        momentum_score += 6
    score += momentum_score
    reasons.append(f"{'✅' if momentum_score >= 13 else '⚠️' if momentum_score >= 7 else '❌'} "
                    f"Momentum (RSI {rsi_val}, MACD hist {round(macd_hist,2) if macd_hist is not None else 'N/A'}, ADX {adx_val})")

    # Candlestick pattern (0-15)
    if pattern.get("bias") == signal_to_bias(signal):
        score += 15
        reasons.append(f"✅ Pattern supports ({pattern.get('pattern')})")
    elif pattern.get("bias") == "neutral":
        score += 5
        reasons.append(f"⚠️ Pattern neutral ({pattern.get('pattern')})")
    else:
        reasons.append(f"❌ Pattern disagrees ({pattern.get('pattern')})")

    # Volume (0-15)
    if volume_info.get("confirmed"):
        vol_score = 15 if volume_info.get("spike") else 10
        score += vol_score
        reasons.append(f"✅ Volume confirms (spike={volume_info.get('spike')})")
    else:
        reasons.append("❌ Volume not confirming")

    # Support/resistance room (0-10) — more room to the next major level = higher score
    closest = sr.get("closest_major_level")
    if closest:
        price = indicators.get("_price")
        room_pct = abs(closest - price) / price * 100 if price else 0
        sr_score = min(10, round(room_pct * 4))
        score += sr_score
        reasons.append(f"{'✅' if sr_score >= 6 else '⚠️'} Room to next major level ({room_pct:.2f}%)")
    else:
        reasons.append("⚠️ No major level nearby to gauge room")

    return min(100, score), reasons


def signal_to_trend(signal):
    return "BULLISH" if signal == "BUY" else ("BEARISH" if signal == "SELL" else None)


def signal_to_bias(signal):
    return "bullish" if signal == "BUY" else ("bearish" if signal == "SELL" else None)


# ---------------------------------------------------------------------------
# Trade Engine
# ---------------------------------------------------------------------------

MIN_CONFIDENCE_FOR_SIGNAL = 55
WARNING_CONFIDENCE_BAND = 15   # within this many points below the threshold -> WARNING instead of WAIT

def decide_trade(structure, mtf, reversal):
    """
    Produces a raw directional bias (BUY/SELL/WAIT) before confidence is
    scored — confidence scoring needs a direction to score against, so this
    picks the direction first from structure + MTF dominant trend, with
    reversal detection able to override toward EXIT/WARNING territory.
    """
    trend = structure.get("trend")
    dominant = mtf.get("dominant")

    if reversal.get("confirmed"):
        # A confirmed reversal against the prevailing trend is treated as a
        # WARNING on the old direction rather than an immediate new signal —
        # one bar of reversal evidence isn't enough to flip a live trade.
        return "WARNING", (trend if trend in ("BULLISH", "BEARISH") else dominant)

    if trend == "BULLISH" and dominant in ("BULLISH", "MIXED"):
        return "BUY", trend
    if trend == "BEARISH" and dominant in ("BEARISH", "MIXED"):
        return "SELL", trend
    return "WAIT", trend


def finalize_signal(raw_direction, confidence):
    """Applies the confidence threshold: a raw BUY/SELL below threshold gets
    downgraded to WARNING (close, not confirmed) or WAIT (not close)."""
    if raw_direction not in ("BUY", "SELL"):
        return raw_direction
    if confidence >= MIN_CONFIDENCE_FOR_SIGNAL:
        return raw_direction
    if confidence >= MIN_CONFIDENCE_FOR_SIGNAL - WARNING_CONFIDENCE_BAND:
        return "WARNING"
    return "WAIT"


# ---------------------------------------------------------------------------
# Trade Monitoring (for an already-open trade)
# ---------------------------------------------------------------------------

def monitor_trade(trade, current_price, atr_val, structure):
    """
    trade: {"direction": "BUY"|"SELL", "entry": f, "stop_loss": f,
            "take_profits": [tp1,tp2,tp3]}
    Returns one of: TP1_HIT / TP2_HIT / TP3_HIT / SL_HIT / WEAKNESS / EXIT / HOLD.
    WEAKNESS fires when structure flips against the trade direction (early
    warning); EXIT fires on a confirmed CHoCH against it (stronger signal).
    """
    direction = trade["direction"]
    stop_loss = trade["stop_loss"]
    tps = trade["take_profits"]

    hit_sl = (current_price <= stop_loss) if direction == "BUY" else (current_price >= stop_loss)
    if hit_sl:
        return "SL_HIT"

    for i, tp in enumerate(reversed(tps)):
        tp_index = len(tps) - i
        hit_tp = (current_price >= tp) if direction == "BUY" else (current_price <= tp)
        if hit_tp:
            return f"TP{tp_index}_HIT"

    trend = structure.get("trend")
    if structure.get("choch"):
        return "EXIT"
    if (direction == "BUY" and trend == "BEARISH") or (direction == "SELL" and trend == "BULLISH"):
        return "WEAKNESS"

    return "HOLD"


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------

def run_backtest(candles, warmup=60):
    """
    Walk-forward simulation: at each bar past `warmup`, runs the same
    structure+indicator logic used live (using only candles up to and
    including that bar — no lookahead) to generate a signal, and if BUY/SELL
    with confidence above threshold, opens a simulated trade sized with
    compute_trade_levels(). The trade is then walked forward bar-by-bar
    until SL or a take-profit is hit, or the data runs out.

    Returns a dict with win/loss counts, win rate, max drawdown, profit
    factor, a simple Sharpe ratio (on per-trade returns), monthly/weekly/
    daily breakdowns, and a trade journal (list of closed trades).
    """
    trades = []
    equity = [0.0]
    open_trade = None

    for i in range(warmup, len(candles)):
        window = candles[:i + 1]
        price = window[-1]["close"]

        if open_trade is not None:
            result = monitor_trade(open_trade, price, atr(window), classify_structure(window))
            if result in ("SL_HIT", "TP1_HIT", "TP2_HIT", "TP3_HIT", "EXIT"):
                pnl_pct = ((price - open_trade["entry"]) / open_trade["entry"] * 100
                           if open_trade["direction"] == "BUY"
                           else (open_trade["entry"] - price) / open_trade["entry"] * 100)
                open_trade.update({"exit_price": price, "exit_time": window[-1]["time"],
                                    "result": result, "pnl_pct": round(pnl_pct, 3)})
                trades.append(open_trade)
                equity.append(equity[-1] + pnl_pct)
                open_trade = None
            continue

        structure = classify_structure(window)
        indicators = compute_indicators(window)
        indicators["_price"] = price
        pattern = detect_candlestick_pattern(window)
        volume_info = analyze_volume(window)
        sr = find_support_resistance(window)
        # Backtest uses single-timeframe data, so approximate MTF agreement
        # with the local structure/ADX read rather than a full multi-fetch.
        mtf = {"dominant": structure["trend"], "agreement_pct": 60 if structure["strength"] != "WEAK" else 30,
               "conflict": False, "trends": {}}
        reversal = detect_reversal(structure, pattern, volume_info)

        raw_direction, _ = decide_trade(structure, mtf, reversal)
        if raw_direction not in ("BUY", "SELL"):
            continue
        confidence, _ = compute_confidence(raw_direction, indicators, structure, mtf, pattern, volume_info, sr)
        signal = finalize_signal(raw_direction, confidence)
        if signal not in ("BUY", "SELL"):
            continue

        levels = compute_trade_levels(price, indicators["atr"], signal)
        if levels is None:
            continue
        open_trade = {"direction": signal, "entry": price, "entry_time": window[-1]["time"],
                       "stop_loss": levels["stop_loss"], "take_profits": levels["take_profits"],
                       "confidence": confidence}

    return {
        "trades": trades,
        "stats": compute_backtest_stats(trades, equity),
    }


def compute_backtest_stats(trades, equity):
    if not trades:
        return {"total_trades": 0, "win_rate": None, "loss_rate": None, "max_drawdown_pct": None,
                "profit_factor": None, "sharpe_ratio": None}

    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    gross_profit = sum(t["pnl_pct"] for t in wins)
    gross_loss = abs(sum(t["pnl_pct"] for t in losses))

    peak = equity[0]
    max_dd = 0.0
    for e in equity:
        peak = max(peak, e)
        max_dd = max(max_dd, peak - e)

    returns = [t["pnl_pct"] for t in trades]
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    stddev = math.sqrt(variance)
    sharpe = round(mean_return / stddev, 3) if stddev > 0 else None

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 2),
        "loss_rate": round(len(losses) / len(trades) * 100, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else None,
        "sharpe_ratio": sharpe,
        "total_return_pct": round(sum(returns), 2),
    }


def performance_by_period(trades, period="monthly"):
    """Buckets closed trades by month/week/day (UTC) and sums pnl_pct per
    bucket. period: 'monthly' | 'weekly' | 'daily'."""
    buckets = {}
    for t in trades:
        dt = datetime.fromtimestamp(t["exit_time"], tz=timezone.utc)
        if period == "monthly":
            key = dt.strftime("%Y-%m")
        elif period == "weekly":
            key = f"{dt.isocalendar().year}-W{dt.isocalendar().week:02d}"
        else:
            key = dt.strftime("%Y-%m-%d")
        buckets.setdefault(key, 0.0)
        buckets[key] += t["pnl_pct"]
    return {k: round(v, 2) for k, v in sorted(buckets.items())}


# ---------------------------------------------------------------------------
# Performance Tracking (live, cumulative — separate from historical backtest)
# ---------------------------------------------------------------------------

def new_performance_state():
    return {"total_signals": 0, "buy": 0, "sell": 0, "wait": 0, "warning": 0, "exit": 0,
            "wins": 0, "losses": 0, "neutral": 0, "current_streak": 0, "best_streak": 0,
            "streak_type": None}


def record_signal_in_state(state, signal):
    state["total_signals"] += 1
    key = signal.lower()
    if key in state:
        state[key] += 1
    return state


def record_outcome_in_state(state, outcome):
    """outcome: 'WIN' | 'LOSS' | 'NEUTRAL'. Tracks a running streak of
    same-type outcomes (win streak or loss streak), resetting on a switch."""
    key_map = {"WIN": "wins", "LOSS": "losses", "NEUTRAL": "neutral"}
    key = key_map.get(outcome)
    if key in state:
        state[key] += 1

    if outcome in ("WIN", "LOSS"):
        if state["streak_type"] == outcome:
            state["current_streak"] += 1
        else:
            state["streak_type"] = outcome
            state["current_streak"] = 1
        if outcome == "WIN":
            state["best_streak"] = max(state["best_streak"], state["current_streak"])
    else:
        state["streak_type"] = None
        state["current_streak"] = 0
    return state


def accuracy_pct(state):
    total = state["wins"] + state["losses"]
    return round(state["wins"] / total * 100, 2) if total else None


# ---------------------------------------------------------------------------
# Final AI Decision Engine — the single entry point eth_bot.py calls
# ---------------------------------------------------------------------------

def analyze():
    """
    Orchestrates every section above into one trading decision. Returns a
    dict ready for eth_bot.py's message formatting, or None if live data
    couldn't be fetched (network hiccup — caller should skip this cycle).
    """
    tf_candles = fetch_multi_timeframe_candles()
    primary = tf_candles.get("1h") or []
    entry_tf = tf_candles.get("15m") or []

    if not primary or not entry_tf:
        logger.warning("Insufficient candle data this cycle — skipping analysis.")
        return None

    price = entry_tf[-1]["close"]

    structure = classify_structure(primary)
    indicators = compute_indicators(primary)
    indicators["_price"] = price
    smc = analyze_smart_money(primary)
    sr = find_support_resistance(primary)
    pattern = detect_candlestick_pattern(entry_tf)
    volume_info = analyze_volume(primary)
    mtf = analyze_multi_timeframe(tf_candles)
    pullback = detect_pullback(primary, structure, indicators)
    reversal = detect_reversal(structure, pattern, volume_info)

    raw_direction, _ = decide_trade(structure, mtf, reversal)
    confidence, reasons = compute_confidence(raw_direction, indicators, structure, mtf, pattern, volume_info, sr)
    signal = finalize_signal(raw_direction, confidence)

    levels = None
    if signal in ("BUY", "SELL"):
        levels = compute_trade_levels(price, indicators.get("atr"), signal)

    return {
        "signal": signal,
        "price": price,
        "confidence": confidence,
        "reasons": reasons,
        "entry": price if signal in ("BUY", "SELL") else None,
        "stop_loss": levels["stop_loss"] if levels else None,
        "take_profits": levels["take_profits"] if levels else None,
        "risk_reward": levels["risk_reward"] if levels else None,
        "trend": structure.get("trend"),
        "structure": structure,
        "smart_money": smc,
        "support": sr.get("closest_major_level") if sr.get("level_status") in ("RETEST",) else structure.get("support"),
        "resistance": structure.get("resistance"),
        "sr_detail": sr,
        "volume": volume_info,
        "pattern": pattern,
        "mtf": mtf,
        "pullback": pullback,
        "reversal": reversal,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }


if __name__ == "__main__":
    # Quick manual smoke test: `python backtest_eth.py`
    logging.basicConfig(level=logging.INFO)
    result = analyze()
    if result:
        import pprint
        pprint.pprint(result)
    else:
        print("analyze() returned None — check network/API access.")
