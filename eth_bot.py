"""
ETH/USDT Telegram Signal Bot
-----------------------------
Price source: Coinbase public spot price endpoint (ETH-USD, no API key required)
Signal logic: MA5 / MA10 / MA30 crossover + RSI(15)
Features: 3-hour rolling BUY/SELL/HOLD breakdown, inline Refresh/Start/Stop buttons,
          /status, /price, /start, /stop commands, Flask self-ping server for Render free tier.

Same structure as the current BTC, PAXG, and ETC bots:
  - MA spread threshold: 0.08%
  - RSI bands: 60 (buy ceiling) / 40 (sell floor)
  - Confirmation streak: 3 cycles (at 30-min polling = 90 min sustained trend)
  - No reversal/exit warning system — BUY/SELL alerts only, one consistent message template
  - Manual /start and /stop commands (+ inline buttons) control proactive alerts
  - 30-minute polling interval
  - Live accuracy tracking persisted to a local JSON state file (survives restarts,
    NOT Render redeploys — free-tier disk resets on every new deploy)

PHASE 1 ADDITION — Market Structure + Support/Resistance:
  Fetches hourly OHLC candles (separate from the spot-price polling used for the
  MA/RSI signal) to detect swing highs/lows, classify trend structure as
  BULLISH (higher highs + higher lows), BEARISH (lower highs + lower lows), or
  RANGING, and track the nearest support/resistance levels. This acts as an
  additional confirmation gate: a BUY signal from MA/RSI is blocked if market
  structure is clearly BEARISH, and a SELL signal is blocked if structure is
  clearly BULLISH. Structure and S/R levels are shown in every message.
"""

import os
import sys
import json
import asyncio
import logging
from collections import deque
from datetime import datetime, timedelta
from threading import Thread

import requests
from flask import Flask
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("eth_bot")

# ---------------------------------------------------------------------------
# Startup environment variable validation
# ---------------------------------------------------------------------------
REQUIRED_ENV_VARS = ["BOT_TOKEN", "CHAT_ID"]

missing = [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]
if missing:
    logger.error(f"Missing required environment variable(s): {', '.join(missing)}")
    logger.error("Set these in Render's Environment tab before redeploying.")
    sys.exit(1)

BOT_TOKEN = os.environ["BOT_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
COINBASE_URL = "https://api.coinbase.com/v2/prices/ETH-USD/spot"

POLL_INTERVAL_SECONDS = 900   # 15 minutes (was 30 — faster detection, see CONFIRMATION_STREAK below)
HISTORY_MAXLEN = 60
RSI_PERIOD = 15
MA_SHORT = 5
MA_MED = 10
MA_LONG = 30
ROLLING_WINDOW_HOURS = 3
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot_state.json")

# --- Market structure / support-resistance / trade-level config (Phase 1) ---
CANDLES_URL = "https://api.exchange.coinbase.com/products/ETH-USD/candles"
STRUCTURE_GRANULARITY_SECONDS = 3600   # 1-hour candles
STRUCTURE_LOOKBACK = 100               # candles requested per fetch
SWING_STRENGTH = 2                     # bars on each side to confirm a swing pivot
ATR_PERIOD = 14
SL_ATR_MULTIPLIER = 2                  # stop-loss = entry -/+ 2x ATR
TP_RR_MULTIPLIERS = [1.0, 1.5, 2.0]    # TP1/TP2/TP3 as multiples of the SL distance

# --- Volume confirmation ---
VOLUME_LOOKBACK = 20                   # candles used to compute the average
VOLUME_CONFIRMATION_MULTIPLIER = 1.0   # current volume must exceed avg * this to confirm (was 1.1 — loosened further)

# --- Higher-timeframe structure ---
# Coinbase's public candle API only supports 60/300/900/3600/21600/86400 second
# granularities — there is no exact 4H option. 6H (21600s) is the closest
# available, so that's what's used here as the "higher timeframe" check.
HTF_GRANULARITY_SECONDS = 21600        # 6-hour candles
HTF_LOOKBACK = 60

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
price_history = deque(maxlen=HISTORY_MAXLEN)
signal_log = deque()
alerts_enabled = True         # controlled by /start and /stop
last_alerted_signal = None    # tracks last signal actually sent, to avoid repeats

# Cached market structure snapshot — refreshed each poll/check cycle since it
# requires a separate candle fetch. Structure: trend, htf_trend, support,
# resistance, atr, volume.
last_structure = {
    "trend": "UNKNOWN", "htf_trend": "UNKNOWN",
    "support": None, "resistance": None, "atr": None, "volume": None,
    "pattern": {"pattern": "UNKNOWN", "bias": "neutral"},
}

# --- Signal confidence filtering ---
MIN_MA_SPREAD_PCT = 0.05   # was 0.08 — loosened to let weaker-but-real trends qualify
RSI_BUY_CEILING = 60
RSI_SELL_FLOOR = 40
CONFIRMATION_STREAK = 2    # 2 * 15min = 30 min sustained trend — loosened from 4 (60 min)
recent_raw_signals = deque(maxlen=CONFIRMATION_STREAK)

# --- Live accuracy tracking ---
EVAL_HORIZON_CYCLES = 12       # cycles ahead to check outcome (12 * 15min poll = 3 hours — same window as before)
MIN_MOVE_THRESHOLD_PCT = 0.05  # minimum move to count as a real win/loss vs noise
pending_evaluations = deque()  # (cycles_remaining, signal_type, entry_price)
accuracy_stats = {"WIN": 0, "LOSS": 0, "NEUTRAL": 0}

# ---------------------------------------------------------------------------
# Flask keep-alive server (Render free tier)
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/")
def home():
    return "ETH/USDT bot is alive.", 200


def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)


# ---------------------------------------------------------------------------
# Price fetching
# ---------------------------------------------------------------------------
def fetch_price():
    """Fetch the latest ETH/USD spot price from Coinbase. Returns float or None.
    No API key required — this is Coinbase's public data endpoint."""
    try:
        resp = requests.get(COINBASE_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        asset = data.get("data")
        if not asset or "amount" not in asset:
            logger.warning(f"Unexpected Coinbase response: {data}")
            return None

        return float(asset["amount"])

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching price: {e}")
        return None
    except (ValueError, KeyError, TypeError) as e:
        logger.error(f"Error parsing price data: {e}")
        return None


# ---------------------------------------------------------------------------
# Market structure / support-resistance (Phase 1)
# ---------------------------------------------------------------------------
def fetch_candles(granularity=STRUCTURE_GRANULARITY_SECONDS, limit=STRUCTURE_LOOKBACK):
    """
    Fetches recent OHLCV candles from Coinbase's public Exchange API (no API
    key required) at the given granularity. Returns a chronological list of
    dicts: {time, low, high, open, close, volume}. Returns [] on failure —
    callers must handle that gracefully rather than assuming data exists.
    """
    try:
        resp = requests.get(
            CANDLES_URL,
            params={"granularity": granularity},
            headers={"User-Agent": "eth-signal-bot/1.0"},
            timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json()
        # Coinbase candle format: [time, low, high, open, close, volume], newest first
        raw.sort(key=lambda c: c[0])
        candles = [
            {"time": c[0], "low": c[1], "high": c[2], "open": c[3], "close": c[4], "volume": c[5]}
            for c in raw
        ][-limit:]
        return candles
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching candles (granularity={granularity}): {e}")
        return []
    except (ValueError, KeyError, TypeError, IndexError) as e:
        logger.error(f"Error parsing candle data (granularity={granularity}): {e}")
        return []


def check_volume_confirmation(candles, lookback=VOLUME_LOOKBACK):
    """
    Compares the most recent candle's volume against the average of the
    preceding `lookback` candles. Returns None if there isn't enough history
    yet — callers should treat None as "unknown", not "unconfirmed", so a
    fresh deploy doesn't block every signal until it has 20+ candles.
    """
    if len(candles) < lookback + 1:
        return None
    recent = candles[-(lookback + 1):-1]  # exclude the current, still-forming candle
    avg_volume = sum(c["volume"] for c in recent) / len(recent)
    current_volume = candles[-1]["volume"]
    if avg_volume == 0:
        return None
    confirmed = current_volume > avg_volume * VOLUME_CONFIRMATION_MULTIPLIER
    return {"confirmed": confirmed, "current_volume": current_volume, "avg_volume": avg_volume}


def compute_atr(candles, period=ATR_PERIOD):
    """Average True Range over the given candle list. Returns None if insufficient data."""
    if len(candles) < period + 1:
        return None
    true_ranges = []
    for i in range(1, len(candles)):
        high = candles[i]["high"]
        low = candles[i]["low"]
        prev_close = candles[i - 1]["close"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)
    return sum(true_ranges[-period:]) / period


def detect_swings(candles, strength=SWING_STRENGTH):
    """
    Simple fractal-style swing detection: a candle is a swing high if its high
    is the max within `strength` bars on each side, and a swing low if its low
    is the min within `strength` bars on each side. Returns a chronological
    list of (index, 'high'|'low', price).
    """
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
    Classifies trend structure from swing highs/lows:
      BULLISH  = most recent two swing highs AND lows are both rising (HH + HL)
      BEARISH  = most recent two swing highs AND lows are both falling (LH + LL)
      RANGING  = mixed / no clear sequence
      UNKNOWN  = not enough candle history yet

    Also returns the nearest support (most recent confirmed swing low) and
    nearest resistance (most recent confirmed swing high) — these are real
    swing points from actual price history, not estimates.
    """
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


def detect_candlestick_pattern(candles):
    """
    Classifies the most recent candle (with the prior candle for engulfing
    patterns) using plain OHLC math — no ML, fully deterministic. Returns
    {"pattern": <name>, "bias": "bullish"|"bearish"|"neutral"}.

    Checked in order: engulfing patterns first (need 2 candles), then Doji,
    then Marubozu, then Hammer/Shooting Star, then a generic long-wick
    rejection as a fallback. First match wins.
    """
    if len(candles) < 2:
        return {"pattern": "UNKNOWN", "bias": "neutral"}

    current = candles[-1]
    previous = candles[-2]

    def body(c):
        return abs(c["close"] - c["open"])

    def candle_range(c):
        return c["high"] - c["low"] if c["high"] != c["low"] else 1e-9

    def upper_wick(c):
        return c["high"] - max(c["open"], c["close"])

    def lower_wick(c):
        return min(c["open"], c["close"]) - c["low"]

    cur_body = body(current)
    cur_range = candle_range(current)
    cur_upper = upper_wick(current)
    cur_lower = lower_wick(current)
    cur_bullish = current["close"] > current["open"]
    cur_bearish = current["close"] < current["open"]

    prev_body = body(previous)
    prev_bullish = previous["close"] > previous["open"]
    prev_bearish = previous["close"] < previous["open"]

    # Bullish engulfing: bullish candle whose body fully engulfs the prior
    # bearish candle's body
    if (cur_bullish and prev_bearish and current["open"] <= previous["close"]
            and current["close"] >= previous["open"] and cur_body > prev_body):
        return {"pattern": "Bullish Engulfing", "bias": "bullish"}

    # Bearish engulfing: mirror of the above
    if (cur_bearish and prev_bullish and current["open"] >= previous["close"]
            and current["close"] <= previous["open"] and cur_body > prev_body):
        return {"pattern": "Bearish Engulfing", "bias": "bearish"}

    # Hammer: small body, long lower wick, little upper wick — bullish reversal.
    # Checked before Doji since both have small bodies — wick asymmetry is
    # what actually distinguishes a hammer from indecision.
    if cur_body > 0 and cur_lower >= cur_body * 2 and cur_upper <= cur_body * 0.5:
        return {"pattern": "Hammer", "bias": "bullish"}

    # Shooting star: small body, long upper wick, little lower wick — bearish reversal
    if cur_body > 0 and cur_upper >= cur_body * 2 and cur_lower <= cur_body * 0.5:
        return {"pattern": "Shooting Star", "bias": "bearish"}

    # Doji: body is negligible relative to the candle's full range, AND wicks
    # are roughly balanced (if they weren't, it would have matched hammer/star
    # above) — genuine indecision, not a directional rejection
    if cur_body <= cur_range * 0.1:
        return {"pattern": "Doji", "bias": "neutral"}

    # Marubozu: body dominates the range, almost no wicks — strong momentum
    if cur_body >= cur_range * 0.9:
        bias = "bullish" if cur_bullish else "bearish"
        return {"pattern": f"{'Bullish' if cur_bullish else 'Bearish'} Marubozu", "bias": bias}

    # Fallback: a long wick on either side even without a clean hammer/star shape
    if cur_lower >= cur_range * 0.5:
        return {"pattern": "Long Lower Rejection Wick", "bias": "bullish"}
    if cur_upper >= cur_range * 0.5:
        return {"pattern": "Long Upper Rejection Wick", "bias": "bearish"}

    return {"pattern": "No clear pattern", "bias": "neutral"}


def get_market_structure():
    """
    Single entry point used by the handlers: fetches 1H candles (main
    structure + volume) and 6H candles (higher-timeframe confirmation),
    returning everything together so callers only need one function call.
    Falls back to last_structure (stale but non-blocking) if a fetch fails,
    so a transient network hiccup doesn't kill signals entirely.
    """
    global last_structure
    candles = fetch_candles(granularity=STRUCTURE_GRANULARITY_SECONDS, limit=STRUCTURE_LOOKBACK)
    if not candles:
        logger.warning("1H candle fetch failed — using last known structure.")
        return last_structure

    structure = classify_structure(candles)
    structure["atr"] = compute_atr(candles)
    structure["volume"] = check_volume_confirmation(candles)
    structure["pattern"] = detect_candlestick_pattern(candles)

    htf_candles = fetch_candles(granularity=HTF_GRANULARITY_SECONDS, limit=HTF_LOOKBACK)
    if htf_candles:
        htf_structure = classify_structure(htf_candles)
        structure["htf_trend"] = htf_structure["trend"]
    else:
        logger.warning("6H candle fetch failed — higher-timeframe check skipped this cycle.")
        structure["htf_trend"] = last_structure.get("htf_trend", "UNKNOWN")

    last_structure = structure
    return structure


def compute_trade_levels(entry_price, atr, direction):
    """
    ATR-based stop-loss and 3-tier take-profit, matching the same 2x-ATR /
    risk:reward approach used in the Deriv autotrader. Returns None if ATR
    isn't available yet (not enough candle history).
    """
    if atr is None:
        return None

    sl_distance = atr * SL_ATR_MULTIPLIER

    if direction == "BUY":
        stop_loss = entry_price - sl_distance
        take_profits = [entry_price + sl_distance * m for m in TP_RR_MULTIPLIERS]
    else:  # SELL
        stop_loss = entry_price + sl_distance
        take_profits = [entry_price - sl_distance * m for m in TP_RR_MULTIPLIERS]

    return {
        "stop_loss": stop_loss,
        "take_profits": take_profits,
        "risk_reward": TP_RR_MULTIPLIERS[-1],  # RR of the furthest TP, e.g. 1:2
    }


def compute_confidence(signal, ma5, ma10, ma30, rsi, structure, price):
    """
    Rule-based confidence score out of 100 — transparent and reproducible,
    NOT a machine-learning prediction. Each component only awards points if
    it genuinely supports the direction of `signal`. Returns (score, reasons)
    where reasons is a list of human-readable checklist lines for the message.

    Weights: MA trend 15, RSI 10, 1H structure 20, 6H structure 15,
    volume 15, candlestick pattern 15, support/resistance room 10. Total 100.
    """
    score = 0
    reasons = []

    # MA trend alignment (0-15) — scaled by how separated the MAs are
    if ma5 is not None and ma10 is not None and ma30 is not None and ma30:
        spread_pct = abs(ma5 - ma30) / ma30 * 100
        trend_matches = (signal == "BUY" and ma5 > ma10 > ma30) or (signal == "SELL" and ma5 < ma10 < ma30)
        if trend_matches:
            ma_score = min(15, round(spread_pct / MIN_MA_SPREAD_PCT * 7.5))
            score += ma_score
            reasons.append(f"✅ MA trend aligned (spread {spread_pct:.2f}%)")
        else:
            reasons.append("❌ MA trend not aligned")

    # RSI positioning (0-10)
    if rsi is not None:
        if signal == "BUY" and rsi < RSI_BUY_CEILING:
            rsi_score = round((RSI_BUY_CEILING - rsi) / RSI_BUY_CEILING * 10)
            score += rsi_score
            reasons.append(f"✅ RSI supportive ({rsi})")
        elif signal == "SELL" and rsi > RSI_SELL_FLOOR:
            rsi_score = round((rsi - RSI_SELL_FLOOR) / (100 - RSI_SELL_FLOOR) * 10)
            score += rsi_score
            reasons.append(f"✅ RSI supportive ({rsi})")
        else:
            reasons.append(f"❌ RSI not confirming ({rsi})")

    # 1H market structure alignment (0-20)
    trend = structure.get("trend")
    if (signal == "BUY" and trend == "BULLISH") or (signal == "SELL" and trend == "BEARISH"):
        score += 20
        reasons.append(f"✅ 1H structure confirms ({trend})")
    elif trend == "RANGING":
        score += 7
        reasons.append("⚠️ 1H structure ranging — weaker confirmation")
    elif trend == "UNKNOWN":
        reasons.append("⚠️ 1H structure not yet available (gathering candle history)")
    else:
        reasons.append(f"❌ 1H structure disagrees ({trend})")

    # 6H (higher-timeframe) structure alignment (0-15)
    htf_trend = structure.get("htf_trend")
    if (signal == "BUY" and htf_trend == "BULLISH") or (signal == "SELL" and htf_trend == "BEARISH"):
        score += 15
        reasons.append(f"✅ 6H trend confirms ({htf_trend})")
    elif htf_trend == "RANGING":
        score += 5
        reasons.append("⚠️ 6H trend ranging — weaker confirmation")
    elif htf_trend == "UNKNOWN":
        reasons.append("⚠️ 6H trend not yet available (gathering candle history)")
    else:
        reasons.append(f"❌ 6H trend disagrees ({htf_trend})")

    # Volume confirmation (0-15)
    volume = structure.get("volume")
    if volume is None:
        reasons.append("⚠️ Volume data not yet available (gathering candle history)")
    elif volume["confirmed"]:
        ratio = volume["current_volume"] / volume["avg_volume"]
        score += 15
        reasons.append(f"✅ Volume confirms breakout ({ratio:.1f}x average)")
    else:
        ratio = volume["current_volume"] / volume["avg_volume"] if volume["avg_volume"] else 0
        reasons.append(f"❌ Volume below average ({ratio:.1f}x) — could be a fake move")

    # Candlestick pattern (0-15) — deterministic OHLC-based classification, not ML
    pattern_info = structure.get("pattern") or {"pattern": "UNKNOWN", "bias": "neutral"}
    pattern_bias = pattern_info.get("bias", "neutral")
    pattern_name = pattern_info.get("pattern", "UNKNOWN")
    if (signal == "BUY" and pattern_bias == "bullish") or (signal == "SELL" and pattern_bias == "bearish"):
        score += 15
        reasons.append(f"✅ Candlestick confirms ({pattern_name})")
    elif pattern_bias == "neutral":
        reasons.append(f"⚠️ Candlestick neutral ({pattern_name})")
    else:
        reasons.append(f"❌ Candlestick disagrees ({pattern_name})")

    # Distance from support/resistance (0-10) — reward not buying into resistance
    # or selling into support
    support = structure.get("support")
    resistance = structure.get("resistance")
    if signal == "BUY" and resistance is not None and price:
        room_pct = (resistance - price) / price * 100
        if room_pct > 0:
            sr_score = min(10, round(room_pct * 2))
            score += sr_score
            reasons.append(f"✅ Room to resistance (${resistance:,.2f}, {room_pct:.1f}% away)")
        else:
            reasons.append(f"❌ Already past resistance (${resistance:,.2f})")
    elif signal == "SELL" and support is not None and price:
        room_pct = (price - support) / price * 100
        if room_pct > 0:
            sr_score = min(10, round(room_pct * 2))
            score += sr_score
            reasons.append(f"✅ Room to support (${support:,.2f}, {room_pct:.1f}% away)")
        else:
            reasons.append(f"❌ Already past support (${support:,.2f})")

    return min(100, score), reasons


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def moving_average(data, period):
    if len(data) < period:
        return None
    return sum(list(data)[-period:]) / period


def calculate_rsi(data, period=RSI_PERIOD):
    """Returns RSI value or None if insufficient data."""
    if len(data) < period + 1:
        return None

    prices = list(data)[-(period + 1):]
    gains = []
    losses = []

    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]
        if change >= 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 2)


def generate_signal():
    """
    Returns one of 'BUY', 'SELL', 'HOLD' based on MA5/MA10/MA30 crossover + RSI(15),
    filtered for confidence: requires meaningful MA separation and RSI clearly
    past the midline. Returns None if not enough data yet.
    """
    ma5 = moving_average(price_history, MA_SHORT)
    ma10 = moving_average(price_history, MA_MED)
    ma30 = moving_average(price_history, MA_LONG)
    rsi = calculate_rsi(price_history)

    if ma5 is None or ma10 is None or ma30 is None:
        return None

    bullish_alignment = ma5 > ma10 > ma30
    bearish_alignment = ma5 < ma10 < ma30

    ma_spread_pct = abs(ma5 - ma30) / ma30 * 100 if ma30 else 0
    strong_spread = ma_spread_pct >= MIN_MA_SPREAD_PCT

    if bullish_alignment and strong_spread and (rsi is None or rsi < RSI_BUY_CEILING):
        return "BUY"
    elif bearish_alignment and strong_spread and (rsi is None or rsi > RSI_SELL_FLOOR):
        return "SELL"
    else:
        return "HOLD"


def confirmed_signal(raw_signal):
    """
    Tracks the last few raw signals and only returns BUY/SELL once the same
    signal has held for CONFIRMATION_STREAK consecutive cycles. Otherwise
    returns 'HOLD' so a single noisy tick can't trigger a false alert.
    """
    recent_raw_signals.append(raw_signal)

    if len(recent_raw_signals) < CONFIRMATION_STREAK:
        return "HOLD"

    if all(s == "BUY" for s in recent_raw_signals):
        return "BUY"
    elif all(s == "SELL" for s in recent_raw_signals):
        return "SELL"
    else:
        return "HOLD"


def apply_structure_filter(signal, structure):
    """
    Blocks a confirmed BUY/SELL if either of the following clearly disagree:
      - 1H structure (e.g. don't BUY into a confirmed BEARISH 1H structure)
      - Volume (require above-average volume to confirm the move isn't just noise)

    6H (higher-timeframe) structure is intentionally NOT a hard gate here —
    it was originally, but stacked with the 1H gate + volume gate + the
    3-cycle MA/RSI confirmation, that was too many simultaneous requirements
    and produced near-zero signals in practice. 6H trend still affects the
    confidence score (see compute_confidence) — it just no longer blocks a
    signal outright the way 1H structure and volume do.

    RANGING/UNKNOWN 1H structure and unknown volume (not enough history yet)
    don't block anything — there's no clear evidence either way, so we let
    the underlying MA/RSI signal stand.
    """
    trend = structure.get("trend")

    if signal == "BUY" and trend == "BEARISH":
        return "HOLD"
    elif signal == "SELL" and trend == "BULLISH":
        return "HOLD"

    # Volume is intentionally NOT a hard gate here — it was, but combined with
    # the 1H structure gate and MA/RSI confirmation, it was blocking too many
    # otherwise-valid setups. Weak volume now reduces the confidence score
    # instead (see compute_confidence) rather than silently killing the signal.
    return signal


def load_state():
    """Loads persisted accuracy_stats and pending_evaluations from disk, if present."""
    global accuracy_stats
    if not os.path.exists(STATE_FILE):
        logger.info("No existing state file found — starting fresh.")
        return
    try:
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
        accuracy_stats.update(data.get("accuracy_stats", {}))
        pending_evaluations.clear()
        for item in data.get("pending_evaluations", []):
            pending_evaluations.append(tuple(item))
        logger.info(
            f"Loaded persisted state: {accuracy_stats}, "
            f"{len(pending_evaluations)} pending evaluation(s)."
        )
    except Exception as e:
        logger.warning(f"Could not load state file, starting fresh: {e}")


def save_state():
    """Persists accuracy_stats and pending_evaluations to disk."""
    try:
        with open(STATE_FILE, "w") as f:
            json.dump({
                "accuracy_stats": accuracy_stats,
                "pending_evaluations": list(pending_evaluations),
            }, f)
    except Exception as e:
        logger.warning(f"Could not save state file: {e}")


def evaluate_pending_signals(current_price):
    """
    Checks any pending signal evaluations that have reached their horizon,
    scores them WIN/LOSS/NEUTRAL against the current price, and updates
    the running accuracy_stats tally.
    """
    still_pending = deque()
    while pending_evaluations:
        cycles_remaining, sig_type, entry_price = pending_evaluations.popleft()
        cycles_remaining -= 1
        if cycles_remaining > 0:
            still_pending.append((cycles_remaining, sig_type, entry_price))
            continue

        move_pct = (current_price - entry_price) / entry_price * 100
        if sig_type == "BUY":
            outcome = "WIN" if move_pct > MIN_MOVE_THRESHOLD_PCT else (
                "LOSS" if move_pct < -MIN_MOVE_THRESHOLD_PCT else "NEUTRAL"
            )
        else:  # SELL
            outcome = "WIN" if move_pct < -MIN_MOVE_THRESHOLD_PCT else (
                "LOSS" if move_pct > MIN_MOVE_THRESHOLD_PCT else "NEUTRAL"
            )
        accuracy_stats[outcome] += 1

    pending_evaluations.extend(still_pending)


def get_accuracy_display():
    """Returns a human-readable accuracy summary string for the message template."""
    total = accuracy_stats["WIN"] + accuracy_stats["LOSS"] + accuracy_stats["NEUTRAL"]
    if total == 0:
        return "Bot accuracy: gathering data..."
    win_rate = round(accuracy_stats["WIN"] / total * 100, 1)
    return (
        f"Bot accuracy: {win_rate}% win rate "
        f"({accuracy_stats['WIN']}W / {accuracy_stats['LOSS']}L / {accuracy_stats['NEUTRAL']}N, {total} scored)"
    )


def record_signal(signal):
    now = datetime.utcnow()
    signal_log.append((now, signal))
    cutoff = now - timedelta(hours=ROLLING_WINDOW_HOURS)
    while signal_log and signal_log[0][0] < cutoff:
        signal_log.popleft()


def get_rolling_breakdown():
    """Returns dict with BUY/SELL/HOLD percentages over the rolling window."""
    if not signal_log:
        return {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}

    total = len(signal_log)
    counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
    for _, sig in signal_log:
        counts[sig] += 1

    return {k: round((v / total) * 100, 1) for k, v in counts.items()}


# ---------------------------------------------------------------------------
# Message formatting — ONE consistent template used everywhere a signal is shown
# ---------------------------------------------------------------------------
def build_status_message(price, signal, rsi, structure):
    ma5 = moving_average(price_history, MA_SHORT)
    ma10 = moving_average(price_history, MA_MED)
    ma30 = moving_average(price_history, MA_LONG)
    breakdown = get_rolling_breakdown()

    rsi_display = f"{rsi}" if rsi is not None else "N/A (gathering data)"
    signal_display = signal if signal is not None else "Gathering data..."
    trend = structure.get("trend", "UNKNOWN")
    htf_trend = structure.get("htf_trend", "UNKNOWN")
    support = structure.get("support")
    resistance = structure.get("resistance")
    atr = structure.get("atr")
    volume = structure.get("volume")

    support_display = f"${support:,.2f}" if support is not None else "N/A (gathering candles)"
    resistance_display = f"${resistance:,.2f}" if resistance is not None else "N/A (gathering candles)"
    if volume is None:
        volume_display = "N/A (gathering candles)"
    else:
        ratio = volume["current_volume"] / volume["avg_volume"] if volume["avg_volume"] else 0
        volume_display = f"{ratio:.1f}x average ({'confirmed' if volume['confirmed'] else 'below threshold'})"

    pattern_info = structure.get("pattern") or {"pattern": "UNKNOWN", "bias": "neutral"}
    pattern_display = pattern_info.get("pattern", "UNKNOWN")

    lines = [
        f"Ξ *ETH/USDT Signal*\n",
        f"💰 Price: ${price:,.2f}",
        f"📈 1H Trend: *{trend}*  |  6H Trend: *{htf_trend}*",
        f"🟢 Support: {support_display}",
        f"🔴 Resistance: {resistance_display}",
        f"📦 Volume: {volume_display}",
        f"🕯️ Candle: {pattern_display}",
        f"📊 Signal: *{signal_display}*",
    ]

    # Only show trade levels + confidence when there's an actual BUY/SELL to size
    if signal in ("BUY", "SELL"):
        levels = compute_trade_levels(price, atr, signal)
        confidence, reasons = compute_confidence(signal, ma5, ma10, ma30, rsi, structure, price)

        if levels is not None:
            tp1, tp2, tp3 = levels["take_profits"]
            lines.append(f"\n🛑 Stop Loss: ${levels['stop_loss']:,.2f}")
            lines.append(
                f"🎯 Take Profit:\nTP1: ${tp1:,.2f}\nTP2: ${tp2:,.2f}\nTP3: ${tp3:,.2f}"
            )
            lines.append(f"⚖️ Risk:Reward: 1:{levels['risk_reward']:.1f}")
        else:
            lines.append("\n🛑 Stop Loss / 🎯 Take Profit: gathering candle history...")

        lines.append(f"\n🔎 Confidence: {confidence}%")
        lines.append("Reason:\n" + "\n".join(reasons))

    ma5_display = f"{ma5:.2f}" if ma5 is not None else "N/A"
    ma10_display = f"{ma10:.2f}" if ma10 is not None else "N/A"
    ma30_display = f"{ma30:.2f}" if ma30 is not None else "N/A"
    lines.append(f"\nMA5: {ma5_display} | MA10: {ma10_display} | MA30: {ma30_display}")
    lines.append(f"RSI(15): {rsi_display}")
    lines.append(
        f"\n📈 Last {ROLLING_WINDOW_HOURS}h breakdown:\n"
        f"  BUY: {breakdown['BUY']}% | SELL: {breakdown['SELL']}% | HOLD: {breakdown['HOLD']}%"
    )
    lines.append(f"\n🏆 {get_accuracy_display()}")
    lines.append(f"\n🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

    return "\n".join(lines)


def build_keyboard():
    keyboard = [
        [
            InlineKeyboardButton("🔄 Refresh", callback_data="refresh"),
        ],
        [
            InlineKeyboardButton("▶️ Start", callback_data="start"),
            InlineKeyboardButton("⏹ Stop", callback_data="stop"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


# ---------------------------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------------------------
async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    price = fetch_price()

    if price is not None:
        price_history.append(price)
        structure = get_market_structure()
        signal = generate_signal()
        signal = apply_structure_filter(signal, structure) if signal else signal
        if signal:
            record_signal(signal)
        rsi = calculate_rsi(price_history)
        msg = build_status_message(price, signal, rsi, structure)
        await update.message.reply_text(
            msg, parse_mode="Markdown", reply_markup=build_keyboard()
        )
    else:
        await update.message.reply_text(
            "⚠️ Couldn't fetch the current ETH price. Try again shortly."
        )


async def price_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    price = fetch_price()
    if price is not None:
        await update.message.reply_text(f"Ξ ETH/USDT: ${price:,.2f}")
    else:
        await update.message.reply_text("⚠️ Couldn't fetch the current ETH price.")


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global alerts_enabled
    alerts_enabled = True
    await update.message.reply_text(
        "▶️ Alerts turned ON. You'll get a message whenever a strict BUY/SELL signal confirms."
    )


async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global alerts_enabled
    alerts_enabled = False
    await update.message.reply_text(
        "⏹ Alerts turned OFF. Use /start to resume, or /status anytime to check manually."
    )


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global alerts_enabled
    query = update.callback_query
    await query.answer()

    if query.data == "refresh":
        price = fetch_price()
        if price is not None:
            price_history.append(price)
            structure = get_market_structure()
            signal = generate_signal()
            signal = apply_structure_filter(signal, structure) if signal else signal
            if signal:
                record_signal(signal)
            rsi = calculate_rsi(price_history)
            msg = build_status_message(price, signal, rsi, structure)
            await query.edit_message_text(
                msg, parse_mode="Markdown", reply_markup=build_keyboard()
            )
        else:
            await query.edit_message_text("⚠️ Couldn't fetch the current ETH price.")

    elif query.data == "start":
        alerts_enabled = True
        await query.answer("Alerts turned ON", show_alert=False)

    elif query.data == "stop":
        alerts_enabled = False
        await query.answer("Alerts turned OFF", show_alert=False)


# ---------------------------------------------------------------------------
# Background polling loop (sends proactive signal updates)
# ---------------------------------------------------------------------------
async def poll_and_alert(context: ContextTypes.DEFAULT_TYPE):
    global last_alerted_signal

    if not alerts_enabled:
        return

    price = fetch_price()
    if price is None:
        logger.warning("Skipping this poll cycle — no price returned.")
        return

    price_history.append(price)
    evaluate_pending_signals(price)
    structure = get_market_structure()
    raw_signal = generate_signal()
    signal = confirmed_signal(raw_signal) if raw_signal is not None else None
    signal = apply_structure_filter(signal, structure) if signal else signal

    if signal:
        record_signal(signal)

    rsi = calculate_rsi(price_history)

    # Only alert when the confirmed signal is different from the last one we
    # actually sent — stops repeat BUY/BUY/BUY spam every single cycle.
    if signal in ("BUY", "SELL") and signal != last_alerted_signal:
        msg = build_status_message(price, signal, rsi, structure)
        try:
            await context.bot.send_message(
                chat_id=CHAT_ID,
                text=msg,
                parse_mode="Markdown",
                reply_markup=build_keyboard(),
            )
            last_alerted_signal = signal
            pending_evaluations.append((EVAL_HORIZON_CYCLES, signal, price))
        except Exception as e:
            logger.error(f"Failed to send proactive alert: {e}")

    save_state()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    load_state()

    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("price", price_command))
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("stop", stop_command))
    application.add_handler(CallbackQueryHandler(button_callback))

    application.job_queue.run_repeating(
        poll_and_alert, interval=POLL_INTERVAL_SECONDS, first=10
    )

    flask_thread = Thread(target=run_flask, daemon=True)
    flask_thread.start()

    logger.info("ETH/USDT bot starting...")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
