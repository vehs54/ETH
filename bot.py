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

POLL_INTERVAL_SECONDS = 1800   # 30 minutes
HISTORY_MAXLEN = 60
RSI_PERIOD = 15
MA_SHORT = 5
MA_MED = 10
MA_LONG = 30
ROLLING_WINDOW_HOURS = 3
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot_state.json")

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
price_history = deque(maxlen=HISTORY_MAXLEN)
signal_log = deque()
alerts_enabled = True         # controlled by /start and /stop
last_alerted_signal = None    # tracks last signal actually sent, to avoid repeats

# --- Signal confidence filtering ---
MIN_MA_SPREAD_PCT = 0.08
RSI_BUY_CEILING = 60
RSI_SELL_FLOOR = 40
CONFIRMATION_STREAK = 3
recent_raw_signals = deque(maxlen=CONFIRMATION_STREAK)

# --- Live accuracy tracking ---
EVAL_HORIZON_CYCLES = 6        # cycles ahead to check outcome (6 * 30min poll = 3 hours)
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
def build_status_message(price, signal, rsi):
    ma5 = moving_average(price_history, MA_SHORT)
    ma10 = moving_average(price_history, MA_MED)
    ma30 = moving_average(price_history, MA_LONG)
    breakdown = get_rolling_breakdown()

    rsi_display = f"{rsi}" if rsi is not None else "N/A (gathering data)"
    ma5_display = f"{ma5:.2f}" if ma5 is not None else "N/A"
    ma10_display = f"{ma10:.2f}" if ma10 is not None else "N/A"
    ma30_display = f"{ma30:.2f}" if ma30 is not None else "N/A"
    signal_display = signal if signal is not None else "Gathering data..."

    msg = (
        f"Ξ *ETH/USDT Signal*\n\n"
        f"💰 Price: ${price:,.2f}\n"
        f"📊 Signal: *{signal_display}*\n\n"
        f"MA5: {ma5_display} | MA10: {ma10_display} | MA30: {ma30_display}\n"
        f"RSI(15): {rsi_display}\n\n"
        f"📈 Last {ROLLING_WINDOW_HOURS}h breakdown:\n"
        f"  BUY: {breakdown['BUY']}% | SELL: {breakdown['SELL']}% | HOLD: {breakdown['HOLD']}%\n\n"
        f"🎯 {get_accuracy_display()}\n\n"
        f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
    )
    return msg


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
        signal = generate_signal()
        if signal:
            record_signal(signal)
        rsi = calculate_rsi(price_history)
        msg = build_status_message(price, signal, rsi)
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
            signal = generate_signal()
            if signal:
                record_signal(signal)
            rsi = calculate_rsi(price_history)
            msg = build_status_message(price, signal, rsi)
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
    raw_signal = generate_signal()
    signal = confirmed_signal(raw_signal) if raw_signal is not None else None

    if signal:
        record_signal(signal)

    rsi = calculate_rsi(price_history)

    # Only alert when the confirmed signal is different from the last one we
    # actually sent — stops repeat BUY/BUY/BUY spam every single cycle.
    if signal in ("BUY", "SELL") and signal != last_alerted_signal:
        msg = build_status_message(price, signal, rsi)
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
