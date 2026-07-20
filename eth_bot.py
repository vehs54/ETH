"""
eth_bot.py
----------
Purpose: everything related to Telegram, scheduling, messaging, and
communication. All trading intelligence lives in backtest_eth.py — this
file's job is: fetch a decision from analyze(), format it, send it, track
state, and respond to commands. It should not contain indicator/structure
math itself.
"""

import os
import sys
import json
import time
import asyncio
import logging
from datetime import datetime, timezone
from threading import Thread
from functools import wraps

import requests
from flask import Flask
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

import backtest_eth as engine

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

REQUIRED_ENV_VARS = ["BOT_TOKEN", "CHAT_ID"]
missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
if missing:
    print(f"Missing required environment variable(s): {', '.join(missing)}", file=sys.stderr)
    print("Set these in your host's Environment tab (e.g. Render) before redeploying.", file=sys.stderr)
    sys.exit(1)

BOT_TOKEN = os.environ["BOT_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL")  # optional, enables self-ping

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POLL_INTERVAL_SECONDS = 900          # 15 minutes
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot_state.json")
SIGNAL_COOLDOWN_SECONDS = 1800       # won't re-alert the same direction within 30 min
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5
CONSECUTIVE_FAILURE_ALERT_THRESHOLD = 6   # ~90 min of failed polls -> notify chat

# ---------------------------------------------------------------------------
# Logging — separate bot/error loggers so error-level noise is easy to grep
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
bot_logger = logging.getLogger("eth_bot")
error_logger = logging.getLogger("eth_bot.errors")

# ---------------------------------------------------------------------------
# State (persisted to STATE_FILE so restarts don't lose stats/active trade —
# note: on Render's free tier this file does NOT survive a redeploy, only a
# process restart, since the disk resets on every new deploy)
# ---------------------------------------------------------------------------

state = {
    "alerts_enabled": True,
    "active_trade": None,          # set when a BUY/SELL signal opens a trade
    "last_alert_direction": None,
    "last_alert_time": None,       # unix seconds
    "performance": engine.new_performance_state(),
    "alert_history": [],           # last N alerts, most recent last
    "last_analysis": None,         # cached result of the most recent analyze()
    "consecutive_failures": 0,
}

ALERT_HISTORY_MAXLEN = 50


def save_state():
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except OSError as e:
        error_logger.error(f"Failed to save state: {e}")


def load_state():
    global state
    if not os.path.exists(STATE_FILE):
        return
    try:
        with open(STATE_FILE) as f:
            saved = json.load(f)
        state.update(saved)
        # performance dict might be missing new keys after an upgrade — backfill
        for k, v in engine.new_performance_state().items():
            state["performance"].setdefault(k, v)
    except (OSError, json.JSONDecodeError) as e:
        error_logger.error(f"Failed to load state, starting fresh: {e}")


# ---------------------------------------------------------------------------
# Retry helper (bot monitoring: retry system)
# ---------------------------------------------------------------------------

def with_retries(func, *args, max_retries=MAX_RETRIES, backoff=RETRY_BACKOFF_SECONDS, **kwargs):
    """Calls func with retries on exception or a falsy/None result, with a
    fixed backoff between attempts. Returns the result, or None if every
    attempt failed."""
    for attempt in range(1, max_retries + 1):
        try:
            result = func(*args, **kwargs)
            if result is not None:
                return result
        except Exception as e:
            error_logger.error(f"{func.__name__} attempt {attempt}/{max_retries} raised: {e}")
        if attempt < max_retries:
            time.sleep(backoff)
    return None


# ---------------------------------------------------------------------------
# Bot Monitoring: connectivity checks
# ---------------------------------------------------------------------------

def check_internet():
    try:
        requests.get("https://api.coinbase.com", timeout=5)
        return True
    except requests.exceptions.RequestException:
        return False


def check_api_connection():
    price = engine.fetch_spot_price()
    return price is not None


# ---------------------------------------------------------------------------
# Flask keep-alive server (Render free tier)
# ---------------------------------------------------------------------------

flask_app = Flask(__name__)


@flask_app.route("/")
def home():
    return "ETH signal bot is alive.", 200


@flask_app.route("/health")
def health():
    ok = check_internet() and check_api_connection()
    return ({"status": "ok" if ok else "degraded"}, 200 if ok else 503)


def run_flask():
    port = int(os.environ.get("PORT", 10000))
    flask_app.run(host="0.0.0.0", port=port)


def self_ping_job():
    """Optional: pings this service's own public URL every 10 minutes so
    Render's free tier doesn't spin it down from inactivity. Only runs if
    RENDER_EXTERNAL_URL is set."""
    if not RENDER_EXTERNAL_URL:
        return
    while True:
        try:
            requests.get(RENDER_EXTERNAL_URL, timeout=10)
        except requests.exceptions.RequestException as e:
            error_logger.warning(f"Self-ping failed: {e}")
        time.sleep(600)


# ---------------------------------------------------------------------------
# Message Formatting
# ---------------------------------------------------------------------------

SIGNAL_EMOJI = {"BUY": "🟢", "SELL": "🔴", "WAIT": "⏸️", "WARNING": "⚠️", "EXIT": "🚪"}


def format_analysis_message(result):
    """Builds the full Telegram message (Markdown) from an analyze() result dict."""
    signal = result["signal"]
    emoji = SIGNAL_EMOJI.get(signal, "❓")
    lines = [f"{emoji} *ETH/USDT — {signal}*", f"💰 Price: ${result['price']:,.2f}"]

    if signal in ("BUY", "SELL") and result.get("stop_loss") is not None:
        tp1, tp2, tp3 = result["take_profits"]
        lines.append(f"\n📍 Entry: ${result['entry']:,.2f}")
        lines.append(f"🛑 Stop Loss: ${result['stop_loss']:,.2f}")
        lines.append(f"🎯 TP1: ${tp1:,.2f} | TP2: ${tp2:,.2f} | TP3: ${tp3:,.2f}")
        lines.append(f"⚖️ Risk:Reward 1:{result['risk_reward']:.1f}")
        lines.append(f"\n🔎 Confidence: {result['confidence']}%")
        lines.append("Reason:\n" + "\n".join(result["reasons"]))

    structure = result["structure"]
    lines.append(f"\n📊 Trend: {result['trend']} (strength: {structure.get('strength')})")
    if result.get("support"):
        lines.append(f"🟩 Support: ${result['support']:,.2f}")
    if result.get("resistance"):
        lines.append(f"🟥 Resistance: ${result['resistance']:,.2f}")

    vol = result["volume"]
    if vol.get("avg_volume") is not None:
        lines.append(f"\n📦 Volume: {vol['pressure']} pressure"
                      f"{' (spike)' if vol.get('spike') else ''}"
                      f"{', divergence: ' + vol['divergence'] if vol.get('divergence') else ''}")

    pattern = result["pattern"]
    lines.append(f"🕯️ Pattern: {pattern.get('pattern')} ({pattern.get('bias')})")

    mtf = result["mtf"]
    lines.append(f"\n🌐 Timeframe agreement: {mtf['agreement_pct']}% ({mtf['dominant']})"
                  f"{' ⚠️ conflict' if mtf.get('conflict') else ''}")

    if result.get("pullback"):
        pb = result["pullback"]
        lines.append(f"↩️ {pb['depth']} ({pb['atr_multiple']}x ATR)")

    lines.append(f"\n🕐 {result['timestamp']}")
    return "\n".join(lines)


def format_stats_message():
    perf = state["performance"]
    lines = [
        "📊 *Signal Stats*",
        f"Total signals: {perf['total_signals']}",
        f"BUY: {perf['buy']} | SELL: {perf['sell']} | WAIT: {perf['wait']} | "
        f"WARNING: {perf['warning']} | EXIT: {perf['exit']}",
    ]
    return "\n".join(lines)


def format_accuracy_message():
    perf = state["performance"]
    acc = engine.accuracy_pct(perf)
    acc_display = f"{acc}%" if acc is not None else "N/A (no closed trades yet)"
    lines = [
        "🏆 *Accuracy*",
        f"Wins: {perf['wins']} | Losses: {perf['losses']} | Neutral: {perf['neutral']}",
        f"Accuracy: {acc_display}",
        f"Current streak: {perf['current_streak']} ({perf['streak_type'] or 'none'})",
        f"Best streak: {perf['best_streak']}",
    ]
    return "\n".join(lines)


def format_settings_message():
    lines = [
        "⚙️ *Settings*",
        f"Alerts: {'ON' if state['alerts_enabled'] else 'OFF'}",
        f"Poll interval: {POLL_INTERVAL_SECONDS // 60} min",
        f"Signal cooldown: {SIGNAL_COOLDOWN_SECONDS // 60} min",
        f"Min confidence for BUY/SELL: {engine.MIN_CONFIDENCE_FOR_SIGNAL}%",
        f"Active trade: {'Yes (' + state['active_trade']['direction'] + ')' if state['active_trade'] else 'None'}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Signal Management
# ---------------------------------------------------------------------------

def is_on_cooldown(direction):
    if state["last_alert_direction"] != direction or state["last_alert_time"] is None:
        return False
    return (time.time() - state["last_alert_time"]) < SIGNAL_COOLDOWN_SECONDS


def record_alert(result):
    state["last_alert_direction"] = result["signal"]
    state["last_alert_time"] = time.time()
    state["alert_history"].append({
        "signal": result["signal"], "price": result["price"],
        "confidence": result["confidence"], "time": result["timestamp"],
    })
    state["alert_history"] = state["alert_history"][-ALERT_HISTORY_MAXLEN:]
    engine.record_signal_in_state(state["performance"], result["signal"])


async def send_alert(context, result):
    try:
        await context.bot.send_message(
            chat_id=CHAT_ID, text=format_analysis_message(result),
            parse_mode="Markdown", reply_markup=build_keyboard(),
        )
    except Exception as e:
        error_logger.error(f"Failed to send alert: {e}")


# ---------------------------------------------------------------------------
# Inline Buttons
# ---------------------------------------------------------------------------

def build_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 Refresh", callback_data="refresh"),
         InlineKeyboardButton("🔎 Force Analysis", callback_data="analysis")],
        [InlineKeyboardButton("▶️ Start", callback_data="start"),
         InlineKeyboardButton("⏹ Stop", callback_data="stop")],
        [InlineKeyboardButton("📈 Status", callback_data="status"),
         InlineKeyboardButton("💰 Price", callback_data="price")],
        [InlineKeyboardButton("📊 Stats", callback_data="stats"),
         InlineKeyboardButton("🏆 Accuracy", callback_data="accuracy")],
    ])


# ---------------------------------------------------------------------------
# Core analysis + active-trade update (shared by commands and the poll loop)
# ---------------------------------------------------------------------------

def run_analysis_cycle():
    """Fetches a fresh analyze() result with retries, updates active-trade
    monitoring and performance stats, and caches the result. Returns the
    result dict, or None if all retries failed."""
    result = with_retries(engine.analyze)

    if result is None:
        state["consecutive_failures"] += 1
        error_logger.error(f"analyze() failed after {MAX_RETRIES} retries "
                            f"(consecutive failures: {state['consecutive_failures']})")
        return None

    state["consecutive_failures"] = 0
    state["last_analysis"] = result

    # Active trade monitoring: check the open trade against this cycle's
    # structure/price before considering any new signal.
    if state["active_trade"] is not None:
        # atr_val isn't used inside monitor_trade's own logic (SL/TP/CHoCH are
        # all price/structure based) so it's safe to pass None here.
        outcome = engine.monitor_trade(
            state["active_trade"], result["price"], None, result["structure"],
        )
        if outcome in ("SL_HIT", "TP1_HIT", "TP2_HIT", "TP3_HIT", "EXIT"):
            win = outcome != "SL_HIT" and outcome != "EXIT"
            engine.record_outcome_in_state(state["performance"], "WIN" if win else "LOSS")
            state["active_trade"] = None

    save_state()
    return result


# ---------------------------------------------------------------------------
# Telegram Commands
# ---------------------------------------------------------------------------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state["alerts_enabled"] = True
    save_state()
    await update.message.reply_text(
        "▶️ Alerts turned ON. You'll get a message whenever a confirmed BUY/SELL signal fires.",
        reply_markup=build_keyboard(),
    )


async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state["alerts_enabled"] = False
    save_state()
    await update.message.reply_text("⏹ Alerts turned OFF. Use /start to resume, or /status anytime.")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    result = state["last_analysis"]
    if result is None:
        await update.message.reply_text("No analysis cached yet — try /analysis for a fresh read.")
        return
    await update.message.reply_text(format_analysis_message(result), parse_mode="Markdown", reply_markup=build_keyboard())


async def price_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    price = with_retries(engine.fetch_spot_price, max_retries=2)
    if price is not None:
        await update.message.reply_text(f"Ξ ETH/USDT: ${price:,.2f}")
    else:
        await update.message.reply_text("⚠️ Couldn't fetch the current ETH price. Try again shortly.")


async def analysis_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔎 Running fresh analysis...")
    result = run_analysis_cycle()
    if result is None:
        await update.message.reply_text("⚠️ Analysis failed — network or API issue. Try again shortly.")
        return
    await update.message.reply_text(format_analysis_message(result), parse_mode="Markdown", reply_markup=build_keyboard())


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(format_stats_message(), parse_mode="Markdown")


async def accuracy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(format_accuracy_message(), parse_mode="Markdown")


async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(format_settings_message(), parse_mode="Markdown")


async def refresh_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await analysis_command(update, context)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "*Commands*\n"
        "/start - turn alerts on\n"
        "/stop - turn alerts off\n"
        "/status - show the last cached analysis\n"
        "/price - current spot price only\n"
        "/analysis or /refresh - run a fresh analysis now\n"
        "/settings - show current configuration\n"
        "/stats - signal counts\n"
        "/accuracy - win/loss record\n"
        "/help - this message",
        parse_mode="Markdown", reply_markup=build_keyboard(),
    )


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    action = query.data

    if action == "start":
        state["alerts_enabled"] = True
        save_state()
        await query.answer("Alerts turned ON", show_alert=False)
    elif action == "stop":
        state["alerts_enabled"] = False
        save_state()
        await query.answer("Alerts turned OFF", show_alert=False)
    elif action == "price":
        price = with_retries(engine.fetch_spot_price, max_retries=2)
        text = f"Ξ ETH/USDT: ${price:,.2f}" if price is not None else "⚠️ Couldn't fetch price."
        await context.bot.send_message(chat_id=query.message.chat_id, text=text)
    elif action == "status":
        result = state["last_analysis"]
        if result is None:
            await context.bot.send_message(chat_id=query.message.chat_id, text="No analysis cached yet.")
        else:
            await context.bot.send_message(chat_id=query.message.chat_id, text=format_analysis_message(result),
                                            parse_mode="Markdown", reply_markup=build_keyboard())
    elif action == "stats":
        await context.bot.send_message(chat_id=query.message.chat_id, text=format_stats_message(), parse_mode="Markdown")
    elif action == "accuracy":
        await context.bot.send_message(chat_id=query.message.chat_id, text=format_accuracy_message(), parse_mode="Markdown")
    elif action in ("refresh", "analysis"):
        result = run_analysis_cycle()
        if result is None:
            await context.bot.send_message(chat_id=query.message.chat_id, text="⚠️ Analysis failed — try again shortly.")
        else:
            await context.bot.send_message(chat_id=query.message.chat_id, text=format_analysis_message(result),
                                            parse_mode="Markdown", reply_markup=build_keyboard())


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    error_logger.error(f"Update {update} caused error: {context.error}")


# ---------------------------------------------------------------------------
# Scheduler: poll every 15 minutes, alert on new confirmed signals
# ---------------------------------------------------------------------------

async def poll_and_alert(context: ContextTypes.DEFAULT_TYPE):
    if not state["alerts_enabled"]:
        return

    result = run_analysis_cycle()
    if result is None:
        if state["consecutive_failures"] == CONSECUTIVE_FAILURE_ALERT_THRESHOLD:
            try:
                await context.bot.send_message(
                    chat_id=CHAT_ID,
                    text=f"⚠️ {CONSECUTIVE_FAILURE_ALERT_THRESHOLD} consecutive analysis failures — "
                         f"check the bot's network/API access.",
                )
            except Exception as e:
                error_logger.error(f"Failed to send failure-streak alert: {e}")
        return

    signal = result["signal"]
    if signal not in ("BUY", "SELL", "WARNING", "EXIT"):
        return  # WAIT never alerts

    if signal in ("BUY", "SELL"):
        if is_on_cooldown(signal):
            return
        await send_alert(context, result)
        record_alert(result)
        if state["active_trade"] is None:
            state["active_trade"] = {
                "direction": signal, "entry": result["entry"],
                "stop_loss": result["stop_loss"], "take_profits": result["take_profits"],
            }
    else:
        # WARNING/EXIT relate to an existing trade — only worth sending if one is open
        if state["active_trade"] is not None:
            await send_alert(context, result)
            record_alert(result)

    save_state()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_state()

    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("stop", stop_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("price", price_command))
    application.add_handler(CommandHandler("analysis", analysis_command))
    application.add_handler(CommandHandler("refresh", refresh_command))
    application.add_handler(CommandHandler("settings", settings_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("accuracy", accuracy_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_error_handler(error_handler)

    application.job_queue.run_repeating(poll_and_alert, interval=POLL_INTERVAL_SECONDS, first=10)

    Thread(target=run_flask, daemon=True).start()
    if RENDER_EXTERNAL_URL:
        Thread(target=self_ping_job, daemon=True).start()

    bot_logger.info("ETH/USDT bot starting...")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
