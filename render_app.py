# render_app.py
import os
import asyncio
import logging
from fastapi import FastAPI, Request, HTTPException
from aiogram import Bot
from aiogram.types import Update

# dp из твоего bot.py
try:
    from bot import dp  # глобальный Dispatcher
except Exception:
    from bot import get_dispatcher
    dp = get_dispatcher()

TOKEN = os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("Set TELEGRAM_TOKEN or TELEGRAM_BOT_TOKEN")

SECRET = os.getenv("WEBHOOK_SECRET", "dev-secret")
WEBHOOK_PATH = f"/webhook/{SECRET}"

bot = Bot(TOKEN)
app = FastAPI()

@app.get("/")
async def health():
    return {"ok": True}

@app.post(WEBHOOK_PATH)
async def telegram_webhook(request: Request):
    data = await request.json()
    upd = Update.model_validate(data)  # aiogram v3
    await dp.feed_update(bot, upd)
    return {"ok": True}

@app.on_event("startup")
async def on_startup():
    # не блокируем старт, всё тяжёлое — в фоне
    asyncio.create_task(_warmup())

async def _warmup():
    try:
        base = (os.getenv("RENDER_EXTERNAL_URL") or os.getenv("BASE_URL") or "").rstrip("/")
        if base:
            url = f"{base}{WEBHOOK_PATH}"
            try:
                # ограничиваем сетевой таймаут, чтобы не зависать
                await bot.set_webhook(url, drop_pending_updates=True, request_timeout=10)
                logging.info("Webhook set to %s", url)
            except Exception as e:
                logging.warning("set_webhook failed: %s", e)
        else:
            logging.warning("RENDER_EXTERNAL_URL is empty; will try again on next restart")
    except Exception as e:
        logging.warning("warmup(webhook) error: %s", e)

    # подгрузка данных/модели — в отдельном потоке, чтобы не блокировать event loop
    try:
        from bot import ensure_data  # должна ничего не делать, если уже инициализировано
        await asyncio.to_thread(ensure_data)
        logging.info("Data/model warmup completed")
    except Exception as e:
        logging.warning("warmup(data) skipped: %s", e)
