
import os
from fastapi import FastAPI, Request, HTTPException
from aiogram import Bot
from aiogram.types import Update
from bot import dp  

TOKEN = os.environ["TELEGRAM_TOKEN"]
SECRET = os.environ.get("WEBHOOK_SECRET", "dev-secret")
BASE = os.environ.get("RENDER_EXTERNAL_URL", "").rstrip("/")
if not BASE:
    BASE = os.environ.get("BASE_URL", "").rstrip("/")

WEBHOOK_PATH = f"/webhook/{SECRET}"
WEBHOOK_URL = f"{BASE}{WEBHOOK_PATH}" if BASE else None

bot = Bot(TOKEN)
app = FastAPI()

@app.on_event("startup")
async def on_startup():
    if not WEBHOOK_URL:
        raise RuntimeError("RENDER_EXTERNAL_URL/BASE_URL не установлен(а)")
    await bot.set_webhook(WEBHOOK_URL, drop_pending_updates=True)

@app.on_event("shutdown")
async def on_shutdown():
    await bot.delete_webhook()

@app.get("/")
async def health():
    return {"ok": True}

@app.post(WEBHOOK_PATH)
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.model_validate(data)
    await dp.feed_update(bot, update)
    return {"ok": True}
