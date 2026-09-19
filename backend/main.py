"""
backend/main.py — SignBridge API server.

Run with:  uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router as api_router
from backend.websocket.live_conversation import live_conversation_endpoint
from backend.db.database import init_db

app = FastAPI(
    title="SignBridge API",
    description="Real-time sign language communication assistant (demo prototype).",
    version="0.1.0",
)

ALLOWED_ORIGINS = os.getenv(
    "CORS_ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5500,http://127.0.0.1:8000,http://localhost:8000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(api_router)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"


@app.get("/")
def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/index.html")
def serve_index_html():
    return FileResponse(FRONTEND_DIR / "index.html")


# Serve static MediaPipe vendor assets under /vendor without overshadowing API routes.
app.mount('/vendor', StaticFiles(directory=str(FRONTEND_DIR / 'vendor')), name='vendor')


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok", "demo_mode": os.getenv("DEMO_MODE", "true")}


@app.websocket("/ws/live-conversation")
async def ws_live_conversation(websocket: WebSocket):
    await live_conversation_endpoint(websocket)

