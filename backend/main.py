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
from backend.config import get_settings
from backend.logging_config import logger

app = FastAPI(
    title="SignBridge API",
    description="Real-time sign language communication assistant (demo prototype).",
    version="0.1.0",
)

current_settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=current_settings.cors_allowed_origins,
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
    s = get_settings()
    logger.info(f"SignBridge backend started | Mode: {'demo' if s.demo_mode else 'production'} | Model: {s.onnx_model_path}")


@app.get("/health")
def health():
    s = get_settings()
    return {"status": "ok", "demo_mode": str(s.demo_mode).lower()}


@app.websocket("/ws/live-conversation")
async def ws_live_conversation(websocket: WebSocket):
    await live_conversation_endpoint(websocket)

