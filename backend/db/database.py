"""
db/database.py

§15 Privacy: conversations are NOT stored by default. This module only
persists a transcript if the client explicitly sets `consent_to_store=True`
when starting a session. Camera frames and raw audio are never written to
disk anywhere in this codebase -- only derived text (recognized tokens,
sentences, transcribed speech) is ever eligible for storage, and only with
consent.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from backend.config import get_settings

DB_PATH = get_settings().db_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    consent_to_store INTEGER NOT NULL DEFAULT 0,
    vocabulary_mode TEXT NOT NULL DEFAULT 'demo'
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    direction TEXT NOT NULL,      -- 'sign_to_speech' | 'speech_to_text'
    content TEXT NOT NULL,
    confidence REAL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions (id)
);
"""


@contextmanager
def get_connection():
    db_path = get_settings().db_path
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()



def init_db():
    with get_connection() as conn:
        conn.executescript(SCHEMA)


def create_session(session_id: str, consent_to_store: bool, vocabulary_mode: str = "demo"):
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO sessions (id, started_at, consent_to_store, vocabulary_mode) VALUES (?, ?, ?, ?)",
            (session_id, datetime.now(timezone.utc).isoformat(), int(consent_to_store), vocabulary_mode),
        )


def log_message(session_id: str, direction: str, content: str, confidence: float | None = None):
    """No-ops unless the session opted in to storage."""
    with get_connection() as conn:
        row = conn.execute("SELECT consent_to_store FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if not row or not row["consent_to_store"]:
            return
        conn.execute(
            "INSERT INTO messages (session_id, direction, content, confidence, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, direction, content, confidence, datetime.now(timezone.utc).isoformat()),
        )
