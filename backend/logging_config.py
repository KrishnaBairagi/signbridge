"""
backend/logging_config.py — Centralized Backend Application Logging.

Configures Python's standard logging system for SignBridge backend events:
server startup, session lifecycle, model loading, prediction confirmations,
and error payloads (without per-frame logging spam or logging sensitive data).
"""

from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("signbridge")
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logging()
