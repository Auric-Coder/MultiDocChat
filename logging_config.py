"""Centralized logging setup for MultiDocChat."""

from __future__ import annotations

import logging
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOG_FILE = LOGS_DIR / "multidocchat.log"


def setup_logging(level: int = logging.INFO) -> None:
    """Initialize file and console logging."""
    LOGS_DIR.mkdir(exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Prevent duplicate handlers
    if not root_logger.handlers:
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """Return named logger instance."""
    setup_logging()
    return logging.getLogger(name)
