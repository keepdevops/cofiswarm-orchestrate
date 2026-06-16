"""Structured JSON logging via structlog.

Standard fields: ts (ISO-8601), level, logger, event, request_id, agent_id,
mode, backend. Output goes to stderr by default; configure_logging() can
redirect to agent_logs/<name>.jsonl.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import structlog

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = REPO_ROOT / "agent_logs"


def configure_logging(
    log_name: str | None = None,
    level: int = logging.INFO,
    log_dir: Path = DEFAULT_LOG_DIR,
) -> None:
    """Configure structlog + stdlib logging to emit JSON lines.

    If log_name is given, also tee to agent_logs/<name>.jsonl. Otherwise stderr only.
    Idempotent: safe to call multiple times.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_name:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / f"{log_name}.jsonl"))

    logging.basicConfig(
        format="%(message)s",
        handlers=handlers,
        level=level,
        force=True,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(sort_keys=True),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> Any:
    return structlog.get_logger(name)


def bind_context(**fields: Any) -> None:
    """Bind request-scoped fields (request_id, mode, agent_id) into structlog context."""
    structlog.contextvars.bind_contextvars(**fields)


def clear_context() -> None:
    structlog.contextvars.clear_contextvars()
