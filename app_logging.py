"""Small structured-logging helpers for the financial dashboard."""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import TextIO
from uuid import uuid4


LOGGER_NAME = "financial_dashboard"
_REQUEST_ID: ContextVar[str] = ContextVar(
    "financial_dashboard_request_id",
    default="unassigned",
)


class JsonLogFormatter(logging.Formatter):
    """Format one log event as a machine-readable JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(
                record.created,
                timezone.utc,
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", record.getMessage()),
            "request_id": _REQUEST_ID.get(),
        }
        context = getattr(record, "context", {})
        if isinstance(context, dict):
            for key, value in context.items():
                if key not in payload:
                    payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, sort_keys=True)


def configure_logging(stream: TextIO | None = None) -> logging.Logger:
    """Configure the application logger once and return it."""
    application_logger = logging.getLogger(LOGGER_NAME)
    application_logger.setLevel(logging.INFO)
    application_logger.propagate = False

    has_application_handler = any(
        getattr(handler, "financial_dashboard_handler", False)
        for handler in application_logger.handlers
    )
    if not has_application_handler:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonLogFormatter())
        handler.financial_dashboard_handler = True
        application_logger.addHandler(handler)
    return application_logger


def get_logger(component: str) -> logging.Logger:
    """Return a child logger that shares the dashboard JSON handler."""
    configure_logging()
    return logging.getLogger(f"{LOGGER_NAME}.{component}")


def start_request(request_id: str | None = None) -> str:
    """Start one correlated Streamlit rerun and return its request ID."""
    assigned_request_id = request_id or uuid4().hex
    _REQUEST_ID.set(assigned_request_id)
    return assigned_request_id


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    *,
    exc_info: bool = False,
    **context: object,
) -> None:
    """Write a stable event name plus a small allowlisted context mapping."""
    logger.log(
        level,
        event,
        extra={"event": event, "context": context},
        exc_info=exc_info,
    )
