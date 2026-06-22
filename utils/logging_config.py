"""
Structured logging configuration for the trading system.

Usage:
    from utils.logging_config import setup_logging
    setup_logging()  # level read from LOG_LEVEL env var, default INFO

Individual modules should then create their own logger:
    import logging
    logger = logging.getLogger(__name__)
"""

from __future__ import annotations

import contextvars
import logging
import os
import uuid

# Correlation id is carried in a ContextVar so it follows the logical flow
# across threads (the request thread, JobManager workers, the scheduler) and
# is injected into every log record by the filter below.
_correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default="-")

LOG_FORMAT = "%(asctime)s %(levelname)s [%(correlation_id)s] %(name)s: %(message)s"


def set_correlation_id(cid: str) -> None:
    """Bind a correlation id to the current context (request/job/cycle)."""
    _correlation_id.set(cid)


def get_correlation_id() -> str:
    """Current correlation id, or '-' when none is bound."""
    return _correlation_id.get()


def new_correlation_id() -> str:
    """Generate, bind, and return a fresh short correlation id."""
    cid = uuid.uuid4().hex[:12]
    _correlation_id.set(cid)
    return cid


class CorrelationIdFilter(logging.Filter):
    """Stamps every record with the current correlation id so LOG_FORMAT can
    reference %(correlation_id)s without each call site supplying it."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = _correlation_id.get()
        return True


def setup_logging(level: str | int | None = None) -> None:
    """Configure root logging for the application.

    Args:
        level: explicit logging level (name like "DEBUG" or numeric constant).
            When None, the level is read from the LOG_LEVEL environment
            variable, defaulting to INFO. Unknown level names fall back to
            INFO rather than raising.
    """
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.strip().upper(), logging.INFO)
        if not isinstance(level, int):
            level = logging.INFO

    # No force=True: basicConfig is a no-op when handlers already exist
    # (pytest, gunicorn), so we never clobber theirs. We still attach the
    # correlation filter to whatever handlers are present.
    logging.basicConfig(level=level, format=LOG_FORMAT)
    root = logging.getLogger()
    for handler in root.handlers:
        if not any(isinstance(f, CorrelationIdFilter) for f in handler.filters):
            handler.addFilter(CorrelationIdFilter())
