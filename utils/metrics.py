"""Phase 4 ops: minimal in-process metrics (Prometheus text exposition).

stdlib-only — no prometheus_client dependency. Counters are process-local,
which is the correct scope: the app runs a single gunicorn worker (load-bearing,
see README), so there is exactly one registry to scrape.
"""
from __future__ import annotations

import threading
import time

_lock = threading.Lock()
_counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
_START = time.monotonic()


def inc(name: str, labels: dict[str, str] | None = None,
        amount: float = 1.0) -> None:
    """Increment a named counter (optionally label-scoped)."""
    key = (name, tuple(sorted((labels or {}).items())))
    with _lock:
        _counters[key] = _counters.get(key, 0.0) + amount


def reset() -> None:
    """Clear all counters — for tests only."""
    with _lock:
        _counters.clear()


def _format_labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    return "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}"


def render() -> str:
    """Render all metrics in Prometheus text exposition format."""
    lines = [
        "# HELP app_uptime_seconds Seconds since the process started.",
        "# TYPE app_uptime_seconds gauge",
        f"app_uptime_seconds {time.monotonic() - _START:.1f}",
    ]
    with _lock:
        snapshot = dict(_counters)
    by_name: dict[str, list[tuple[tuple[tuple[str, str], ...], float]]] = {}
    for (name, labels), value in snapshot.items():
        by_name.setdefault(name, []).append((labels, value))
    for name in sorted(by_name):
        lines.append(f"# TYPE {name} counter")
        for labels, value in sorted(by_name[name]):
            lines.append(f"{name}{_format_labels(labels)} {value:g}")
    return "\n".join(lines) + "\n"
