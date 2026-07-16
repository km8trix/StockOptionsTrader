"""Causal liquidity measurements shared by execution simulations."""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd


def has_executable_share_capacity(adv, participation_cap: float) -> bool:
    """Whether the ADV cap permits at least one whole share."""
    if adv is None or not math.isfinite(float(adv)) or adv <= 0:
        return False
    return int(participation_cap * adv) >= 1


def capped_fill_quantity(desired: int, adv, participation_cap: float,
                         *, strict: bool = False):
    """Return whole-share ``(fill, remainder)`` under an ADV cap.

    Legacy simulations floor a positive-ADV cap at one share. Strict research
    simulations instead return a zero fill when the fractional capacity is
    below one share; their pending-intent loop then defers and expires it.
    """
    if adv is None or not math.isfinite(float(adv)) or adv <= 0:
        return desired, 0
    cap = int(participation_cap * adv)
    if cap < 1:
        cap = 0 if strict else 1
    return (desired, 0) if desired <= cap else (cap, desired - cap)


def requeue_remainder(intent: Dict, remainder: int,
                      enabled: bool) -> Optional[Dict]:
    """Build a later-day intent for an unfilled whole-share remainder."""
    if remainder <= 0 or not enabled:
        return None
    follow = dict(intent)
    follow['quantity'] = remainder
    follow['accumulate'] = intent['signal'] in ('BUY', 'SHORT')
    follow['liquidity_remainder'] = True
    follow['days_waiting'] = 0
    return follow


def trailing_average_daily_volume(data, date, window: int):
    """Mean valid volume strictly before ``date``; ``None`` if unavailable."""
    if data is None or "volume" not in getattr(data, "columns", []):
        return None
    prior = data[data.index < pd.Timestamp(date)].sort_index()
    if prior.empty:
        return None
    volumes = pd.to_numeric(
        prior["volume"].tail(window), errors="coerce").dropna()
    values = volumes.to_numpy(dtype=float)
    if not len(values) or (values < 0).any() or not np.isfinite(values).all():
        return None
    average = float(values.mean())
    return average if math.isfinite(average) and average > 0 else None


def require_credible_delisting_terms(
        strict: bool, source: str, quality_flags, symbol: str) -> None:
    """Invalidate qualifying research when terminal stock value is unknown.

    A documented cash/zero-recovery corporate-action term is a settlement,
    not a market fill.  Falling back to an uncapped final close is acceptable
    only for legacy diagnostics because it overstates executable exit value.
    """
    if strict and (source == 'final_tradable_close'
                   or 'delisting_terms_unavailable' in quality_flags):
        raise RuntimeError(
            f"qualifying research lacks delisting terms for {symbol}")


__all__ = [
    "capped_fill_quantity",
    "has_executable_share_capacity",
    "requeue_remainder",
    "require_credible_delisting_terms",
    "trailing_average_daily_volume",
]
