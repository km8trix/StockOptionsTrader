"""Point-in-time size bucketing for size-conditioned cross-sectional signals.

The insider edge is validated WITHIN size buckets; pooling across sizes dilutes
it. The trap: the tickers table's ``scalemarketcap`` band is the CURRENT band (as
of data download), NOT point-in-time — bucketing on it leaks the future (a name's
band reflects where it ended up). This module buckets on POINT-IN-TIME market cap
(``PitWarehouse.daily_metric(..., 'marketcap')``, known ON the date) by
cross-sectional quantile, so it is both PIT and unit-agnostic — only the ordering
of caps matters, no magic dollar thresholds that could drift with the data unit.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def size_buckets(marketcaps: Dict[str, float], n: int = 3) -> Dict[str, int]:
    """Assign each name to a size bucket ``0..n-1`` by cross-sectional market-cap
    quantile (0 = smallest, n-1 = largest), split into n roughly equal groups.

    Names with missing / non-finite / non-positive cap are dropped. Returns ``{}``
    if fewer than ``n`` names have a usable cap (can't form n balanced buckets).
    """
    valid = {k: float(v) for k, v in marketcaps.items()
             if v is not None and np.isfinite(v) and v > 0}
    if len(valid) < n:
        return {}
    order = sorted(valid, key=lambda k: valid[k])
    m = len(order)
    # rank i (0-based, ascending cap) -> bucket floor(i*n/m), clamped to n-1
    return {name: min(n - 1, i * n // m) for i, name in enumerate(order)}


def pit_marketcaps(provider, names: Sequence[str], date) -> Dict[str, float]:
    """Point-in-time market cap per name on ``date`` via ``daily_metric`` — the
    PIT size source (NOT ``scalemarketcap``, which is current-not-PIT). Names with
    no daily row on ``date`` are omitted."""
    out: Dict[str, float] = {}
    for s in names:
        mc = provider.daily_metric(s, date, 'marketcap')
        if mc is not None:
            out[s] = mc
    return out
