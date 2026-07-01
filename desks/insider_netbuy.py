"""Insider net-buy desk — graduating the validated insider signal to a Desk.

The full-universe screen (scripts/insider_screen.py) found a 63d insider
buy-vs-sell long/short that clears the honest bar WITHIN size buckets (pooling
across sizes dilutes it). This desk trades ONE point-in-time size sleeve: within
its band it longs the strongest net-BUYERS and shorts the strongest net-SELLERS,
on the shared CrossSectionalLongShortDesk book. Run three (micro/small/mid) and
combine with the capital allocator (see docs/insider_desk_spec.md).

Two traps this desk is built around (both fixed in the foundation):
  * SIZE is point-in-time — bucket by ``daily_metric`` market cap, NEVER the
    current-not-PIT ``scalemarketcap`` band (data/size_buckets.py).
  * PRICES are survivorship-free — run under
    ``BacktestEngine(desk=..., market_data=WarehouseMarketData())`` so delisted
    names are still priced (data/warehouse_feed.py).

FIXED factor: committee=[] so walk_forward_fits=[] and validation runs at
n_trials=1 (no deflation) — honest, the rule is pre-specified. Wide RiskManager
(the signal is monthly; a 2% price stop would churn it).

PERFORMANCE: the base calls ``_alpha_scores`` every simulated day, but the
signal only changes monthly, and each score needs PIT provider reads (marketcap +
insider aggregate) per name — far too slow to redo daily. So scores are cached
per calendar month and only recomputed when the month rolls over.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from data.size_buckets import pit_marketcaps, size_buckets
from desks.cross_sectional import CrossSectionalLongShortDesk
from portfolio.risk_manager import RiskManager

_BANDS = {'micro': 0, 'small': 1, 'mid': 2}   # ascending PIT market-cap tercile


class InsiderNetBuyDesk(CrossSectionalLongShortDesk):
    """Long strongest net-buyers / short strongest net-sellers, within one PIT
    size band (tercile). ``provider`` exposes daily_metric + insider_net_buys
    (PitWarehouse or PitCache); it is queried with the simulated ``date`` as the
    point-in-time boundary."""

    def __init__(self, band: str = 'small', *, provider=None,
                 capital_allocation: float = 1.0,
                 risk_manager: Optional[RiskManager] = None,
                 lookback_days: int = 90, quantile: float = 0.2,
                 n_bands: int = 3, long_only: bool = False):
        if band not in _BANDS:
            raise ValueError(f"band {band!r} must be one of {list(_BANDS)}")
        if provider is None:
            from data.pit_warehouse import PitWarehouse
            provider = PitWarehouse()
        if risk_manager is None:
            risk_manager = RiskManager(position_stop_loss=0.50)
        super().__init__(
            key=f'insider_{band}',
            name=f'Insider Net-Buy ({band}-cap)',
            description=('Long the strongest insider net-buyers, short the '
                         f'strongest net-sellers, within the PIT {band}-cap '
                         'tercile (trailing insider window).'),
            accent='#1f883d',
            note_label=f'Insider-{band}',
            reason_prefix=f'insider-{band}',
            committee=[],
            model_label=f'insider net-buy ({band}-cap, fixed factor)',
            capital_allocation=capital_allocation,
            risk_manager=risk_manager,
            quantile=quantile,
            long_only=long_only,
        )
        self._provider = provider
        self._band_idx = _BANDS[band]
        self._n_bands = n_bands
        self._lookback = lookback_days
        self._cache_month: Optional[tuple] = None
        self._cache_scores: Optional[Dict[str, float]] = None

    def _alpha_scores(self, all_data: Dict[str, pd.DataFrame],
                      date) -> Optional[Dict[str, float]]:
        ts = pd.Timestamp(date)
        month = (ts.year, ts.month)
        if month == self._cache_month:
            return self._cache_scores        # signal is monthly; reuse

        symbols = list(all_data.keys())
        caps = pit_marketcaps(self._provider, symbols, date)   # PIT market cap
        buckets = size_buckets(caps, self._n_bands)            # tercile per name
        scores: Dict[str, float] = {}
        for sym, b in buckets.items():
            if b != self._band_idx:
                continue
            nb = self._provider.insider_net_buys(
                sym, date, lookback_days=self._lookback)
            if (nb['n_buys'] + nb['n_sells']) == 0 or nb['net_shares'] == 0:
                continue                     # no directional insider activity
            # Score = SIGN only. The screen validated that the DIRECTION (net
            # buyer vs seller) predicts and the dollar MAGNITUDE does not — so
            # ranking by net_value would concentrate the book on non-predictive
            # big-dollar extremes. +1 ranks every buyer long, -1 every seller
            # short; within-side selection (top/bottom quantile) is then a wash.
            scores[sym] = float(np.sign(nb['net_shares']))

        result = scores if len(scores) >= self.min_scored else None
        self._cache_month, self._cache_scores = month, result
        return result
