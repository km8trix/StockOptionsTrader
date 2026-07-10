"""PEAD desk — post-earnings-announcement drift on true SUE (small/mid cap).

Trades the drift after earnings surprises: long the strongest positive
standardized surprises (SUE), short the strongest negative, on the shared
CrossSectionalLongShortDesk book. The signal is the drift-adjusted
Bernard–Thomas SUE from PIT SF1 filings (data/earnings_surprise.py over
PitWarehouse.eps_quarterly), kept only while FRESH (filing datekey within
fresh_days of the simulated date) — stale filings carry no drift.

PIT discipline: a filing is visible from its SEC ``datekey`` (median ~41 days
after quarter end — conservative vs press-release dating and lookahead-clean).
The desk pulls the quarterly EPS table ONCE per run and slices it to
``datekey <= date`` before every SUE computation, so scores at t never see a
later filing (pinned by test). Optional ``band`` restricts the book to one PIT
market-cap tercile (data/size_buckets.py — never scalemarketcap).

FIXED factor: committee=[] so walk_forward_fits=[] and validation runs at
n_trials=1 (no deflation) — honest, the rule is pre-specified in
docs/vix_pead_desks_spec.md. Wide RiskManager (the signal is monthly; a 2%
price stop would churn it). Scores are cached per calendar month (the base
calls _alpha_scores daily) — including a cached None, which keeps the book
flat until the month rolls even if filings land mid-month. That is the same
monthly cadence the screen validated (BMS rebalances) and mirrors the insider
desk's documented behavior; trading mid-month would be an untested,
faster-cadence variant. Run under
``BacktestEngine(desk=..., market_data=WarehouseMarketData())`` so delisted
names are still priced (survivorship-free).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from data.earnings_surprise import (apply_announcement_dating,
                                    latest_fresh_sue, sue_table)
from data.size_buckets import pit_marketcaps, size_buckets
from desks.cross_sectional import CrossSectionalLongShortDesk
from portfolio.risk_manager import RiskManager

_BANDS = {'micro': 0, 'small': 1, 'mid': 2}   # ascending PIT market-cap tercile


class PEADDesk(CrossSectionalLongShortDesk):
    """Long the top SUE quantile, short the bottom, among names with a FRESH
    earnings filing. ``provider`` exposes eps_quarterly (+ daily_metric when
    ``band`` is set); it is queried with the simulated ``date`` as the
    point-in-time boundary."""

    def __init__(self, band: Optional[str] = None, *, provider=None,
                 capital_allocation: float = 1.0,
                 risk_manager: Optional[RiskManager] = None,
                 fresh_days: int = 63, quantile: float = 0.2,
                 n_bands: int = 3, long_only: bool = False,
                 dating: str = 'filing'):
        if band is not None and band not in _BANDS:
            raise ValueError(f"band {band!r} must be one of {list(_BANDS)}")
        if dating not in ('filing', 'announce'):
            raise ValueError(f"dating {dating!r} must be 'filing' or "
                             f"'announce'")
        if provider is None:
            from data.pit_warehouse import PitWarehouse
            provider = PitWarehouse()
        if risk_manager is None:
            risk_manager = RiskManager(position_stop_loss=0.50)
        label = band or 'all'
        super().__init__(
            key='pead' if band is None else f'pead_{band}',
            name=('PEAD Desk' if band is None
                  else f'PEAD Desk ({band}-cap)'),
            description=('Post-earnings-announcement drift: long the '
                         'strongest positive standardized earnings surprises '
                         '(SUE from PIT SF1 filings), short the strongest '
                         'negative, while the filing is fresh'
                         + ('' if band is None
                            else f', within the PIT {band}-cap tercile')
                         + '. Requires the Sharadar PIT warehouse.'),
            accent='#9a6700',
            note_label=f'PEAD-{label}',
            reason_prefix=f'pead-{label}',
            committee=[],
            model_label=f'PEAD SUE ({label}, fixed factor)',
            capital_allocation=capital_allocation,
            risk_manager=risk_manager,
            quantile=quantile,
            long_only=long_only,
        )
        self._provider = provider
        self._band_idx = None if band is None else _BANDS[band]
        self._n_bands = n_bands
        self._fresh_days = fresh_days
        self.dating = dating
        self._events: Optional[pd.DataFrame] = None
        self._eps: Optional[pd.DataFrame] = None   # cumulative pull, sliced PIT
        self._eps_symbols: set = set()             # symbols covered by _eps
        self._cache_month: Optional[tuple] = None
        self._cache_scores: Optional[Dict[str, float]] = None

    def _alpha_scores(self, all_data: Dict[str, pd.DataFrame],
                      date) -> Optional[Dict[str, float]]:
        ts = pd.Timestamp(date)
        month = (ts.year, ts.month)
        if month == self._cache_month:
            return self._cache_scores        # signal is monthly; reuse

        symbols = list(all_data.keys())
        # One warehouse scan per run PLUS a re-pull whenever the engine hands
        # us symbols not yet covered (mid-window IPOs enter all_data only once
        # they have bars — a day-one-only pull would silently exclude them for
        # the whole backtest). Every read slices this frame to datekey <= date,
        # which IS the PIT boundary (pinned by test).
        if self._eps is None or not set(symbols) <= self._eps_symbols:
            self._eps_symbols |= set(symbols)
            syms = sorted(self._eps_symbols)
            self._eps = self._provider.eps_quarterly(syms)
            if self.dating == 'announce':
                self._events = self._provider.earnings_events(syms)
        if self._band_idx is not None:
            caps = pit_marketcaps(self._provider, symbols, date)
            buckets = size_buckets(caps, self._n_bands)
            keep = {s for s, b in buckets.items() if b == self._band_idx}
        else:
            keep = set(symbols)

        if self.dating == 'announce':
            # Include rows announced by ts but FILED up to ~60d later, then
            # re-date each SUE to its 8-K press release; latest_fresh_sue's
            # own <= ts filter then runs on announcement dates (unannounced
            # padded rows keep datekey > ts and drop out). The 8-K-EPS ==
            # filed-EPS approximation is documented on
            # apply_announcement_dating.
            visible = self._eps[self._eps['datekey']
                                <= ts + pd.Timedelta(days=60)]
            rows = apply_announcement_dating(sue_table(visible), self._events)
        else:
            visible = self._eps[self._eps['datekey'] <= ts]
            rows = sue_table(visible)
        sue = latest_fresh_sue(rows, ts, fresh_days=self._fresh_days)
        scores = {sym: float(sue[sym]) for sym in keep
                  if sym in sue.index and np.isfinite(sue[sym])}

        result = scores if len(scores) >= self.min_scored else None
        self._cache_month, self._cache_scores = month, result
        return result
