"""Value+Quality composite desk — the screened two-factor rank blend.

Graduates the validated screen composite (scripts/factor_screen.py value pb +
scripts/quality_screen.py netmargin, combined per the 2026-07-02 rank-average
study: composite beat each leg on both t and magnitude) to a desk. Score =
mean of the cross-sectional percentile ranks of CHEAPNESS (low DAILY pb,
non-positive pb dropped — value traps are not "cheap") and PROFITABILITY
(latest PIT SF1 netmargin; negative margins are legitimate short-leg members).

PIT discipline: pb comes from the DAILY table (point-in-time by nature) on
the simulated date; netmargin from the latest ARQ filing with datekey <= date,
required to be RECENT (within stale_days) so dead filers drop out rather than
carrying a years-old margin. Monthly score cache (the base calls
_alpha_scores daily), monthly effective cadence — the cadence the screens
validated.

FIXED factor: committee=[] => walk_forward_fits=[] => n_trials=1 validation.
Wide RiskManager (monthly signal; a 2% stop would churn it). Run under
``BacktestEngine(desk=..., market_data=WarehouseMarketData())``.

HONEST CONTEXT (screen record, do not oversell): the composite's edge is
micro-concentrated and regime-dependent (value dead 2015-2019); the tradeable
ex-micro slice was only t~1.9-2.6. This desk exists primarily as the
decorrelated second leg for the PEAD combine (docs/vix_pead_desks_spec.md
unlock c), not as a standalone promotion candidate.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from data.size_buckets import pit_marketcaps, size_buckets
from desks.cross_sectional import CrossSectionalLongShortDesk
from portfolio.risk_manager import RiskManager


class ValueQualityDesk(CrossSectionalLongShortDesk):
    """Long cheap+profitable, short rich+unprofitable, by rank composite.

    exclude_micro drops the bottom PIT market-cap tercile: the validated
    tradeable slice is EX-MICRO (the screen's micro strength is illiquid),
    and it makes the book DISJOINT from a micro-band PEAD leg in a fund —
    overlapping cross-sectional books on a shared portfolio fight each
    other's positions (orphan-sweep churn)."""

    def __init__(self, *, provider=None, capital_allocation: float = 1.0,
                 risk_manager: Optional[RiskManager] = None,
                 quantile: float = 0.2, long_only: bool = False,
                 stale_days: int = 270, exclude_micro: bool = False):
        if provider is None:
            from data.pit_warehouse import PitWarehouse
            provider = PitWarehouse()
        if risk_manager is None:
            risk_manager = RiskManager(position_stop_loss=0.50)
        super().__init__(
            key='value_quality',
            name='Value+Quality Desk',
            description=('Cross-sectional value+quality composite: long the '
                         'cheapest (PIT price/book) most profitable (PIT '
                         'net margin) names, short the richest least '
                         'profitable, rank-blended. Requires the Sharadar '
                         'PIT warehouse (sf1 + daily ingested).'),
            accent='#0969da',
            note_label='Value+Quality',
            reason_prefix='value-quality',
            committee=[],
            model_label='value+quality rank composite (fixed factor)',
            capital_allocation=capital_allocation,
            risk_manager=risk_manager,
            quantile=quantile,
            long_only=long_only,
        )
        self._provider = provider
        self._stale_days = stale_days
        self._exclude_micro = exclude_micro
        self._nm: Optional[pd.DataFrame] = None    # cumulative pull, PIT-cut
        self._nm_symbols: set = set()
        self._cache_month: Optional[tuple] = None
        self._cache_scores: Optional[Dict[str, float]] = None

    def _alpha_scores(self, all_data: Dict[str, pd.DataFrame],
                      date) -> Optional[Dict[str, float]]:
        ts = pd.Timestamp(date)
        month = (ts.year, ts.month)
        if month == self._cache_month:
            return self._cache_scores        # signal is monthly; reuse

        symbols = list(all_data.keys())
        if self._exclude_micro:
            caps = pit_marketcaps(self._provider, symbols, ts)
            buckets = size_buckets(caps, 3)
            symbols = [s for s in symbols if buckets.get(s, 0) != 0]
        if self._nm is None or not set(symbols) <= self._nm_symbols:
            self._nm_symbols |= set(symbols)
            self._nm = self._provider.fundamentals_quarterly(
                sorted(self._nm_symbols), fields=('netmargin',))

        # Latest PIT netmargin per name, recent filings only.
        vis = self._nm[(self._nm['datekey'] <= ts)
                       & (self._nm['datekey']
                          >= ts - pd.Timedelta(days=self._stale_days))]
        nm = (vis.sort_values('datekey').groupby('ticker', sort=False)
              .tail(1).set_index('ticker')['netmargin'])
        # PIT price/book on the simulated date; value traps dropped.
        pbs = self._provider.daily_fields_bulk(symbols, ts, fields=('pb',))

        rows = {}
        for sym in symbols:
            pb = (pbs.get(sym) or {}).get('pb')
            margin = nm.get(sym)
            if pb is None or not np.isfinite(pb) or pb <= 0:
                continue
            if margin is None or not np.isfinite(margin):
                continue
            rows[sym] = (pb, float(margin))
        result: Optional[Dict[str, float]] = None
        if len(rows) >= self.min_scored:
            frame = pd.DataFrame.from_dict(rows, orient='index',
                                           columns=['pb', 'nm'])
            cheap = (-frame['pb']).rank(pct=True)      # low pb = high rank
            profit = frame['nm'].rank(pct=True)        # high margin = high
            composite = (cheap + profit) / 2.0
            result = {sym: float(v) for sym, v in composite.items()}

        self._cache_month, self._cache_scores = month, result
        return result
