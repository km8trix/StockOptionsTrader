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

STRENGTHENING VARIANTS (2026-07-10, both opt-in, defaults byte-identical —
the two merged screen facts behind them, both pre-registered, no tuning):

  include_gp_assets  adds the gp/assets (Novy-Marx gross profitability) rank
      to the composite at an EQUAL rank weight — the strongest quality factor
      ever screened here (pooled 63d net +3.60% t=+4.44) and nearly
      orthogonal to netmargin (per-date Spearman +0.067). Computed via
      data/quality_ratios.computed_ratio on the SAME PIT SF1 row the desk
      reads netmargin from; names with no computable gp_assets rank on the
      available factors only (mean of the available ranks) rather than
      dropping out — requiring the raws would shrink the base book.

  issuance_filter    drops the TOP QUINTILE (0.2, rank-based among the
      rebalance's scored names — the desk's own quantile convention, no
      absolute threshold) of YoY net share issuers from LONG candidacy.
      HIGH issuance predicts LOW returns (8/8 BH screen) but the long-only
      issuance leg was weak (PR #96 negative) — as a FILTER the short-leg
      information is harvested without shorting. Missing issuance is NOT
      evidence of issuance: no-data names are never excluded. Longs only.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from data.quality_ratios import computed_ratio
from data.share_issuance import issuance_table
from data.size_buckets import pit_marketcaps, size_buckets
from desks.cross_sectional import CrossSectionalLongShortDesk
from portfolio.risk_manager import RiskManager

#: issuance_filter exclusion quantile — pre-registered at the desk's own
#: top-quintile convention (0.2, rank-based per rebalance), never tuned.
_ISSUANCE_FILTER_QUINTILE = 0.2
#: issuance_filter staleness cut, days — the issuance-desk/screen default
#: (400). The composite's filing-recency screen bounds NETMARGIN freshness,
#: not issuance freshness (different pulls, different NULL filters, plus
#: issuance_table's own guards), so without this cut a name whose last
#: COMPUTABLE issuance row is years old would be barred from longs on
#: ancient history. Stale issuance = no data = NOT excluded.
_ISSUANCE_STALE_DAYS = 400


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
                 stale_days: int = 270, exclude_micro: bool = False,
                 include_gp_assets: bool = False,
                 issuance_filter: bool = False):
        if provider is None:
            from data.pit_warehouse import PitWarehouse
            provider = PitWarehouse()
        if risk_manager is None:
            risk_manager = RiskManager(position_stop_loss=0.50)
        super().__init__(
            # STABLE key: fund allocation dicts address this desk by .key
            # (tests/test_vq_fund_gate.py key contract) — never derive it
            # from params (variants included).
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
        self._include_gp_assets = include_gp_assets
        self._issuance_filter = issuance_filter
        self._nm: Optional[pd.DataFrame] = None    # cumulative pull, PIT-cut
        self._nm_symbols: set = set()
        # issuance_filter state: cumulative share-count pull (the issuance
        # desk's one-shot-pull pattern verbatim) + the monthly exclusion set
        # (computed alongside the score cache, served via _long_exclusions).
        self._shares: Optional[pd.DataFrame] = None
        self._share_symbols: set = set()
        self._cache_month: Optional[tuple] = None
        self._cache_scores: Optional[Dict[str, float]] = None
        self._cache_excluded: set = set()

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
            # gp/assets raws ride the SAME pull (and so the same PIT SF1
            # row) as netmargin when the gp_assets extension is on; the
            # default pull is byte-identical. fundamentals_quarterly drops
            # rows where the FIRST field is NULL, so the netmargin row set
            # is unchanged either way.
            fields = (('netmargin', 'gp', 'assets')
                      if self._include_gp_assets else ('netmargin',))
            self._nm = self._provider.fundamentals_quarterly(
                sorted(self._nm_symbols), fields=fields)

        # Latest PIT filing per name, recent filings only.
        vis = self._nm[(self._nm['datekey'] <= ts)
                       & (self._nm['datekey']
                          >= ts - pd.Timedelta(days=self._stale_days))]
        latest = (vis.sort_values('datekey').groupby('ticker', sort=False)
                  .tail(1).set_index('ticker'))
        nm = latest['netmargin']
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
            gpa = np.nan
            if self._include_gp_assets:
                # gp/assets from the SAME PIT SF1 row as the margin; NaN
                # raws map to None so computed_ratio's non-null guard fires
                # (mirroring the missing-netmargin detection above). A name
                # with no computable gp_assets keeps a NaN here and ranks on
                # the available factors only — it is NOT dropped.
                row = latest.loc[sym]
                fund = {f: (None if pd.isna(row[f]) else row[f])
                        for f in ('gp', 'assets')}
                v = computed_ratio(fund, 'gp', 'assets')
                gpa = np.nan if v is None or not np.isfinite(v) else float(v)
            rows[sym] = (pb, float(margin), gpa)
        result: Optional[Dict[str, float]] = None
        if len(rows) >= self.min_scored:
            frame = pd.DataFrame.from_dict(rows, orient='index',
                                           columns=['pb', 'nm', 'gpa'])
            cheap = (-frame['pb']).rank(pct=True)      # low pb = high rank
            profit = frame['nm'].rank(pct=True)        # high margin = high
            if self._include_gp_assets:
                # EQUAL rank weights (pre-registered — no weight tuning):
                # the mean of the AVAILABLE ranks. pandas keeps NaN out of
                # both the rank (na_option='keep') and the row mean
                # (skipna), so a name with no computable gp_assets averages
                # its pb+netmargin ranks alone.
                quality2 = frame['gpa'].rank(pct=True)  # high gp/assets = high
                composite = pd.concat([cheap, profit, quality2],
                                      axis=1).mean(axis=1)
            else:
                composite = (cheap + profit) / 2.0
            result = {sym: float(v) for sym, v in composite.items()}

        # Monthly exclusion set rides the score cache: rank-based among THIS
        # rebalance's scored names, served to the base via _long_exclusions.
        self._cache_excluded = (
            self._issuance_exclusions(sorted(result), ts)
            if self._issuance_filter and result else set())
        self._cache_month, self._cache_scores = month, result
        return result

    def _issuance_exclusions(self, symbols: List[str], ts: pd.Timestamp
                             ) -> set:
        """The top-``_ISSUANCE_FILTER_QUINTILE`` heaviest YoY net issuers
        among ``symbols`` (this rebalance's scored names) — barred from longs.

        HIGH issuance predicts LOW returns (data/share_issuance sign
        convention; 8/8 BH screen), and the long-only issuance DESK was weak
        (PR #96 negative) — so the short-leg information is harvested as a
        long-candidacy FILTER instead. Rank-based per rebalance (the desk's
        own quantile convention, floor'd like the base's k so fewer than 5
        computable names exclude nobody), ties broken by symbol. Names with
        no computable issuance are NOT excluded — missing data is not
        evidence of issuance — and a latest computable row older than
        ``_ISSUANCE_STALE_DAYS`` counts as missing (the issuance-desk/screen
        staleness convention; see the constant's rationale).
        """
        # One warehouse scan per run plus a cumulative re-pull for symbols
        # not yet covered — desks/issuance.py's one-shot-pull pattern
        # verbatim; every read slices to datekey <= ts (the PIT boundary).
        if self._shares is None or not set(symbols) <= self._share_symbols:
            self._share_symbols |= set(symbols)
            self._shares = self._provider.fundamentals_quarterly(
                sorted(self._share_symbols),
                fields=('sharesbas', 'sharefactor'))
        visible = self._shares[self._shares['datekey'] <= ts]
        table = issuance_table(visible)
        if not len(table):
            return set()
        latest = (table.sort_values('datekey').groupby('ticker', sort=False)
                  .tail(1).set_index('ticker'))
        fresh = latest[latest['datekey']
                       >= ts - pd.Timedelta(days=_ISSUANCE_STALE_DAYS)]
        latest = fresh['issuance']
        scored = [(sym, float(latest[sym])) for sym in symbols
                  if sym in latest.index and np.isfinite(latest[sym])]
        k = int(_ISSUANCE_FILTER_QUINTILE * len(scored))
        if k <= 0:
            return set()
        scored.sort(key=lambda item: (-item[1], item[0]))  # heaviest first
        return {sym for sym, _ in scored[:k]}

    def _long_exclusions(self, ranked_symbols: List[str], date) -> set:
        """Serve the monthly issuance-filter set to the base's long-candidacy
        filter (empty when ``issuance_filter`` is off — byte-identical)."""
        return self._cache_excluded
