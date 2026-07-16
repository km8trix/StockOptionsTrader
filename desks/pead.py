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

FIXED factor: committee=[] so walk_forward_fits=[]. Historical validation used
n_trials=1, which is not qualifying; future DSR uses program-wide breadth from
the append-only research ledger. Wide RiskManager (the signal is monthly; a 2%
price stop would churn it). Scores are cached per calendar month (the base
calls _alpha_scores daily) — including a cached None, which keeps the book
flat until the month rolls even if filings land mid-month. That is the same
monthly cadence the screen validated (BMS rebalances) and mirrors the insider
desk's documented behavior; trading mid-month would be an untested,
faster-cadence variant. Run under
``BacktestEngine(desk=..., market_data=WarehouseMarketData())`` so delisted
names are still priced (survivorship-free).

PRE-REGISTERED SURGE VARIANTS (2026-07-10, both opt-in, defaults
byte-identical; rules and constants declared HERE before any variant
backtest ran — 2 trials this round, no further variant joins after seeing
results). These declarations predate the sealed ledger and remain development
provenance, not a qualifying preregistration. SURGE = the standardized
revenue-growth surprise
(Jegadeesh-Livnat 2006, who found earnings+revenue surprise jointly beat
either alone): sue_table(column='revenue') over a SEPARATE
fundamentals_quarterly(fields=('revenue',)) pull. Separate, not a joint
('epsdil', 'revenue') pull: the warehouse NULL-filters and dedups on the
FIRST field, so only a revenue-first pull yields the honest first-known
revenue row set (a joint pull would drop revenue-only filings and date each
quarter by its first EPSDIL-bearing filing) — and the default desk's
eps_quarterly call stays literally untouched. SURGE rows then get EXACTLY
the SUE treatment: the same PIT slice (datekey <= date; announce mode's
+60d padding and 8-K re-dating included) and the same ``fresh_days``
freshness window.

  surge_confirm   longs must ALSO have SURGE >= the MEDIAN
      (_SURGE_CONFIRM_PCTL = 0.5, rank-based among the rebalance's scored
      names with a computable fresh SURGE — never an absolute threshold).
      Names with NO computable fresh SURGE are NOT excluded — missing data
      is not evidence of a weak revenue surprise (the ValueQualityDesk
      issuance-filter convention, staleness included: a revenue filing
      older than ``fresh_days`` counts as missing). Longs only, served via
      the base's _long_exclusions hook (the top-k backfills; a blocked
      held long is closed by the normal reconcile).

  rank_combine    desk score = the EQUAL-WEIGHT mean of the SUE percentile
      rank and the SURGE percentile rank among the rebalance's scored
      names (weights pre-registered, never tuned). A scored name missing
      SURGE ranks on SUE alone — the mean of its AVAILABLE ranks (the
      ValueQualityDesk gp_assets convention); the scored set itself is
      unchanged (names with a computable fresh SUE).

The two variants are SEPARATE pre-registered trials: composing them was
never pre-registered, so the constructor rejects both-on.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from data.earnings_surprise import (apply_announcement_dating,
                                    latest_fresh_sue, sue_table)
from data.size_buckets import pit_marketcaps, size_buckets
from desks.cross_sectional import CrossSectionalLongShortDesk
from portfolio.risk_manager import RiskManager

_BANDS = {'micro': 0, 'small': 1, 'mid': 2}   # ascending PIT market-cap tercile

#: surge_confirm long-candidacy threshold — pre-registered at the MEDIAN
#: (0.5, rank-based among the rebalance's scored names with a computable
#: fresh SURGE; module docstring), declared before any variant backtest ran
#: and never tuned.
_SURGE_CONFIRM_PCTL = 0.5


class PEADDesk(CrossSectionalLongShortDesk):
    """Long the top SUE quantile, short the bottom, among names with a FRESH
    earnings filing. ``provider`` exposes eps_quarterly (+ daily_metric when
    ``band`` is set, + fundamentals_quarterly when a SURGE variant is on);
    it is queried with the simulated ``date`` as the point-in-time
    boundary."""

    def __init__(self, band: Optional[str] = None, *, provider=None,
                 capital_allocation: float = 1.0,
                 risk_manager: Optional[RiskManager] = None,
                 fresh_days: int = 63, quantile: float = 0.2,
                 n_bands: int = 3, long_only: bool = False,
                 dating: str = 'filing', surge_confirm: bool = False,
                 rank_combine: bool = False):
        if band is not None and band not in _BANDS:
            raise ValueError(f"band {band!r} must be one of {list(_BANDS)}")
        if dating not in ('filing', 'announce'):
            raise ValueError(f"dating {dating!r} must be 'filing' or "
                             f"'announce'")
        if surge_confirm and rank_combine:
            raise ValueError(
                "surge_confirm and rank_combine are SEPARATE pre-registered "
                "trials (module docstring); composing them was never "
                "pre-registered — pick one")
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
        self._surge_confirm = surge_confirm
        self._rank_combine = rank_combine
        self._events: Optional[pd.DataFrame] = None
        self._eps: Optional[pd.DataFrame] = None   # cumulative pull, sliced PIT
        self._eps_symbols: set = set()             # symbols covered by _eps
        # SURGE state (variant-only): cumulative revenue pull mirroring _eps,
        # plus the monthly surge_confirm exclusion set (computed alongside
        # the score cache, served via _long_exclusions — the ValueQualityDesk
        # issuance-filter pattern verbatim).
        self._rev: Optional[pd.DataFrame] = None
        self._rev_symbols: set = set()
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

        if self._rank_combine and scores:
            # Pre-registered combine (module docstring): the EQUAL-WEIGHT
            # mean of the SUE and SURGE percentile ranks among the scored
            # names. pandas keeps a missing SURGE out of both the rank
            # (na_option='keep') and the row mean (skipna), so a scored name
            # with no computable fresh SURGE ranks on SUE alone — the VQ
            # gp_assets mean-of-available convention. The scored SET is
            # unchanged: SURGE never adds or drops a name.
            sue_rank = pd.Series(scores).rank(pct=True)
            surge = self._fresh_surge(sorted(scores), ts)
            surge_rank = (pd.Series(surge, dtype=float)
                          .reindex(sue_rank.index).rank(pct=True))
            composite = pd.concat([sue_rank, surge_rank], axis=1).mean(axis=1)
            scores = {sym: float(v) for sym, v in composite.items()}

        result = scores if len(scores) >= self.min_scored else None
        # Monthly exclusion set rides the score cache (VQ issuance-filter
        # pattern): rank-based among THIS rebalance's scored names, served
        # to the base via _long_exclusions. Empty unless surge_confirm.
        self._cache_excluded = (
            self._surge_exclusions(sorted(result), ts)
            if self._surge_confirm and result else set())
        self._cache_month, self._cache_scores = month, result
        return result

    def _fresh_surge(self, symbols: List[str], ts: pd.Timestamp
                     ) -> Dict[str, float]:
        """symbol -> latest FRESH SURGE (standardized revenue-growth
        surprise) for the ``symbols`` with one computable at ``ts``.

        Mirrors the SUE pipeline EXACTLY: the same cumulative one-pull
        pattern (a separate revenue-first pull — module docstring), the same
        PIT slice per dating mode (announce mode's +60d filing padding and
        8-K re-dating via the SAME events table included), and the same
        ``fresh_days`` freshness window. Names with no computable fresh
        SURGE are simply absent from the result."""
        if self._rev is None or not set(symbols) <= self._rev_symbols:
            self._rev_symbols |= set(symbols)
            self._rev = self._provider.fundamentals_quarterly(
                sorted(self._rev_symbols), fields=('revenue',))
        if self.dating == 'announce':
            visible = self._rev[self._rev['datekey']
                                <= ts + pd.Timedelta(days=60)]
            rows = apply_announcement_dating(
                sue_table(visible, column='revenue'), self._events)
        else:
            visible = self._rev[self._rev['datekey'] <= ts]
            rows = sue_table(visible, column='revenue')
        surge = latest_fresh_sue(rows, ts, fresh_days=self._fresh_days)
        return {sym: float(surge[sym]) for sym in symbols
                if sym in surge.index and np.isfinite(surge[sym])}

    def _surge_exclusions(self, symbols: List[str], ts: pd.Timestamp) -> set:
        """Scored names whose fresh SURGE sits strictly BELOW the median —
        barred from long candidacy (surge_confirm, module docstring).

        The cut is ``_SURGE_CONFIRM_PCTL`` (the median), rank-based among
        THIS rebalance's scored names with a computable fresh SURGE — never
        an absolute threshold. Names with no computable fresh SURGE are not
        in the population and are NOT excluded (missing data is not evidence
        of a weak revenue surprise — the VQ issuance-filter convention);
        SURGE == the median passes (rank >= median)."""
        surge = self._fresh_surge(symbols, ts)
        if not surge:
            return set()
        vals = pd.Series(surge, dtype=float)
        cut = vals.quantile(_SURGE_CONFIRM_PCTL)
        return set(vals.index[vals < cut])

    def _long_exclusions(self, ranked_symbols: List[str], date) -> set:
        """Serve the monthly surge_confirm set to the base's long-candidacy
        filter (empty when ``surge_confirm`` is off — byte-identical)."""
        return self._cache_excluded
