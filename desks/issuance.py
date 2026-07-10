"""Net-issuance desk — YoY split-adjusted share growth, buybacks long.

Graduates the validated issuance screen (scripts/issuance_screen.py,
2026-07-10: 8/8 BH survivors, pooled 63d net +4.83% t=+3.42, and uniquely a
LIVE MID-CAP slice — 63d net +1.89% t=+2.79 — where value/quality/PEAD all
died; measured long-leg turnover 7.3%/rebalance) to a desk. The signal math
is the shared data/share_issuance.issuance_table — one definition for screen
and desk.

SIGN CONVENTION (explicit, academic — Pontiff-Woodgate / Daniel-Titman):
HIGH issuance predicts LOW returns. The base machinery longs the HIGHEST
alpha scores, so the score is the NEGATED issuance value: a net buyback
(negative issuance) scores positive and is longed; a heavy issuer scores
negative and is shorted (when not long_only). Pinned by test with a planted
buyback vs issuer.

PIT discipline: the quarterly share-count table is pulled ONCE per run
(cumulative re-pull only when the engine hands over new symbols — the PEAD
desk pattern, so a 600-name 10y gate stays ~minutes) and sliced to
``datekey <= date`` before every issuance computation, so scores at t never
see a later filing (pinned by test). Names whose latest computable filing is
older than ``stale_days`` (default 400 — the screen's --stale-days default:
a live quarterly filer is always fresh; only names that stopped filing over
a year ago fall out) are dropped.

FUND SLOT (defaults): ``exclude_micro=True`` — PEAD owns the micro band, VQ
is ex-micro; issuance rides the same tradeable ex-micro slice, where its
screen uniquely survived, and stays disjoint from the micro PEAD leg
(overlapping cross-sectional books churn on a shared portfolio).
``long_only=True`` — small-cap short legs bleed (insider/PEAD lesson).
``key='issuance'`` is STABLE regardless of params: fund allocation keys must
equal desk.key (the vq_fund_gate key contract) or risk-parity silently
degenerates to the equal-weight fallback.

FIXED factor: committee=[] => walk_forward_fits=[] => n_trials=1 validation.
Wide RiskManager (0.50 stop — monthly signal; a 2% stop would churn it).
Monthly score cache (the base calls _alpha_scores daily), monthly effective
cadence — the BMS cadence the screen validated. Run under
``BacktestEngine(desk=..., market_data=WarehouseMarketData())``.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from data.share_issuance import issuance_table
from data.size_buckets import pit_marketcaps, size_buckets
from desks.cross_sectional import CrossSectionalLongShortDesk
from portfolio.risk_manager import RiskManager


class IssuanceDesk(CrossSectionalLongShortDesk):
    """Long the lowest-issuance (buyback) quantile of names with a fresh
    filing, short the heaviest issuers (suppressed by default: long_only).
    ``provider`` exposes fundamentals_quarterly (+ daily_marketcaps /
    daily_metric when ``exclude_micro``); it is queried with the simulated
    ``date`` as the point-in-time boundary."""

    def __init__(self, *, provider=None, capital_allocation: float = 1.0,
                 risk_manager: Optional[RiskManager] = None,
                 quantile: float = 0.2, long_only: bool = True,
                 stale_days: int = 400, exclude_micro: bool = True):
        if provider is None:
            from data.pit_warehouse import PitWarehouse
            provider = PitWarehouse()
        if risk_manager is None:
            risk_manager = RiskManager(position_stop_loss=0.50)
        super().__init__(
            # STABLE key: fund allocation dicts address this desk by .key
            # (tests/test_vq_fund_gate.py key contract) — never derive it
            # from params.
            key='issuance',
            name='Net-Issuance Desk',
            description=('Cross-sectional net share issuance: long the '
                         'names retiring shares (buybacks — lowest YoY '
                         'split-adjusted share growth from PIT SF1 '
                         'sharesbas*sharefactor), short the heaviest '
                         'issuers. Requires the Sharadar PIT warehouse '
                         '(sf1 + daily ingested).'),
            accent='#8250df',
            note_label='Issuance',
            reason_prefix='issuance',
            committee=[],
            model_label='YoY net issuance (fixed factor)',
            capital_allocation=capital_allocation,
            risk_manager=risk_manager,
            quantile=quantile,
            long_only=long_only,
        )
        self._provider = provider
        self._stale_days = stale_days
        self._exclude_micro = exclude_micro
        self._shares: Optional[pd.DataFrame] = None  # cumulative pull
        self._share_symbols: set = set()             # symbols it covers
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
        # us symbols not yet covered (mid-window IPOs enter all_data only
        # once they have bars). Every read slices this frame to datekey <=
        # date, which IS the PIT boundary (pinned by test) — the PEAD desk's
        # one-shot-pull pattern verbatim.
        if self._shares is None or not set(symbols) <= self._share_symbols:
            self._share_symbols |= set(symbols)
            self._shares = self._provider.fundamentals_quarterly(
                sorted(self._share_symbols),
                fields=('sharesbas', 'sharefactor'))
        if self._exclude_micro:
            # Drop the bottom PIT market-cap tercile (data/size_buckets —
            # never scalemarketcap): the fund's micro band belongs to PEAD,
            # and the screen's mid-cap slice is where issuance uniquely
            # survived. Names with no usable PIT cap are dropped too (VQ
            # convention: bucket default 0 == micro).
            caps = pit_marketcaps(self._provider, symbols, ts)
            buckets = size_buckets(caps, 3)
            keep = {s for s in symbols if buckets.get(s, 0) != 0}
        else:
            keep = set(symbols)

        visible = self._shares[self._shares['datekey'] <= ts]
        rows = issuance_table(visible)
        if len(rows):
            # Latest computable filing per name, recent only: a name whose
            # newest issuance row is staler than stale_days stopped filing
            # (or stopped being computable) — its "YoY" growth describes
            # ancient history, so it drops out rather than carrying forward.
            fresh = rows[rows['datekey']
                         >= ts - pd.Timedelta(days=self._stale_days)]
            latest = (fresh.sort_values('datekey')
                      .groupby('ticker', sort=False)
                      .tail(1).set_index('ticker')['issuance'])
        else:
            latest = pd.Series(dtype=float)
        # SIGN: HIGH issuance predicts LOW returns, and the base longs the
        # HIGHEST scores — so score = MINUS issuance. A buyback (negative
        # issuance) scores positive => long; a heavy issuer scores negative
        # => bottom of the book.
        scores = {sym: -float(latest[sym]) for sym in keep
                  if sym in latest.index and np.isfinite(latest[sym])}

        result = scores if len(scores) >= self.min_scored else None
        self._cache_month, self._cache_scores = month, result
        return result
