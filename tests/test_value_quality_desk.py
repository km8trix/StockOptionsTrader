"""Tests for the ValueQualityDesk strengthening variants: the pinned default
composite (byte-identity), the opt-in gp_assets rank extension (equal
weights, missing-field fallback), the opt-in issuance long-filter
(rank-based top quintile, no-data names kept), their composition, the
one-pull caching, and the stable-key contract. All hermetic — in-memory
fake provider, no network, no warehouse (the test_issuance_desk.py desk-test
pattern)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.share_issuance import _share_rows
from desks.value_quality import ValueQualityDesk
from portfolio.manager import PortfolioManager


def _fund_row(ticker, nm, gp=np.nan, assets=np.nan):
    """One PIT SF1 row per name (reportperiod 2022-03-31, filed +40d)."""
    return {'ticker': ticker, 'reportperiod': pd.Timestamp('2022-03-31'),
            'datekey': pd.Timestamp('2022-05-10'), 'netmargin': nm,
            'gp': gp, 'assets': assets}


class _FakeProvider:
    """Minimal PitWarehouse stand-in: canned SF1 rows + daily pb + shares.

    fundamentals_quarterly dispatches on ``fields`` exactly like the desk
    calls it (netmargin-family pull vs the sharesbas/sharefactor pull) and
    counts each kind, so the one-pull caching is pinned per table. Only the
    REQUESTED fields come back (the warehouse contract)."""

    def __init__(self, funds, pbs, shares=None):
        self._funds = funds
        self._pbs = pbs
        self._shares = shares
        self.nm_pulls = 0
        self.share_pulls = 0
        self.nm_fields = []          # the fields tuple of every nm pull

    def fundamentals_quarterly(self, tickers=None, *, fields=('netmargin',)):
        fields = tuple(fields)
        if fields == ('sharesbas', 'sharefactor'):
            self.share_pulls += 1
            df = self._shares
            if df is None:
                df = pd.DataFrame(columns=['ticker', 'reportperiod',
                                           'datekey', 'sharesbas',
                                           'sharefactor'])
            if tickers is not None:
                df = df[df['ticker'].isin(tickers)]
            return df.reset_index(drop=True)
        assert fields[0] == 'netmargin'
        self.nm_pulls += 1
        self.nm_fields.append(fields)
        rows = [r for r in self._funds
                if tickers is None or r['ticker'] in tickers]
        df = pd.DataFrame(rows)
        return df[['ticker', 'reportperiod', 'datekey']
                  + list(fields)].reset_index(drop=True)

    def daily_fields_bulk(self, tickers, date, *, fields=('pb',)):
        return {t: {'pb': self._pbs[t]} for t in tickers if t in self._pbs}


def _frames(names, start, periods=15):
    idx = pd.bdate_range(start, periods=periods)
    return {n: pd.DataFrame({'close': np.full(periods, 50.0)}, index=idx)
            for n in names}, idx


# ---------------------------------------------------------------------------
# Default byte-identity pin
# ---------------------------------------------------------------------------
def test_default_composite_is_pinned():
    # THE byte-identity pin for the strengthening round: exact composite
    # values of the pb+netmargin rank blend, the exact ('netmargin',)
    # fundamentals pull (no gp/assets riding along by default), no share
    # pull, no long exclusions. Any default drift fails here.
    funds = [_fund_row('A', 0.30), _fund_row('B', -0.10),
             _fund_row('C', 0.10), _fund_row('D', 0.05)]
    pbs = {'A': 0.8, 'B': 12.0, 'C': 3.0, 'D': 5.0}
    prov = _FakeProvider(funds, pbs)
    desk = ValueQualityDesk(provider=prov)
    assert desk._include_gp_assets is False
    assert desk._issuance_filter is False
    frames, idx = _frames(['A', 'B', 'C', 'D'], '2022-06-01')
    scores = desk._alpha_scores(frames, idx[-1])
    # cheap ranks A 1.0 C .75 D .5 B .25; profit identical -> mean the same
    assert scores == {'A': 1.0, 'C': 0.75, 'D': 0.5, 'B': 0.25}
    assert prov.nm_fields == [('netmargin',)]
    assert prov.share_pulls == 0
    assert desk._long_exclusions(sorted(scores), idx[-1]) == set()


def test_key_is_stable_regardless_of_params():
    # The fund allocation dict addresses the desk by .key
    # (tests/test_vq_fund_gate.py key contract): 'value_quality', never
    # derived from params — variants included.
    prov = _FakeProvider([_fund_row('A', 0.1)], {'A': 1.0})
    assert ValueQualityDesk(provider=prov).key == 'value_quality'
    assert ValueQualityDesk(provider=prov, include_gp_assets=True,
                            issuance_filter=True, long_only=True,
                            exclude_micro=True).key == 'value_quality'


# ---------------------------------------------------------------------------
# Param A — gp_assets composite extension
# ---------------------------------------------------------------------------
_GPA_FUNDS = [
    # A: top pb+nm but NO gp raw -> gp_assets not computable (fallback);
    # X: mid pb/nm, TOP gp/assets (0.8) — the planted Novy-Marx winner;
    # Y: slightly better pb/nm than X, near-zero gp/assets (0.02);
    # F: unit-garbage assets (the VIVS row) -> computed_ratio guard -> fallback.
    _fund_row('A', 0.50, gp=np.nan, assets=100e6),
    _fund_row('X', 0.30, gp=80e6, assets=100e6),
    _fund_row('Y', 0.35, gp=2e6, assets=100e6),
    _fund_row('D', 0.01, gp=30e6, assets=100e6),
    _fund_row('E', -0.05, gp=10e6, assets=100e6),
    _fund_row('F', 0.02, gp=1_677_000.0, assets=3_628.0),
]
_GPA_PBS = {'A': 0.5, 'Y': 1.5, 'X': 2.0, 'D': 5.0, 'E': 8.0, 'F': 10.0}
_GPA_NAMES = sorted(_GPA_PBS)


def test_gp_assets_rank_is_added_at_equal_weight():
    frames, idx = _frames(_GPA_NAMES, '2022-06-01')
    base = ValueQualityDesk(provider=_FakeProvider(_GPA_FUNDS, _GPA_PBS))
    s_base = base._alpha_scores(frames, idx[-1])
    prov = _FakeProvider(_GPA_FUNDS, _GPA_PBS)
    gpa = ValueQualityDesk(provider=prov, include_gp_assets=True)
    s_gpa = gpa._alpha_scores(frames, idx[-1])
    assert s_base is not None and s_gpa is not None
    # Default: Y beats X on both pb and nm. With the gp_assets rank added at
    # an EQUAL weight, the planted high-gp/assets name outranks it:
    # X = mean(4/6, 4/6, 1.0) = 7/9 vs Y = mean(5/6, 5/6, 0.25).
    assert s_base['Y'] > s_base['X']
    assert s_gpa['X'] > s_gpa['Y']
    assert s_gpa['X'] == pytest.approx(7 / 9)
    assert s_gpa['Y'] == pytest.approx((5 / 6 + 5 / 6 + 0.25) / 3)
    # The raws ride the SAME (single) netmargin pull.
    assert prov.nm_fields == [('netmargin', 'gp', 'assets')]


def test_missing_gp_assets_ranks_on_available_factors_only():
    frames, idx = _frames(_GPA_NAMES, '2022-06-01')
    desk = ValueQualityDesk(provider=_FakeProvider(_GPA_FUNDS, _GPA_PBS),
                            include_gp_assets=True)
    scores = desk._alpha_scores(frames, idx[-1])
    assert scores is not None and set(scores) == set(_GPA_NAMES)
    # A (gp NULL) is NOT dropped: it averages its pb+nm ranks alone (both
    # 1.0). F's assets fail the _MIN_ASSETS unit-inconsistency guard, so it
    # falls back the same way: mean(1/6 pb, 3/6 nm) = 1/3 — never the
    # exploded 462x ratio.
    assert scores['A'] == pytest.approx(1.0)
    assert scores['F'] == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# Param B — issuance long-filter
# ---------------------------------------------------------------------------
def _filter_universe():
    """6 scored names. Composite order HVY > NOD > M1 > M2 > M3 > M4.

    Share histories (9 quarters ending 2022-03-31, filed 2022-05-10):
    HVY +0.30 YoY in the FINAL quarter only (the planted heavy issuer — and
    the BEST composite, so only the filter can keep it out of the long
    book; 0.0 before, so the q8 filing date is the PIT boundary); M1 +0.25
    from q5 on (second-heaviest at q8 — must NOT be excluded, the quintile
    is rank-based); M2 0.0; M3 -0.10; M4 -0.20; NOD has NO share rows at
    all (missing data is not evidence of issuance)."""
    funds = [_fund_row('HVY', 0.50), _fund_row('NOD', 0.40),
             _fund_row('M1', 0.30), _fund_row('M2', 0.20),
             _fund_row('M3', 0.10), _fund_row('M4', 0.05)]
    pbs = {'HVY': 0.5, 'NOD': 0.6, 'M1': 1.0, 'M2': 2.0, 'M3': 3.0,
           'M4': 4.0}
    shares = pd.DataFrame(
        _share_rows('HVY', [100.0] * 8 + [130.0])
        + _share_rows('M1', [100.0] * 5 + [125.0] * 4)
        + _share_rows('M2', [100.0] * 9)
        + _share_rows('M3', [100.0] * 8 + [90.0])
        + _share_rows('M4', [100.0] * 8 + [80.0]))
    return funds, pbs, shares


def test_filter_drops_planted_heavy_issuer_from_longs_keeps_no_data_name():
    funds, pbs, shares = _filter_universe()
    desk = ValueQualityDesk(provider=_FakeProvider(funds, pbs, shares),
                            issuance_filter=True, long_only=True)
    frames, idx = _frames(sorted(pbs), '2022-06-01')
    date = idx[-1]
    desk.set_clock(date)
    intents = desk.generate_intents(frames, date,
                                    PortfolioManager(100_000.0))
    # k = int(0.2 * 6) = 1 long. HVY is composite #1 but the top-quintile
    # issuer -> barred from long candidacy; the backfilled long is NOD,
    # which has NO issuance data and is therefore never excluded.
    assert [i.asset.symbol for i in intents if i.action == 'BUY'] == ['NOD']
    assert all(i.asset.symbol != 'HVY' for i in intents)
    assert desk._cache_excluded == {'HVY'}


def test_filter_leaves_the_short_side_alone():
    # Longs only: with the short leg on, the heavy issuer is still eligible
    # to be SHORTED (it ranks top of the composite here so it won't be, but
    # nothing removes it from scores/ranking) — pinned via the score dict:
    # HVY stays scored, at the top, in BOTH variants.
    funds, pbs, shares = _filter_universe()
    desk = ValueQualityDesk(provider=_FakeProvider(funds, pbs, shares),
                            issuance_filter=True)
    frames, idx = _frames(sorted(pbs), '2022-06-01')
    scores = desk._alpha_scores(frames, idx[-1])
    assert scores is not None and 'HVY' in scores
    assert scores['HVY'] == max(scores.values())


def test_exclusion_is_rank_based_top_quintile_not_absolute():
    funds, pbs, shares = _filter_universe()
    desk = ValueQualityDesk(provider=_FakeProvider(funds, pbs, shares),
                            issuance_filter=True)
    ts = pd.Timestamp('2022-06-21')
    # 5 computable issuances -> int(0.2 * 5) = 1 exclusion: ONLY the
    # heaviest (HVY +0.30). M1 (+0.25) survives — there is no absolute
    # threshold to trip.
    assert desk._issuance_exclusions(['HVY', 'M1', 'M2', 'M3', 'M4'],
                                     ts) == {'HVY'}
    # 4 computable -> int(0.2 * 4) = 0: nobody excluded (the base's own
    # int-floor convention; no forced minimum exclusion).
    assert desk._issuance_exclusions(['M1', 'M2', 'M3', 'M4'], ts) == set()


def test_filing_after_the_rebalance_is_invisible_to_the_filter():
    # HVY's +30% quarter files 2022-05-10; a May-1 exclusion pass must not
    # see it (datekey <= ts slice — the PIT convention, pinned here): with
    # HVY still reading 0.0, the top-quintile issuer is M1 (+0.25 since q5).
    funds, pbs, shares = _filter_universe()
    desk = ValueQualityDesk(provider=_FakeProvider(funds, pbs, shares),
                            issuance_filter=True)
    symbols = ['HVY', 'M1', 'M2', 'M3', 'M4']
    assert desk._issuance_exclusions(
        symbols, pd.Timestamp('2022-05-01')) == {'M1'}
    # ... and the SAME desk sees the q8 filing once it lands.
    assert desk._issuance_exclusions(
        symbols, pd.Timestamp('2022-06-01')) == {'HVY'}


def test_stale_issuance_counts_as_missing_not_excluded():
    # All histories file their last quarter 2022-05-10. Within 400 days the
    # heavy issuer is excluded; past 400 days its issuance is STALE and
    # counts as missing (stale data must not bar a 2024 long on 2019-vintage
    # issuance — the issuance-desk/screen convention, adversarial-review
    # finding 2026-07-10).
    funds, pbs, shares = _filter_universe()
    desk = ValueQualityDesk(provider=_FakeProvider(funds, pbs, shares),
                            issuance_filter=True)
    symbols = ['HVY', 'M1', 'M2', 'M3', 'M4']
    assert desk._issuance_exclusions(
        symbols, pd.Timestamp('2023-06-01')) == {'HVY'}   # 387d: fresh
    assert desk._issuance_exclusions(
        symbols, pd.Timestamp('2023-07-01')) == set()     # 417d: stale


def test_one_pull_per_run_and_monthly_cache():
    funds, pbs, shares = _filter_universe()
    prov = _FakeProvider(funds, pbs, shares)
    desk = ValueQualityDesk(provider=prov, issuance_filter=True)
    frames, idx = _frames(sorted(pbs), '2022-06-01')
    scores = desk._alpha_scores(frames, idx[-1])
    assert scores is not None
    assert prov.nm_pulls == 1 and prov.share_pulls == 1
    # Same month => cached object, provider untouched.
    again = desk._alpha_scores(frames, idx[-1] + pd.Timedelta(days=1))
    assert again is scores
    assert prov.nm_pulls == 1 and prov.share_pulls == 1
    # New month, same symbols => recomputed from the ONE cumulative pull.
    frames2, idx2 = _frames(sorted(pbs), '2022-07-01')
    assert desk._alpha_scores(frames2, idx2[-1]) is not None
    assert prov.nm_pulls == 1 and prov.share_pulls == 1


def test_new_symbols_trigger_repull():
    # A name entering all_data mid-run (IPO) must be scored and filterable,
    # not frozen out by the first pull (the PEAD adversarial-review finding)
    # — for BOTH cumulative tables.
    funds, pbs, shares = _filter_universe()
    funds.append(_fund_row('NEW', 0.45))
    pbs = dict(pbs, NEW=0.55)
    shares = pd.concat(
        [shares, pd.DataFrame(_share_rows('NEW', [100.0] * 5 + [140.0] * 4))],
        ignore_index=True)
    prov = _FakeProvider(funds, pbs, shares)
    desk = ValueQualityDesk(provider=prov, issuance_filter=True)
    old = sorted(set(pbs) - {'NEW'})
    frames1, idx1 = _frames(old, '2022-06-01')
    assert desk._alpha_scores(frames1, idx1[-1]) is not None
    assert prov.nm_pulls == 1 and prov.share_pulls == 1
    frames2, idx2 = _frames(old + ['NEW'], '2022-07-01')
    scores2 = desk._alpha_scores(frames2, idx2[-1])
    assert scores2 is not None and 'NEW' in scores2
    assert prov.nm_pulls == 2 and prov.share_pulls == 2
    # NEW is now the heaviest issuer (+0.40 > HVY's +0.30): rank-based top
    # quintile of 6 computable = int(1.2) = 1 -> NEW replaces HVY.
    assert desk._cache_excluded == {'NEW'}


# ---------------------------------------------------------------------------
# Composition — both params on
# ---------------------------------------------------------------------------
def test_both_params_compose():
    funds, pbs, shares = _filter_universe()
    # gp/assets raws: HVY tops the gpa rank too (still composite #1 with the
    # extension on) and stays the planted heavy issuer; NOD has no gp raw
    # (fallback) and no share rows (never excluded).
    by = {f['ticker']: f for f in funds}
    for t, gp in (('HVY', 50e6), ('M1', 40e6), ('M2', 30e6), ('M3', 20e6),
                  ('M4', 10e6)):
        by[t]['gp'] = gp
        by[t]['assets'] = 100e6
    prov = _FakeProvider(funds, pbs, shares)
    desk = ValueQualityDesk(provider=prov, include_gp_assets=True,
                            issuance_filter=True, long_only=True)
    frames, idx = _frames(sorted(pbs), '2022-06-01')
    date = idx[-1]
    desk.set_clock(date)
    intents = desk.generate_intents(frames, date,
                                    PortfolioManager(100_000.0))
    # Param A active: the 3-factor composite scored HVY a perfect 1.0 ...
    assert desk._cache_scores['HVY'] == pytest.approx(1.0)
    assert prov.nm_fields == [('netmargin', 'gp', 'assets')]
    # ... and param B still bars it from the long book.
    assert [i.asset.symbol for i in intents if i.action == 'BUY'] == ['NOD']
    assert desk._cache_excluded == {'HVY'}
