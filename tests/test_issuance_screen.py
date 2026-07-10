"""Tests for the net-issuance screen: YoY split-adjusted share-growth math,
the PIT collect, the low-issuance-long sign convention, and the long-leg
turnover measurement. All hermetic — no network, no real warehouse."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.share_issuance import _share_rows, issuance_table
from scripts.factor_screen import factor_study
from scripts.issuance_screen import collect_issuance_events, long_leg_turnover


def test_screen_reexports_the_data_seam():
    # The signal math lives in data/share_issuance (the sue_table seam
    # convention); the screen re-imports it. Identity — not equality — pins
    # that there is exactly ONE definition, so screen and desk can never
    # drift apart.
    import scripts.issuance_screen as screen
    assert screen.issuance_table is issuance_table
    assert screen._share_rows is _share_rows


# ---------------------------------------------------------------------------
# issuance_table — YoY split-adjusted share growth
# ---------------------------------------------------------------------------
class TestIssuanceTable:
    def test_hand_computed_value_and_min_history(self):
        # 100 shares for a year, then 110: q4..q7 read 0, q8 reads +0.10.
        sh = pd.DataFrame(_share_rows('AAA', [100] * 8 + [110]))
        out = issuance_table(sh)
        # q0..q3 have <4 prior filings (and no seasonal match) — dropped.
        assert list(out['reportperiod']) == [
            pd.Timestamp('2020-03-31') + pd.DateOffset(months=3 * q)
            for q in range(4, 9)]
        assert out['issuance'].iloc[:-1].abs().max() < 1e-12
        assert out['issuance'].iloc[-1] == pytest.approx(0.10)

    def test_buyback_is_negative(self):
        sh = pd.DataFrame(_share_rows('BBK', [100] * 5 + [90] * 4))
        out = issuance_table(sh)
        assert out['issuance'].iloc[-1] == pytest.approx(-0.10)

    def test_split_is_not_issuance(self):
        # Real Sharadar split semantics: the vendor retroactively restates
        # sharesbas to the CURRENT split basis (verified 2026-07-10 against
        # the warehouse — NVDA/TSLA/AAPL forward, CERO/UNCY reverse), so a
        # split is invisible in the data a run ever sees: both quarters
        # carry the same basis, sharefactor stays constant.
        rows = _share_rows('SPL', [200.0] * 9)   # post-2:1 restated vintage
        assert issuance_table(pd.DataFrame(rows))['issuance'].abs().max() < 1e-12

    def test_adr_ratio_change_is_not_issuance(self):
        # sharefactor is an ADR/unit multiplier, near-constant per ticker
        # (BRK.B, V, ONON class). On a ratio change the vendor moves the
        # count and the factor inversely; using each quarter's OWN factor
        # keeps the economic share count flat — no phantom issuance.
        rows = (_share_rows('ADR', [100.0] * 5)
                + _share_rows('ADR', [200.0] * 4, start='2021-06-30',
                              factor=0.5))
        assert issuance_table(pd.DataFrame(rows))['issuance'].abs().max() < 1e-12

    def test_zero_denominator_guarded(self):
        # sharefactor==0 garbage: neither divides (year-later row) nor emits
        # a row itself; every surviving issuance stays finite.
        rows = _share_rows('ZRO', [100.0] * 9)
        rows[2]['sharefactor'] = 0.0
        out = issuance_table(pd.DataFrame(rows))
        assert pd.Timestamp('2021-09-30') not in set(out['reportperiod'])
        assert np.isfinite(out['issuance']).all()

    def test_null_field_at_either_quarter_skips(self):
        rows = _share_rows('NUL', [100.0] * 9)
        rows[4]['sharesbas'] = np.nan               # current-quarter NULL
        rows[1]['sharefactor'] = np.nan             # year-ago NULL (for q5)
        out = issuance_table(pd.DataFrame(rows))
        got = set(out['reportperiod'])
        assert pd.Timestamp('2021-03-31') not in got     # q4 itself NULL
        assert pd.Timestamp('2021-06-30') not in got     # q5's match NULL

    def test_missing_year_ago_quarter_skips_not_misaligns(self):
        rows = _share_rows('GAP', [100.0] * 4 + [200.0] * 5)
        del rows[1]                                  # lose q1 entirely
        out = issuance_table(pd.DataFrame(rows))
        # q5's seasonal match (q1, 330-410d back) is gone — no row, and no
        # silent fallback to q0 or q2.
        assert pd.Timestamp('2021-06-30') not in set(out['reportperiod'])

    def test_out_of_order_filing_dropped(self):
        rows = _share_rows('ZZZ', [100.0] * 9)
        rows[8]['datekey'] = pd.Timestamp('2020-06-01')  # filed before q0..q7
        out = issuance_table(pd.DataFrame(rows))
        assert pd.Timestamp('2022-03-31') not in set(out['reportperiod'])

    def test_empty_input(self):
        empty = issuance_table(pd.DataFrame(
            columns=['ticker', 'reportperiod', 'datekey', 'sharesbas',
                     'sharefactor']))
        assert list(empty.columns) == ['ticker', 'reportperiod', 'datekey',
                                       'issuance']


# ---------------------------------------------------------------------------
# collect_issuance_events — PIT visibility + staleness
# ---------------------------------------------------------------------------
class _FakeProv:
    """Minimal PitWarehouse stand-in: rising prices + canned share rows."""

    def __init__(self, rows):
        self._rows = pd.DataFrame(rows)
        self._idx = pd.bdate_range('2020-01-01', periods=700)

    def fundamentals_quarterly(self, names, *, fields):
        assert tuple(fields) == ('sharesbas', 'sharefactor')
        return self._rows[self._rows['ticker'].isin(names)]

    def prices(self, name, start, end):
        return pd.Series(np.linspace(10, 20, len(self._idx)),
                         index=self._idx, name=name)

    def daily_metric(self, name, t):
        return {'marketcap': 5e8}


def test_collect_is_pit_and_respects_staleness():
    prov = _FakeProv(_share_rows('AAA', [100] * 8 + [110]))
    rebal = pd.bdate_range('2020-06-01', '2024-06-01', freq='BMS')
    ev, n = collect_issuance_events(prov, ['AAA'], rebal, [21],
                                    stale_days=400)
    assert n == 1
    # First computable filing is q4 (reportperiod 2021-03-31, datekey +40d):
    # nothing may appear at any earlier rebalance (PIT).
    assert ev['date'].min() >= pd.Timestamp('2021-05-10')
    # Last filing q8 datekey 2022-05-10 + 400 stale days ~ 2023-06-14: the
    # 2023-07+ rebalances carry no stale signal.
    assert ev['date'].max() <= pd.Timestamp('2023-06-14')
    assert {'issuance', 'mcap', 'fwd_21'} <= set(ev.columns)


def test_collect_datekey_equal_to_rebalance_not_usable():
    # side='left': a filing whose datekey IS the rebalance date is invisible.
    rows = _share_rows('AAA', [100] * 9)
    dk = pd.Timestamp(rows[8]['datekey'])
    prov = _FakeProv(rows)
    ev, _ = collect_issuance_events(prov, ['AAA'], [dk], [21], stale_days=30)
    assert len(ev) == 0                    # q7 is >30d stale, q8 not yet known


# ---------------------------------------------------------------------------
# Sign convention + turnover
# ---------------------------------------------------------------------------
def _planted_events(edge=0.002, n_dates=24, k=20, seed=0):
    """HIGH issuance -> LOWER forward return (the academic direction)."""
    rng = np.random.default_rng(seed)
    recs = []
    for d in range(n_dates):
        date = pd.Timestamp('2021-01-01') + pd.offsets.BMonthBegin(d)
        for i in range(k):
            iss = (i - k / 2) / k                   # negative..positive
            recs.append({'date': date, 'name': f'N{i}', 'issuance': iss,
                         'fwd_21': -edge * iss + rng.normal(0, 1e-4)})
    return pd.DataFrame(recs)


def test_low_issuance_long_earns_planted_spread():
    s = factor_study(_planted_events(), 'issuance', [21], cost_bps=0.0,
                     cheap_is_long=True, drop_nonpositive=False)[0]
    assert s['gross_spread'] > 0 and s['t'] > 3


def test_drop_nonpositive_would_gut_the_long_leg():
    # The buyback (negative-issuance) names ARE the long leg; value-style
    # nonpositive-dropping must change the book — pin that the screen's
    # drop_nonpositive=False choice is load-bearing.
    ev = _planted_events()
    kept = factor_study(ev, 'issuance', [21], 0.0, cheap_is_long=True,
                        drop_nonpositive=False)[0]
    dropped = factor_study(ev, 'issuance', [21], 0.0, cheap_is_long=True,
                           drop_nonpositive=True)[0]
    assert kept['gross_spread'] != dropped['gross_spread']


def test_long_leg_turnover_static_and_full_churn():
    ev = _planted_events()
    to = long_leg_turnover(ev)
    assert to['n_pairs'] == 23 and to['per_rebalance'] == 0.0
    assert to['annualized'] == 0.0
    # Full churn: k=4 of 20, shift which names hold the bottom ranks by 4
    # each date so the leg never repeats a member.
    recs = []
    for d in range(6):
        date = pd.Timestamp('2021-01-01') + pd.offsets.BMonthBegin(d)
        for i in range(20):
            recs.append({'date': date, 'name': f'N{(i + 4 * d) % 20}',
                         'issuance': i / 20.0, 'fwd_21': 0.0})
    to2 = long_leg_turnover(pd.DataFrame(recs))
    assert to2['per_rebalance'] == 1.0     # bottom-4 leg fully replaced

    assert long_leg_turnover(ev.iloc[:20]) == {}   # single date -> no pairs
