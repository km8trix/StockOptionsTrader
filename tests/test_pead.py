"""Tests for the PEAD stack: SUE math, the eps_quarterly warehouse read, and
the PEADDesk PIT/caching invariants.

SUE math is hand-computed; the warehouse read runs against a hermetic DuckDB-
written parquet fixture (test_pit_warehouse pattern); the desk runs against an
in-memory fake provider — no network, no real warehouse.
"""

from __future__ import annotations

import inspect

import duckdb
import numpy as np
import pandas as pd
import pytest

from data.earnings_surprise import (apply_announcement_dating,
                                    latest_fresh_sue, sue_table)
from data.pit_warehouse import PitWarehouse
from desks.pead import PEADDesk
from portfolio.manager import PortfolioManager


def _eps_rows(ticker, eps_values, start='2020-03-31', lag_days=40):
    """Quarterly EPS rows: reportperiod every 3 months, datekey +lag_days."""
    rows = []
    for q, e in enumerate(eps_values):
        rp = pd.Timestamp(start) + pd.DateOffset(months=3 * q)
        rows.append({'ticker': ticker, 'reportperiod': rp,
                     'datekey': rp + pd.Timedelta(days=lag_days),
                     'epsdil': float(e)})
    return rows


# ---------------------------------------------------------------------------
# sue_table — drift-adjusted Bernard–Thomas SUE
# ---------------------------------------------------------------------------
class TestSueTable:
    def test_hand_computed_value(self):
        # Seasonal diffs by construction: d4=1, d5=2, d6=1, d7=2, d8=3.
        # For q8: past = [1,2,1,2] (>=4 diffs), mean 1.5, std ddof=1 = 0.57735;
        # SUE = (3 - 1.5) / 0.57735 = +2.598.
        eps = pd.DataFrame(_eps_rows('AAA', [1, 1, 1, 1, 2, 3, 2, 3, 5]))
        out = sue_table(eps)
        assert list(out['ticker']) == ['AAA']          # only q8 is computable
        assert out['sue'].iloc[0] == pytest.approx(2.598, abs=1e-3)
        assert out['reportperiod'].iloc[0] == pd.Timestamp('2022-03-31')

    def test_steady_growth_is_not_a_surprise(self):
        # Constant seasonal growth => diffs constant => std 0 => no SUE rows
        # (the drift-adjusted numerator is 0 too; degenerate std is skipped).
        eps = pd.DataFrame(_eps_rows('BBB', [1 + 0.1 * q for q in range(12)]))
        assert len(sue_table(eps)) == 0

    def test_missing_quarter_skips_diff_instead_of_misaligning(self):
        rows = _eps_rows('CCC', [1, 1, 1, 1, 2, 3, 2, 3, 5])
        del rows[2]                                    # lose q2 entirely
        out = sue_table(pd.DataFrame(rows))
        # q6's seasonal match (q2) is gone -> d6 NaN -> q8 has only 3 past
        # diffs -> below min_diffs -> nothing computable.
        assert len(out) == 0

    def test_out_of_order_filing_is_dropped(self):
        # A delinquent filer: q2's FIRST filing lands after q8's — every SUE
        # whose window straddles the disorder would embed EPS not yet public
        # at its own datekey, so those rows must be dropped.
        rows = _eps_rows('ZZZ', [1, 1, 1, 1, 2, 3, 2, 3, 5])
        rows[2]['datekey'] = pd.Timestamp('2023-06-01')   # filed years late
        out = sue_table(pd.DataFrame(rows))
        assert len(out) == 0                  # q8's SUE (the only one) killed

    def test_min_diffs_enforced_and_empty_input(self):
        eps = pd.DataFrame(_eps_rows('DDD', [1, 1, 1, 1, 2, 3]))
        assert len(sue_table(eps)) == 0                # only 2 diffs exist
        empty = sue_table(pd.DataFrame(
            columns=['ticker', 'reportperiod', 'datekey', 'epsdil']))
        assert list(empty.columns) == ['ticker', 'reportperiod', 'datekey',
                                       'sue']

    def test_column_default_pins_epsdil_behavior(self):
        # The column parameter must be a pure generalization for existing
        # desk/screen callers. Comparing default vs column='epsdil' on the
        # NEW code would be tautological; the durable pin is (a) the default
        # IS 'epsdil' and (b) the default path still produces the exact
        # hand-computed SUE the pre-parameter code did (2.598 for this
        # fixture — same value test_revenue_column_scale_free pins for the
        # renamed column). Old-vs-new frame identity was additionally
        # fuzz-verified across NaN/missing-quarter/delinquent fixtures in
        # review (2026-07-10).
        assert (inspect.signature(sue_table).parameters['column'].default
                == 'epsdil')
        eps = pd.DataFrame(_eps_rows('AAA', [1, 1, 1, 1, 2, 3, 2, 3, 5]))
        out = sue_table(eps)
        assert out['sue'].iloc[0] == pytest.approx(2.598, abs=1e-3)

    def test_revenue_column_scale_free(self):
        # SURGE = the same machinery on 'revenue'. Standardization is
        # scale-free, so dollar-scale revenue must reproduce the EPS SUE
        # exactly (same series x 1e9, different column name).
        vals = [1, 1, 1, 1, 2, 3, 2, 3, 5]
        eps = pd.DataFrame(_eps_rows('AAA', vals))
        rev = eps.rename(columns={'epsdil': 'revenue'})
        rev['revenue'] *= 1e9
        out = sue_table(rev, column='revenue')
        assert list(out.columns) == ['ticker', 'reportperiod', 'datekey',
                                     'sue']
        assert out['sue'].iloc[0] == pytest.approx(2.598, abs=1e-3)

    def test_degenerate_guard_holds_at_revenue_scale(self):
        # Steady dollar-scale growth with sub-dollar noise: seasonal diffs
        # of ~$4e8 whose std (~$0.07) is huge vs any ABSOLUTE 1e-9 epsilon
        # but noise vs the diff level. The RELATIVE epsilon
        # (1e-9 * |mean diff| = $0.4 here) must still read this as
        # degenerate. Scope honestly: the guard rejects FLOAT-ARITHMETIC
        # noise only — a grower whose diffs vary at reporting granularity
        # (whole dollars) still standardizes, exactly as sub-cent EPS
        # variation always did.
        rev = pd.DataFrame(_eps_rows(
            'BBB', [1e9 + 1e8 * q + 0.05 * (q % 3) for q in range(12)]))
        rev = rev.rename(columns={'epsdil': 'revenue'})
        assert len(sue_table(rev, column='revenue')) == 0


class TestLatestFreshSue:
    def _sues(self):
        return sue_table(pd.DataFrame(
            _eps_rows('AAA', [1, 1, 1, 1, 2, 3, 2, 3, 5])))

    def test_pit_visibility_and_freshness(self):
        sues = self._sues()
        dk = sues['datekey'].iloc[0]                   # q8 filing date
        assert len(latest_fresh_sue(sues, dk - pd.Timedelta(days=1))) == 0
        s = latest_fresh_sue(sues, dk + pd.Timedelta(days=10))
        assert s['AAA'] == pytest.approx(2.598, abs=1e-3)
        stale = latest_fresh_sue(sues, dk + pd.Timedelta(days=100),
                                 fresh_days=63)
        assert len(stale) == 0

    def test_empty(self):
        assert len(latest_fresh_sue(None, '2020-01-01')) == 0


class TestAnnouncementDating:
    def _sues(self):
        return sue_table(pd.DataFrame(
            _eps_rows('AAA', [1, 1, 1, 1, 2, 3, 2, 3, 5])))

    def test_redates_to_announcement_within_window(self):
        sues = self._sues()                      # one row, datekey rp+40d
        rp = sues['reportperiod'].iloc[0]
        ann = pd.DataFrame({'ticker': ['AAA'], 'date': [rp
                            + pd.Timedelta(days=12)]})
        out = apply_announcement_dating(sues, ann)
        assert out['datekey'].iloc[0] == rp + pd.Timedelta(days=12)

    def test_no_announcement_keeps_filing_date(self):
        sues = self._sues()
        ann = pd.DataFrame({'ticker': ['ZZZ'],
                            'date': [pd.Timestamp('2022-04-15')]})
        out = apply_announcement_dating(sues, ann)
        assert out['datekey'].iloc[0] == sues['datekey'].iloc[0]

    def test_announcement_before_quarter_end_not_used(self):
        sues = self._sues()
        rp = sues['reportperiod'].iloc[0]
        ann = pd.DataFrame({'ticker': ['AAA'],
                            'date': [rp - pd.Timedelta(days=5)]})
        out = apply_announcement_dating(sues, ann)
        assert out['datekey'].iloc[0] == sues['datekey'].iloc[0]

    def test_empty_events_passthrough(self):
        sues = self._sues()
        out = apply_announcement_dating(
            sues, pd.DataFrame(columns=['ticker', 'date']))
        assert out is sues


# ---------------------------------------------------------------------------
# PitWarehouse.eps_quarterly — hermetic parquet fixture
# ---------------------------------------------------------------------------
def _write(path, columns, rows):
    sel = ", ".join(f"{lit} AS {name}"
                    for (name, _t), lit in zip(columns, rows[0]))
    rest = "".join(
        " UNION ALL SELECT " + ", ".join(str(x) for x in r) for r in rows[1:])
    duckdb.connect().execute(
        f"COPY (SELECT {sel}{rest}) TO '{path}' (FORMAT PARQUET)")


class TestEpsQuarterly:
    def test_dedupe_earliest_datekey_and_filters(self, tmp_path):
        cols = [('ticker', 'VARCHAR'), ('dimension', 'VARCHAR'),
                ('reportperiod', 'DATE'), ('datekey', 'DATE'),
                ('epsdil', 'DOUBLE')]
        _write(tmp_path / "sf1.parquet", cols, [
            ("'AAA'", "'ARQ'", "DATE '2020-03-31'", "DATE '2020-05-10'", "1.0"),
            # amendment for the same quarter filed later: must be DROPPED
            ("'AAA'", "'ARQ'", "DATE '2020-03-31'", "DATE '2020-08-01'", "9.0"),
            ("'AAA'", "'ARQ'", "DATE '2020-06-30'", "DATE '2020-08-09'", "1.1"),
            ("'BBB'", "'ARQ'", "DATE '2020-03-31'", "DATE '2020-05-12'", "2.0"),
            # NULL epsdil rows are excluded
            ("'BBB'", "'ARQ'", "DATE '2020-06-30'", "DATE '2020-08-11'", "NULL"),
        ])
        wh = PitWarehouse(str(tmp_path))
        df = wh.eps_quarterly()
        assert len(df) == 3
        aaa_q1 = df[(df['ticker'] == 'AAA')
                    & (df['reportperiod'] == pd.Timestamp('2020-03-31'))]
        assert len(aaa_q1) == 1
        assert aaa_q1['epsdil'].iloc[0] == 1.0         # earliest filing kept
        assert aaa_q1['datekey'].iloc[0] == pd.Timestamp('2020-05-10')
        # tickers filter
        assert set(wh.eps_quarterly(['BBB'])['ticker']) == {'BBB'}
        assert len(wh.eps_quarterly([])) == 0
        # asof filter: only filings KNOWN by the date
        asof = wh.eps_quarterly(asof='2020-06-01')
        assert set(zip(asof['ticker'], asof['epsdil'])) == {('AAA', 1.0),
                                                            ('BBB', 2.0)}

    def test_missing_sf1_returns_empty(self, tmp_path):
        df = PitWarehouse(str(tmp_path)).eps_quarterly(['AAA'])
        assert df.empty and list(df.columns) == ['ticker', 'reportperiod',
                                                 'datekey', 'epsdil']


# ---------------------------------------------------------------------------
# PEADDesk — PIT slicing, monthly cache, banding
# ---------------------------------------------------------------------------
class _FakeProvider:
    def __init__(self, eps, caps=None, events=None, rev=None):
        self._eps = eps
        self._caps = caps or {}
        self._events = events
        self._rev = rev
        self.eps_calls = 0
        self.rev_calls = 0

    def eps_quarterly(self, tickers=None, *, asof=None):
        self.eps_calls += 1
        df = self._eps
        if tickers is not None:
            df = df[df['ticker'].isin(tickers)]
        if asof is not None:
            df = df[df['datekey'] <= pd.Timestamp(asof)]
        return df.reset_index(drop=True)

    def fundamentals_quarterly(self, tickers=None, *, fields=('epsdil',)):
        # The desk's ONLY fundamentals_quarterly use is the variant-only
        # SURGE pull — a separate revenue-FIRST pull (desks/pead.py module
        # docstring: the warehouse NULL-filters on the first field). Any
        # other shape here is a contract drift.
        assert tuple(fields) == ('revenue',), fields
        self.rev_calls += 1
        df = (self._rev if self._rev is not None
              else pd.DataFrame(columns=['ticker', 'reportperiod',
                                         'datekey', 'revenue']))
        if tickers is not None:
            df = df[df['ticker'].isin(tickers)]
        return df.reset_index(drop=True)

    def earnings_events(self, tickers=None):
        if self._events is None:
            return pd.DataFrame(columns=['ticker', 'date'])
        df = self._events
        if tickers is not None:
            df = df[df['ticker'].isin(tickers)]
        return df.reset_index(drop=True)

    def daily_marketcaps(self, tickers, date):
        return {t: self._caps[t] for t in tickers if t in self._caps}


def _desk_frames(names, start, periods=15):
    idx = pd.bdate_range(start, periods=periods)
    return {n: pd.DataFrame({'close': np.full(periods, 50.0)}, index=idx)
            for n in names}, idx


def _surprise_eps():
    """4 names, 12 quarters; big final-quarter surprises for BBB(+) / CCC(-);
    AAA/DDD grow with alternating seasonal diffs (0.2/0.3) so their SUE is
    computable (non-degenerate std) but small."""
    rows = []
    for name, jump in (('AAA', 0.0), ('BBB', 4.0), ('CCC', -4.0), ('DDD', 0.0)):
        eps = [1.0 + 0.05 * q for q in range(4)]
        for q in range(4, 12):
            eps.append(eps[q - 4] + (0.2 if q % 2 else 0.3))
        eps[11] += jump
        rows.extend(_eps_rows(name, eps))
    return pd.DataFrame(rows)


def _jump_rows(jumps, column='epsdil', lag_days=40):
    """12 quarters per name (the _surprise_eps base: alternating 0.2/0.3
    seasonal diffs so the surprise std is non-degenerate) with each name's
    ``jumps`` value added to the FINAL quarter — the resulting SUE/SURGE
    ordering is monotone in the jump. q11 reports 2022-12-31 and files
    ``lag_days`` later (2023-02-09 at the default 40)."""
    rows = []
    for name, jump in jumps.items():
        vals = [1.0 + 0.05 * q for q in range(4)]
        for q in range(4, 12):
            vals.append(vals[q - 4] + (0.2 if q % 2 else 0.3))
        vals[11] += jump
        for q, v in enumerate(vals):
            rp = pd.Timestamp('2020-03-31') + pd.DateOffset(months=3 * q)
            rows.append({'ticker': name, 'reportperiod': rp,
                         'datekey': rp + pd.Timedelta(days=lag_days),
                         column: float(v)})
    return pd.DataFrame(rows)


class TestPEADDesk:
    def test_scores_sign_and_monthly_cache(self):
        prov = _FakeProvider(_surprise_eps())
        desk = PEADDesk(provider=prov, quantile=0.25)
        frames, idx = _desk_frames(['AAA', 'BBB', 'CCC', 'DDD'], '2023-02-01')
        scores = desk._alpha_scores(frames, idx[-1])   # after q11 datekeys
        assert scores is not None
        assert scores['BBB'] > 2 and scores['CCC'] < -2
        assert abs(scores['AAA']) < 2 and abs(scores['DDD']) < 2
        # Same month => cached, provider not re-queried.
        calls = prov.eps_calls
        again = desk._alpha_scores(frames, idx[-1] + pd.Timedelta(days=1))
        assert again is scores and prov.eps_calls == calls

    def test_pit_slicing_hides_future_filings(self):
        prov = _FakeProvider(_surprise_eps())
        desk = PEADDesk(provider=prov, quantile=0.25)
        # q11 (the surprise quarter) files 2023-02-09; a January date must
        # not see it even though the desk's one-shot pull contains it.
        frames, _ = _desk_frames(['AAA', 'BBB', 'CCC', 'DDD'], '2023-01-02')
        early = pd.Timestamp('2023-01-20')
        s = desk._alpha_scores(frames, early)
        if s is not None:
            assert abs(s.get('BBB', 0.0)) < 2, s       # surprise invisible
        # ...and the SAME desk sees it once the filing lands (new month).
        frames2, idx2 = _desk_frames(['AAA', 'BBB', 'CCC', 'DDD'],
                                     '2023-02-01')
        s2 = desk._alpha_scores(frames2, idx2[-1])
        assert s2 is not None and s2['BBB'] > 2

    def test_band_filter_uses_provider_caps(self):
        caps = {'AAA': 1e8, 'BBB': 5e8, 'CCC': 2e9, 'DDD': 8e9}
        prov = _FakeProvider(_surprise_eps(), caps=caps)
        desk = PEADDesk('micro', provider=prov, quantile=0.25)
        desk.min_scored = 1                            # tiny universe
        frames, idx = _desk_frames(['AAA', 'BBB', 'CCC', 'DDD'], '2023-02-01')
        scores = desk._alpha_scores(frames, idx[-1])
        # micro tercile of 4 names by cap = bucket 0 => AAA (and BBB w/ n=4//3)
        assert scores is not None
        assert set(scores) <= {'AAA', 'BBB'}
        assert 'CCC' not in scores and 'DDD' not in scores

    def test_new_symbols_trigger_repull(self):
        """A name entering all_data mid-run (IPO) must be scored, not frozen
        out by the first pull (adversarial-review finding)."""
        eps = _surprise_eps()
        extra = _eps_rows('EEE', [1.0 + 0.05 * q + (0.2 if q % 2 else 0.3)
                                  * (q // 4 + 1) for q in range(12)])
        prov = _FakeProvider(pd.concat([eps, pd.DataFrame(extra)],
                                       ignore_index=True))
        desk = PEADDesk(provider=prov, quantile=0.25)
        frames1, idx1 = _desk_frames(['AAA', 'BBB', 'CCC', 'DDD'],
                                     '2023-02-01')
        assert desk._alpha_scores(frames1, idx1[-1]) is not None
        calls = prov.eps_calls
        # New month AND a new symbol: the desk must re-pull and score EEE.
        frames2, idx2 = _desk_frames(['AAA', 'BBB', 'CCC', 'DDD', 'EEE'],
                                     '2023-03-01')
        scores2 = desk._alpha_scores(frames2, idx2[-1])
        assert prov.eps_calls == calls + 1
        assert scores2 is not None and 'EEE' in scores2

    def test_below_min_scored_flattens(self):
        prov = _FakeProvider(_surprise_eps())
        desk = PEADDesk(provider=prov)                 # min_scored default 4
        frames, idx = _desk_frames(['AAA', 'BBB'], '2023-02-01')
        assert desk._alpha_scores(frames, idx[-1]) is None

    def test_announce_dating_sees_surprise_before_filing(self):
        """The unlock: q11 announces 2023-01-10 but files 2023-02-09; an
        announce-dated desk trades the surprise a month earlier."""
        eps = _surprise_eps()
        names = ['AAA', 'BBB', 'CCC', 'DDD']
        ann_rows = []
        for n in names:
            for rp in eps[eps['ticker'] == n]['reportperiod']:
                ann_rows.append({'ticker': n,
                                 'date': rp + pd.Timedelta(days=10)})
        events = pd.DataFrame(ann_rows)         # every quarter: rp+10d
        asof = pd.Timestamp('2023-01-20')       # after ann, before filing
        frames, _ = _desk_frames(names, '2023-01-02')
        filing_desk = PEADDesk(provider=_FakeProvider(eps), quantile=0.25)
        s_filing = filing_desk._alpha_scores(frames, asof)
        assert s_filing is None or abs(s_filing.get('BBB', 0.0)) < 2
        ann_desk = PEADDesk(provider=_FakeProvider(eps, events=events),
                            quantile=0.25, dating='announce')
        s_ann = ann_desk._alpha_scores(frames, asof)
        assert s_ann is not None and s_ann['BBB'] > 2 and s_ann['CCC'] < -2

    def test_band_validation(self):
        with pytest.raises(ValueError):
            PEADDesk('mega', provider=_FakeProvider(_surprise_eps()))
        with pytest.raises(ValueError):
            PEADDesk(provider=_FakeProvider(_surprise_eps()),
                     dating='psychic')

    def test_fixed_rule_has_no_walk_forward_fits(self):
        desk = PEADDesk(provider=_FakeProvider(_surprise_eps()))
        assert desk.walk_forward_fits == []


# ---------------------------------------------------------------------------
# Pre-registered SURGE variants (surge_confirm / rank_combine)
# ---------------------------------------------------------------------------
#: SUE jumps: strict long ranking N1 > N2 > ... > N6 (6 scored names).
_SUE_JUMPS = {'N1': 6.0, 'N2': 5.0, 'N3': 4.0, 'N4': 3.0,
              'N5': -2.0, 'N6': -3.0}
#: Revenue (SURGE) jumps — N2 and N6 have NO revenue rows at all (missing
#: data is never evidence). Computable SURGEs order N3 > N4 > N5 > N1 with
#: median between N4 and N5, so N1 and N5 sit strictly below it.
_REV_JUMPS = {'N1': -4.0, 'N3': 3.0, 'N4': 2.0, 'N5': -3.0}


class TestPEADSurgeVariants:
    def test_default_desk_never_pulls_revenue_or_excludes(self):
        # THE byte-identity pin: with both variant flags off (the default)
        # the desk never touches fundamentals_quarterly (rev present in the
        # provider but unpulled), scores stay RAW SUE (not ranks), and the
        # long-exclusion hook serves the base's empty set.
        prov = _FakeProvider(_surprise_eps(),
                             rev=_jump_rows(_REV_JUMPS, column='revenue'))
        desk = PEADDesk(provider=prov, quantile=0.25)
        assert desk._surge_confirm is False and desk._rank_combine is False
        frames, idx = _desk_frames(['AAA', 'BBB', 'CCC', 'DDD'], '2023-02-01')
        scores = desk._alpha_scores(frames, idx[-1])
        assert scores is not None and scores['BBB'] > 2     # raw SUE scale
        assert prov.rev_calls == 0
        assert desk._long_exclusions(sorted(scores), idx[-1]) == set()

    def test_variants_are_separate_trials_not_composable(self):
        # Composing the two was never pre-registered (2 trials this round).
        with pytest.raises(ValueError):
            PEADDesk(provider=_FakeProvider(_surprise_eps()),
                     surge_confirm=True, rank_combine=True)

    def test_surge_confirm_drops_low_surge_long_keeps_no_data_name(self):
        prov = _FakeProvider(_jump_rows(_SUE_JUMPS),
                             rev=_jump_rows(_REV_JUMPS, column='revenue'))
        desk = PEADDesk(provider=prov, quantile=0.4, long_only=True,
                        surge_confirm=True)
        frames, idx = _desk_frames(sorted(_SUE_JUMPS), '2023-02-01')
        date = idx[-1]
        desk.set_clock(date)
        intents = desk.generate_intents(frames, date,
                                        PortfolioManager(100_000.0))
        # k = int(0.4 * 6) = 2 longs. SUE ranks N1 first, but N1's SURGE
        # (-4 jump) sits strictly below the computable-SURGE median -> barred
        # from long candidacy. N2 has NO revenue data and is KEPT (missing
        # data never excludes); N3 backfills the second slot.
        assert ([i.asset.symbol for i in intents if i.action == 'BUY']
                == ['N2', 'N3'])
        assert desk._cache_excluded == {'N1', 'N5'}
        # The scores themselves stay raw SUE — the filter only touches long
        # candidacy (N1 still tops the ranking).
        assert desk._cache_scores['N1'] == max(desk._cache_scores.values())
        # One cumulative pull per table; same month => cached, untouched.
        assert prov.eps_calls == 1 and prov.rev_calls == 1
        desk._alpha_scores(frames, date + pd.Timedelta(days=1))
        assert prov.eps_calls == 1 and prov.rev_calls == 1

    def test_rank_combine_planted_ordering_and_mean_of_available(self):
        sue_jumps = {'A': 4.0, 'B': 3.0, 'C': 2.0, 'D': 1.0}
        rev_jumps = {'A': -4.0, 'B': 4.0, 'D': 2.0}     # C: no revenue data
        prov = _FakeProvider(_jump_rows(sue_jumps),
                             rev=_jump_rows(rev_jumps, column='revenue'))
        desk = PEADDesk(provider=prov, quantile=0.25, rank_combine=True)
        frames, idx = _desk_frames(sorted(sue_jumps), '2023-02-01')
        scores = desk._alpha_scores(frames, idx[-1])
        # The scored SET is unchanged: SURGE never adds or drops a name.
        assert scores is not None and set(scores) == set(sue_jumps)
        # SUE pct ranks A 1.0 / B .75 / C .5 / D .25; SURGE pct ranks among
        # the three computable B 1.0 / D 2/3 / A 1/3. Equal-weight mean of
        # the AVAILABLE ranks — C (no SURGE) ranks on SUE alone.
        assert scores['B'] == pytest.approx((0.75 + 1.0) / 2)
        assert scores['A'] == pytest.approx((1.0 + 1 / 3) / 2)
        assert scores['C'] == pytest.approx(0.5)
        assert scores['D'] == pytest.approx((0.25 + 2 / 3) / 2)
        # The combine genuinely reorders the book: B overtakes the top SUE
        # name A on the strength of its revenue surprise.
        assert scores['B'] == max(scores.values())

    def test_revenue_pit_boundary_and_freshness_mirror_sue(self):
        # Revenue files 80d after quarter end here: q10 (no jumps) files
        # 2022-12-19, the q11 jump quarter 2023-03-21.
        rev = _jump_rows(_REV_JUMPS, column='revenue', lag_days=80)
        prov = _FakeProvider(_jump_rows(_SUE_JUMPS), rev=rev)
        desk = PEADDesk(provider=prov, surge_confirm=True)
        symbols = sorted(_SUE_JUMPS)
        # PIT: before the q11 revenue files, the latest fresh row is q10 —
        # every computable name reads the SAME no-jump SURGE, so nobody is
        # strictly below the median. The jumps are INVISIBLE.
        assert (desk._surge_exclusions(symbols, pd.Timestamp('2023-02-10'))
                == set())
        # ... and the SAME desk sees them once the q11 revenue files.
        assert (desk._surge_exclusions(symbols, pd.Timestamp('2023-03-25'))
                == {'N1', 'N5'})
        # Freshness mirrors SUE's window: 72d after the filing (> the
        # desk's fresh_days=63) the row is STALE = missing = NOT excluded...
        assert (desk._surge_exclusions(symbols, pd.Timestamp('2023-06-01'))
                == set())
        # ... while a desk with a wider fresh_days (the SAME constructor
        # knob SUE uses) still sees it — no hardcoded 63 for SURGE.
        wide = PEADDesk(provider=_FakeProvider(_jump_rows(_SUE_JUMPS),
                                               rev=rev),
                        surge_confirm=True, fresh_days=200)
        assert (wide._surge_exclusions(symbols, pd.Timestamp('2023-06-01'))
                == {'N1', 'N5'})

    def test_announce_dating_applies_to_surge_rows_too(self):
        # q11 announces 2023-01-10 (rp+10d) but files 2023-02-09; the SURGE
        # confirm-filter must ride the SAME 8-K re-dating as SUE and see the
        # revenue jumps a month before their filings.
        eps = _jump_rows(_SUE_JUMPS)
        rev = _jump_rows(_REV_JUMPS, column='revenue')
        ann_rows = [{'ticker': n, 'date': rp + pd.Timedelta(days=10)}
                    for n, g in eps.groupby('ticker')
                    for rp in g['reportperiod']]
        prov = _FakeProvider(eps, events=pd.DataFrame(ann_rows), rev=rev)
        desk = PEADDesk(provider=prov, quantile=0.4, long_only=True,
                        surge_confirm=True, dating='announce')
        frames, _ = _desk_frames(sorted(_SUE_JUMPS), '2023-01-02')
        asof = pd.Timestamp('2023-01-20')   # after the 8-K, before filing
        scores = desk._alpha_scores(frames, asof)
        assert scores is not None and len(scores) == 6
        assert desk._cache_excluded == {'N1', 'N5'}


# ---------------------------------------------------------------------------
# scripts/pead_desk_gate.py — the variant registry + construction pins
# ---------------------------------------------------------------------------
def test_pead_variant_kwargs_are_the_preregistered_trials():
    # The strengthening round is pre-registered: base is EMPTY (byte-
    # identical default desk) and exactly two trial variants exist — the
    # SURGE median confirm-filter and the SUE+SURGE equal-weight rank
    # combine. Adding or tuning an entry must fail this test and be argued
    # in review, not slipped in.
    import scripts.pead_desk_gate as pdg
    assert pdg.PEAD_VARIANT_KWARGS == {
        'base': {},
        'surge_confirm': {'surge_confirm': True},
        'rank_combine': {'rank_combine': True}}


def test_run_gate_applies_variant_kwargs(monkeypatch):
    # Constructor-level pin (the test_vq_fund_gate stubbing pattern): the
    # solo gate threads --variant into PEADDesk as its registered kwargs,
    # and the default call is byte-identical to before --variant existed.
    import scripts.pead_desk_gate as pdg
    captured = {}
    report = {'summary': {'total_return_pct': 1.0},
              'portfolio_history': [], 'closed_trades': []}

    class FakeDesk:
        def __init__(self, band, **kwargs):
            captured['band'] = band
            captured['kwargs'] = kwargs

    class FakeEngine:
        def __init__(self, **kwargs):
            pass

        def run(self, symbols, start, end):
            return dict(report)

    monkeypatch.setattr(pdg, 'PitWarehouse', lambda: 'WH')
    monkeypatch.setattr(pdg, 'resolve_universe',
                        lambda wh, rb, scales: ['AAA', 'BBB'])
    monkeypatch.setattr(pdg, 'WarehouseMarketData', lambda wh: 'FEED')
    monkeypatch.setattr(pdg, 'PEADDesk', FakeDesk)
    monkeypatch.setattr(pdg, 'BacktestEngine', FakeEngine)
    monkeypatch.setattr(pdg, '_daily_returns_with_years',
                        lambda history: ('R', 'Y'))
    monkeypatch.setattr(pdg, 'validate_strategy_oos',
                        lambda returns, years, psr_threshold: {'psr': 0.5})

    pdg.run_gate('micro', '2015-01-01', '2024-12-31', long_only=True,
                 dating='announce', variant='surge_confirm')
    assert captured['band'] == 'micro'
    assert captured['kwargs'] == {'provider': 'WH', 'long_only': True,
                                  'dating': 'announce',
                                  'surge_confirm': True}
    # Default: NO variant kwargs at all — byte-identical construction.
    pdg.run_gate('micro', '2015-01-01', '2024-12-31')
    assert captured['kwargs'] == {'provider': 'WH', 'long_only': False,
                                  'dating': 'filing'}
