"""Tests for the pure factor_study core of scripts/factor_screen."""

import numpy as np
import pandas as pd
import pytest

from analysis.research_stats import benjamini_hochberg
from scripts.factor_screen import _print_report, factor_study
from scripts.value_validate import _date_spreads


def _events(n_dates=24, k_per=20, edge=0.02, seed=0):
    """Each date: k_per names with a value ratio in [1, k_per]; the CHEAP names
    (low ratio) get a higher forward return by `edge`, so a cheap-is-long
    long-short should earn ~`edge` (+ a per-date market shock that cancels)."""
    rng = np.random.default_rng(seed)
    recs = []
    for d in range(n_dates):
        shock = rng.normal(0, 0.05)                 # common market move (cancels)
        date = pd.Timestamp('2015-01-01') + pd.offsets.BMonthBegin(d)
        for i in range(k_per):
            ratio = float(i + 1)                    # 1 = cheapest
            # cheaper (low ratio) -> higher return: linear tilt around the mean
            tilt = edge * (k_per / 2 - i) / k_per
            recs.append({'date': date, 'name': f'N{i}', 'pb': ratio,
                         'fwd_21': shock + tilt + rng.normal(0, 1e-4)})
    return pd.DataFrame(recs)


def test_cheap_long_earns_positive_spread():
    s = factor_study(_events(), 'pb', [21], cost_bps=0.0)[0]
    assert s['gross_spread'] > 0        # cheap outperforms rich
    assert s['t'] > 3
    assert s['n_dates'] == 24


def test_positive_spread_survives_market_shock():
    # the per-date common shock cancels in the long-short; spread stays positive
    base = factor_study(_events(seed=1), 'pb', [21], cost_bps=0.0)[0]
    assert base['gross_spread'] > 0


def test_nonpositive_factor_values_dropped():
    ev = _events()
    ev.loc[ev['name'] == 'N0', 'pb'] = -1.0     # a "cheap" trap (neg ratio)
    s = factor_study(ev, 'pb', [21], cost_bps=0.0)[0]
    # still forms a book from the remaining positive names; no crash
    assert not s.get('insufficient')


def test_one_way_cost_charges_entry_and_exit_on_both_legs():
    s = factor_study(_events(), 'pb', [21], cost_bps=25.0)[0]
    # Long entry + long exit + short entry + short exit = four 25bp trades.
    assert abs(s['net_spread'] - (s['gross_spread'] - 0.0100)) < 1e-12
    assert s['mean_total_cost_bps'] == 100.0
    assert s['cost_model'] == 'fixed_one_way_four_trade_round_trip'


def test_inference_and_bh_use_net_period_spreads_not_gross():
    # The planted gross edge is highly significant, but 200bp/leg makes the
    # per-date net series significantly NEGATIVE. The directional p-value must
    # be near one so BH cannot promote a losing-after-cost signal.
    s = factor_study(_events(), 'pb', [21], cost_bps=200.0)[0]
    assert s['raw_gross_spread'] > 0
    assert s['raw_gross_t'] > 3
    assert s['raw_net_spread'] < 0
    assert s['t'] < -3
    assert s['p'] > 0.5
    assert s['p_two_sided'] < 0.05       # retained diagnostic, not BH input
    assert benjamini_hochberg([s['p']])['rejected_bh'] == [False]


def test_dated_total_spread_costs_are_deducted_per_period():
    ev = _events()
    dates = sorted(ev['date'].unique())
    # Total long+short strategy-return drag; deliberately time-varying so the
    # HAC test sees the actual net array, not merely a shifted headline mean.
    costs = {d: (10.0 if i % 2 == 0 else 90.0)
             for i, d in enumerate(dates)}
    s = factor_study(ev, 'pb', [21], cost_bps=999.0,
                     spread_cost_bps_by_date=costs)[0]
    assert s['cost_model'] == 'dated_total_spread_cost'
    assert s['mean_total_cost_bps'] == pytest.approx(50.0)
    assert s['raw_net_spread'] == pytest.approx(
        s['raw_gross_spread'] - 50.0 / 1e4)
    # Variation in dated costs changes the uncertainty, proving t is fit on
    # the per-date net values rather than copied from the gross fit.
    assert s['raw_net_t'] != pytest.approx(s['raw_gross_t'])


def test_persisted_inference_series_recomputes_reported_economics():
    s = factor_study(
        _events(), 'pb', [21], cost_bps=25.0,
        winsor_returns=0.1, include_series=True,
    )[0]
    series = s['inference_series']

    assert len(series['formation_dates']) == s['n_dates']
    assert np.mean(series['raw_gross_spreads']) == pytest.approx(
        s['raw_gross_spread'])
    assert np.mean(series['raw_net_spreads']) == pytest.approx(
        s['raw_net_spread'])
    assert np.mean(series['total_cost_drags']) * 1e4 == pytest.approx(
        s['mean_total_cost_bps'])
    assert np.asarray(series['raw_gross_spreads']) - np.asarray(
        series['total_cost_drags']) == pytest.approx(
            series['raw_net_spreads'])
    assert np.mean(series['robust_net_spreads']) == pytest.approx(
        s['robust_net_spread'])


def test_dated_costs_fail_closed_when_a_formation_date_is_missing():
    ev = _events()
    dates = sorted(ev['date'].unique())
    with pytest.raises(ValueError, match='missing spread cost'):
        factor_study(ev, 'pb', [21], 0.0,
                     spread_cost_bps_by_date={d: 20.0 for d in dates[:-1]})


def test_too_few_dates_flagged():
    s = factor_study(_events(n_dates=6), 'pb', [21], cost_bps=0.0)[0]
    assert s['insufficient'] is True


def test_winsor_returns_tames_outlier_leg():
    # one +100x name in the cheap (long) leg blows up the raw spread; per-date
    # winsorization clips it back to the leg's bulk, shrinking the spread.
    ev = _events()
    ev.loc[ev['name'] == 'N0', 'fwd_21'] = 100.0          # N0 = cheapest -> long
    raw = factor_study(ev, 'pb', [21], 0.0)[0]
    wins = factor_study(ev, 'pb', [21], 0.0, winsor_returns=0.1)[0]
    assert wins['raw_gross_spread'] == raw['raw_gross_spread']
    assert wins['raw_gross_spread'] > 5 * wins['robust_gross_spread']
    # Compatibility aliases and BH stay on the raw executable series;
    # winsorized values are diagnostics and cannot create a promotion result.
    assert wins['gross_spread'] == wins['raw_gross_spread']
    assert wins['net_spread'] == wins['raw_net_spread']
    assert wins['inference_basis'] == 'raw_net'
    assert wins['inference_is_executable'] is True
    assert wins['robust_inference_basis'] == 'winsorized_net_diagnostic'
    assert wins['p'] == wins['raw_net_p']
    assert wins['t'] == wins['raw_net_t']


def test_print_report_separates_raw_economics_from_robust_inference(capsys):
    stats = [{'factor': 'pb', 'h': 21, 'n_dates': 24,
              'raw_gross_spread': 0.0123, 'raw_net_spread': 0.0063,
              'robust_net_spread': 0.0051, 'gross_spread': 0.0111,
              'net_spread': 0.0051, 't': 2.5, 'p': 0.0062}]
    _print_report(stats, None, 30.0, 10, 240)
    out = capsys.readouterr().out
    assert 'raw-g%' in out and 'raw-n%' in out and 'rob-n%' in out
    assert "         pb  21     24   +1.230   +0.630   +0.510   +2.50   0.0062" in out
    assert 'winsorized robustness diagnostic only' in out
    assert 'never drives BH or promotion' in out
    assert 'one-sided H1 net-mean>0 p-values' in out
    _print_report(stats, None, 30.0, 10, 240, width=15)
    wide = capsys.readouterr().out
    assert "             pb  21" in wide          # factor column widened


def test_print_report_verdict_requires_positive_raw_net_economics(capsys):
    stats = [{'factor': 'pb', 'h': 21, 'n_dates': 24,
              'raw_gross_spread': 0.002, 'raw_net_spread': -0.004,
              'robust_net_spread': 0.005, 'gross_spread': 0.011,
              'net_spread': 0.005, 't': 3.0, 'p': 0.001}]
    bh = {'m': 1, 'alpha': 0.05, 'n_significant_bh': 1,
          'rejected_bh': [True]}
    _print_report(stats, bh, 30.0, 10, 240)
    out = capsys.readouterr().out
    assert 'no value factor clears net-inference BH + positive raw-net' in out


def test_date_spreads_series_matches_factor_study_mean():
    # _date_spreads keeps the per-date spread series; its mean must equal the
    # gross spread factor_study collapses to (same book, same horizon).
    ev = _events()
    ev = ev.rename(columns={'fwd_21': 'fwd_63'})     # _date_spreads defaults h=63
    series = _date_spreads(ev, 'pb', h=63)
    assert len(series) == 24
    gross = factor_study(ev, 'pb', [63], cost_bps=0.0)[0]['gross_spread']
    assert abs(series.mean() - gross) < 1e-12
