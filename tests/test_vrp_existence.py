"""Hermetic tests for the VRP existence seam (2026-07-10; no network — the
conftest hard-blocks sockets).

Covers:
  * scripts/ingest_massive_options.py — straddle selection logic (nearest-30-
    DTE-in-window, nearest-ATM-strike, both tie rules), canned reference +
    aggregates JSON parsing, next_url pagination (verbatim, auth header
    only) + 429 retry, SIP session-close derivation, monthly selection
    dates, CLI defaults pinned.
  * data/pit_warehouse.py — option_bars_eod write/read round-trip on a tmp
    dir; REGISTRY BYTE-IDENTITY pins: _TABLES and _INTRADAY_TABLES exactly
    as before this seam, _OPTION_TABLES additive, Sharadar ingest still
    rejects the new table.
  * scripts/vrp_screen.py — straddle P&L math incl. the pre-registered
    costs (hand-computed), settlement fallback + unsettled-month drop, IV
    inversion round-trip vs desks.options_pricing, end-to-end screen on a
    tmp warehouse, --selftest.
"""

import datetime as dt

import pandas as pd
import pytest

from data.pit_warehouse import (_BARS_1M_COLUMNS, _INTRADAY_TABLES,
                                _OPTION_BARS_EOD_COLUMNS, _OPTION_TABLES,
                                _TABLES, PitWarehouse)
from desks.options_pricing import black_scholes_price
from scripts import ingest_massive_options as imo
from scripts import vrp_screen as vrp

# Reference-style contract dicts -------------------------------------------


def _contract(ctype, strike, expiry, underlying='SPY'):
    k = int(round(float(strike) * 1000))
    e = pd.Timestamp(expiry)
    occ = (f"O:{underlying}{e.strftime('%y%m%d')}"
           f"{'C' if ctype == 'call' else 'P'}{k:08d}")
    return {'ticker': occ, 'underlying_ticker': underlying,
            'contract_type': ctype, 'strike_price': float(strike),
            'expiration_date': e.date().isoformat()}


class _Resp:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code = status
        self._body = body or {}
        self.headers = headers or {}

    def raise_for_status(self):
        assert self.status_code == 200

    def json(self):
        return self._body


# ---------------------------------------------------------------------------
# Selection logic — nearest-30-DTE-in-window, nearest-ATM strike, ties
# ---------------------------------------------------------------------------
def test_pick_expiry_nearest_30_dte_in_window():
    sel = '2025-01-06'
    exps = [pd.Timestamp(sel) + pd.Timedelta(days=d) for d in (15, 25, 33, 50)]
    # 15 and 50 are outside [21, 45]; |33-30|=3 beats |25-30|=5.
    assert imo.pick_expiry(exps, sel) == dt.date(2025, 2, 8)


def test_pick_expiry_tie_prefers_earlier():
    sel = '2025-01-06'
    exps = [pd.Timestamp(sel) + pd.Timedelta(days=d) for d in (35, 25)]
    # |25-30| == |35-30| == 5 -> the EARLIER expiry (25 DTE) wins.
    assert imo.pick_expiry(exps, sel) == dt.date(2025, 1, 31)


def test_pick_expiry_none_in_window():
    sel = '2025-01-06'
    exps = [pd.Timestamp(sel) + pd.Timedelta(days=d) for d in (7, 60)]
    assert imo.pick_expiry(exps, sel) is None
    assert imo.pick_expiry([], sel) is None


def test_pick_strike_nearest_and_tie_lower():
    assert imo.pick_strike([95.0, 100.0, 105.0], 100.3) == 100.0
    # 102.5 is equidistant from 100 and 105 -> the LOWER strike.
    assert imo.pick_strike([100.0, 105.0], 102.5) == 100.0
    assert imo.pick_strike([], 100.0) is None


def test_select_straddle_requires_both_legs():
    """Strike 100 lists only a call; 99/101 list both. Spot 100.2 ->
    nearest both-leg strike is 101 (|101-100.2| = 0.8 < |99-100.2| = 1.2)."""
    exp = '2025-02-05'                                # 30 DTE from 2025-01-06
    far = '2025-02-14'                                # 39 DTE — loses to 30
    contracts = [
        _contract('call', 100, exp),                  # no matching put
        _contract('call', 99, exp), _contract('put', 99, exp),
        _contract('call', 101, exp), _contract('put', 101, exp),
        _contract('call', 100, far), _contract('put', 100, far),
    ]
    pick = imo.select_straddle(contracts, '2025-01-06', 100.2)
    assert pick is not None
    call, put = pick
    assert call['contract_type'] == 'call' and put['contract_type'] == 'put'
    assert call['strike_price'] == put['strike_price'] == 101.0
    assert call['expiration_date'] == put['expiration_date'] == exp
    # No strike with both legs -> None (never a naked substitute).
    assert imo.select_straddle([_contract('call', 100, exp)],
                               '2025-01-06', 100.2) is None
    assert imo.select_straddle([], '2025-01-06', 100.2) is None


# ---------------------------------------------------------------------------
# Canned reference JSON — pagination verbatim, auth-only headers, 429 retry
# ---------------------------------------------------------------------------
def test_fetch_contracts_params_pagination_and_429():
    exp = '2025-02-05'
    next_url = ('https://api.polygon.io/v3/reference/options/contracts'
                '?cursor=xyz789')
    pages = [
        _Resp(200, {'results': [_contract('call', 100, exp)],
                    'next_url': next_url}),
        _Resp(429, headers={'Retry-After': '7'}),      # retry same URL
        _Resp(200, {'results': [_contract('put', 100, exp)]}),
    ]
    calls, sleeps = [], []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append((url, params, dict(headers)))
        return pages[len(calls) - 1]

    auth = {'Authorization': 'Bearer TESTTOKEN'}
    items = imo.fetch_contracts('SPY', '2025-01-06', 600.0, get=fake_get,
                                headers=auth, sleep=sleeps.append)
    assert len(items) == 2 and sleeps == [7.0]
    url0, params0, _ = calls[0]
    assert url0 == imo.CONTRACTS_URL
    assert params0 == {
        'underlying_ticker': 'SPY', 'as_of': '2025-01-06',
        'expiration_date.gte': '2025-01-27',           # sel + 21d
        'expiration_date.lte': '2025-02-20',           # sel + 45d
        'strike_price.gte': 540.0, 'strike_price.lte': 660.0,  # ±10%
        'limit': 1000,
    }
    # 429 retried the SAME next_url; cursor URL followed VERBATIM, no params
    # re-sent — only the Authorization header rides along.
    assert calls[1][0] == next_url and calls[2][0] == next_url
    assert calls[1][1] is None and calls[2][1] is None
    assert all(h == auth for _, _, h in calls)


def test_massive_paged_gives_up_after_max_429s():
    resp = _Resp(429, headers={'Retry-After': '1'})
    with pytest.raises(RuntimeError, match='consecutive 429s'):
        imo.massive_paged('https://api.polygon.io/x', {}, label='X',
                          get=lambda *a, **k: resp, headers={},
                          sleep=lambda _: None)


# ---------------------------------------------------------------------------
# Canned aggregates JSON -> option_bars_eod frame
# ---------------------------------------------------------------------------
WINTER_T = 1736173800000   # 2025-01-06 14:30:00 UTC == 09:30 EST
SUMMER_T = 1751895000000   # 2025-07-07 13:30:00 UTC == 09:30 EDT


def _abar(t, o=2.0, h=2.2, low=1.9, c=2.1, v=350):
    return {'t': t, 'o': o, 'h': h, 'l': low, 'c': c, 'v': v}


def test_option_bars_frame_normalizes_to_session_date():
    df = imo.option_bars_frame(
        [_abar(WINTER_T), _abar(SUMMER_T)], underlying='SPY',
        contract='O:SPY250205C00600000', ctype='call', strike=600.0,
        expiry='2025-02-05', selection_date='2025-01-06')
    assert list(df.columns) == list(_OPTION_BARS_EOD_COLUMNS)
    assert df['ts'].dt.tz is None
    assert df['ts'].iloc[0] == pd.Timestamp('2025-01-06')   # EST, normalized
    assert df['ts'].iloc[1] == pd.Timestamp('2025-07-07')   # EDT, normalized
    assert df['expiry'].iloc[0] == pd.Timestamp('2025-02-05')
    assert df['selection_date'].iloc[0] == pd.Timestamp('2025-01-06')
    assert df['close'].iloc[0] == pytest.approx(2.1)
    assert df['volume'].iloc[0] == 350
    assert imo.option_bars_frame([], underlying='SPY', contract='X',
                                 ctype='put', strike=1.0, expiry='2025-02-05',
                                 selection_date='2025-01-06').empty


# ---------------------------------------------------------------------------
# SIP session closes + monthly selection dates
# ---------------------------------------------------------------------------
def _minute_df(rows):
    return pd.DataFrame(
        [{'ticker': 'SPY', 'ts': pd.Timestamp(ts), 'open': c, 'high': c,
          'low': c, 'close': c, 'volume': 100, 'trade_count': 1, 'vwap': c}
         for ts, c in rows], columns=list(_BARS_1M_COLUMNS))


def test_sip_session_closes_last_rth_bar(tmp_path):
    wh = PitWarehouse(str(tmp_path))
    rows = [
        ('2025-01-06 09:30', 599.0), ('2025-01-06 15:59', 601.5),
        ('2025-01-06 16:00', 990.0), ('2025-01-06 17:00', 991.0),  # ext hrs
        ('2025-01-07 09:30', 602.0), ('2025-01-07 15:30', 603.25),  # early cut
        ('2025-01-07 04:30', 500.0),                              # pre-market
    ]
    assert wh.write_bars_1m('SPY', _minute_df(rows), table='bars_1m_sip') == 7
    closes = imo.sip_session_closes(wh, 'SPY', '2025-01-01', '2025-01-31')
    assert closes.index.tolist() == [pd.Timestamp('2025-01-06'),
                                     pd.Timestamp('2025-01-07')]
    assert closes.iloc[0] == 601.5          # 15:59 bar close, not the 16:00+
    assert closes.iloc[1] == 603.25         # last RTH bar when 15:59 missing
    assert imo.sip_session_closes(wh, 'QQQ', '2025-01-01', '2025-01-31').empty


def test_monthly_selection_dates_first_session():
    sessions = pd.DatetimeIndex([
        '2024-07-31', '2024-08-02', '2024-08-05', '2024-09-03', '2024-09-04',
        '2024-11-01',                                   # October missing
    ])
    got = imo.monthly_selection_dates(sessions, '2024-08', '2024-11')
    assert got == [pd.Timestamp('2024-08-02'), pd.Timestamp('2024-09-03'),
                   pd.Timestamp('2024-11-01')]
    assert imo.monthly_selection_dates(sessions, '2025-01', '2025-02') == []


# ---------------------------------------------------------------------------
# Registry byte-identity pins + warehouse round-trip
# ---------------------------------------------------------------------------
def test_existing_registries_byte_identical_and_option_table_additive():
    assert _TABLES == {
        'tickers': ('SHARADAR/TICKERS', {'table': 'SF1'}),
        'sep': ('SHARADAR/SEP', {}),
        'sf1': ('SHARADAR/SF1', {'dimension': 'ARQ'}),
        'sf2': ('SHARADAR/SF2', {}),
        'sf3': ('SHARADAR/SF3', {}),
        'daily': ('SHARADAR/DAILY', {}),
        'actions': ('SHARADAR/ACTIONS', {}),
        'events': ('SHARADAR/EVENTS', {}),
    }
    assert _INTRADAY_TABLES == {
        'bars_1m': ('ALPACA/v2/stocks/{symbol}/bars',
                    {'timeframe': '1Min', 'feed': 'iex'}),
        'bars_1m_sip': ('MASSIVE/v2/aggs/ticker/{symbol}/range/1/minute',
                        {'adjusted': 'false', 'limit': 50000}),
    }
    assert _OPTION_TABLES == {
        'option_bars_eod': ('MASSIVE/v3/reference/options/contracts '
                            '+ v2/aggs/ticker/{contract}/range/1/day',
                            {'adjusted': 'false', 'limit': 50000}),
    }
    # The Sharadar bulk ingest keeps rejecting the new table.
    with pytest.raises(ValueError, match='unknown table'):
        PitWarehouse('/nonexistent').ingest_table('option_bars_eod')


def test_option_bars_eod_roundtrip(tmp_path):
    wh = PitWarehouse(str(tmp_path))
    df = imo.option_bars_frame(
        [_abar(WINTER_T, c=2.1), _abar(WINTER_T + 86_400_000, c=1.8)],
        underlying='SPY', contract='O:SPY250205C00600000', ctype='call',
        strike=600.0, expiry='2025-02-05', selection_date='2025-01-06')
    assert wh.write_option_bars_eod('SPY', df) == 2
    assert (tmp_path / 'option_bars_eod' / 'SPY.parquet').exists()
    back = wh.option_bars_eod('SPY')
    assert list(back.columns) == list(_OPTION_BARS_EOD_COLUMNS)
    assert len(back) == 2 and back['close'].tolist() == [2.1, 1.8]
    assert back['expiry'].iloc[0] == pd.Timestamp('2025-02-05')
    # Missing underlying -> empty frame with the contract columns; empty
    # write -> nothing written; missing columns -> ValueError.
    assert wh.option_bars_eod('QQQ').empty
    assert wh.write_option_bars_eod(
        'QQQ', pd.DataFrame(columns=list(_OPTION_BARS_EOD_COLUMNS))) == 0
    assert not (tmp_path / 'option_bars_eod' / 'QQQ.parquet').exists()
    with pytest.raises(ValueError, match='missing columns'):
        wh.write_option_bars_eod('QQQ', pd.DataFrame({'close': [1.0]}))


def test_ingest_cli_defaults_pinned():
    args = imo.build_parser().parse_args([])
    assert args.underlyings == ['SPY', 'QQQ', 'IWM']
    assert (args.start_month, args.end_month) == ('2024-08', '2026-06')
    assert (args.bars_start, args.bars_end) == ('2024-07-01', '2026-07-31')
    assert args.force is False and args.pace == 13.0
    assert (imo.MIN_DTE, imo.MAX_DTE, imo.TARGET_DTE) == (21, 45, 30)


# ---------------------------------------------------------------------------
# Straddle P&L math — hand-computed, incl. the pre-registered costs
# ---------------------------------------------------------------------------
def test_straddle_pnl_otm_win_hand_computed():
    """c=3.00 p=2.80 K=100 S_T=100: premium 5.80, haircut .15+.14=.29,
    commission 2*.65/100=.013, intrinsic 0 -> +$549.70 per straddle."""
    r = vrp.straddle_pnl(3.00, 2.80, 100.0, 100.0)
    assert r['premium_gross'] == pytest.approx(5.80)
    assert r['haircut'] == pytest.approx(0.29)
    assert r['commission'] == pytest.approx(0.013)
    assert r['intrinsic'] == 0.0
    assert r['pnl_dollars'] == pytest.approx(549.70)


def test_straddle_pnl_itm_blowout_hand_computed():
    """c=p=2.50 K=100 S_T=130: premium 5.00, haircut .25, intrinsic 30 ->
    5.00-.25-30-.013 = -25.263/share = -$2526.30."""
    r = vrp.straddle_pnl(2.50, 2.50, 100.0, 130.0)
    assert r['intrinsic'] == pytest.approx(30.0)
    assert r['pnl_dollars'] == pytest.approx(-2526.30)


def test_straddle_pnl_haircut_floor_on_cheap_legs():
    """A $0.40 leg: 5% = $0.02 < the $0.05 floor -> floor applies per leg."""
    r = vrp.straddle_pnl(0.40, 0.40, 100.0, 100.0)
    assert r['haircut'] == pytest.approx(0.10)
    assert r['pnl_dollars'] == pytest.approx((0.80 - 0.10 - 0.013) * 100)


# ---------------------------------------------------------------------------
# IV inversion round-trip vs desks.options_pricing
# ---------------------------------------------------------------------------
def test_implied_vol_roundtrip_and_refusals():
    for right, vol in (('call', 0.234), ('put', 0.187), ('call', 0.85)):
        px = black_scholes_price(605.0, 600.0, 32 / 365.0, vol, 0.0425, right)
        iv = vrp.implied_vol(px, 605.0, 600.0, 32 / 365.0, 0.0425, right)
        assert iv == pytest.approx(vol, abs=1e-6)
    # Below-intrinsic print, expired, and junk inputs -> None, never a vol.
    assert vrp.implied_vol(0.50, 100.0, 90.0, 30 / 365.0, 0.045, 'call') is None
    assert vrp.implied_vol(5.0, 100.0, 100.0, 0.0, 0.045, 'call') is None
    assert vrp.implied_vol(-1.0, 100.0, 100.0, 0.1, 0.045, 'put') is None


def test_selftest_runs():
    vrp._selftest()


# ---------------------------------------------------------------------------
# Screen end-to-end on a tmp warehouse — settlement fallback + unsettled drop
# ---------------------------------------------------------------------------
def _seed_warehouse(tmp_path):
    """Two straddle months on SPY + a third unsettled one.

    Month 1 (sel 2025-01-06, K=600, expiry 2025-02-05 = a session): OTM-ish.
    Month 2 (sel 2025-02-03, K=610, expiry 2025-03-08 = a SATURDAY, not a
    session): settles at the last prior session close (fallback).
    Month 3 (sel 2025-03-03, expiry 2025-04-04 AFTER the last session):
    unsettled -> dropped.
    """
    wh = PitWarehouse(str(tmp_path))
    sessions = pd.bdate_range('2024-11-01', '2025-03-07')
    minute_rows = [(s + pd.Timedelta(hours=15, minutes=59),
                    600.0 + 0.05 * i) for i, s in enumerate(sessions)]
    assert wh.write_bars_1m('SPY', _minute_df(minute_rows),
                            table='bars_1m_sip') == len(sessions)
    closes = imo.sip_session_closes(wh, 'SPY', '2024-11-01', '2025-03-31')

    def bars(contract, ctype, strike, expiry, sel, entry_close):
        ms = int(pd.Timestamp(sel).tz_localize('America/New_York')
                 .tz_convert('UTC').timestamp() * 1000)
        return imo.option_bars_frame(
            [_abar(ms, c=entry_close)], underlying='SPY', contract=contract,
            ctype=ctype, strike=strike, expiry=expiry, selection_date=sel)

    frames = [
        bars('O:SPY250205C00600000', 'call', 600.0, '2025-02-05',
             '2025-01-06 09:30', 9.00),
        bars('O:SPY250205P00600000', 'put', 600.0, '2025-02-05',
             '2025-01-06 09:30', 8.00),
        bars('O:SPY250308C00610000', 'call', 610.0, '2025-03-08',
             '2025-02-03 09:30', 9.50),
        bars('O:SPY250308P00610000', 'put', 610.0, '2025-03-08',
             '2025-02-03 09:30', 8.50),
        bars('O:SPY250404C00615000', 'call', 615.0, '2025-04-04',
             '2025-03-03 09:30', 9.00),
        bars('O:SPY250404P00615000', 'put', 615.0, '2025-04-04',
             '2025-03-03 09:30', 8.00),
    ]
    # option_bars_frame normalizes ts to the session date; selection_date
    # must match that convention (the ingest passes the session Timestamp).
    df = pd.concat(frames, ignore_index=True)
    df['selection_date'] = df['selection_date'].dt.normalize()
    assert wh.write_option_bars_eod('SPY', df) == 6
    return wh, closes


def test_screen_settlement_fallback_and_unsettled_drop(tmp_path):
    wh, closes = _seed_warehouse(tmp_path)
    res = vrp.screen_underlying(wh, 'SPY', closes=closes)
    assert res['skipped'] == {'no_entry_close': 0, 'no_underlying_close': 0,
                              'unsettled': 1}
    assert len(res['months']) == 2
    m1, m2 = res['months']
    # Month 1: expiry IS a session -> no fallback; P&L matches straddle_pnl
    # against that session's SIP close.
    assert not m1['settle_fallback']
    settle1 = float(closes.loc[pd.Timestamp('2025-02-05')])
    exp1 = vrp.straddle_pnl(9.00, 8.00, 600.0, settle1)['pnl_dollars']
    assert m1['pnl'] == pytest.approx(exp1)
    assert m1['premium_pct'] == pytest.approx(
        17.0 / float(closes.loc[pd.Timestamp('2025-01-06')]))
    # Month 2: Saturday expiry -> settles at the LAST PRIOR session close.
    assert m2['settle_fallback']
    assert m2['settle_date'] == pd.Timestamp('2025-03-07')
    settle2 = float(closes.loc[pd.Timestamp('2025-03-07')])
    exp2 = vrp.straddle_pnl(9.50, 8.50, 610.0, settle2)['pnl_dollars']
    assert m2['pnl'] == pytest.approx(exp2)
    # The direct VRP fields exist: entry IV inverted, RV measured.
    assert m1['entry_iv'] is not None and m1['rv'] is not None
    assert m1['iv_minus_rv'] == pytest.approx(m1['entry_iv'] - m1['rv'])


def test_screen_stats_and_pooled_report(tmp_path, capsys):
    wh, closes = _seed_warehouse(tmp_path)
    res = vrp.screen_underlying(wh, 'SPY', closes=closes)
    s = vrp._stats(res['months'])
    assert s['n'] == 2 and 0.0 <= s['win_rate'] <= 1.0
    assert s['worst_pnl'] == min(m['pnl'] for m in res['months'])
    assert s['n_fallback_settles'] == 1
    assert s['nw_t'] is None                       # n=2 < 3 -> no HAC fit
    pooled = vrp.pooled_monthly([res])
    assert len(pooled) == 2                        # one row per month
    assert pooled[0]['pnl'] == pytest.approx(res['months'][0]['pnl'])
    vrp.print_report([res])
    out = capsys.readouterr().out
    assert 'HONESTY' in out and 'NOT' not in out.split('HONESTY')[0][:1]
    assert 'EXISTENCE measurement, not a' in out
    assert 'NO promotion pathway' in out
    assert 'POOLED' in out


def test_nw_tstat_matches_factor_screen_construction():
    import numpy as np
    import statsmodels.api as sm
    rng = np.random.default_rng(7)
    x = rng.normal(0.5, 1.0, 24)
    t, p = vrp.nw_tstat(x)
    fit = sm.OLS(x, np.ones(24)).fit(cov_type='HAC',
                                     cov_kwds={'maxlags': vrp.NW_LAG})
    assert t == pytest.approx(float(fit.tvalues[0]))
    assert p == pytest.approx(float(fit.pvalues[0]))
    assert vrp.nw_tstat([1.0, 2.0]) == (None, None)
    assert vrp.NW_LAG == 2                         # pre-registered
