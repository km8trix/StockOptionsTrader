"""Hermetic tests for the Alpaca ORB pilot (ingest + warehouse seam + gate).

Three surfaces, no network anywhere (conftest hard-blocks sockets anyway):
  * scripts/orb_gate.py     — session-simulator math on synthetic minute bars,
                              every pre-registered rule pinned by hand-computed
                              fills/PnL.
  * scripts/ingest_alpaca_bars.py — pagination + JSON parse against a canned
                              Alpaca payload via an injected fake transport.
  * data/pit_warehouse.py   — bars_1m round-trip on a tmp dir, and byte-identity
                              of the EXISTING Sharadar table registry.
"""

import pandas as pd
import pytest

from data.pit_warehouse import (_BARS_1M_COLUMNS, _INTRADAY_TABLES, _TABLES,
                                PitWarehouse)
from scripts.ingest_alpaca_bars import (MIN_RETRY_SLEEP, RETRY_429_DEFAULT,
                                        _retry_after_seconds, bars_to_df,
                                        fetch_symbol_bars)
from scripts.orb_gate import (COMMISSION, HALF_DAY_MIN_PROBE_BARS,
                              INITIAL_CAPITAL, MAX_LEVERAGE, OR_REQUIRED_BARS,
                              RISK_FRAC, SLIPPAGE, equal_weight, make_session,
                              run_symbol, simulate_session, trade_stats)

EQ = 100_000.0


def _or_block():
    """09:30-09:34 bars pinning OR-high=101.0 / OR-low=100.0 (range 1.0)."""
    return [('09:30', 100.5, 101.0, 100.0, 100.5),
            ('09:31', 100.5, 100.9, 100.1, 100.5),
            ('09:32', 100.5, 100.9, 100.1, 100.5),
            ('09:33', 100.5, 100.9, 100.1, 100.5),
            ('09:34', 100.5, 100.9, 100.1, 100.5)]


def _probe_block(n=30, px=100.5):
    """``n`` bars in [13:15, 15:45) at ``px`` (in the _or_block range, so
    they never break out or hit a stop). 30+ marks the synthetic session as
    a REGULAR day under the amended data-driven session-end rule."""
    return [(f'13:{m}', px, px, px, px) for m in range(15, 15 + n)]


def _day(extra, date='2021-01-04'):
    """Regular-day session: OR block + extras + 30 probe bars (sorted)."""
    return make_session(date, _or_block() + extra + _probe_block()).sort_index()


# ---------------------------------------------------------------------------
# Simulator math — every rule pinned
# ---------------------------------------------------------------------------
def test_breakout_long_entry_is_bar_close_plus_slippage_eod_flat():
    """Entry = first trade THROUGH the level, filled at the breakout bar's
    CLOSE + slippage (never the level itself); EOD exit at last bar close."""
    day = _day([('09:35', 100.5, 100.8, 100.2, 100.6),      # no break
                ('09:36', 100.7, 101.5, 100.6, 101.2),      # THROUGH 101.0
                ('09:37', 101.2, 101.6, 100.8, 101.4),
                ('15:59', 101.8, 102.1, 101.7, 102.0)])
    tr = simulate_session(day, EQ)
    assert tr.direction == 1 and not tr.stopped
    assert tr.entry_ts == pd.Timestamp('2021-01-04 09:36')
    assert tr.entry_price == pytest.approx(101.2 + SLIPPAGE)   # close, not 101.0
    assert tr.exit_price == pytest.approx(102.0 - SLIPPAGE)    # EOD, adverse
    # shares = 1% * 100k / OR-range 1.0 = 1000; pnl = 1000*0.78 - 2*.0035*1000
    assert tr.shares == pytest.approx(1000.0)
    assert tr.pnl == pytest.approx(773.0)


def test_stop_out_intrabar_low_touches_stop():
    day = _day([('09:36', 100.7, 101.5, 100.6, 101.2),      # long @ 101.21
                ('09:40', 100.5, 100.6, 99.9, 100.0),       # low<=100 -> stop
                ('15:59', 100.2, 100.3, 100.1, 100.2)])
    tr = simulate_session(day, EQ)
    assert tr.stopped and tr.exit_ts == pd.Timestamp('2021-01-04 09:40')
    assert tr.exit_price == pytest.approx(100.0 - SLIPPAGE)   # stop level fill
    assert tr.pnl == pytest.approx(1000 * (99.99 - 101.21) - 7.0)  # -1227.0


def test_stop_gap_through_fills_at_open_not_stop():
    day = _day([('09:36', 100.7, 101.5, 100.6, 101.2),
                ('09:40', 99.5, 99.6, 99.0, 99.1),          # opens below stop
                ('15:59', 100.2, 100.3, 100.1, 100.2)])
    tr = simulate_session(day, EQ)
    assert tr.stopped
    assert tr.exit_price == pytest.approx(99.5 - SLIPPAGE)   # open fill: 99.49
    assert tr.pnl == pytest.approx(1000 * (99.49 - 101.21) - 7.0)  # -1727.0


def test_no_trade_day_touch_is_not_through():
    """high == OR-high / low == OR-low is a touch, not a trade THROUGH."""
    day = _day([('09:36', 100.5, 101.0, 100.0, 100.5),
                ('15:59', 100.5, 100.9, 100.1, 100.5)])
    assert simulate_session(day, EQ) is None


def test_short_breakout_mirrored():
    day = _day([('09:36', 100.2, 100.4, 99.5, 99.7),        # THROUGH OR-low
                ('10:00', 99.6, 99.8, 99.2, 99.4),
                ('15:59', 99.1, 99.2, 98.9, 99.0)])
    tr = simulate_session(day, EQ)
    assert tr.direction == -1 and not tr.stopped
    assert tr.entry_price == pytest.approx(99.7 - SLIPPAGE)  # adverse for short
    assert tr.exit_price == pytest.approx(99.0 + SLIPPAGE)
    assert tr.pnl == pytest.approx(1000 * (99.69 - 99.01) - 7.0)  # +673.0


def test_short_stop_at_or_high():
    day = _day([('09:36', 100.2, 100.4, 99.5, 99.7),        # short @ 99.69
                ('10:00', 100.5, 101.2, 100.4, 101.1),      # high>=101 -> stop
                ('15:59', 101.0, 101.1, 100.9, 101.0)])
    tr = simulate_session(day, EQ)
    assert tr.stopped
    assert tr.exit_price == pytest.approx(101.0 + SLIPPAGE)
    assert tr.pnl == pytest.approx(1000 * (99.69 - 101.01) - 7.0)


def test_both_sides_in_first_breaking_bar_skips_day():
    day = _day([('09:36', 100.5, 101.5, 99.5, 100.5),       # breaches BOTH
                ('15:59', 100.5, 100.9, 100.1, 100.5)])
    assert simulate_session(day, EQ) is None


def test_first_breakout_wins_one_trade_per_day():
    """Down-break first: the later up-break must not flip or re-enter."""
    day = _day([('09:36', 100.2, 100.4, 99.5, 99.7),        # short first
                ('10:30', 100.8, 101.6, 100.7, 101.5),      # up-break AND stop
                ('15:59', 102.0, 102.2, 101.9, 102.1)])
    tr = simulate_session(day, EQ)
    assert tr.direction == -1 and tr.stopped                 # stopped, not long


def test_no_entries_at_or_after_1555():
    day = _day([('15:55', 100.7, 101.5, 100.6, 101.2),      # too late
                ('15:59', 101.8, 102.1, 101.7, 102.0)])
    assert simulate_session(day, EQ) is None


def test_sizing_risk_fraction_and_leverage_cap():
    # Tight OR: 101.0/100.9 -> range 0.1 -> risk shares 10_000; the 4x
    # notional cap at the 101.21 entry fill binds first.
    tight = [('09:30', 100.95, 101.0, 100.9, 100.95),
             ('09:31', 100.95, 101.0, 100.9, 100.95),
             ('09:32', 100.95, 101.0, 100.9, 100.95),
             ('09:33', 100.95, 101.0, 100.9, 100.95),
             ('09:34', 100.95, 101.0, 100.9, 100.95),
             ('09:36', 101.0, 101.3, 100.95, 101.2)]
    tight += _probe_block(px=100.95)             # regular day, no stop touch
    tight += [('15:59', 101.4, 101.5, 101.3, 101.4)]
    tr = simulate_session(make_session('2021-01-04', tight), EQ)
    assert RISK_FRAC * EQ / 0.1 == pytest.approx(10_000.0)   # un-capped size
    assert tr.shares == pytest.approx(MAX_LEVERAGE * EQ / 101.21)


def test_costs_commission_both_sides():
    day = _day([('09:36', 100.7, 101.5, 100.6, 101.2),
                ('15:59', 101.2, 101.3, 101.15, 101.21)])
    tr = simulate_session(day, EQ)
    # Fills net out exactly to -2*slippage; the residue is pure commission.
    gross = tr.shares * (tr.exit_price - tr.entry_price)
    assert tr.pnl == pytest.approx(gross - 2 * COMMISSION * tr.shares)


def test_long_only_cut_skips_shorts():
    day = _day([('09:36', 100.2, 100.4, 99.5, 99.7),        # short signal
                ('15:59', 99.1, 99.2, 98.9, 99.0)])
    assert simulate_session(day, EQ).direction == -1         # combined: short
    assert simulate_session(day, EQ, long_only=True) is None  # cut: flat


def test_zero_or_range_or_empty_session_no_trade():
    flat = [(f'09:3{i}', 100.0, 100.0, 100.0, 100.0) for i in range(5)]
    flat += [('09:36', 100.0, 100.5, 99.5, 100.2), ('15:59', 100, 100, 100, 100)]
    assert simulate_session(make_session('2021-01-04', flat), EQ) is None
    empty = make_session('2021-01-04', [('08:15', 100, 101, 99, 100)])  # pre-mkt
    assert simulate_session(empty, EQ) is None


def test_run_symbol_compounds_and_zero_fills_no_trade_days():
    d1 = _day([('09:36', 100.7, 101.5, 100.6, 101.2),
               ('15:59', 101.8, 102.1, 101.7, 102.0)], date='2021-01-04')
    d2 = _day([('09:36', 100.5, 101.0, 100.0, 100.5),       # no-trade day
               ('15:59', 100.5, 100.9, 100.1, 100.5)], date='2021-01-05')
    d3 = _day([('09:36', 100.7, 101.5, 100.6, 101.2),
               ('09:40', 100.5, 100.6, 99.9, 100.0),        # stop-out
               ('15:59', 100.2, 100.3, 100.1, 100.2)], date='2021-01-06')
    bars = pd.concat([d1, d2, d3])
    dates, rets, trades, equity = run_symbol(bars, initial_capital=EQ)
    assert [d.isoformat() for d in dates] == ['2021-01-04', '2021-01-05',
                                              '2021-01-06']
    assert rets[0] == pytest.approx(773.0 / EQ)
    assert rets[1] == 0.0
    # Day 3 sizes off the COMPOUNDED equity 100_773.
    eq3 = EQ + 773.0
    shares3 = RISK_FRAC * eq3 / 1.0
    pnl3 = shares3 * (99.99 - 101.21) - 2 * COMMISSION * shares3
    assert rets[2] == pytest.approx(pnl3 / eq3)
    assert len(trades) == 2 and equity == pytest.approx(eq3 + pnl3)


def test_equal_weight_means_over_symbols_with_data():
    from datetime import date
    d1, d2, d3 = date(2021, 1, 4), date(2021, 1, 5), date(2021, 1, 6)
    days, rets = equal_weight({'A': ([d1, d2], [0.01, 0.02]),
                               'B': ([d2, d3], [0.04, 0.06])})
    assert days == [d1, d2, d3]
    assert rets == pytest.approx([0.01, 0.03, 0.06])


def test_trade_stats_cuts():
    class T:
        def __init__(self, pnl):
            self.pnl = pnl
    s = trade_stats([T(100.0), T(-50.0), T(300.0)], n_days=504)
    assert s['n_trades'] == 3 and s['trades_per_year'] == pytest.approx(1.5)
    assert s['win_rate'] == pytest.approx(2 / 3)
    assert s['avg_win'] == pytest.approx(200.0)
    assert s['avg_loss'] == pytest.approx(-50.0)
    assert INITIAL_CAPITAL == 100_000.0                      # pre-registered


# ---------------------------------------------------------------------------
# Amended spec 2026-07-10 — data-driven early close + full-OR requirement
# (pre-registered amendments: no ORB backtest had run when these were added)
# ---------------------------------------------------------------------------
def _half_day(rth_extra, ah, date='2021-11-26'):
    """OR block + pre-13:00 extras + sparse after-hours prints. NO probe
    filler, so the data-driven rule reads the session as a 13:00 early
    close (fewer than HALF_DAY_MIN_PROBE_BARS bars in [13:15, 15:45))."""
    return make_session(date, _or_block() + rth_extra + ah)


def test_half_day_phantom_ah_breakout_suppressed():
    """No OR break before the 13:00 close; a sparse AH print trades THROUGH
    OR-high. Old [09:30,16:00) filter would have entered on it -> phantom
    trade. Amended rule: no trade at all."""
    day = _half_day(
        [('10:00', 100.5, 100.9, 100.2, 100.6),
         ('12:59', 100.5, 100.8, 100.3, 100.6)],       # never breaks by close
        [('13:30', 101.2, 101.6, 101.1, 101.5),        # AH print THROUGH high
         ('15:50', 101.4, 101.5, 101.3, 101.4)])
    assert simulate_session(day, EQ) is None


def test_half_day_real_breakout_exits_at_last_bar_before_1300():
    """Real 09:40 breakout on a half day: EOD flat exit fills at the LAST
    bar strictly before 13:00 — never an AH bar (the 13:30 AH print goes
    through the stop and must not fill anything)."""
    day = _half_day(
        [('09:40', 100.7, 101.5, 100.6, 101.2),        # breakout -> long
         ('12:59', 101.5, 101.9, 101.4, 101.8)],       # last pre-close bar
        [('13:30', 99.0, 99.1, 98.5, 98.6),            # AH: through the stop
         ('14:10', 98.6, 98.7, 98.4, 98.5)])
    tr = simulate_session(day, EQ)
    assert tr is not None and not tr.stopped
    assert tr.exit_ts == pd.Timestamp('2021-11-26 12:59')
    assert tr.exit_price == pytest.approx(101.8 - SLIPPAGE)
    assert tr.pnl == pytest.approx(1000 * (101.79 - 101.21) - 7.0)  # +573.0


def test_half_day_entries_stop_six_minutes_before_1300():
    """Mirror of the 15:54 rule: last entry bar on a half day is 12:54."""
    day = _half_day(
        [('12:55', 100.7, 101.5, 100.6, 101.2),        # too late on half day
         ('12:59', 101.5, 101.9, 101.4, 101.8)],
        [('13:30', 102.0, 102.2, 101.9, 102.1)])
    assert simulate_session(day, EQ) is None


def test_regular_day_probe_threshold_boundary():
    """Exactly HALF_DAY_MIN_PROBE_BARS (30) probe bars => regular day, EOD
    exit at 15:59 with the same pinned +773 PnL as
    test_breakout_long_entry_is_bar_close_plus_slippage_eod_flat (which now
    carries 30 probe bars via _day and still passes — the existing pin);
    one bar fewer => early close, EOD exit at the last pre-13:00 bar."""
    assert HALF_DAY_MIN_PROBE_BARS == 30                     # pre-registered
    core = [('09:36', 100.7, 101.5, 100.6, 101.2),           # breakout
            ('12:59', 101.6, 101.9, 101.5, 101.8)]
    tail = [('15:59', 101.8, 102.1, 101.7, 102.0)]
    reg = make_session('2021-01-04',
                       _or_block() + core + _probe_block(30) + tail)
    tr = simulate_session(reg, EQ)
    assert tr.exit_ts == pd.Timestamp('2021-01-04 15:59')
    assert tr.pnl == pytest.approx(773.0)
    half = make_session('2021-01-04',
                        _or_block() + core + _probe_block(29) + tail)
    tr = simulate_session(half, EQ)
    assert tr.exit_ts == pd.Timestamp('2021-01-04 12:59')
    assert tr.exit_price == pytest.approx(101.8 - SLIPPAGE)


def test_partial_opening_range_no_trade():
    """Only 4 of the five 09:30-09:34 bars -> no trade that session, even
    with a clean breakout after (OR_REQUIRED_BARS data-quality guard)."""
    assert OR_REQUIRED_BARS == 5                             # pre-registered
    bars = _or_block()[:4] + [('09:36', 100.7, 101.5, 100.6, 101.2)]
    bars += _probe_block() + [('15:59', 101.8, 102.1, 101.7, 102.0)]
    assert simulate_session(make_session('2021-01-04', bars), EQ) is None


# ---------------------------------------------------------------------------
# Ingest — canned Alpaca JSON, injected transport (no network)
# ---------------------------------------------------------------------------
class _Resp:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code = status
        self._body = body or {}
        self.headers = headers or {}

    def raise_for_status(self):
        assert self.status_code == 200

    def json(self):
        return self._body


def _bar(t, o=100.0, h=101.0, low=99.0, c=100.5, v=1200, n=17, vw=100.25):
    return {'t': t, 'o': o, 'h': h, 'l': low, 'c': c, 'v': v, 'n': n, 'vw': vw}


def test_bars_to_df_converts_utc_to_naive_eastern():
    # 14:30 UTC on a January day == 09:30 EST; July DST: 13:30 UTC == 09:30 EDT
    df = bars_to_df([_bar('2021-01-04T14:30:00Z'),
                     _bar('2021-07-06T13:30:00Z')], 'SPY')
    assert list(df.columns) == list(_BARS_1M_COLUMNS)
    assert df['ts'].dt.tz is None                            # tz-NAIVE ET
    assert df['ts'].iloc[0] == pd.Timestamp('2021-01-04 09:30:00')
    assert df['ts'].iloc[1] == pd.Timestamp('2021-07-06 09:30:00')
    assert (df['ticker'] == 'SPY').all()
    assert df['volume'].iloc[0] == 1200 and df['trade_count'].iloc[0] == 17
    assert df['vwap'].iloc[0] == pytest.approx(100.25)
    assert bars_to_df([], 'SPY').empty


def test_fetch_pagination_429_and_token_passing():
    pages = [
        _Resp(200, {'bars': [_bar('2021-01-04T14:30:00Z'),
                             _bar('2021-01-04T14:31:00Z')],
                    'next_page_token': 'tok1'}),
        _Resp(429, headers={'Retry-After': '3'}),            # retried, same page
        _Resp(200, {'bars': [_bar('2021-01-04T14:32:00Z')],
                    'next_page_token': None}),
    ]
    calls, sleeps = [], []
    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append((url, dict(params)))
        return pages[len(calls) - 1]
    df = fetch_symbol_bars('SPY', '2021-01-04', '2021-01-04',
                           get=fake_get, headers={'APCA-API-KEY-ID': 'x'},
                           sleep=sleeps.append)
    assert len(df) == 3 and sleeps == [3.0]
    assert calls[0][0].endswith('/v2/stocks/SPY/bars')
    assert 'page_token' not in calls[0][1]                   # first page
    assert calls[0][1]['timeframe'] == '1Min'
    assert calls[0][1]['feed'] == 'iex'
    assert calls[0][1]['limit'] == 10_000
    # 429 retry re-requests the SAME page token; page 2 carries tok1.
    assert calls[1][1] == calls[2][1]
    assert calls[2][1]['page_token'] == 'tok1'
    assert df['ts'].iloc[-1] == pd.Timestamp('2021-01-04 09:32:00')


def test_fetch_dedups_overlapping_pages():
    dup = _bar('2021-01-04T14:30:00Z')
    df = bars_to_df([dup, dup, _bar('2021-01-04T14:31:00Z')], 'SPY')
    assert len(df) == 2 and df['ts'].is_monotonic_increasing


def test_retry_after_parse_falls_back_and_floors():
    """Amended 2026-07-10: the HTTP-date form (RFC 9110 allows it) and any
    junk fall back to the default; 0/sub-second values are floored so a
    'Retry-After: 0' server cannot induce a hot retry loop."""
    assert _retry_after_seconds('3') == 3.0                  # numeric form
    assert _retry_after_seconds('Wed, 21 Oct 2026 07:28:00 GMT') \
        == RETRY_429_DEFAULT
    assert _retry_after_seconds('0') == MIN_RETRY_SLEEP      # no hot-loop
    assert _retry_after_seconds('0.25') == MIN_RETRY_SLEEP   # floored
    assert _retry_after_seconds(None) == RETRY_429_DEFAULT   # header absent
    assert _retry_after_seconds('') == RETRY_429_DEFAULT
    assert MIN_RETRY_SLEEP == 1.0 and RETRY_429_DEFAULT == 10.0


# ---------------------------------------------------------------------------
# Warehouse — bars_1m round-trip on tmp dir; existing registry byte-identical
# ---------------------------------------------------------------------------
def _sample_frame():
    return bars_to_df([_bar('2021-01-04T14:30:00Z', c=100.5),
                       _bar('2021-01-04T14:31:00Z', c=100.7),
                       _bar('2021-01-05T14:30:00Z', c=101.5)], 'SPY')


def test_bars_roundtrip_single_day_and_range(tmp_path):
    wh = PitWarehouse(str(tmp_path))
    assert wh.write_bars_1m('SPY', _sample_frame()) == 3
    day = wh.ohlcv_intraday('SPY', '2021-01-04')
    assert list(day.index) == [pd.Timestamp('2021-01-04 09:30'),
                               pd.Timestamp('2021-01-04 09:31')]
    assert list(day.columns) == ['open', 'high', 'low', 'close', 'volume',
                                 'trade_count', 'vwap']
    assert day['close'].tolist() == [100.5, 100.7]
    assert day.index.tz is None                              # tz-naive ET
    rng = wh.ohlcv_intraday_range('SPY', '2021-01-04', '2021-01-05')
    assert len(rng) == 3 and rng.index.is_monotonic_increasing
    # Other days / other symbols / bad dates are empty, same columns.
    assert wh.ohlcv_intraday('SPY', '2021-01-06').empty
    assert wh.ohlcv_intraday('QQQ', '2021-01-04').empty
    assert wh.ohlcv_intraday('SPY', 'not-a-date').empty
    assert list(wh.ohlcv_intraday('QQQ', '2021-01-04').columns) == \
        list(day.columns)


def test_write_bars_validates_and_skips_empty(tmp_path):
    wh = PitWarehouse(str(tmp_path))
    with pytest.raises(ValueError, match='missing columns'):
        wh.write_bars_1m('SPY', pd.DataFrame({'ts': []}))
    # Empty frame writes NOTHING (failed fetch must not look ingested).
    assert wh.write_bars_1m('SPY', _sample_frame().iloc[0:0]) == 0
    assert not (tmp_path / 'bars_1m' / 'SPY.parquet').exists()
    assert wh.ohlcv_intraday('SPY', '2021-01-04').empty


def test_write_is_per_symbol_overwrite(tmp_path):
    wh = PitWarehouse(str(tmp_path))
    wh.write_bars_1m('SPY', _sample_frame())
    qqq = _sample_frame().assign(ticker='QQQ')
    wh.write_bars_1m('QQQ', qqq)
    # Overwriting SPY leaves QQQ untouched (per-symbol Parquet layout).
    one = _sample_frame().iloc[:1]
    assert wh.write_bars_1m('SPY', one) == 1
    assert len(wh.ohlcv_intraday_range('SPY', '2021-01-01', '2021-12-31')) == 1
    assert len(wh.ohlcv_intraday_range('QQQ', '2021-01-01', '2021-12-31')) == 3


def test_existing_sharadar_registry_byte_identical():
    """The Sharadar bulk-export registry must be untouched by the intraday
    seam — pinned entry-for-entry (and bars_1m must NOT be in it, or
    ingest_table would try a Sharadar export of an Alpaca table)."""
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
    # bars_1m_sip joined the intraday registry 2026-07-10 (Massive SIP
    # sibling); the full registry is pinned in test_massive_sip_trial.py.
    assert set(_INTRADAY_TABLES) == {'bars_1m', 'bars_1m_sip'}
    assert not (set(_INTRADAY_TABLES) & set(_TABLES))
    with pytest.raises(ValueError, match='unknown table'):
        PitWarehouse('/nonexistent').ingest_table('bars_1m')
