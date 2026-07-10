"""Hermetic tests for the Massive (ex-Polygon) SIP minute-bar seam.

Covers the 2026-07-10 additions (no network anywhere; conftest hard-blocks
sockets):
  * scripts/ingest_alpaca_bars.py — ``--provider massive``: canned Massive
    JSON page, ``next_url`` pagination (followed VERBATIM, auth header only),
    429 retry, and the epoch-ms UTC -> tz-naive US/Eastern conversion pinned
    on a WINTER (EST) and a SUMMER (EDT) timestamp.
  * data/pit_warehouse.py — sibling-table (``--table``) round-trip on a tmp
    dir, isolated from the default ``bars_1m`` table.
  * DEFAULT-PATH BYTE-IDENTITY PINS for all three touched files: warehouse
    registry/read-helper defaults, ingest provider/table defaults, and the
    orb_gate bars-table/gated-cut defaults (combined = the registered
    2016-2024 pilot behavior).
"""

import inspect

import pandas as pd
import pytest

from data.pit_warehouse import (_BARS_1M_COLUMNS, _INTRADAY_TABLES,
                                PitWarehouse)
from scripts import ingest_alpaca_bars as ingest
from scripts import orb_gate as orb
from scripts.ingest_alpaca_bars import (MASSIVE_PAGE_LIMIT,
                                        fetch_symbol_bars_massive,
                                        massive_bars_to_df)

# Pinned epoch-ms UTC instants (see test below):
WINTER_T = 1736173800000   # 2025-01-06 14:30:00 UTC == 09:30 EST (winter)
SUMMER_T = 1751895000000   # 2025-07-07 13:30:00 UTC == 09:30 EDT (summer)


def _mbar(t, o=100.0, h=101.0, low=99.0, c=100.5, v=1200, n=17, vw=100.25):
    """Massive aggregate item: same o/h/l/c/v/n/vw keys as Alpaca, but ``t``
    is epoch-MILLISECONDS UTC (int), not RFC-3339."""
    return {'t': t, 'o': o, 'h': h, 'l': low, 'c': c, 'v': v, 'n': n, 'vw': vw}


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
# Epoch-ms UTC -> tz-naive US/Eastern (DST-correct, same convention as Alpaca)
# ---------------------------------------------------------------------------
def test_massive_bars_to_df_epoch_ms_to_naive_eastern():
    """Winter: 14:30 UTC == 09:30 EST. Summer: 13:30 UTC == 09:30 EDT.
    Both must land on the 09:30 wall-clock open, tz-NAIVE."""
    df = massive_bars_to_df([_mbar(WINTER_T), _mbar(SUMMER_T)], 'SPY')
    assert list(df.columns) == list(_BARS_1M_COLUMNS)
    assert df['ts'].dt.tz is None                            # tz-NAIVE ET
    assert df['ts'].iloc[0] == pd.Timestamp('2025-01-06 09:30:00')  # EST
    assert df['ts'].iloc[1] == pd.Timestamp('2025-07-07 09:30:00')  # EDT
    assert (df['ticker'] == 'SPY').all()
    assert df['volume'].iloc[0] == 1200 and df['trade_count'].iloc[0] == 17
    assert df['vwap'].iloc[0] == pytest.approx(100.25)
    assert massive_bars_to_df([], 'SPY').empty


def test_massive_dedups_and_sorts_like_alpaca_path():
    dup = _mbar(WINTER_T)
    df = massive_bars_to_df([_mbar(WINTER_T + 60_000), dup, dup], 'SPY')
    assert len(df) == 2 and df['ts'].is_monotonic_increasing


# ---------------------------------------------------------------------------
# Pagination — canned Massive JSON, next_url followed VERBATIM, 429 retry
# ---------------------------------------------------------------------------
def test_massive_fetch_next_url_pagination_and_429():
    next_url = ('https://api.polygon.io/v2/aggs/ticker/SPY/range/1/minute/'
                '2025-01-02/2026-06-30?cursor=abc123')
    pages = [
        _Resp(200, {'status': 'OK', 'resultsCount': 2,
                    'results': [_mbar(WINTER_T), _mbar(WINTER_T + 60_000)],
                    'next_url': next_url}),
        _Resp(429, headers={'Retry-After': '12'}),           # retry same URL
        _Resp(200, {'status': 'OK', 'resultsCount': 1,
                    'results': [_mbar(WINTER_T + 120_000)]}),  # no next_url
    ]
    calls, sleeps = [], []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append((url, params, dict(headers)))
        return pages[len(calls) - 1]

    auth = {'Authorization': 'Bearer TESTTOKEN'}
    df = fetch_symbol_bars_massive('SPY', '2025-01-02', '2026-06-30',
                                   get=fake_get, headers=auth,
                                   sleep=sleeps.append)
    assert len(df) == 3 and sleeps == [12.0]
    url0, params0, headers0 = calls[0]
    assert url0 == ('https://api.polygon.io/v2/aggs/ticker/SPY'
                    '/range/1/minute/2025-01-02/2026-06-30')
    assert params0 == {'limit': 50_000, 'adjusted': 'false'}
    assert MASSIVE_PAGE_LIMIT == 50_000
    # The 429 retried the SAME next_url; the cursor URL is followed VERBATIM
    # with NO params re-sent — ONLY the Authorization header rides along.
    assert calls[1][0] == next_url and calls[2][0] == next_url
    assert calls[1][1] is None and calls[2][1] is None
    assert all(h == auth for _, _, h in calls)
    assert df['ts'].iloc[-1] == pd.Timestamp('2025-01-06 09:32:00')


def test_massive_fetch_gives_up_after_max_429s():
    resp_429 = _Resp(429, headers={'Retry-After': '1'})

    def always_429(url, params=None, headers=None, timeout=None):
        return resp_429

    with pytest.raises(RuntimeError, match='consecutive 429s'):
        fetch_symbol_bars_massive('SPY', '2025-01-02', '2025-01-03',
                                  get=always_429, headers={}, sleep=lambda _: None)


# ---------------------------------------------------------------------------
# Warehouse sibling table — round-trip on tmp dir, isolated from bars_1m
# ---------------------------------------------------------------------------
def test_sibling_table_roundtrip_isolated_from_default(tmp_path):
    wh = PitWarehouse(str(tmp_path))
    df = massive_bars_to_df([_mbar(WINTER_T, c=600.5),
                             _mbar(WINTER_T + 60_000, c=600.7),
                             _mbar(SUMMER_T, c=620.1)], 'SPY')
    assert wh.write_bars_1m('SPY', df, table='bars_1m_sip') == 3
    assert (tmp_path / 'bars_1m_sip' / 'SPY.parquet').exists()
    assert not (tmp_path / 'bars_1m' / 'SPY.parquet').exists()
    day = wh.ohlcv_intraday('SPY', '2025-01-06', table='bars_1m_sip')
    assert day['close'].tolist() == [600.5, 600.7]
    assert day.index.tz is None                              # tz-naive ET
    rng = wh.ohlcv_intraday_range('SPY', '2025-01-01', '2025-12-31',
                                  table='bars_1m_sip')
    assert len(rng) == 3 and rng.index.is_monotonic_increasing
    # The DEFAULT table sees NOTHING — the SIP sibling never shadows the
    # IEX pilot data (and vice versa).
    assert wh.ohlcv_intraday('SPY', '2025-01-06').empty
    assert wh.ohlcv_intraday_range('SPY', '2025-01-01', '2025-12-31').empty


def test_run_gate_reads_from_bars_table(tmp_path):
    """run_gate(bars_table='bars_1m_sip') must read the SIBLING table: with
    only default-table data present it errors, naming the missing table."""
    wh = PitWarehouse(str(tmp_path))
    df = massive_bars_to_df([_mbar(WINTER_T)], 'SPY')
    assert wh.write_bars_1m('SPY', df) == 1                  # bars_1m only
    with pytest.raises(SystemExit, match='no bars_1m_sip data for SPY'):
        orb.run_gate(['SPY'], '2025-01-02', '2026-06-30',
                     warehouse_dir=str(tmp_path), bars_table='bars_1m_sip')


# ---------------------------------------------------------------------------
# Default-path byte-identity pins — registry, provider, gated-cut
# ---------------------------------------------------------------------------
def test_warehouse_defaults_pinned_to_bars_1m():
    """Every intraday helper defaults to 'bars_1m' — the pre-sibling default
    path stays byte-identical. The bars_1m registry entry is unchanged and
    bars_1m_sip is a SIBLING, still rejected by the Sharadar ingest."""
    for fn in (PitWarehouse._bars_pq, PitWarehouse.write_bars_1m,
               PitWarehouse._bars_query, PitWarehouse.ohlcv_intraday,
               PitWarehouse.ohlcv_intraday_range):
        assert inspect.signature(fn).parameters['table'].default == 'bars_1m'
    assert _INTRADAY_TABLES == {
        'bars_1m': ('ALPACA/v2/stocks/{symbol}/bars',
                    {'timeframe': '1Min', 'feed': 'iex'}),
        'bars_1m_sip': ('MASSIVE/v2/aggs/ticker/{symbol}/range/1/minute',
                        {'adjusted': 'false', 'limit': 50000}),
    }
    with pytest.raises(ValueError, match='unknown table'):
        PitWarehouse('/nonexistent').ingest_table('bars_1m_sip')


def test_ingest_cli_defaults_pinned_to_alpaca_bars_1m():
    args = ingest.build_parser().parse_args([])
    assert args.provider == 'alpaca'                         # byte-identical
    assert args.table == 'bars_1m'
    assert args.symbols == ['SPY', 'QQQ', 'IWM', 'TQQQ']
    assert (args.start, args.end) == ('2016-01-01', '2024-12-31')


def test_gate_cli_and_run_gate_defaults_pinned_to_registered_config():
    args = orb.build_parser().parse_args([])
    assert args.bars_table == 'bars_1m'                      # byte-identical
    assert args.gated_cut == 'combined'
    params = inspect.signature(orb.run_gate).parameters
    assert params['bars_table'].default == 'bars_1m'
    assert params['gated_cut'].default == 'combined'


# ---------------------------------------------------------------------------
# print_report — default output unchanged; long_only cut flips the verdict key
# ---------------------------------------------------------------------------
def _fake_gate(passed):
    return {'psr': 0.90, 'psr_pass': False, 'n_periods_tested': 1,
            'bh': {'n_significant_bh': 0, 'rejected_bh': [False]},
            'fold_pass': False, 'fold_labels': ['2025'],
            'fold_pvalues': [0.5], 'passed': passed,
            'risk_free_rate': 0.02}


def _fake_res(**extra):
    sym = {'n_days': 10, 'final_equity': 100_000.0, 'total_return_pct': 0.0,
           'n_trades': 0, 'trades_per_year': 0.0, 'win_rate': 0.0,
           'avg_win': 0.0, 'avg_loss': 0.0}
    return {'symbols': {'SPY': dict(sym)}, 'long_only': {'SPY': dict(sym)},
            'gate': _fake_gate(True), 'gate_long_only_reported':
            _fake_gate(False), **extra}


def test_print_report_default_verdict_is_combined(capsys):
    """No gated_cut/bars_table keys (pre-2026-07-10 res shape) -> the
    combined gate drives the verdict and the IEX caveat prints — the
    registered pilot's output, byte-identical."""
    orb.print_report(_fake_res(), ['SPY'], '2016-01-01', '2024-12-31')
    out = capsys.readouterr().out
    assert 'VERDICT (combined long+short): PASS' in out
    assert '[IEX-only minute bars' in out
    assert 'LONG+SHORT (gated)' in out and 'LONG-ONLY (reported)' in out
    assert 'GATE [COMBINED long+short portfolio]:' in out    # no suffix
    assert 'GATE [long-only portfolio (reported, not gated)]:' in out


def test_print_report_long_only_cut_flips_gate_and_caveat(capsys):
    res = _fake_res(bars_table='bars_1m_sip', gated_cut='long_only')
    orb.print_report(res, ['SPY'], '2025-01-02', '2026-06-30')
    out = capsys.readouterr().out
    # Verdict now reads the LONG-ONLY validation (False here), not combined.
    assert 'VERDICT (long-only): FAIL' in out
    assert 'bars table: bars_1m_sip' in out
    assert '[IEX-only minute bars' not in out
    assert 'LONG-ONLY (gated)' in out and 'LONG+SHORT (reported)' in out
    assert 'long-only portfolio (GATED — SIP confirmation trial)' in out
    assert 'COMBINED long+short portfolio (reported, not gated)' in out
