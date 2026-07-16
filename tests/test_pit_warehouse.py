"""Hermetic tests for PitWarehouse — DuckDB-over-Parquet PIT reads.

Parquet fixtures are written with DuckDB itself (no pyarrow dep), with dates as
native DATE so the read SQL's CAST(... AS DATE) gating is exercised end to end.
Every test asserts the point-in-time / survivorship invariant the method exists
to guarantee — the same contract as data/pit_provider.PitCache.
"""

import duckdb
import pandas as pd
import pytest

from data.pit_warehouse import (
    PitWarehouse,
    SecurityLifecycleError,
    WarehouseQueryError,
    WarehouseSchemaError,
    WarehouseTableMissingError,
)


def _write(path, columns, rows):
    """Write `rows` to a Parquet file at `path` with the given typed columns.

    `columns` is a list of (name, sql_type); rows are tuples of SQL literals
    already formatted (e.g. "DATE '2020-01-02'", "'AAPL'", "100.0", "NULL").
    """
    sel = ", ".join(f"{lit} AS {name}"
                    for (name, _t), lit in zip(columns, rows[0]))
    rest = "".join(
        " UNION ALL SELECT " + ", ".join(str(x) for x in r) for r in rows[1:])
    duckdb.connect().execute(
        f"COPY (SELECT {sel}{rest}) TO '{path}' (FORMAT PARQUET)")


@pytest.fixture
def wh(tmp_path):
    return PitWarehouse(str(tmp_path)), tmp_path


# ---------------------------------------------------------------------------
# Missing warehouse -> empty, side-effect-free construction
# ---------------------------------------------------------------------------
def test_missing_parquet_returns_empty(wh):
    w, _ = wh
    assert w.universe_asof('2020-01-01') == []
    assert w.fundamentals_asof('AAPL', '2020-01-01') is None
    assert w.prices('AAPL', '2020-01-01', '2020-12-31').empty
    assert w.insider_net_buys('AAPL', '2020-01-01')['n_buys'] == 0
    assert w.institutional_asof('AAPL', '2020-01-01') is None
    assert w.daily_metric('AAPL', '2020-01-01') is None


def test_construction_creates_nothing(tmp_path):
    d = tmp_path / "wh"
    PitWarehouse(str(d))
    assert not d.exists()


# ---------------------------------------------------------------------------
# universe_asof — survivorship-free
# ---------------------------------------------------------------------------
def test_universe_survivorship_free(wh):
    w, tmp = wh
    cols = [('ticker', 'VARCHAR'), ('firstpricedate', 'DATE'),
            ('lastpricedate', 'DATE'), ('category', 'VARCHAR'),
            ('scalemarketcap', 'VARCHAR'), ('isdelisted', 'VARCHAR'),
            ('permaticker', 'BIGINT')]
    _write(tmp / "tickers.parquet", cols, [
        # live across 2020, delisted 2018 (must appear in a 2017 universe), and
        # not-yet-listed until 2021 (must NOT appear in a 2020 universe).
        ("'LIVE'", "DATE '2010-01-01'", "DATE '2026-07-10'",
         "'Domestic Common Stock'", "'4 - Mid'", "'N'", "1001"),
        ("'DEAD'", "DATE '2010-01-01'", "DATE '2018-06-30'",
         "'Domestic Common Stock'", "'2 - Micro'", "'Y'", "1002"),
        ("'NEW'", "DATE '2021-01-01'", "DATE '2026-07-10'",
         "'Domestic Common Stock'", "'4 - Mid'", "'N'", "1003"),
        ("'ETF'", "DATE '2010-01-01'", "DATE '2026-07-10'", "'ETF'",
         "'6 - Mega'", "'N'", "1004"),
    ])
    _write(tmp / "sep.parquet", [('ticker', 'VARCHAR'), ('date', 'DATE')], [
        ("'DEAD'", "DATE '2018-06-30'"),
    ])
    u2017 = w.universe_asof('2017-06-30')
    assert 'DEAD' in u2017          # survivorship-free: delisted name kept for live span
    assert 'LIVE' in u2017
    u2020 = w.universe_asof('2020-06-30')
    assert 'DEAD' not in u2020      # already delisted
    assert 'NEW' not in u2020       # not yet listed
    assert 'ETF' not in u2020       # category filter (common stock only)
    # category=None lifts the filter; scalemarketcap scopes size
    assert 'ETF' in w.universe_asof('2020-06-30', category=None)
    assert w.universe_asof('2020-06-30', scalemarketcap=['4 - Mid']) == ['LIVE']
    assert w.delisting_date('DEAD').isoformat() == '2018-06-30'
    assert w.delisting_date('LIVE') is None
    assert w.delisting_date('MISSING') is None


# ---------------------------------------------------------------------------
# security_lifecycle — strict identity and delisting evidence
# ---------------------------------------------------------------------------
def test_security_lifecycle_validates_identity_status_and_final_sep_date(wh):
    w, tmp = wh
    columns = [
        ('ticker', 'VARCHAR'), ('permaticker', 'BIGINT'),
        ('isdelisted', 'VARCHAR'), ('lastpricedate', 'DATE'),
    ]
    _write(tmp / 'tickers.parquet', columns, [
        ("'DEAD'", '1001', "'Y'", "DATE '2020-01-06'"),
        ("'LIVE'", '1002', "'N'", "DATE '2026-07-10'"),
    ])
    _write(tmp / 'sep.parquet',
           [('ticker', 'VARCHAR'), ('date', 'DATE')], [
               ("'DEAD'", "DATE '2020-01-03'"),
               ("'DEAD'", "DATE '2020-01-06'"),
           ])

    dead = w.security_lifecycle('DEAD')
    assert dead == {
        'ticker': 'DEAD',
        'permaticker': 1001,
        'isdelisted': 'Y',
        'lastpricedate': pd.Timestamp('2020-01-06').date(),
        'sep_lastpricedate': pd.Timestamp('2020-01-06').date(),
    }
    live = w.security_lifecycle('LIVE')
    assert live['isdelisted'] == 'N'
    assert live['lastpricedate'] == pd.Timestamp('2026-07-10').date()
    assert live['sep_lastpricedate'] is None


@pytest.mark.parametrize(('rows', 'message'), [
    ([
        ("'DEAD'", '1001', "'Y'", "DATE '2020-01-06'"),
        ("'DEAD'", '1002', "'Y'", "DATE '2020-01-06'"),
    ], 'exactly one'),
    ([("'DEAD'", '1001', "'true'", "DATE '2020-01-06'")],
     'literal N or Y'),
    ([("'DEAD'", 'NULL', "'Y'", "DATE '2020-01-06'")],
     'invalid permaticker'),
    ([("'DEAD'", '1001', "'Y'", 'NULL')], 'lastpricedate'),
])
def test_security_lifecycle_rejects_invalid_tickers_rows(wh, rows, message):
    w, tmp = wh
    _write(tmp / 'tickers.parquet', [
        ('ticker', 'VARCHAR'), ('permaticker', 'BIGINT'),
        ('isdelisted', 'VARCHAR'), ('lastpricedate', 'DATE'),
    ], rows)

    with pytest.raises(SecurityLifecycleError, match=message):
        w.security_lifecycle('DEAD')


def test_security_lifecycle_rejects_missing_schema_and_final_date_mismatch(wh):
    w, tmp = wh
    _write(tmp / 'tickers.parquet', [
        ('ticker', 'VARCHAR'), ('isdelisted', 'VARCHAR'),
        ('lastpricedate', 'DATE'),
    ], [
        ("'DEAD'", "'Y'", "DATE '2020-01-06'"),
    ])
    with pytest.raises(SecurityLifecycleError, match='permaticker'):
        w.security_lifecycle('DEAD')

    (tmp / 'tickers.parquet').unlink()
    _write(tmp / 'tickers.parquet', [
        ('ticker', 'VARCHAR'), ('permaticker', 'BIGINT'),
        ('isdelisted', 'VARCHAR'), ('lastpricedate', 'DATE'),
    ], [
        ("'DEAD'", '1001', "'Y'", "DATE '2020-01-06'"),
    ])
    _write(tmp / 'sep.parquet',
           [('ticker', 'VARCHAR'), ('date', 'DATE')], [
               ("'DEAD'", "DATE '2020-01-03'"),
           ])
    with pytest.raises(SecurityLifecycleError, match='does not match'):
        w.security_lifecycle('DEAD')
    # The convenience method cannot accidentally treat contradictory evidence
    # as a proven delisting.
    assert w.delisting_date('DEAD') is None


def test_security_currency_is_a_strict_separate_identity_read(wh):
    w, tmp = wh
    _write(
        tmp / 'tickers.parquet',
        [('ticker', 'VARCHAR'), ('currency', 'VARCHAR')],
        [("'AAPL'", "'usd'")],
    )
    assert w.security_currency('AAPL') == {
        'ticker': 'AAPL',
        'currency': 'USD',
    }
    with pytest.raises(WarehouseSchemaError, match='exactly one'):
        w.security_currency('MSFT')


def test_corporate_action_slice_is_exact_ordered_and_semantics_free(wh):
    w, tmp = wh
    columns = [
        ('date', 'DATE'), ('action', 'VARCHAR'), ('ticker', 'VARCHAR'),
        ('name', 'VARCHAR'), ('value', 'DOUBLE'),
        ('contraticker', 'VARCHAR'), ('contraname', 'VARCHAR'),
    ]
    _write(tmp / 'actions.parquet', columns, [
        ("DATE '2020-01-05'", "'acquisitionby'", "'ABC'", "'ABC Corp'",
         '50000.0', "'XYZ'", "'XYZ Corp'"),
        ("DATE '2020-01-03'", "'dividend'", "'ABC'", "'ABC Corp'",
         '0.25', 'NULL', 'NULL'),
        ("DATE '2020-01-03'", "'dividend'", "'OTHER'", "'Other Corp'",
         '0.50', 'NULL', 'NULL'),
    ])

    assert w.corporate_actions_for_tickers(
        ['ABC'], '2020-01-01', '2020-01-31'
    ) == [
        {
            'date': '2020-01-03',
            'action': 'dividend',
            'ticker': 'ABC',
            'name': 'ABC Corp',
            'value': 0.25,
            'contraticker': None,
            'contraname': None,
        },
        {
            'date': '2020-01-05',
            'action': 'acquisitionby',
            'ticker': 'ABC',
            'name': 'ABC Corp',
            # This remains generic vendor metadata; the reader does not call it
            # per-share settlement consideration.
            'value': 50000.0,
            'contraticker': 'XYZ',
            'contraname': 'XYZ Corp',
        },
    ]

    with pytest.raises(WarehouseSchemaError, match='unique'):
        w.corporate_actions_for_tickers(
            ['ABC', 'ABC'], '2020-01-01', '2020-01-31'
        )
    with pytest.raises(WarehouseSchemaError, match='ordered'):
        w.corporate_actions_for_tickers(
            ['ABC'], '2020-02-01', '2020-01-31'
        )


# ---------------------------------------------------------------------------
# fundamentals_asof — datekey gating (no restatement lookahead)
# ---------------------------------------------------------------------------
def test_fundamentals_point_in_time(wh):
    w, tmp = wh
    cols = [('ticker', 'VARCHAR'), ('dimension', 'VARCHAR'), ('datekey', 'DATE'),
            ('calendardate', 'DATE'), ('revenue', 'DOUBLE')]
    _write(tmp / "sf1.parquet", cols, [
        ("'AAPL'", "'ARQ'", "DATE '2020-02-01'", "DATE '2019-12-31'", "100.0"),
        ("'AAPL'", "'ARQ'", "DATE '2020-05-01'", "DATE '2020-03-31'", "110.0"),
    ])
    # On 2020-03-01 only the Feb-filed row is known.
    rec = w.fundamentals_asof('AAPL', '2020-03-01')
    assert rec['revenue'] == 100.0
    # After the May filing, the newer row wins.
    assert w.fundamentals_asof('AAPL', '2020-06-01')['revenue'] == 110.0
    # Before any filing -> None (the May number is invisible on 2020-01-01).
    assert w.fundamentals_asof('AAPL', '2020-01-01') is None


# ---------------------------------------------------------------------------
# prices — range + closeadj default
# ---------------------------------------------------------------------------
def test_prices_range_and_field(wh):
    w, tmp = wh
    cols = [('ticker', 'VARCHAR'), ('date', 'DATE'), ('close', 'DOUBLE'),
            ('closeadj', 'DOUBLE'), ('closeunadj', 'DOUBLE')]
    _write(tmp / "sep.parquet", cols, [
        ("'AAPL'", "DATE '2020-01-02'", "100.0", "50.0", "99.0"),
        ("'AAPL'", "DATE '2020-01-03'", "101.0", "50.5", "100.0"),
        ("'AAPL'", "DATE '2020-02-01'", "110.0", "55.0", "109.0"),
    ])
    s = w.prices('AAPL', '2020-01-01', '2020-01-31')
    assert list(s.index) == [pd.Timestamp('2020-01-02'), pd.Timestamp('2020-01-03')]
    assert list(s.values) == [50.0, 50.5]          # closeadj by default (total return)
    assert s.name == 'AAPL'
    assert list(w.prices('AAPL', '2020-01-01', '2020-01-31', field='close').values) == [100.0, 101.0]
    assert list(w.prices(
        'AAPL', '2020-01-01', '2020-01-31',
        field='closeunadj').values) == [99.0, 100.0]
    with pytest.raises(ValueError, match='unsupported price field'):
        w.prices('AAPL', '2020-01-02', '2020-01-02', field='bogus')
    with pytest.raises(ValueError, match='unsupported price field'):
        w.prices_bulk(
            ['AAPL'], '2020-01-02', '2020-01-02', field='bogus')


def test_prices_strict_returns_exact_field_without_date_shift(wh):
    w, tmp = wh
    cols = [('ticker', 'VARCHAR'), ('date', 'DATE'),
            ('closeadj', 'DOUBLE'), ('closeunadj', 'DOUBLE')]
    _write(tmp / 'sep.parquet', cols, [
        ("'AAPL'", "DATE '2020-01-02'", "50.0", "99.0"),
        ("'AAPL'", "DATE '2020-01-03'", "50.5", "100.0"),
    ])

    adjusted = w.prices_strict(
        'AAPL', '2020-01-02', '2020-01-02', 'closeadj')
    unadjusted = w.prices_strict(
        'AAPL', '2020-01-02', '2020-01-02', 'closeunadj')
    assert adjusted.iloc[0] == 50.0
    assert unadjusted.iloc[0] == 99.0
    assert not adjusted.equals(unadjusted)

    # Exact-date reads never pull the prior or following observed session.
    assert w.prices_strict(
        'AAPL', '2020-01-04', '2020-01-04', 'closeunadj').empty


def test_prices_strict_distinguishes_missing_query_and_schema_failures(wh):
    w, tmp = wh
    with pytest.raises(WarehouseTableMissingError, match='sep'):
        w.prices_strict(
            'AAPL', '2020-01-02', '2020-01-02', 'closeunadj')

    (tmp / 'sep.parquet').write_bytes(b'not parquet')
    with pytest.raises(WarehouseQueryError, match='query failed'):
        w.prices_strict(
            'AAPL', '2020-01-02', '2020-01-02', 'closeunadj')

    (tmp / 'sep.parquet').unlink()
    _write(tmp / 'sep.parquet',
           [('ticker', 'VARCHAR'), ('date', 'DATE'), ('closeadj', 'DOUBLE')], [
               ("'AAPL'", "DATE '2020-01-02'", "50.0"),
           ])
    with pytest.raises(WarehouseSchemaError, match='closeunadj'):
        w.prices_strict(
            'AAPL', '2020-01-02', '2020-01-02', 'closeunadj')


@pytest.mark.parametrize('bad_value', ['NULL', "CAST('NaN' AS DOUBLE)"])
def test_prices_strict_rejects_non_finite_values(wh, bad_value):
    w, tmp = wh
    _write(tmp / 'sep.parquet',
           [('ticker', 'VARCHAR'), ('date', 'DATE'), ('closeadj', 'DOUBLE')], [
               ("'AAPL'", "DATE '2020-01-02'", bad_value),
           ])

    with pytest.raises(WarehouseSchemaError, match='non-finite|non-numeric'):
        w.prices_strict('AAPL', '2020-01-02', '2020-01-02', 'closeadj')


def test_prices_strict_rejects_duplicate_dates(wh):
    w, tmp = wh
    _write(tmp / 'sep.parquet',
           [('ticker', 'VARCHAR'), ('date', 'DATE'), ('closeadj', 'DOUBLE')], [
               ("'AAPL'", "DATE '2020-01-02'", "50.0"),
               ("'AAPL'", "DATE '2020-01-02'", "51.0"),
           ])

    with pytest.raises(WarehouseSchemaError, match='sorted and unique'):
        w.prices_strict('AAPL', '2020-01-02', '2020-01-02', 'closeadj')


# ---------------------------------------------------------------------------
# insider_net_buys — sign convention (PR #67) + filingdate window
# ---------------------------------------------------------------------------
def test_insider_net_buys_sign_and_window(wh):
    w, tmp = wh
    cols = [('ticker', 'VARCHAR'), ('filingdate', 'DATE'),
            ('transactioncode', 'VARCHAR'), ('transactionshares', 'DOUBLE'),
            ('transactionvalue', 'DOUBLE')]
    _write(tmp / "sf2.parquet", cols, [
        # buy: +1000 sh / +$150k
        ("'AAPL'", "DATE '2020-06-10'", "'P'", "1000", "150000.0"),
        # sell: Sharadar signs shares NEGATIVE, value POSITIVE magnitude.
        # Correct netting must subtract abs of both -> net = buy - sell.
        ("'AAPL'", "DATE '2020-06-15'", "'S'", "-400", "60000.0"),
        # grant (code 'G') -> ignored (no directional signal)
        ("'AAPL'", "DATE '2020-06-12'", "'G'", "500", "0.0"),
        # outside the lookback window -> excluded
        ("'AAPL'", "DATE '2020-01-01'", "'P'", "9999", "999999.0"),
        # filed AFTER asof -> invisible (point-in-time)
        ("'AAPL'", "DATE '2020-07-01'", "'P'", "8888", "888888.0"),
    ])
    r = w.insider_net_buys('AAPL', '2020-06-30', lookback_days=90)
    assert r['net_shares'] == 600.0          # 1000 - 400, NOT 1000 + 400
    assert r['net_value'] == 90000.0         # 150k - 60k
    assert r['n_buys'] == 1 and r['n_sells'] == 1
    assert r['window_start'] == '2020-04-01'


# ---------------------------------------------------------------------------
# institutional_asof — 45-day 13F lag
# ---------------------------------------------------------------------------
def test_institutional_lag(wh):
    w, tmp = wh
    cols = [('ticker', 'VARCHAR'), ('calendardate', 'DATE'),
            ('investorname', 'VARCHAR'), ('value', 'DOUBLE'), ('units', 'DOUBLE')]
    _write(tmp / "sf3.parquet", cols, [
        ("'AAPL'", "DATE '2020-03-31'", "'Fund A'", "1000.0", "10.0"),
        ("'AAPL'", "DATE '2020-03-31'", "'Fund B'", "2000.0", "20.0"),
    ])
    # 2020-03-31 + 45d = 2020-05-15. Before that the quarter is not yet public.
    assert w.institutional_asof('AAPL', '2020-05-01') is None
    got = w.institutional_asof('AAPL', '2020-05-20')
    assert got['calendardate'] == '2020-03-31'
    assert got['total_value'] == 3000.0      # summed across both investors
    assert got['n_investors'] == 2


def test_institutional_calendardate_is_bare_date_even_if_timestamp(wh):
    # If calendardate is stored TIMESTAMP (not DATE), the CAST must still return
    # a bare 'YYYY-MM-DD' to match SQLite PitCache — not '...T00:00:00'.
    w, tmp = wh
    cols = [('ticker', 'VARCHAR'), ('calendardate', 'TIMESTAMP'),
            ('investorname', 'VARCHAR'), ('value', 'DOUBLE'), ('units', 'DOUBLE')]
    _write(tmp / "sf3.parquet", cols, [
        ("'AAPL'", "TIMESTAMP '2020-03-31 00:00:00'", "'Fund A'", "1000.0", "10.0"),
    ])
    got = w.institutional_asof('AAPL', '2020-05-20')
    assert got['calendardate'] == '2020-03-31'


# ---------------------------------------------------------------------------
# daily_metric — exact-date row + field selection
# ---------------------------------------------------------------------------
def test_daily_metric(wh):
    w, tmp = wh
    cols = [('ticker', 'VARCHAR'), ('date', 'DATE'), ('marketcap', 'DOUBLE'),
            ('pe', 'DOUBLE'), ('pb', 'DOUBLE'), ('ps', 'DOUBLE'),
            ('ev', 'DOUBLE'), ('evebit', 'DOUBLE'), ('evebitda', 'DOUBLE')]
    _write(tmp / "daily.parquet", cols, [
        ("'AAPL'", "DATE '2020-06-30'", "2.0e12", "29.0", "45.0", "7.0", "2.1e12", "22.0", "20.0"),
    ])
    rec = w.daily_metric('AAPL', '2020-06-30')
    assert rec['pe'] == 29.0 and rec['pb'] == 45.0
    assert w.daily_metric('AAPL', '2020-06-30', field='ps') == 7.0
    assert w.daily_metric('AAPL', '2020-07-01') is None   # no row that day
