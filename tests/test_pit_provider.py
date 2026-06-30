"""PitCache correctness gate (PLAN.md Phase 0). The two invariants the whole
honest-backtest effort depends on: survivorship-free universe_asof, and
datekey-gated point-in-time fundamentals_asof. Hermetic: :memory: cache, injected
rows, zero network.
"""

from __future__ import annotations

import pytest

from data.pit_provider import PitCache

# A survivor, a name DELISTED in 2018, a name not LISTED until 2017, and an ETF.
_TICKERS = [
    {'ticker': 'SURV', 'permaticker': 1, 'name': 'Survivor Inc',
     'category': 'Domestic Common Stock', 'isdelisted': 'N',
     'scalemarketcap': '4 - Mid', 'firstpricedate': '2005-01-03',
     'lastpricedate': '2026-06-29'},
    {'ticker': 'DEAD', 'permaticker': 2, 'name': 'Dead Co',
     'category': 'Domestic Common Stock', 'isdelisted': 'Y',
     'scalemarketcap': '2 - Micro', 'firstpricedate': '2010-01-04',
     'lastpricedate': '2018-06-29'},
    {'ticker': 'NEW', 'permaticker': 3, 'name': 'Newly Listed',
     'category': 'Domestic Common Stock', 'isdelisted': 'N',
     'scalemarketcap': '3 - Small', 'firstpricedate': '2017-01-03',
     'lastpricedate': '2026-06-29'},
    {'ticker': 'ETFX', 'permaticker': 4, 'name': 'Some ETF',
     'category': 'ETF', 'isdelisted': 'N', 'scalemarketcap': '5 - Large',
     'firstpricedate': '2005-01-03', 'lastpricedate': '2026-06-29'},
]


@pytest.fixture
def cache():
    c = PitCache(':memory:')
    c.store_tickers(_TICKERS)
    return c


# --- Survivorship: the "dead name" gate ------------------------------------

def test_universe_includes_delisted_name_during_its_live_span(cache):
    """THE acceptance test: a name delisted in 2018 MUST appear in the 2015
    universe (it was live then) and MUST NOT appear in the 2020 universe."""
    u2015 = cache.universe_asof('2015-06-30')
    assert 'DEAD' in u2015                    # was live in 2015
    u2020 = cache.universe_asof('2020-01-01')
    assert 'DEAD' not in u2020                # delisted 2018-06-29


def test_universe_excludes_not_yet_listed(cache):
    assert 'NEW' not in cache.universe_asof('2015-06-30')   # firstprice 2017
    assert 'NEW' in cache.universe_asof('2018-06-30')


def test_universe_is_survivorship_free_as_of_each_date(cache):
    assert cache.universe_asof('2008-01-01') == ['SURV']            # only SURV live
    assert cache.universe_asof('2015-06-30') == ['DEAD', 'SURV']    # DEAD live too
    assert cache.universe_asof('2020-01-01') == ['NEW', 'SURV']     # DEAD gone, NEW listed


def test_category_filter_excludes_non_common_stock(cache):
    assert 'ETFX' not in cache.universe_asof('2020-01-01')          # ETF filtered
    assert 'ETFX' in cache.universe_asof('2020-01-01', category=None)


def test_scalemarketcap_filter_scopes_by_size(cache):
    # NEW (3 - Small) and SURV (4 - Mid) are live in 2020 and in the size set;
    # DEAD (Micro, delisted) and ETFX (Large, ETF) are excluded.
    small_mid = cache.universe_asof('2020-01-01',
                                    scalemarketcap=['3 - Small', '4 - Mid'])
    assert small_mid == ['NEW', 'SURV']


# --- Point-in-time fundamentals: datekey gating ----------------------------

def test_fundamentals_are_point_in_time_by_datekey(cache):
    """A number FILED 2015-05-01 must be invisible to a 2015-03-01 query —
    gating on datekey (filing date), never calendardate."""
    cache.store_sf1([
        {'ticker': 'SURV', 'dimension': 'ARQ', 'datekey': '2015-02-01',
         'calendardate': '2014-12-31', 'eps': 1.0, 'revenue': 100},
        {'ticker': 'SURV', 'dimension': 'ARQ', 'datekey': '2015-05-01',
         'calendardate': '2015-03-31', 'eps': 1.2, 'revenue': 110},
    ])
    assert cache.fundamentals_asof('SURV', '2015-01-01') is None    # nothing filed yet
    assert cache.fundamentals_asof('SURV', '2015-03-01')['eps'] == 1.0  # only Feb filing known
    assert cache.fundamentals_asof('SURV', '2015-06-01')['eps'] == 1.2  # May filing now known


def test_fundamentals_respects_dimension(cache):
    cache.store_sf1([
        {'ticker': 'SURV', 'dimension': 'ARQ', 'datekey': '2015-02-01',
         'calendardate': '2014-12-31', 'eps': 1.0},
        {'ticker': 'SURV', 'dimension': 'ART', 'datekey': '2015-02-01',
         'calendardate': '2014-12-31', 'eps': 4.0},  # trailing-twelve
    ])
    assert cache.fundamentals_asof('SURV', '2015-03-01', dimension='ARQ')['eps'] == 1.0
    assert cache.fundamentals_asof('SURV', '2015-03-01', dimension='ART')['eps'] == 4.0


# --- Prices ----------------------------------------------------------------

def test_prices_returns_total_return_adjusted_series(cache):
    cache.store_sep([
        {'ticker': 'SURV', 'date': '2015-01-02', 'open': 10, 'high': 11,
         'low': 9, 'close': 10.5, 'volume': 1000, 'closeadj': 9.8},
        {'ticker': 'SURV', 'date': '2015-01-05', 'open': 10.5, 'high': 10.7,
         'low': 10.4, 'close': 10.6, 'volume': 1100, 'closeadj': 9.9},
    ])
    s = cache.prices('SURV', '2015-01-01', '2015-01-31')
    assert list(s) == [9.8, 9.9]                       # closeadj by default
    assert list(cache.prices('SURV', '2015-01-01', '2015-01-31', field='close')) \
        == [10.5, 10.6]
    assert cache.prices('SURV', '2015-01-03', '2015-01-04').empty   # gap between bars


# --- Insiders (SF2): point-in-time by filingdate -----------------------------

def test_insider_net_buys_is_point_in_time_and_filters_codes(cache):
    cache.store_sf2([
        {'ticker': 'SURV', 'filingdate': '2015-02-10', 'transactiondate': '2015-02-08',
         'transactioncode': 'P', 'transactionshares': 1000, 'transactionvalue': 1_000_000,
         'ownername': 'CEO'},                                   # open-market BUY
        {'ticker': 'SURV', 'filingdate': '2015-03-05', 'transactiondate': '2015-03-03',
         'transactioncode': 'S', 'transactionshares': 400, 'transactionvalue': 400_000,
         'ownername': 'CFO'},                                   # open-market SELL
        {'ticker': 'SURV', 'filingdate': '2015-02-20', 'transactiondate': '2015-02-18',
         'transactioncode': 'A', 'transactionshares': 5000, 'transactionvalue': None,
         'ownername': 'CEO'},                                   # grant -> IGNORED
        {'ticker': 'SURV', 'filingdate': '2015-06-01', 'transactiondate': '2015-05-30',
         'transactioncode': 'P', 'transactionshares': 9999, 'transactionvalue': 9_000_000,
         'ownername': 'CEO'},                                   # filed AFTER asof -> invisible
    ])
    r = cache.insider_net_buys('SURV', '2015-03-31', lookback_days=90)
    assert r['n_buys'] == 1 and r['n_sells'] == 1              # grant + future buy excluded
    assert r['net_value'] == 600_000.0                         # 1,000,000 - 400,000
    assert r['net_shares'] == 600.0
    # widen past the June filing -> it becomes visible
    assert cache.insider_net_buys('SURV', '2015-07-01', lookback_days=400)['n_buys'] == 2


def test_insider_net_buys_handles_sharadar_signed_shares(cache):
    """Sharadar signs transactionshares (negative for sells); transactionvalue is
    a positive magnitude. net_shares must come out -400 for a sell, not +400 — a
    double-negative previously made every seller count as a buyer."""
    cache.store_sf2([
        {'ticker': 'SURV', 'filingdate': '2015-02-10', 'transactiondate': '2015-02-08',
         'transactioncode': 'P', 'transactionshares': 1000, 'transactionvalue': 100000,
         'ownername': 'CEO'},
        {'ticker': 'SURV', 'filingdate': '2015-02-12', 'transactiondate': '2015-02-10',
         'transactioncode': 'S', 'transactionshares': -400, 'transactionvalue': 40000,
         'ownername': 'CFO'},                                  # Sharadar-signed sell
    ])
    r = cache.insider_net_buys('SURV', '2015-03-01', lookback_days=90)
    assert r['n_buys'] == 1 and r['n_sells'] == 1
    assert r['net_shares'] == 600.0       # 1000 - 400 (NOT 1000 + 400)
    assert r['net_value'] == 60000.0      # 100000 - 40000


# --- Institutional (SF3): the 45-day 13F filing lag (PIT) ---------------------

def test_institutional_asof_lags_13f_by_45_days(cache):
    """13F for the 2015-Q1 (calendardate 2015-03-31) is only public ~45 days
    later (~2015-05-15). It MUST be invisible on 2015-04-15 and visible on
    2015-06-01 — skipping the lag is lookahead."""
    cache.store_sf3([
        {'ticker': 'SURV', 'calendardate': '2015-03-31', 'investorname': 'FundA',
         'securitytype': 'SHR', 'value': 1_000_000, 'units': 10_000},
        {'ticker': 'SURV', 'calendardate': '2015-03-31', 'investorname': 'FundB',
         'securitytype': 'SHR', 'value': 500_000, 'units': 5_000},
    ])
    assert cache.institutional_asof('SURV', '2015-04-15') is None   # lag not elapsed
    h = cache.institutional_asof('SURV', '2015-06-01')
    assert h['calendardate'] == '2015-03-31'
    assert h['total_value'] == 1_500_000                           # summed across investors
    assert h['n_investors'] == 2


# --- DAILY metrics -----------------------------------------------------------

def test_daily_metric(cache):
    cache.store_daily([
        {'ticker': 'SURV', 'date': '2015-01-02', 'marketcap': 1000, 'pe': 15.0,
         'pb': 2.0, 'ps': 3.0, 'ev': 1200, 'evebit': 10, 'evebitda': 8},
    ])
    m = cache.daily_metric('SURV', '2015-01-02')
    assert m['pb'] == 2.0 and m['pe'] == 15.0
    assert cache.daily_metric('SURV', '2015-01-02', field='pb') == 2.0
    assert cache.daily_metric('SURV', '2015-01-03') is None        # no row that day


# --- Side-effect-free construction (mirror EarningsCache contract) ----------

def test_missing_cache_reads_empty_and_creates_no_file(tmp_path):
    p = tmp_path / 'nope.db'
    c = PitCache(str(p))
    assert c.universe_asof('2015-06-30') == []
    assert c.fundamentals_asof('X', '2015-06-30') is None
    assert c.prices('X', '2015-01-01', '2015-12-31').empty
    assert c.insider_net_buys('X', '2015-06-30')['n_buys'] == 0
    assert c.institutional_asof('X', '2015-06-30') is None
    assert c.daily_metric('X', '2015-06-30') is None
    assert not p.exists()                              # reads created no file
