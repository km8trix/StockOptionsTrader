"""Tests for data.earnings_cache.EarningsCache.

Offline and network-free: ``next_earnings`` reads only SQLite, and
``ingest`` is exercised with a FAKE fetcher so no test ever touches
yfinance. Pins the contract Renaissance's earnings gate relies on:
earliest-date-on-or-after semantics (matching SyntheticEarningsCalendar),
empty-cache no-op, persistence, and EarningsCalendar protocol conformance.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from data.earnings_cache import EarningsCache
from desks.options_pricing import EarningsCalendar, SyntheticEarningsCalendar


def test_next_earnings_returns_earliest_on_or_after():
    cache = EarningsCache(':memory:')
    cache.store('AAPL', ['2026-01-30', '2026-04-30', '2026-07-30'])
    # On-or-after, inclusive of the date itself.
    assert cache.next_earnings('AAPL', '2026-02-01') == dt.date(2026, 4, 30)
    assert cache.next_earnings('AAPL', '2026-04-30') == dt.date(2026, 4, 30)
    # Past the last known date -> None (shallow-history reality).
    assert cache.next_earnings('AAPL', '2026-08-01') is None
    # Unknown symbol -> None.
    assert cache.next_earnings('NVDA', '2026-01-01') is None


def test_empty_cache_is_a_noop():
    # An un-ingested cache returns None for everything: this is what keeps the
    # registry flag byte-identical until the operator runs ingest.
    cache = EarningsCache(':memory:')
    assert cache.next_earnings('AAPL', '2026-01-01') is None


def test_matches_synthetic_calendar_semantics():
    dates = ['2026-03-15', '2026-06-15']
    cache = EarningsCache(':memory:')
    cache.store('MSFT', dates)
    synth = SyntheticEarningsCalendar({'MSFT': dates})
    for probe in ['2026-01-01', '2026-03-15', '2026-03-16', '2026-07-01']:
        assert cache.next_earnings('MSFT', probe) == synth.next_earnings('MSFT', probe)


def test_conforms_to_earnings_calendar_protocol():
    assert isinstance(EarningsCache(':memory:'), EarningsCalendar)


def test_ingest_uses_injected_fetcher_no_network():
    calls = []

    def fake_fetch(symbol):
        calls.append(symbol)
        return [dt.date(2026, 5, 1)]

    cache = EarningsCache(':memory:')
    counts = cache.ingest(['AAPL', 'MSFT'], fetcher=fake_fetch)
    assert counts == {'AAPL': 1, 'MSFT': 1}
    assert calls == ['AAPL', 'MSFT']
    assert cache.next_earnings('AAPL', '2026-01-01') == dt.date(2026, 5, 1)


def test_store_is_idempotent_and_skips_bad_dates():
    cache = EarningsCache(':memory:')
    assert cache.store('AAPL', ['2026-01-30', 'not-a-date']) == 1  # bad one skipped
    assert cache.store('AAPL', ['2026-01-30']) == 1  # re-store, no duplicate row
    assert cache.next_earnings('AAPL', '2026-01-01') == dt.date(2026, 1, 30)


def test_store_skips_nat_without_corrupting_cache():
    # pd.NaT/None slip past a (TypeError, ValueError) guard because
    # pd.Timestamp(NaT).date().isoformat() == 'NaT' (a string, no exception),
    # which would then crash next_earnings. They must be skipped instead.
    cache = EarningsCache(':memory:')
    assert cache.store('AAPL', [pd.NaT, None, '2026-01-30']) == 1
    assert cache.next_earnings('AAPL', '2026-01-01') == dt.date(2026, 1, 30)  # no crash


def test_persists_across_reopen(tmp_path):
    db = str(tmp_path / 'earnings.db')
    EarningsCache(db).store('AAPL', ['2026-09-01'])
    # A fresh handle on the same file sees the stored date.
    assert EarningsCache(db).next_earnings('AAPL', '2026-01-01') == dt.date(2026, 9, 1)


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_make_edgar_fetcher_extracts_8k_item_202_dates(monkeypatch):
    # No live network: canned SEC JSON routed by URL. Earnings = 8-K Item 2.02
    # filing dates, across filings.recent AND the older filings.files batch.
    import data.earnings_cache as ec

    tickers = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
    submissions = {"filings": {
        "recent": {
            "form": ["8-K", "8-K", "10-Q"],
            "items": ["2.02,9.01", "5.02", ""],     # 5.02 (mgmt change) and 10-Q ignored
            "filingDate": ["2026-01-30", "2026-02-15", "2026-02-01"],
        },
        "files": [{"name": "CIK0000320193-submissions-001.json"}],
    }}
    older = {"form": ["8-K"], "items": ["2.02"], "filingDate": ["2019-01-29"]}

    def fake_get(url, timeout=None, headers=None):
        if "company_tickers" in url:
            return _FakeResp(tickers)
        if url.endswith("001.json"):
            return _FakeResp(older)
        return _FakeResp(submissions)

    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)

    fetch = ec.make_edgar_fetcher(user_agent="Test test@example.com")
    assert fetch("AAPL") == [dt.date(2019, 1, 29), dt.date(2026, 1, 30)]  # sorted, 2.02 only
    assert fetch("ZZZZ") == []  # unknown ticker -> US filers only


def test_make_edgar_fetcher_requires_user_agent(monkeypatch):
    import data.earnings_cache as ec
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(ValueError, match="SEC_USER_AGENT"):
        ec.make_edgar_fetcher()


def test_read_symbols_file_skips_blanks_and_comments(tmp_path):
    from data.earnings_cache import _read_symbols_file
    p = tmp_path / 'universe.txt'
    p.write_text('AAPL\n\n  MSFT  \n# a comment\nNVDA\n')
    assert _read_symbols_file(str(p)) == ['AAPL', 'MSFT', 'NVDA']
