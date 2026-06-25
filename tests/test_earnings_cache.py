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
