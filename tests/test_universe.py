"""Universe lists, ETF holdings fallback, and batch_fetch isolation.

Offline: data.universe._get_openbb is monkeypatched; batch_fetch runs
against a fake handler and an injected sleep_fn (never actually sleeps).
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

import data.universe as universe
from data.universe import (
    CORE_ETFS,
    LARGE_CAP_100,
    SECTOR_ETFS,
    STATIC_HOLDINGS,
    batch_fetch,
    get_etf_holdings,
)


class TestStaticLists:
    def test_core_etfs(self):
        assert CORE_ETFS == ['SPY', 'QQQ', 'IWM', 'DIA']

    def test_sector_etfs_are_the_11_spdrs(self):
        assert len(SECTOR_ETFS) == 11
        assert len(set(SECTOR_ETFS)) == 11
        assert all(s.startswith('XL') for s in SECTOR_ETFS)

    def test_large_cap_list_sanity(self):
        assert 90 <= len(LARGE_CAP_100) <= 110
        assert len(set(LARGE_CAP_100)) == len(LARGE_CAP_100)  # no dups
        assert all(s == s.upper() for s in LARGE_CAP_100)
        assert LARGE_CAP_100 == sorted(LARGE_CAP_100)  # alphabetized

    def test_static_holdings_cover_spy_qqq_and_sectors(self):
        expected_keys = {'SPY', 'QQQ'} | set(SECTOR_ETFS)
        assert set(STATIC_HOLDINGS.keys()) == expected_keys
        for etf, holdings in STATIC_HOLDINGS.items():
            assert len(holdings) >= 8, etf
            assert len(set(holdings)) == len(holdings), etf
            assert all(s == s.upper() for s in holdings), etf


class TestGetEtfHoldings:
    def test_openbb_success_path(self, monkeypatch):
        fake_obb = SimpleNamespace(etf=SimpleNamespace(
            holdings=lambda symbol: SimpleNamespace(results=[
                SimpleNamespace(symbol='aapl'),
                SimpleNamespace(symbol='MSFT'),
                SimpleNamespace(symbol=None),  # skipped, not crashed
            ])
        ))
        monkeypatch.setattr(universe, '_get_openbb', lambda: fake_obb)

        assert get_etf_holdings('SPY') == ['AAPL', 'MSFT']

    def test_falls_back_to_static_when_openbb_missing(self, monkeypatch):
        monkeypatch.setattr(universe, '_get_openbb', lambda: None)

        assert get_etf_holdings('SPY') == STATIC_HOLDINGS['SPY']

    def test_falls_back_to_static_when_openbb_raises(self, monkeypatch):
        def _boom(symbol):
            raise RuntimeError("provider exploded")

        fake_obb = SimpleNamespace(etf=SimpleNamespace(holdings=_boom))
        monkeypatch.setattr(universe, '_get_openbb', lambda: fake_obb)

        assert get_etf_holdings('xlf') == STATIC_HOLDINGS['XLF']

    def test_unknown_etf_returns_empty_list(self, monkeypatch):
        monkeypatch.setattr(universe, '_get_openbb', lambda: None)

        assert get_etf_holdings('ZZZT') == []


class FakeHandler:
    """fetch_stock_data behavior per symbol: 'ok' | 'empty' | Exception.

    from_cache controls what get_last_fetch_info reports per symbol.
    """

    def __init__(self, behaviors, from_cache=None):
        self.behaviors = behaviors
        self.from_cache = from_cache or {}
        self.fetch_calls = []

    def fetch_stock_data(self, symbol, start_date, end_date):
        self.fetch_calls.append(symbol)
        behavior = self.behaviors[symbol]
        if isinstance(behavior, Exception):
            raise behavior
        if behavior == 'empty':
            return pd.DataFrame()
        return pd.DataFrame({'close': [1.0]},
                            index=pd.DatetimeIndex(['2023-01-03'],
                                                   name='date'))

    def get_last_fetch_info(self, symbol):
        if symbol not in self.fetch_calls:
            return None
        return {
            'provider': None if self.from_cache.get(symbol) else 'fake',
            'from_cache': bool(self.from_cache.get(symbol)),
            'fetched_at': '2026-06-11T10:00:00',
            'failures': [],
            'start_date': '2023-01-01',
            'end_date': '2023-01-31',
        }


class TestBatchFetch:
    def test_error_is_isolated_and_statuses_correct(self):
        handler = FakeHandler({
            'A': 'ok',
            'B': ValueError('boom'),
            'C': 'empty',
            'D': 'ok',
        })
        sleeps = []

        results = batch_fetch(handler, ['A', 'B', 'C', 'D'],
                              '2023-01-01', '2023-01-31',
                              sleep_fn=sleeps.append)

        # The erroring symbol never aborts the batch.
        assert handler.fetch_calls == ['A', 'B', 'C', 'D']
        assert results == {
            'A': 'ok',
            'B': 'error: boom',
            'C': 'empty',
            'D': 'ok',
        }

    def test_sleeps_between_provider_fetches_not_after_last(self):
        handler = FakeHandler({'A': 'ok', 'B': 'ok', 'C': 'ok'})
        sleeps = []

        batch_fetch(handler, ['A', 'B', 'C'], '2023-01-01', '2023-01-31',
                    delay_seconds=0.5, sleep_fn=sleeps.append)

        assert sleeps == [0.5, 0.5]  # between fetches only, none after C

    def test_never_sleeps_on_cache_hits(self):
        handler = FakeHandler(
            {'A': 'ok', 'B': 'ok', 'C': 'ok'},
            from_cache={'A': True, 'B': True, 'C': True},
        )
        sleeps = []

        results = batch_fetch(handler, ['A', 'B', 'C'],
                              '2023-01-01', '2023-01-31',
                              sleep_fn=sleeps.append)

        assert sleeps == []
        assert results == {'A': 'ok', 'B': 'ok', 'C': 'ok'}

    def test_mixed_cache_and_provider_sleeps(self):
        handler = FakeHandler(
            {'A': 'ok', 'B': 'ok', 'C': 'ok'},
            from_cache={'A': True, 'B': False, 'C': True},
        )
        sleeps = []

        batch_fetch(handler, ['A', 'B', 'C'], '2023-01-01', '2023-01-31',
                    delay_seconds=0.2, sleep_fn=sleeps.append)

        assert sleeps == [0.2]  # only after B, the lone provider fetch

    def test_progress_logged_every_10_symbols(self, caplog):
        import logging
        symbols = [f'S{i:02d}' for i in range(25)]
        handler = FakeHandler({s: 'ok' for s in symbols},
                              from_cache={s: True for s in symbols})

        with caplog.at_level(logging.INFO, logger='data.universe'):
            batch_fetch(handler, symbols, '2023-01-01', '2023-01-31',
                        sleep_fn=lambda _s: None)

        progress = [r for r in caplog.records if 'progress' in r.message]
        assert len(progress) == 2  # at 10 and 20 of 25

    def test_handler_without_get_last_fetch_info_still_works(self):
        class BareHandler:
            def fetch_stock_data(self, symbol, start_date, end_date):
                return pd.DataFrame({'close': [1.0]})

        sleeps = []
        results = batch_fetch(BareHandler(), ['A', 'B'],
                              '2023-01-01', '2023-01-31',
                              sleep_fn=sleeps.append)

        assert results == {'A': 'ok', 'B': 'ok'}
        assert len(sleeps) == 1  # treated as provider fetches
