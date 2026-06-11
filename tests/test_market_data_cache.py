"""MarketDataHandler cache integration + get_last_fetch_info contract.

Offline: MarketDataHandler._get_openbb is monkeypatched to a fake OpenBB
object whose provider calls are counted; SQLite lives under tmp_path.
"""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from data.cache import OHLCVCache
from data.market_data import MarketDataHandler

START = "2023-01-02"
END = "2023-01-06"

CONTRACT_KEYS = {
    'provider', 'from_cache', 'fetched_at', 'failures',
    'start_date', 'end_date',
}


def _bars(start: str = START, n: int = 5):
    """Synthetic OBB result items (one bar per business day)."""
    items = []
    d = date.fromisoformat(start)
    made = 0
    while made < n:
        if d.weekday() < 5:
            price = 100.0 + made
            items.append(SimpleNamespace(
                date=d, open=price, high=price + 1.0, low=price - 1.0,
                close=price + 0.5, volume=1_000_000.0,
            ))
            made += 1
        d += timedelta(days=1)
    return items


class FakeOBB:
    """Counts provider calls; per-provider behavior: items list or Exception."""

    def __init__(self, behaviors):
        self.behaviors = behaviors
        self.calls = []
        self.equity = SimpleNamespace(
            price=SimpleNamespace(historical=self._historical)
        )

    def _historical(self, symbol, start_date, end_date, provider):
        self.calls.append(provider)
        behavior = self.behaviors[provider]
        if isinstance(behavior, Exception):
            raise behavior
        return SimpleNamespace(results=behavior)


@pytest.fixture
def shared_cache(tmp_path):
    return OHLCVCache(db_path=str(tmp_path / "md_cache.db"))


def make_handler(monkeypatch, fake_obb, cache, providers):
    handler = MarketDataHandler(cache=cache)
    handler.providers = list(providers)
    monkeypatch.setattr(MarketDataHandler, "_get_openbb",
                        lambda self: fake_obb)
    return handler


class TestFetchFlow:
    def test_first_fetch_hits_provider(self, monkeypatch, shared_cache):
        fake = FakeOBB({"good": _bars()})
        handler = make_handler(monkeypatch, fake, shared_cache, ["good"])

        df = handler.fetch_stock_data("AAPL", START, END)

        assert fake.calls == ["good"]
        assert len(df) == 5
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        info = handler.get_last_fetch_info("AAPL")
        assert info["provider"] == "good"
        assert info["from_cache"] is False
        assert info["failures"] == []

    def test_second_fetch_same_handler_uses_memory_no_provider_call(
            self, monkeypatch, shared_cache):
        fake = FakeOBB({"good": _bars()})
        handler = make_handler(monkeypatch, fake, shared_cache, ["good"])
        first = handler.fetch_stock_data("AAPL", START, END)

        second = handler.fetch_stock_data("AAPL", START, END)

        assert fake.calls == ["good"]  # exactly one provider call ever
        pd.testing.assert_frame_equal(first, second)
        info = handler.get_last_fetch_info("AAPL")
        assert info["from_cache"] is True
        assert info["provider"] is None

    def test_fresh_handler_served_from_sqlite_no_provider_call(
            self, monkeypatch, shared_cache):
        fake = FakeOBB({"good": _bars()})
        first_handler = make_handler(monkeypatch, fake, shared_cache, ["good"])
        first = first_handler.fetch_stock_data("AAPL", START, END)
        assert fake.calls == ["good"]

        # New process simulation: fresh handler, same SQLite file.
        second_handler = make_handler(monkeypatch, fake, shared_cache, ["good"])
        second = second_handler.fetch_stock_data("AAPL", START, END)

        assert fake.calls == ["good"]  # provider NOT called again
        pd.testing.assert_frame_equal(first, second, check_freq=False)
        info = second_handler.get_last_fetch_info("AAPL")
        assert info["from_cache"] is True
        assert info["provider"] is None

    def test_cache_false_disables_sqlite(self, monkeypatch, tmp_path):
        fake = FakeOBB({"good": _bars()})
        handler1 = make_handler(monkeypatch, fake, False, ["good"])
        handler1.fetch_stock_data("AAPL", START, END)

        handler2 = make_handler(monkeypatch, fake, False, ["good"])
        handler2.fetch_stock_data("AAPL", START, END)

        assert fake.calls == ["good", "good"]  # no cross-handler cache


class TestDateCanonicalizationEndToEnd:
    """Regression: a strptime-valid but non-padded request ('2023-1-2')
    against padded cached coverage used to return an EMPTY frame as a valid
    hit (from_cache=True, failures=[]) without consulting the provider that
    held the real bars. Must now serve the real bars."""

    def test_non_padded_request_after_padded_fetch(self, monkeypatch,
                                                   shared_cache):
        fake = FakeOBB({"good": _bars()})
        handler = make_handler(monkeypatch, fake, shared_cache, ["good"])
        first = handler.fetch_stock_data("AAPL", START, END)  # caches 5 bars
        assert fake.calls == ["good"]
        assert len(first) == 5

        # Fresh handler: user-typed non-padded start reaches the SQLite path.
        fresh = make_handler(monkeypatch, fake, shared_cache, ["good"])
        df = fresh.fetch_stock_data("AAPL", "2023-1-2", END)

        assert len(df) == 5  # the real bars, never an empty "covered" frame
        pd.testing.assert_frame_equal(df, first, check_freq=False)
        info = fresh.get_last_fetch_info("AAPL")
        assert info["from_cache"] is True
        assert info["failures"] == []
        assert fake.calls == ["good"]  # provider untouched on the second call


class TestGetLastFetchInfoContract:
    def test_exact_keys(self, monkeypatch, shared_cache):
        fake = FakeOBB({"good": _bars()})
        handler = make_handler(monkeypatch, fake, shared_cache, ["good"])
        handler.fetch_stock_data("AAPL", START, END)

        info = handler.get_last_fetch_info("AAPL")

        assert set(info.keys()) == CONTRACT_KEYS
        assert isinstance(info["fetched_at"], str)
        assert "T" in info["fetched_at"]  # ISO-8601 datetime
        assert info["start_date"] == START
        assert info["end_date"] == END

    def test_failures_populated_when_provider_raises(
            self, monkeypatch, shared_cache):
        fake = FakeOBB({"bad": ValueError("boom"), "good": _bars()})
        handler = make_handler(monkeypatch, fake, shared_cache,
                               ["bad", "good"])

        df = handler.fetch_stock_data("AAPL", START, END)

        assert not df.empty
        info = handler.get_last_fetch_info("AAPL")
        assert info["provider"] == "good"
        assert info["failures"] == [{"provider": "bad", "error": "boom"}]

    def test_all_providers_fail(self, monkeypatch, shared_cache):
        fake = FakeOBB({"bad1": ValueError("boom1"),
                        "bad2": RuntimeError("boom2")})
        handler = make_handler(monkeypatch, fake, shared_cache,
                               ["bad1", "bad2"])

        df = handler.fetch_stock_data("AAPL", START, END)

        assert df.empty
        info = handler.get_last_fetch_info("AAPL")
        assert info["provider"] is None
        assert info["from_cache"] is False
        assert info["failures"] == [
            {"provider": "bad1", "error": "boom1"},
            {"provider": "bad2", "error": "boom2"},
        ]

    def test_never_fetched_returns_none(self, monkeypatch, shared_cache):
        fake = FakeOBB({})
        handler = make_handler(monkeypatch, fake, shared_cache, [])

        assert handler.get_last_fetch_info("NEVER") is None
