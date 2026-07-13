"""OHLCVCache: round-trips, coverage boundaries, and the staleness policy.

Offline and deterministic: tmp_path SQLite files everywhere, and the
cache clock is injected (now_fn) so time never actually passes.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import pytest

from data.cache import OHLCVCache

FIXED_NOW = datetime(2026, 6, 11, 10, 0, 0)


class FakeClock:
    """Mutable injectable clock for OHLCVCache(now_fn=...)."""

    def __init__(self, now: datetime = FIXED_NOW):
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now = self.now + timedelta(**kwargs)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def cache(tmp_path, clock):
    return OHLCVCache(db_path=str(tmp_path / "cache.db"), now_fn=clock)


class TestRoundTrip:
    def test_store_get_preserves_values_and_index(self, cache, make_ohlcv):
        df = make_ohlcv(n_days=10, start="2023-01-02")
        cache.store("AAPL", df, "fmp", "2023-01-01", "2023-01-14")

        result = cache.get("AAPL", "2023-01-01", "2023-01-14")

        assert result is not None
        expected = df.copy()
        expected.index.name = "date"
        pd.testing.assert_frame_equal(result, expected, check_freq=False)
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.name == "date"
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]

    def test_sub_range_is_served(self, cache, make_ohlcv):
        df = make_ohlcv(n_days=10, start="2023-01-02")
        cache.store("AAPL", df, "fmp", "2023-01-01", "2023-01-14")

        result = cache.get("AAPL", "2023-01-04", "2023-01-10")

        assert result is not None
        assert result.index.min() >= pd.Timestamp("2023-01-04")
        assert result.index.max() <= pd.Timestamp("2023-01-10")
        assert len(result) == 5  # business days Jan 4..10 2023

    def test_upsert_overwrites_existing_rows(self, cache, make_ohlcv):
        df = make_ohlcv(n_days=10, start="2023-01-02")
        cache.store("AAPL", df, "fmp", "2023-01-01", "2023-01-14")
        revised = df * 2.0
        cache.store("AAPL", revised, "polygon", "2023-01-01", "2023-01-14")

        result = cache.get("AAPL", "2023-01-01", "2023-01-14")

        assert result["close"].iloc[0] == pytest.approx(df["close"].iloc[0] * 2.0)
        assert len(result) == len(df)  # replaced, not duplicated

    def test_symbols_are_isolated(self, cache, make_ohlcv):
        df = make_ohlcv(n_days=10, start="2023-01-02")
        cache.store("AAPL", df, "fmp", "2023-01-01", "2023-01-14")

        assert cache.get("MSFT", "2023-01-01", "2023-01-14") is None


class TestCoverageBoundaries:
    def test_exact_boundary_dates_hit(self, cache, make_ohlcv):
        df = make_ohlcv(n_days=10, start="2023-01-02")
        cache.store("AAPL", df, "fmp", "2023-01-02", "2023-01-13")

        # start == cov_start AND end == cov_end must hit.
        assert cache.get("AAPL", "2023-01-02", "2023-01-13") is not None

    def test_end_beyond_coverage_misses(self, cache, make_ohlcv):
        df = make_ohlcv(n_days=10, start="2023-01-02")
        cache.store("AAPL", df, "fmp", "2023-01-02", "2023-01-13")

        assert cache.get("AAPL", "2023-01-02", "2023-01-14") is None

    def test_start_before_coverage_misses(self, cache, make_ohlcv):
        df = make_ohlcv(n_days=10, start="2023-01-02")
        cache.store("AAPL", df, "fmp", "2023-01-02", "2023-01-13")

        assert cache.get("AAPL", "2023-01-01", "2023-01-13") is None


class TestStalenessPolicy:
    """fetched_at is FIXED_NOW (2026-06-11 10:00) in every store below."""

    def test_historical_range_never_expires(self, cache, clock, make_ohlcv):
        # Range ends years before fetch: fully historical, immutable.
        df = make_ohlcv(n_days=10, start="2023-01-02")
        cache.store("AAPL", df, "fmp", "2023-01-01", "2023-01-14")

        clock.advance(hours=OHLCVCache.CACHE_MAX_AGE_HOURS + 100)

        assert cache.get("AAPL", "2023-01-01", "2023-01-14") is not None

    def test_recent_range_fresh_within_max_age(self, cache, clock, make_ohlcv):
        # Range ends on the fetch date itself: not historical, but fresh.
        df = make_ohlcv(n_days=9, start="2026-06-01")
        cache.store("SPY", df, "fmp", "2026-06-01", "2026-06-11")

        clock.advance(hours=OHLCVCache.CACHE_MAX_AGE_HOURS - 1)

        assert cache.get("SPY", "2026-06-01", "2026-06-11") is not None

    def test_recent_range_expires_after_max_age(self, cache, clock, make_ohlcv):
        df = make_ohlcv(n_days=9, start="2026-06-01")
        cache.store("SPY", df, "fmp", "2026-06-01", "2026-06-11")

        clock.advance(hours=OHLCVCache.CACHE_MAX_AGE_HOURS, minutes=1)

        assert cache.get("SPY", "2026-06-01", "2026-06-11") is None

    def test_two_day_immutability_boundary(self, cache, clock, make_ohlcv):
        # fetched 2026-06-11: end <= 2026-06-09 is historical (immortal),
        # end == 2026-06-10 is not and must expire with the max age.
        df = make_ohlcv(n_days=8, start="2026-06-01")
        cache.store("IMM", df, "fmp", "2026-06-01", "2026-06-09")
        cache.store("MUT", df, "fmp", "2026-06-01", "2026-06-10")

        clock.advance(hours=OHLCVCache.CACHE_MAX_AGE_HOURS, minutes=1)

        assert cache.get("IMM", "2026-06-01", "2026-06-09") is not None
        assert cache.get("MUT", "2026-06-01", "2026-06-10") is None


class TestDateCanonicalization:
    """Caller dates are parsed and re-emitted as padded ISO before any SQL
    comparison. The trap: '2023-01-02' <= '2023-1-2' is TRUE as strings, so
    a non-padded request used to QUALIFY coverage while the bar slice
    ('date' >= '2023-1-2') matched zero rows — an empty frame served as a
    valid hit. That must never happen: either correct bars or a clean miss.
    """

    def test_non_padded_request_returns_correct_bars_not_empty_hit(
            self, cache, make_ohlcv):
        df = make_ohlcv(n_days=5, start="2023-01-02")
        cache.store("AAPL", df, "fmp", "2023-01-02", "2023-01-06")

        result = cache.get("AAPL", "2023-1-2", "2023-01-06")

        # Canonicalization makes this a correct hit (5 bars), never the
        # silently-empty "covered" frame.
        assert result is not None
        assert len(result) == 5
        assert result.index.min() == pd.Timestamp("2023-01-02")
        assert result.index.max() == pd.Timestamp("2023-01-06")

    def test_non_padded_store_serves_padded_request(self, cache, make_ohlcv):
        df = make_ohlcv(n_days=5, start="2023-01-02")
        cache.store("AAPL", df, "fmp", "2023-1-2", "2023-1-6")

        result = cache.get("AAPL", "2023-01-02", "2023-01-06")

        assert result is not None
        assert len(result) == 5

    def test_unparseable_get_is_clean_miss(self, cache, make_ohlcv, caplog):
        df = make_ohlcv(n_days=5, start="2023-01-02")
        cache.store("AAPL", df, "fmp", "2023-01-02", "2023-01-06")

        with caplog.at_level(logging.WARNING, logger="data.cache"):
            result = cache.get("AAPL", "02/01/2023", "2023-01-06")

        assert result is None  # miss -> caller refetches from a provider
        assert any("unparseable" in r.message for r in caplog.records)

    def test_unparseable_store_claims_only_actual_bars(self, cache,
                                                       make_ohlcv, caplog):
        df = make_ohlcv(n_days=5, start="2023-01-02")

        with caplog.at_level(logging.WARNING, logger="data.cache"):
            cache.store("AAPL", df, "fmp", "garbage", "2023-01-06")

        assert any("unparseable" in r.message for r in caplog.records)
        # Coverage fell back to the stored bars' own range (conservative).
        assert cache.get("AAPL", "2023-01-02", "2023-01-06") is not None
        assert cache.get("AAPL", "2023-01-01", "2023-01-06") is None


class TestTruncationWarning:
    """store() must make silent provider history truncation observable."""

    def test_truncated_history_warns_with_context(self, cache, make_ohlcv,
                                                  caplog):
        # Provider returned bars starting two months after requested start.
        df = make_ohlcv(n_days=5, start="2023-03-01")

        with caplog.at_level(logging.WARNING, logger="data.cache"):
            cache.store("AAPL", df, "fmp", "2023-01-01", "2023-03-07")

        msgs = [r.message for r in caplog.records]
        assert any("truncated" in m and "AAPL" in m and "fmp" in m
                   and "2023-01-01" in m and "2023-03-01" in m for m in msgs)

    def test_no_warning_when_first_bar_near_start(self, cache, make_ohlcv,
                                                  caplog):
        # 2023-01-01 is a Sunday; first bar Jan 2 is the next business day.
        df = make_ohlcv(n_days=5, start="2023-01-02")

        with caplog.at_level(logging.WARNING, logger="data.cache"):
            cache.store("AAPL", df, "fmp", "2023-01-01", "2023-01-06")

        assert not any("truncated" in r.message for r in caplog.records)

    def test_five_business_day_boundary(self, cache, make_ohlcv, caplog):
        # Requested start Mon 2023-01-02. First bar Mon Jan 9 -> gap is
        # exactly 5 business days: no warning. First bar Tue Jan 10 -> 6
        # business days: warning.
        df_ok = make_ohlcv(n_days=3, start="2023-01-09")
        with caplog.at_level(logging.WARNING, logger="data.cache"):
            cache.store("OK", df_ok, "fmp", "2023-01-02", "2023-01-11")
        assert not any("truncated" in r.message for r in caplog.records)

        df_bad = make_ohlcv(n_days=3, start="2023-01-10")
        with caplog.at_level(logging.WARNING, logger="data.cache"):
            cache.store("BAD", df_bad, "fmp", "2023-01-02", "2023-01-12")
        assert any("truncated" in r.message and "BAD" in r.message
                   for r in caplog.records)


class TestCoverageQuality:
    """Non-empty is not synonymous with complete requested history."""

    def test_truncated_start_claims_only_observed_history(
            self, cache, make_ohlcv):
        df = make_ohlcv(n_days=5, start="2023-03-01")

        quality = cache.store(
            "AAPL", df, "fmp", "2023-01-03", "2023-03-07")

        assert quality is not None
        assert not quality.request_complete
        assert quality.reason == "missing_start"
        assert quality.observed_start.isoformat() == "2023-03-01"
        assert quality.covered_start.isoformat() == "2023-03-01"
        # The false full-range hit is gone, but the actually observed range
        # remains reusable.
        assert cache.get("AAPL", "2023-01-03", "2023-03-07") is None
        assert cache.get("AAPL", "2023-03-01", "2023-03-07") is not None

    def test_partial_end_does_not_claim_requested_end(self, cache,
                                                       make_ohlcv):
        df = make_ohlcv(n_days=4, start="2023-01-03")  # through Jan 6

        quality = cache.store(
            "AAPL", df, "fmp", "2023-01-03", "2023-01-09")

        assert quality is not None
        assert not quality.request_complete
        assert quality.reason == "missing_end"
        assert quality.covered_end.isoformat() == "2023-01-06"
        assert cache.get("AAPL", "2023-01-03", "2023-01-09") is None

    def test_internal_partial_response_claims_no_interval(self, cache,
                                                           make_ohlcv):
        df = make_ohlcv(n_days=4, start="2023-01-03")
        # Keep both requested edges but remove a market session in the middle.
        df = df.drop(pd.Timestamp("2023-01-05"))

        quality = cache.store(
            "AAPL", df, "fmp", "2023-01-03", "2023-01-06")

        assert quality is not None
        assert quality.reason == "missing_interior"
        assert quality.missing_interior_sessions == 1
        assert not quality.cache_eligible
        assert cache.get("AAPL", "2023-01-03", "2023-01-06") is None

    def test_quality_metadata_is_persisted(self, cache, make_ohlcv):
        df = make_ohlcv(n_days=3, start="2023-01-04")
        cache.store("AAPL", df, "fmp", "2023-01-03", "2023-01-06")

        conn = sqlite3.connect(cache.db_path)
        row = conn.execute('''
            SELECT requested_start_date, requested_end_date,
                   observed_start_date, observed_end_date,
                   start_date, end_date, request_complete,
                   quality_reason, quality_version
            FROM fetch_coverage
        ''').fetchone()
        conn.close()

        assert row == (
            "2023-01-03", "2023-01-06",
            "2023-01-04", "2023-01-06",
            "2023-01-04", "2023-01-06",
            0, "missing_start", 1,
        )

    @pytest.mark.parametrize(
        "requested_start,requested_end,observed",
        [
            # Weekend boundary before the Monday session.
            ("2023-01-07", "2023-01-09", "2023-01-09"),
            # Good Friday is not an expected US equity-market session.
            ("2024-03-29", "2024-04-01", "2024-04-01"),
        ],
    )
    def test_non_trading_boundaries_are_legitimate_complete_coverage(
            self, cache, make_ohlcv, requested_start, requested_end, observed):
        df = make_ohlcv(n_days=1, start=observed)

        quality = cache.store(
            "SPY", df, "fmp", requested_start, requested_end)

        assert quality is not None and quality.request_complete
        result = cache.get("SPY", requested_start, requested_end)
        assert result is not None and len(result) == 1


class TestDbPathResolution:
    def test_env_trading_db_path_honored(self, tmp_path, monkeypatch, make_ohlcv):
        env_db = tmp_path / "env_cache.db"
        monkeypatch.setenv("TRADING_DB_PATH", str(env_db))

        cache = OHLCVCache(now_fn=FakeClock())

        assert cache.db_path == str(env_db)
        assert env_db.exists()

        df = make_ohlcv(n_days=5, start="2023-01-02")
        cache.store("AAPL", df, "fmp", "2023-01-01", "2023-01-06")
        assert cache.get("AAPL", "2023-01-01", "2023-01-06") is not None

    def test_explicit_db_path_beats_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRADING_DB_PATH", str(tmp_path / "env.db"))
        explicit = tmp_path / "explicit.db"

        cache = OHLCVCache(db_path=str(explicit), now_fn=FakeClock())

        assert cache.db_path == str(explicit)
        assert explicit.exists()
