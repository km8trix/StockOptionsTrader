"""Tests for utils/market_hours.py — the static 2026/2027 NYSE calendar.

Offline and deterministic: pure date arithmetic, no clock reads, no
network. Every expected date below was derived from the NYSE rules
(third-Monday floats, last-Monday Memorial Day, Good Friday from Easter,
weekend-observance shifts) and is asserted literally.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from utils.market_hours import NYSE_TZ, MarketHours

NY = ZoneInfo("America/New_York")
UTC = timezone.utc


@pytest.fixture()
def mh():
    return MarketHours()


def ny(y, m, d, hh=12, mm=0, ss=0):
    """Aware America/New_York datetime."""
    return datetime(y, m, d, hh, mm, ss, tzinfo=NY)


# ==========================================================================
# Holiday calendar (full-day closes), including observed-shift cases
# ==========================================================================

FULL_HOLIDAYS_2026 = [
    (date(2026, 1, 1), "New Year's Day"),
    (date(2026, 1, 19), 'MLK Day (3rd Monday)'),
    (date(2026, 2, 16), "Washington's Birthday (3rd Monday)"),
    (date(2026, 4, 3), 'Good Friday'),
    (date(2026, 5, 25), 'Memorial Day (last Monday)'),
    (date(2026, 6, 19), 'Juneteenth'),
    (date(2026, 7, 3), 'Independence Day observed (Jul 4 = Saturday)'),
    (date(2026, 9, 7), 'Labor Day (1st Monday)'),
    (date(2026, 11, 26), 'Thanksgiving (4th Thursday)'),
    (date(2026, 12, 25), 'Christmas'),
]

FULL_HOLIDAYS_2027 = [
    (date(2027, 1, 1), "New Year's Day"),
    (date(2027, 1, 18), 'MLK Day (3rd Monday)'),
    (date(2027, 2, 15), "Washington's Birthday (3rd Monday)"),
    (date(2027, 3, 26), 'Good Friday'),
    (date(2027, 5, 31), 'Memorial Day (last Monday)'),
    (date(2027, 6, 18), 'Juneteenth observed (Jun 19 = Saturday)'),
    (date(2027, 7, 5), 'Independence Day observed (Jul 4 = Sunday)'),
    (date(2027, 9, 6), 'Labor Day (1st Monday)'),
    (date(2027, 11, 25), 'Thanksgiving (4th Thursday)'),
    (date(2027, 12, 24), 'Christmas observed (Dec 25 = Saturday)'),
]


class TestHolidayCalendar:
    @pytest.mark.parametrize(
        'holiday,label', FULL_HOLIDAYS_2026 + FULL_HOLIDAYS_2027,
        ids=lambda v: str(v))
    def test_full_holidays_closed_all_day(self, mh, holiday, label):
        noon = datetime.combine(holiday, time(12, 0), tzinfo=NY)
        assert mh.is_market_open(noon) is False, label
        assert mh.session_close(holiday) is None, label

    def test_exactly_ten_holidays_per_year(self, mh):
        """One row per NYSE holiday — a duplicate or a miss both fail."""
        from utils.market_hours import _FULL_HOLIDAYS
        for year, expected in ((2026, FULL_HOLIDAYS_2026),
                               (2027, FULL_HOLIDAYS_2027)):
            got = sorted(d for d in _FULL_HOLIDAYS if d.year == year)
            assert got == sorted(d for d, _ in expected)

    @pytest.mark.parametrize('observed,actual', [
        # Observed-shift cases: the OBSERVED weekday closes; the actual
        # weekend date needs no holiday row (weekends are closed anyway).
        (date(2026, 7, 3), date(2026, 7, 4)),    # Sat -> prior Friday
        (date(2027, 6, 18), date(2027, 6, 19)),  # Sat -> prior Friday
        (date(2027, 12, 24), date(2027, 12, 25)),  # Sat -> prior Friday
        (date(2027, 7, 5), date(2027, 7, 4)),    # Sun -> following Monday
    ])
    def test_observed_shift_closes_the_weekday(self, mh, observed, actual):
        assert mh.is_market_open(
            datetime.combine(observed, time(12, 0), tzinfo=NY)) is False
        assert mh.session_close(observed) is None
        # The weekend date itself is closed too (it is a weekend).
        assert mh.session_close(actual) is None

    def test_business_days_around_observed_holidays_stay_open(self, mh):
        # Thursday before the observed July 4th 2026, Tuesday after the
        # observed July 4th 2027: regular full sessions.
        assert mh.is_market_open(ny(2026, 7, 2)) is True
        assert mh.is_market_open(ny(2027, 7, 6)) is True


# ==========================================================================
# Half days (13:00 ET close)
# ==========================================================================

class TestHalfDays:
    @pytest.mark.parametrize('half_day', [
        date(2026, 11, 27),  # day after Thanksgiving 2026
        date(2026, 12, 24),  # Christmas Eve 2026 (Thursday)
        date(2027, 11, 26),  # day after Thanksgiving 2027
    ])
    def test_half_day_closes_at_1pm(self, mh, half_day):
        assert mh.session_close(half_day) == datetime.combine(
            half_day, time(13, 0), tzinfo=NY)
        # Trading at 12:59:59, closed at 13:00:00 sharp.
        assert mh.is_market_open(
            datetime.combine(half_day, time(12, 59, 59), tzinfo=NY)) is True
        assert mh.is_market_open(
            datetime.combine(half_day, time(13, 0), tzinfo=NY)) is False

    def test_christmas_eve_2027_is_a_full_holiday_not_a_half_day(self, mh):
        """Dec 24 2027 is the OBSERVED Christmas (Dec 25 = Saturday): the
        full-day close takes precedence over the Christmas Eve half day."""
        assert mh.session_close(date(2027, 12, 24)) is None
        assert mh.is_market_open(ny(2027, 12, 24, 10, 0)) is False

    def test_regular_day_closes_at_4pm(self, mh):
        d = date(2026, 6, 10)  # a plain Wednesday
        assert mh.session_close(d) == datetime.combine(
            d, time(16, 0), tzinfo=NY)


# ==========================================================================
# Session boundaries, weekends, DST transitions
# ==========================================================================

class TestSessionBoundaries:
    def test_open_inclusive_close_exclusive(self, mh):
        # Wednesday 2026-06-10.
        assert mh.is_market_open(ny(2026, 6, 10, 9, 29, 59)) is False
        assert mh.is_market_open(ny(2026, 6, 10, 9, 30, 0)) is True
        assert mh.is_market_open(ny(2026, 6, 10, 15, 59, 59)) is True
        assert mh.is_market_open(ny(2026, 6, 10, 16, 0, 0)) is False

    @pytest.mark.parametrize('weekend_day', [
        date(2026, 6, 13),  # Saturday
        date(2026, 6, 14),  # Sunday
    ])
    def test_weekend_closed(self, mh, weekend_day):
        noon = datetime.combine(weekend_day, time(12, 0), tzinfo=NY)
        assert mh.is_market_open(noon) is False
        assert mh.session_close(weekend_day) is None

    def test_spring_forward_2026_boundary(self, mh):
        """2026-03-08 is the EST->EDT transition (a Sunday). The Friday
        before opens at 14:30 UTC (EST, UTC-5); the Monday after opens at
        13:30 UTC (EDT, UTC-4) — asserted from UTC so a fixed-offset bug
        cannot pass."""
        # Friday 2026-03-06 (EST): 09:30 ET == 14:30 UTC.
        assert mh.is_market_open(datetime(2026, 3, 6, 14, 30, tzinfo=UTC))
        assert not mh.is_market_open(
            datetime(2026, 3, 6, 14, 29, 59, tzinfo=UTC))
        # Monday 2026-03-09 (EDT): 09:30 ET == 13:30 UTC.
        assert mh.is_market_open(datetime(2026, 3, 9, 13, 30, tzinfo=UTC))
        assert not mh.is_market_open(
            datetime(2026, 3, 9, 13, 29, 59, tzinfo=UTC))
        # Transition Sunday itself: closed, and next open is Monday 09:30
        # EDT with a -4h UTC offset.
        sunday = datetime(2026, 3, 8, 12, 0, tzinfo=NY)
        assert mh.is_market_open(sunday) is False
        nxt = mh.next_market_open(sunday)
        assert nxt == ny(2026, 3, 9, 9, 30)
        assert nxt.utcoffset() == timedelta(hours=-4)

    def test_fall_back_2026_boundary(self, mh):
        """2026-11-01 is the EDT->EST transition (a Sunday). Friday before
        opens 13:30 UTC (EDT); Monday after opens 14:30 UTC (EST)."""
        # Friday 2026-10-30 (EDT): 09:30 ET == 13:30 UTC.
        assert mh.is_market_open(datetime(2026, 10, 30, 13, 30, tzinfo=UTC))
        # Monday 2026-11-02 (EST): 09:30 ET == 14:30 UTC; 13:30 UTC is
        # 08:30 ET — pre-open.
        assert mh.is_market_open(datetime(2026, 11, 2, 14, 30, tzinfo=UTC))
        assert not mh.is_market_open(
            datetime(2026, 11, 2, 13, 30, tzinfo=UTC))
        nxt = mh.next_market_open(datetime(2026, 11, 1, 12, 0, tzinfo=NY))
        assert nxt == ny(2026, 11, 2, 9, 30)
        assert nxt.utcoffset() == timedelta(hours=-5)

    def test_close_times_straddling_fall_back_week(self, mh):
        """Friday before and Monday after the fall-back keep their 16:00
        WALL-CLOCK close; the UTC instants differ by the DST shift."""
        fri = mh.session_close(date(2026, 10, 30))
        mon = mh.session_close(date(2026, 11, 2))
        assert fri.astimezone(UTC).hour == 20  # 16:00 EDT == 20:00 UTC
        assert mon.astimezone(UTC).hour == 21  # 16:00 EST == 21:00 UTC


# ==========================================================================
# next_market_open
# ==========================================================================

class TestNextMarketOpen:
    def test_before_open_same_day(self, mh):
        assert mh.next_market_open(ny(2026, 6, 10, 8, 0)) == \
            ny(2026, 6, 10, 9, 30)

    def test_during_session_returns_next_day(self, mh):
        assert mh.next_market_open(ny(2026, 6, 10, 12, 0)) == \
            ny(2026, 6, 11, 9, 30)

    def test_at_exact_open_returns_next_session(self, mh):
        """Strictly after: 09:30 itself is already open."""
        assert mh.next_market_open(ny(2026, 6, 10, 9, 30)) == \
            ny(2026, 6, 11, 9, 30)

    def test_friday_evening_skips_weekend(self, mh):
        assert mh.next_market_open(ny(2026, 6, 12, 18, 0)) == \
            ny(2026, 6, 15, 9, 30)

    def test_skips_thanksgiving_to_the_half_day(self, mh):
        """Wednesday evening before Thanksgiving 2026 -> Friday 09:30 (the
        half day still OPENS at 09:30; only the close moves)."""
        assert mh.next_market_open(ny(2026, 11, 25, 17, 0)) == \
            ny(2026, 11, 27, 9, 30)

    def test_skips_long_july_4th_weekend_2026(self, mh):
        """Thu 2026-07-02 after close: Fri Jul 3 is the observed holiday,
        Sat/Sun are the weekend -> Monday Jul 6."""
        assert mh.next_market_open(ny(2026, 7, 2, 16, 30)) == \
            ny(2026, 7, 6, 9, 30)


# ==========================================================================
# Guard rails: naive datetimes and calendar coverage
# ==========================================================================

class TestGuardRails:
    NAIVE = datetime(2026, 6, 10, 12, 0)

    def test_is_market_open_rejects_naive(self, mh):
        with pytest.raises(ValueError, match='aware'):
            mh.is_market_open(self.NAIVE)

    def test_next_market_open_rejects_naive(self, mh):
        with pytest.raises(ValueError, match='aware'):
            mh.next_market_open(self.NAIVE)

    def test_session_close_rejects_naive_datetime(self, mh):
        with pytest.raises(ValueError, match='aware'):
            mh.session_close(self.NAIVE)

    def test_session_close_accepts_aware_datetime_and_plain_date(self, mh):
        d = date(2026, 6, 10)
        from_date = mh.session_close(d)
        from_dt = mh.session_close(ny(2026, 6, 10, 11, 0))
        assert from_date == from_dt == datetime.combine(
            d, time(16, 0), tzinfo=NYSE_TZ)

    @pytest.mark.parametrize('outside', [
        datetime(2025, 12, 31, 12, 0, tzinfo=NY),
        datetime(2028, 1, 3, 12, 0, tzinfo=NY),
    ])
    def test_years_outside_static_calendar_raise(self, mh, outside):
        """Fail LOUD outside 2026/2027 — never silently report 'open'."""
        with pytest.raises(ValueError, match='static'):
            mh.is_market_open(outside)
        with pytest.raises(ValueError, match='static'):
            mh.session_close(outside.date())

    def test_next_open_walking_past_calendar_end_raises(self, mh):
        """Friday 2027-12-31 after close: the next session is in 2028,
        beyond the static calendar -> ValueError, not a wrong answer."""
        with pytest.raises(ValueError, match='static'):
            mh.next_market_open(ny(2027, 12, 31, 18, 0))
