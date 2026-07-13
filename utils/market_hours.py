"""
NYSE market-hours calendar for the live scheduler (Phase 9 follow-up).

Regular session: 09:30-16:00 America/New_York, Monday-Friday. The holiday
calendar is STATIC and covers 2026 and 2027 ONLY — extend it annually by
appending the next year's rows to _FULL_HOLIDAYS / _HALF_DAYS below (the
dates are spelled out literally so an extension is a reviewable diff, not
a rules engine). Any query outside the covered years raises ValueError:
an unknown holiday calendar must fail loudly, never silently report the
market open.

Holidays follow NYSE observance rules: a holiday falling on Saturday is
observed the preceding Friday, one falling on Sunday the following Monday
(e.g. Independence Day 2026 -> Friday 2026-07-03; Juneteenth 2027 ->
Friday 2027-06-18; Christmas 2027 -> Friday 2027-12-24). Half days
(13:00 ET close) are the day after Thanksgiving, plus Christmas Eve when
it is a weekday and not itself the observed Christmas holiday.

ALL datetime inputs must be timezone-aware; naive datetimes raise
ValueError. Comparisons are done in America/New_York via zoneinfo, so
DST transitions (2026-03-08 spring-forward, 2026-11-01 fall-back) are
handled by the tz database, not by fixed UTC offsets.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta
from typing import Optional, Union
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

#: Exchange timezone — every session boundary is defined in this zone.
NYSE_TZ = ZoneInfo("America/New_York")

#: Years the static calendar below covers. EXTEND ANNUALLY.
SUPPORTED_YEARS = (2026, 2027)

# Full-day closes (observed dates — weekend holidays already shifted):
#   New Year's Day, MLK Day, Washington's Birthday, Good Friday,
#   Memorial Day, Juneteenth, Independence Day, Labor Day, Thanksgiving,
#   Christmas.
_FULL_HOLIDAYS = frozenset({
    # ---- 2026 ----
    date(2026, 1, 1),    # New Year's Day (Thursday)
    date(2026, 1, 19),   # MLK Day (3rd Monday of January)
    date(2026, 2, 16),   # Washington's Birthday (3rd Monday of February)
    date(2026, 4, 3),    # Good Friday (Easter 2026-04-05)
    date(2026, 5, 25),   # Memorial Day (last Monday of May)
    date(2026, 6, 19),   # Juneteenth (Friday)
    date(2026, 7, 3),    # Independence Day OBSERVED (Jul 4 is a Saturday)
    date(2026, 9, 7),    # Labor Day (1st Monday of September)
    date(2026, 11, 26),  # Thanksgiving (4th Thursday of November)
    date(2026, 12, 25),  # Christmas (Friday)
    # ---- 2027 ----
    date(2027, 1, 1),    # New Year's Day (Friday)
    date(2027, 1, 18),   # MLK Day (3rd Monday of January)
    date(2027, 2, 15),   # Washington's Birthday (3rd Monday of February)
    date(2027, 3, 26),   # Good Friday (Easter 2027-03-28)
    date(2027, 5, 31),   # Memorial Day (last Monday of May)
    date(2027, 6, 18),   # Juneteenth OBSERVED (Jun 19 is a Saturday)
    date(2027, 7, 5),    # Independence Day OBSERVED (Jul 4 is a Sunday)
    date(2027, 9, 6),    # Labor Day (1st Monday of September)
    date(2027, 11, 25),  # Thanksgiving (4th Thursday of November)
    date(2027, 12, 24),  # Christmas OBSERVED (Dec 25 is a Saturday)
})

# 13:00 ET closes. NOTE 2027-12-24 is NOT here: Christmas Eve 2027 is the
# observed Christmas FULL holiday, which takes precedence over a half day.
_HALF_DAYS = frozenset({
    # ---- 2026 ----
    date(2026, 11, 27),  # day after Thanksgiving (Friday)
    date(2026, 12, 24),  # Christmas Eve (Thursday)
    # ---- 2027 ----
    date(2027, 11, 26),  # day after Thanksgiving (Friday)
})


class MarketHours:
    """NYSE regular-session calendar; static through 2027 (extend annually).

    API (all tz-aware; naive datetimes raise ValueError):
        is_market_open(dt)    -> bool
        next_market_open(dt)  -> datetime (America/New_York)
        session_close(d)      -> datetime | None (None = no session that day)
    """

    OPEN_TIME = time(9, 30)
    CLOSE_TIME = time(16, 0)
    HALF_DAY_CLOSE_TIME = time(13, 0)

    # ------------------------------------------------------------------
    @staticmethod
    def _require_aware(dt: datetime) -> datetime:
        if not isinstance(dt, datetime):
            raise ValueError(f"expected a datetime, got {type(dt).__name__}")
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            raise ValueError(
                "naive datetime passed to MarketHours — every input must be "
                "timezone-aware (use zoneinfo / datetime.now(timezone.utc))")
        return dt.astimezone(NYSE_TZ)

    @staticmethod
    def _require_supported_year(d: date) -> None:
        if d.year not in SUPPORTED_YEARS:
            raise ValueError(
                f"MarketHours calendar is static through "
                f"{max(SUPPORTED_YEARS)} and does not cover {d.year} — "
                f"extend utils/market_hours.py with the next year's NYSE "
                f"holidays")

    def _coerce_date(self, d: Union[date, datetime]) -> date:
        """Accept a date, or an AWARE datetime (converted to NY first)."""
        if isinstance(d, datetime):
            return self._require_aware(d).date()
        if isinstance(d, date):
            return d
        raise ValueError(f"expected a date or datetime, "
                         f"got {type(d).__name__}")

    # ------------------------------------------------------------------
    def is_trading_day(self, d: Union[date, datetime]) -> bool:
        """True when the NYSE holds a session (full or half) on day ``d``."""
        d = self._coerce_date(d)
        self._require_supported_year(d)
        return d.weekday() < 5 and d not in _FULL_HOLIDAYS

    def is_half_day(self, d: Union[date, datetime]) -> bool:
        """True for 13:00-ET-close sessions."""
        d = self._coerce_date(d)
        self._require_supported_year(d)
        return d in _HALF_DAYS

    def session_open(self, d: Union[date, datetime]) -> Optional[datetime]:
        """Aware 09:30 ET open for day ``d``, or None when closed all day."""
        d = self._coerce_date(d)
        if not self.is_trading_day(d):
            return None
        return datetime.combine(d, self.OPEN_TIME, tzinfo=NYSE_TZ)

    def session_close(self, d: Union[date, datetime]) -> Optional[datetime]:
        """Aware close (16:00 ET, or 13:00 ET on half days) for day ``d``.

        None when there is no session that day (weekend / full holiday).
        """
        d = self._coerce_date(d)
        if not self.is_trading_day(d):
            return None
        close = (self.HALF_DAY_CLOSE_TIME if d in _HALF_DAYS
                 else self.CLOSE_TIME)
        return datetime.combine(d, close, tzinfo=NYSE_TZ)

    # ------------------------------------------------------------------
    def is_market_open(self, dt: datetime) -> bool:
        """True while the regular session is trading at aware instant ``dt``.

        Open is inclusive (09:30:00 counts), close exclusive (16:00:00 /
        13:00:00 does not).
        """
        local = self._require_aware(dt)
        if not self.is_trading_day(local.date()):
            return False
        open_dt = self.session_open(local.date())
        close_dt = self.session_close(local.date())
        # is_trading_day() above guarantees both session boundaries exist.
        assert open_dt is not None and close_dt is not None
        return open_dt <= local < close_dt

    def next_market_open(self, dt: datetime) -> datetime:
        """The next session open STRICTLY after aware instant ``dt``.

        If ``dt`` is before today's open on a trading day, that open is
        returned; otherwise the search walks forward day by day (raising
        ValueError if it walks past the static calendar's last year).
        """
        local = self._require_aware(dt)
        day = local.date()
        # Bounded walk: the longest NYSE gap is a long weekend (4 days);
        # 10 days is generous and keeps a broken calendar from spinning.
        for _ in range(10):
            self._require_supported_year(day)
            if day.weekday() < 5 and day not in _FULL_HOLIDAYS:
                open_dt = datetime.combine(day, self.OPEN_TIME,
                                           tzinfo=NYSE_TZ)
                if open_dt > local:
                    return open_dt
            day += timedelta(days=1)
        raise ValueError(
            f"no market open found within 10 days of {local.isoformat()} — "
            f"the static calendar looks wrong")
