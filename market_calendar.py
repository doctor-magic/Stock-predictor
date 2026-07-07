"""US market session guard — weekend + NYSE full-day holidays, anchored to ET.

Zero dependencies (stdlib only) so any cron entrypoint OR db.py can import it
with no coupling / no circular-import risk. Anchored to America/New_York so the
trading-date check matches db._signal_date() — this avoids the Asia/Jerusalem
date-rollover bug (a 20:30 UTC cron is already 23:30 in Israel; we must judge the
US trading day, not the server's local date).

Used to stop crons (and endpoint-driven setup_log writes) from logging phantom
rows on days the US market is closed — see the Juneteenth-2026 pollution incident.
"""
import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")

# NYSE FULL-DAY closures for 2026. Half-days (2026-11-27, 2026-12-24) are NOT
# here on purpose — the market is open, so crons SHOULD run those days.
NYSE_HOLIDAYS_2026 = frozenset({
    "2026-01-01",  # New Year's Day
    "2026-01-19",  # MLK Jr. Day
    "2026-02-16",  # Washington's Birthday
    "2026-04-03",  # Good Friday
    "2026-05-25",  # Memorial Day
    "2026-06-19",  # Juneteenth
    "2026-07-03",  # Independence Day (observed — Jul 4 is a Saturday)
    "2026-09-07",  # Labor Day
    "2026-11-26",  # Thanksgiving
    "2026-12-25",  # Christmas
})

_COVERED_YEARS = frozenset({2026})

# NYSE HALF-DAYS (13:00 ET close). The market IS open — is_us_market_session()
# returns True — but time-windowed logic (Power Hour, HOD gate) must derive its
# windows from the ACTUAL close, not a hardcoded 16:00.
EARLY_CLOSES_2026 = frozenset({
    "2026-11-27",  # day after Thanksgiving
    "2026-12-24",  # Christmas Eve
})


def us_trading_date() -> date:
    """Today's date in US-Eastern — same anchor as db._signal_date()."""
    return datetime.now(_ET).date()


def is_us_market_session(d: "date | None" = None) -> bool:
    """True iff `d` (default: today in ET) is a NYSE full trading day.

    Weekends and 2026 NYSE holidays -> False. If `d` falls in a year whose
    holiday table is not loaded, we still apply the weekend filter and warn
    loudly (so the table gets extended) rather than silently mislabel a holiday.
    """
    if d is None:
        d = us_trading_date()
    if d.weekday() >= 5:                      # Sat=5, Sun=6
        return False
    if d.year not in _COVERED_YEARS:
        print(f"[market_calendar] WARNING: {d.year} NYSE holidays not in table — "
              f"extend NYSE_HOLIDAYS (weekend filter still applied)", file=sys.stderr, flush=True)
    return d.isoformat() not in NYSE_HOLIDAYS_2026


def session_close_hour(d: "date | None" = None) -> int:
    """ET hour the session ends (exclusive): 16 on full days, 13 on half-days.

    Callers deriving intraday windows (Power Hour = last hour before close,
    HOD gate = 10:00 → close) must use this instead of a hardcoded 16.
    """
    if d is None:
        d = us_trading_date()
    return 13 if d.isoformat() in EARLY_CLOSES_2026 else 16


def has_session_opened(now: "datetime | None" = None) -> bool:
    """True from 09:30 ET on a trading day until midnight ET. False pre-open.

    Guards signal LOGGING against the pre-open window (midnight→09:30 ET):
    the ET date has already rolled to today, but Yahoo screeners still serve
    the PREVIOUS session's quotes — logging then stamps yesterday's data with
    today's date (the Jul-7-2026 QNT incident: 25 stale pre-open rows, which
    then BLOCK the real intraday rows via UNIQUE(source,symbol,date)).
    After the close (16:00→24:00 ET) logging stays allowed — quotes then ARE
    today's data; the per-day dedup prevents duplicates.
    """
    if now is None:
        now = datetime.now(_ET)
    if not is_us_market_session(now.date()):
        return False
    session_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    return now >= session_open
