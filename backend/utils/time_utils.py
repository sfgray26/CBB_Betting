"""Time utilities — all game/metric dates use ET, never UTC."""

from datetime import datetime, date
from zoneinfo import ZoneInfo


def now_et() -> datetime:
    """Return current datetime in ET (America/New_York)."""
    return datetime.now(ZoneInfo("America/New_York"))


def today_et() -> date:
    """Return today's date in ET."""
    return now_et().date()
