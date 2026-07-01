"""Event calendar used as model features.

The build covers a Pakistan-centric marketplace, so the calendar models the
events that actually move retail demand here: Ramadan, the two Eids, national
festivals, and disaster windows.

Hijri dates drift ~11 days earlier each Gregorian year and depend on moon
sighting, so Ramadan/Eid windows are hard-coded per year (approximate, ±1 day).
Festivals are fixed Gregorian dates. Disasters can't be forecast, so they enter
the model as a context flag the caller sets (flood / earthquake / heatwave).
"""

from __future__ import annotations

from datetime import date

# Ramadan window (inclusive) per Gregorian year — approximate moon-sighting dates.
RAMADAN = {
    2023: (date(2023, 3, 23), date(2023, 4, 20)),
    2024: (date(2024, 3, 11), date(2024, 4, 9)),
    2025: (date(2025, 3, 1), date(2025, 3, 30)),
    2026: (date(2026, 2, 18), date(2026, 3, 19)),
    2027: (date(2027, 2, 8), date(2027, 3, 9)),
}

# Eid windows (the 2-3 high-spend days around each Eid).
EID_FITR = {
    2023: (date(2023, 4, 21), date(2023, 4, 23)),
    2024: (date(2024, 4, 10), date(2024, 4, 12)),
    2025: (date(2025, 3, 31), date(2025, 4, 2)),
    2026: (date(2026, 3, 20), date(2026, 3, 22)),
    2027: (date(2027, 3, 10), date(2027, 3, 12)),
}
EID_ADHA = {
    2023: (date(2023, 6, 28), date(2023, 6, 30)),
    2024: (date(2024, 6, 16), date(2024, 6, 18)),
    2025: (date(2025, 6, 6), date(2025, 6, 8)),
    2026: (date(2026, 5, 27), date(2026, 5, 29)),
    2027: (date(2027, 5, 16), date(2027, 5, 18)),
}

# Fixed-date national / cultural festivals (month, day) -> label.
FESTIVALS = {
    (1, 1): "new_year",
    (2, 5): "kashmir_day",
    (3, 23): "pakistan_day",
    (5, 1): "labour_day",
    (8, 14): "independence_day",
    (9, 6): "defence_day",
    (11, 9): "iqbal_day",
    (12, 25): "quaid_day",  # also Christmas
}


def _within(d: date, window) -> bool:
    start, end = window
    return start <= d <= end


def is_ramadan(d: date) -> bool:
    win = RAMADAN.get(d.year)
    return bool(win and _within(d, win))


def eid_type(d: date) -> str:
    """Returns 'fitr', 'adha', or 'none'."""
    if d.year in EID_FITR and _within(d, EID_FITR[d.year]):
        return "fitr"
    if d.year in EID_ADHA and _within(d, EID_ADHA[d.year]):
        return "adha"
    return "none"


def festival_of(d: date) -> str:
    """Returns the festival label for the date, or 'none'.

    Basant (spring kite festival, Punjab) is an approximate mid-February window.
    """
    label = FESTIVALS.get((d.month, d.day))
    if label:
        return label
    if d.month == 2 and 5 <= d.day <= 14:
        return "basant"
    return "none"


def event_label(d: date, disaster: str = "none") -> str:
    """Single dominant event label for a date (used for grouping / display).

    Priority: disaster > Eid > Ramadan > festival > none. Disasters are caller
    supplied because they can't be predicted from the calendar.
    """
    if disaster and disaster != "none":
        return f"disaster_{disaster}"
    et = eid_type(d)
    if et != "none":
        return f"eid_{et}"
    if is_ramadan(d):
        return "ramadan"
    fest = festival_of(d)
    if fest != "none":
        return "festival"
    return "none"
