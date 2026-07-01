"""Feature engineering: turn a prediction *context* into the model's columns.

The feature set is exactly the one the product brief asked for:
age, time (hour), date-of-month, season, year, day(-of-week), events
(Ramadan / Eid / festivals / disasters), location, and product (demand).

A scikit-learn ``ColumnTransformer`` one-hot-encodes the categoricals and passes
the numerics through, so the rest of the code only ever deals with these plain
dict / DataFrame rows.
"""

from __future__ import annotations

from datetime import date, datetime

import events

# ── Vocabularies ────────────────────────────────────────────────────────────

# Product categories whose demand we forecast.
CATEGORIES = [
    "groceries",
    "food",
    "clothing",
    "electronics",
    "pharmacy",
    "services",
    "gifts",
]

# Cities/regions with a population weight (rough demand scale).
LOCATIONS = {
    "karachi": {"lat": 24.86, "lng": 67.01, "weight": 1.30},
    "lahore": {"lat": 31.55, "lng": 74.34, "weight": 1.20},
    "islamabad": {"lat": 33.68, "lng": 73.04, "weight": 1.00},
    "multan": {"lat": 30.18, "lng": 71.49, "weight": 0.90},
    "peshawar": {"lat": 34.01, "lng": 71.58, "weight": 0.85},
    "quetta": {"lat": 30.18, "lng": 66.97, "weight": 0.70},
}

AGE_GROUPS = ["youth", "adult", "middle", "senior"]
SEASONS = ["winter", "spring", "summer", "autumn"]
DISASTERS = ["none", "flood", "earthquake", "heatwave"]

# Categorical vs numeric split for the ColumnTransformer.
CATEGORICAL_FEATURES = [
    "age_group",
    "season",
    "day_of_week",
    "location",
    "category",
    "event",
    "disaster",
]
NUMERIC_FEATURES = [
    "age",
    "hour",
    "day_of_month",
    "year",
    "is_weekend",
    "pop_weight",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def season_of(month: int) -> str:
    """Pakistan-leaning seasons (summer absorbs the monsoon)."""
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8, 9):
        return "summer"
    return "autumn"


def age_group_of(age: int) -> str:
    if age < 25:
        return "youth"
    if age < 40:
        return "adult"
    if age < 60:
        return "middle"
    return "senior"


_WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def parse_date(value) -> date:
    """Accepts a ``date``, ISO string, or ``None`` (today)."""
    if value is None:
        return date.today()
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return datetime.fromisoformat(str(value)[:19]).date()


def build_features(
    *,
    when: date,
    hour: int,
    age: int,
    location: str,
    category: str,
    disaster: str = "none",
    event_override: str | None = None,
) -> dict:
    """Build one model row from a context.

    ``event_override`` lets the UI force an event (e.g. preview "what happens on
    Eid") regardless of the calendar; otherwise the event is derived from the
    date.
    """
    loc = LOCATIONS.get(location, LOCATIONS["islamabad"])
    dow = when.weekday()  # 0 = Monday
    event = event_override or events.event_label(when, disaster)
    return {
        "age": int(age),
        "age_group": age_group_of(int(age)),
        "hour": int(hour),
        "day_of_month": when.day,
        "season": season_of(when.month),
        "year": when.year,
        "day_of_week": _WEEKDAYS[dow],
        "is_weekend": 1 if dow >= 5 else 0,
        "location": location,
        "pop_weight": loc["weight"],
        "category": category,
        "event": event,
        "disaster": disaster,
    }
