"""Synthetic-but-realistic training data.

A fresh marketplace has no labelled "customer behaviour" history, so we generate
a dataset whose demand/intent obey believable domain effects — Ramadan shifts
food to the evening, Eid spikes clothing & gifts, disasters spike pharmacy and
crush discretionary spend, weekends and population scale everything, etc. The
RandomForests then *learn* those effects back from the features, so the served
predictions are genuine model output, not the hand-written rules.

If you later export real ``requests`` from RTDB, drop them in here in the same
column shape and the rest of the pipeline is unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import events
import features as F

# Base daily demand per product category (orders), before any multipliers.
BASE_DEMAND = {
    "groceries": 60,
    "food": 80,
    "clothing": 38,
    "electronics": 24,
    "pharmacy": 30,
    "services": 34,
    "gifts": 18,
}

# Seasonal multipliers per category.
SEASON_MULT = {
    "winter": {"clothing": 1.35, "food": 1.2, "pharmacy": 1.25, "electronics": 1.1},
    "spring": {"services": 1.2, "clothing": 1.1, "gifts": 1.1},
    "summer": {"pharmacy": 1.2, "groceries": 1.1, "electronics": 1.15, "food": 0.95},
    "autumn": {"clothing": 1.15, "services": 1.1},
}


def _hour_curve(hour: int, ramadan: bool) -> float:
    """Demand weight by hour of day (0..1.0-ish)."""
    if ramadan:
        # Sehri (pre-dawn) and post-iftar peaks; daytime suppressed (fasting).
        pre_dawn = np.exp(-((hour - 4) ** 2) / 4.0)
        evening = np.exp(-((hour - 21) ** 2) / 6.0)
        daytime = 0.15 if 7 <= hour <= 16 else 0.3
        return float(max(pre_dawn, evening, daytime))
    lunch = np.exp(-((hour - 13) ** 2) / 6.0)
    evening = np.exp(-((hour - 20) ** 2) / 8.0)
    return float(max(lunch, evening, 0.12))


def _event_mult(event: str, category: str) -> float:
    """Demand multiplier for an event × category pair."""
    table = {
        "ramadan": {
            "food": 1.6, "groceries": 1.4, "clothing": 1.3, "gifts": 1.2,
            "electronics": 0.9, "services": 0.9,
        },
        "eid_fitr": {
            "clothing": 2.3, "gifts": 2.1, "food": 1.8, "groceries": 1.5,
            "services": 1.2, "electronics": 1.3,
        },
        "eid_adha": {
            "food": 2.0, "services": 1.5, "clothing": 1.4, "groceries": 1.3,
            "gifts": 1.3,
        },
        "festival": {
            "clothing": 1.3, "food": 1.2, "gifts": 1.25, "services": 1.05,
        },
    }
    if event.startswith("disaster_"):
        return {
            "pharmacy": 2.1, "groceries": 1.7, "food": 1.35, "services": 0.7,
            "clothing": 0.55, "electronics": 0.4, "gifts": 0.3,
        }.get(category, 0.8)
    return table.get(event, {}).get(category, 1.0)


# Baseline purchase appetite per category (additive intent bias).
CATEGORY_PREF = {
    "food": 0.25,
    "groceries": 0.18,
    "pharmacy": 0.12,
    "clothing": 0.08,
    "services": 0.05,
    "electronics": 0.0,
    "gifts": -0.02,
}


def _age_intent(age_group: str, category: str, hour: int) -> float:
    """Additive purchase-intent bias by shopper age × context."""
    bias = 0.0
    if age_group == "youth":
        bias += 0.10 + (0.12 if hour >= 21 else 0.0)
        bias += 0.08 if category in ("food", "clothing", "electronics") else 0.0
    elif age_group == "adult":
        bias += 0.12
        bias += 0.06 if category in ("groceries", "food", "gifts") else 0.0
    elif age_group == "middle":
        bias += 0.08 + (0.06 if 9 <= hour <= 18 else 0.0)
        bias += 0.05 if category in ("groceries", "services") else 0.0
    else:  # senior
        bias += (0.10 if 8 <= hour <= 17 else -0.05)
        bias += 0.12 if category == "pharmacy" else 0.0
    return bias


def generate(n: int = 24000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    years = [2023, 2024, 2025, 2026]
    rows = []
    for _ in range(n):
        year = int(rng.choice(years))
        month = int(rng.integers(1, 13))
        day = int(rng.integers(1, 28))  # clamp to 28 so every month is valid
        from datetime import date as _date

        when = _date(year, month, day)
        hour = int(rng.integers(0, 24))
        age = int(np.clip(rng.normal(36, 14), 16, 78))
        location = str(rng.choice(list(F.LOCATIONS.keys())))
        category = str(rng.choice(F.CATEGORIES))

        # Inject occasional disaster windows so the model learns their effect.
        disaster = "none"
        if rng.random() < 0.04:
            disaster = str(rng.choice(["flood", "earthquake", "heatwave"]))

        row = F.build_features(
            when=when, hour=hour, age=age, location=location,
            category=category, disaster=disaster,
        )
        ramadan = events.is_ramadan(when)

        base = BASE_DEMAND[category]
        season_mult = SEASON_MULT.get(row["season"], {}).get(category, 1.0)
        weekend_mult = 1.0 + (0.22 if row["is_weekend"] else 0.0)
        hour_mult = 0.5 + 1.3 * _hour_curve(hour, ramadan)
        ev_mult = _event_mult(row["event"], category)

        demand = (
            base
            * row["pop_weight"]
            * season_mult
            * weekend_mult
            * hour_mult
            * ev_mult
        )
        demand *= rng.normal(1.0, 0.12)  # noise
        demand = max(0.0, demand)

        # Purchase intent: a logistic of a propensity built directly from the
        # context features (not the noisy demand), so the classifier has a clean,
        # learnable signal. The scale factor sharpens class separation.
        z = (
            -1.0
            + 1.8 * _hour_curve(hour, ramadan)
            + 0.9 * (ev_mult - 1.0)
            + _age_intent(row["age_group"], category, hour)
            + (0.5 if row["is_weekend"] else 0.0)
            + 0.8 * (row["pop_weight"] - 1.0)
            + CATEGORY_PREF.get(category, 0.0)
            + rng.normal(0.0, 0.25)  # irreducible noise
        )
        prob = 1.0 / (1.0 + np.exp(-2.2 * z))
        will_order = int(rng.random() < prob)

        row["demand"] = round(demand, 2)
        row["will_order"] = will_order
        rows.append(row)

    return pd.DataFrame(rows)
