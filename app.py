"""FastAPI service that serves scikit-learn customer-behaviour & demand
predictions to the Flutter admin "Insights" screen.

Endpoints
  GET  /health                      → liveness + model status
  POST /predict                     → single-context demand + purchase intent
  GET  /insights?...                → full dashboard payload (charts-ready)

The Flutter app calls ``GET /insights`` and renders the pies/bars from the JSON.
Models train lazily on first start if ``models/`` is empty.
"""

from __future__ import annotations

import os
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import features as F
import train as T

app = FastAPI(title="Realtim Prediction Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # admin dashboard is a trusted first-party caller
    allow_methods=["*"],
    allow_headers=["*"],
)

_REG = None
_CLF = None
_META: dict = {}

# Representative age per group for grid scoring.
AGE_REP = {"youth": 22, "adult": 32, "middle": 50, "senior": 66}
WEEKDAY_SHORT = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
EVENT_LABELS = {
    "none": "Normal",
    "ramadan": "Ramadan",
    "eid_fitr": "Eid-ul-Fitr",
    "eid_adha": "Eid-ul-Adha",
    "festival": "Festival",
    "disaster": "Disaster",
}


def _ensure_models():
    """Load persisted models, training them once if absent."""
    global _REG, _CLF, _META
    if _REG is not None:
        return
    if not (os.path.exists(T.REG_PATH) and os.path.exists(T.CLF_PATH)):
        T.train()
    _REG = joblib.load(T.REG_PATH)
    _CLF = joblib.load(T.CLF_PATH)
    import json

    with open(T.META_PATH, encoding="utf-8") as fh:
        _META = json.load(fh)


@app.on_event("startup")
def _startup():
    _ensure_models()


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)[F.FEATURE_COLUMNS]


def _demand(rows: list[dict]) -> np.ndarray:
    return np.clip(_REG.predict(_frame(rows)), 0, None)


def _intent(rows: list[dict]) -> np.ndarray:
    return _CLF.predict_proba(_frame(rows))[:, 1]


def _resolve_event(event: str, disaster: str):
    """Map the UI event selection to (event_override, disaster_value)."""
    event = (event or "auto").lower()
    disaster = (disaster or "none").lower()
    if event == "auto":
        return None, (disaster if disaster != "none" else "none")
    if event == "disaster":
        dis = disaster if disaster != "none" else "flood"
        return f"disaster_{dis}", dis
    return event, "none"


# ── Single prediction ───────────────────────────────────────────────────────


class PredictIn(BaseModel):
    date: str | None = None
    hour: int = 19
    age: int = 32
    location: str = "islamabad"
    category: str = "food"
    event: str = "auto"
    disaster: str = "none"


@app.post("/predict")
def predict(body: PredictIn):
    _ensure_models()
    when = F.parse_date(body.date)
    override, dis = _resolve_event(body.event, body.disaster)
    row = F.build_features(
        when=when, hour=body.hour, age=body.age, location=body.location,
        category=body.category, disaster=dis, event_override=override,
    )
    demand = float(_demand([row])[0])
    intent = float(_intent([row])[0])
    return {
        "ok": True,
        "predicted_demand": round(demand, 1),
        "purchase_intent": round(intent, 3),
        "behavior": _intent_label(intent),
        "context": row,
    }


def _intent_label(p: float) -> str:
    if p >= 0.66:
        return "High intent"
    if p >= 0.40:
        return "Medium intent"
    if p >= 0.20:
        return "Low intent"
    return "Unlikely"


# ── Dashboard insights ───────────────────────────────────────────────────────


@app.get("/insights")
def insights(
    date: str | None = None,
    location: str = "islamabad",
    age_group: str = "adult",
    event: str = "auto",
    disaster: str = "none",
):
    _ensure_models()
    base_date = F.parse_date(date)
    age = AGE_REP.get(age_group, 32)
    override, dis = _resolve_event(event, disaster)

    hours_sample = [8, 11, 13, 16, 19, 21]

    def rows_for(day, ev_override, ev_disaster, hours, cats=None):
        cats = cats or F.CATEGORIES
        out = []
        for c in cats:
            for h in hours:
                out.append(
                    F.build_features(
                        when=day, hour=h, age=age, location=location,
                        category=c, disaster=ev_disaster,
                        event_override=ev_override,
                    )
                )
        return out

    # 1) 7-day demand forecast (sum across categories, averaged over the hour grid).
    forecast_by_day = []
    for i in range(7):
        day = base_date + timedelta(days=i)
        rows = rows_for(day, override, dis, hours_sample)
        # mean over hours, summed over categories ≈ representative daily demand.
        d = _demand(rows).reshape(len(F.CATEGORIES), len(hours_sample)).mean(axis=1).sum()
        forecast_by_day.append(
            {"label": f"{WEEKDAY_SHORT[day.weekday()]} {day.day}", "value": round(float(d), 1)}
        )

    # 2) Hourly demand curve for the base date (sum across categories).
    demand_by_hour = []
    for h in range(24):
        rows = rows_for(base_date, override, dis, [h])
        demand_by_hour.append({"label": f"{h:02d}", "value": round(float(_demand(rows).sum()), 1)})

    # 3) Product-demand share (pie) at the base date, summed over the hour grid.
    cat_rows = rows_for(base_date, override, dis, hours_sample)
    cat_demand = _demand(cat_rows).reshape(len(F.CATEGORIES), len(hours_sample)).sum(axis=1)
    cat_total = float(cat_demand.sum()) or 1.0
    category_share = [
        {"label": c, "value": round(float(v), 1), "pct": round(float(v) / cat_total * 100, 1)}
        for c, v in sorted(zip(F.CATEGORIES, cat_demand), key=lambda x: -x[1])
    ]

    # 4) Behaviour segments (pie): score a shopper population, bucket the intent.
    pop_rows = []
    for ag, a in AGE_REP.items():
        for c in F.CATEGORIES:
            for h in [10, 14, 18, 21]:
                pop_rows.append(
                    F.build_features(
                        when=base_date, hour=h, age=a, location=location,
                        category=c, disaster=dis, event_override=override,
                    )
                )
    probs = _intent(pop_rows)
    buckets = {"High intent": 0, "Medium intent": 0, "Low intent": 0, "Unlikely": 0}
    for p in probs:
        buckets[_intent_label(float(p))] += 1
    seg_total = len(probs) or 1
    behavior_segments = [
        {"label": k, "value": v, "pct": round(v / seg_total * 100, 1)}
        for k, v in buckets.items()
    ]
    avg_intent = float(probs.mean())

    # 5) Event impact (bar): total base-date demand under each event vs normal.
    base_rows = rows_for(base_date, "none", "none", hours_sample)
    base_total = float(_demand(base_rows).sum()) or 1.0
    event_impact = []
    for ev in ["none", "ramadan", "eid_fitr", "eid_adha", "festival", "disaster"]:
        ov, dv = _resolve_event(ev, "flood" if ev == "disaster" else "none")
        ov = "none" if ev == "none" else ov
        rows = rows_for(base_date, ov, dv, hours_sample)
        total = float(_demand(rows).sum())
        event_impact.append(
            {"label": EVENT_LABELS[ev], "value": round(total / base_total, 2)}
        )

    # 6) Feature importance (bar): which factors drive the predictions.
    labels = _META.get("feature_labels", {})
    factors = _META.get("factor_importances", {})
    feature_importance = [
        {"label": labels.get(k, k), "value": round(v * 100, 1)}
        for k, v in sorted(factors.items(), key=lambda x: -x[1])
    ]

    # KPIs + narrative.
    peak_day = max(forecast_by_day, key=lambda x: x["value"])
    peak_hour = max(demand_by_hour, key=lambda x: x["value"])
    demand_7d = round(sum(x["value"] for x in forecast_by_day), 0)
    top_cat = category_share[0]["label"] if category_share else "—"
    top_factor = feature_importance[0]["label"] if feature_importance else "—"

    narrative = [
        f"Forecast demand over the next 7 days is ~{int(demand_7d)} orders, "
        f"peaking on {peak_day['label']}.",
        f"Busiest hour is around {peak_hour['label']}:00; '{top_cat}' is the "
        f"top product category.",
        f"Average purchase intent across shopper segments is "
        f"{round(avg_intent * 100)}%.",
        f"The strongest predictor of behaviour is {top_factor.lower()}.",
    ]
    if event and event not in ("auto", "none"):
        mult = next((e["value"] for e in event_impact
                     if e["label"] == EVENT_LABELS.get(event)), None)
        if mult:
            narrative.insert(
                0,
                f"{EVENT_LABELS.get(event, event)} shifts demand to "
                f"{mult}× the normal baseline.",
            )

    return {
        "ok": True,
        "source": "model",
        "meta": {
            "model": _META.get("model", "RandomForest (scikit-learn)"),
            "trained_at": _META.get("trained_at"),
            "metrics": _META.get("metrics", {}),
            "rows": _META.get("rows"),
        },
        "context": {
            "date": base_date.isoformat(),
            "location": location,
            "age_group": age_group,
            "event": event,
            "disaster": disaster,
        },
        "kpis": {
            "demand_7d": demand_7d,
            "peak_day": peak_day["label"],
            "peak_hour": f"{peak_hour['label']}:00",
            "avg_intent_pct": round(avg_intent * 100, 1),
            "top_category": top_cat,
        },
        "forecast_by_day": forecast_by_day,
        "demand_by_hour": demand_by_hour,
        "category_share": category_share,
        "behavior_segments": behavior_segments,
        "event_impact": event_impact,
        "feature_importance": feature_importance,
        "narrative": narrative,
        "options": {
            "locations": _META.get("locations", list(F.LOCATIONS.keys())),
            "age_groups": _META.get("age_groups", F.AGE_GROUPS),
            "categories": _META.get("categories", F.CATEGORIES),
        },
    }


@app.get("/health")
def health():
    loaded = _REG is not None and _CLF is not None
    return {
        "ok": True,
        "model_loaded": loaded,
        "model": _META.get("model"),
        "trained_at": _META.get("trained_at"),
        "metrics": _META.get("metrics", {}),
    }
