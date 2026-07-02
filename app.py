"""FastAPI service that serves customer-behaviour & demand predictions to the
Flutter admin "Insights" screen — via the Cloudflare Worker, which is the only
intended caller (it RBAC-checks every app request and holds the API key).

Endpoints
  GET  /health                      → liveness + model/registry status (open)
  GET  /version                     → model registry (open)
  POST /predict                     → single-context demand + purchase intent
  GET  /insights?...                → market-level dashboard payload (legacy)
  POST /insights                    → tenant-aware payload: the Worker sends the
                                      tenant's real daily series + category mix;
                                      forecasts come from THAT series (weekday
                                      seasonality + trend + P10/P90 bands,
                                      backtest-validated), with the model
                                      supplying hour-of-day shape & event lift.
  POST /churn/score                 → batch churn probabilities (real model when
                                      trained, labelled heuristic otherwise)
  POST /admin/retrain               → pull real rows from the Worker feed, blend
                                      with the synthetic prior, retrain, bump
                                      the registry version
  POST /admin/drift                 → PSI drift check: recent real data vs the
                                      distribution the models were trained on

Auth: when the ML_API_KEY env var is set, every endpoint except /health and
/version requires the matching ``x-api-key`` header.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import churn as CH
import features as F
import forecasting as FC
import real_data as RD
import registry as REG
import train as T

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


@asynccontextmanager
async def _lifespan(app: FastAPI):
    _ensure_models()
    yield


app = FastAPI(title="Realtim Prediction Service", version="2.0.0", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # the Worker is the only real caller; key gates writes
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_key(x_api_key: str | None = Header(default=None)):
    """Shared-secret gate. No-op until ML_API_KEY is configured, so a fresh
    deployment keeps working while the secret is being set on both ends."""
    expected = os.environ.get("ML_API_KEY", "")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="bad or missing x-api-key")


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


def _intent_label(p: float) -> str:
    if p >= 0.66:
        return "High intent"
    if p >= 0.40:
        return "Medium intent"
    if p >= 0.20:
        return "Low intent"
    return "Unlikely"


# ── Single prediction ───────────────────────────────────────────────────────


class PredictIn(BaseModel):
    date: str | None = None
    hour: int = 19
    age: int = 32
    location: str = "islamabad"
    category: str = "food"
    event: str = "auto"
    disaster: str = "none"


@app.post("/predict", dependencies=[Depends(require_key)])
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


# ── Dashboard insights ───────────────────────────────────────────────────────


class SeriesPoint(BaseModel):
    date: str
    orders: float = 0


class InsightsIn(BaseModel):
    date: str | None = None
    location: str | None = None
    age_group: str | None = None
    event: str | None = None
    disaster: str | None = None
    tenant_id: str | None = None
    # The tenant's REAL history, materialized by the Worker's rollups.
    series: list[SeriesPoint] | None = None
    category_mix: dict[str, float] | None = None


def _build_insights(
    *,
    date: str | None,
    location: str,
    age_group: str,
    event: str,
    disaster: str,
    series: list[dict] | None = None,
    category_mix: dict | None = None,
) -> dict:
    _ensure_models()
    base_date = F.parse_date(date)
    age = AGE_REP.get(age_group, 32)
    override, dis = _resolve_event(event, disaster)
    reg_meta = REG.load()

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

    # ── Forecast: the tenant's own series when it carries enough signal ──
    tenant_fc = FC.tenant_forecast([p for p in (series or [])]) if series else None
    source = "tenant_model" if tenant_fc else "market_model"

    # Event lift from the model (used to scale scenario previews).
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
    sel_mult = 1.0
    if event and event not in ("auto", "none"):
        sel_mult = next(
            (e["value"] for e in event_impact if e["label"] == EVENT_LABELS.get(event)),
            1.0,
        )

    backtest_smape = None
    if tenant_fc:
        forecast_by_day = [
            {
                "label": p["label"],
                "value": round(p["value"] * sel_mult, 1),
                "p10": round(p["p10"] * sel_mult, 1),
                "p90": round(p["p90"] * sel_mult, 1),
            }
            for p in tenant_fc["points"]
        ]
        backtest_smape = tenant_fc["backtest_smape"]
        daily_scale = max(
            0.1, float(np.mean([p["value"] for p in forecast_by_day]))
        )
    else:
        forecast_by_day = []
        for i in range(7):
            day = base_date + timedelta(days=i)
            rows = rows_for(day, override, dis, hours_sample)
            d = _demand(rows).reshape(len(F.CATEGORIES), len(hours_sample)).mean(axis=1).sum()
            forecast_by_day.append(
                {"label": f"{WEEKDAY_SHORT[day.weekday()]} {day.day}", "value": round(float(d), 1)}
            )
        # A tenant with SOME history but not enough to forecast still gets the
        # market curve anchored to their own average volume — server-side now.
        if series:
            hist = [float(p["orders"]) for p in series]
            hist_mean = float(np.mean([h for h in hist])) if hist else 0.0
            model_mean = float(np.mean([p["value"] for p in forecast_by_day])) or 1.0
            if hist_mean > 0:
                k = hist_mean * sel_mult / model_mean
                for p in forecast_by_day:
                    p["value"] = round(p["value"] * k, 1)
                source = "market_model_scaled"
        daily_scale = max(0.1, float(np.mean([p["value"] for p in forecast_by_day])))

    # ── Hourly curve: model shape, scaled to the forecast's daily volume ──
    raw_hours = []
    for h in range(24):
        rows = rows_for(base_date, override, dis, [h])
        raw_hours.append(float(_demand(rows).sum()))
    hour_total = sum(raw_hours) or 1.0
    demand_by_hour = [
        {"label": f"{h:02d}", "value": round(v / hour_total * daily_scale, 2)}
        for h, v in enumerate(raw_hours)
    ]

    # ── Category share: REAL mix when the tenant has one ──
    if category_mix:
        total_mix = sum(float(v) for v in category_mix.values()) or 1.0
        category_share = [
            {
                "label": k,
                "value": round(float(v), 1),
                "pct": round(float(v) / total_mix * 100, 1),
            }
            for k, v in sorted(category_mix.items(), key=lambda x: -float(x[1]))
        ][:8]
        category_source = "real"
    else:
        cat_rows = rows_for(base_date, override, dis, hours_sample)
        cat_demand = _demand(cat_rows).reshape(len(F.CATEGORIES), len(hours_sample)).sum(axis=1)
        cat_total = float(cat_demand.sum()) or 1.0
        category_share = [
            {"label": c, "value": round(float(v), 1), "pct": round(float(v) / cat_total * 100, 1)}
            for c, v in sorted(zip(F.CATEGORIES, cat_demand), key=lambda x: -x[1])
        ]
        category_source = "model"

    # ── Behaviour segments (model-scored population) ──
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

    # ── Feature importance ──
    labels = _META.get("feature_labels", {})
    factors = _META.get("factor_importances", {})
    feature_importance = [
        {"label": labels.get(k, k), "value": round(v * 100, 1)}
        for k, v in sorted(factors.items(), key=lambda x: -x[1])
    ]

    # ── KPIs + narrative ──
    peak_day = max(forecast_by_day, key=lambda x: x["value"])
    peak_hour = max(demand_by_hour, key=lambda x: x["value"])
    demand_7d = round(sum(x["value"] for x in forecast_by_day), 0)
    top_cat = category_share[0]["label"] if category_share else "—"
    top_factor = feature_importance[0]["label"] if feature_importance else "—"

    narrative = []
    if source == "tenant_model":
        narrative.append(
            f"Forecast from YOUR order history ({tenant_fc['history_days']} days, "
            f"{tenant_fc['history_orders']} orders): ~{int(demand_7d)} orders in "
            f"the next 7 days, peaking on {peak_day['label']}."
        )
        if backtest_smape is not None:
            narrative.append(
                f"Backtest on your own last week: {backtest_smape}% average error "
                f"(SMAPE) — the shaded band shows the P10–P90 range."
            )
    elif source == "market_model_scaled":
        narrative.append(
            f"~{int(demand_7d)} orders expected in the next 7 days (market model "
            f"anchored to your recent volume — the forecast switches to your own "
            f"history once you have 2+ weeks of orders)."
        )
    else:
        narrative.append(
            f"Market-level forecast: ~{int(demand_7d)} orders over the next 7 "
            f"days, peaking on {peak_day['label']}."
        )
    narrative.append(
        f"Busiest hour is around {peak_hour['label']}:00; "
        f"'{top_cat}' leads {'your real order mix' if category_source == 'real' else 'product demand'}."
    )
    narrative.append(
        f"Average {_META.get('behavior_label', 'purchase intent')} across segments is "
        f"{round(avg_intent * 100)}%; the strongest predictor is {top_factor.lower()}."
    )
    if event and event not in ("auto", "none") and sel_mult != 1.0:
        narrative.insert(
            0,
            f"{EVENT_LABELS.get(event, event)} scenario applied: demand scaled to "
            f"{sel_mult}× the normal baseline.",
        )

    return {
        "ok": True,
        "source": source,
        "meta": {
            "model": (
                f"Tenant series forecaster + {_META.get('model', 'RandomForest')}"
                if source == "tenant_model"
                else _META.get("model", "RandomForest (scikit-learn)")
            ),
            "version": reg_meta.get("version", 0),
            "trained_at": _META.get("trained_at"),
            "rows": _META.get("rows"),
            "rows_real_demand": _META.get("rows_real_demand", 0),
            "rows_real_behavior": _META.get("rows_real_behavior", 0),
            "behavior_label": _META.get("behavior_label"),
            "metrics": {
                **_META.get("metrics", {}),
                **({"backtest_smape": backtest_smape} if backtest_smape is not None else {}),
            },
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
        "category_source": category_source,
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


@app.get("/insights", dependencies=[Depends(require_key)])
def insights(
    date: str | None = None,
    location: str = "islamabad",
    age_group: str = "adult",
    event: str = "auto",
    disaster: str = "none",
):
    return _build_insights(
        date=date, location=location, age_group=age_group,
        event=event, disaster=disaster,
    )


@app.post("/insights", dependencies=[Depends(require_key)])
def insights_tenant(body: InsightsIn):
    series = [p.model_dump() for p in body.series] if body.series else None
    return _build_insights(
        date=body.date,
        location=body.location or "islamabad",
        age_group=body.age_group or "adult",
        event=body.event or "auto",
        disaster=body.disaster or "none",
        series=series,
        category_mix=body.category_mix,
    )


# ── Churn scoring ─────────────────────────────────────────────────────────────


class ChurnIn(BaseModel):
    customers: list[dict]


@app.post("/churn/score", dependencies=[Depends(require_key)])
def churn_score(body: ChurnIn):
    probs, model = CH.score(body.customers)
    return {"ok": True, "probs": probs, "model": model, "n": len(probs)}


# ── Admin: retrain on real data + drift check ────────────────────────────────


class RetrainIn(BaseModel):
    days: int = 90
    synth_rows: int = 24000


@app.post("/admin/retrain", dependencies=[Depends(require_key)])
def admin_retrain(body: RetrainIn):
    global _REG, _CLF, _META
    rows = RD.fetch_rows(days=min(365, max(7, body.days)))
    demand_df, behavior_df, snapshot = RD.to_frames(rows)
    meta = T.train(
        n=body.synth_rows,
        real={"demand_df": demand_df, "behavior_df": behavior_df, "snapshot": snapshot},
    )
    # Churn model from the same feed (trains only when there's enough signal).
    churn_metrics = None
    feats, labels = RD.churn_snapshots(rows)
    if len(feats) > 0:
        churn_metrics = CH.train_churn(feats, labels)
        if churn_metrics:
            REG.save({"churn_metrics": churn_metrics})
    # Hot-swap the serving models.
    _REG = joblib.load(T.REG_PATH)
    _CLF = joblib.load(T.CLF_PATH)
    _META = meta
    reg_meta = REG.load()
    return {
        "ok": True,
        "version": reg_meta.get("version"),
        "rows_real": len(rows),
        "rows_real_demand": meta.get("rows_real_demand"),
        "rows_real_behavior": meta.get("rows_real_behavior"),
        "behavior_label": meta.get("behavior_label"),
        "metrics": meta.get("metrics"),
        "churn": churn_metrics or "insufficient real data — heuristic serving",
    }


@app.post("/admin/drift", dependencies=[Depends(require_key)])
def admin_drift():
    reg_meta = REG.load()
    trained = (reg_meta.get("training_snapshot") or {})
    if not trained or not trained.get("rows"):
        return {"ok": True, "drifted": False, "reason": "no real training snapshot yet"}
    rows = RD.fetch_rows(days=14)
    _, _, recent = RD.to_frames(rows)
    if not recent or not recent.get("rows"):
        return {"ok": True, "drifted": False, "reason": "no recent real rows"}
    scores = {
        dim: RD.psi(trained.get(dim, {}), recent.get(dim, {}))
        for dim in ("category", "hour", "city")
    }
    worst = max(scores.values())
    return {
        "ok": True,
        "psi": worst,
        "scores": scores,
        "drifted": worst > 0.25,  # standard PSI alert threshold
        "trained_rows": trained.get("rows"),
        "recent_rows": recent.get("rows"),
    }


# ── Liveness / registry ──────────────────────────────────────────────────────


@app.get("/version")
def version():
    return {"ok": True, **REG.load()}


@app.get("/health")
def health():
    loaded = _REG is not None and _CLF is not None
    reg_meta = REG.load()
    return {
        "ok": True,
        "model_loaded": loaded,
        "model": _META.get("model"),
        "version": reg_meta.get("version", 0),
        "trained_at": _META.get("trained_at"),
        "rows_real_demand": _META.get("rows_real_demand", 0),
        "rows_real_behavior": _META.get("rows_real_behavior", 0),
        "metrics": _META.get("metrics", {}),
        "auth": bool(os.environ.get("ML_API_KEY")),
    }
