"""Real marketplace data: pull the Worker's training feed and shape it.

The Cloudflare Worker exports one compact row per real order
(``GET {EXPORT_URL}?days=N`` guarded by ``x-ml-key``):

    {ts, t, v, cat, lat, lng, amt, cur, st: f|x|o, paid, cx, cu}

This module turns those raw facts into the exact column shape ``synth.py``
produces — so, as its docstring always promised, the rest of the training
pipeline is unchanged — plus a distribution snapshot for drift monitoring and
per-customer order sequences for the churn model.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

import features as F

EXPORT_URL = os.environ.get(
    "EXPORT_URL",
    "https://realtim-proxy.shumailakhan-syed.workers.dev/ml/export",
)
ML_SYNC_KEY = os.environ.get("ML_SYNC_KEY", "")

DAY_MS = 86_400_000


def fetch_rows(days: int = 90) -> list[dict]:
    """Pull raw real-order rows from the Worker. Empty list when the feed is
    unreachable/unconfigured — training then proceeds on synthetic data only."""
    headers = {"x-ml-key": ML_SYNC_KEY} if ML_SYNC_KEY else {}
    try:
        r = requests.get(EXPORT_URL, params={"days": days}, headers=headers, timeout=60)
        r.raise_for_status()
        body = r.json()
        if body.get("ok") and isinstance(body.get("rows"), list):
            return body["rows"]
    except Exception:
        pass
    return []


def nearest_city(lat, lng) -> str:
    """Map raw coordinates onto the model's city vocabulary (else 'other')."""
    if lat is None or lng is None:
        return "other"
    best, bd = "other", float("inf")
    for name, loc in F.LOCATIONS.items():
        d = (float(lat) - loc["lat"]) ** 2 + (float(lng) - loc["lng"]) ** 2
        if d < bd:
            bd, best = d, name
    return best if bd < 0.8**2 else "other"  # ~80 km box


# Marketplace categories seen in real orders don't always match the model's
# 7-category vocabulary; fold the common vertical categories onto it.
CATEGORY_MAP = {
    "grocery": "groceries",
    "meal": "food",
    "restaurant": "food",
    "medicine": "pharmacy",
    "medicines": "pharmacy",
    "health": "pharmacy",
    "delivery": "services",
    "parcel": "services",
    "repair": "services",
    "booking": "services",
    "appointment": "services",
    "complaint": "services",
    "apparel": "clothing",
    "gift": "gifts",
}


def model_category(raw: str) -> str:
    c = str(raw or "other").lower()
    if c in F.CATEGORIES:
        return c
    return CATEGORY_MAP.get(c, "services")


def _row_context(ts_ms: int, city: str, category: str) -> dict:
    when = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return F.build_features(
        when=when.date(),
        hour=when.hour,
        age=32,  # shopper age isn't collected yet — recorded as a limitation
        location=city,
        category=category,
        disaster="none",
    )


def to_frames(rows: list[dict]):
    """→ (demand_df, behavior_df, snapshot). Any frame may be empty."""
    if not rows:
        return pd.DataFrame(), pd.DataFrame(), {}

    recs = []
    for r in rows:
        ts = int(r.get("ts") or 0)
        if ts <= 0:
            continue
        recs.append(
            {
                "ts": ts,
                "city": nearest_city(r.get("lat"), r.get("lng")),
                "category": model_category(r.get("cat")),
                "st": str(r.get("st") or "o"),
                "cu": r.get("cu"),
            }
        )
    if not recs:
        return pd.DataFrame(), pd.DataFrame(), {}
    raw = pd.DataFrame(recs)
    raw["dt"] = pd.to_datetime(raw["ts"], unit="ms", utc=True)

    # ── Demand: real order counts per (date, hour, city, category) cell ──
    raw["date"] = raw["dt"].dt.date
    raw["hour"] = raw["dt"].dt.hour
    grp = (
        raw.groupby(["date", "hour", "city", "category"])
        .size()
        .reset_index(name="count")
    )
    demand_rows = []
    for _, g in grp.iterrows():
        row = _row_context(
            int(pd.Timestamp(g["date"]).value // 10**6) + int(g["hour"]) * 3_600_000,
            g["city"],
            g["category"],
        )
        row["demand"] = float(g["count"])  # calibrated against synth in train()
        demand_rows.append(row)
    demand_df = pd.DataFrame(demand_rows)

    # ── Behaviour: real completion outcomes (fulfilled=1, cancelled=0) ──
    done = raw[raw["st"].isin(["f", "x"])]
    behavior_rows = []
    for _, g in done.iterrows():
        row = _row_context(int(g["ts"]), g["city"], g["category"])
        row["will_order"] = 1 if g["st"] == "f" else 0
        behavior_rows.append(row)
    behavior_df = pd.DataFrame(behavior_rows)

    # ── Distribution snapshot for drift monitoring ──
    def dist(series) -> dict:
        vc = series.value_counts(normalize=True)
        return {str(k): round(float(v), 4) for k, v in vc.items()}

    snapshot = {
        "category": dist(raw["category"]),
        "hour": dist(raw["dt"].dt.hour),
        "city": dist(raw["city"]),
        "rows": int(len(raw)),
    }
    return demand_df, behavior_df, snapshot


def churn_snapshots(rows: list[dict], now_ms: int | None = None, window_days: int = 45):
    """Per-customer training snapshots for the churn model.

    Observation point = ``window_days`` before now: features are computed from
    each customer's orders BEFORE that point, and the label is whether they
    ordered again in the window after it — a leakage-free real churn label.
    """
    now_ms = now_ms or int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    cutoff = now_ms - window_days * DAY_MS

    by_customer: dict[str, list[int]] = {}
    for r in rows:
        cu, ts = r.get("cu"), int(r.get("ts") or 0)
        if not cu or ts <= 0:
            continue
        by_customer.setdefault(str(cu), []).append(ts)

    feats, labels = [], []
    for ts_list in by_customer.values():
        ts_list.sort()
        before = [t for t in ts_list if t < cutoff]
        if not before:
            continue  # no history before the observation point
        after = [t for t in ts_list if t >= cutoff]
        last, first, n = before[-1], before[0], len(before)
        tenure = max(1.0, (cutoff - first) / DAY_MS)
        cadence = tenure / (n - 1) if n > 1 else 30.0
        feats.append(
            {
                "recency_days": (cutoff - last) / DAY_MS,
                "frequency": n,
                "monetary": 0.0,  # spend isn't in the anonymised feed
                "tenure_days": tenure,
                "cadence_days": cadence,
                "points": 0.0,
            }
        )
        labels.append(0 if after else 1)  # churned = silent in the window
    return feats, np.array(labels, dtype=int)


def psi(expected: dict, actual: dict) -> float:
    """Population-stability index between two categorical distributions."""
    keys = set(expected) | set(actual)
    total = 0.0
    for k in keys:
        e = max(1e-4, float(expected.get(k, 0)))
        a = max(1e-4, float(actual.get(k, 0)))
        total += (a - e) * np.log(a / e)
    return round(float(total), 4)
