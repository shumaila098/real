"""Train the two scikit-learn models and persist them.

  • RandomForestRegressor  → product *demand* (orders) for a context.
  • RandomForestClassifier → customer *behaviour* (completion propensity when
    real outcome labels exist; purchase intent from the synthetic prior until
    then — the metadata says which).

REAL marketplace rows (from the Worker's training feed, see ``real_data.py``)
are blended in whenever available: real demand cells are calibrated onto the
synthetic scale and up-weighted, and the behaviour model switches to training
on real fulfilled/cancelled outcomes entirely once there are enough of them.
Both are wrapped in a Pipeline(ColumnTransformer + forest) so the same raw
feature dict scores at serve time without manual one-hot alignment. Feature
importances are summed back to the original (pre-one-hot) feature names so the
UI can show "which factors drive demand".
"""

from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import features as F
import registry
import synth

# Real rows count this much more than synthetic ones in the demand blend.
REAL_WEIGHT = 3.0
# The behaviour model trains on real outcomes alone once it has this many.
MIN_REAL_BEHAVIOR = 300

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
REG_PATH = os.path.join(MODEL_DIR, "demand_regressor.joblib")
CLF_PATH = os.path.join(MODEL_DIR, "behavior_classifier.joblib")
META_PATH = os.path.join(MODEL_DIR, "metadata.json")

# Friendly labels for the feature-importance chart.
FEATURE_LABELS = {
    "age": "Age",
    "age_group": "Age group",
    "hour": "Time of day",
    "day_of_month": "Date of month",
    "season": "Season",
    "year": "Year",
    "day_of_week": "Day of week",
    "is_weekend": "Weekend",
    "location": "Location",
    "pop_weight": "City scale",
    "category": "Product",
    "event": "Event (Ramadan/Eid/…)",
    "disaster": "Disaster",
}


def _make_pipeline(estimator) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                F.CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", F.NUMERIC_FEATURES),
        ]
    )
    return Pipeline([("pre", pre), ("model", estimator)])


def _grouped_importances(pipe: Pipeline) -> dict:
    """Sum one-hot importances back to original feature names."""
    pre: ColumnTransformer = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]
    importances = model.feature_importances_

    names = []
    ohe: OneHotEncoder = pre.named_transformers_["cat"]
    for col, cats in zip(F.CATEGORICAL_FEATURES, ohe.categories_):
        names.extend([col] * len(cats))
    names.extend(F.NUMERIC_FEATURES)

    grouped: dict[str, float] = {}
    for name, imp in zip(names, importances):
        grouped[name] = grouped.get(name, 0.0) + float(imp)
    total = sum(grouped.values()) or 1.0
    return {k: round(v / total, 4) for k, v in grouped.items()}


def train(n: int = 24000, seed: int = 42, real: dict | None = None) -> dict:
    """Train both models. ``real`` (optional) carries the marketplace's own
    data from ``real_data.to_frames``: {demand_df, behavior_df, snapshot}."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    real = real or {}
    demand_real = real.get("demand_df")
    behavior_real = real.get("behavior_df")
    snapshot = real.get("snapshot") or {}
    n_real_demand = 0 if demand_real is None else int(len(demand_real))
    n_real_behavior = 0 if behavior_real is None else int(len(behavior_real))

    df = synth.generate(n=n, seed=seed)

    # ── Demand regressor: synth prior + calibrated real cells ──────────
    X = df[F.FEATURE_COLUMNS]
    yr = df["demand"]
    weights = np.ones(len(df))
    calibration_k = None
    if n_real_demand > 0:
        # Real cells are order COUNTS per (date,hour,city,category); scale them
        # onto the synthetic demand magnitude so the forest learns shape from
        # both. Serving magnitude comes from each tenant's own series anyway.
        real_mean = float(demand_real["demand"].mean())
        calibration_k = float(yr.mean() / real_mean) if real_mean > 0 else 1.0
        dr = demand_real.copy()
        dr["demand"] = dr["demand"] * calibration_k
        X = pd.concat([X, dr[F.FEATURE_COLUMNS]], ignore_index=True)
        yr = pd.concat([yr, dr["demand"]], ignore_index=True)
        weights = np.concatenate([weights, np.full(len(dr), REAL_WEIGHT)])

    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed)
    reg = _make_pipeline(
        RandomForestRegressor(n_estimators=60, max_depth=9, n_jobs=-1, random_state=seed)
    )
    reg.fit(X.iloc[tr], yr.iloc[tr], model__sample_weight=weights[tr])
    r2 = r2_score(yr.iloc[te], reg.predict(X.iloc[te]))
    # Honest extra: R² measured on the REAL held-out cells alone.
    real_r2 = None
    real_te = te[te >= len(df)]
    if len(real_te) >= 30:
        real_r2 = r2_score(yr.iloc[real_te], reg.predict(X.iloc[real_te]))

    # ── Behaviour classifier ────────────────────────────────────────────
    # With enough real fulfilled/cancelled outcomes, train on REAL data only
    # (label = completion propensity). Mixing intent-flavoured synthetic labels
    # with real outcome labels would blur what the number means.
    behavior_label = "purchase intent (synthetic prior)"
    real_holdout = None
    if n_real_behavior >= MIN_REAL_BEHAVIOR and behavior_real["will_order"].nunique() > 1:
        Xb, yb = behavior_real[F.FEATURE_COLUMNS], behavior_real["will_order"]
        behavior_label = "completion propensity (real outcomes)"
    else:
        Xb, yb = df[F.FEATURE_COLUMNS], df["will_order"]
    Xtr, Xte, ytr, yte = train_test_split(
        Xb, yb, test_size=0.2, random_state=seed, stratify=yb
    )
    clf = _make_pipeline(
        RandomForestClassifier(
            n_estimators=60, max_depth=9, n_jobs=-1, random_state=seed
        )
    )
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    acc = accuracy_score(yte, (proba >= 0.5).astype(int))
    try:
        auc = roc_auc_score(yte, proba)
    except ValueError:
        auc = 0.5
    if behavior_label.startswith("completion"):
        real_holdout = {"behavior_real_auc": round(float(auc), 3)}

    joblib.dump(reg, REG_PATH)
    joblib.dump(clf, CLF_PATH)

    reg_imp = _grouped_importances(reg)
    clf_imp = _grouped_importances(clf)
    # Average the two models' importances for the headline "factors" chart.
    factors = {
        k: round((reg_imp.get(k, 0) + clf_imp.get(k, 0)) / 2, 4)
        for k in F.FEATURE_COLUMNS
    }

    metrics = {
        "demand_r2": round(float(r2), 3),
        "behavior_accuracy": round(float(acc), 3),
        "behavior_auc": round(float(auc), 3),
    }
    if real_r2 is not None:
        metrics["demand_real_r2"] = round(float(real_r2), 3)
    if real_holdout:
        metrics.update(real_holdout)

    meta = {
        "trained_at": int(time.time() * 1000),
        "rows": int(len(df)),
        "rows_real_demand": n_real_demand,
        "rows_real_behavior": n_real_behavior,
        "behavior_label": behavior_label,
        "calibration_k": None if calibration_k is None else round(calibration_k, 3),
        "model": "RandomForest (scikit-learn)"
        + (" + real marketplace data" if n_real_demand or n_real_behavior else ""),
        "metrics": metrics,
        "feature_labels": FEATURE_LABELS,
        "demand_importances": reg_imp,
        "behavior_importances": clf_imp,
        "factor_importances": factors,
        "categories": F.CATEGORIES,
        "locations": list(F.LOCATIONS.keys()),
        "age_groups": F.AGE_GROUPS,
    }
    with open(META_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    # Version the artifact + remember the real-data distribution for drift.
    registry.save(
        {
            "trained_at": meta["trained_at"],
            "rows_synth": int(len(df)),
            "rows_real_demand": n_real_demand,
            "rows_real_behavior": n_real_behavior,
            "behavior_label": behavior_label,
            "metrics": metrics,
            "training_snapshot": snapshot,
        }
    )

    print(
        f"Trained on {len(df)} synth + {n_real_demand} real demand cells "
        f"+ {n_real_behavior} real outcomes | demand R2={r2:.3f} | "
        f"behaviour acc={acc:.3f} auc={auc:.3f} [{behavior_label}]"
    )
    print(f"Saved -> {MODEL_DIR}")
    return meta


if __name__ == "__main__":
    train()
