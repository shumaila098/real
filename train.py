"""Train the two scikit-learn models and persist them.

  • RandomForestRegressor  → product *demand* (orders) for a context.
  • RandomForestClassifier → customer *behaviour*: will this shopper order?

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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import features as F
import synth

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


def train(n: int = 24000, seed: int = 42) -> dict:
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = synth.generate(n=n, seed=seed)
    X = df[F.FEATURE_COLUMNS]

    # ── Demand regressor ────────────────────────────────────────────────
    yr = df["demand"]
    Xtr, Xte, ytr, yte = train_test_split(X, yr, test_size=0.2, random_state=seed)
    reg = _make_pipeline(
        RandomForestRegressor(n_estimators=60, max_depth=9, n_jobs=-1, random_state=seed)
    )
    reg.fit(Xtr, ytr)
    r2 = r2_score(yte, reg.predict(Xte))

    # ── Behaviour classifier ────────────────────────────────────────────
    yc = df["will_order"]
    Xtr, Xte, ytr, yte = train_test_split(
        X, yc, test_size=0.2, random_state=seed, stratify=yc
    )
    clf = _make_pipeline(
        RandomForestClassifier(
            n_estimators=60, max_depth=9, n_jobs=-1, random_state=seed
        )
    )
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    acc = accuracy_score(yte, (proba >= 0.5).astype(int))
    auc = roc_auc_score(yte, proba)

    joblib.dump(reg, REG_PATH)
    joblib.dump(clf, CLF_PATH)

    reg_imp = _grouped_importances(reg)
    clf_imp = _grouped_importances(clf)
    # Average the two models' importances for the headline "factors" chart.
    factors = {
        k: round((reg_imp.get(k, 0) + clf_imp.get(k, 0)) / 2, 4)
        for k in F.FEATURE_COLUMNS
    }

    meta = {
        "trained_at": int(time.time() * 1000),
        "rows": int(len(df)),
        "model": "RandomForest (scikit-learn)",
        "metrics": {
            "demand_r2": round(float(r2), 3),
            "behavior_accuracy": round(float(acc), 3),
            "behavior_auc": round(float(auc), 3),
        },
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

    print(
        f"Trained on {len(df)} rows | demand R2={r2:.3f} | "
        f"behaviour acc={acc:.3f} auc={auc:.3f}"
    )
    print(f"Saved -> {MODEL_DIR}")
    return meta


if __name__ == "__main__":
    train()
