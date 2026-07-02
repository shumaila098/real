"""Churn-probability scoring for the CRM segments + win-back trigger.

Trains a logistic model on REAL leakage-free labels (did the customer order
again in the 45 days after the observation point — built by
``real_data.churn_snapshots``) once there's enough data; until then a
calibrated cadence heuristic (identical to the Worker's fallback) serves, and
every response says which one produced the numbers.
"""

from __future__ import annotations

import math
import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
CHURN_PATH = os.path.join(MODEL_DIR, "churn_model.joblib")

FEATURES = ["recency_days", "frequency", "monetary", "tenure_days", "cadence_days", "points"]
MIN_TRAIN = 200  # snapshots needed (with both classes) before the model takes over

_model = None
_loaded = False


def _load():
    global _model, _loaded
    if not _loaded:
        _model = joblib.load(CHURN_PATH) if os.path.exists(CHURN_PATH) else None
        _loaded = True
    return _model


def _vector(f: dict) -> list[float]:
    return [float(f.get(k) or 0.0) for k in FEATURES]


def heuristic_prob(f: dict) -> float:
    """Cadence-relative risk (mirrors the Worker's fallback exactly)."""
    cadence = max(3.0, float(f.get("cadence_days") or 30))
    gaps = float(f.get("recency_days") or 999) / cadence
    z = (
        -2.2
        + 1.15 * gaps
        - 0.25 * math.log(1 + float(f.get("frequency") or 0))
        - 0.1 * math.log(1 + float(f.get("monetary") or 0) / 500)
    )
    return round(1.0 / (1.0 + math.exp(-z)), 3)


def train_churn(feats: list[dict], labels: np.ndarray) -> dict | None:
    """Fit + persist the churn model when the real dataset is big enough.
    Returns metrics, or None when data is still too thin (heuristic keeps
    serving — never train a junk model just to say we did)."""
    global _model, _loaded
    if len(feats) < MIN_TRAIN or len(set(labels.tolist())) < 2:
        return None
    X = np.array([_vector(f) for f in feats])
    Xtr, Xte, ytr, yte = train_test_split(X, labels, test_size=0.25, random_state=7, stratify=labels)
    pipe = Pipeline(
        [("scale", StandardScaler()), ("lr", LogisticRegression(max_iter=500, class_weight="balanced"))]
    )
    pipe.fit(Xtr, ytr)
    auc = roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipe, CHURN_PATH)
    _model, _loaded = pipe, True
    return {"churn_auc": round(float(auc), 3), "churn_rows": int(len(feats))}


def score(customers: list[dict]) -> tuple[list[float], str]:
    """Batch-score churn probability. Returns (probs, model_name)."""
    model = _load()
    if model is not None:
        X = np.array([_vector(f) for f in customers])
        probs = model.predict_proba(X)[:, 1]
        return [round(float(p), 3) for p in probs], "logistic (real labels)"
    return [heuristic_prob(f) for f in customers], "heuristic"
