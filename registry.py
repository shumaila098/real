"""Tiny model registry: versioned metadata for every artifact this service
serves, persisted next to the models. `/version`, `/health` and the Insights UI
read it so provenance ("model vN, trained <date> on R real + S synthetic rows,
backtest SMAPE x%") is always visible — no more silent model swaps."""

from __future__ import annotations

import json
import os
import time

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
REGISTRY_PATH = os.path.join(MODEL_DIR, "registry.json")


def load() -> dict:
    if os.path.exists(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass
    return {"version": 0}


def save(update: dict) -> dict:
    """Merge `update` into the registry, bumping the version."""
    reg = load()
    reg.update(update)
    reg["version"] = int(reg.get("version", 0)) + 1
    reg["updated_at"] = int(time.time() * 1000)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as fh:
        json.dump(reg, fh, indent=2)
    return reg
