# Realtim Prediction Service (scikit-learn)

A small FastAPI microservice that predicts **customer behaviour** and **product
demand** for the admin *Insights* screen. It runs separately from the Cloudflare
Worker because scikit-learn is Python and cannot execute on the Worker runtime.

## Features the models use

Exactly the factors requested in the brief:

| Feature | Source |
|---|---|
| Age / age group | shopper context |
| Time (hour of day) | context / hour grid |
| Date of month | calendar |
| Season | derived from month |
| Year | calendar |
| Day of week / weekend | calendar |
| Events — Ramadan, Eid-ul-Fitr, Eid-ul-Adha, national festivals | `events.py` calendar |
| Disasters — flood / earthquake / heatwave | caller-supplied flag |
| Location (city) | context (+ population weight) |
| Product demand (category) | target / context |

## Models

- `RandomForestRegressor` → product **demand** (orders) for a context.
- `RandomForestClassifier` → **purchase intent** (will the shopper order?).

Both are scikit-learn `Pipeline(ColumnTransformer + forest)`. On first run they
train on a synthetic-but-realistic dataset (`synth.py`) that encodes domain
effects (Ramadan evening peaks, Eid clothing/gift spikes, disaster pharmacy
surge, weekend/population scaling). The forests learn those effects back, so the
served predictions are genuine model output. Swap in real RTDB `requests` later
by feeding rows of the same shape.

## Run

```bash
cd ml_service
python -m venv .venv && .venv\Scripts\activate    # Windows (PowerShell: .venv\Scripts\Activate.ps1)
pip install -r requirements.txt
python train.py            # optional: trains + prints R²/accuracy (auto-runs on first request too)
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then `GET http://localhost:8000/insights` returns the dashboard JSON.

## Endpoints

- `GET  /health` — liveness + model metrics.
- `POST /predict` — single context → `{predicted_demand, purchase_intent, behavior}`.
- `GET  /insights?date=&location=&age_group=&event=&disaster=` — full charts payload.

`event` ∈ `auto | none | ramadan | eid_fitr | eid_adha | festival | disaster`.
`disaster` ∈ `none | flood | earthquake | heatwave` (used when `event=disaster`).

## Pointing the Flutter app at it

The app reads `ML_BASE_URL` (default `http://localhost:8000`):

```bash
flutter run --dart-define=ML_BASE_URL=http://localhost:8000      # web / desktop
flutter run --dart-define=ML_BASE_URL=http://10.0.2.2:8000       # Android emulator
```

If the service is unreachable the Insights screen falls back to an on-device
heuristic and labels the data source as *offline sample*.
