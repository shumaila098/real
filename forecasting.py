"""Tenant-aware demand forecasting from the tenant's OWN daily order series.

This replaces grid-scoring a context regressor with a real (if intentionally
simple) time-series method: weekday seasonality + a robust recent trend +
quantile bands from in-sample residuals, honestly validated by backtesting the
last week (SMAPE). No heavy dependencies — plain numpy on the ~60-point daily
series the Worker materializes per tenant.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

WEEKDAY_SHORT = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

MIN_DAYS = 14
MIN_ORDERS = 5


def _weekday_factors(y: np.ndarray, weekdays: np.ndarray) -> np.ndarray:
    """Mean per weekday / overall mean, shrunk toward 1 when data is thin."""
    overall = y.mean() if y.mean() > 0 else 1.0
    factors = np.ones(7)
    for wd in range(7):
        vals = y[weekdays == wd]
        if len(vals) == 0:
            continue
        raw = vals.mean() / overall
        weight = min(1.0, len(vals) / 6.0)  # full trust at ~6 observations
        factors[wd] = 1.0 + (raw - 1.0) * weight
    return np.clip(factors, 0.25, 4.0)


def _trend_per_day(y: np.ndarray) -> float:
    """Robust multiplicative daily trend from the last 28 days (log1p OLS,
    slope clipped so one viral day can't forecast the moon)."""
    tail = y[-28:]
    if len(tail) < 7:
        return 1.0
    x = np.arange(len(tail), dtype=float)
    ly = np.log1p(tail.astype(float))
    slope = float(np.polyfit(x, ly, 1)[0])
    slope = float(np.clip(slope, -0.05, 0.05))  # ±5%/day cap
    return float(np.exp(slope))


def _fit_predict(y: np.ndarray, weekdays: np.ndarray, horizon: int):
    """Fit on (y, weekdays) and forecast `horizon` days ahead. Returns
    (forecast, residual_std)."""
    factors = _weekday_factors(y, weekdays)
    trend = _trend_per_day(y)

    # Level: exponentially-weighted mean of the last 14 deseasonalized days.
    tail = y[-14:]
    tail_wd = weekdays[-14:]
    deseason = tail / factors[tail_wd]
    w = np.exp(np.linspace(-1.5, 0, len(deseason)))
    level = float(np.average(deseason, weights=w)) if len(deseason) else 0.0

    # In-sample one-step residuals for the uncertainty band.
    fitted = level * factors[weekdays[-28:]] if len(y) >= 28 else level * factors[weekdays]
    actual = y[-len(fitted):]
    resid_std = float(np.std(actual - fitted)) if len(fitted) > 3 else max(1.0, level * 0.5)

    last_wd = int(weekdays[-1])
    forecast = np.array(
        [
            max(0.0, level * (trend ** (t + 1)) * factors[(last_wd + t + 1) % 7])
            for t in range(horizon)
        ]
    )
    return forecast, resid_std


def smape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = (np.abs(actual) + np.abs(pred)) / 2.0
    mask = denom > 0
    if not mask.any():
        return 0.0
    return round(float(np.mean(np.abs(actual[mask] - pred[mask]) / denom[mask]) * 100), 1)


def tenant_forecast(series: list[dict], horizon: int = 7):
    """``series``: [{date: 'YYYY-MM-DD', orders: n}, …] oldest→newest,
    zero-filled. Returns None when there isn't enough real signal yet."""
    if not series or len(series) < MIN_DAYS:
        return None
    y = np.array([float(s.get("orders") or 0) for s in series])
    if y.sum() < MIN_ORDERS:
        return None
    dates = [date.fromisoformat(str(s["date"])) for s in series]
    weekdays = np.array([d.weekday() for d in dates])

    # Honest validation: refit without the last week and score it.
    backtest_smape = None
    if len(y) >= MIN_DAYS + 7 and y[:-7].sum() >= MIN_ORDERS:
        held_pred, _ = _fit_predict(y[:-7], weekdays[:-7], 7)
        backtest_smape = smape(y[-7:], held_pred)

    forecast, resid_std = _fit_predict(y, weekdays, horizon)
    z10, z90 = 1.2816, 1.2816
    points = []
    for t in range(horizon):
        d = dates[-1] + timedelta(days=t + 1)
        p50 = round(float(forecast[t]), 1)
        points.append(
            {
                "label": f"{WEEKDAY_SHORT[d.weekday()]} {d.day}",
                "date": d.isoformat(),
                "value": p50,
                "p10": round(max(0.0, float(forecast[t]) - z10 * resid_std), 1),
                "p90": round(float(forecast[t]) + z90 * resid_std, 1),
            }
        )
    return {
        "points": points,
        "backtest_smape": backtest_smape,
        "history_days": len(y),
        "history_orders": int(y.sum()),
        "method": "weekday-seasonal trend (tenant series)",
    }
