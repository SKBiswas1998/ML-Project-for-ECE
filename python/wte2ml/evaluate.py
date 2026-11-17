"""Error metrics. Targets are already log10 (or mV/dec for swing), so MAE/RMSE are
in decades (or mV/dec) directly."""
from __future__ import annotations

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE, MSE and R^2 of one prediction vector (any shape, flattened)."""
    y, p = np.ravel(y_true).astype(float), np.ravel(y_pred).astype(float)
    err = p - y
    mse = float(np.mean(err**2))
    var = float(np.var(y))
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(mse)),
        "MSE": mse,
        "R2": 1.0 - mse / var if var > 0 else float("nan"),
        "n": int(y.size),
    }


def skill_score(mse_model: float, mse_best_naive: float) -> float:
    """1 - MSE_model / MSE_best_naive: 0 = as good as the best naive baseline, 1 = perfect,
    negative = worse than the baseline."""
    return 1.0 - mse_model / mse_best_naive if mse_best_naive > 0 else float("nan")
