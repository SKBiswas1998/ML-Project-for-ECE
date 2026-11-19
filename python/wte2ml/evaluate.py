"""Error metrics. Targets are already log10 (or mV/dec for swing), so MAE/RMSE are
in decades (or mV/dec) directly."""
from __future__ import annotations

import numpy as np
import pandas as pd


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


def comparison_table(classical: pd.DataFrame, mlp_per_seed: pd.DataFrame) -> pd.DataFrame:
    """One row per (protocol, target, model): MAE, RMSE, R2, skill (MLP = mean over seeds,
    with MAE_std), the rank of the model by MAE within its task, and whether it is best.

    classical: results/metrics_classical.csv; mlp_per_seed: results/metrics_mlp.csv.
    """
    task = ["protocol", "feature", "target"]
    cols = ["MAE", "RMSE", "R2", "skill"]
    mlp = mlp_per_seed.groupby(task)[cols].mean().join(
        mlp_per_seed.groupby(task).MAE.std().rename("MAE_std")).reset_index().assign(model="mlp")
    both = pd.concat([classical[[*task, "model", *cols]], mlp], ignore_index=True)
    both["rank"] = both.groupby(task).MAE.rank(method="min").astype(int)
    return both.assign(best=both["rank"] == 1).sort_values([*task, "rank"], ignore_index=True)
