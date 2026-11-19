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
    """One row per (protocol, target, model): MAE, RMSE, R2, skill (MLP = mean over seeds of the per-seed metrics,
    with MAE_std), the rank of the model by MAE within its task, and whether it is best.

    classical: results/metrics_classical.csv; mlp_per_seed: results/metrics_mlp.csv.
    """
    task = ["protocol", "feature", "target"]
    cols = ["MAE", "RMSE", "R2", "skill"]
    mlp = mlp_per_seed.groupby(task)[cols].mean().join(
        mlp_per_seed.groupby(task).MAE.std().rename("MAE_std")).reset_index().assign(model="mlp")
    both = pd.concat([classical[[*task, "model", *cols]], mlp], ignore_index=True)
    both["rank"] = both.groupby(task).MAE.rank(method="min").astype(int)
    naive = (classical[classical.model.isin(["naive_vac_mean", "naive_nn"])]
             .pivot_table(index=task, columns="model", values="MAE").add_prefix("MAE_").reset_index())
    both = both.merge(naive, on=task, how="left")   # both naive baselines shown on every row
    return both.assign(best=both["rank"] == 1).sort_values([*task, "rank"], ignore_index=True)


def pooled_predictions(classical: pd.DataFrame, mlp: pd.DataFrame) -> pd.DataFrame:
    """All test predictions in one table: classical models as stored, the MLP as the mean of
    its seeds (results/predictions_classical.csv.gz and predictions_mlp.csv.gz)."""
    keys = ["protocol", "feature", "target", "T_K", "vac_pct", "gate_V"]
    m = (mlp.assign(gate_V=mlp.gate_V.fillna(-1)).groupby(keys, as_index=False)
         .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean")).assign(model="mlp", y_std=np.nan))
    m["gate_V"] = m.gate_V.replace(-1, np.nan)
    return pd.concat([classical, m], ignore_index=True)
