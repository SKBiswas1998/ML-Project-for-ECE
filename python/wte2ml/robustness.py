"""Input-robustness test: how much does a fitted model's error grow when its test inputs
are perturbed by Gaussian noise?

Motivation (project proposal): a surrogate is only useful if a small error in the inputs
(a temperature reading off by a kelvin, a vacancy density known only roughly) does not
change its prediction wildly. Noise is added to the TEST inputs only; the model is fitted
once on clean training data. Noise is measured in grid steps of each input, so one
number means the same thing for temperature (2 K steps) and vacancy (0.1 % steps).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from wte2ml.evaluate import regression_metrics


def input_steps(cfg: dict[str, Any], n_columns: int) -> np.ndarray:
    """Grid step of each X column, in its own unit: [T step K, vacancy step %][, gate step V]."""
    s = cfg["splits"]
    steps = [s["temp_step_K"], s["vac_step_pct"], cfg["robustness"]["gate_step_V"]]
    return np.asarray(steps[:n_columns], float)


def noisy(X: np.ndarray, sigma: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """X + independent Gaussian noise with standard deviation sigma[j] in column j."""
    return X + rng.standard_normal(X.shape) * sigma[None, :]


def noise_curve(model: Any, X_test: np.ndarray, y_test: np.ndarray, steps: np.ndarray,
                fractions: list[float], repeats: int, seed: int) -> pd.DataFrame:
    """Test MAE for each noise level (fraction of a grid step) and repeat.

    Fraction 0 is evaluated once (it is the clean test error). Returns one row per
    (noise_fraction, repeat) with MAE and RMSE in the target's units.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for frac in fractions:
        for rep in range(1 if frac == 0 else repeats):
            X = X_test if frac == 0 else noisy(X_test, frac * steps, rng)
            m = regression_metrics(y_test, np.ravel(model.predict(X)))
            rows.append({"noise_fraction": frac, "repeat": rep, "MAE": m["MAE"], "RMSE": m["RMSE"]})
    return pd.DataFrame(rows)


def summarise(curves: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean and std of MAE over repeats per noise level, plus the increase over clean MAE."""
    s = curves.groupby([*keys, "noise_fraction"]).MAE.agg(MAE_mean="mean", MAE_std="std").reset_index()
    clean = s[s.noise_fraction == 0].set_index(keys).MAE_mean.rename("MAE_clean")
    s = s.merge(clean, left_on=keys, right_index=True)
    return s.assign(MAE_increase=s.MAE_mean - s.MAE_clean)
