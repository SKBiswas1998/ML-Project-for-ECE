"""Uncertainty of the reported test errors, computed from saved predictions (no refits).

Cluster bootstrap: curves of one vacancy level share one layout and are strongly
dependent, so the resampling unit is the whole vacancy LEVEL, never a single curve
(resampling curves would treat 38 near-copies as 38 independent samples and give
falsely tight intervals). With few levels in a test set (5 for the vacancy tests) the
intervals are coarse and wide; that is the honest answer, not a defect.

GP coverage: share of test points whose truth lies inside the GP's nominal 68 % and
95 % predictive intervals (mean +/- 1 and 1.96 std).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

Z68, Z95 = 1.0, 1.959964   # standard-normal half-widths of the 68.27 % and 95 % intervals


def level_sums(pred: pd.DataFrame, step: float) -> pd.DataFrame:
    """Per vacancy level: sum of |error| and number of test rows (the bootstrap clusters)."""
    level = np.rint(pred.vac_pct.to_numpy(float) / step).astype(int)
    err = (pred.y_pred - pred.y_true).abs().to_numpy()
    return pd.DataFrame({"level": level, "abs_err": err}).groupby("level").abs_err.agg(["sum", "count"])


def resample_levels(n_levels: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, n_levels) indices: each bootstrap sample draws n_levels levels with replacement."""
    return rng.integers(0, n_levels, size=(n_boot, n_levels))


def bootstrap_mae(sums: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    """MAE of each bootstrap sample: pooled |error| over the drawn levels / their row count."""
    s, c = sums["sum"].to_numpy(), sums["count"].to_numpy()
    return s[idx].sum(axis=1) / c[idx].sum(axis=1)


def percentile_ci(samples: np.ndarray, level: float) -> tuple[float, float]:
    """Two-sided percentile interval at confidence ``level`` (e.g. 0.95)."""
    lo, hi = np.percentile(samples, [50 * (1 - level), 50 * (1 + level)])
    return float(lo), float(hi)


def gp_coverage(pred: pd.DataFrame) -> dict[str, float]:
    """Empirical coverage of the nominal 68 % and 95 % GP intervals, plus the mean predictive std.
    pred: rows with y_true, y_pred, y_std (GP only)."""
    z = ((pred.y_true - pred.y_pred).abs() / pred.y_std).to_numpy()
    return {"coverage_68": float(np.mean(z <= Z68)), "coverage_95": float(np.mean(z <= Z95)),
            "mean_std": float(pred.y_std.mean()), "n": len(z)}


def task_bootstrap(pred: pd.DataFrame, models: list[str], reference: str, cfg_u: dict[str, Any],
                   step: float, n_comparisons: int) -> pd.DataFrame:
    """For one task (protocol, target): MAE with cluster-bootstrap CI for each model, and the
    paired difference MAE_model - MAE_reference on the SAME resampled levels, with a 95 % and a
    Bonferroni-adjusted interval (n_comparisons = number of model-vs-reference tests reported)."""
    sums = {m: level_sums(pred[pred.model == m], step) for m in models}
    levels = sums[reference].index
    idx = resample_levels(len(levels), cfg_u["n_boot"], np.random.default_rng(cfg_u["seed"]))
    boot = {m: bootstrap_mae(sums[m].reindex(levels), idx) for m in models}
    level_bonf = 1 - (1 - cfg_u["confidence"]) / n_comparisons
    rows = []
    for m in models:
        diff = boot[m] - boot[reference]
        rows.append({"model": m, "n_levels": len(levels),
                     "MAE": float(sums[m]["sum"].sum() / sums[m]["count"].sum()),
                     **dict(zip(("MAE_lo", "MAE_hi"), percentile_ci(boot[m], cfg_u["confidence"]))),
                     **dict(zip(("diff_lo", "diff_hi"), percentile_ci(diff, cfg_u["confidence"]))),
                     **dict(zip(("diff_lo_bonf", "diff_hi_bonf"), percentile_ci(diff, level_bonf)))})
    return pd.DataFrame(rows).assign(reference=reference, n_comparisons=n_comparisons)
