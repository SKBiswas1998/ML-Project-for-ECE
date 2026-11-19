"""Uncertainty tests: the bootstrap resamples whole vacancy levels; GP coverage is computed right."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wte2ml.uncertainty import (
    bootstrap_mae,
    gp_coverage,
    level_sums,
    percentile_ci,
    resample_levels,
    task_bootstrap,
)

CFG_U = {"n_boot": 2000, "confidence": 0.95, "seed": 0}


def _clustered(n_levels: int = 8, per_level: int = 38, seed: int = 0) -> pd.DataFrame:
    """Test predictions whose error is shared within a level (like one layout): strong clustering."""
    rng = np.random.default_rng(seed)
    level_err = rng.exponential(1.0, n_levels)
    vac = np.repeat(np.arange(n_levels) * 0.1, per_level)
    err = np.repeat(level_err, per_level) + 0.01 * rng.standard_normal(n_levels * per_level)
    return pd.DataFrame({"vac_pct": vac, "y_true": 0.0, "y_pred": err})


def test_cluster_bootstrap_is_wider_than_naive_curve_bootstrap():
    """THE point of the cluster bootstrap: with errors shared inside a level, resampling single
    curves pretends there are 304 independent samples and gives a falsely narrow interval."""
    pred = _clustered()
    sums = level_sums(pred, 0.1)
    rng = np.random.default_rng(0)
    lo, hi = percentile_ci(bootstrap_mae(sums, resample_levels(len(sums), 2000, rng)), 0.95)
    err = (pred.y_pred - pred.y_true).abs().to_numpy()
    naive = err[rng.integers(0, len(err), size=(2000, len(err)))].mean(axis=1)
    nlo, nhi = percentile_ci(naive, 0.95)
    assert (hi - lo) > 5 * (nhi - nlo)


def test_level_sums_group_by_level_and_reproduce_mae():
    pred = _clustered(4, 10)
    sums = level_sums(pred, 0.1)
    assert sums.index.tolist() == [0, 1, 2, 3] and (sums["count"] == 10).all()
    full = bootstrap_mae(sums, np.arange(4)[None, :])[0]           # all levels once = plain MAE
    assert full == pytest.approx((pred.y_pred - pred.y_true).abs().mean())


def test_paired_difference_against_itself_is_zero_and_flags_a_clearly_worse_model():
    good = _clustered().assign(model="good")
    bad = good.assign(model="bad", y_pred=good.y_pred + 1.0)       # 1 decade worse everywhere
    out = task_bootstrap(pd.concat([good, bad]), ["bad", "good"], "good", CFG_U, 0.1, n_comparisons=1)
    g, b = out.set_index("model").loc["good"], out.set_index("model").loc["bad"]
    assert g.diff_lo == g.diff_hi == 0.0
    assert b.diff_lo_bonf > 0.9 and b.diff_hi_bonf < 1.1


def test_gp_coverage_of_a_calibrated_gaussian():
    rng = np.random.default_rng(1)
    std = rng.uniform(0.5, 2.0, 20000)
    pred = pd.DataFrame({"y_pred": 0.0, "y_std": std, "y_true": std * rng.standard_normal(20000)})
    c = gp_coverage(pred)
    assert c["coverage_68"] == pytest.approx(0.6827, abs=0.01)
    assert c["coverage_95"] == pytest.approx(0.95, abs=0.006)
