"""Robustness tests: noise has the intended scale per input, and its effect is measured honestly."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from wte2ml.config import load_config
from wte2ml.robustness import input_steps, noise_curve, noisy, summarise


def test_noise_is_scaled_per_column_in_grid_steps():
    """1 grid step of noise means ~2 K on temperature and ~0.1 % on vacancy, not one shared std."""
    cfg = load_config()
    steps = input_steps(cfg, 2)
    assert steps.tolist() == [2.0, 0.1]
    X = np.zeros((20000, 2))
    e = noisy(X, steps, np.random.default_rng(0))
    assert e.std(axis=0) == pytest.approx(steps, rel=0.03)


def test_clean_level_equals_plain_test_error_and_error_grows_with_noise():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, (200, 2))
    y = 3 * X[:, 0] - 2 * X[:, 1]
    model = LinearRegression().fit(X, y)                      # exact on clean inputs
    curve = noise_curve(model, X, y, np.array([0.1, 0.1]), [0.0, 0.5, 1.0], repeats=3, seed=0)
    assert (curve.noise_fraction == 0).sum() == 1             # clean level evaluated once
    assert curve[curve.noise_fraction == 0].MAE.iloc[0] == pytest.approx(0, abs=1e-12)
    s = summarise(curve.assign(model="lin"), ["model"])
    assert s.MAE_mean.is_monotonic_increasing and (s.MAE_increase >= 0).all()


def test_same_seed_same_noise():
    X, model = np.ones((50, 2)), LinearRegression().fit(np.eye(2), [0.0, 1.0])
    a = noise_curve(model, X, np.zeros(50), np.array([1.0, 1.0]), [1.0], repeats=2, seed=7)
    b = noise_curve(model, X, np.zeros(50), np.array([1.0, 1.0]), [1.0], repeats=2, seed=7)
    pd.testing.assert_frame_equal(a, b)
