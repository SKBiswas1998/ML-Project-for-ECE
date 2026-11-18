"""MLP tests: reproducible, early stopping holds out whole curves, learns a simple function."""
from __future__ import annotations

import numpy as np

from wte2ml.neural import MLPRegressor, curve_holdout_mask

FAST = {"width": 32, "depth": 2, "max_epochs": 300, "patience": 50, "eval_every": 5,
        "batch_size": 64, "learning_rate": 1e-2}


def _data(rng_seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """6 T x 6 vacancy curves x 5 gate points, y = smooth function of all three inputs."""
    T, v, g = np.meshgrid(np.arange(4, 16, 2), np.linspace(0, 1, 6), np.linspace(0, 2, 5), indexing="ij")
    X = np.column_stack([T.ravel(), v.ravel(), g.ravel()]).astype(float)
    return X, -5 - 3 * X[:, 2] * (1 - 0.3 * X[:, 1]) + 0.01 * X[:, 0]


def test_early_stopping_holds_out_whole_curves():
    X, _ = _data()
    mask = curve_holdout_mask(X, 0.25, np.random.default_rng(0))
    curves = {tuple(r) for r in X[mask, :2]}
    for c in curves:   # every row of a held-out curve is held out
        assert mask[(X[:, 0] == c[0]) & (X[:, 1] == c[1])].all()
    assert len(curves) == 9   # 25 % of 36 curves


def test_same_seed_gives_identical_predictions():
    X, y = _data()
    p1 = MLPRegressor(**FAST, seed=3).fit(X, y).predict(X)
    p2 = MLPRegressor(**FAST, seed=3).fit(X, y).predict(X)
    p3 = MLPRegressor(**FAST, seed=4).fit(X, y).predict(X)
    assert np.array_equal(p1, p2) and not np.array_equal(p1, p3)


def test_learns_smooth_function():
    X, y = _data()
    p = MLPRegressor(**FAST).fit(X, y).predict(X)
    assert np.mean(np.abs(p - y)) < 0.1 * np.std(y)


def test_multi_output_predicts_whole_curve():
    """One row per curve, 5 outputs (the gate points): prediction shape and accuracy."""
    X, y = _data()
    Xc, Yc = X[::5, :2], y.reshape(-1, 5)          # 36 curves x 5 gate points
    m = MLPRegressor(**FAST).fit(Xc, Yc)
    P = m.predict(Xc)
    assert P.shape == (36, 5)
    assert np.mean(np.abs(P - Yc)) < 0.1 * np.std(Yc)
