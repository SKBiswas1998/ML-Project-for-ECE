"""Model tests: baselines and wrappers do what they claim, on tiny synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wte2ml.config import load_config
from wte2ml.evaluate import regression_metrics, skill_score
from wte2ml.models import Gatewise, VacancyLevelMean, build_model, tune_and_fit
from wte2ml.splits import fold_indices, temperature_forward

GATES = np.array([0.0, 0.5, 1.0])


def _grid_table() -> pd.DataFrame:
    """Synthetic 4 T x 5 vacancy x 3 gate table, y = T/100 - vac + 2*gate."""
    T, v, g = np.meshgrid([4, 6, 8, 10], [0.0, 0.1, 0.2, 0.3, 0.4], GATES, indexing="ij")
    df = pd.DataFrame({"T_K": T.ravel(), "vac_pct": v.ravel(), "gate_V": g.ravel()})
    return df.assign(y=df.T_K / 100 - df.vac_pct + 2 * df.gate_V)


def test_vacancy_mean_uses_nearest_training_level():
    X = np.array([[4, 0.0], [6, 0.0], [4, 0.5]])
    m = VacancyLevelMean().fit(X, np.array([1.0, 3.0, 10.0]))
    assert m.predict(np.array([[9, 0.1], [9, 0.4]])).tolist() == [2.0, 10.0]


def test_gatewise_routes_each_gate_to_its_own_model():
    df = _grid_table()
    X, y = df[["T_K", "vac_pct", "gate_V"]].to_numpy(float), df.y.to_numpy()
    m = Gatewise(VacancyLevelMean(), decimals=6).fit(X, y)
    assert sorted(m.models_) == GATES.tolist()
    # Same (T, vac), different gate -> predictions differ by exactly 2 * delta gate.
    p = m.predict(np.array([[6, 0.2, 0.0], [6, 0.2, 1.0]]))
    assert p[1] - p[0] == pytest.approx(2.0)


def test_gp_reproduces_smooth_function_and_returns_std():
    cfg = load_config()
    df = _grid_table()
    est, grid = build_model("gp", cfg, pointwise=True)
    assert grid == {}
    X = df[["T_K", "vac_pct", "gate_V"]].to_numpy(float)
    est.fit(X, df.y.to_numpy())
    mean, std = est.predict(X, return_std=True)
    assert np.allclose(mean, df.y, atol=1e-2) and np.all(std >= 0)


def test_models_without_grid_skip_cv():
    cfg = load_config()
    est, grid = build_model("naive_nn", cfg, pointwise=False)
    _, params, cv_mse = tune_and_fit(est, grid, np.array([[4, 0.0], [6, 0.1]]),
                                     np.array([1.0, 2.0]), cv=[], cfg=cfg)
    assert params == {} and np.isnan(cv_mse)


def test_fold_indices_are_positions_of_the_right_curves():
    cfg = load_config()
    df = _grid_table().assign(split="train")
    folds = [temperature_forward(df, {**cfg, "splits": {**cfg["splits"], "temp_train_max_K": 6}})]
    (tr, va), = fold_indices(df, folds)
    assert set(df.loc[tr, "T_K"]) == {4, 6} and set(df.loc[va, "T_K"]) == {8, 10}
    assert len(tr) + len(va) == len(df)


def test_fold_indices_refuses_shuffled_index():
    df = _grid_table().sample(frac=1, random_state=0)
    with pytest.raises(ValueError):
        fold_indices(df, [])


def test_metrics_and_skill():
    m = regression_metrics(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 3.0]))
    assert m["MAE"] == pytest.approx(1 / 3) and m["MSE"] == pytest.approx(1 / 3)
    assert skill_score(0.25, 1.0) == 0.75 and skill_score(2.0, 1.0) == -1.0
