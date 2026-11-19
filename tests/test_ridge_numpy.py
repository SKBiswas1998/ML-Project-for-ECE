"""NumPy ridge must reproduce the pipeline's sklearn ridge (StandardScaler + Ridge) exactly."""
from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from wte2ml.config import load_config
from wte2ml.data import load_dataset
from wte2ml.models import build_model, features
from wte2ml.ridge_numpy import coefficient_path, effective_dof, fit_ridge
from wte2ml.targets import curve_targets


@pytest.fixture(scope="module")
def real_xy():
    """Real curve-level task: X = (T_K, vac_pct) with cubic terms, y = log10 Ion/Ioff."""
    cfg = load_config()
    ds = load_dataset(cfg)
    curves = curve_targets(ds.metrics_pooled(), ds.iv_pooled(), cfg)
    X = PolynomialFeatures(3, include_bias=False).fit_transform(features(curves, "vac_pct", None))
    return X, curves["log10_Ion_Ioff"].to_numpy()


@pytest.mark.parametrize("alpha", [1e-4, 1e-2, 1.0, 100.0])
def test_matches_sklearn_pipeline_ridge(real_xy, alpha):
    X, y = real_xy
    ours = fit_ridge(X, y, alpha)
    sk = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X, y)
    np.testing.assert_allclose(ours.predict(X), sk.predict(X), rtol=0, atol=1e-8)
    np.testing.assert_allclose(ours.beta, sk[-1].coef_, rtol=1e-8, atol=1e-8)
    assert ours.intercept == pytest.approx(sk[-1].intercept_, abs=1e-10)


def test_same_model_as_the_pipeline_builds():
    """The pipeline's own 'ridge' (models.build_model) is the model reproduced here."""
    est, grid = build_model("ridge", load_config(), pointwise=False)
    assert [type(s).__name__ for s in est] == ["StandardScaler", "Ridge"] and "ridge__alpha" in grid


def test_effective_dof_limits_and_monotone(real_xy):
    X, _ = real_xy
    p = X.shape[1]
    assert effective_dof(X, 0.0) == pytest.approx(p)
    dofs = [effective_dof(X, a) for a in (1e-2, 1.0, 100.0, 1e6)]
    assert all(a > b for a, b in pairwise(dofs)) and dofs[-1] < 0.01


def test_coefficient_path_shrinks(real_xy):
    X, y = real_xy
    path = coefficient_path(X, y, np.array([1e-4, 1.0, 1e4]))
    norms = np.linalg.norm(path, axis=1)
    assert norms[0] > norms[1] > norms[2]
