"""Model zoo behind one interface: sklearn-style ``fit(X, y)`` / ``predict(X)``.

Feature matrix X columns (built by ``features``): [T_K, vacancy] for curve-level
targets, [T_K, vacancy, gate_V] for log10_I. Every model that needs it standardises
X inside its own Pipeline, so scaling statistics come from the training fold only.

Any object with fit/predict (and optionally predict(X, return_std=True)) plugs in,
which is the slot a later PINN would use (``Regressor`` protocol below).
"""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from wte2ml.neural import MLPRegressor

NAIVE_MODELS = ("naive_vac_mean", "naive_nn")
MODELS = (*NAIVE_MODELS, "ridge", "poly_ridge", "boosting", "gp")


class Regressor(Protocol):
    """What run_experiments needs from a model (sklearn estimators satisfy it)."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> Regressor: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


def features(table: pd.DataFrame, vacancy_feature: str, gate_column: str | None) -> np.ndarray:
    """X matrix (n_rows, 2 or 3): T_K [K], vacancy [% or atoms], optionally gate [V]."""
    cols = ["T_K", vacancy_feature] + ([gate_column] if gate_column else [])
    return table[cols].to_numpy(float)


class VacancyLevelMean(RegressorMixin, BaseEstimator):
    """Naive baseline: mean target of the training curves at the nearest vacancy level.

    Uses only column 1 of X (vacancy). For a held-out level the nearest training
    level is used (ties -> the lower level), which is the honest naive guess.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> VacancyLevelMean:
        s = pd.Series(np.asarray(y, float)).groupby(np.asarray(X)[:, 1]).mean()
        self.levels_, self.means_ = s.index.to_numpy(float), s.to_numpy(float)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        idx = np.argmin(np.abs(np.asarray(X)[:, [1]] - self.levels_[None, :]), axis=1)
        return self.means_[idx]


class Gatewise(RegressorMixin, BaseEstimator):
    """Fit one copy of ``estimator`` per gate voltage, on the (T, vacancy) columns.

    Turns any curve-level model into a log10_I model. Exists so the GP never sees
    more than one point per curve (<= 638 points, not 12,760) and so the naive
    baselines compare like with like at each gate point. Last X column = gate [V].
    """

    def __init__(self, estimator: Any = None, decimals: int = 6, n_jobs: int = 1) -> None:
        self.estimator = estimator
        self.decimals = decimals
        self.n_jobs = n_jobs

    def _gates(self, X: np.ndarray) -> np.ndarray:
        return np.round(np.asarray(X)[:, -1], self.decimals)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Gatewise:
        g, y = self._gates(X), np.asarray(y, float)
        gates = np.unique(g)
        fitted = Parallel(n_jobs=self.n_jobs)(
            delayed(clone(self.estimator).fit)(X[g == v, :-1], y[g == v]) for v in gates)
        self.models_ = dict(zip(gates, fitted))
        return self

    def predict(self, X: np.ndarray, return_std: bool = False):
        g = self._gates(X)
        mean, std = np.full(len(g), np.nan), np.full(len(g), np.nan)
        for v in np.unique(g):
            m = g == v
            out = self.models_[v].predict(X[m, :-1], **({"return_std": True} if return_std else {}))
            mean[m], std[m] = (out if return_std else (out, np.nan))
        return (mean, std) if return_std else mean


def _gp_kernel(name: str, n_dims: int, cfg_gp: dict[str, Any]):
    """Amplitude x (RBF | Matern 5/2) with one bounded length scale per input, plus white noise."""
    ls, bounds = np.ones(n_dims), tuple(cfg_gp["length_scale_bounds"])
    base = {"rbf": RBF(ls, bounds), "matern25": Matern(ls, bounds, nu=2.5)}[name]
    return ConstantKernel(1.0) * base + WhiteKernel(cfg_gp["noise_init"], tuple(cfg_gp["noise_bounds"]))


def _curve_model(name: str, cfg: dict[str, Any]) -> tuple[Any, dict[str, list]]:
    """Estimator and grid for a model acting on (T, vacancy) columns only."""
    m, seed = cfg["models"], cfg["models"]["seed"]
    if name == "naive_vac_mean":
        return VacancyLevelMean(), {}
    if name == "naive_nn":
        return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=1)), {}
    if name == "gp":
        # Kernel family chosen by blocked CV (Stage 7 decision); length scales and noise of the
        # chosen kernel are still fitted by marginal likelihood inside each fit.
        gp = GaussianProcessRegressor(normalize_y=True, random_state=seed,
                                      n_restarts_optimizer=m["gp"]["n_restarts_optimizer"])
        kernels = [_gp_kernel(k, 2, m["gp"]) for k in m["gp"]["kernels"]]
        return make_pipeline(StandardScaler(), gp), {"gaussianprocessregressor__kernel": kernels}
    raise ValueError(name)


def _joint_model(name: str, cfg: dict[str, Any]) -> tuple[Any, dict[str, list]]:
    """Estimator and grid for a model acting on all columns of X jointly."""
    m, seed = cfg["models"], cfg["models"]["seed"]
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": m["ridge"]["alpha"]}
    if name == "poly_ridge":
        pipe = Pipeline([("scale", StandardScaler()), ("poly", PolynomialFeatures()), ("ridge", Ridge())])
        return pipe, {"poly__degree": m["poly_ridge"]["degree"], "ridge__alpha": m["poly_ridge"]["alpha"]}
    if name == "boosting":
        b = m["boosting"]
        gbm = HistGradientBoostingRegressor(learning_rate=b["learning_rate"], random_state=seed)
        return gbm, {"max_depth": b["max_depth"], "max_iter": b["max_iter"]}
    raise ValueError(name)


def _mlp_model(cfg: dict[str, Any]) -> tuple[Any, dict[str, list]]:
    """PyTorch MLP; width, depth and weight decay tuned by blocked CV. For log10_I the
    MLP step feeds it one row per curve with 20 outputs (see run_experiments)."""
    m, c = cfg["models"]["mlp"], cfg["compute"]
    est = MLPRegressor(dropout=m["dropout"], learning_rate=m["learning_rate"],
                       max_epochs=m["max_epochs"], patience=m["patience"],
                       eval_every=m["eval_every"], batch_size=m["batch_size"],
                       early_stop_fraction=m["early_stop_fraction"], seed=m["seeds"][0],
                       device=c["device"], threads=c["torch_threads"])
    return est, {"width": m["width"], "depth": m["depth"], "weight_decay": m["weight_decay"]}


def build_model(name: str, cfg: dict[str, Any], pointwise: bool) -> tuple[Any, dict[str, list]]:
    """(unfitted estimator, hyperparameter grid) for model ``name``.

    pointwise=True means X has a gate column; curve-level models (naive, GP) are
    then wrapped in ``Gatewise`` and their grid keys get the ``estimator__`` prefix.
    """
    if name in ("ridge", "poly_ridge", "boosting"):
        return _joint_model(name, cfg)
    if name == "mlp":
        return _mlp_model(cfg)
    est, grid = _curve_model(name, cfg)
    if not pointwise:
        return est, grid
    wrapped = Gatewise(est, decimals=cfg["inputs"]["gate_decimals"], n_jobs=cfg["compute"]["n_jobs"])
    return wrapped, {f"estimator__{k}": v for k, v in grid.items()}


def tune_and_fit(est: Any, grid: dict[str, list], X: np.ndarray, y: np.ndarray,
                 cv: list[tuple[np.ndarray, np.ndarray]], cfg: dict[str, Any]
                 ) -> tuple[Any, dict[str, Any], float]:
    """Pick hyperparameters by blocked CV (given fold indices), refit on all of X.

    Returns (fitted model, best params, best mean CV MSE). CV only exists to choose
    hyperparameters, so a model with no grid (naive baselines, GP) is fitted directly
    and its CV MSE is NaN.
    """
    if not grid:
        return clone(est).fit(X, y), {}, float("nan")
    search = GridSearchCV(est, grid or {}, cv=cv, scoring=cfg["experiments"]["cv_scoring"],
                          n_jobs=cfg["compute"]["n_jobs"], refit=True, error_score="raise")
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, -float(search.best_score_)
