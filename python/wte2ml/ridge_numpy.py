"""Ridge regression written out in NumPy, to show what the sklearn model in the pipeline computes.

Same model as models.build_model("ridge"): StandardScaler (training mean, population std)
followed by sklearn Ridge with an unpenalised intercept. In standardised inputs
Z = (X - mean) / std and centred target, the coefficients solve

    (Z^T Z + alpha I) beta = Z^T (y - mean(y)),   intercept = mean(y)  (on the Z scale).

A test checks that predictions and coefficients agree with sklearn to numerical precision.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RidgeFit:
    """Fitted ridge model. beta: (p,) coefficients on standardised inputs; intercept: mean of y."""

    x_mean: np.ndarray
    x_std: np.ndarray
    beta: np.ndarray
    intercept: float
    alpha: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        """(n,) predictions for raw inputs X (n, p)."""
        return ((np.asarray(X, float) - self.x_mean) / self.x_std) @ self.beta + self.intercept


def _standardise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z, mean, std with sklearn StandardScaler's convention (ddof = 0; constant columns -> std 1)."""
    mean, std = X.mean(axis=0), X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std, mean, std


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> RidgeFit:
    """Closed-form ridge on standardised X; X (n, p) raw inputs, y (n,) target, alpha >= 0."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    Z, mean, std = _standardise(X)
    yc = y - y.mean()
    beta = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ yc)
    return RidgeFit(mean, std, beta, float(y.mean()), float(alpha))


def effective_dof(X: np.ndarray, alpha: float) -> float:
    """Effective number of parameters, trace of the hat matrix: sum_j d_j^2 / (d_j^2 + alpha),
    with d_j the singular values of standardised X. Equals p at alpha = 0, falls to 0 as alpha grows."""
    d = np.linalg.svd(_standardise(np.asarray(X, float))[0], compute_uv=False)
    return float(np.sum(d**2 / (d**2 + alpha)))


def coefficient_path(X: np.ndarray, y: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """(len(alphas), p) coefficients for each alpha: how the penalty shrinks them towards 0."""
    return np.vstack([fit_ridge(X, y, a).beta for a in alphas])
