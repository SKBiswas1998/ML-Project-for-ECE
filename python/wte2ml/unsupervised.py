"""PCA of the log10 I-V curves and clustering of devices (unsupervised learning).

Descriptive only: everything here is fitted on all 798 curves to see how much
structure the I-V curves have, and whether devices fall into natural groups that
match a physics label (on/off ratio classes). Nothing here feeds a prediction, so
it does not touch the train/test protocols.

Input is the (n_curves, 20) matrix of log10 I [decades] from ``data.curve_matrix``.
Columns share one unit, so PCA centres but does not rescale them.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture


def fit_pca(Y: np.ndarray, n_components: int) -> PCA:
    """PCA of the curves (rows) over gate points (columns), mean-centred, not scaled."""
    return PCA(n_components=n_components).fit(Y)


def explained_variance_table(pca: PCA) -> pd.DataFrame:
    """Per component: share of total variance and cumulative share (both 0-1)."""
    ratio = pca.explained_variance_ratio_
    return pd.DataFrame({"component": np.arange(1, ratio.size + 1),
                         "explained_variance_ratio": ratio, "cumulative": np.cumsum(ratio)})


def n_components_for(pca: PCA, share: float) -> int:
    """Fewest leading components whose cumulative explained variance reaches ``share``
    (all fitted components if the share is never reached)."""
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    return int(min(np.searchsorted(cumulative, share) + 1, cumulative.size))


def component_table(pca: PCA, gate_V: np.ndarray) -> pd.DataFrame:
    """Long table of component loadings vs gate voltage, plus the mean curve (component 0).
    Shows what each component does to a curve, e.g. 'deepens the off state'."""
    rows = [pd.DataFrame({"component": 0, "gate_V": gate_V, "loading": pca.mean_})]
    rows += [pd.DataFrame({"component": i + 1, "gate_V": gate_V, "loading": v})
             for i, v in enumerate(pca.components_)]
    return pd.concat(rows, ignore_index=True)


def ion_ioff_classes(log10_ratio: np.ndarray, edges: list[float]) -> np.ndarray:
    """Physics label per curve: 0 below edges[0], 1 between, 2 at or above edges[1]."""
    return np.digitize(np.asarray(log10_ratio, float), edges)


def cluster(Z: np.ndarray, method: str, k: int, cfg_u: dict[str, Any], seed: int
            ) -> tuple[np.ndarray, float]:
    """Cluster the rows of Z (PCA scores) by k-means or a Gaussian mixture.
    Returns (labels, BIC); BIC (lower = better fit per parameter) is NaN for k-means."""
    if method == "kmeans":
        return KMeans(k, n_init=cfg_u["n_init"], random_state=seed).fit_predict(Z), float("nan")
    if method == "gmm":
        gmm = GaussianMixture(k, covariance_type=cfg_u["gmm_covariance"], n_init=cfg_u["n_init"],
                              random_state=seed).fit(Z)
        return gmm.predict(Z), float(gmm.bic(Z))
    raise ValueError(method)

def cluster_scores(Z: np.ndarray, classes: np.ndarray, cfg_u: dict[str, Any], seed: int
                   ) -> pd.DataFrame:
    """For each method and k: silhouette (-1..1, higher = better separated), BIC (GMM only)
    and adjusted Rand index against the Ion/Ioff classes (0 = chance, 1 = identical)."""
    rows = []
    for method in ("kmeans", "gmm"):
        for k in cfg_u["k_range"]:
            lab, bic = cluster(Z, method, k, cfg_u, seed)
            rows.append({"method": method, "k": k, "silhouette": float(silhouette_score(Z, lab)), "bic": bic,
                         "ari_vs_ion_ioff_classes": float(adjusted_rand_score(classes, lab)),
                         "smallest_cluster": int(np.bincount(lab).min())})
    return pd.DataFrame(rows)
