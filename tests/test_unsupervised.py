"""Unsupervised tests: PCA bookkeeping, class edges, and clustering on data with a known answer."""
from __future__ import annotations

import numpy as np
import pytest

from wte2ml.config import load_config
from wte2ml.unsupervised import (
    cluster,
    cluster_scores,
    explained_variance_table,
    fit_pca,
    ion_ioff_classes,
    n_components_for,
)


def _two_groups(n: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Curves from two families (deep vs shallow off state) plus tiny noise; true labels."""
    rng = np.random.default_rng(0)
    gate = np.linspace(0, 1.9, 20)
    deep, shallow = -5 - 8 * gate, -5 - 2 * gate
    truth = np.repeat([0, 1], n // 2)
    Y = np.where(truth[:, None] == 0, deep, shallow) + 0.01 * rng.standard_normal((n, 20))
    return Y, truth


def test_explained_variance_is_cumulative_and_bounded():
    Y, _ = _two_groups()
    t = explained_variance_table(fit_pca(Y, 5))
    assert np.all(np.diff(t.cumulative) >= 0) and t.cumulative.iloc[-1] <= 1 + 1e-12
    assert t.explained_variance_ratio.iloc[0] > 0.99      # one family difference dominates


def test_n_components_for_share():
    Y, _ = _two_groups()
    pca = fit_pca(Y, 5)
    assert n_components_for(pca, 0.5) == 1
    assert n_components_for(pca, 1.0 - 1e-12) <= 5


def test_ion_ioff_class_edges_are_inclusive_upper():
    assert ion_ioff_classes(np.array([3.99, 4.0, 8.99, 9.0, 13.4]), [4.0, 9.0]).tolist() == [0, 1, 1, 2, 2]


@pytest.mark.parametrize("method", ["kmeans", "gmm"])
def test_clustering_recovers_known_groups(method):
    """Two well-separated families must come out as the two clusters (ARI = 1)."""
    Y, truth = _two_groups()
    Z = fit_pca(Y, 3).transform(Y)
    u = load_config()["unsupervised"]
    labels, bic = cluster(Z, method, 2, u, seed=0)
    assert len(set(zip(labels, truth))) == 2               # a relabelling of the truth
    assert np.isnan(bic) == (method == "kmeans")
    scores = cluster_scores(Z, truth, {**u, "k_range": [2]}, seed=0)
    assert np.allclose(scores.ari_vs_ion_ioff_classes, 1.0) and (scores.silhouette > 0.9).all()
