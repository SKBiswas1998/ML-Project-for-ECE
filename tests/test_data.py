"""Loader tests: the real data loads, matches the Stage 0 facts, and bad data is rejected."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from wte2ml.config import load_config
from wte2ml.data import check_curves, curve_matrix, load_dataset, load_layout


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def ds(cfg):
    return load_dataset(cfg)


def test_row_counts(ds):
    assert (len(ds.iv_train), len(ds.iv_test)) == (12760, 3200)
    assert (len(ds.metrics_train), len(ds.metrics_test)) == (638, 160)


def test_pooled_grid_is_full_38_by_21(ds):
    m = ds.metrics_pooled()
    assert m["T_K"].nunique() == 38 and m["vac_pct"].nunique() == 21
    assert len(m) == 38 * 21 == len(m.drop_duplicates(["T_K", "vac_pct"]))


def test_supplied_test_split_is_interpolation_only(ds):
    t = ds.metrics_test
    assert t["T_K"].between(6, 76).all() and t["vac_pct"].between(0.1, 1.9).all()
    assert ds.metrics_train["T_K"].nunique() == 38 and ds.metrics_train["vac_pct"].nunique() == 21


def test_log10_column_matches_current(ds):
    iv = ds.iv_pooled()
    assert np.allclose(iv["log10_I"], np.log10(iv["I_A"]), atol=1e-10)


def test_curve_matrix_shape_and_order(ds, cfg):
    keys, gate_V, Y = curve_matrix(ds.iv_pooled(), cfg)
    assert Y.shape == (798, 20) and gate_V.shape == (20,)
    assert np.all(np.diff(gate_V) > 0)
    assert keys.iloc[0].tolist() == [4, 0.0]
    # Row of the matrix equals the original rows of that curve.
    rows = ds.iv_pooled().query("T_K == 4 and vac_pct == 0.0").sort_values("VTG_minus_VBG_V")
    assert np.array_equal(Y[0], rows["log10_I"].to_numpy())


def test_supplied_ion_ioff_are_first_and_last_gate_points(ds, cfg):
    """Stage 0 finding: Ion = I(V=0), Ioff = I(V=1.9 V), NOT the curve minimum."""
    keys, _, Y = curve_matrix(ds.iv_pooled(), cfg, value="I_A")
    m = ds.metrics_pooled().set_index(["T_K", "vac_pct"]).loc[list(map(tuple, keys.to_numpy()))]
    assert np.allclose(m["Ion_A"], Y[:, 0], rtol=1e-12)
    assert np.allclose(m["Ioff_A"], Y[:, -1], rtol=1e-12)


def test_layout_is_nested_and_matches_csv(ds):
    layout = load_layout(load_config())
    assert layout.is_nested()
    assert [len(a) for a in layout.vac_atoms] == layout.n_vac.tolist()
    per_level = ds.metrics_pooled().groupby("vac_pct")["n_vac_atoms"].first()
    assert np.array_equal(per_level.to_numpy(), layout.n_vac)
    assert min(a.min() for a in layout.vac_atoms if a.size) >= 1   # 1-based, no index 0
    assert max(a.max() for a in layout.vac_atoms if a.size) <= layout.n_atoms


def test_missing_point_is_rejected(ds, cfg):
    broken = ds.iv_pooled().iloc[1:]
    with pytest.raises(ValueError, match="points"):
        check_curves(broken, cfg)


def test_shifted_gate_grid_is_rejected(ds, cfg):
    broken = ds.iv_pooled().copy()
    broken.loc[0, "VTG_minus_VBG_V"] += 1e-3
    with pytest.raises(ValueError, match="gate grids"):
        check_curves(broken, cfg)


def test_wrong_row_count_is_rejected(cfg):
    bad = copy.deepcopy(cfg)
    bad["data"]["expected_rows"]["iv_test"] = 3199
    with pytest.raises(ValueError, match="rows"):
        load_dataset(bad)
