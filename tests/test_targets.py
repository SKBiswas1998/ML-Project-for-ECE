"""Target tests: synthetic curves with known answers, plus invariants on the real data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wte2ml.config import load_config
from wte2ml.data import load_dataset
from wte2ml.targets import (
    boltzmann_swing,
    curve_targets,
    dynamic_power,
    falling_branch,
    raw_grid_swing,
    steepest_swing,
)

GATE = np.round(np.arange(20) * 0.1, 10)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def table(cfg):
    ds = load_dataset(cfg)
    return curve_targets(ds.metrics_pooled(), ds.iv_pooled(), cfg)


def test_straight_line_gives_exact_swing():
    # 10 decades per volt everywhere -> 100 mV/dec by both methods.
    y = -5.0 - 10.0 * GATE
    assert steepest_swing(GATE, y, 2001, 1.0) == pytest.approx(100.0, rel=1e-9)
    assert raw_grid_swing(GATE, y) == pytest.approx(100.0, rel=1e-9)


def test_upturn_after_minimum_is_ignored():
    # Falls 5 dec/V to 1.0 V, then rises 50 dec/V. The steep RISE must not count.
    y = np.where(GATE <= 1.0, -5 - 5 * GATE, -10 + 50 * (GATE - 1.0))
    assert falling_branch(y) == slice(0, 11)
    assert steepest_swing(GATE, y, 2001, 1.0) == pytest.approx(200.0, rel=1e-6)


def test_flat_curve_has_no_swing():
    y = -5.0 - 0.05 * GATE                     # drops < 1 decade in total
    assert np.isnan(steepest_swing(GATE, y, 2001, 1.0))


def test_boltzmann_limit_values():
    kb = load_config()["physics"]["k_B_eV_per_K"]
    assert boltzmann_swing(300, kb) == pytest.approx(59.5, abs=0.1)
    assert boltzmann_swing([4, 78], kb) == pytest.approx([0.794, 15.48], abs=0.01)


def test_unknown_power_convention_rejected():
    with pytest.raises(ValueError):
        dynamic_power(pd.Series([1.0]), pd.Series([1.0]), "DED_times_f")


def test_table_has_one_row_per_curve_and_no_nan(table):
    assert len(table) == 798 and not table.isna().any().any()
    assert table["split"].value_counts().to_dict() == {"train": 638, "test": 160}


def test_supplied_ratio_reproduced(table, cfg):
    m = load_dataset(cfg).metrics_pooled().sort_values(["T_K", "vac_pct"], ignore_index=True)
    assert np.allclose(table["log10_Ion_Ioff"], m["log10_Ion_Ioff"], atol=1e-9)


def test_min_ratio_never_below_supplied_ratio(table):
    assert (table["log10_Ion_Ioff_min"] >= table["log10_Ion_Ioff"] - 1e-12).all()


def test_pchip_swing_never_above_raw_grid_swing(table):
    # PCHIP slope inside an interval is >= the secant slope of its steepest step.
    assert (table["swing_mV_per_dec"] <= table["swing_raw_mV_per_dec"] + 1e-9).all()


def test_static_power_is_vds_times_ioff(table, cfg):
    m = load_dataset(cfg).metrics_pooled().sort_values(["T_K", "vac_pct"], ignore_index=True)
    v_ds = cfg["physics"]["V_DS_V"]
    assert np.allclose(table["log10_SPD"], np.log10(v_ds * m["Ioff_A"]), atol=1e-9)
