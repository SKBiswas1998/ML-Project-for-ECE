"""Split tests: prove no curve leaks between sides, and every protocol has the intended ranges."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from wte2ml.config import load_config
from wte2ml.data import load_dataset
from wte2ml.splits import (
    all_protocols,
    blocked_cv_folds,
    curve_keys,
    vacancy_level,
)
from wte2ml.targets import curve_targets


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def ds(cfg):
    return load_dataset(cfg)


@pytest.fixture(scope="module")
def curves(ds, cfg):
    return curve_targets(ds.metrics_pooled(), ds.iv_pooled(), cfg)


@pytest.fixture(scope="module")
def protocols(curves, cfg):
    return {s.name: s for s in all_protocols(curves, cfg)}


def _pairs(keys, step=0.1):
    return set(zip(keys["T_K"].astype(int), vacancy_level(keys["vac_pct"], step)))


def _levels(keys):
    return sorted(set(vacancy_level(keys["vac_pct"], 0.1)))


@pytest.mark.parametrize("name", ["supplied", "temperature_forward", "vacancy_block",
                                  "vacancy_extrapolation"])
def test_train_and_test_curves_are_disjoint(protocols, name):
    s = protocols[name]
    assert not (_pairs(s.train) & _pairs(s.test))
    assert len(s.train) > 0 and len(s.test) > 0


@pytest.mark.parametrize("name", ["supplied", "temperature_forward", "vacancy_block",
                                  "vacancy_extrapolation"])
def test_iv_rows_follow_their_curve(protocols, ds, name):
    """Applying a split to the 15,960 IV rows keeps all 20 points of a curve together."""
    tr, te = protocols[name].apply(ds.iv_pooled())
    assert len(tr) == 20 * len(protocols[name].train) and len(te) == 20 * len(protocols[name].test)
    assert not (_pairs(curve_keys(tr)) & _pairs(curve_keys(te)))


def test_supplied_split_matches_files(protocols, ds):
    s = protocols["supplied"]
    assert (len(s.train), len(s.test)) == (638, 160)
    assert _pairs(s.test) == _pairs(curve_keys(ds.metrics_test))


def test_temperature_forward_ranges(protocols):
    s = protocols["temperature_forward"]
    assert s.train["T_K"].max() == 60 and s.test["T_K"].min() == 62 and s.test["T_K"].max() == 78
    assert (len(s.train), len(s.test)) == (29 * 21, 9 * 21)


def test_vacancy_block_ranges(protocols):
    s = protocols["vacancy_block"]
    assert _levels(s.test) == [8, 9, 10, 11, 12]
    assert not set(_levels(s.train)) & {8, 9, 10, 11, 12}
    assert s.test["T_K"].nunique() == 38 and (len(s.train), len(s.test)) == (16 * 38, 5 * 38)


def test_vacancy_extrapolation_ranges(protocols):
    s = protocols["vacancy_extrapolation"]
    assert _levels(s.train) == list(range(16)) and _levels(s.test) == list(range(16, 21))


def test_float_vacancy_values_map_to_correct_level():
    assert vacancy_level([0.30000000000000004, 0.7000000000000001, 2.0], 0.1).tolist() == [3, 7, 20]


@pytest.mark.parametrize("name", ["supplied", "temperature_forward", "vacancy_block",
                                  "vacancy_extrapolation"])
def test_cv_uses_only_training_pool_and_respects_embargo(protocols, cfg, name):
    """THE leakage test: CV folds never see a final-test curve, and every training
    level is more than `cv_embargo_levels` grid steps away from the validation block."""
    pool = protocols[name].train
    test_pairs = _pairs(protocols[name].test)
    emb = cfg["splits"]["cv_embargo_levels"]
    folds = blocked_cv_folds(pool, cfg)
    assert len(folds) == cfg["splits"]["cv_n_blocks"]
    for f in folds:
        assert not (_pairs(f.train) | _pairs(f.test)) & test_pairs
        assert not _pairs(f.train) & _pairs(f.test)
        val = np.array(_levels(f.test))
        gaps = [np.min(np.abs(val - lv)) for lv in _levels(f.train)]
        assert min(gaps) > emb


def test_cv_validation_blocks_partition_the_pool(protocols, cfg):
    pool = protocols["supplied"].train
    folds = blocked_cv_folds(pool, cfg)
    seen = [p for f in folds for p in _pairs(f.test)]
    assert len(seen) == len(set(seen)) == len(pool)


def test_cv_blocks_are_consecutive_levels(protocols, cfg):
    pool_levels = _levels(protocols["supplied"].train)
    for f in blocked_cv_folds(protocols["supplied"].train, cfg):
        val = _levels(f.test)
        i = pool_levels.index(val[0])
        assert val == pool_levels[i:i + len(val)]


def test_zero_embargo_drops_nothing(protocols, cfg):
    c = copy.deepcopy(cfg)
    c["splits"]["cv_embargo_levels"] = 0
    pool = protocols["supplied"].train
    for f in blocked_cv_folds(pool, c):
        assert len(f.train) + len(f.test) == len(pool)
