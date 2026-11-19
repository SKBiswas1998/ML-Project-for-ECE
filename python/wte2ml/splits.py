"""Evaluation protocols and blocked cross-validation, all defined on CURVES.

A split is a pair of curve-key tables (T_K, vac_pct). Rows of any table (IV rows
or curve targets) are assigned by key, so the 20 gate points of one sweep can
never land on both sides. Random row splits are deliberately not offered.

Vacancy levels are compared through an integer level index round(vac_pct / step)
(step from config), which avoids float comparisons such as 0.30000000000000004.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

KEY = ["T_K", "vac_pct"]


def vacancy_level(vac_pct: pd.Series | np.ndarray | list, step: float) -> np.ndarray:
    """Integer vacancy-level index: with step 0.1, 0.0 % -> 0, 0.1 % -> 1, ..., 2.0 % -> 20."""
    return np.rint(np.asarray(vac_pct, float) / step).astype(int)


def curve_keys(table: pd.DataFrame) -> pd.DataFrame:
    """Unique (T_K, vac_pct) of a table, sorted. Works for IV rows and curve rows."""
    return table[KEY].drop_duplicates().sort_values(KEY, ignore_index=True)


def _key_set(table: pd.DataFrame, step: float) -> set[tuple[int, int]]:
    """Set of (T_K, level index) pairs present in ``table``."""
    return set(zip(table["T_K"].astype(int), vacancy_level(table["vac_pct"], step)))


@dataclass(frozen=True)
class Split:
    """Named train/test partition of curve keys.

    ``step`` is the vacancy grid spacing; ``cv_axis`` says along which input the
    hyperparameter CV inside ``train`` is blocked ("vacancy" or "temperature"), chosen
    to resemble the final test of the protocol.
    """

    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    step: float
    cv_axis: str = "vacancy"

    def fingerprint(self) -> str:
        """Short hash of the exact train and test curves and CV axis (used in cache keys, so
        a cached result is reused only if it was computed on this identical split)."""
        parts = [self.name, self.cv_axis] + [sorted(_key_set(k, self.step)) for k in (self.train, self.test)]
        return hashlib.sha1(repr(parts).encode()).hexdigest()[:8]

    def apply(self, table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Rows of ``table`` whose curve is in train / in test (original row order kept)."""
        return self._rows(table, self.train), self._rows(table, self.test)

    def _rows(self, table: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
        wanted = _key_set(keys, self.step)
        rows = zip(table["T_K"].astype(int), vacancy_level(table["vac_pct"], self.step))
        return table.loc[[k in wanted for k in rows]]


def _make(name: str, keys: pd.DataFrame, test: np.ndarray, drop: np.ndarray,
          step: float, cv_axis: str = "vacancy") -> Split:
    """Build a Split: ``test`` curves form the test set; ``drop`` curves are excluded
    from training only (the embargo)."""
    return Split(name, keys[~test & ~drop].reset_index(drop=True),
                 keys[test].reset_index(drop=True), step, cv_axis)


def _near(levels: np.ndarray, lo: int, hi: int, margin: int) -> np.ndarray:
    """Levels within ``margin`` grid steps of the closed range [lo, hi] (range included)."""
    return (levels >= lo - margin) & (levels <= hi + margin)


def supplied_split(curves: pd.DataFrame, cfg: dict[str, Any]) -> Split:
    """Protocol 1: the supplied train/test labels (interpolation only)."""
    step, keys = cfg["splits"]["vac_step_pct"], curve_keys(curves)
    test_set = _key_set(curves[curves["split"] == "test"], step)
    test = np.array([k in test_set for k in _key_set_list(keys, step)])
    return _make("supplied", keys, test, np.zeros(len(keys), bool), step)


def _key_set_list(keys: pd.DataFrame, step: float) -> list[tuple[int, int]]:
    """(T_K, level) pairs of ``keys`` in row order."""
    return list(zip(keys["T_K"].astype(int), vacancy_level(keys["vac_pct"], step)))


def temperature_forward(curves: pd.DataFrame, cfg: dict[str, Any]) -> Split:
    """Protocol 2 (forward-chaining in T): train T <= temp_train_max_K, test above it.
    Its hyperparameters are tuned by temperature-blocked CV (``cv_axis="temperature"``)."""
    keys = curve_keys(curves)
    test = keys["T_K"].to_numpy() > cfg["splits"]["temp_train_max_K"]
    return _make("temperature_forward", keys, test, np.zeros(len(keys), bool),
                 cfg["splits"]["vac_step_pct"], cv_axis="temperature")


def vacancy_block(curves: pd.DataFrame, cfg: dict[str, Any]) -> Split:
    """Protocol 3a: hold out a block of consecutive vacancy levels at all temperatures."""
    s = cfg["splits"]
    keys = curve_keys(curves)
    lv = vacancy_level(keys["vac_pct"], s["vac_step_pct"])
    lo, hi = vacancy_level(s["vac_block_holdout_pct"], s["vac_step_pct"])
    test = _near(lv, lo, hi, 0)
    drop = _near(lv, lo, hi, s["holdout_embargo_levels"]) & ~test
    return _make("vacancy_block", keys, test, drop, s["vac_step_pct"])


def vacancy_extrapolation(curves: pd.DataFrame, cfg: dict[str, Any]) -> Split:
    """Protocol 3b: train on levels <= vac_train_max_pct, test on every higher level."""
    s = cfg["splits"]
    keys = curve_keys(curves)
    lv = vacancy_level(keys["vac_pct"], s["vac_step_pct"])
    cut = int(vacancy_level([s["vac_train_max_pct"]], s["vac_step_pct"])[0])
    test = lv > cut
    drop = ~test & (lv > cut - s["holdout_embargo_levels"])
    return _make("vacancy_extrapolation", keys, test, drop, s["vac_step_pct"])


def all_protocols(curves: pd.DataFrame, cfg: dict[str, Any]) -> list[Split]:
    """The four final evaluation splits, in reporting order. ``curves`` = pooled 798."""
    return [supplied_split(curves, cfg), temperature_forward(curves, cfg),
            vacancy_block(curves, cfg), vacancy_extrapolation(curves, cfg)]


def _cv_levels(keys: pd.DataFrame, cfg: dict[str, Any], axis: str) -> tuple[np.ndarray, int, int]:
    """Integer level of each curve along the CV axis, plus that axis' (n_blocks, embargo)."""
    s = cfg["splits"]
    if axis == "vacancy":
        return vacancy_level(keys["vac_pct"], s["vac_step_pct"]), s["cv_n_blocks"], s["cv_embargo_levels"]
    if axis == "temperature":
        levels = np.rint(keys["T_K"].to_numpy(float) / s["temp_step_K"]).astype(int)
        return levels, s["cv_temp_n_blocks"], s["cv_temp_embargo_levels"]
    raise ValueError(f"unknown cv axis {axis!r}")


def blocked_cv_folds(train_keys: pd.DataFrame, cfg: dict[str, Any], axis: str = "vacancy"
                     ) -> list[Split]:
    """Hyperparameter CV inside one training pool: blocks of consecutive levels of ``axis``.

    The pool's distinct levels (vacancy levels, or temperatures) are cut into
    contiguous blocks. Fold i validates on block i and trains on the other levels minus
    those within the embargo (grid steps) of block i. For vacancy, nested layouts mean
    this REDUCES leakage between levels; it cannot remove it.
    """
    keys = curve_keys(train_keys)
    lv, n_blocks, embargo = _cv_levels(keys, cfg, axis)
    folds = []
    for i, block in enumerate(np.array_split(np.unique(lv), n_blocks)):
        val = np.isin(lv, block)
        drop = _near(lv, block.min(), block.max(), embargo) & ~val
        folds.append(_make(f"cv_fold{i}", keys, val, drop, cfg["splits"]["vac_step_pct"], axis))
    return folds

def fold_indices(table: pd.DataFrame, folds: list[Split]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Positional (train_idx, val_idx) of ``table`` rows for each fold (GridSearchCV ``cv``).

    ``table`` must have a default RangeIndex (positions == labels); rows of curves that a
    fold drops (embargo) appear in neither array.
    """
    if not table.index.equals(pd.RangeIndex(len(table))):
        raise ValueError("fold_indices needs a table with a fresh RangeIndex")
    out = []
    for f in folds:
        tr, va = f.apply(table)
        out.append((tr.index.to_numpy(), va.index.to_numpy()))
    return out
