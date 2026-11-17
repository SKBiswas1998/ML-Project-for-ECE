"""Load and validate the NEGF dataset (CSVs + vacancy layout from NEGF_all.mat).

This is the only module that touches the raw files. Everything downstream gets
validated pandas tables or numpy arrays, so a changed or corrupted data file
fails here, loudly, instead of silently shifting a result.

Units follow the CSV column names: T_K in kelvin, vac_pct in percent of edge
atoms, VTG_minus_VBG_V in volt, I_A in ampere, log10_I = log10(I / 1 A).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
import scipy.io as sio

from wte2ml.config import repo_path

IV_COLUMNS = (
    "T_K", "vac_pct", "vac_pct_actual", "n_vac_atoms", "lambda_eV", "VTG_minus_VBG_V",
    "Ez_V_per_A", "I_A", "log10_I", "Cq_F_per_m", "Cq_F_per_m2", "CQ_F", "CT_F",
    "reliable", "seed",
)
METRIC_COLUMNS = (
    "T_K", "vac_pct", "vac_pct_actual", "n_vac_atoms", "Ion_A", "Ioff_A", "Ion_Ioff",
    "log10_Ion_Ioff", "CQ_avg_F", "C_eff_F", "tau_d_s", "DED_J", "SPD_W", "seed",
)


@dataclass(frozen=True)
class Dataset:
    """The four supplied tables. ``split`` column added: 'train' or 'test'."""

    iv_train: pd.DataFrame
    iv_test: pd.DataFrame
    metrics_train: pd.DataFrame
    metrics_test: pd.DataFrame

    def iv_pooled(self) -> pd.DataFrame:
        """All 798 curves' gate points (train + test), for the pooled protocols."""
        return pd.concat([self.iv_train, self.iv_test], ignore_index=True)

    def metrics_pooled(self) -> pd.DataFrame:
        """All 798 curve-level rows (train + test)."""
        return pd.concat([self.metrics_train, self.metrics_test], ignore_index=True)


@dataclass(frozen=True)
class VacancyLayout:
    """Vacancy configuration per level, from NEGF_all.mat.

    vac_atoms[k] holds the removed atoms of level vac_pct[k] as 1-based MATLAB
    atom indices (kept as stored; never used to index a Python array).
    """

    vac_pct: np.ndarray            # (21,) nominal levels, percent
    n_vac: np.ndarray              # (21,) number of removed atoms
    vac_atoms: tuple[np.ndarray, ...]
    n_atoms: int                   # atoms in the device (1620)

    def is_nested(self) -> bool:
        """True if every level keeps all atoms removed at the previous level."""
        sets = [set(a.tolist()) for a in self.vac_atoms]
        return all(a <= b for a, b in pairwise(sets))


def _read_table(path, columns: tuple[str, ...], split: str) -> pd.DataFrame:
    """Read one CSV, check the expected columns exist, tag rows with ``split``."""
    df = pd.read_csv(path)
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    return df.assign(split=split)


def _check_table(df: pd.DataFrame, name: str, n_expected: int, seed: int) -> None:
    """Row count, no NaN, all points reliable, single layout seed."""
    if len(df) != n_expected:
        raise ValueError(f"{name}: {len(df)} rows, expected {n_expected}")
    if df.isna().any().any():
        raise ValueError(f"{name}: contains NaN")
    if "reliable" in df and not (df["reliable"] == 1).all():
        raise ValueError(f"{name}: rows with reliable != 1")
    if not (df["seed"] == seed).all():
        raise ValueError(f"{name}: seed differs from {seed}")


def check_curves(iv: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Every curve has ``points_per_curve`` points on one shared gate grid."""
    key, gate = cfg["data"]["curve_key"], cfg["data"]["gate_column"]
    n_points = cfg["data"]["points_per_curve"]
    sizes = iv.groupby(key).size()
    if not (sizes == n_points).all():
        raise ValueError(f"curves with != {n_points} points: {sizes[sizes != n_points]}")
    grids = iv.sort_values(key + [gate]).groupby(key)[gate].apply(np.asarray)
    spread = np.ptp(np.stack(grids.to_numpy()), axis=0).max()
    if spread > cfg["data"]["gate_grid_tolerance_V"]:
        raise ValueError(f"gate grids differ between curves by {spread} V")


def check_split_consistency(ds: Dataset, cfg: dict[str, Any]) -> None:
    """IV and metrics tables describe the same curves; train and test are disjoint."""
    key = cfg["data"]["curve_key"]
    for split in ("train", "test"):
        iv_keys = set(map(tuple, getattr(ds, f"iv_{split}")[key].drop_duplicates().to_numpy()))
        m_keys = set(map(tuple, getattr(ds, f"metrics_{split}")[key].to_numpy()))
        if iv_keys != m_keys:
            raise ValueError(f"{split}: IV and metrics curve sets differ")
    overlap = set(map(tuple, ds.metrics_train[key].to_numpy())) & set(
        map(tuple, ds.metrics_test[key].to_numpy()))
    if overlap:
        raise ValueError(f"{len(overlap)} curves in both train and test")


def load_dataset(cfg: dict[str, Any]) -> Dataset:
    """Read the four CSVs and run every check. Returns validated tables."""
    d, folder = cfg["data"], repo_path(cfg, "data_dir")
    tables = {}
    for name in ("iv_train", "iv_test", "metrics_train", "metrics_test"):
        cols = IV_COLUMNS if name.startswith("iv") else METRIC_COLUMNS
        df = _read_table(folder / d[name], cols, split=name.split("_")[1])
        _check_table(df, name, d["expected_rows"][name], d["expected_seed"])
        tables[name] = df
    ds = Dataset(**tables)
    check_curves(ds.iv_pooled(), cfg)
    check_split_consistency(ds, cfg)
    return ds


def curve_matrix(iv: pd.DataFrame, cfg: dict[str, Any], value: str = "log10_I"
                 ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Reshape gate-point rows into one row per curve.

    Returns (keys, gate_V, Y): keys is a (n_curves, 2) table of T_K, vac_pct
    sorted by T then vacancy; gate_V the (20,) shared gate grid in volt; Y the
    (n_curves, 20) matrix of ``value``. Used by swing extraction and PCA.
    """
    key, gate = cfg["data"]["curve_key"], cfg["data"]["gate_column"]
    wide = iv.pivot_table(index=key, columns=gate, values=value).sort_index()
    return wide.index.to_frame(index=False), wide.columns.to_numpy(float), wide.to_numpy(float)


def load_layout(cfg: dict[str, Any]) -> VacancyLayout:
    """Read vacancy layout arrays from NEGF_all.mat (cell arrays -> tuple of int arrays)."""
    raw = sio.loadmat(repo_path(cfg, "data_dir") / cfg["data"]["negf_all"], squeeze_me=True)
    atoms = tuple(np.atleast_1d(a).astype(int) for a in raw["vac_atoms"])
    return VacancyLayout(
        vac_pct=np.asarray(raw["vac_list"], float),
        n_vac=np.asarray(raw["n_vac"], int),
        vac_atoms=atoms,
        n_atoms=int(raw["n_at"]),
    )
