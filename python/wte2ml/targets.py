"""Learning targets derived from the validated tables.

Curve-level targets (one row per (T_K, vac_pct)) are all in log10, because
currents and powers span about 12 decades. Per-gate-point target log10_I is
used directly from the IV table.

Physics conventions (config ``physics``): SPD = V_DS * Ioff, DED = C_eff * V_DS^2,
tau_d = C_eff * V_DS / Ion, and dynamic power P_dyn = DED / tau_d.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from wte2ml.data import curve_matrix

CURVE_TARGETS = (
    "log10_Ion_Ioff", "log10_Ion_Ioff_min", "log10_SPD", "log10_DED",
    "log10_tau_d", "log10_P_dyn", "swing_mV_per_dec",
)


def falling_branch(log10_current: np.ndarray) -> slice:
    """Indices of the first descent: from the on state (index 0) to the curve minimum.

    The upturn after the minimum (present in 761 of 798 curves) is excluded,
    because it is a different mechanism from the switching we want to measure.
    """
    return slice(0, int(np.argmin(log10_current)) + 1)


def steepest_swing(gate_V: np.ndarray, log10_current: np.ndarray, n_fine: int,
                   min_drop: float) -> float:
    """Subthreshold swing of one sweep, in mV/decade.

    gate_V: (20,) increasing gate voltages in volt; log10_current: (20,) log10(I/A).
    PCHIP-interpolates log10 I(V) on the falling branch (shape-preserving: no
    overshoot between points) and returns min over V of |dV / dlog10 I| * 1000.
    Returns NaN when the branch falls less than ``min_drop`` decades (no switching).
    """
    br = falling_branch(log10_current)
    v, y = gate_V[br], log10_current[br]
    if v.size < 2 or y[0] - y[-1] < min_drop:
        return float("nan")
    fine = np.linspace(v[0], v[-1], n_fine)
    slope = PchipInterpolator(v, y).derivative()(fine)     # decades per volt
    return 1e3 / float(np.max(-slope))


def raw_grid_swing(gate_V: np.ndarray, log10_current: np.ndarray) -> float:
    """Swing from the steepest single grid step (mV/dec): an upper bound on the true value."""
    br = falling_branch(log10_current)
    step = -np.diff(log10_current[br]) / np.diff(gate_V[br])
    return 1e3 / float(step.max()) if step.size and step.max() > 0 else float("nan")


def boltzmann_swing(T_K: np.ndarray | float, k_B_eV_per_K: float) -> np.ndarray:
    """Thermionic limit kT ln(10)/q in mV/dec (k_B in eV/K, so kT/q is in volt)."""
    return 1e3 * k_B_eV_per_K * np.asarray(T_K, float) * np.log(10)


def swing_table(iv: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Per-curve swing (PCHIP and raw-grid) with the Boltzmann limit at that T."""
    keys, gate_V, Y = curve_matrix(iv, cfg)
    sw = cfg["targets"]["swing"]
    pchip = [steepest_swing(gate_V, y, sw["pchip_points"], sw["min_drop_decades"]) for y in Y]
    raw = [raw_grid_swing(gate_V, y) for y in Y]
    return keys.assign(
        swing_mV_per_dec=pchip,
        swing_raw_mV_per_dec=raw,
        swing_boltzmann_mV_per_dec=boltzmann_swing(keys["T_K"], cfg["physics"]["k_B_eV_per_K"]),
        idx_min=np.argmin(Y, axis=1),
    )


def curve_targets(metrics: pd.DataFrame, iv: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """One row per curve: inputs, split label and every log10 / swing target.

    metrics, iv: validated tables covering the same curves (e.g. pooled).
    log10_Ion_Ioff uses the supplied Ioff (current at the last gate point);
    log10_Ion_Ioff_min uses the minimum current of the sweep.
    """
    key = cfg["data"]["curve_key"]
    i_min = iv.groupby(key)["I_A"].min().rename("Ioff_min_A")
    m = metrics.merge(i_min, left_on=key, right_index=True, validate="one_to_one")
    m = m.merge(swing_table(iv, cfg), on=key, validate="one_to_one")
    p_dyn = dynamic_power(m["DED_J"], m["tau_d_s"], cfg["physics"]["dynamic_power"])
    out = m[[*key, "vac_pct_actual", "n_vac_atoms", "split"]].assign(
        log10_Ion_Ioff=np.log10(m["Ion_A"] / m["Ioff_A"]),
        log10_Ion_Ioff_min=np.log10(m["Ion_A"] / m["Ioff_min_A"]),
        log10_SPD=np.log10(m["SPD_W"]),
        log10_DED=np.log10(m["DED_J"]),
        log10_tau_d=np.log10(m["tau_d_s"]),
        log10_P_dyn=np.log10(p_dyn),
        swing_mV_per_dec=m["swing_mV_per_dec"],
        swing_raw_mV_per_dec=m["swing_raw_mV_per_dec"],
        swing_boltzmann_mV_per_dec=m["swing_boltzmann_mV_per_dec"],
    )
    return out.sort_values(key, ignore_index=True)


def dynamic_power(ded_J: pd.Series, tau_s: pd.Series, convention: str) -> pd.Series:
    """Dynamic power in watt. The single convention lives in config ``physics.dynamic_power``."""
    if convention != "DED_over_tau":
        raise ValueError(f"unknown dynamic-power convention {convention!r}")
    return ded_J / tau_s
