"""Produce figures/*.png. Run from the repo root:  .\\.venv\\Scripts\\python python\\make_figures.py

Stage 2 adds only the swing-extraction check; later stages add result figures
that read results/*.csv.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from wte2ml.config import load_config, repo_path
from wte2ml.data import curve_matrix, load_dataset
from wte2ml.targets import falling_branch, swing_table


def _plot_one(ax, gate_V: np.ndarray, y: np.ndarray, title: str, swing: float, n_fine: int,
              tangent_half_width_V: float) -> None:
    """One panel: raw points, falling branch, PCHIP fit, steepest-slope tangent."""
    br = falling_branch(y)
    ax.plot(gate_V, y, "o", ms=3, color="0.6", label="NEGF points")
    ax.plot(gate_V[br], y[br], "o", ms=3, color="C0", label="falling branch")
    if br.stop >= 2:
        fine = np.linspace(gate_V[0], gate_V[br.stop - 1], n_fine)
        f = PchipInterpolator(gate_V[br], y[br])
        ax.plot(fine, f(fine), "-", lw=1, color="C0")
        if np.isfinite(swing):
            k = np.argmin(f.derivative()(fine))
            near = np.abs(fine - fine[k]) <= tangent_half_width_V
            tang = f(fine[k]) + f.derivative()(fine[k]) * (fine[near] - fine[k])
            ax.plot(fine[near], tang, "--", lw=1.0, color="C3", label="steepest tangent")
    ax.set_title(title, fontsize=7)
    ax.tick_params(labelsize=7)


def figure_swing_check(cfg: dict[str, Any], out_dir: Path) -> Path:
    """Grid of sample curves (T x vacancy from config) showing how swing is extracted."""
    iv = load_dataset(cfg).iv_pooled()
    keys, gate_V, Y = curve_matrix(iv, cfg)
    table = swing_table(iv, cfg).set_index(["T_K", "vac_pct"])
    sw = cfg["targets"]["swing"]
    Ts, vacs = sw["check_T_K"], sw["check_vac_pct"]
    fig, axes = plt.subplots(len(Ts), len(vacs), figsize=(3.2 * len(vacs), 2.6 * len(Ts)),
                             sharex=True, squeeze=False)
    lookup = {(t, v): i for i, (t, v) in enumerate(keys.itertuples(index=False))}
    for (r, T), (c, vac) in ((a, b) for a in enumerate(Ts) for b in enumerate(vacs)):
        s = table.loc[(T, vac)]
        title = (f"{T} K, {vac:.1f} %: S={s.swing_mV_per_dec:.1f} (raw {s.swing_raw_mV_per_dec:.0f},"
                 f" kTln10 {s.swing_boltzmann_mV_per_dec:.1f})")
        _plot_one(axes[r, c], gate_V, Y[lookup[(T, vac)]], title, s.swing_mV_per_dec,
                  sw["pchip_points"], sw["tangent_half_width_V"])
    for ax in axes[-1]:
        ax.set_xlabel("V_TG - V_BG (V)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("log10 I (A)", fontsize=8)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Swing extraction check (S in mV/dec; dashed = steepest PCHIP tangent)", fontsize=9)
    fig.tight_layout()
    path = out_dir / "swing_check.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_variance(ax, ev: pd.DataFrame) -> None:
    """Bars: share of variance per component; line: cumulative share."""
    ax.bar(ev.component, ev.explained_variance_ratio, color="C0", label="per component")
    ax.plot(ev.component, ev.cumulative, "o-", color="C3", ms=3, label="cumulative")
    ax.set(xlabel="principal component", ylabel="share of variance", ylim=(0, 1.02),
           title="(a) PCA of log10 I-V curves")
    ax.legend(fontsize=7)


def _plot_loadings(ax, comps: pd.DataFrame) -> None:
    """What PC1 and PC2 add to the mean curve, per gate voltage."""
    for c, style in ((1, "-"), (2, "--")):
        sub = comps[comps.component == c]
        ax.plot(sub.gate_V, sub.loading, style, label=f"PC{c} loading")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set(xlabel="V_TG - V_BG (V)", ylabel="loading (decades per unit score)",
           title="(b) component shapes")
    ax.legend(fontsize=7)


def _plot_clusters(ax, assign: pd.DataFrame, column: str, class_names: list[str]) -> None:
    """PC1-PC2 scores coloured by cluster, marker by Ion/Ioff class."""
    markers = dict(zip(class_names, ("x", "o", "^")))
    for cls, m in markers.items():
        sub = assign[assign.ion_ioff_class == cls]
        ax.scatter(sub.PC1, sub.PC2, c=sub[column], cmap="viridis", vmin=0,
                   vmax=assign[column].max(), marker=m, s=14, lw=0.8, label=cls)
    ax.set(xlabel="PC1 score", ylabel="PC2 score", title=f"(c) {column} (colour) vs Ion/Ioff class (marker)")
    leg = ax.legend(fontsize=7, title="log10 Ion/Ioff class", title_fontsize=7)
    for handle in leg.legend_handles:    # marker shape = class; colour belongs to the clusters
        handle.set_color("0.35")


def figure_pca_clusters(cfg: dict[str, Any], out_dir: Path, column: str = "kmeans_k3") -> Path:
    """Three panels from results/pca_*.csv and cluster_assignments.csv (Stage 6)."""
    res = repo_path(cfg, "results_dir")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    _plot_variance(axes[0], pd.read_csv(res / "pca_explained_variance.csv"))
    _plot_loadings(axes[1], pd.read_csv(res / "pca_components.csv"))
    _plot_clusters(axes[2], pd.read_csv(res / "cluster_assignments.csv"), column,
                   cfg["unsupervised"]["class_names"])
    fig.tight_layout()
    path = out_dir / "pca_clusters.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    plt.switch_backend("Agg")   # file output only, no window
    cfg = load_config()
    out_dir = repo_path(cfg, "figures_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(figure_swing_check(cfg, out_dir))
    print(figure_pca_clusters(cfg, out_dir))


if __name__ == "__main__":
    main()
