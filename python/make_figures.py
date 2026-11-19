"""Produce figures/*.png. Run from the repo root:  .\\.venv\\Scripts\\python python\\make_figures.py

Diagnostic figures (swing check, PCA) and the Stage 7 report figures. Result figures read
results/*.csv only; plotting primitives live in wte2ml.plots.
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
from wte2ml.evaluate import comparison_table
from wte2ml.plots import error_profile, heatmap, iv_with_bands, predicted_vs_true, robustness_lines
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


PROTOCOLS = ["supplied", "temperature_forward", "vacancy_block", "vacancy_extrapolation"]
MODELS = ["naive_vac_mean", "naive_nn", "ridge", "poly_ridge", "boosting", "gp", "mlp"]


def _save(fig, out_dir: Path, name: str) -> Path:
    fig.tight_layout()
    path = out_dir / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def load_predictions(res: Path) -> pd.DataFrame:
    """All test predictions: classical models as stored; MLP averaged over its seeds."""
    classical = pd.read_csv(res / "predictions_classical.csv.gz")
    mlp = pd.read_csv(res / "predictions_mlp.csv.gz")
    keys = ["protocol", "feature", "target", "T_K", "vac_pct", "gate_V"]
    mlp = (mlp.assign(gate_V=mlp.gate_V.fillna(-1)).groupby(keys, as_index=False)
           .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
           .assign(model="mlp", y_std=np.nan))
    mlp["gate_V"] = mlp.gate_V.replace(-1, np.nan)
    return pd.concat([classical, mlp], ignore_index=True)


def figure_ratio_heatmap(res: Path, out_dir: Path) -> Path:
    """log10 Ion/Ioff (supplied end-point definition) over the full T x vacancy grid."""
    curves = pd.read_csv(res / "curve_targets.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = heatmap(ax, curves, "log10_Ion_Ioff", "log10 Ion/Ioff (all 798 curves)")
    fig.colorbar(im, ax=ax, label="log10 Ion/Ioff (decades)")
    return _save(fig, out_dir, "ion_ioff_heatmap.png")


def figure_pred_vs_true(pred: pd.DataFrame, target: str, out_dir: Path) -> Path:
    """Grid: rows = protocols, columns = models; predicted vs true for one target."""
    sub = pred[pred.target == target]
    lims = (float(sub.y_true.min()) - 0.5, float(sub.y_true.max()) + 0.5)
    fig, axes = plt.subplots(len(PROTOCOLS), len(MODELS), figsize=(2.1 * len(MODELS), 2.1 * len(PROTOCOLS)))
    for r, protocol in enumerate(PROTOCOLS):
        for c, model in enumerate(MODELS):
            cell = sub[(sub.protocol == protocol) & (sub.model == model)]
            predicted_vs_true(axes[r, c], cell, f"{protocol}\n{model}", lims)
    fig.supxlabel(f"true {target}"); fig.supylabel(f"predicted {target}")
    return _save(fig, out_dir, f"pred_vs_true_{target}.png")


def figure_error_maps(pred: pd.DataFrame, target: str, models: list[str], out_dir: Path) -> Path:
    """|error| of each test curve over the T x vacancy grid; rows = protocols, cols = models."""
    sub = pred[pred.target == target].assign(abs_err=lambda d: (d.y_pred - d.y_true).abs())
    vmax = float(sub.abs_err.quantile(0.98))
    fig, axes = plt.subplots(len(PROTOCOLS), len(models), figsize=(3.4 * len(models), 2.9 * len(PROTOCOLS)),
                             layout="constrained")
    for r, protocol in enumerate(PROTOCOLS):
        for c, model in enumerate(models):
            cell = sub[(sub.protocol == protocol) & (sub.model == model)]
            im = heatmap(axes[r, c], cell, "abs_err", f"{protocol}: {model}", "magma", 0, vmax)
            axes[r, c].title.set_fontsize(8)
    fig.colorbar(im, ax=axes, label=f"|error| in {target} (decades)", shrink=0.6)
    path = out_dir / f"error_map_{target}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_error_profiles(pred: pd.DataFrame, target: str, out_dir: Path) -> Path:
    """MAE vs vacancy (top) and vs temperature (bottom), one column per protocol."""
    sub = pred[pred.target == target]
    fig, axes = plt.subplots(2, len(PROTOCOLS), figsize=(4 * len(PROTOCOLS), 6.5))
    for c, protocol in enumerate(PROTOCOLS):
        cell = sub[sub.protocol == protocol]
        error_profile(axes[0, c], cell, "vac_pct", MODELS, f"{protocol}: error vs vacancy")
        error_profile(axes[1, c], cell, "T_K", MODELS, f"{protocol}: error vs temperature")
    axes[0, 0].legend(fontsize=6)
    return _save(fig, out_dir, f"error_profiles_{target}.png")


def figure_iv_samples(pred: pd.DataFrame, cfg: dict[str, Any], out_dir: Path) -> Path:
    """Sample I-V curves (rows = protocols) with GP mean +/- 2 std, boosting and MLP."""
    iv = pred[pred.target == cfg["experiments"]["point_target"]]
    fig, axes = plt.subplots(len(PROTOCOLS), 3, figsize=(3.6 * 3, 2.8 * len(PROTOCOLS)))
    for r, protocol in enumerate(PROTOCOLS):
        cell = iv[iv.protocol == protocol]
        T = cfg["figures"]["iv_sample_T_K"][protocol]
        levels = np.sort(cell[cell.T_K == T].vac_pct.unique())
        for c, vac in enumerate(levels[[0, len(levels) // 2, -1]]):   # lowest, middle, highest test level
            curve = cell[(cell.T_K == T) & np.isclose(cell.vac_pct, vac)]
            iv_with_bands(axes[r, c], curve, ["gp", "boosting", "mlp"], f"{protocol}: {T} K, {vac:.1f} %")
    axes[0, 0].legend(fontsize=6)
    return _save(fig, out_dir, "iv_samples_gp_bands.png")


def figure_robustness(res: Path, out_dir: Path) -> Path:
    """Test MAE vs Gaussian input-noise level, one panel per protocol."""
    s = pd.read_csv(res / "robustness_summary.csv")
    protocols = list(dict.fromkeys(s.protocol))
    fig, axes = plt.subplots(1, len(protocols), figsize=(5.5 * len(protocols), 4), squeeze=False)
    for ax, protocol in zip(axes[0], protocols):
        robustness_lines(ax, s[s.protocol == protocol], f"{protocol}: {s.target.iloc[0]}")
    axes[0, 0].legend(fontsize=7)
    return _save(fig, out_dir, "robustness_noise.png")


def figure_comparison_table(table: pd.DataFrame, target: str, out_dir: Path) -> Path:
    """Model x protocol MAE table for one target as an image (best per protocol in bold)."""
    wide = table[table.target == target].pivot(index="model", columns="protocol", values="MAE")
    wide = wide.reindex(index=MODELS, columns=PROTOCOLS)
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(MODELS) + 1))
    ax.axis("off")
    t = ax.table(cellText=wide.map(lambda v: f"{v:.3f}").to_numpy(), rowLabels=wide.index,
                 colLabels=wide.columns, loc="center")
    for c, protocol in enumerate(PROTOCOLS):
        t[(int(np.argmin(wide[protocol].to_numpy())) + 1, c)].set_text_props(fontweight="bold")
    ax.set_title(f"MAE (decades), {target}; bold = best per protocol", fontsize=9)
    return _save(fig, out_dir, f"model_comparison_{target}.png")


def report_figures(cfg: dict[str, Any], out_dir: Path) -> list[Path]:
    """Every Stage 7 figure plus results/model_comparison.csv."""
    res = repo_path(cfg, "results_dir")
    feature = cfg["inputs"]["vacancy_features"][0]          # never mix vacancy encodings in a figure
    table = comparison_table(pd.read_csv(res / "metrics_classical.csv"), pd.read_csv(res / "metrics_mlp.csv"))
    table = table[table.feature == feature]
    table.to_csv(res / "model_comparison.csv", index=False)
    pred = load_predictions(res)
    pred = pred[pred.feature == feature]
    head = cfg["figures"]["headline_target"]
    paths = [figure_ratio_heatmap(res, out_dir), figure_iv_samples(pred, cfg, out_dir),
             figure_robustness(res, out_dir)]
    for target in (head, cfg["experiments"]["point_target"]):
        paths += [figure_pred_vs_true(pred, target, out_dir), figure_error_profiles(pred, target, out_dir),
                  figure_comparison_table(table, target, out_dir),
                  figure_error_maps(pred, target, cfg["figures"]["error_map_models"], out_dir)]
    return paths


def main() -> None:
    plt.switch_backend("Agg")   # file output only, no window
    cfg = load_config()
    out_dir = repo_path(cfg, "figures_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(figure_swing_check(cfg, out_dir))
    print(figure_pca_clusters(cfg, out_dir))
    for path in report_figures(cfg, out_dir):
        print(path)


if __name__ == "__main__":
    main()
