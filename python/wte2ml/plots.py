"""Plotting functions for the Stage 7 report figures.

Each function draws on a given matplotlib Axes from tables that came from
results/*.csv (passed in as DataFrames), so the figures can never compute their own
numbers: make_figures.py only reads the CSVs, calls these, and saves the PNGs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["T_K", "vac_pct"]


def heatmap(ax, table: pd.DataFrame, value: str, title: str, cmap: str = "viridis",
            vmin: float | None = None, vmax: float | None = None):
    """Colour map of ``value`` over the T x vacancy grid (T on y, vacancy on x).
    Curves absent from ``table`` (e.g. not in a test set) are left blank."""
    grid = table.assign(vac=table.vac_pct.round(1)).pivot_table(index="T_K", columns="vac", values=value)
    im = ax.imshow(grid.to_numpy(), origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[grid.columns.min() - 0.05, grid.columns.max() + 0.05,
                           grid.index.min() - 1, grid.index.max() + 1])
    ax.set(xlabel="edge vacancy (%)", ylabel="T (K)", title=title)
    return im


def predicted_vs_true(ax, pred: pd.DataFrame, title: str, lims: tuple[float, float]) -> None:
    """Scatter of predicted vs true with the y = x line; MAE in the corner."""
    ax.scatter(pred.y_true, pred.y_pred, s=5, alpha=0.6)
    ax.plot(lims, lims, color="0.4", lw=0.8)
    mae = float(np.mean(np.abs(pred.y_pred - pred.y_true)))
    ax.text(0.04, 0.92, f"MAE {mae:.3f}", transform=ax.transAxes, fontsize=7)
    ax.set(xlim=lims, ylim=lims, title=title)
    ax.title.set_fontsize(8)
    ax.tick_params(labelsize=6)


def error_profile(ax, pred: pd.DataFrame, by: str, models: list[str], title: str) -> None:
    """Mean absolute error of each model grouped by ``by`` (T_K or vac_pct), one line per model."""
    for model in models:
        sub = pred[pred.model == model]
        g = sub.assign(err=(sub.y_pred - sub.y_true).abs(), key=sub[by].round(1)).groupby("key").err.mean()
        ax.plot(g.index, g.to_numpy(), "o-", ms=3, lw=1, label=model)
    ax.set(xlabel={"T_K": "T (K)", "vac_pct": "edge vacancy (%)"}[by], ylabel="MAE (decades)", title=title)
    ax.title.set_fontsize(8)


def iv_with_bands(ax, curve: pd.DataFrame, models: list[str], title: str) -> None:
    """One I-V curve: NEGF truth, the GP mean with a +/- 2 std band, other models' predictions."""
    truth = curve[curve.model == models[0]].sort_values("gate_V")
    ax.plot(truth.gate_V, truth.y_true, "ko", ms=3, label="NEGF")
    for model in models:
        sub = curve[curve.model == model].sort_values("gate_V")
        line, = ax.plot(sub.gate_V, sub.y_pred, "-", lw=1.2, label=model)
        if model == "gp":
            ax.fill_between(sub.gate_V, sub.y_pred - 2 * sub.y_std, sub.y_pred + 2 * sub.y_std,
                            color=line.get_color(), alpha=0.25, label="GP +/- 2 std")
    ax.set(xlabel="V_TG - V_BG (V)", ylabel="log10 I (A)", title=title)
    ax.title.set_fontsize(8)


def robustness_lines(ax, summary: pd.DataFrame, title: str) -> None:
    """Test MAE vs input-noise level (fraction of one grid step), mean +/- std over repeats."""
    for model, sub in summary.groupby("model"):
        sub = sub.sort_values("noise_fraction")
        ax.errorbar(sub.noise_fraction, sub.MAE_mean, yerr=sub.MAE_std.fillna(0), marker="o",
                    ms=3, lw=1, capsize=2, label=model)
    ax.set(xlabel="input noise (std, fraction of one grid step)", ylabel="test MAE (decades)", title=title)
    ax.title.set_fontsize(8)


def coefficient_path_lines(ax, alphas: np.ndarray, path: np.ndarray, names: list[str]) -> None:
    """Ridge coefficients (standardised inputs) vs the penalty alpha, one line per feature."""
    for j, name in enumerate(names):
        ax.plot(alphas, path[:, j], lw=1, label=name)
    ax.set_xscale("log")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set(xlabel="penalty alpha", ylabel="coefficient (standardised input)", title="Ridge coefficient path")
    ax.legend(fontsize=6, ncol=2)


def coverage_bars(ax, cover: pd.DataFrame, target: str) -> None:
    """Empirical coverage of the GP's 68 % and 95 % intervals per protocol, against nominal lines."""
    sub = cover[cover.target == target].reset_index(drop=True)
    x = np.arange(len(sub))
    ax.bar(x - 0.2, sub.coverage_68, 0.4, label="inside +/- 1 std")
    ax.bar(x + 0.2, sub.coverage_95, 0.4, label="inside +/- 1.96 std")
    for level, style in ((0.6827, "--"), (0.95, ":")):
        ax.axhline(level, color="0.3", ls=style, lw=1)
    ax.set_xticks(x, sub.protocol, rotation=15, fontsize=7)
    ax.set(ylim=(0, 1.05), ylabel="share of test points", title=f"GP interval coverage: {target}")
    ax.legend(fontsize=7)
