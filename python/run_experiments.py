"""Run pipeline steps and write results/*.csv. From the repo root:

    .\\.venv\\Scripts\\python python\\run_experiments.py targets

Each step writes its own CSVs and times itself. Later stages add model steps.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd

from wte2ml.config import load_config, repo_path
from wte2ml.data import load_dataset
from wte2ml.splits import Split, all_protocols, blocked_cv_folds
from wte2ml.targets import curve_targets


def step_targets(cfg: dict[str, Any], out_dir: Path) -> list[Path]:
    """Curve-level targets for all 798 curves plus a per-vacancy swing summary."""
    ds = load_dataset(cfg)
    table = curve_targets(ds.metrics_pooled(), ds.iv_pooled(), cfg)
    summary = table.groupby("vac_pct").agg(
        swing_median_mV_per_dec=("swing_mV_per_dec", "median"),
        swing_raw_median_mV_per_dec=("swing_raw_mV_per_dec", "median"),
        log10_Ion_Ioff_median=("log10_Ion_Ioff", "median"),
        log10_Ion_Ioff_min_median=("log10_Ion_Ioff_min", "median"),
    ).reset_index()
    paths = [out_dir / "curve_targets.csv", out_dir / "swing_by_vacancy.csv"]
    table.to_csv(paths[0], index=False)
    summary.to_csv(paths[1], index=False)
    return paths


def _describe(split: Split, protocol: str) -> dict[str, Any]:
    """One summary row: sizes and T / vacancy ranges of both sides of a split."""
    row: dict[str, Any] = {"protocol": protocol, "split": split.name}
    for side in ("train", "test"):
        k = getattr(split, side)
        levels = sorted(k["vac_pct"].round(1).unique())
        row |= {f"{side}_curves": len(k), f"{side}_T_K": f"{k.T_K.min()}-{k.T_K.max()}",
                f"{side}_vac_pct": " ".join(f"{v:g}" for v in levels)}
    return row


def step_splits(cfg: dict[str, Any], out_dir: Path) -> list[Path]:
    """Every final split and every CV fold inside its training pool, as one table."""
    ds = load_dataset(cfg)
    curves = curve_targets(ds.metrics_pooled(), ds.iv_pooled(), cfg)
    rows = []
    for split in all_protocols(curves, cfg):
        rows.append(_describe(split, split.name))
        rows += [_describe(f, split.name) for f in blocked_cv_folds(split.train, cfg)]
    path = out_dir / "split_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return [path]


STEPS = {"targets": step_targets, "splits": step_splits}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("steps", nargs="+", choices=sorted(STEPS))
    args = parser.parse_args()
    cfg = load_config()
    out_dir = repo_path(cfg, "results_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    timings = []
    for name in args.steps:
        t0 = time.perf_counter()
        for p in STEPS[name](cfg, out_dir):
            print(f"wrote {p}")
        timings.append({"step": name, "seconds": round(time.perf_counter() - t0, 2)})
    pd.DataFrame(timings).to_csv(out_dir / "timings_last_run.csv", index=False)
    print(pd.DataFrame(timings).to_string(index=False))


if __name__ == "__main__":
    main()
