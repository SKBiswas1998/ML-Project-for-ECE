"""Run pipeline steps and write results/*.csv. From the repo root:

    .\\.venv\\Scripts\\python python\\run_experiments.py targets

Each step writes its own CSVs and times itself. Later stages add model steps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wte2ml.config import load_config, repo_path
from wte2ml.data import load_dataset
from wte2ml.evaluate import regression_metrics, skill_score
from wte2ml.models import MODELS, NAIVE_MODELS, build_model, features, tune_and_fit
from wte2ml.splits import Split, all_protocols, blocked_cv_folds, fold_indices
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


def _config_hash(cfg: dict[str, Any]) -> str:
    """Short hash of every config section that changes model results (cache key)."""
    parts = {k: cfg[k] for k in ("data", "targets", "splits", "inputs", "experiments", "models")}
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:8]


def _predict(model: Any, X: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Mean prediction, plus predictive std for the GP (NaN for other models)."""
    if name == "gp":
        return model.predict(X, return_std=True)
    return model.predict(X), np.full(len(X), np.nan)


def run_job(cfg: dict[str, Any], split: Split, table: pd.DataFrame, feature: str,
            target: str, model_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Tune one model by blocked CV inside ``split.train``, refit, predict ``split.test`` once.

    ``table`` is the row table of the task (IV rows for log10_I, curves otherwise).
    Returns the test predictions (one row per test row) and a metadata dict.
    """
    gate = cfg["data"]["gate_column"] if target == cfg["experiments"]["point_target"] else None
    train, test = (t.reset_index(drop=True) for t in split.apply(table))
    cv = fold_indices(train, blocked_cv_folds(train, cfg))
    est, grid = build_model(model_name, cfg, pointwise=gate is not None)
    t0 = time.perf_counter()
    model, params, cv_mse = tune_and_fit(est, grid, features(train, feature, gate), train[target].to_numpy(), cv, cfg)
    mean, std = _predict(model, features(test, feature, gate), model_name)
    cols = ["T_K", "vac_pct"] + ([gate] if gate else [])
    pred = test[cols].rename(columns={gate: "gate_V"} if gate else {}).assign(
        y_true=test[target].to_numpy(), y_pred=mean, y_std=std)
    meta = {"cv_mse": cv_mse, "best_params": json.dumps({k: str(v) for k, v in params.items()}),
            "fit_seconds": round(time.perf_counter() - t0, 2)}
    return pred, meta


def _cached_job(cfg: dict[str, Any], cache: Path, job: dict[str, str], split: Split,
                table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """run_job with an on-disk cache, so an interrupted run resumes where it stopped."""
    path = cache / ("__".join(job.values()) + ".pkl")
    if path.exists():
        return pd.read_pickle(path)
    pred, meta = run_job(cfg, split, table, job["feature"], job["target"], job["model"])
    pred = pred.assign(**job)
    pd.to_pickle((pred, meta), path)
    print(f"  {job} cv_mse={meta['cv_mse']:.4g} ({meta['fit_seconds']} s)", flush=True)
    return pred, meta


def _jobs(cfg: dict[str, Any], splits: list[Split]):
    """Every (protocol, vacancy feature, target, model) combination, in run order."""
    e = cfg["experiments"]
    tasks = [(e["point_target"], f) for f in e["point_vacancy_features"]] + [
        (t, f) for t in e["curve_targets"] for f in cfg["inputs"]["vacancy_features"]]
    for split in splits:
        for target, feature in tasks:
            for model in MODELS:
                yield split, {"protocol": split.name, "feature": feature, "target": target, "model": model}


def _score(pred: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Metrics per job plus skill against the best naive baseline of the same task."""
    task = ["protocol", "feature", "target"]
    rows = [{**dict(zip([*task, "model"], k)), **regression_metrics(g.y_true, g.y_pred)}
            for k, g in pred.groupby([*task, "model"], sort=False)]
    m = pd.DataFrame(rows).merge(meta, on=[*task, "model"])
    best_naive = m[m.model.isin(NAIVE_MODELS)].groupby(task).MSE.min().rename("MSE_best_naive")
    m = m.merge(best_naive, left_on=task, right_index=True)
    return m.assign(skill=[skill_score(a, b) for a, b in zip(m.MSE, m.MSE_best_naive)])


def step_classical(cfg: dict[str, Any], out_dir: Path) -> list[Path]:
    """Baselines, ridge, polynomial ridge, boosting and GP on every protocol and target."""
    ds = load_dataset(cfg)
    curves = curve_targets(ds.metrics_pooled(), ds.iv_pooled(), cfg)
    iv = ds.iv_pooled().reset_index(drop=True)
    cache = repo_path(cfg, "cache_dir") / f"classical_{_config_hash(cfg)}"
    cache.mkdir(parents=True, exist_ok=True)
    preds, metas = [], []
    for split, job in _jobs(cfg, all_protocols(curves, cfg)):
        table = iv if job["target"] == cfg["experiments"]["point_target"] else curves
        pred, meta = _cached_job(cfg, cache, job, split, table)
        preds.append(pred)
        metas.append({**job, **meta})
    pred = pd.concat(preds, ignore_index=True)
    paths = [out_dir / "predictions_classical.csv.gz", out_dir / "metrics_classical.csv"]
    pred.to_csv(paths[0], index=False)
    _score(pred, pd.DataFrame(metas)).to_csv(paths[1], index=False)
    return paths


STEPS = {"targets": step_targets, "splits": step_splits, "classical": step_classical}


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
