"""Run pipeline steps and write results/*.csv. From the repo root:

    .\\.venv\\Scripts\\python python\\run_experiments.py targets

Each step writes its own CSVs and times itself. Later stages add model steps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone

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


MLP_KEYS = ("mlp", "mlp_targets", "mlp_vacancy_feature")   # config keys only the MLP step uses


def _config_hash(cfg: dict[str, Any], step: str) -> str:
    """Short hash of the config settings that change ``step``'s results (cache key).

    The classical step ignores MLP-only keys (and vice versa for model settings), so
    adding or tuning the MLP does not invalidate the 19-minute classical cache.
    """
    parts = {k: cfg[k] for k in ("data", "targets", "splits", "inputs", "experiments", "models")}
    if step == "classical":
        parts = {k: ({kk: vv for kk, vv in v.items() if kk not in MLP_KEYS} if isinstance(v, dict) else v)
                 for k, v in parts.items()}
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
    d = _task_data(cfg, split, table, feature, target)
    est, grid = build_model(model_name, cfg, pointwise=d["gate"] is not None)
    t0 = time.perf_counter()
    model, params, cv_mse = tune_and_fit(est, grid, d["X_train"], d["y_train"], d["cv"], cfg)
    mean, std = _predict(model, d["X_test"], model_name)
    pred = d["test_keys"].assign(y_true=d["y_test"], y_pred=mean, y_std=std)
    meta = {"cv_mse": cv_mse, "best_params": json.dumps({k: str(v) for k, v in params.items()}),
            "fit_seconds": round(time.perf_counter() - t0, 2)}
    return pred, meta


def _task_data(cfg: dict[str, Any], split: Split, table: pd.DataFrame, feature: str,
               target: str) -> dict[str, Any]:
    """Everything one job needs: X/y for train and test, blocked-CV fold indices inside
    the training pool, the gate column (None for curve targets) and test row keys."""
    gate = cfg["data"]["gate_column"] if target == cfg["experiments"]["point_target"] else None
    train, test = (t.reset_index(drop=True) for t in split.apply(table))
    keys = test[["T_K", "vac_pct"] + ([gate] if gate else [])]
    return {
        "gate": gate,
        "X_train": features(train, feature, gate), "y_train": train[target].to_numpy(),
        "X_test": features(test, feature, gate), "y_test": test[target].to_numpy(),
        "cv": fold_indices(train, blocked_cv_folds(train, cfg)),
        "test_keys": keys.rename(columns={gate: "gate_V"} if gate else {}),
    }


def _cached(cache: Path, job: dict[str, str], compute: Callable[[], tuple[pd.DataFrame, dict]]
            ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a job's (predictions, metadata) from disk, or run ``compute`` and store it.
    Makes long runs resumable: an interrupted run skips every finished job."""
    path = cache / ("__".join(job.values()) + ".pkl")
    if path.exists():
        return pd.read_pickle(path)
    pred, meta = compute()
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
    cache = repo_path(cfg, "cache_dir") / f"classical_{_config_hash(cfg, 'classical')}"
    cache.mkdir(parents=True, exist_ok=True)
    preds, metas = [], []
    for split, job in _jobs(cfg, all_protocols(curves, cfg)):
        table = iv if job["target"] == cfg["experiments"]["point_target"] else curves
        pred, meta = _cached(cache, job, partial(run_job, cfg, split, table, job["feature"], job["target"], job["model"]))
        preds.append(pred)
        metas.append({**job, **meta})
    pred = pd.concat(preds, ignore_index=True)
    paths = [out_dir / "predictions_classical.csv.gz", out_dir / "metrics_classical.csv"]
    pred.to_csv(paths[0], index=False)
    _score(pred, pd.DataFrame(metas)).to_csv(paths[1], index=False)
    return paths


def _fit_seed(best: Any, seed: int, d: dict[str, Any]) -> tuple[np.ndarray, int]:
    """Refit the tuned MLP with one initialisation seed; return test predictions, best epoch."""
    model = clone(best).set_params(seed=seed).fit(d["X_train"], d["y_train"])
    return model.predict(d["X_test"]), model.best_epoch_


def run_mlp_job(cfg: dict[str, Any], split: Split, table: pd.DataFrame, target: str
                ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Tune the MLP by blocked CV with the first seed, then refit with every seed in
    ``models.mlp.seeds`` (in parallel). Returns predictions (one block per seed) and metadata."""
    d = _task_data(cfg, split, table, cfg["experiments"]["mlp_vacancy_feature"], target)
    est, grid = build_model("mlp", cfg, pointwise=d["gate"] is not None)
    t0 = time.perf_counter()
    best, params, cv_mse = tune_and_fit(est, grid, d["X_train"], d["y_train"], d["cv"], cfg)
    seeds = cfg["models"]["mlp"]["seeds"]
    runs = Parallel(n_jobs=cfg["compute"]["n_jobs"])(delayed(_fit_seed)(best, s, d) for s in seeds)
    pred = pd.concat([d["test_keys"].assign(seed=s, y_true=d["y_test"], y_pred=p, y_std=np.nan)
                      for s, (p, _) in zip(seeds, runs)], ignore_index=True)
    meta = {"cv_mse": cv_mse, "best_params": json.dumps({k: str(v) for k, v in params.items()}),
            "best_epochs": json.dumps([int(e) for _, e in runs]),
            "fit_seconds": round(time.perf_counter() - t0, 2)}
    return pred, meta


def _mlp_score(pred: pd.DataFrame, meta: pd.DataFrame, classical: pd.DataFrame) -> pd.DataFrame:
    """Per-seed metrics and skill against the best naive baseline from the classical run."""
    task = ["protocol", "feature", "target"]
    rows = [{**dict(zip([*task, "seed"], k)), **regression_metrics(g.y_true, g.y_pred)}
            for k, g in pred.groupby([*task, "seed"], sort=False)]
    naive = classical[classical.model.isin(NAIVE_MODELS)].groupby(task).MSE.min().rename("MSE_best_naive")
    m = pd.DataFrame(rows).merge(meta, on=task).merge(naive, left_on=task, right_index=True)
    return m.assign(model="mlp", skill=[skill_score(a, b) for a, b in zip(m.MSE, m.MSE_best_naive)])


def _mlp_summary(per_seed: pd.DataFrame, classical: pd.DataFrame) -> pd.DataFrame:
    """MLP mean +/- std over seeds next to the best classical (non-naive) and best naive model."""
    task = ["protocol", "feature", "target"]
    s = per_seed.groupby(task).agg(mlp_MAE_mean=("MAE", "mean"), mlp_MAE_std=("MAE", "std"),
                                   mlp_skill_mean=("skill", "mean"), mlp_skill_std=("skill", "std"))
    c = classical.set_index(task).loc[s.index.unique()]
    best = (c[~c.model.isin(NAIVE_MODELS)].reset_index().sort_values("MAE").groupby(task).first()
            [["model", "MAE", "skill"]].add_prefix("best_classical_"))
    naive = c[c.model.isin(NAIVE_MODELS)].groupby(task).MAE.min().rename("best_naive_MAE")
    out = s.join(best).join(naive).reset_index()
    return out.assign(mlp_beats_best_classical=out.mlp_MAE_mean < out.best_classical_MAE)


def step_mlp(cfg: dict[str, Any], out_dir: Path) -> list[Path]:
    """MLP on every protocol for ``experiments.mlp_targets``; needs the classical results."""
    ds = load_dataset(cfg)
    curves = curve_targets(ds.metrics_pooled(), ds.iv_pooled(), cfg)
    iv = ds.iv_pooled().reset_index(drop=True)
    classical = pd.read_csv(out_dir / "metrics_classical.csv")
    cache = repo_path(cfg, "cache_dir") / f"mlp_{_config_hash(cfg, 'mlp')}"
    cache.mkdir(parents=True, exist_ok=True)
    preds, metas = [], []
    for split in all_protocols(curves, cfg):
        for target in cfg["experiments"]["mlp_targets"]:
            job = {"protocol": split.name, "feature": cfg["experiments"]["mlp_vacancy_feature"], "target": target}
            table = iv if target == cfg["experiments"]["point_target"] else curves
            pred, meta = _cached(cache, job, partial(run_mlp_job, cfg, split, table, target))
            preds.append(pred)
            metas.append({**job, **meta})
    pred = pd.concat(preds, ignore_index=True)
    per_seed = _mlp_score(pred, pd.DataFrame(metas), classical)
    paths = [out_dir / "predictions_mlp.csv.gz", out_dir / "metrics_mlp.csv", out_dir / "mlp_vs_classical.csv"]
    pred.to_csv(paths[0], index=False)
    per_seed.to_csv(paths[1], index=False)
    _mlp_summary(per_seed, classical).to_csv(paths[2], index=False)
    return paths


STEPS = {"targets": step_targets, "splits": step_splits, "classical": step_classical, "mlp": step_mlp}


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
