"""Runner test: the curve-output MLP layout lines up with the per-point rows it is scored on."""
from __future__ import annotations

import numpy as np
import pandas as pd

import run_experiments as R
from wte2ml.config import load_config
from wte2ml.data import load_dataset
from wte2ml.splits import all_protocols
from wte2ml.targets import curve_targets


def test_curve_output_rows_match_iv_rows():
    """If the 20 outputs of a curve were flattened in the wrong order, every I-V score would be
    computed against the wrong gate point. Check y_test equals the IV table row by row."""
    cfg = load_config()
    ds = load_dataset(cfg)
    iv = ds.iv_pooled().reset_index(drop=True)
    split = all_protocols(curve_targets(ds.metrics_pooled(), iv, cfg), cfg)[2]   # vacancy_block
    d = R._curve_output_data(cfg, split, iv, "vac_pct")
    assert d["X_train"].shape == (len(split.train), 2) and d["y_train"].shape == (len(split.train), 20)
    merged = d["test_keys"].assign(y=d["y_test"]).merge(
        iv.rename(columns={"VTG_minus_VBG_V": "gate_V"}), on=["T_K", "vac_pct", "gate_V"])
    assert len(merged) == len(d["test_keys"]) == 20 * len(split.test)
    assert np.array_equal(merged["y"].to_numpy(), merged["log10_I"].to_numpy())
    # CV folds are curve positions inside the training pool, never test curves.
    assert all(max(tr.max(), va.max()) < len(split.train) for tr, va in d["cv"])


def test_memory_log_appends_one_row_per_job(tmp_path):
    """Each newly computed job adds a row with its time and this process's memory, so growth
    over a run can be read off the log (cached jobs add nothing)."""
    log = tmp_path / "memory_log.csv"
    job = {"protocol": "p", "feature": "f", "target": "t"}
    R._log_memory(log, job, 1.5)
    R._log_memory(log, job, 2.5)
    rows = pd.read_csv(log)
    assert len(rows) == 2 and rows.job_seconds.tolist() == [1.5, 2.5]
    assert (rows.rss_MB > 0).all() and {"peak_MB", "pid"} <= set(rows.columns)
