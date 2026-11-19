"""Generate notebooks/WTe2_TIFET_standalone.ipynb: the whole pipeline as notebook cells.

Run from the repo root:  .\\.venv\\Scripts\\python python\\build_standalone_notebook.py

The notebook is generated FROM the package source (one cell per function or class, in
data-flow order), so it never drifts from the code: after any code change, re-run this
script. Only the data files are downloaded from GitHub; every result is recomputed.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

PYTHON = Path(__file__).resolve().parent
REPO = PYTHON.parent
OUT = REPO / "notebooks" / "WTe2_TIFET_standalone.ipynb"
RAW = "https://raw.githubusercontent.com/SKBiswas1998/ML-Project-for-ECE/{branch}/{path}"

# (file, section title) in the order the data flows; config.py is replaced by a notebook cell.
MODULES = [
    ("wte2ml/data.py", "Loading and validating the data"),
    ("wte2ml/targets.py", "Learning targets and subthreshold swing"),
    ("wte2ml/splits.py", "Evaluation protocols and blocked cross-validation"),
    ("wte2ml/evaluate.py", "Metrics, skill and comparison tables"),
    ("wte2ml/neural.py", "The MLP (PyTorch, CPU)"),
    ("wte2ml/models.py", "All models behind one fit/predict interface"),
    ("wte2ml/unsupervised.py", "PCA and clustering"),
    ("wte2ml/robustness.py", "Input-noise robustness"),
    ("wte2ml/uncertainty.py", "Cluster bootstrap and GP coverage"),
    ("wte2ml/ridge_numpy.py", "Ridge regression from scratch"),
    ("wte2ml/plots.py", "Plotting primitives"),
    ("run_experiments.py", "Experiment steps (each writes results/*.csv)"),
    ("make_figures.py", "Figures (read results/*.csv, write figures/*.png)"),
]
SKIP = {"main"}                                   # command-line entry points are not needed
DATA_FILES = ["ML_IV_train.csv", "ML_IV_test.csv", "ML_metrics_train.csv", "ML_metrics_test.csv", "NEGF_all.mat"]
REUSABLE = {"classical": ["metrics_classical.csv", "predictions_classical.csv.gz"],
            "mlp": ["metrics_mlp.csv", "predictions_mlp.csv.gz", "mlp_vs_classical.csv"],
            "epoch_sensitivity": ["epoch_sensitivity.csv"]}


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def _is_local_import(node: ast.stmt) -> bool:
    """Imports of the package itself, of __future__, and of matplotlib backend switching."""
    if isinstance(node, ast.ImportFrom):
        return (node.module or "").startswith(("wte2ml", "__future__"))
    return False


def _segment(src: str, node: ast.stmt) -> str:
    """Source of a top-level statement including its decorators and the comments directly above it."""
    lines = src.splitlines()
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]) - 1
    while start > 0 and lines[start - 1].lstrip().startswith("#"):
        start -= 1
    return "\n".join(lines[start:node.end_lineno])


def _first_paragraph(doc: str | None) -> str:
    return (doc or "").strip().split("\n\n")[0].replace("\n", " ") if doc else ""


def module_cells(rel: str, title: str) -> list[dict]:
    """Markdown intro from the module docstring, one cell of imports/constants, then one cell per def."""
    src = (PYTHON / rel).read_text(encoding="utf-8")
    tree = ast.parse(src)
    cells = [md(f"## {title}\n*Source: `python/{rel}`*\n\n{ast.get_docstring(tree) or ''}")]
    body = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant))]
    pending: list[str] = []          # consecutive imports/constants, kept in source order
    for n in body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            if pending:
                cells.append(code("\n".join(pending)))
                pending = []
            if n.name not in SKIP:
                kind = "class" if isinstance(n, ast.ClassDef) else "function"
                cells.append(md(f"### `{n.name}` ({kind})\n{_first_paragraph(ast.get_docstring(n))}"))
                cells.append(code(_segment(src, n)))
        elif not isinstance(n, ast.If) and not _is_local_import(n):   # skip `if __name__ == "__main__"`
            pending.append(_segment(src, n))
    if pending:
        cells.append(code("\n".join(pending)))
    return cells


def intro_cells() -> list[dict]:
    config_text = (PYTHON / "config.yaml").read_text(encoding="utf-8")
    return [
        md("""
# WTe₂ TIFET ML pipeline: complete standalone notebook
**ECE 455/555 course project, Shuvankar Biswas**

This notebook contains **the entire codebase** as cells: data loading and validation, targets and swing, evaluation protocols with blocked CV, naive baselines, ridge, polynomial ridge, gradient boosting, Gaussian processes, the MLP, PCA and clustering, input-noise robustness, cluster-bootstrap confidence intervals, GP coverage, the epoch-cap check and every figure. **Only the data** (the four NEGF CSV files and `NEGF_all.mat`) is downloaded from GitHub; everything else is computed here.

**How to run (Kaggle recommended):** turn **Internet on** (needed for the data download), choose a CPU machine, then *Run All*. Settings are in the next two cells. A full run uses 4 parallel fits; expect about 1–1.5 h on Kaggle (2 h 16 min with a serial MLP on the author's laptop). Finished jobs are cached under `results/cache`, so re-running after an interruption continues where it stopped.

The notebook was generated from the package source by `python/build_standalone_notebook.py`, so it matches the repository code; the theory behind each step is in `WTe2_TIFET_theory.ipynb`.
"""),
        md("## Run settings"),
        code(f"""
BRANCH = "test_1"                 # GitHub branch to download the data from
RUN_STEPS = ["targets", "splits", "classical", "mlp", "unsupervised", "robustness",
             "uncertainty", "epoch_sensitivity"]           # remove a step to skip it
REUSE_GITHUB_RESULTS = []         # e.g. ["classical", "mlp", "epoch_sensitivity"]: download those results
                                  # from GitHub instead of recomputing (minutes instead of about an hour)
N_JOBS = 4                        # parallel fits for every step (Kaggle: 4 CPU cores, about 30 GB RAM)
REUSABLE = {REUSABLE!r}
"""),
        md("## Setup: packages, working folder, data download from GitHub"),
        code(f"""
import json, os, subprocess, sys, time, urllib.request
from pathlib import Path
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyyaml", "psutil", "scikit-learn>=1.5"], check=True)
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings; warnings.filterwarnings("ignore")

WORK = Path("/kaggle/working/wte2_run") if Path("/kaggle/working").exists() else Path.cwd() / "wte2_run"
RAW = "{RAW}"
def download(path: str) -> Path:
    \"\"\"Fetch one file of the repository (data or saved results) into WORK, unless already there.\"\"\"
    target = WORK / path
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(RAW.format(branch=BRANCH, path=path), target)
    return target
for name in {DATA_FILES!r}:
    print("data:", download("ML_edge_vacancy_seed2026/" + name), flush=True)
"""),
        md("## Configuration\nEvery constant of the pipeline (the repository's `python/config.yaml`), with the Kaggle parallelism applied below it. Edit here to change a setting."),
        code(f'''
import yaml
CONFIG_YAML = r"""
{config_text}"""
_CONFIG = yaml.safe_load(CONFIG_YAML)
_CONFIG["compute"]["n_jobs"] = N_JOBS        # notebook setting: parallel classical fits
_CONFIG["compute"]["mlp_n_jobs"] = N_JOBS    # and parallel MLP fits (enough memory on Kaggle)
REPO_ROOT = WORK

def load_config(path=None):
    """The configuration dict (a fresh copy each call, so callers cannot change it for others)."""
    return json.loads(json.dumps(_CONFIG))

def repo_path(cfg, key):
    """Absolute path of cfg['paths'][key] inside the working folder."""
    return REPO_ROOT / cfg["paths"][key]
'''),
        md("## Imports used by all modules"),
        code("from __future__ import annotations"),
    ]


def run_cells() -> list[dict]:
    return [
        md("## Run the pipeline\nEach step writes its `results/*.csv`; times are appended to `results/timings_log.csv`. Steps listed in `REUSE_GITHUB_RESULTS` download the saved results instead."),
        code("""
cfg = {**load_config(), "run": {"protocols": None}}
out_dir = repo_path(cfg, "results_dir"); out_dir.mkdir(parents=True, exist_ok=True)
repo_path(cfg, "figures_dir").mkdir(parents=True, exist_ok=True)
timings = []
for step in RUN_STEPS:
    t0 = time.perf_counter()
    if step in REUSE_GITHUB_RESULTS:
        for f in REUSABLE[step]:
            download("results/" + f)
        how = "downloaded"
    else:
        for p in STEPS[step](cfg, out_dir):
            print("  wrote", p.name, flush=True)
        how = "computed"
    timings.append({"step": step, "how": how, "seconds": round(time.perf_counter() - t0, 1)})
    print(f"{step}: {how} in {timings[-1]['seconds']} s", flush=True)
import pandas as pd
log = out_dir / "timings_log.csv"          # appended, so every run's step times stay on record
pd.DataFrame(timings).assign(finished=time.strftime("%Y-%m-%d %H:%M:%S"), n_jobs=N_JOBS).to_csv(
    log, mode="a", header=not log.exists(), index=False)
pd.DataFrame(timings)
"""),
        md("## Make every figure"),
        code("""
import matplotlib.pyplot as plt
from IPython.display import Image, display
fig_dir = repo_path(cfg, "figures_dir")
paths = [figure_swing_check(cfg, fig_dir), figure_pca_clusters(cfg, fig_dir), *report_figures(cfg, fig_dir)]
plt.close("all")
for p in paths:
    print(p.name); display(Image(str(p)))
"""),
        md("## Key result tables"),
        code("""
comparison = pd.read_csv(out_dir / "model_comparison.csv")
comparison[comparison.target == "log10_Ion_Ioff"].pivot(index="model", columns="protocol", values="MAE").round(3)
"""),
        code("""
boot = pd.read_csv(out_dir / "bootstrap_ci.csv")
best = comparison[comparison["rank"] == 1][["protocol", "target", "model"]]
best.merge(boot, on=["protocol", "target", "model"])[["target", "protocol", "model", "n_levels", "MAE", "MAE_lo", "MAE_hi"]].round(3)
"""),
        code('pd.read_csv(out_dir / "gp_coverage.csv")[["protocol", "target", "coverage_68", "coverage_95"]].round(3)'),
        code('pd.read_csv(out_dir / "robustness_summary.csv").pivot_table(index=["protocol", "model"], columns="noise_fraction", values="MAE_mean").round(3)'),
        code('pd.read_csv(out_dir / "timings_log.csv")'),
    ]


def build(out: Path = OUT) -> Path:
    cells = intro_cells()
    for rel, title in MODULES:
        cells += module_cells(rel, title)
    cells += run_cells()
    for i, c in enumerate(cells):
        c["id"] = f"cell-{i:03d}"
    nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}}
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


if __name__ == "__main__":
    print("wrote", build())
