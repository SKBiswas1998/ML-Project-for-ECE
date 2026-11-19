# Python ML pipeline (CPU only)

## Setup (PowerShell, from the repository root)
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Check
```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m ruff check python tests
```

Every constant is in `config.yaml`. The data flow and glossary are in `docs/ARCHITECTURE.md`.

## Run in Google Colab or Kaggle
Open `notebooks/WTe2_TIFET_pipeline.ipynb`:
- **Colab:** https://colab.research.google.com/github/SKBiswas1998/ML-Project-for-ECE/blob/test_1/notebooks/WTe2_TIFET_pipeline.ipynb
- **Kaggle:** *File → Import Notebook → GitHub*, then paste the repository URL and choose the notebook. Turn **Internet on** in the notebook settings, because it clones the repository.

The notebook clones the repository, installs only the missing packages, runs the tests, and shows every stage's results from `results/*.csv` (about 2 min). Set `RERUN_CLASSICAL` / `RERUN_MLP` to `True` to recompute the models from scratch (about 31 / 100 min on a laptop CPU; robustness about 4 min).

## Theory notebook
`notebooks/WTe2_TIFET_theory.ipynb` explains every component in the order the data flows through the pipeline: what we do, the theory, what is optimised, and its figures and result tables, all loaded from the repository. It runs in under a minute and needs no model re-runs.
- **Colab:** https://colab.research.google.com/github/SKBiswas1998/ML-Project-for-ECE/blob/test_1/notebooks/WTe2_TIFET_theory.ipynb
- **Kaggle:** *File → Import Notebook → GitHub*, with Internet on.

## Standalone notebook (whole pipeline, only the data from GitHub)
`notebooks/WTe2_TIFET_standalone.ipynb` contains the entire codebase as cells (one per function or class, in data-flow order) and downloads only the four NEGF CSV files and `NEGF_all.mat` from GitHub. Every result and figure is recomputed.
- **Kaggle (recommended):** *File → Import Notebook → Upload* (or GitHub), **Internet on**, CPU machine, *Run All*. It uses 4 parallel fits. A full run is estimated at about 1–1.5 h on Kaggle; this is not yet measured there. Finished jobs are cached, so a re-run resumes.
- **Faster look:** in the first code cell, set `REUSE_GITHUB_RESULTS = ["classical", "mlp", "epoch_sensitivity"]` to download those saved results instead of recomputing them (about 5 min in total).
- **Keep it in sync:** the notebook is generated from the source. After any code change run `.\.venv\Scripts\python python\build_standalone_notebook.py`; a test fails if you forget.
