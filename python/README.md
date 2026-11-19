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
