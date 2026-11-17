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
