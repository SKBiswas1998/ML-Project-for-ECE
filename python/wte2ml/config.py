"""Load config.yaml and resolve paths against the repository root."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PYTHON_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PYTHON_DIR.parent
DEFAULT_CONFIG = PYTHON_DIR / "config.yaml"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read the YAML config into a plain dict.

    Exists so that every module gets its constants from one file instead of
    hard-coding them; the dict is passed explicitly (no global state).
    """
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def repo_path(cfg: dict[str, Any], key: str) -> Path:
    """Absolute path of ``cfg['paths'][key]``, relative paths taken from the repo root."""
    return REPO_ROOT / cfg["paths"][key]
