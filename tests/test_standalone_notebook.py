"""Drift guard: the standalone notebook must equal what the generator makes from the current code."""
from __future__ import annotations

import json
from pathlib import Path

import build_standalone_notebook as B


def _cells(path: Path) -> list[tuple[str, str]]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return [(c["cell_type"], "".join(c["source"]).replace("\r\n", "\n")) for c in nb["cells"]]


def test_standalone_notebook_is_up_to_date(tmp_path):
    """Fails if package code changed without re-running python/build_standalone_notebook.py."""
    fresh = B.build(tmp_path / "fresh.ipynb")
    assert _cells(fresh) == _cells(B.OUT), "re-run: python python/build_standalone_notebook.py"


def test_standalone_notebook_has_no_package_imports():
    code = [s for kind, s in _cells(B.OUT) if kind == "code"]
    assert not any("from wte2ml" in s or "import wte2ml" in s for s in code)
    assert any("def run_job(" in s for s in code) and any("class MLPRegressor" in s for s in code)
