"""Layering rules the architecture relies on.

``ffrefine/charges/`` and ``ffrefine/io/`` decide things; they never start a
process. That is what lets their numerics be tested with no ORCA, xtb or
GROMACS installed (milestone 5 ruling R26: charges/apply.py once imported
CHARGE_TOLERANCE from qm.xtb and so loaded the runner and subprocess).
Checked on the import statements themselves, relative imports resolved.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).parent.parent / "ffrefine"

FORBIDDEN = {
    "charges": ("subprocess", "ffrefine.runner", "ffrefine.qm"),
    "io": ("subprocess", "ffrefine.runner", "ffrefine.qm"),
}


def _imported_modules(path: Path) -> list[str]:
    """Every module an import statement in ``path`` names, relative imports resolved."""
    package = ["ffrefine", *path.relative_to(PACKAGE).parts[:-1]]
    out: list[str] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            out.append(module)
            out += [f"{module}.{alias.name}" for alias in node.names]
    return out


def _modules(subpackage: str) -> list[Path]:
    return sorted((PACKAGE / subpackage).glob("*.py"))


@pytest.mark.parametrize(
    "subpackage, path",
    [(sub, p) for sub in FORBIDDEN for p in _modules(sub)],
    ids=lambda v: v if isinstance(v, str) else v.name,
)
def test_pure_subpackages_start_no_process(subpackage, path):
    forbidden = FORBIDDEN[subpackage]
    bad = [
        m
        for m in _imported_modules(path)
        if any(m == f or m.startswith(f + ".") for f in forbidden)
    ]
    assert not bad, f"ffrefine/{subpackage}/{path.name} imports {bad}"


def test_the_checker_resolves_relative_imports():
    """Guard against a checker that silently passes everything."""
    apply = PACKAGE / "charges" / "apply.py"
    assert "ffrefine.io.topology" in _imported_modules(apply)
