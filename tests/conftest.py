"""Shared pytest fixtures."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def butane_itp() -> Path:
    return FIXTURES / "butane" / "butane.itp"


@pytest.fixture
def butane_pdb() -> Path:
    return FIXTURES / "butane" / "butane.pdb"


@pytest.fixture
def butane_gro() -> Path:
    return FIXTURES / "butane" / "butane.gro"


@pytest.fixture
def gmx_bin() -> str:
    """Path to gmx, or skip the test when GROMACS is unavailable."""
    exe = shutil.which("gmx") or shutil.which("gmx_mpi")
    if exe is None:
        pytest.skip("gmx not on PATH")
    return exe
