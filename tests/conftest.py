"""Shared pytest fixtures."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from ffrefine.io.structure import Structure

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


@pytest.fixture
def xtb_bin() -> str:
    """Path to xtb, or skip the test when xtb is unavailable."""
    exe = shutil.which("xtb")
    if exe is None:
        pytest.skip("xtb not on PATH")
    return exe


@pytest.fixture
def rotated_frames() -> Callable[[Structure, int], np.ndarray]:
    """Factory building frames that sample a molecule's target torsion.

    Rotates the "moving" side of the butane fixture (atoms 8-14) rigidly
    about the C2-C3 bond (atoms 5-8), which sweeps the C1-C2-C3-C4 dihedral
    through a full turn while leaving every other internal coordinate fixed.

    Used by tests/test_selfcheck.py and tests/test_roundtrip.py (see Task 13
    ruling R2+R13: this lives here, not as a helper imported from another test
    module, since tests/ has no __init__.py and cross-module test imports
    cannot resolve under pytest's default rootdir-relative collection).
    """

    def _make(structure: Structure, n: int = 12) -> np.ndarray:
        base = structure.coords
        axis_a, axis_b = base[4], base[7]  # atoms 5 and 8, 0-based
        axis = (axis_b - axis_a) / np.linalg.norm(axis_b - axis_a)
        moving = [7, 8, 9, 10, 11, 12, 13]  # C3 side: 0-based indices of atoms 8..14
        out = []
        for angle in np.linspace(0.0, 2 * np.pi, n, endpoint=False):
            c = base.copy()
            k = axis
            for idx in moving:
                v = c[idx] - axis_a
                c[idx] = (
                    axis_a
                    + v * np.cos(angle)
                    + np.cross(k, v) * np.sin(angle)
                    + k * np.dot(k, v) * (1 - np.cos(angle))
                )
            out.append(c)
        return np.stack(out)

    return _make


@pytest.fixture
def butane_xyz(tmp_path, butane_pdb) -> Path:
    """An .xyz of the butane fixture, written from the .pdb.

    Derived from the .pdb rather than checked in so the two can never drift
    apart: an .xyz carries elements and coordinates only, which is exactly the
    point of the tests that use it.
    """
    from ffrefine.io.structure import read_pdb, write_xyz

    s = read_pdb(butane_pdb)
    out = tmp_path / "butane.xyz"
    write_xyz(out, s, comment="butane, from butane.pdb")
    return out
