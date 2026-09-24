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
def orca_bin() -> str:
    """Path to orca, or skip the test when ORCA is unavailable."""
    exe = shutil.which("orca")
    if exe is None:
        pytest.skip("orca not on PATH")
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
def rotate_side() -> Callable[..., np.ndarray]:
    """Factory rotating one side of a bond rigidly about that bond.

    ``fixed`` and ``moving`` are 1-based atom numbers of the bond. Every atom
    reachable from ``moving`` without crossing the bond turns about the
    fixed->moving axis, once per angle in ``angles_deg``; the rest stay put,
    so only torsions about this bond change. A bond in a ring has no such
    side and is refused.
    """

    def _make(
        coords: np.ndarray,
        bonds: list[tuple[int, int]],
        fixed: int,
        moving: int,
        angles_deg,
    ) -> np.ndarray:
        adjacency: dict[int, set[int]] = {}
        for a, b in bonds:
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)
        side, stack = {moving}, [moving]
        while stack:
            atom = stack.pop()
            for n in adjacency.get(atom, ()):
                if atom == moving and n == fixed:
                    continue
                if n == fixed:
                    raise AssertionError(f"bond {fixed}-{moving} is in a ring")
                if n not in side:
                    side.add(n)
                    stack.append(n)
        idx = sorted(a - 1 for a in side)  # 1-based -> 0-based
        origin = coords[fixed - 1]
        axis = coords[moving - 1] - origin
        axis = axis / np.linalg.norm(axis)
        frames = []
        for angle in np.radians(np.asarray(angles_deg, dtype=float)):
            c = np.array(coords, dtype=float, copy=True)
            v = c[idx] - origin
            c[idx] = (
                origin
                + v * np.cos(angle)
                + np.cross(axis, v) * np.sin(angle)
                + np.outer(v @ axis, axis) * (1.0 - np.cos(angle))
            )
            frames.append(c)
        return np.stack(frames)

    return _make


# CODATA 2018, written out so this file's arithmetic does not lean on
# ffrefine.units, which is part of what the RRMS tests in test_cli_charges.py check.
BOHR_IN_NM = 0.0529177210903


def _write_chelpg_vpot(path, atoms_bohr, grid_bohr, v):
    """Write a ``.vpot`` in the layout a CHELPG run leaves: ``V x y z``, bohr."""
    lines = [f"{len(atoms_bohr):6d} {len(grid_bohr):6d}"]
    lines += [f"   {x: .10E} {y: .10E} {z: .10E}" for x, y, z in atoms_bohr]
    lines += [f"   {p: .10E} {x: .10E} {y: .10E} {z: .10E}" for p, (x, y, z) in zip(v, grid_bohr)]
    path.write_text("\n".join(lines) + "\n")


def _shell_points_bohr(atoms_bohr, n=400, seed=0):
    """Points on shells well outside the molecule, so none sits on an atom."""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    centre = atoms_bohr.mean(axis=0)
    radius = np.linalg.norm(atoms_bohr - centre, axis=1).max() + 4.0
    return centre + directions * radius * rng.uniform(1.0, 1.5, size=(n, 1))


def _inverse_distances_bohr(atoms_bohr, grid_bohr):
    return 1.0 / np.linalg.norm(grid_bohr[:, None, :] - atoms_bohr[None, :, :], axis=-1)


# BUTANE_ESP and FakeEsp: stand-in "ORCA" charges for butane and the fake ESP
# job that writes them, so cmd_charges runs end to end without ORCA.
BUTANE_ESP = np.array(
    [-0.40, 0.09, 0.09, 0.09, 0.20, -0.035, -0.035, 0.20, -0.035, -0.035, -0.40, 0.09, 0.09, 0.09]
)
"""Stand-in "ORCA" charges for butane: sum to zero, equal within each
equivalence group (so averaging leaves them alone), and unlike the fixture's
own charges, so a selected-mode result mixes two genuinely different sets."""


class FakeEsp:
    """What the fake ESP job wrote, and an independent RRMS against it."""

    def __init__(self) -> None:
        self.chelpg = BUTANE_ESP.copy()
        self.calls = 0
        self.atoms_bohr: np.ndarray | None = None
        self.grid_bohr: np.ndarray | None = None
        self.v: np.ndarray | None = None

    def rrms(self, charges) -> float:
        """RRMS from the arrays the fake wrote, in bohr, with no ffrefine code."""
        inv_r = _inverse_distances_bohr(self.atoms_bohr, self.grid_bohr)
        fit = inv_r @ np.asarray(charges, float)
        return float(np.sqrt(((self.v - fit) ** 2).sum() / (self.v**2).sum()))


@pytest.fixture
def fake_esp(monkeypatch):
    """Stand in for the ORCA ESP job, so cmd_charges runs end to end without ORCA.

    Writes an ``esp.chelpg.xyz`` carrying ``FakeEsp.chelpg`` and an
    ``esp.vpot`` whose potential those same charges produce exactly, in the
    layouts ORCA uses. Tests pass ``--charge-geom input`` so xtb is not
    needed either.
    """
    import ffrefine.cli as cli

    fake = FakeEsp()

    def fake_run_esp_job(binaries, structure, coords_nm, workdir, keywords, **kwargs):
        from pathlib import Path

        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        fake.calls += 1
        fake.atoms_bohr = np.asarray(coords_nm, float) / BOHR_IN_NM
        fake.grid_bohr = _shell_points_bohr(fake.atoms_bohr)
        fake.v = _inverse_distances_bohr(fake.atoms_bohr, fake.grid_bohr) @ fake.chelpg
        _write_chelpg_vpot(workdir / "esp.vpot", fake.atoms_bohr, fake.grid_bohr, fake.v)
        lines = [str(len(fake.chelpg)), "# ORCA CHELPG point charges xyz file"]
        for element, (x, y, z), q in zip(structure.elements, coords_nm * 10.0, fake.chelpg):
            lines.append(f"{element:<2s} {x:12.6f} {y:12.6f} {z:12.6f} {q:12.8f}")
        (workdir / "esp.chelpg.xyz").write_text("\n".join(lines) + "\n")
        return workdir / "esp"

    monkeypatch.setattr(cli, "run_esp_job", fake_run_esp_job)
    monkeypatch.setattr(cli, "_orca_version", lambda binaries: "fake ORCA")
    return fake


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
