# ffrefine Milestones 1-3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the torsion-refinement core of `ffrefine` — chemistry file parsers, a GROMACS single-point energy engine with verified torsion zeroing, and a cosine-basis dihedral fitter with model selection — such that the entire fitting pipeline is validated without any QM calculation.

**Architecture:** A pure-Python package in three layers. `io/` parses PDB/GRO/XYZ and GROMACS topologies in-house into explicit object models. `mm/` drives GROMACS as a subprocess for single-point energies, constructs a torsion-zeroed topology, and proves the zeroing was exact by comparing `E_full - E_zeroed` against an independent analytic evaluation of the removed terms. `fit/` solves the dihedral refit as linear least squares in a cosine basis and selects multiplicities by BIC. Every external binary invocation is logged and its output captured.

**Tech Stack:** Python 3.10+, numpy, scipy, matplotlib (runtime deps — no others permitted); pytest + black (dev only); GROMACS 2023+ driven via subprocess.

**Spec:** `ffrefine_spec.md` (repo root, committed at ff699af)

## Global Constraints

Every task inherits this section. Values are copied verbatim from the spec.

- Python 3.10+. Hard runtime dependencies limited to `numpy`, `scipy`, `matplotlib`. No RDKit, no OpenBabel, no MDAnalysis, no AmberTools. Parse all chemistry file formats in-house.
- GROMACS 2023 or newer. Locate binaries with `shutil.which`, allow explicit overrides via CLI or config.
- No internet at runtime.
- **Never modify input files in place.**
- Hartree to kJ/mol: `2625.499639`. GROMACS is kJ/mol, CHARMM `.prm` is kcal/mol (`4.184` kJ/kcal), xtb is Hartree. **Convert at one place only** (`ffrefine/units.py`).
- Wrap angles consistently to `(-180, 180]` for fitting and plotting; the fit basis must be periodic.
- All user-facing atom indices are 1-based; internal arrays are 0-based; convert at the boundary and document where.
- Improper dihedrals (funct 2 and 4) are never modified.
- Never remove `[ pairs ]`. 1-4 interactions stay on in both MM energies.
- Read `fudgeQQ` and `fudgeLJ` from the topology, never assume.
- Log every external command with its full command line; capture stdout and stderr to files under the relevant stage directory. Nonzero exit gives a clear error including the tail of the output.
- Type hints throughout, docstrings on public functions, `black` formatting, `pyproject.toml`.

## Local Environment (verified 2026-09-04)

Binaries live inside WSL; the repo lives on the Windows filesystem. Run everything through WSL:

```bash
wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && <command>"
```

| Tool | Version | Path |
|---|---|---|
| `gmx` | 2025.4 conda-forge, mixed precision, thread_mpi | `/home/marzazil/miniconda3/envs/gromacs/bin/gmx` |
| `xtb` | 6.7.1 | `/home/marzazil/miniconda3/envs/gromacs/bin/xtb` |
| `orca`, `orca_vpot` | 6.1.1 shared build | `/mnt/c/Users/hrach/Downloads/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg/` |
| python | 3.11.14 (numpy 1.26.4, scipy 1.14.0, matplotlib 3.10.9) | `/home/marzazil/miniconda3/envs/gromacs/bin/python` |

Shorthand used throughout: `PY=/home/marzazil/miniconda3/envs/gromacs/bin/python`, `GMX=/home/marzazil/miniconda3/envs/gromacs/bin/gmx`.

Facts already probed — do not re-derive:

- ORCA is **not** on PATH and is a *shared* build: `LD_LIBRARY_PATH` must include its own directory or it will not start.
- `orca_vpot` argument order is `orca_vpot GBWName PName XYZName POTName [DensName]`.
- xtb takes the constraint/scan file via `-I` / `--input FILE`.
- Milestones 4-6 (xtb scan driver, ORCA charges, `.str` parsing) are **out of scope for this plan.**

## Note on spec build-order ordering

Spec section 10 places the round-trip refit test in milestone 2 but "fitting" in milestone 3. The round-trip test cannot run without a fitter. Resolution: the *core* linear cosine fit (Task 11) lands before the round-trip test (Task 13); model selection, weighting, outputs, plots and the validation stage stay in milestone 3. Ordering decision only — no spec requirement is dropped.

## File Structure

```
pyproject.toml                  packaging, deps, black/pytest config
README.md                       worked butane example + physical assumptions
ffrefine/
  __init__.py                   version
  errors.py                     FFRefineError hierarchy
  units.py                      THE single conversion point (sec 9.8)
  geometry.py                   dihedral measurement, angle wrap, Kabsch RMSD
  config.py                     config JSON + CLI merge, binary discovery
  runner.py                     logged subprocess execution (sec 8)
  stages.py                     stage dirs, input hashing, --resume manifests
  io/
    structure.py                PDB/GRO/XYZ read + multi-frame write
    topology.py                 topology parser + object model (sec 4.1)
    topwriter.py                emit .itp/.top, incl. zeroed variant
    prm.py                      CHARMM DIHEDRALS block (kcal/mol)
  mm/
    torsionset.py               collect set T sharing central bond (sec 4.2)
    analytic.py                 analytic proper-dihedral energy
    gromacs.py                  mdp/grompp/mdrun -rerun/energy-by-name
    selfcheck.py                mandatory sec 4.5 consistency check
  fit/
    basis.py                    cosine design matrix
    linear.py                   linear LSQ solve -> k_n, delta_n
    weights.py                  uniform / emax cut / Boltzmann
    selection.py                subset enumeration, BIC ranking
    report.py                   fit statistics, minima, barriers
  plots.py                      matplotlib figures
  cli.py                        argparse subcommands
tests/
  conftest.py                   fixtures, skip-if-no-gmx marker
  test_units.py test_geometry.py test_structure.py test_topology.py
  test_torsionset.py test_analytic.py test_gromacs.py test_selfcheck.py
  test_fit_linear.py test_weights.py test_selection.py test_roundtrip.py
fixtures/butane/
  butane.pdb butane.gro butane.itp   hand-authored CGenFF-style n-butane
```

## Task Index

| # | Task | Milestone |
|---|---|---|
| 1 | Project scaffold and unit conversions | 1 |
| 2 | Geometry primitives | 1 |
| 3 | Structure parsers (PDB/GRO/XYZ) | 1 |
| 4 | Topology object model and parser | 1 |
| 5 | Structure/topology cross-validation | 1 |
| 6 | Butane fixture and PDB/GRO equivalence | 1 |
| 7 | Config, binary discovery, logged runner | 2 |
| 8 | Target torsion set collection | 2 |
| 9 | Analytic proper-dihedral energy | 2 |
| 10 | Topology writer and zeroed topology | 2 |
| 11 | Core cosine-basis linear fit | 2 |
| 12 | GROMACS single-point energy engine | 2 |
| 13 | Mandatory self-consistency check + round-trip refit | 2 |
| 14 | Weighting schemes | 3 |
| 15 | BIC model selection | 3 |
| 16 | Free-phase fit and asymmetry detection | 3 |
| 17 | Fit report statistics | 3 |
| 18 | Output writers (itp, prm, fit.json) | 3 |
| 19 | Plots | 3 |
| 20 | Stage manifests and resume | 3 |
| 21 | Validation stage | 3 |
| 22 | CLI wiring and README | 3 |

---

## MILESTONE 1 — Parsers and topology object model

### Task 1: Project scaffold and unit conversions

**Files:**
- Create: `pyproject.toml`, `ffrefine/__init__.py`, `ffrefine/errors.py`, `ffrefine/units.py`
- Test: `tests/test_units.py`

**Interfaces:**
- Consumes: nothing
- Produces: `HARTREE_TO_KJ: float`, `KCAL_TO_KJ: float`, `hartree_to_kj(x)`, `kj_to_hartree(x)`, `kcal_to_kj(x)`, `kj_to_kcal(x)`, `deg_to_rad(x)`, `rad_to_deg(x)` — all accept float or `np.ndarray` and return the same kind. `errors.FFRefineError`, `errors.ParseError`, `errors.ValidationError`, `errors.ExternalToolError`.

- [ ] **Step 1: Write the failing test** — `tests/test_units.py`

```python
import numpy as np
import pytest
from ffrefine import units


def test_hartree_constant_exact():
    assert units.HARTREE_TO_KJ == 2625.499639


def test_hartree_round_trip():
    x = 0.0371
    assert units.kj_to_hartree(units.hartree_to_kj(x)) == pytest.approx(x, abs=1e-15)


def test_kcal_round_trip():
    x = 12.5
    assert units.kcal_to_kj(units.kj_to_kcal(x)) == pytest.approx(x, abs=1e-12)


def test_kcal_constant_is_thermochemical():
    assert units.KCAL_TO_KJ == 4.184
    assert units.kcal_to_kj(1.0) == pytest.approx(4.184)


def test_degree_round_trip_array():
    d = np.array([-180.0, -90.0, 0.0, 179.9])
    assert np.allclose(units.rad_to_deg(units.deg_to_rad(d)), d)


def test_conversions_accept_arrays():
    a = np.array([1.0, 2.0])
    assert isinstance(units.hartree_to_kj(a), np.ndarray)
    assert units.hartree_to_kj(a)[1] == pytest.approx(2 * 2625.499639)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_units.py -v"`
Expected: FAIL — `ModuleNotFoundError: No module named 'ffrefine'`

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "ffrefine"
version = "0.1.0"
description = "Small-molecule force field parameter refinement (torsions and charges)"
requires-python = ">=3.10"
dependencies = ["numpy", "scipy", "matplotlib"]

[project.scripts]
ffrefine = "ffrefine.cli:main"

[project.optional-dependencies]
dev = ["pytest", "black"]

[tool.setuptools.packages.find]
include = ["ffrefine*"]

[tool.black]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["needs_gmx: requires a working gmx binary"]
```

- [ ] **Step 4: Write `ffrefine/errors.py`**

```python
"""Exception hierarchy for ffrefine."""


class FFRefineError(Exception):
    """Base class for all ffrefine failures."""


class ParseError(FFRefineError):
    """A chemistry input file could not be parsed."""


class ValidationError(FFRefineError):
    """Inputs parsed but are mutually inconsistent."""


class ExternalToolError(FFRefineError):
    """An external binary failed or produced unusable output."""
```

- [ ] **Step 5: Write `ffrefine/units.py`**

Every unit conversion in the package goes through this module (spec section 9.8). No other file may define a conversion factor.

```python
"""The single place where unit conversions live (spec section 9.8).

GROMACS works in kJ/mol, CHARMM .prm files in kcal/mol, xtb and ORCA in
Hartree.  No other module in ffrefine may define a conversion constant.
"""

from __future__ import annotations

import numpy as np

HARTREE_TO_KJ: float = 2625.499639
"""kJ/mol per Hartree."""

KCAL_TO_KJ: float = 4.184
"""kJ per thermochemical kcal."""

Numeric = float | np.ndarray


def hartree_to_kj(x: Numeric) -> Numeric:
    """Convert Hartree to kJ/mol."""
    return x * HARTREE_TO_KJ


def kj_to_hartree(x: Numeric) -> Numeric:
    """Convert kJ/mol to Hartree."""
    return x / HARTREE_TO_KJ


def kcal_to_kj(x: Numeric) -> Numeric:
    """Convert kcal/mol to kJ/mol."""
    return x * KCAL_TO_KJ


def kj_to_kcal(x: Numeric) -> Numeric:
    """Convert kJ/mol to kcal/mol."""
    return x / KCAL_TO_KJ


def deg_to_rad(x: Numeric) -> Numeric:
    """Convert degrees to radians."""
    return np.deg2rad(x)


def rad_to_deg(x: Numeric) -> Numeric:
    """Convert radians to degrees."""
    return np.rad2deg(x)
```

- [ ] **Step 6: Write `ffrefine/__init__.py`**

```python
"""ffrefine: small-molecule force field parameter refinement."""

__version__ = "0.1.0"
```

- [ ] **Step 7: Install into the conda env and run tests**

```bash
wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && /home/marzazil/miniconda3/envs/gromacs/bin/python -m pip install -q pytest black && /home/marzazil/miniconda3/envs/gromacs/bin/python -m pip install -q -e ."
wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && /home/marzazil/miniconda3/envs/gromacs/bin/python -m pytest tests/test_units.py -v"
```
Expected: 6 passed. If `pip install pytest` fails for lack of internet, stop and report — it blocks every later task.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml ffrefine/ tests/test_units.py .gitattributes
git commit -m "feat: project scaffold, error hierarchy, unit conversions

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Geometry primitives

**Files:**
- Create: `ffrefine/geometry.py`
- Test: `tests/test_geometry.py`

**Interfaces:**
- Consumes: nothing (numpy only)
- Produces:
  - `wrap_deg(a: float | np.ndarray) -> float | np.ndarray` — wraps to `(-180, 180]`
  - `dihedral_deg(p0, p1, p2, p3) -> float` — signed IUPAC dihedral, each `p` shape `(3,)`, unit-agnostic, result in `(-180, 180]`
  - `dihedral_deg_frames(coords: np.ndarray, idx: tuple[int, int, int, int]) -> np.ndarray` — `coords` shape `(nframes, natoms, 3)`, `idx` **0-based**, returns `(nframes,)`
  - `kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float` — both `(n, 3)`, RMSD after optimal superposition

- [ ] **Step 1: Write the failing test** — `tests/test_geometry.py`

```python
import numpy as np
import pytest
from ffrefine import geometry as g


def test_wrap_maps_minus_180_to_positive_180():
    assert g.wrap_deg(180.0) == pytest.approx(180.0)
    assert g.wrap_deg(-180.0) == pytest.approx(180.0)


def test_wrap_range():
    a = np.array([-540.0, -181.0, -179.0, 0.0, 181.0, 359.0, 361.0])
    w = g.wrap_deg(a)
    assert np.all(w > -180.0) and np.all(w <= 180.0)
    assert w[2] == pytest.approx(-179.0)
    assert w[5] == pytest.approx(-1.0)
    assert w[6] == pytest.approx(1.0)


def _four_points(phi_deg):
    """Four points whose dihedral is exactly phi_deg, by construction."""
    phi = np.deg2rad(phi_deg)
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.5, 0.0, 0.0])
    p0 = np.array([-0.5, 1.0, 0.0])
    p3 = np.array([2.0, np.cos(phi), np.sin(phi)])
    return p0, p1, p2, p3


@pytest.mark.parametrize("phi", [0.0, 60.0, 90.0, 120.0, 180.0, -60.0, -120.0])
def test_dihedral_recovers_constructed_angle(phi):
    p0, p1, p2, p3 = _four_points(phi)
    assert g.dihedral_deg(p0, p1, p2, p3) == pytest.approx(g.wrap_deg(phi), abs=1e-8)


def test_dihedral_sign_flips_under_mirror():
    p0, p1, p2, p3 = _four_points(60.0)
    m = np.diag([1.0, 1.0, -1.0])
    assert g.dihedral_deg(p0 @ m, p1 @ m, p2 @ m, p3 @ m) == pytest.approx(-60.0, abs=1e-8)


def test_dihedral_frames_shape_and_values():
    frames = np.stack([np.stack(_four_points(p)) for p in (0.0, 60.0, -60.0)])
    out = g.dihedral_deg_frames(frames, (0, 1, 2, 3))
    assert out.shape == (3,)
    assert out == pytest.approx([0.0, 60.0, -60.0], abs=1e-8)


def test_kabsch_zero_for_rotated_translated_copy():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(8, 3))
    t = 0.7
    rot = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    b = a @ rot.T + np.array([3.0, -1.0, 2.0])
    assert g.kabsch_rmsd(a, b) == pytest.approx(0.0, abs=1e-9)


def test_kabsch_detects_real_difference():
    a = np.zeros((4, 3))
    a[1] = [1, 0, 0]
    a[2] = [0, 1, 0]
    a[3] = [0, 0, 1]
    b = a.copy()
    b[3] = [0, 0, 2.0]
    assert g.kabsch_rmsd(a, b) > 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_geometry.py -v"`
Expected: FAIL — `ModuleNotFoundError: No module named 'ffrefine.geometry'`

- [ ] **Step 3: Write `ffrefine/geometry.py`**

Wrap convention is `(-180, 180]`, so `-180` maps to `+180`. Implemented as `180 - ((180 - a) mod 360)`.

```python
"""Geometric primitives: dihedral measurement, angle wrapping, RMSD.

All atom indices taken by this module are 0-based; callers convert from the
user-facing 1-based indices at the CLI boundary.
"""

from __future__ import annotations

import numpy as np

Numeric = float | np.ndarray


def wrap_deg(a: Numeric) -> Numeric:
    """Wrap an angle in degrees to the half-open interval (-180, 180].

    -180 maps to +180 so the convention is consistent across the package
    (spec section 9.6).
    """
    return 180.0 - np.mod(180.0 - np.asarray(a, dtype=float), 360.0)


def dihedral_deg(p0, p1, p2, p3) -> float:
    """Signed IUPAC dihedral angle p0-p1-p2-p3 in degrees, wrapped to (-180, 180].

    Uses the numerically stable atan2 formulation. Unit-agnostic.
    """
    b0 = np.asarray(p0, float) - np.asarray(p1, float)
    b1 = np.asarray(p2, float) - np.asarray(p1, float)
    b2 = np.asarray(p3, float) - np.asarray(p2, float)
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return float(wrap_deg(np.degrees(np.arctan2(y, x))))


def dihedral_deg_frames(coords: np.ndarray, idx: tuple[int, int, int, int]) -> np.ndarray:
    """Measure one dihedral across many frames.

    Args:
        coords: shape (nframes, natoms, 3).
        idx: 0-based atom indices (i, j, k, l).

    Returns:
        Array of shape (nframes,) in degrees, wrapped to (-180, 180].
    """
    coords = np.asarray(coords, float)
    i, j, k, l = idx
    b0 = coords[:, i] - coords[:, j]
    b1 = coords[:, k] - coords[:, j]
    b2 = coords[:, l] - coords[:, k]
    b1n = b1 / np.linalg.norm(b1, axis=1)[:, None]
    v = b0 - (np.sum(b0 * b1n, axis=1)[:, None]) * b1n
    w = b2 - (np.sum(b2 * b1n, axis=1)[:, None]) * b1n
    x = np.sum(v * w, axis=1)
    y = np.sum(np.cross(b1n, v) * w, axis=1)
    return wrap_deg(np.degrees(np.arctan2(y, x)))


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """RMSD between two coordinate sets after optimal rigid superposition.

    Args:
        a, b: shape (n, 3), same atom ordering.

    Returns:
        RMSD in the input distance unit.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    ac = a - a.mean(axis=0)
    bc = b - b.mean(axis=0)
    cov = ac.T @ bc
    u, _, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    diff = ac @ rot.T - bc
    return float(np.sqrt((diff**2).sum() / len(a)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_geometry.py -v"`
Expected: 13 passed (7 parametrised + 6)

- [ ] **Step 5: Commit**

```bash
git add ffrefine/geometry.py tests/test_geometry.py
git commit -m "feat: dihedral measurement, angle wrapping, Kabsch RMSD

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Structure parsers (PDB / GRO / XYZ)

**Files:**
- Create: `ffrefine/io/__init__.py`, `ffrefine/io/structure.py`
- Test: `tests/test_structure.py`

**Interfaces:**
- Consumes: `ffrefine.errors`
- Produces:
  - `@dataclass Structure` with fields `names: list[str]`, `resnames: list[str]`, `resids: list[int]`, `elements: list[str]`, `coords: np.ndarray` (shape `(natoms, 3)`, **always nm internally**), `title: str`, `box: np.ndarray | None` (shape `(3,)`, nm); property `natoms -> int`
  - `read_structure(path: str | Path) -> Structure` — dispatches on suffix `.pdb`, `.gro`, `.xyz`
  - `read_pdb(path)`, `read_gro(path)`, `read_xyz(path)` -> `Structure`
  - `write_multimodel_pdb(path, structure: Structure, frames: np.ndarray) -> None` — `frames` shape `(nframes, natoms, 3)` in nm, emits `MODEL`/`ENDMDL` records using names and resnames from `structure`
  - `guess_element(atom_name: str, mass: float | None = None) -> str`

**Critical unit convention:** PDB and XYZ files are in Angstrom, GRO in nm. `Structure.coords` is **always nm**. Conversion happens in the readers and writers only; the factor `0.1 nm/A` lives in `structure.py` as `ANGSTROM_TO_NM` since it is a length convention, not one of the energy conversions owned by `units.py`.

- [ ] **Step 1: Write the failing test** — `tests/test_structure.py`

```python
import numpy as np
import pytest
from ffrefine.errors import ParseError
from ffrefine.io import structure as st


PDB_TEXT = """\
TITLE     n-butane
ATOM      1  C1  BUT     1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  C2  BUT     1       1.520   0.000   0.000  1.00  0.00           C
ATOM      3  H11 BUT     1      -0.360   1.020   0.000  1.00  0.00           H
END
"""

GRO_TEXT = """\
n-butane
    3
    1BUT     C1    1   0.000   0.000   0.000
    1BUT     C2    2   0.152   0.000   0.000
    1BUT    H11    3  -0.036   0.102   0.000
   2.00000   2.00000   2.00000
"""


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_pdb_parses_names_and_converts_to_nm(tmp_path):
    s = st.read_pdb(_write(tmp_path, "m.pdb", PDB_TEXT))
    assert s.natoms == 3
    assert s.names == ["C1", "C2", "H11"]
    assert s.resnames == ["BUT", "BUT", "BUT"]
    # 1.520 Angstrom -> 0.152 nm
    assert s.coords[1, 0] == pytest.approx(0.152)


def test_pdb_reads_element_column():
    pass  # covered by test_pdb_elements


def test_pdb_elements(tmp_path):
    s = st.read_pdb(_write(tmp_path, "m.pdb", PDB_TEXT))
    assert s.elements == ["C", "C", "H"]


def test_gro_parses_in_nm(tmp_path):
    s = st.read_gro(_write(tmp_path, "m.gro", GRO_TEXT))
    assert s.natoms == 3
    assert s.names == ["C1", "C2", "H11"]
    assert s.coords[1, 0] == pytest.approx(0.152)
    assert s.box == pytest.approx([2.0, 2.0, 2.0])


def test_gro_negative_coordinate_columns(tmp_path):
    """GRO is fixed-width; a negative number must not eat the previous column."""
    s = st.read_gro(_write(tmp_path, "m.gro", GRO_TEXT))
    assert s.coords[2, 0] == pytest.approx(-0.036)
    assert s.coords[2, 1] == pytest.approx(0.102)


def test_pdb_and_gro_agree(tmp_path):
    a = st.read_pdb(_write(tmp_path, "m.pdb", PDB_TEXT))
    b = st.read_gro(_write(tmp_path, "m.gro", GRO_TEXT))
    assert a.names == b.names
    assert np.allclose(a.coords, b.coords, atol=1e-6)


def test_read_structure_dispatches(tmp_path):
    assert st.read_structure(_write(tmp_path, "m.pdb", PDB_TEXT)).natoms == 3
    assert st.read_structure(_write(tmp_path, "m.gro", GRO_TEXT)).natoms == 3


def test_read_structure_rejects_unknown_suffix(tmp_path):
    with pytest.raises(ParseError, match="unsupported"):
        st.read_structure(_write(tmp_path, "m.mol2", "junk"))


def test_gro_atom_count_mismatch_raises(tmp_path):
    bad = GRO_TEXT.replace("    3\n", "    9\n")
    with pytest.raises(ParseError, match="expected 9"):
        st.read_gro(_write(tmp_path, "bad.gro", bad))


def test_multimodel_pdb_round_trip(tmp_path):
    s = st.read_pdb(_write(tmp_path, "m.pdb", PDB_TEXT))
    frames = np.stack([s.coords, s.coords + 0.01, s.coords + 0.02])
    out = tmp_path / "traj.pdb"
    st.write_multimodel_pdb(out, s, frames)
    text = out.read_text()
    assert text.count("MODEL") == 3
    assert text.count("ENDMDL") == 3
    # names must survive so that mdrun -rerun matches the topology
    assert "C2" in text


def test_guess_element_from_name():
    assert st.guess_element("C1") == "C"
    assert st.guess_element("H11") == "H"
    assert st.guess_element("CL2") == "Cl"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_structure.py -v"`
Expected: FAIL — `ModuleNotFoundError: No module named 'ffrefine.io'`

- [ ] **Step 3: Write `ffrefine/io/__init__.py`**

```python
"""In-house parsers for chemistry file formats (no external toolkits)."""
```

- [ ] **Step 4: Write `ffrefine/io/structure.py`**

PDB is strictly column-based, not whitespace-split — atom names like `HG12` and negative coordinates break naive splitting. GRO is likewise fixed-width (`%8.3f` coordinate fields).

```python
"""PDB, GRO and XYZ readers and writers.

Coordinates are held in nanometres throughout the package.  PDB and XYZ are
Angstrom on disk and are converted on read and on write; GRO is already nm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..errors import ParseError

ANGSTROM_TO_NM: float = 0.1
"""Length convention for PDB/XYZ.  Energy conversions live in units.py."""

_TWO_LETTER = {"CL", "BR", "NA", "MG", "ZN", "CA", "FE", "SI", "LI", "AL"}


@dataclass
class Structure:
    """A single molecular geometry with naming metadata.

    Attributes:
        names: atom names, in topology order.
        resnames: residue name per atom.
        resids: residue number per atom.
        elements: element symbol per atom, capitalised (e.g. "C", "Cl").
        coords: shape (natoms, 3) in nanometres.
        title: free-text title from the file.
        box: shape (3,) box vector lengths in nm, or None.
    """

    names: list[str]
    resnames: list[str]
    resids: list[int]
    elements: list[str]
    coords: np.ndarray
    title: str = ""
    box: np.ndarray | None = field(default=None)

    @property
    def natoms(self) -> int:
        """Number of atoms."""
        return len(self.names)


def guess_element(atom_name: str, mass: float | None = None) -> str:
    """Infer an element symbol from an atom name, optionally checked by mass."""
    name = atom_name.strip().lstrip("0123456789")
    if not name:
        return "X"
    two = name[:2].upper()
    if two in _TWO_LETTER:
        return two.capitalize()
    return name[0].upper()


def read_pdb(path: str | Path) -> Structure:
    """Read the first model of a PDB file.  Columns are parsed by position."""
    names: list[str] = []
    resnames: list[str] = []
    resids: list[int] = []
    elements: list[str] = []
    xyz: list[list[float]] = []
    title = ""
    for line in Path(path).read_text().splitlines():
        if line.startswith("TITLE"):
            title = line[10:].strip()
        elif line.startswith(("ATOM", "HETATM")):
            try:
                names.append(line[12:16].strip())
                resnames.append(line[17:20].strip())
                resids.append(int(line[22:26]))
                xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except (ValueError, IndexError) as exc:
                raise ParseError(f"{path}: malformed ATOM record:\n{line}") from exc
            elem = line[76:78].strip() if len(line) >= 78 else ""
            elements.append(elem.capitalize() if elem else guess_element(names[-1]))
        elif line.startswith("ENDMDL"):
            break
    if not names:
        raise ParseError(f"{path}: no ATOM or HETATM records found")
    return Structure(
        names=names,
        resnames=resnames,
        resids=resids,
        elements=elements,
        coords=np.array(xyz, dtype=float) * ANGSTROM_TO_NM,
        title=title,
    )


def read_gro(path: str | Path) -> Structure:
    """Read a GRO file (already in nm).  Fixed-width fields per the format spec."""
    lines = Path(path).read_text().splitlines()
    if len(lines) < 3:
        raise ParseError(f"{path}: file too short to be a GRO")
    title = lines[0].strip()
    try:
        natoms = int(lines[1])
    except ValueError as exc:
        raise ParseError(f"{path}: second line must be the atom count") from exc
    body = lines[2 : 2 + natoms]
    if len(body) != natoms:
        raise ParseError(f"{path}: expected {natoms} atom lines, found {len(body)}")
    names, resnames, resids, elements = [], [], [], []
    xyz = []
    for line in body:
        try:
            resids.append(int(line[0:5]))
            resnames.append(line[5:10].strip())
            names.append(line[10:15].strip())
            xyz.append([float(line[20:28]), float(line[28:36]), float(line[36:44])])
        except (ValueError, IndexError) as exc:
            raise ParseError(f"{path}: malformed GRO atom line:\n{line}") from exc
        elements.append(guess_element(names[-1]))
    box = None
    if len(lines) > 2 + natoms:
        parts = lines[2 + natoms].split()
        if len(parts) >= 3:
            box = np.array([float(p) for p in parts[:3]])
    return Structure(
        names=names,
        resnames=resnames,
        resids=resids,
        elements=elements,
        coords=np.array(xyz, dtype=float),
        title=title,
        box=box,
    )


def read_xyz(path: str | Path) -> Structure:
    """Read an XYZ file (Angstrom) into nm."""
    lines = Path(path).read_text().splitlines()
    if len(lines) < 2:
        raise ParseError(f"{path}: file too short to be an XYZ")
    try:
        natoms = int(lines[0].split()[0])
    except (ValueError, IndexError) as exc:
        raise ParseError(f"{path}: first line must be the atom count") from exc
    body = lines[2 : 2 + natoms]
    if len(body) != natoms:
        raise ParseError(f"{path}: expected {natoms} atom lines, found {len(body)}")
    elements, xyz = [], []
    for line in body:
        parts = line.split()
        if len(parts) < 4:
            raise ParseError(f"{path}: malformed XYZ line:\n{line}")
        elements.append(parts[0].capitalize())
        xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
    names = [f"{e}{i + 1}" for i, e in enumerate(elements)]
    return Structure(
        names=names,
        resnames=["MOL"] * natoms,
        resids=[1] * natoms,
        elements=elements,
        coords=np.array(xyz, dtype=float) * ANGSTROM_TO_NM,
        title=lines[1].strip(),
    )


def read_structure(path: str | Path) -> Structure:
    """Read a structure, dispatching on file suffix."""
    suffix = Path(path).suffix.lower()
    readers = {".pdb": read_pdb, ".gro": read_gro, ".xyz": read_xyz}
    if suffix not in readers:
        raise ParseError(f"{path}: unsupported structure format {suffix!r}; use .pdb, .gro or .xyz")
    return readers[suffix](path)


def write_multimodel_pdb(path: str | Path, structure: Structure, frames: np.ndarray) -> None:
    """Write frames as a multi-MODEL PDB using the naming from ``structure``.

    The atom and residue names must match the topology so that
    ``gmx mdrun -rerun`` accepts the trajectory (spec section 4.4).

    Args:
        path: output file.
        structure: supplies names, resnames, resids and elements.
        frames: shape (nframes, natoms, 3) in nm.
    """
    frames = np.asarray(frames, float)
    if frames.ndim != 3 or frames.shape[1] != structure.natoms:
        raise ValueError(
            f"frames shape {frames.shape} incompatible with {structure.natoms} atoms"
        )
    out = [f"TITLE     {structure.title or 'ffrefine scan'}"]
    for m, frame in enumerate(frames, start=1):
        out.append(f"MODEL     {m:>4d}")
        for i in range(structure.natoms):
            x, y, z = frame[i] / ANGSTROM_TO_NM
            name = structure.names[i]
            # PDB puts one-to-three character names at column 14, four at 13.
            name_field = f"{name:<4s}" if len(name) >= 4 else f" {name:<3s}"
            out.append(
                f"ATOM  {i + 1:>5d} {name_field}{structure.resnames[i]:>3s}  "
                f"{structure.resids[i]:>4d}    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          "
                f"{structure.elements[i]:>2s}"
            )
        out.append("ENDMDL")
    out.append("END")
    Path(path).write_text("\n".join(out) + "\n")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_structure.py -v"`
Expected: all PASS

- [ ] **Step 6: Verify GROMACS accepts the multi-model PDB**

This is the real acceptance criterion for `write_multimodel_pdb`; a file pytest likes but `gmx` rejects is useless.

```bash
wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && /home/marzazil/miniconda3/envs/gromacs/bin/gmx check -f /tmp/traj.pdb 2>&1 | tail -20"
```
Write a throwaway 3-frame PDB to `/tmp/traj.pdb` first. Expected: `gmx check` reports 3 frames without a format error. Record the observed output in the commit message.

- [ ] **Step 7: Commit**

```bash
git add ffrefine/io/ tests/test_structure.py
git commit -m "feat: in-house PDB, GRO and XYZ parsers with multi-model PDB writer

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Topology object model and parser

**Files:**
- Create: `ffrefine/io/topology.py`
- Test: `tests/test_topology.py`

**Interfaces:**
- Consumes: `ffrefine.errors`
- Produces (all atom indices stored **1-based**, matching the topology file itself; documented on the dataclass):
  - `@dataclass Atom`: `nr: int`, `atype: str`, `resnr: int`, `resname: str`, `name: str`, `cgnr: int`, `charge: float`, `mass: float | None`
  - `@dataclass Dihedral`: `ai, aj, ak, al: int`, `funct: int`, `params: list[float]` (empty when the line carries no explicit parameters), `comment: str`
  - `@dataclass Defaults`: `nbfunc: int`, `comb_rule: int`, `gen_pairs: bool`, `fudge_lj: float`, `fudge_qq: float`
  - `@dataclass Topology`: `name: str`, `nrexcl: int`, `atoms: list[Atom]`, `bonds: list[tuple[int, int]]`, `pairs: list[tuple[int, int]]`, `angles: list[tuple[int, int, int]]`, `propers: list[Dihedral]`, `impropers: list[Dihedral]`, `defaults: Defaults | None`, `raw_lines: list[str]`, `source: Path`; property `natoms`; method `bonded_neighbours(nr: int) -> list[int]`
  - `read_topology(path: str | Path, defines: set[str] | None = None) -> Topology`

**Classification rule (spec sections 4.1, 9.4):** `funct` 1, 4 and 9 in `[ dihedrals ]` are *propers*; `funct` 2 is an *improper*. Note funct 4 is ambiguous in GROMACS (periodic improper) — classify funct 4 as **improper** when the `[ dihedrals ]` section header comment or a preceding `; impropers` marker says so, otherwise by position: CHARMM-style topologies list impropers in a second `[ dihedrals ]` block. Concretely: funct 2 -> improper, funct 9 and 1 -> proper, funct 4 -> improper. Impropers are never modified, so misclassifying a proper as an improper is the safe direction.

- [ ] **Step 1: Write the failing test** — `tests/test_topology.py`

```python
import pytest
from ffrefine.errors import ParseError
from ffrefine.io import topology as tp


TOP = """\
[ defaults ]
; nbfunc  comb-rule  gen-pairs  fudgeLJ  fudgeQQ
1         2          yes        1.0      1.0

#include "charmm36.itp"

[ moleculetype ]
; name  nrexcl
BUT     3

[ atoms ]
;  nr  type  resnr  residue  atom  cgnr  charge    mass
    1  CG331     1  BUT      C1       1  -0.270  12.0110
    2  CG321     1  BUT      C2       1  -0.180  12.0110
    3  CG321     1  BUT      C3       2  -0.180  12.0110
    4  CG331     1  BUT      C4       2  -0.270  12.0110

[ bonds ]
    1  2
    2  3
    3  4

[ pairs ]
    1  4

[ angles ]
    1  2  3
    2  3  4

[ dihedrals ]
;   ai  aj  ak  al  funct   phi0     kphi    mult
     1   2   3   4      9   0.000   0.5858      3
     1   2   3   4      9 180.000   0.4184      2

[ dihedrals ]
; impropers
     1   2   3   4      2   0.000  10.0000

#ifdef POSRES
[ position_restraints ]
     1   1  1000  1000  1000
#endif
"""


def _top(tmp_path, text=TOP, name="mol.itp"):
    p = tmp_path / name
    p.write_text(text)
    return tp.read_topology(p)


def test_defaults_parsed(tmp_path):
    t = _top(tmp_path)
    assert t.defaults.fudge_qq == pytest.approx(1.0)
    assert t.defaults.fudge_lj == pytest.approx(1.0)
    assert t.defaults.comb_rule == 2
    assert t.defaults.gen_pairs is True


def test_moleculetype(tmp_path):
    t = _top(tmp_path)
    assert t.name == "BUT"
    assert t.nrexcl == 3


def test_atoms(tmp_path):
    t = _top(tmp_path)
    assert t.natoms == 4
    assert t.atoms[0].atype == "CG331"
    assert t.atoms[0].charge == pytest.approx(-0.270)
    assert t.atoms[0].mass == pytest.approx(12.0110)
    assert t.atoms[2].cgnr == 2


def test_atom_indices_are_one_based(tmp_path):
    t = _top(tmp_path)
    assert [a.nr for a in t.atoms] == [1, 2, 3, 4]


def test_bonds_pairs_angles(tmp_path):
    t = _top(tmp_path)
    assert t.bonds == [(1, 2), (2, 3), (3, 4)]
    assert t.pairs == [(1, 4)]
    assert t.angles == [(1, 2, 3), (2, 3, 4)]


def test_propers_multiline_funct9(tmp_path):
    t = _top(tmp_path)
    assert len(t.propers) == 2
    assert all(d.funct == 9 for d in t.propers)
    assert t.propers[0].params == pytest.approx([0.0, 0.5858, 3])
    assert t.propers[1].params == pytest.approx([180.0, 0.4184, 2])


def test_impropers_not_picked_up_as_propers(tmp_path):
    """Spec section 7.5 and 9.4: funct 2 is an improper and must stay untouched."""
    t = _top(tmp_path)
    assert len(t.impropers) == 1
    assert t.impropers[0].funct == 2
    assert all(d.funct != 2 for d in t.propers)


def test_dihedral_without_explicit_parameters(tmp_path):
    text = TOP.replace(
        "     1   2   3   4      9   0.000   0.5858      3\n"
        "     1   2   3   4      9 180.000   0.4184      2\n",
        "     1   2   3   4      9\n",
    )
    t = _top(tmp_path, text)
    assert len(t.propers) == 1
    assert t.propers[0].params == []


def test_ifdef_block_skipped_by_default(tmp_path):
    t = _top(tmp_path)
    assert not any("position_restraints" in line for line in t.raw_lines if line.startswith("["))


def test_ifdef_block_included_when_defined(tmp_path):
    p = tmp_path / "m.itp"
    p.write_text(TOP)
    t = tp.read_topology(p, defines={"POSRES"})
    assert any("POSRES" not in line for line in t.raw_lines)


def test_include_directive_does_not_crash(tmp_path):
    t = _top(tmp_path)
    assert t.natoms == 4


def test_comments_stripped_from_data(tmp_path):
    t = _top(tmp_path)
    assert t.atoms[0].name == "C1"


def test_bonded_neighbours(tmp_path):
    t = _top(tmp_path)
    assert sorted(t.bonded_neighbours(2)) == [1, 3]
    assert sorted(t.bonded_neighbours(1)) == [2]


def test_missing_atoms_section_raises(tmp_path):
    p = tmp_path / "bad.itp"
    p.write_text("[ moleculetype ]\nBUT 3\n")
    with pytest.raises(ParseError, match="atoms"):
        tp.read_topology(p)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_topology.py -v"`
Expected: FAIL — `ModuleNotFoundError: No module named 'ffrefine.io.topology'`

- [ ] **Step 3: Write `ffrefine/io/topology.py`**

```python
"""GROMACS topology parser and object model (spec section 4.1).

Atom indices are stored exactly as they appear in the file, i.e. **1-based**.
Conversion to 0-based array indices happens where coordinates are indexed,
never here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..errors import ParseError

PROPER_FUNCTS = (1, 9)
"""[ dihedrals ] funct values treated as proper dihedrals."""

IMPROPER_FUNCTS = (2, 4)
"""[ dihedrals ] funct values treated as impropers; never modified (section 9.4)."""


@dataclass
class Atom:
    """One [ atoms ] entry.  ``nr`` is 1-based as written in the file."""

    nr: int
    atype: str
    resnr: int
    resname: str
    name: str
    cgnr: int
    charge: float
    mass: float | None = None


@dataclass
class Dihedral:
    """One [ dihedrals ] line.  Indices are 1-based.

    ``params`` is empty when the line carries no explicit parameters, in which
    case GROMACS looks the term up in [ dihedraltypes ].
    """

    ai: int
    aj: int
    ak: int
    al: int
    funct: int
    params: list[float] = field(default_factory=list)
    comment: str = ""

    @property
    def central_bond(self) -> tuple[int, int]:
        """The (j, k) central pair, unordered-normalised to (min, max)."""
        return (min(self.aj, self.ak), max(self.aj, self.ak))


@dataclass
class Defaults:
    """The [ defaults ] directive (spec section 9.3)."""

    nbfunc: int
    comb_rule: int
    gen_pairs: bool
    fudge_lj: float
    fudge_qq: float


@dataclass
class Topology:
    """Parsed molecule topology."""

    name: str
    nrexcl: int
    atoms: list[Atom]
    bonds: list[tuple[int, int]]
    pairs: list[tuple[int, int]]
    angles: list[tuple[int, int, int]]
    propers: list[Dihedral]
    impropers: list[Dihedral]
    defaults: Defaults | None
    raw_lines: list[str]
    source: Path

    @property
    def natoms(self) -> int:
        """Number of atoms."""
        return len(self.atoms)

    def bonded_neighbours(self, nr: int) -> list[int]:
        """1-based indices bonded to atom ``nr``."""
        out = []
        for a, b in self.bonds:
            if a == nr:
                out.append(b)
            elif b == nr:
                out.append(a)
        return out


def _strip(line: str) -> str:
    """Remove a trailing comment and surrounding whitespace."""
    return line.split(";", 1)[0].strip()


def _comment(line: str) -> str:
    """Return the trailing comment text, without the semicolon."""
    return line.split(";", 1)[1].strip() if ";" in line else ""


def read_topology(path: str | Path, defines: set[str] | None = None) -> Topology:
    """Parse an .itp or .top file into a :class:`Topology`.

    Handles ``#include`` (recorded but not followed -- force field includes are
    resolved by grompp, not by us), ``#ifdef``/``#ifndef``/``#else``/``#endif``
    conditional blocks, and the ``[ defaults ]`` directive.

    Args:
        path: topology file.
        defines: preprocessor symbols considered defined.

    Returns:
        The parsed topology.

    Raises:
        ParseError: on malformed input or a missing [ atoms ] section.
    """
    path = Path(path)
    defines = set(defines or ())
    raw_lines = path.read_text().splitlines()

    section = ""
    dihedral_block = 0
    atoms: list[Atom] = []
    bonds: list[tuple[int, int]] = []
    pairs: list[tuple[int, int]] = []
    angles: list[tuple[int, int, int]] = []
    propers: list[Dihedral] = []
    impropers: list[Dihedral] = []
    defaults: Defaults | None = None
    name, nrexcl = "", 3
    seen_atoms_section = False

    # Conditional-compilation stack: True means "emit lines here".
    emit: list[bool] = [True]

    for lineno, raw in enumerate(raw_lines, start=1):
        stripped = raw.strip()
        if stripped.startswith("#"):
            token = stripped.split()
            directive = token[0]
            if directive == "#ifdef":
                emit.append(emit[-1] and token[1] in defines)
            elif directive == "#ifndef":
                emit.append(emit[-1] and token[1] not in defines)
            elif directive == "#else":
                if len(emit) < 2:
                    raise ParseError(f"{path}:{lineno}: #else without #ifdef")
                emit[-1] = emit[-2] and not emit[-1]
            elif directive == "#endif":
                if len(emit) < 2:
                    raise ParseError(f"{path}:{lineno}: #endif without #ifdef")
                emit.pop()
            elif directive == "#define":
                if emit[-1] and len(token) > 1:
                    defines.add(token[1])
            # #include is recorded in raw_lines and left for grompp.
            continue
        if not emit[-1]:
            continue

        body = _strip(raw)
        if not body:
            continue
        if body.startswith("[") and body.endswith("]"):
            section = body[1:-1].strip().lower()
            if section == "dihedrals":
                dihedral_block += 1
            if section == "atoms":
                seen_atoms_section = True
            continue

        fields = body.split()
        try:
            if section == "defaults":
                gen = fields[2].lower() in ("yes", "y") if len(fields) > 2 else False
                defaults = Defaults(
                    nbfunc=int(fields[0]),
                    comb_rule=int(fields[1]),
                    gen_pairs=gen,
                    fudge_lj=float(fields[3]) if len(fields) > 3 else 1.0,
                    fudge_qq=float(fields[4]) if len(fields) > 4 else 1.0,
                )
            elif section == "moleculetype":
                name = fields[0]
                nrexcl = int(fields[1])
            elif section == "atoms":
                atoms.append(
                    Atom(
                        nr=int(fields[0]),
                        atype=fields[1],
                        resnr=int(fields[2]),
                        resname=fields[3],
                        name=fields[4],
                        cgnr=int(fields[5]),
                        charge=float(fields[6]),
                        mass=float(fields[7]) if len(fields) > 7 else None,
                    )
                )
            elif section == "bonds":
                bonds.append((int(fields[0]), int(fields[1])))
            elif section == "pairs":
                pairs.append((int(fields[0]), int(fields[1])))
            elif section == "angles":
                angles.append((int(fields[0]), int(fields[1]), int(fields[2])))
            elif section == "dihedrals":
                funct = int(fields[4])
                dih = Dihedral(
                    ai=int(fields[0]),
                    aj=int(fields[1]),
                    ak=int(fields[2]),
                    al=int(fields[3]),
                    funct=funct,
                    params=[float(f) for f in fields[5:]],
                    comment=_comment(raw),
                )
                if funct in IMPROPER_FUNCTS:
                    impropers.append(dih)
                elif funct in PROPER_FUNCTS:
                    propers.append(dih)
                else:
                    raise ParseError(
                        f"{path}:{lineno}: unsupported dihedral funct {funct}"
                    )
        except (IndexError, ValueError) as exc:
            raise ParseError(f"{path}:{lineno}: malformed [ {section} ] line:\n{raw}") from exc

    if not seen_atoms_section or not atoms:
        raise ParseError(f"{path}: no [ atoms ] section found")

    return Topology(
        name=name,
        nrexcl=nrexcl,
        atoms=atoms,
        bonds=bonds,
        pairs=pairs,
        angles=angles,
        propers=propers,
        impropers=impropers,
        defaults=defaults,
        raw_lines=raw_lines,
        source=path,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_topology.py -v"`
Expected: all PASS. If `test_ifdef_block_skipped_by_default` fails, the `emit` stack logic is wrong — check that `#else` consults `emit[-2]`.

- [ ] **Step 5: Commit**

```bash
git add ffrefine/io/topology.py tests/test_topology.py
git commit -m "feat: GROMACS topology parser and object model

Impropers (funct 2, 4) are kept separate from propers so they are never
modified, per spec section 9.4.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Structure/topology cross-validation

**Files:**
- Create: `ffrefine/io/validate.py`
- Test: `tests/test_validate.py`

**Interfaces:**
- Consumes: `ffrefine.io.structure.Structure`, `ffrefine.io.topology.Topology`, `ffrefine.errors.ValidationError`
- Produces: `validate_structure_topology(structure: Structure, topology: Topology, strict_names: bool = True) -> list[str]` — returns a list of non-fatal warnings; raises `ValidationError` on a fatal mismatch (spec section 4.1: "Abort with a clear message on mismatch")

- [ ] **Step 1: Write the failing test** — `tests/test_validate.py`

```python
import numpy as np
import pytest
from ffrefine.errors import ValidationError
from ffrefine.io.structure import Structure
from ffrefine.io.topology import Atom, Topology
from ffrefine.io.validate import validate_structure_topology


def _struct(names, elements):
    n = len(names)
    return Structure(
        names=list(names),
        resnames=["BUT"] * n,
        resids=[1] * n,
        elements=list(elements),
        coords=np.zeros((n, 3)),
    )


def _topo(names, masses):
    atoms = [
        Atom(nr=i + 1, atype="CG331", resnr=1, resname="BUT", name=nm, cgnr=1, charge=0.0, mass=m)
        for i, (nm, m) in enumerate(zip(names, masses))
    ]
    return Topology(
        name="BUT", nrexcl=3, atoms=atoms, bonds=[], pairs=[], angles=[],
        propers=[], impropers=[], defaults=None, raw_lines=[], source="x.itp",
    )


def test_matching_inputs_produce_no_warnings():
    s = _struct(["C1", "C2"], ["C", "C"])
    t = _topo(["C1", "C2"], [12.011, 12.011])
    assert validate_structure_topology(s, t) == []


def test_atom_count_mismatch_is_fatal():
    s = _struct(["C1", "C2"], ["C", "C"])
    t = _topo(["C1"], [12.011])
    with pytest.raises(ValidationError, match="atom count"):
        validate_structure_topology(s, t)


def test_element_ordering_mismatch_is_fatal():
    """A carbon in the structure against a hydrogen mass in the topology."""
    s = _struct(["C1", "C2"], ["C", "C"])
    t = _topo(["C1", "C2"], [12.011, 1.008])
    with pytest.raises(ValidationError, match="element"):
        validate_structure_topology(s, t)


def test_name_mismatch_warns_but_does_not_abort():
    s = _struct(["C1", "CX"], ["C", "C"])
    t = _topo(["C1", "C2"], [12.011, 12.011])
    warnings = validate_structure_topology(s, t, strict_names=False)
    assert any("name" in w for w in warnings)


def test_name_mismatch_is_fatal_when_strict():
    s = _struct(["C1", "CX"], ["C", "C"])
    t = _topo(["C1", "C2"], [12.011, 12.011])
    with pytest.raises(ValidationError, match="name"):
        validate_structure_topology(s, t, strict_names=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_validate.py -v"`
Expected: FAIL — no module `ffrefine.io.validate`

- [ ] **Step 3: Write `ffrefine/io/validate.py`**

```python
"""Cross-checks between a structure and its topology (spec section 4.1)."""

from __future__ import annotations

from ..errors import ValidationError
from .structure import Structure, guess_element
from .topology import Topology

_ELEMENT_MASSES = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
    "P": 30.974, "S": 32.06, "Cl": 35.45, "Br": 79.904, "I": 126.904,
}


def _element_from_mass(mass: float) -> str | None:
    """Nearest element by atomic mass, or None if nothing is within 0.5 amu."""
    best, best_diff = None, 0.5
    for sym, ref in _ELEMENT_MASSES.items():
        diff = abs(mass - ref)
        if diff < best_diff:
            best, best_diff = sym, diff
    return best


def validate_structure_topology(
    structure: Structure, topology: Topology, strict_names: bool = True
) -> list[str]:
    """Verify that a structure and topology describe the same molecule.

    Checks atom count, element ordering (via topology masses where available)
    and atom names.  Atom order must be identical between the two files.

    Args:
        structure: parsed coordinates.
        topology: parsed topology.
        strict_names: treat an atom-name mismatch as fatal rather than a warning.

    Returns:
        Non-fatal warning strings.

    Raises:
        ValidationError: on atom-count or element mismatch, or on a name
            mismatch when ``strict_names`` is set.
    """
    warnings: list[str] = []
    if structure.natoms != topology.natoms:
        raise ValidationError(
            f"atom count mismatch: structure has {structure.natoms} atoms, "
            f"topology has {topology.natoms}. The two files must describe the "
            f"same molecule in the same order."
        )

    name_mismatches = []
    for i, (satom_name, atom) in enumerate(zip(structure.names, topology.atoms), start=1):
        if atom.mass is not None:
            expected = _element_from_mass(atom.mass)
            actual = structure.elements[i - 1] or guess_element(satom_name)
            if expected is not None and expected != actual:
                raise ValidationError(
                    f"element mismatch at atom {i} ({satom_name}/{atom.name}): "
                    f"structure says {actual}, topology mass {atom.mass} implies "
                    f"{expected}. Atom order must be identical in both files."
                )
        if satom_name != atom.name:
            name_mismatches.append(f"atom {i}: structure {satom_name!r} vs topology {atom.name!r}")

    if name_mismatches:
        detail = "; ".join(name_mismatches[:10])
        message = f"atom name mismatch ({len(name_mismatches)} atoms): {detail}"
        if strict_names:
            raise ValidationError(
                message + ". Pass --no-strict-names to downgrade this to a warning."
            )
        warnings.append(message)
    return warnings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_validate.py -v"`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add ffrefine/io/validate.py tests/test_validate.py
git commit -m "feat: structure/topology consistency validation

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Butane fixture and PDB/GRO equivalence test

**Files:**
- Create: `fixtures/butane/butane.pdb`, `fixtures/butane/butane.gro`, `fixtures/butane/butane.itp`, `tests/conftest.py`
- Test: `tests/test_fixture_equivalence.py`

**Interfaces:**
- Consumes: Tasks 3, 4, 5
- Produces: pytest fixtures `butane_pdb`, `butane_gro`, `butane_itp` (all `Path`), `gmx_bin` (str or skip); pytest marker `needs_gmx`

**Note on provenance:** No CGenFF distribution is installed, so this topology is **hand-authored** with CGenFF-style atom types and literature-plausible parameters. It must carry a header comment saying exactly that. Its purpose is to exercise the pipeline, not to be a reference parameter set. The 14 atoms are n-butane in an all-trans geometry.

- [ ] **Step 1: Create `fixtures/butane/butane.itp`**

Charges sum to exactly zero. The `[ dihedrals ]` block carries the C1-C2-C3-C4 torsion as two funct-9 lines, which is what the refit will replace.

```
; n-butane, CGenFF-style atom types.
;
; HAND-AUTHORED FIXTURE -- not from a CGenFF distribution.
; Parameters are literature-plausible and exist to exercise the ffrefine
; pipeline. Do not use for production simulation.

[ defaults ]
; nbfunc  comb-rule  gen-pairs  fudgeLJ  fudgeQQ
1         2          yes        1.0      1.0

[ atomtypes ]
;name  at.num  mass      charge  ptype  sigma        epsilon
CG331  6       12.011000  0.000  A      0.363961     0.326352
CG321  6       12.011000  0.000  A      0.358142     0.234304
HGA2   1        1.008000  0.000  A      0.238761     0.100416
HGA3   1        1.008000  0.000  A      0.238761     0.100416

[ moleculetype ]
; name  nrexcl
BUT     3

[ atoms ]
;  nr  type   resnr  residue  atom  cgnr   charge     mass
    1  CG331      1  BUT      C1       1  -0.270   12.011000
    2  HGA3       1  BUT      H11      1   0.090    1.008000
    3  HGA3       1  BUT      H12      1   0.090    1.008000
    4  HGA3       1  BUT      H13      1   0.090    1.008000
    5  CG321      1  BUT      C2       2  -0.180   12.011000
    6  HGA2       1  BUT      H21      2   0.090    1.008000
    7  HGA2       1  BUT      H22      2   0.090    1.008000
    8  CG321      1  BUT      C3       3  -0.180   12.011000
    9  HGA2       1  BUT      H31      3   0.090    1.008000
   10  HGA2       1  BUT      H32      3   0.090    1.008000
   11  CG331      1  BUT      C4       4  -0.270   12.011000
   12  HGA3       1  BUT      H41      4   0.090    1.008000
   13  HGA3       1  BUT      H42      4   0.090    1.008000
   14  HGA3       1  BUT      H43      4   0.090    1.008000

[ bonds ]
;  ai  aj  funct       b0         kb
    1   2      1   0.111100   269449.6
    1   3      1   0.111100   269449.6
    1   4      1   0.111100   269449.6
    1   5      1   0.152800   186188.0
    5   6      1   0.111100   258571.2
    5   7      1   0.111100   258571.2
    5   8      1   0.153000   186188.0
    8   9      1   0.111100   258571.2
    8  10      1   0.111100   258571.2
    8  11      1   0.152800   186188.0
   11  12      1   0.111100   269449.6
   11  13      1   0.111100   269449.6
   11  14      1   0.111100   269449.6

[ pairs ]
;  ai  aj  funct
    2   6      1
    2   7      1
    2   8      1
    3   6      1
    3   7      1
    3   8      1
    4   6      1
    4   7      1
    4   8      1
    1   9      1
    1  10      1
    1  11      1
    6   9      1
    6  10      1
    6  11      1
    7   9      1
    7  10      1
    7  11      1
    5  12      1
    5  13      1
    5  14      1
    9  12      1
    9  13      1
    9  14      1
   10  12      1
   10  13      1
   10  14      1

[ angles ]
;  ai  aj  ak  funct     th0        cth
    2   1   3      1  108.400    297.064
    2   1   4      1  108.400    297.064
    3   1   4      1  108.400    297.064
    2   1   5      1  110.100    289.532
    3   1   5      1  110.100    289.532
    4   1   5      1  110.100    289.532
    1   5   6      1  110.100    289.532
    1   5   7      1  110.100    289.532
    6   5   7      1  109.000    297.064
    1   5   8      1  113.600    488.273
    6   5   8      1  110.100    289.532
    7   5   8      1  110.100    289.532
    5   8   9      1  110.100    289.532
    5   8  10      1  110.100    289.532
    9   8  10      1  109.000    297.064
    5   8  11      1  113.600    488.273
    9   8  11      1  110.100    289.532
   10   8  11      1  110.100    289.532
    8  11  12      1  110.100    289.532
    8  11  13      1  110.100    289.532
    8  11  14      1  110.100    289.532
   12  11  13      1  108.400    297.064
   12  11  14      1  108.400    297.064
   13  11  14      1  108.400    297.064

[ dihedrals ]
;  ai  aj  ak  al  funct     phi0       kphi  mult
;  --- the target torsion C1-C2-C3-C4, central bond 5-8 ---
    1   5   8  11      9    0.000     0.58576     3
    1   5   8  11      9  180.000     0.41840     2
;  --- other torsions about the same central bond 5-8 ---
    1   5   8   9      9    0.000     0.66944     3
    1   5   8  10      9    0.000     0.66944     3
    6   5   8  11      9    0.000     0.66944     3
    7   5   8  11      9    0.000     0.66944     3
    6   5   8   9      9    0.000     0.66944     3
    6   5   8  10      9    0.000     0.66944     3
    7   5   8   9      9    0.000     0.66944     3
    7   5   8  10      9    0.000     0.66944     3
;  --- methyl rotations, different central bonds ---
    2   1   5   6      9    0.000     0.66944     3
    2   1   5   7      9    0.000     0.66944     3
    2   1   5   8      9    0.000     0.66944     3
    3   1   5   6      9    0.000     0.66944     3
    3   1   5   7      9    0.000     0.66944     3
    3   1   5   8      9    0.000     0.66944     3
    4   1   5   6      9    0.000     0.66944     3
    4   1   5   7      9    0.000     0.66944     3
    4   1   5   8      9    0.000     0.66944     3
    5   8  11  12      9    0.000     0.66944     3
    5   8  11  13      9    0.000     0.66944     3
    5   8  11  14      9    0.000     0.66944     3
    9   8  11  12      9    0.000     0.66944     3
    9   8  11  13      9    0.000     0.66944     3
    9   8  11  14      9    0.000     0.66944     3
   10   8  11  12      9    0.000     0.66944     3
   10   8  11  13      9    0.000     0.66944     3
   10   8  11  14      9    0.000     0.66944     3
```

Note the atom numbering: the target torsion C1-C2-C3-C4 is atoms `1, 5, 8, 11`, and its central bond is `(5, 8)`. Nine dihedral entries share that central bond — this is exactly the set `T` from spec section 4.2 that Task 8 must collect.

- [ ] **Step 2: Generate the two structure files**

Build all-trans n-butane and write both formats from one geometry so they are guaranteed to agree. Save as `scripts/make_butane.py`, run once, and commit the outputs.

```python
"""Generate the butane fixture geometry in both PDB and GRO form."""

import numpy as np

from ffrefine.io.structure import Structure, write_multimodel_pdb

NAMES = ["C1", "H11", "H12", "H13", "C2", "H21", "H22",
         "C3", "H31", "H32", "C4", "H41", "H42", "H43"]
ELEMENTS = ["C", "H", "H", "H", "C", "H", "H", "C", "H", "H", "C", "H", "H", "H"]

# All-trans n-butane, Angstrom, generated from ideal internal coordinates.
COORDS_A = np.array([
    [-1.9264,  0.2646,  0.0000], [-2.0324,  0.8974, -0.8842],
    [-2.0324,  0.8974,  0.8842], [-2.7440, -0.4570,  0.0000],
    [-0.5949, -0.4746,  0.0000], [-0.5359, -1.1236,  0.8792],
    [-0.5359, -1.1236, -0.8792], [ 0.5949,  0.4746,  0.0000],
    [ 0.5359,  1.1236,  0.8792], [ 0.5359,  1.1236, -0.8792],
    [ 1.9264, -0.2646,  0.0000], [ 2.7440,  0.4570,  0.0000],
    [ 2.0324, -0.8974,  0.8842], [ 2.0324, -0.8974, -0.8842],
])


def main() -> None:
    s = Structure(
        names=NAMES,
        resnames=["BUT"] * 14,
        resids=[1] * 14,
        elements=ELEMENTS,
        coords=COORDS_A * 0.1,
        title="n-butane all-trans (hand-authored fixture)",
        box=np.array([3.0, 3.0, 3.0]),
    )
    write_multimodel_pdb("fixtures/butane/butane.pdb", s, s.coords[None, :, :])
    lines = [s.title, f"{s.natoms:5d}"]
    for i in range(s.natoms):
        x, y, z = s.coords[i]
        lines.append(
            f"{s.resids[i]:5d}{s.resnames[i]:<5s}{s.names[i]:>5s}{i + 1:5d}"
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
        )
    lines.append(f"{s.box[0]:10.5f}{s.box[1]:10.5f}{s.box[2]:10.5f}")
    with open("fixtures/butane/butane.gro", "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
```

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY scripts/make_butane.py"`

- [ ] **Step 3: Write `tests/conftest.py`**

```python
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
```

- [ ] **Step 4: Write the equivalence test** — `tests/test_fixture_equivalence.py`

This is spec section 7.3.

```python
import numpy as np
import pytest
from ffrefine.io.structure import read_gro, read_pdb
from ffrefine.io.topology import read_topology
from ffrefine.io.validate import validate_structure_topology


def test_butane_topology_parses(butane_itp):
    t = read_topology(butane_itp)
    assert t.natoms == 14
    assert t.name == "BUT"
    assert t.defaults.fudge_qq == pytest.approx(1.0)
    assert len(t.impropers) == 0


def test_butane_charges_sum_to_zero(butane_itp):
    t = read_topology(butane_itp)
    assert sum(a.charge for a in t.atoms) == pytest.approx(0.0, abs=1e-9)


def test_pdb_and_gro_give_identical_coordinates(butane_pdb, butane_gro):
    """Spec section 7.3: the same molecule in both formats must agree."""
    a = read_pdb(butane_pdb)
    b = read_gro(butane_gro)
    assert a.names == b.names
    assert a.resnames == b.resnames
    assert np.allclose(a.coords, b.coords, atol=1e-3)


def test_both_formats_validate_against_topology(butane_pdb, butane_gro, butane_itp):
    t = read_topology(butane_itp)
    assert validate_structure_topology(read_pdb(butane_pdb), t) == []
    assert validate_structure_topology(read_gro(butane_gro), t) == []


def test_target_torsion_is_all_trans(butane_pdb):
    from ffrefine.geometry import dihedral_deg

    s = read_pdb(butane_pdb)
    # C1-C2-C3-C4 is 1-based atoms 1, 5, 8, 11 -> 0-based 0, 4, 7, 10
    phi = dihedral_deg(s.coords[0], s.coords[4], s.coords[7], s.coords[10])
    assert abs(phi) == pytest.approx(180.0, abs=1.0)
```

- [ ] **Step 5: Run the whole suite**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/ -v"`
Expected: all PASS. If `test_target_torsion_is_all_trans` fails, the hand-built coordinates are wrong — fix `COORDS_A`, not the test.

- [ ] **Step 6: Confirm GROMACS accepts the fixture**

The topology has to survive `grompp`, otherwise milestone 2 is built on sand.

```bash
wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && mkdir -p /tmp/butanecheck && cp fixtures/butane/butane.itp /tmp/butanecheck/ && printf '#include \"butane.itp\"\n\n[ system ]\nbutane\n\n[ molecules ]\nBUT 1\n' > /tmp/butanecheck/topol.top && printf 'integrator=md\nnsteps=0\npbc=no\ncutoff-scheme=Verlet\nverlet-buffer-tolerance=-1\nnstlist=0\nrlist=0\ncoulombtype=Cut-off\nrcoulomb=0\nvdwtype=Cut-off\nrvdw=0\n' > /tmp/butanecheck/sp.mdp && cd /tmp/butanecheck && /home/marzazil/miniconda3/envs/gromacs/bin/gmx grompp -f sp.mdp -c /mnt/c/Users/hrach/Projects/ffrefine/fixtures/butane/butane.gro -p topol.top -o sp.tpr -maxwarn 2 2>&1 | tail -25"
```

Expected: a `.tpr` is produced. **Record whether zero cutoffs were accepted** — this answers, ahead of Task 12, whether the spec section 4.4 fallback path (10 nm box, 3 nm cutoffs) is needed on GROMACS 2025.4. Write the answer into the commit message; Task 12 depends on it.

- [ ] **Step 7: Commit**

```bash
git add fixtures/ scripts/make_butane.py tests/conftest.py tests/test_fixture_equivalence.py
git commit -m "test: hand-authored butane fixture, PDB/GRO equivalence

Fixture is hand-written CGenFF-style, not from a CGenFF distribution.
grompp acceptance of zero cutoffs: <RECORD RESULT HERE>

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

**Milestone 1 checkpoint:** all parsers tested, fixture verified against real `grompp`. Stop and report before starting Task 7.

---

## MILESTONE 2 — GROMACS engine, torsion zeroing, verified refit

### Task 7: Config, binary discovery, logged runner

**Files:**
- Create: `ffrefine/config.py`, `ffrefine/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `ffrefine.errors.ExternalToolError`
- Produces:
  - `config.Binaries` dataclass: `gmx: str`, `xtb: str | None`, `orca: str | None`, `orca_vpot: str | None`, `orca_libdir: str | None`
  - `config.discover_binaries(overrides: dict[str, str] | None = None) -> Binaries` — `shutil.which` plus explicit overrides (spec section 1)
  - `config.load_config(path: str | Path | None, cli_overrides: dict) -> dict` — JSON file merged under CLI values (CLI wins)
  - `runner.run_command(argv: list[str], workdir: Path, log_stem: str, extra_env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess` — logs the full command line, writes `<log_stem>.stdout` and `<log_stem>.stderr` into `workdir`, raises `ExternalToolError` with the tail of the output on nonzero exit (spec section 8)

**ORCA detail:** ORCA here is a shared build not on PATH. `discover_binaries` must accept an explicit `orca` path and derive `orca_libdir` as its parent directory; `run_command` prepends that to `LD_LIBRARY_PATH` via `extra_env` when invoking ORCA. This is why `run_command` takes `extra_env`.

- [ ] **Step 1: Write the failing test** — `tests/test_runner.py`

```python
import sys

import pytest
from ffrefine import config, runner
from ffrefine.errors import ExternalToolError


def test_run_command_captures_stdout(tmp_path):
    r = runner.run_command([sys.executable, "-c", "print('hello')"], tmp_path, "probe")
    assert r.returncode == 0
    assert "hello" in (tmp_path / "probe.stdout").read_text()


def test_run_command_logs_full_command_line(tmp_path):
    runner.run_command([sys.executable, "-c", "print(1)"], tmp_path, "probe")
    logged = (tmp_path / "probe.cmd").read_text()
    assert sys.executable in logged
    assert "print(1)" in logged


def test_run_command_raises_with_output_tail(tmp_path):
    code = "import sys; sys.stderr.write('BOOM detail line\\n'); sys.exit(3)"
    with pytest.raises(ExternalToolError, match="BOOM detail line"):
        runner.run_command([sys.executable, "-c", code], tmp_path, "fail")


def test_run_command_check_false_returns_nonzero(tmp_path):
    r = runner.run_command([sys.executable, "-c", "import sys; sys.exit(3)"],
                           tmp_path, "soft", check=False)
    assert r.returncode == 3


def test_run_command_applies_extra_env(tmp_path):
    code = "import os; print(os.environ.get('FFREFINE_PROBE', 'unset'))"
    r = runner.run_command([sys.executable, "-c", code], tmp_path, "env",
                           extra_env={"FFREFINE_PROBE": "set-value"})
    assert "set-value" in r.stdout


def test_discover_binaries_honours_override(tmp_path):
    fake = tmp_path / "mygmx"
    fake.write_text("")
    b = config.discover_binaries({"gmx": str(fake)})
    assert b.gmx == str(fake)


def test_discover_binaries_derives_orca_libdir(tmp_path):
    orca = tmp_path / "orcadir" / "orca"
    orca.parent.mkdir()
    orca.write_text("")
    b = config.discover_binaries({"orca": str(orca)})
    assert b.orca_libdir == str(orca.parent)


def test_load_config_cli_overrides_file(tmp_path):
    cfg_file = tmp_path / "c.json"
    cfg_file.write_text('{"step": 15, "emax": 40.0}')
    merged = config.load_config(cfg_file, {"step": 30})
    assert merged["step"] == 30
    assert merged["emax"] == 40.0


def test_load_config_ignores_none_cli_values(tmp_path):
    cfg_file = tmp_path / "c.json"
    cfg_file.write_text('{"step": 15}')
    merged = config.load_config(cfg_file, {"step": None})
    assert merged["step"] == 15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_runner.py -v"`
Expected: FAIL — no module `ffrefine.runner`

- [ ] **Step 3: Write `ffrefine/runner.py`**

```python
"""Logged execution of external binaries (spec section 8).

Every external command is written to ``<log_stem>.cmd`` with its full argument
vector, and its streams to ``<log_stem>.stdout`` / ``<log_stem>.stderr`` inside
the stage directory.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
from pathlib import Path

from .errors import ExternalToolError

log = logging.getLogger("ffrefine")

TAIL_LINES = 30
"""How many lines of captured output to quote in an error message."""


def run_command(
    argv: list[str],
    workdir: str | Path,
    log_stem: str,
    extra_env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run an external command, logging it and capturing its output.

    Args:
        argv: full argument vector; argv[0] is the executable.
        workdir: directory to run in and to write log files into.
        log_stem: base name for the .cmd/.stdout/.stderr log files.
        extra_env: environment variables layered on top of os.environ.
        check: raise on a nonzero exit status.

    Returns:
        The completed process, with ``stdout`` and ``stderr`` populated.

    Raises:
        ExternalToolError: on nonzero exit when ``check`` is set, or when the
            executable cannot be run at all.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    printable = " ".join(shlex.quote(a) for a in argv)
    log.info("running in %s: %s", workdir, printable)
    (workdir / f"{log_stem}.cmd").write_text(printable + "\n")

    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)

    try:
        proc = subprocess.run(
            argv, cwd=str(workdir), env=env, capture_output=True, text=True
        )
    except OSError as exc:
        raise ExternalToolError(f"could not execute {argv[0]!r}: {exc}") from exc

    (workdir / f"{log_stem}.stdout").write_text(proc.stdout or "")
    (workdir / f"{log_stem}.stderr").write_text(proc.stderr or "")

    if check and proc.returncode != 0:
        combined = ((proc.stdout or "") + (proc.stderr or "")).splitlines()
        tail = "\n".join(combined[-TAIL_LINES:])
        raise ExternalToolError(
            f"command failed with exit code {proc.returncode}:\n"
            f"  {printable}\n"
            f"  workdir: {workdir}\n"
            f"  logs: {log_stem}.stdout, {log_stem}.stderr\n"
            f"--- last {TAIL_LINES} lines of output ---\n{tail}"
        )
    return proc
```

- [ ] **Step 4: Write `ffrefine/config.py`**

```python
"""Binary discovery and configuration merging (spec sections 1 and 3)."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .errors import FFRefineError


@dataclass
class Binaries:
    """Resolved paths to the external programs ffrefine drives.

    ``orca_libdir`` is the directory holding the ORCA shared libraries; ORCA
    shared builds will not start unless it is on LD_LIBRARY_PATH.
    """

    gmx: str | None = None
    xtb: str | None = None
    orca: str | None = None
    orca_vpot: str | None = None
    orca_libdir: str | None = None

    def require(self, name: str) -> str:
        """Return a binary path or raise a clear error naming the override flag."""
        value = getattr(self, name, None)
        if not value:
            raise FFRefineError(
                f"{name} was not found on PATH. Install it, or pass "
                f"--{name.replace('_', '-')}-bin /path/to/{name}, or set "
                f'"binaries": {{"{name}": "..."}} in the config file.'
            )
        return value


def discover_binaries(overrides: dict[str, str] | None = None) -> Binaries:
    """Locate external binaries via shutil.which, honouring explicit overrides.

    Args:
        overrides: mapping of binary name to an explicit path; wins over PATH.

    Returns:
        Resolved binaries.  Missing entries are left as None; call
        :meth:`Binaries.require` at the point of use so the error names the
        program actually needed.
    """
    overrides = dict(overrides or {})
    resolved = Binaries()
    for name in ("gmx", "xtb", "orca", "orca_vpot"):
        explicit = overrides.get(name)
        setattr(resolved, name, explicit or shutil.which(name))
    if resolved.gmx is None and not overrides.get("gmx"):
        resolved.gmx = shutil.which("gmx_mpi")
    if resolved.orca:
        resolved.orca_libdir = str(Path(resolved.orca).resolve().parent)
        if not resolved.orca_vpot:
            candidate = Path(resolved.orca).resolve().parent / "orca_vpot"
            if candidate.exists():
                resolved.orca_vpot = str(candidate)
    return resolved


def orca_env(binaries: Binaries) -> dict[str, str]:
    """Environment additions needed to run an ORCA shared build."""
    if not binaries.orca_libdir:
        return {}
    import os

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    joined = f"{binaries.orca_libdir}:{existing}" if existing else binaries.orca_libdir
    return {"LD_LIBRARY_PATH": joined}


def load_config(path: str | Path | None, cli_overrides: dict) -> dict:
    """Merge a JSON config file with CLI values; CLI wins (spec section 3).

    CLI values that are None mean "not supplied on the command line" and never
    override the file.
    """
    merged: dict = {}
    if path is not None:
        try:
            merged.update(json.loads(Path(path).read_text()))
        except (OSError, json.JSONDecodeError) as exc:
            raise FFRefineError(f"could not read config file {path}: {exc}") from exc
    for key, value in cli_overrides.items():
        if value is not None:
            merged[key] = value
    return merged
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_runner.py -v"`
Expected: 9 passed

- [ ] **Step 6: Confirm the ORCA env plumbing works against the real binary**

```bash
wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && /home/marzazil/miniconda3/envs/gromacs/bin/python -c \"
from ffrefine import config, runner
b = config.discover_binaries({'orca': '/mnt/c/Users/hrach/Downloads/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg/orca'})
print('libdir:', b.orca_libdir)
print('vpot  :', b.orca_vpot)
r = runner.run_command([b.orca], '/tmp/orcaprobe', 'probe', extra_env=config.orca_env(b), check=False)
print('stdout:', r.stdout.strip()[:80])
\""
```
Expected: `libdir` and `vpot` resolved, and ORCA prints its "requires the name of a parameterfile" usage message rather than a shared-library error.

- [ ] **Step 7: Commit**

```bash
git add ffrefine/config.py ffrefine/runner.py tests/test_runner.py
git commit -m "feat: binary discovery, config merging, logged subprocess runner

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Target torsion set collection

**Files:**
- Create: `ffrefine/mm/__init__.py`, `ffrefine/mm/torsionset.py`
- Test: `tests/test_torsionset.py`

**Interfaces:**
- Consumes: `ffrefine.io.topology.Topology`, `Dihedral`
- Produces:
  - `collect_torsion_set(topology: Topology, quad: tuple[int, int, int, int]) -> list[Dihedral]` — every **proper** dihedral whose central pair is `(j, k)` or `(k, j)`; 1-based input. This is set `T` of spec section 4.2.
  - `find_target_entries(topology: Topology, quad) -> list[Dihedral]` — the subset of `T` matching the exact quadruplet in either direction
  - `validate_quad(topology: Topology, quad) -> None` — raises `ValidationError` when indices are out of range, when `j-k` is not a bond, or when `T` is empty

**Spec section 9.1 is the whole point of this task:** zero *all* torsions sharing the central bond, not only the selected quadruplet. In the butane fixture, `T` for `(1,5,8,11)` has **nine** members.

- [ ] **Step 1: Write the failing test** — `tests/test_torsionset.py`

```python
import pytest
from ffrefine.errors import ValidationError
from ffrefine.io.topology import read_topology
from ffrefine.mm.torsionset import collect_torsion_set, find_target_entries, validate_quad


def test_collects_all_nine_sharing_central_bond(butane_itp):
    """Spec section 9.1: every torsion about bond 5-8, not just the target."""
    t = read_topology(butane_itp)
    tset = collect_torsion_set(t, (1, 5, 8, 11))
    assert len(tset) == 9
    for d in tset:
        assert d.central_bond == (5, 8)


def test_reversed_quad_gives_same_set(butane_itp):
    t = read_topology(butane_itp)
    a = collect_torsion_set(t, (1, 5, 8, 11))
    b = collect_torsion_set(t, (11, 8, 5, 1))
    assert {(d.ai, d.aj, d.ak, d.al) for d in a} == {(d.ai, d.aj, d.ak, d.al) for d in b}


def test_target_entries_are_the_two_funct9_lines(butane_itp):
    t = read_topology(butane_itp)
    entries = find_target_entries(t, (1, 5, 8, 11))
    assert len(entries) == 2
    assert {int(e.params[2]) for e in entries} == {2, 3}


def test_target_entries_match_reversed_direction(butane_itp):
    t = read_topology(butane_itp)
    assert len(find_target_entries(t, (11, 8, 5, 1))) == 2


def test_impropers_never_included(butane_itp):
    t = read_topology(butane_itp)
    tset = collect_torsion_set(t, (1, 5, 8, 11))
    assert all(d.funct not in (2, 4) for d in tset)


def test_methyl_torsions_excluded(butane_itp):
    """Torsions about bonds 1-5 and 8-11 must not appear in T for bond 5-8."""
    t = read_topology(butane_itp)
    tset = collect_torsion_set(t, (1, 5, 8, 11))
    assert all(d.central_bond != (1, 5) for d in tset)
    assert all(d.central_bond != (8, 11) for d in tset)


def test_validate_rejects_out_of_range_index(butane_itp):
    t = read_topology(butane_itp)
    with pytest.raises(ValidationError, match="out of range"):
        validate_quad(t, (1, 5, 8, 99))


def test_validate_rejects_non_bonded_central_pair(butane_itp):
    t = read_topology(butane_itp)
    with pytest.raises(ValidationError, match="not bonded"):
        validate_quad(t, (2, 1, 8, 11))


def test_validate_accepts_good_quad(butane_itp):
    t = read_topology(butane_itp)
    validate_quad(t, (1, 5, 8, 11))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_torsionset.py -v"`
Expected: FAIL — no module `ffrefine.mm`

- [ ] **Step 3: Write `ffrefine/mm/__init__.py`**

```python
"""Molecular-mechanics side: GROMACS driving, torsion zeroing, self-checks."""
```

- [ ] **Step 4: Write `ffrefine/mm/torsionset.py`**

```python
"""Selection of the dihedral terms to remove (spec sections 4.2 and 9.1).

All indices in this module are 1-based, matching both the topology file and
the user-facing --dihedral flag.
"""

from __future__ import annotations

from ..errors import ValidationError
from ..io.topology import Dihedral, Topology

Quad = tuple[int, int, int, int]


def _normalise_bond(j: int, k: int) -> tuple[int, int]:
    """Order a bond pair so it can be compared regardless of direction."""
    return (min(j, k), max(j, k))


def validate_quad(topology: Topology, quad: Quad) -> None:
    """Check that a user-supplied dihedral quadruplet is usable.

    Args:
        topology: parsed topology.
        quad: 1-based (i, j, k, l).

    Raises:
        ValidationError: index out of range, central pair not bonded, or no
            proper dihedral in the topology uses that central bond.
    """
    if len(quad) != 4:
        raise ValidationError(f"dihedral must have 4 atom indices, got {len(quad)}")
    for idx in quad:
        if not 1 <= idx <= topology.natoms:
            raise ValidationError(
                f"atom index {idx} out of range; topology has {topology.natoms} atoms "
                f"(indices are 1-based)"
            )
    if len(set(quad)) != 4:
        raise ValidationError(f"dihedral {quad} repeats an atom index")

    j, k = quad[1], quad[2]
    bonds = {_normalise_bond(a, b) for a, b in topology.bonds}
    if _normalise_bond(j, k) not in bonds:
        raise ValidationError(
            f"atoms {j} and {k} are not bonded; the central pair of a proper "
            f"dihedral must be a bond"
        )
    if not collect_torsion_set(topology, quad):
        raise ValidationError(
            f"no proper dihedral in the topology has central bond ({j}, {k}); "
            f"nothing to refit"
        )


def collect_torsion_set(topology: Topology, quad: Quad) -> list[Dihedral]:
    """Every proper dihedral sharing the central bond of ``quad`` (set T).

    Spec section 9.1: all of these are zeroed for the MM reference, not only
    the selected quadruplet.  Impropers are excluded by construction because
    the topology model keeps them in a separate list.

    Args:
        topology: parsed topology.
        quad: 1-based (i, j, k, l); only j and k matter here.

    Returns:
        Matching proper dihedral entries, in topology order.  A funct-9 torsion
        written as several lines contributes one entry per line.
    """
    target = _normalise_bond(quad[1], quad[2])
    return [d for d in topology.propers if d.central_bond == target]


def find_target_entries(topology: Topology, quad: Quad) -> list[Dihedral]:
    """The members of T matching the exact quadruplet, in either direction."""
    forward = tuple(quad)
    reverse = tuple(reversed(quad))
    return [
        d
        for d in collect_torsion_set(topology, quad)
        if (d.ai, d.aj, d.ak, d.al) in (forward, reverse)
    ]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_torsionset.py -v"`
Expected: 9 passed. `test_collects_all_nine_sharing_central_bond` is the one that matters — if it reports 2 instead of 9, `collect_torsion_set` is matching the quadruplet rather than the central bond.

- [ ] **Step 6: Commit**

```bash
git add ffrefine/mm/ tests/test_torsionset.py
git commit -m "feat: collect all proper dihedrals sharing a central bond (set T)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: Analytic proper-dihedral energy

**Files:**
- Create: `ffrefine/mm/analytic.py`
- Test: `tests/test_analytic.py`

**Interfaces:**
- Consumes: `ffrefine.io.topology.Dihedral`, `ffrefine.geometry`
- Produces:
  - `dihedral_energy(params: list[float], phi_deg: np.ndarray) -> np.ndarray` — one funct-9 term, `params = [phi0_deg, k_kj, mult]`, returns kJ/mol
  - `torsion_set_energy(dihedrals: list[Dihedral], coords: np.ndarray) -> np.ndarray` — total energy of a list of dihedral entries over frames; `coords` shape `(nframes, natoms, 3)`, indices inside `Dihedral` are 1-based and converted here
  - `series_energy(k: np.ndarray, delta_deg: np.ndarray, mult: np.ndarray, phi_deg: np.ndarray) -> np.ndarray` — the fitted series `sum k_n [1 + cos(n*phi - delta_n)]`

**The GROMACS funct-9 form** is `V = k [1 + cos(n*phi - phi0)]` with `phi0` in degrees, `k` in kJ/mol, `n` integer. This module is the independent reference that Task 13 checks GROMACS against, so it must be written from the formula and never calibrated against GROMACS output.

- [ ] **Step 1: Write the failing test** — `tests/test_analytic.py`

```python
import numpy as np
import pytest
from ffrefine.io.topology import Dihedral, read_topology
from ffrefine.mm import analytic


def test_single_term_matches_formula():
    phi = np.array([0.0, 60.0, 120.0, 180.0])
    out = analytic.dihedral_energy([0.0, 2.0, 3], phi)
    expected = 2.0 * (1.0 + np.cos(3 * np.deg2rad(phi) - 0.0))
    assert out == pytest.approx(expected)


def test_phase_180_shifts_curve():
    phi = np.array([0.0, 90.0])
    out = analytic.dihedral_energy([180.0, 1.5, 2], phi)
    expected = 1.5 * (1.0 + np.cos(2 * np.deg2rad(phi) - np.pi))
    assert out == pytest.approx(expected)


def test_energy_is_periodic_over_360():
    phi = np.linspace(-180.0, 180.0, 37)
    a = analytic.dihedral_energy([0.0, 1.0, 3], phi)
    b = analytic.dihedral_energy([0.0, 1.0, 3], phi + 360.0)
    assert a == pytest.approx(b)


def test_energy_is_non_negative():
    phi = np.linspace(-180.0, 180.0, 73)
    assert np.all(analytic.dihedral_energy([0.0, 1.0, 3], phi) >= -1e-12)


def test_threefold_minima_at_trans_and_gauche():
    phi = np.linspace(-180.0, 180.0, 361)
    e = analytic.dihedral_energy([0.0, 1.0, 3], phi)
    minima = phi[np.r_[True, e[1:] < e[:-1]][:-1] & np.r_[e[:-1] < e[1:], True][:-1]]
    assert np.any(np.isclose(np.abs(minima), 180.0, atol=2.0))


def test_torsion_set_energy_over_frames(butane_itp):
    from ffrefine.io.structure import read_pdb
    from ffrefine.mm.torsionset import collect_torsion_set

    t = read_topology(butane_itp)
    tset = collect_torsion_set(t, (1, 5, 8, 11))
    coords = np.zeros((2, 14, 3))
    coords[0] = np.arange(42).reshape(14, 3) * 0.1
    coords[1] = coords[0] + 0.01
    e = analytic.torsion_set_energy(tset, coords)
    assert e.shape == (2,)
    assert np.all(np.isfinite(e))


def test_torsion_set_energy_sums_members():
    d1 = Dihedral(1, 2, 3, 4, 9, [0.0, 1.0, 3])
    d2 = Dihedral(1, 2, 3, 4, 9, [180.0, 2.0, 2])
    coords = np.random.default_rng(1).normal(size=(3, 4, 3))
    total = analytic.torsion_set_energy([d1, d2], coords)
    apart = analytic.torsion_set_energy([d1], coords) + analytic.torsion_set_energy([d2], coords)
    assert total == pytest.approx(apart)


def test_torsion_set_energy_ignores_entries_without_parameters():
    """A line with no explicit parameters contributes nothing analytically."""
    d = Dihedral(1, 2, 3, 4, 9, [])
    coords = np.random.default_rng(2).normal(size=(2, 4, 3))
    assert analytic.torsion_set_energy([d], coords) == pytest.approx([0.0, 0.0])


def test_series_energy_matches_sum_of_terms():
    phi = np.linspace(-180.0, 180.0, 25)
    k = np.array([1.0, 0.5])
    delta = np.array([0.0, 180.0])
    mult = np.array([3, 2])
    out = analytic.series_energy(k, delta, mult, phi)
    expected = (analytic.dihedral_energy([0.0, 1.0, 3], phi)
                + analytic.dihedral_energy([180.0, 0.5, 2], phi))
    assert out == pytest.approx(expected)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_analytic.py -v"`
Expected: FAIL — no module `ffrefine.mm.analytic`

- [ ] **Step 3: Write `ffrefine/mm/analytic.py`**

```python
"""Analytic evaluation of GROMACS proper-dihedral terms.

This is the independent reference used by the mandatory self-consistency check
(spec section 4.5).  It is derived from the functional form alone and must
never be tuned to reproduce GROMACS output -- that would defeat the check.

Functional form, GROMACS funct 9 (and funct 1):

    V(phi) = k * [1 + cos(n * phi - phi0)]

with phi0 in degrees, k in kJ/mol and n a non-negative integer.
"""

from __future__ import annotations

import numpy as np

from ..geometry import dihedral_deg_frames
from ..io.topology import Dihedral


def dihedral_energy(params: list[float], phi_deg: np.ndarray) -> np.ndarray:
    """Energy of a single funct-9 dihedral term in kJ/mol.

    Args:
        params: ``[phi0_deg, k_kj_per_mol, multiplicity]``.
        phi_deg: dihedral values in degrees; any range, the form is periodic.

    Returns:
        Energies in kJ/mol, same shape as ``phi_deg``.
    """
    phi0, k, mult = float(params[0]), float(params[1]), int(round(float(params[2])))
    phi = np.deg2rad(np.asarray(phi_deg, dtype=float))
    return k * (1.0 + np.cos(mult * phi - np.deg2rad(phi0)))


def torsion_set_energy(dihedrals: list[Dihedral], coords: np.ndarray) -> np.ndarray:
    """Total energy of a set of dihedral entries across frames.

    Entries with no explicit parameters contribute zero: their parameters live
    in [ dihedraltypes ] and cannot be evaluated without the force field, which
    is exactly why the section 4.5 check compares against GROMACS instead.

    Args:
        dihedrals: entries with **1-based** atom indices.
        coords: shape (nframes, natoms, 3).

    Returns:
        Array of shape (nframes,) in kJ/mol.
    """
    coords = np.asarray(coords, dtype=float)
    total = np.zeros(coords.shape[0], dtype=float)
    for d in dihedrals:
        if not d.params:
            continue
        idx = (d.ai - 1, d.aj - 1, d.ak - 1, d.al - 1)  # 1-based -> 0-based
        phi = dihedral_deg_frames(coords, idx)
        total += dihedral_energy(d.params, phi)
    return total


def series_energy(
    k: np.ndarray, delta_deg: np.ndarray, mult: np.ndarray, phi_deg: np.ndarray
) -> np.ndarray:
    """Evaluate a fitted cosine series sum_n k_n [1 + cos(n*phi - delta_n)].

    Args:
        k: force constants in kJ/mol, shape (nterms,).
        delta_deg: phases in degrees, shape (nterms,).
        mult: multiplicities, shape (nterms,).
        phi_deg: dihedral values in degrees.

    Returns:
        Energies in kJ/mol, same shape as ``phi_deg``.
    """
    phi = np.deg2rad(np.asarray(phi_deg, dtype=float))
    out = np.zeros_like(phi, dtype=float)
    for kn, dn, n in zip(np.atleast_1d(k), np.atleast_1d(delta_deg), np.atleast_1d(mult)):
        out += kn * (1.0 + np.cos(int(n) * phi - np.deg2rad(dn)))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_analytic.py -v"`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add ffrefine/mm/analytic.py tests/test_analytic.py
git commit -m "feat: analytic proper-dihedral energy as independent reference

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: Topology writer and zeroed topology

**Files:**
- Create: `ffrefine/io/topwriter.py`
- Test: `tests/test_topwriter.py`

**Interfaces:**
- Consumes: `ffrefine.io.topology.Topology`, `Dihedral`, `ffrefine.mm.torsionset`
- Produces:
  - `write_zeroed_topology(topology: Topology, quad, out_path: Path) -> list[Dihedral]` — copies the source file line by line, replacing every member of `T` with `i j k l 9 0.000 0.000 1`; returns the removed entries
  - `write_refit_topology(topology: Topology, quad, k, delta_deg, mult, out_path: Path, distribute: bool = False) -> None` — replaces `T` with the fitted series (spec section 4.2)
  - `make_system_top(itp_path: Path, out_path: Path, molname: str, includes: list[str] | None = None) -> None` — minimal `.top` wrapping an `.itp` for grompp

**Line-oriented rewriting, not regeneration.** The source topology may contain `[ atomtypes ]`, `#include` and comments this package does not model. Rewriting only the `[ dihedrals ]` lines that belong to `T` preserves everything else and honours "never modify input files in place" — the output is always a new file.

**Spec section 4.4 caveat:** explicit parameters on a `[ dihedrals ]` line are supposed to suppress `[ dihedraltypes ]` lookup. Task 13 verifies that at runtime rather than trusting it.

- [ ] **Step 1: Write the failing test** — `tests/test_topwriter.py`

```python
import numpy as np
import pytest
from ffrefine.io.topology import read_topology
from ffrefine.io.topwriter import make_system_top, write_refit_topology, write_zeroed_topology
from ffrefine.mm.torsionset import collect_torsion_set


def test_zeroed_topology_removes_all_nine(tmp_path, butane_itp):
    t = read_topology(butane_itp)
    out = tmp_path / "zeroed.itp"
    removed = write_zeroed_topology(t, (1, 5, 8, 11), out)
    assert len(removed) == 9
    z = read_topology(out)
    tset = collect_torsion_set(z, (1, 5, 8, 11))
    assert len(tset) == 9
    for d in tset:
        assert d.params[1] == pytest.approx(0.0)


def test_zeroed_lines_use_funct_9_with_explicit_zeros(tmp_path, butane_itp):
    t = read_topology(butane_itp)
    out = tmp_path / "zeroed.itp"
    write_zeroed_topology(t, (1, 5, 8, 11), out)
    z = read_topology(out)
    for d in collect_torsion_set(z, (1, 5, 8, 11)):
        assert d.funct == 9
        assert len(d.params) == 3
        assert d.params == pytest.approx([0.0, 0.0, 1.0])


def test_input_file_is_not_modified(tmp_path, butane_itp):
    before = butane_itp.read_text()
    t = read_topology(butane_itp)
    write_zeroed_topology(t, (1, 5, 8, 11), tmp_path / "z.itp")
    assert butane_itp.read_text() == before


def test_other_sections_survive_untouched(tmp_path, butane_itp):
    t = read_topology(butane_itp)
    out = tmp_path / "zeroed.itp"
    write_zeroed_topology(t, (1, 5, 8, 11), out)
    z = read_topology(out)
    assert z.natoms == 14
    assert len(z.bonds) == 13
    assert len(z.pairs) == 27
    assert len(z.angles) == 24
    assert "[ atomtypes ]" in out.read_text()


def test_pairs_are_never_removed(tmp_path, butane_itp):
    """Spec section 9.2: 1-4 interactions must survive in both topologies."""
    t = read_topology(butane_itp)
    out = tmp_path / "zeroed.itp"
    write_zeroed_topology(t, (1, 5, 8, 11), out)
    assert read_topology(out).pairs == t.pairs


def test_methyl_torsions_survive_zeroing(tmp_path, butane_itp):
    t = read_topology(butane_itp)
    out = tmp_path / "zeroed.itp"
    write_zeroed_topology(t, (1, 5, 8, 11), out)
    z = read_topology(out)
    methyl = [d for d in z.propers if d.central_bond == (1, 5)]
    assert len(methyl) == 9
    assert all(d.params[1] > 0 for d in methyl)


def test_refit_topology_writes_series_on_target_only(tmp_path, butane_itp):
    t = read_topology(butane_itp)
    out = tmp_path / "refit.itp"
    write_refit_topology(
        t, (1, 5, 8, 11), np.array([1.25, 0.4]), np.array([0.0, 180.0]),
        np.array([3, 2]), out, distribute=False,
    )
    r = read_topology(out)
    target = [d for d in r.propers
              if (d.ai, d.aj, d.ak, d.al) in ((1, 5, 8, 11), (11, 8, 5, 1))]
    assert len(target) == 2
    assert {int(d.params[2]) for d in target} == {2, 3}
    others = [d for d in collect_torsion_set(r, (1, 5, 8, 11))
              if (d.ai, d.aj, d.ak, d.al) not in ((1, 5, 8, 11), (11, 8, 5, 1))]
    assert all(d.params[1] == pytest.approx(0.0) for d in others)


def test_refit_distribute_divides_by_set_size(tmp_path, butane_itp):
    """Spec section 9.7: with --distribute, k_n is divided by N_T."""
    t = read_topology(butane_itp)
    out = tmp_path / "refit.itp"
    write_refit_topology(
        t, (1, 5, 8, 11), np.array([9.0]), np.array([0.0]),
        np.array([3]), out, distribute=True,
    )
    r = read_topology(out)
    tset = collect_torsion_set(r, (1, 5, 8, 11))
    assert len(tset) == 9
    for d in tset:
        assert d.params[1] == pytest.approx(1.0)  # 9.0 / 9 members


def test_make_system_top(tmp_path, butane_itp):
    itp = tmp_path / "mol.itp"
    itp.write_text(butane_itp.read_text())
    top = tmp_path / "topol.top"
    make_system_top(itp, top, "BUT")
    text = top.read_text()
    assert 'mol.itp' in text
    assert "[ molecules ]" in text
    assert "BUT" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_topwriter.py -v"`
Expected: FAIL — no module `ffrefine.io.topwriter`

- [ ] **Step 3: Write `ffrefine/io/topwriter.py`**

The rewriter walks the source lines, tracks the current section and the `#ifdef` state exactly as the parser does, and swaps out only lines that parse to a member of `T`.

```python
"""Emit modified topologies without touching the input (spec section 4.4).

Rewriting is line-oriented: everything the parser does not model -- comments,
[ atomtypes ], #include directives -- is copied through verbatim, and only the
[ dihedrals ] lines belonging to the target set are replaced.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..io.topology import Dihedral, Topology
from ..mm.torsionset import collect_torsion_set

Quad = tuple[int, int, int, int]

ZEROED_TEMPLATE = "{ai:6d}{aj:6d}{ak:6d}{al:6d}{funct:7d}{phi0:12.4f}{k:12.5f}{mult:6d}"


def _entry_key(d: Dihedral) -> tuple[int, int, int, int, int]:
    """Identity of a dihedral line for matching during rewriting."""
    return (d.ai, d.aj, d.ak, d.al, d.funct)


def _format_line(ai, aj, ak, al, funct, phi0, k, mult, comment="") -> str:
    line = ZEROED_TEMPLATE.format(
        ai=ai, aj=aj, ak=ak, al=al, funct=funct, phi0=phi0, k=k, mult=int(mult)
    )
    return f"{line}  ; {comment}" if comment else line


def _rewrite_dihedrals(
    topology: Topology, targets: set, replacement: dict
) -> list[str]:
    """Copy the source file, substituting the dihedral lines named in ``targets``.

    Args:
        topology: parsed source (supplies ``raw_lines``).
        targets: entry keys to replace or drop.
        replacement: entry key -> list of replacement lines (may be empty to drop).

    Returns:
        The output file as a list of lines.
    """
    out: list[str] = []
    section = ""
    emitted: set = set()
    for raw in topology.raw_lines:
        body = raw.split(";", 1)[0].strip()
        if body.startswith("[") and body.endswith("]"):
            section = body[1:-1].strip().lower()
            out.append(raw)
            continue
        if section != "dihedrals" or not body:
            out.append(raw)
            continue
        fields = body.split()
        try:
            key = (int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3]), int(fields[4]))
        except (IndexError, ValueError):
            out.append(raw)
            continue
        if key not in targets:
            out.append(raw)
            continue
        if key not in emitted:
            out.extend(replacement.get(key, []))
            emitted.add(key)
        # Subsequent lines of a multi-line funct-9 term are dropped; the
        # replacement already carries every retained multiplicity.
    return out


def write_zeroed_topology(topology: Topology, quad: Quad, out_path: str | Path) -> list[Dihedral]:
    """Write a copy of the topology with every member of T set to zero.

    Each removed entry becomes a single explicit line ``i j k l 9 0.000 0.000 1``.
    Explicit parameters suppress the [ dihedraltypes ] lookup -- a claim the
    section 4.5 self-check verifies at runtime rather than trusting.

    [ pairs ] is never touched (spec section 9.2).

    Args:
        topology: parsed source topology.
        quad: 1-based target quadruplet.
        out_path: destination; the input is never modified.

    Returns:
        The dihedral entries that were removed, for the self-consistency check.
    """
    tset = collect_torsion_set(topology, quad)
    targets = {_entry_key(d) for d in tset}
    replacement = {
        key: [
            _format_line(key[0], key[1], key[2], key[3], 9, 0.0, 0.0, 1,
                         comment="zeroed by ffrefine")
        ]
        for key in targets
    }
    lines = _rewrite_dihedrals(topology, targets, replacement)
    Path(out_path).write_text("\n".join(lines) + "\n")
    return tset


def write_refit_topology(
    topology: Topology,
    quad: Quad,
    k: np.ndarray,
    delta_deg: np.ndarray,
    mult: np.ndarray,
    out_path: str | Path,
    distribute: bool = False,
) -> None:
    """Write the topology carrying the fitted torsion series (spec section 4.2).

    Args:
        topology: parsed source topology.
        quad: 1-based target quadruplet.
        k: fitted force constants in kJ/mol.
        delta_deg: fitted phases in degrees.
        mult: retained multiplicities.
        out_path: destination.
        distribute: spread k_n over all N_T members as k_n / N_T, giving a
            transferable dihedral type.  Default writes the series onto the
            target quadruplet only and zeroes the rest, which is exact for this
            molecule but molecule-specific.
    """
    tset = collect_torsion_set(topology, quad)
    targets = {_entry_key(d) for d in tset}
    forward, reverse = tuple(quad), tuple(reversed(quad))
    n_t = len(tset)

    replacement: dict = {}
    for key in targets:
        ai, aj, ak, al, _ = key
        is_target = (ai, aj, ak, al) in (forward, reverse)
        if distribute:
            scale, note = 1.0 / n_t, f"ffrefine fit, distributed over {n_t} terms"
        elif is_target:
            scale, note = 1.0, "ffrefine fit"
        else:
            replacement[key] = [
                _format_line(ai, aj, ak, al, 9, 0.0, 0.0, 1,
                             comment="zeroed by ffrefine (fit carried by target quad)")
            ]
            continue
        replacement[key] = [
            _format_line(ai, aj, ak, al, 9, float(d), float(kk) * scale, int(n), comment=note)
            for kk, d, n in zip(np.atleast_1d(k), np.atleast_1d(delta_deg), np.atleast_1d(mult))
        ]

    lines = _rewrite_dihedrals(topology, targets, replacement)
    Path(out_path).write_text("\n".join(lines) + "\n")


def make_system_top(
    itp_path: str | Path,
    out_path: str | Path,
    molname: str,
    includes: list[str] | None = None,
) -> None:
    """Write a minimal .top wrapping an .itp so grompp can process it.

    Args:
        itp_path: molecule .itp; referenced by name, so keep it beside the .top.
        out_path: destination .top.
        molname: the [ moleculetype ] name.
        includes: force field includes to emit before the molecule include.
    """
    lines = ["; generated by ffrefine -- do not edit", ""]
    for inc in includes or []:
        lines.append(f'#include "{inc}"')
    lines.append(f'#include "{Path(itp_path).name}"')
    lines += ["", "[ system ]", f"{molname} in vacuum", "", "[ molecules ]", f"{molname}    1", ""]
    Path(out_path).write_text("\n".join(lines))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_topwriter.py -v"`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add ffrefine/io/topwriter.py tests/test_topwriter.py
git commit -m "feat: zeroed and refit topology writers, line-preserving

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 11: Core cosine-basis linear fit

**Files:**
- Create: `ffrefine/fit/__init__.py`, `ffrefine/fit/basis.py`, `ffrefine/fit/linear.py`
- Test: `tests/test_fit_linear.py`

**Interfaces:**
- Consumes: `ffrefine.geometry.wrap_deg`
- Produces:
  - `basis.design_matrix(phi_deg: np.ndarray, mults: Sequence[int], free_phase: bool = False) -> np.ndarray` — columns `[1, cos(n1 phi), cos(n2 phi), ...]`, plus `sin(n phi)` columns when `free_phase`
  - `@dataclass linear.FitResult`: `mults: np.ndarray`, `k: np.ndarray` (kJ/mol), `delta_deg: np.ndarray`, `constant: float`, `rmse: float`, `max_dev: float`, `residuals: np.ndarray`, `predicted: np.ndarray`, `free_phase: bool`, `n_params: int`
  - `linear.fit_cosine_series(phi_deg, target_kj, mults, weights=None, free_phase=False) -> FitResult`

**Spec section 4.7 is explicit that this is linear least squares, not a nonlinear optimisation.** Fit `V = c + sum_n a_n cos(n phi)`; then `k_n = |a_n|` and `delta_n = 0` when `a_n > 0` else `180`. With `free_phase`, fit `a_n cos + b_n sin` and recover `k_n = hypot(a_n, b_n)`, `delta_n = atan2(-b_n, a_n)` in degrees.

Sign derivation, recorded here so the implementer does not have to re-derive it: the target model is `k[1 + cos(n phi - delta)] = k + k cos(delta) cos(n phi) + k sin(delta) sin(n phi)`. Matching to `a cos(n phi) + b sin(n phi)` gives `a = k cos(delta)`, `b = k sin(delta)`, hence `k = hypot(a, b)` and `delta = atan2(b, a)`.

- [ ] **Step 1: Write the failing test** — `tests/test_fit_linear.py`

```python
import numpy as np
import pytest
from ffrefine.fit.basis import design_matrix
from ffrefine.fit.linear import fit_cosine_series
from ffrefine.mm.analytic import series_energy


def test_design_matrix_shape_and_constant_column():
    phi = np.linspace(-180, 180, 13)
    x = design_matrix(phi, [1, 2, 3])
    assert x.shape == (13, 4)
    assert np.allclose(x[:, 0], 1.0)


def test_design_matrix_free_phase_doubles_columns():
    phi = np.linspace(-180, 180, 13)
    x = design_matrix(phi, [1, 2, 3], free_phase=True)
    assert x.shape == (13, 7)


def test_recovers_single_term_exactly():
    phi = np.linspace(-180, 180, 73, endpoint=False)
    target = series_energy(np.array([2.5]), np.array([0.0]), np.array([3]), phi)
    r = fit_cosine_series(phi, target, [1, 2, 3])
    got = dict(zip(r.mults.tolist(), r.k.tolist()))
    assert got[3] == pytest.approx(2.5, abs=1e-9)
    assert got[1] == pytest.approx(0.0, abs=1e-9)
    assert r.rmse == pytest.approx(0.0, abs=1e-9)


def test_recovers_phase_180_term():
    phi = np.linspace(-180, 180, 73, endpoint=False)
    target = series_energy(np.array([1.75]), np.array([180.0]), np.array([2]), phi)
    r = fit_cosine_series(phi, target, [1, 2, 3])
    i = list(r.mults).index(2)
    assert r.k[i] == pytest.approx(1.75, abs=1e-9)
    assert r.delta_deg[i] == pytest.approx(180.0, abs=1e-6)


def test_recovers_multi_term_series_to_1e_minus_6():
    """This is the numerical core of the spec section 7.1 round-trip test."""
    phi = np.linspace(-180, 180, 73, endpoint=False)
    k_true = np.array([0.6, 1.2, 2.4])
    d_true = np.array([180.0, 0.0, 0.0])
    m_true = np.array([1, 2, 3])
    target = series_energy(k_true, d_true, m_true, phi)
    r = fit_cosine_series(phi, target, [1, 2, 3, 4, 6])
    for kt, dt, mt in zip(k_true, d_true, m_true):
        i = list(r.mults).index(mt)
        assert r.k[i] == pytest.approx(kt, abs=1e-6)
        assert r.delta_deg[i] == pytest.approx(dt, abs=1e-6)
    assert r.rmse == pytest.approx(0.0, abs=1e-9)


def test_unused_multiplicities_come_back_zero():
    phi = np.linspace(-180, 180, 73, endpoint=False)
    target = series_energy(np.array([2.0]), np.array([0.0]), np.array([3]), phi)
    r = fit_cosine_series(phi, target, [1, 2, 3, 4, 6])
    for m in (1, 2, 4, 6):
        assert r.k[list(r.mults).index(m)] == pytest.approx(0.0, abs=1e-9)


def test_k_is_never_negative():
    phi = np.linspace(-180, 180, 73, endpoint=False)
    rng = np.random.default_rng(3)
    target = rng.normal(size=phi.size)
    r = fit_cosine_series(phi, target, [1, 2, 3, 4, 6])
    assert np.all(r.k >= 0.0)


def test_constant_offset_is_absorbed_not_fitted():
    phi = np.linspace(-180, 180, 73, endpoint=False)
    base = series_energy(np.array([2.0]), np.array([0.0]), np.array([3]), phi)
    a = fit_cosine_series(phi, base, [1, 2, 3])
    b = fit_cosine_series(phi, base + 500.0, [1, 2, 3])
    assert a.k == pytest.approx(b.k, abs=1e-9)


def test_free_phase_recovers_arbitrary_phase():
    phi = np.linspace(-180, 180, 73, endpoint=False)
    target = series_energy(np.array([1.4]), np.array([57.0]), np.array([2]), phi)
    r = fit_cosine_series(phi, target, [1, 2, 3], free_phase=True)
    i = list(r.mults).index(2)
    assert r.k[i] == pytest.approx(1.4, abs=1e-8)
    assert r.delta_deg[i] == pytest.approx(57.0, abs=1e-6)


def test_weights_are_honoured():
    phi = np.linspace(-180, 180, 73, endpoint=False)
    target = series_energy(np.array([2.0]), np.array([0.0]), np.array([3]), phi)
    spoiled = target.copy()
    spoiled[0] += 1000.0
    w = np.ones_like(phi)
    w[0] = 0.0
    r = fit_cosine_series(phi, spoiled, [1, 2, 3], weights=w)
    assert r.k[list(r.mults).index(3)] == pytest.approx(2.0, abs=1e-6)


def test_residuals_and_predicted_are_consistent():
    phi = np.linspace(-180, 180, 37, endpoint=False)
    rng = np.random.default_rng(4)
    target = rng.normal(size=phi.size)
    r = fit_cosine_series(phi, target, [1, 2, 3])
    assert r.residuals == pytest.approx(target - r.predicted, abs=1e-12)
    assert r.rmse == pytest.approx(np.sqrt(np.mean(r.residuals**2)), abs=1e-12)
    assert r.max_dev == pytest.approx(np.max(np.abs(r.residuals)), abs=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_fit_linear.py -v"`
Expected: FAIL — no module `ffrefine.fit`

- [ ] **Step 3: Write `ffrefine/fit/__init__.py`**

```python
"""Dihedral series fitting: cosine basis, weighting, model selection."""
```

- [ ] **Step 4: Write `ffrefine/fit/basis.py`**

```python
"""Design matrices for the cosine fitting basis (spec section 4.7)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def design_matrix(
    phi_deg: np.ndarray, mults: Sequence[int], free_phase: bool = False
) -> np.ndarray:
    """Build the linear design matrix for a dihedral cosine series.

    Columns are ``[1, cos(n1 phi), cos(n2 phi), ...]`` and, when ``free_phase``
    is set, the matching ``sin(n phi)`` columns.  The constant column absorbs
    the arbitrary energy offset and is discarded afterwards.

    Args:
        phi_deg: dihedral values in degrees, shape (npoints,).
        mults: multiplicities to include, e.g. (1, 2, 3, 4, 6).
        free_phase: add sine columns so arbitrary phases can be fitted.

    Returns:
        Shape (npoints, 1 + nmults) or (npoints, 1 + 2*nmults).
    """
    phi = np.deg2rad(np.asarray(phi_deg, dtype=float))
    cols = [np.ones_like(phi)]
    cols += [np.cos(n * phi) for n in mults]
    if free_phase:
        cols += [np.sin(n * phi) for n in mults]
    return np.column_stack(cols)
```

- [ ] **Step 5: Write `ffrefine/fit/linear.py`**

```python
"""Linear least-squares dihedral fitting (spec section 4.7).

The refit is exactly linear in a cosine basis, so no nonlinear optimiser is
needed.  Fit ``V = c + sum_n a_n cos(n phi)``, then convert:

    k_n = |a_n|,  delta_n = 0 if a_n > 0 else 180

With ``free_phase`` the basis gains sine terms and the conversion becomes
``k_n = hypot(a_n, b_n)``, ``delta_n = atan2(b_n, a_n)``, since

    k [1 + cos(n phi - delta)] = k + k cos(delta) cos(n phi) + k sin(delta) sin(n phi)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..geometry import wrap_deg
from .basis import design_matrix


@dataclass
class FitResult:
    """Outcome of one cosine-series fit.

    Attributes:
        mults: multiplicities in the model, ascending.
        k: force constants in kJ/mol, same order, all non-negative.
        delta_deg: phases in degrees, wrapped to (-180, 180].
        constant: the absorbed energy offset, discarded by consumers.
        rmse: weighted root-mean-square error in kJ/mol.
        max_dev: largest absolute residual in kJ/mol.
        residuals: target minus predicted, kJ/mol.
        predicted: fitted values at the input angles, kJ/mol.
        free_phase: whether sine terms were used.
        n_params: free parameters, for BIC.
    """

    mults: np.ndarray
    k: np.ndarray
    delta_deg: np.ndarray
    constant: float
    rmse: float
    max_dev: float
    residuals: np.ndarray
    predicted: np.ndarray
    free_phase: bool
    n_params: int


def fit_cosine_series(
    phi_deg: np.ndarray,
    target_kj: np.ndarray,
    mults: Sequence[int],
    weights: np.ndarray | None = None,
    free_phase: bool = False,
) -> FitResult:
    """Fit a dihedral cosine series by weighted linear least squares.

    Args:
        phi_deg: measured dihedral values in degrees.
        target_kj: the target difference curve in kJ/mol.
        mults: multiplicities to include.
        weights: per-point weights; None means uniform.  Zero excludes a point.
        free_phase: allow arbitrary phases rather than 0/180 only.

    Returns:
        The fit, with k in kJ/mol and phases in degrees.
    """
    phi = np.asarray(phi_deg, dtype=float)
    y = np.asarray(target_kj, dtype=float)
    mults = list(mults)
    x = design_matrix(phi, mults, free_phase=free_phase)

    if weights is None:
        w = np.ones_like(y)
    else:
        w = np.asarray(weights, dtype=float)
    sw = np.sqrt(w)
    coeffs, *_ = np.linalg.lstsq(x * sw[:, None], y * sw, rcond=None)

    constant = float(coeffs[0])
    n = len(mults)
    a = coeffs[1 : 1 + n]
    if free_phase:
        b = coeffs[1 + n : 1 + 2 * n]
        k = np.hypot(a, b)
        delta = wrap_deg(np.degrees(np.arctan2(b, a)))
    else:
        k = np.abs(a)
        delta = np.where(a >= 0.0, 0.0, 180.0)

    predicted = x @ coeffs
    residuals = y - predicted
    nz = w > 0
    rmse = float(np.sqrt(np.average(residuals[nz] ** 2, weights=w[nz]))) if nz.any() else 0.0
    max_dev = float(np.max(np.abs(residuals[nz]))) if nz.any() else 0.0

    return FitResult(
        mults=np.array(mults, dtype=int),
        k=np.asarray(k, dtype=float),
        delta_deg=np.asarray(delta, dtype=float),
        constant=constant,
        rmse=rmse,
        max_dev=max_dev,
        residuals=residuals,
        predicted=predicted,
        free_phase=free_phase,
        n_params=int(coeffs.size),
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_fit_linear.py -v"`
Expected: 11 passed. `test_recovers_multi_term_series_to_1e_minus_6` is the gate for spec section 7.1.

- [ ] **Step 7: Commit**

```bash
git add ffrefine/fit/ tests/test_fit_linear.py
git commit -m "feat: linear least-squares cosine dihedral fit

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 12: GROMACS single-point energy engine

**Files:**
- Create: `ffrefine/mm/gromacs.py`
- Test: `tests/test_gromacs.py`

**Interfaces:**
- Consumes: `ffrefine.runner.run_command`, `ffrefine.config.Binaries`, `ffrefine.io.structure.write_multimodel_pdb`
- Produces:
  - `SP_MDP: str` — the vacuum single-point mdp from spec section 4.4, verbatim
  - `FALLBACK_MDP: str` — the 10 nm box / 3 nm cutoff variant
  - `write_mdp(path: Path, fallback: bool = False) -> None`
  - `parse_energy_xvg(path: Path) -> tuple[list[str], np.ndarray]` — returns term names from the `@ s<N> legend` lines and a `(nframes, nterms)` array
  - `@dataclass EnergyResult`: `terms: dict[str, np.ndarray]`, `nframes: int`, `used_fallback: bool`; method `potential() -> np.ndarray`
  - `single_point_energies(binaries, topology_path, structure, frames, workdir, log_stem, nthreads=1, terms=None) -> EnergyResult`

**Term selection by name, never by menu index (spec section 9.9).** `gmx energy` is fed term names on stdin; the parsed `.xvg` legend is then matched back by name. Requested terms: `Potential`, `Bond`, `Angle`, `Proper Dih.`, `Improper Dih.`, `LJ-14`, `Coulomb-14`, `LJ (SR)`, `Coulomb (SR)`. Terms the build does not offer are skipped with a warning rather than aborting.

**Cutoff fallback (spec section 4.4).** Task 6 Step 6 already recorded whether GROMACS 2025.4 accepts zero cutoffs. Implement both paths regardless and detect at runtime: try `SP_MDP`, and on a `grompp` failure mentioning cutoffs or `rlist`, retry with `FALLBACK_MDP` and set `used_fallback`. Log which path was taken.

**Frame count assertion (spec section 4.4):** exactly one energy frame per input frame, checked and raised as `ExternalToolError` otherwise.

- [ ] **Step 1: Write the failing test** — `tests/test_gromacs.py`

The pure-parsing tests run everywhere; the ones driving `gmx` carry `needs_gmx`.

```python
import numpy as np
import pytest
from ffrefine.errors import ExternalToolError
from ffrefine.io.structure import read_pdb
from ffrefine.io.topology import read_topology
from ffrefine.io.topwriter import make_system_top, write_zeroed_topology
from ffrefine.mm import gromacs

XVG = """\
# comment
@    title "Gromacs Energies"
@    xaxis  label "Time (ps)"
@ s0 legend "Bond"
@ s1 legend "Angle"
@ s2 legend "Proper Dih."
@ s3 legend "Potential"
    0.000000    1.500000    2.500000    3.500000    7.500000
    1.000000    1.600000    2.600000    3.600000    7.800000
"""


def test_parse_energy_xvg_names_and_values(tmp_path):
    p = tmp_path / "e.xvg"
    p.write_text(XVG)
    names, data = gromacs.parse_energy_xvg(p)
    assert names == ["Bond", "Angle", "Proper Dih.", "Potential"]
    assert data.shape == (2, 4)
    assert data[0, 3] == pytest.approx(7.5)
    assert data[1, 0] == pytest.approx(1.6)


def test_parse_energy_xvg_ignores_comments(tmp_path):
    p = tmp_path / "e.xvg"
    p.write_text(XVG)
    _, data = gromacs.parse_energy_xvg(p)
    assert np.all(np.isfinite(data))


def test_sp_mdp_is_vacuum_single_point():
    """The mdp must match spec section 4.4 exactly."""
    for line in ("integrator              = md", "nsteps                  = 0",
                 "pbc                     = no", "rcoulomb                = 0",
                 "rvdw                    = 0", "epsilon_r               = 1",
                 "constraints             = none"):
        assert line in gromacs.SP_MDP


def test_fallback_mdp_uses_large_box_cutoffs():
    assert "pbc                     = xyz" in gromacs.FALLBACK_MDP
    assert "rvdw                    = 3" in gromacs.FALLBACK_MDP
    assert "rcoulomb                = 3" in gromacs.FALLBACK_MDP


def test_energy_result_potential_lookup():
    r = gromacs.EnergyResult(
        terms={"Potential": np.array([1.0, 2.0]), "Bond": np.array([0.5, 0.5])},
        nframes=2, used_fallback=False,
    )
    assert r.potential() == pytest.approx([1.0, 2.0])


@pytest.mark.needs_gmx
def test_single_point_runs_on_butane(tmp_path, butane_pdb, butane_itp, gmx_bin):
    from ffrefine.config import Binaries

    s = read_pdb(butane_pdb)
    frames = np.stack([s.coords, s.coords, s.coords])
    itp = tmp_path / "mol.itp"
    itp.write_text(butane_itp.read_text())
    top = tmp_path / "topol.top"
    make_system_top(itp, top, "BUT")

    result = gromacs.single_point_energies(
        Binaries(gmx=gmx_bin), top, s, frames, tmp_path, "full"
    )
    assert result.nframes == 3
    assert "Potential" in result.terms
    # identical frames must give identical energies
    assert result.potential()[0] == pytest.approx(result.potential()[1], abs=1e-6)


@pytest.mark.needs_gmx
def test_frame_count_matches_input(tmp_path, butane_pdb, butane_itp, gmx_bin):
    from ffrefine.config import Binaries

    s = read_pdb(butane_pdb)
    rng = np.random.default_rng(5)
    frames = np.stack([s.coords + rng.normal(scale=0.002, size=s.coords.shape)
                       for _ in range(7)])
    itp = tmp_path / "mol.itp"
    itp.write_text(butane_itp.read_text())
    top = tmp_path / "topol.top"
    make_system_top(itp, top, "BUT")
    r = gromacs.single_point_energies(Binaries(gmx=gmx_bin), top, s, frames, tmp_path, "sp")
    assert r.nframes == 7
    assert r.potential().shape == (7,)


@pytest.mark.needs_gmx
def test_zeroing_lowers_proper_dihedral_energy(tmp_path, butane_pdb, butane_itp, gmx_bin):
    """Sanity: the zeroed topology must have less Proper Dih. energy."""
    from ffrefine.config import Binaries

    s = read_pdb(butane_pdb)
    frames = s.coords[None, :, :]
    t = read_topology(butane_itp)

    full_itp = tmp_path / "full.itp"
    full_itp.write_text(butane_itp.read_text())
    full_top = tmp_path / "full.top"
    make_system_top(full_itp, full_top, "BUT")

    zero_itp = tmp_path / "zeroed.itp"
    write_zeroed_topology(t, (1, 5, 8, 11), zero_itp)
    zero_top = tmp_path / "zeroed.top"
    make_system_top(zero_itp, zero_top, "BUT")

    b = Binaries(gmx=gmx_bin)
    ef = gromacs.single_point_energies(b, full_top, s, frames, tmp_path / "f", "full")
    ez = gromacs.single_point_energies(b, zero_top, s, frames, tmp_path / "z", "zero")
    assert ez.terms["Proper Dih."][0] < ef.terms["Proper Dih."][0]
```

- [ ] **Step 2: Register the marker and run to verify failure**

Add to `tests/conftest.py`:

```python
def pytest_configure(config):
    config.addinivalue_line("markers", "needs_gmx: requires a working gmx binary")
```

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_gromacs.py -v"`
Expected: FAIL — no module `ffrefine.mm.gromacs`

- [ ] **Step 3: Write `ffrefine/mm/gromacs.py`**

```python
"""GROMACS single-point energy engine (spec section 4.4).

Energies are obtained with ``gmx mdrun -rerun`` over a multi-model PDB and
extracted with ``gmx energy``.  Terms are always selected by name: the menu
numbering differs between GROMACS versions (spec section 9.9).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Binaries
from ..errors import ExternalToolError
from ..io.structure import Structure, write_multimodel_pdb
from ..runner import run_command

log = logging.getLogger("ffrefine")

SP_MDP = """\
integrator              = md
nsteps                  = 0
pbc                     = no
cutoff-scheme           = Verlet
verlet-buffer-tolerance = -1
nstlist                 = 0
rlist                   = 0
coulombtype             = Cut-off
rcoulomb                = 0
vdwtype                 = Cut-off
rvdw                    = 0
epsilon_r               = 1
constraints             = none
continuation            = yes
nstcalcenergy           = 1
nstenergy               = 1
"""

FALLBACK_MDP = """\
integrator              = md
nsteps                  = 0
pbc                     = xyz
cutoff-scheme           = Verlet
verlet-buffer-tolerance = -1
nstlist                 = 0
rlist                   = 3.0
coulombtype             = Cut-off
rcoulomb                = 3.0
vdwtype                 = Cut-off
rvdw                    = 3.0
epsilon_r               = 1
constraints             = none
continuation            = yes
nstcalcenergy           = 1
nstenergy               = 1
"""

DEFAULT_TERMS = (
    "Potential",
    "Bond",
    "Angle",
    "Proper Dih.",
    "Improper Dih.",
    "LJ-14",
    "Coulomb-14",
    "LJ (SR)",
    "Coulomb (SR)",
)

FALLBACK_BOX_NM = 10.0
"""Box edge used when the installed GROMACS rejects zero cutoffs."""


@dataclass
class EnergyResult:
    """Decomposed single-point energies, one row per input frame."""

    terms: dict[str, np.ndarray]
    nframes: int
    used_fallback: bool

    def potential(self) -> np.ndarray:
        """Total potential energy in kJ/mol."""
        if "Potential" not in self.terms:
            raise ExternalToolError(
                f"gmx energy did not return a Potential term; got {sorted(self.terms)}"
            )
        return self.terms["Potential"]


def write_mdp(path: str | Path, fallback: bool = False) -> None:
    """Write the single-point mdp, optionally the large-box fallback variant."""
    Path(path).write_text(FALLBACK_MDP if fallback else SP_MDP)


def parse_energy_xvg(path: str | Path) -> tuple[list[str], np.ndarray]:
    """Parse a ``gmx energy`` .xvg into legend names and a data array.

    Args:
        path: the .xvg file.

    Returns:
        ``(names, data)`` where ``names`` has one entry per data column after
        the time column and ``data`` has shape (nframes, nterms).
    """
    names: list[str] = []
    rows: list[list[float]] = []
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("@") and " legend " in stripped and stripped.split()[1].startswith("s"):
            names.append(stripped.split('"')[1])
        elif stripped.startswith(("#", "@")) or not stripped:
            continue
        else:
            rows.append([float(v) for v in stripped.split()])
    if not rows:
        raise ExternalToolError(f"{path}: no energy data rows found")
    data = np.array(rows, dtype=float)[:, 1:]  # drop the time column
    if data.shape[1] != len(names):
        raise ExternalToolError(
            f"{path}: {data.shape[1]} data columns but {len(names)} legends"
        )
    return names, data


def _box_line(fallback: bool) -> np.ndarray | None:
    return np.array([FALLBACK_BOX_NM] * 3) if fallback else None


def single_point_energies(
    binaries: Binaries,
    topology_path: str | Path,
    structure: Structure,
    frames: np.ndarray,
    workdir: str | Path,
    log_stem: str,
    nthreads: int = 1,
    terms: tuple[str, ...] = DEFAULT_TERMS,
) -> EnergyResult:
    """Compute single-point energies for every frame with ``mdrun -rerun``.

    Tries the vacuum zero-cutoff mdp first and falls back to a large box with
    3 nm cutoffs if the installed GROMACS rejects it (spec section 4.4).

    Args:
        binaries: resolved binary paths; ``gmx`` is required.
        topology_path: a .top that grompp can process.
        structure: supplies atom and residue names for the rerun trajectory.
        frames: shape (nframes, natoms, 3) in nm.
        workdir: stage directory for inputs and logs.
        log_stem: prefix for log files.
        nthreads: passed to ``mdrun -nt``.
        terms: energy terms to request, matched by name.

    Returns:
        The decomposed energies.

    Raises:
        ExternalToolError: on any GROMACS failure or a frame-count mismatch.
    """
    gmx = binaries.require("gmx")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frames, dtype=float)
    nframes = frames.shape[0]

    traj = workdir / f"{log_stem}_traj.pdb"
    write_multimodel_pdb(traj, structure, frames)

    used_fallback = False
    for attempt, fallback in enumerate((False, True)):
        mdp = workdir / f"{log_stem}.mdp"
        write_mdp(mdp, fallback=fallback)

        conf = workdir / f"{log_stem}_conf.gro"
        _write_conf_gro(conf, structure, frames[0], box=_box_line(fallback))

        try:
            run_command(
                [gmx, "grompp", "-f", mdp.name, "-c", conf.name, "-p",
                 str(Path(topology_path).resolve()), "-o", f"{log_stem}.tpr", "-maxwarn", "5"],
                workdir, f"{log_stem}_grompp",
            )
        except ExternalToolError as exc:
            text = str(exc).lower()
            cutoff_problem = any(t in text for t in ("rlist", "cut-off", "cutoff", "rvdw", "rcoulomb"))
            if attempt == 0 and cutoff_problem:
                log.warning(
                    "grompp rejected the zero-cutoff vacuum mdp; retrying with the "
                    "%.0f nm box and 3 nm cutoff fallback (spec section 4.4)",
                    FALLBACK_BOX_NM,
                )
                continue
            raise
        used_fallback = fallback
        break

    log.info("GROMACS single point using %s path", "large-box fallback" if used_fallback else "vacuum")

    run_command(
        [gmx, "mdrun", "-s", f"{log_stem}.tpr", "-rerun", traj.name,
         "-e", f"{log_stem}.edr", "-g", f"{log_stem}.log",
         "-o", f"{log_stem}.trr", "-nt", str(nthreads)],
        workdir, f"{log_stem}_mdrun",
    )

    # gmx energy takes term names on stdin; blank line ends the selection.
    selection = "\n".join(terms) + "\n\n"
    proc = run_command(
        [gmx, "energy", "-f", f"{log_stem}.edr", "-o", f"{log_stem}.xvg"],
        workdir, f"{log_stem}_energy", check=False,
    )
    _feed_stdin_energy(gmx, workdir, log_stem, selection)

    names, data = parse_energy_xvg(workdir / f"{log_stem}.xvg")
    if data.shape[0] != nframes:
        raise ExternalToolError(
            f"expected exactly one energy frame per input frame: got "
            f"{data.shape[0]} energy frames for {nframes} input frames"
        )
    missing = [t for t in terms if t not in names]
    if missing:
        log.warning("gmx energy did not provide these terms: %s", ", ".join(missing))

    return EnergyResult(
        terms={name: data[:, i] for i, name in enumerate(names)},
        nframes=nframes,
        used_fallback=used_fallback,
    )


def _feed_stdin_energy(gmx: str, workdir: Path, log_stem: str, selection: str) -> None:
    """Run ``gmx energy`` with the term names piped to stdin."""
    import subprocess

    argv = [gmx, "energy", "-f", f"{log_stem}.edr", "-o", f"{log_stem}.xvg"]
    (workdir / f"{log_stem}_energy.cmd").write_text(" ".join(argv) + f"\n<<< {selection!r}\n")
    proc = subprocess.run(
        argv, cwd=str(workdir), input=selection, capture_output=True, text=True
    )
    (workdir / f"{log_stem}_energy.stdout").write_text(proc.stdout or "")
    (workdir / f"{log_stem}_energy.stderr").write_text(proc.stderr or "")
    if proc.returncode != 0:
        tail = "\n".join(((proc.stdout or "") + (proc.stderr or "")).splitlines()[-30:])
        raise ExternalToolError(f"gmx energy failed:\n{tail}")


def _write_conf_gro(
    path: Path, structure: Structure, coords: np.ndarray, box: np.ndarray | None
) -> None:
    """Write a single-frame GRO for grompp."""
    edge = box if box is not None else np.array([3.0, 3.0, 3.0])
    lines = ["ffrefine single point", f"{structure.natoms:5d}"]
    for i in range(structure.natoms):
        x, y, z = coords[i]
        lines.append(
            f"{structure.resids[i]:5d}{structure.resnames[i]:<5s}"
            f"{structure.names[i]:>5s}{i + 1:5d}{x:8.3f}{y:8.3f}{z:8.3f}"
        )
    lines.append(f"{edge[0]:10.5f}{edge[1]:10.5f}{edge[2]:10.5f}")
    path.write_text("\n".join(lines) + "\n")
```

Note: remove the unused first `run_command` call for `gmx energy` — `_feed_stdin_energy` supersedes it. The implementer should delete that stray `proc = run_command(...)` block; it is listed above only because the surrounding flow is easier to read with the stdin helper called out separately.

- [ ] **Step 4: Run the parsing tests**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_gromacs.py -v -m 'not needs_gmx'"`
Expected: 5 passed

- [ ] **Step 5: Run the GROMACS tests against the real binary**

```bash
wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && PATH=/home/marzazil/miniconda3/envs/gromacs/bin:\$PATH /home/marzazil/miniconda3/envs/gromacs/bin/python -m pytest tests/test_gromacs.py -v"
```
Expected: all 8 pass. If `test_single_point_runs_on_butane` fails inside `grompp`, read `full_grompp.stderr` in the tmp directory — the usual cause is the `[ atomtypes ]` section needing to precede `[ moleculetype ]`, which the fixture already satisfies.

- [ ] **Step 6: Commit**

```bash
git add ffrefine/mm/gromacs.py tests/test_gromacs.py tests/conftest.py
git commit -m "feat: GROMACS single-point energy engine with name-based term selection

Cutoff path used on GROMACS 2025.4: <RECORD vacuum OR fallback>

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 13: Mandatory self-consistency check and round-trip refit

**Files:**
- Create: `ffrefine/mm/selfcheck.py`
- Test: `tests/test_selfcheck.py`, `tests/test_roundtrip.py`

**Interfaces:**
- Consumes: Tasks 8-12
- Produces:
  - `@dataclass SelfCheckResult`: `max_abs_error: float`, `passed: bool`, `gromacs_difference: np.ndarray`, `analytic_difference: np.ndarray`, `phi_deg: np.ndarray`, `tolerance: float`
  - `check_torsion_removal(binaries, topology, quad, structure, frames, workdir, nthreads=1, tolerance=1e-3) -> SelfCheckResult` — raises `ValidationError` on failure, per spec section 4.5 ("abort with a diagnostic, do not proceed to fitting")

**This is the most important test in the package.** It proves the topology edit did what was intended: `E_full(phi) - E_zeroed(phi)` from GROMACS must equal the analytic sum of the removed terms to better than 1e-3 kJ/mol at every point. It also settles empirically whether explicit parameters really do suppress `[ dihedraltypes ]` lookup — the claim spec section 4.4 says not to trust.

- [ ] **Step 1: Write the failing tests** — `tests/test_selfcheck.py`

```python
import numpy as np
import pytest
from ffrefine.errors import ValidationError
from ffrefine.io.structure import read_pdb
from ffrefine.io.topology import read_topology
from ffrefine.mm import selfcheck


def _rotated_frames(structure, n=12):
    """Rotate the terminal methyl about the central bond to sample the torsion."""
    from ffrefine.geometry import dihedral_deg_frames

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


@pytest.mark.needs_gmx
def test_self_consistency_passes_on_butane(tmp_path, butane_pdb, butane_itp, gmx_bin):
    """Spec section 4.5: agreement better than 1e-3 kJ/mol at every point."""
    from ffrefine.config import Binaries

    s = read_pdb(butane_pdb)
    t = read_topology(butane_itp)
    frames = _rotated_frames(s, 12)
    r = selfcheck.check_torsion_removal(
        Binaries(gmx=gmx_bin), t, (1, 5, 8, 11), s, frames, tmp_path
    )
    assert r.passed
    assert r.max_abs_error < 1e-3


@pytest.mark.needs_gmx
def test_self_check_samples_a_real_torsion_range(tmp_path, butane_pdb, butane_itp, gmx_bin):
    from ffrefine.config import Binaries

    s = read_pdb(butane_pdb)
    t = read_topology(butane_itp)
    frames = _rotated_frames(s, 12)
    r = selfcheck.check_torsion_removal(
        Binaries(gmx=gmx_bin), t, (1, 5, 8, 11), s, frames, tmp_path
    )
    assert r.phi_deg.ptp() > 180.0
    assert np.abs(r.analytic_difference).max() > 0.5


@pytest.mark.needs_gmx
def test_self_check_raises_when_tolerance_impossible(tmp_path, butane_pdb, butane_itp, gmx_bin):
    """A deliberately absurd tolerance must abort rather than warn."""
    from ffrefine.config import Binaries

    s = read_pdb(butane_pdb)
    t = read_topology(butane_itp)
    frames = _rotated_frames(s, 6)
    with pytest.raises(ValidationError, match="self-consistency"):
        selfcheck.check_torsion_removal(
            Binaries(gmx=gmx_bin), t, (1, 5, 8, 11), s, frames, tmp_path,
            tolerance=1e-18,
        )
```

- [ ] **Step 2: Write the round-trip test** — `tests/test_roundtrip.py`

This is spec section 7.1. GROMACS supplies the MM energies with the original parameters; those become the pseudo-QM reference; the torsion is zeroed and refitted; the recovered coefficients must match the originals.

```python
import numpy as np
import pytest
from ffrefine.config import Binaries
from ffrefine.fit.linear import fit_cosine_series
from ffrefine.io.structure import read_pdb
from ffrefine.io.topology import read_topology
from ffrefine.io.topwriter import make_system_top, write_zeroed_topology
from ffrefine.mm import gromacs
from ffrefine.mm.torsionset import collect_torsion_set
from tests.test_selfcheck import _rotated_frames


@pytest.mark.needs_gmx
def test_round_trip_refit_recovers_original_parameters(
    tmp_path, butane_pdb, butane_itp, gmx_bin
):
    """Spec section 7.1: refit must recover the original coefficients.

    The whole pipeline except xtb and ORCA is exercised here.
    """
    s = read_pdb(butane_pdb)
    t = read_topology(butane_itp)
    frames = _rotated_frames(s, 36)
    b = Binaries(gmx=gmx_bin)

    full_itp = tmp_path / "full.itp"
    full_itp.write_text(butane_itp.read_text())
    full_top = tmp_path / "full.top"
    make_system_top(full_itp, full_top, "BUT")

    zero_itp = tmp_path / "zeroed.itp"
    removed = write_zeroed_topology(t, (1, 5, 8, 11), zero_itp)
    zero_top = tmp_path / "zeroed.top"
    make_system_top(zero_itp, zero_top, "BUT")

    e_full = gromacs.single_point_energies(b, full_top, s, frames, tmp_path / "f", "full")
    e_zero = gromacs.single_point_energies(b, zero_top, s, frames, tmp_path / "z", "zero")

    from ffrefine.geometry import dihedral_deg_frames

    phi = dihedral_deg_frames(frames, (0, 4, 7, 10))

    # Treat the full-topology MM energies as the "QM" reference (spec 7.1).
    qm = e_full.potential() - e_full.potential().min()
    mm0 = e_zero.potential() - e_zero.potential().min()
    target = qm - mm0

    r = fit_cosine_series(phi, target, [1, 2, 3, 4, 6])

    # The removed set is nine terms about the same central bond; their summed
    # analytic profile is what the fit must reproduce.
    from ffrefine.mm.analytic import torsion_set_energy

    reference = torsion_set_energy(removed, frames)
    reference = reference - reference.min()
    predicted = r.predicted - r.predicted.min()
    assert np.max(np.abs(predicted - reference)) < 1e-6


@pytest.mark.needs_gmx
def test_round_trip_residuals_are_numerically_zero(tmp_path, butane_pdb, butane_itp, gmx_bin):
    s = read_pdb(butane_pdb)
    t = read_topology(butane_itp)
    frames = _rotated_frames(s, 36)
    b = Binaries(gmx=gmx_bin)

    full_itp = tmp_path / "full.itp"
    full_itp.write_text(butane_itp.read_text())
    full_top = tmp_path / "full.top"
    make_system_top(full_itp, full_top, "BUT")
    zero_itp = tmp_path / "zeroed.itp"
    write_zeroed_topology(t, (1, 5, 8, 11), zero_itp)
    zero_top = tmp_path / "zeroed.top"
    make_system_top(zero_itp, zero_top, "BUT")

    e_full = gromacs.single_point_energies(b, full_top, s, frames, tmp_path / "f", "full")
    e_zero = gromacs.single_point_energies(b, zero_top, s, frames, tmp_path / "z", "zero")

    from ffrefine.geometry import dihedral_deg_frames

    phi = dihedral_deg_frames(frames, (0, 4, 7, 10))
    target = (e_full.potential() - e_full.potential().min()) - (
        e_zero.potential() - e_zero.potential().min()
    )
    r = fit_cosine_series(phi, target, [1, 2, 3, 4, 6])
    assert r.rmse < 1e-6
```

- [ ] **Step 3: Run to verify failure**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_selfcheck.py -v"`
Expected: FAIL — no module `ffrefine.mm.selfcheck`

- [ ] **Step 4: Write `ffrefine/mm/selfcheck.py`**

```python
"""The mandatory torsion-removal self-consistency check (spec section 4.5).

E_full(phi) - E_zeroed(phi) computed by GROMACS must equal the analytic sum of
the removed torsion terms, evaluated at the measured dihedral values with the
original parameters, to better than 1e-3 kJ/mol at every point.

Failing this means the topology edit did not do what was intended -- most
often that explicit parameters did not suppress the [ dihedraltypes ] lookup as
assumed.  On failure the pipeline aborts; it never proceeds to fitting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Binaries
from ..errors import ValidationError
from ..geometry import dihedral_deg_frames
from ..io.structure import Structure
from ..io.topology import Topology
from ..io.topwriter import make_system_top, write_zeroed_topology
from .analytic import torsion_set_energy
from .gromacs import single_point_energies

log = logging.getLogger("ffrefine")

DEFAULT_TOLERANCE = 1e-3
"""kJ/mol; spec section 4.5."""


@dataclass
class SelfCheckResult:
    """Outcome of the section 4.5 check."""

    max_abs_error: float
    passed: bool
    gromacs_difference: np.ndarray
    analytic_difference: np.ndarray
    phi_deg: np.ndarray
    tolerance: float


def check_torsion_removal(
    binaries: Binaries,
    topology: Topology,
    quad: tuple[int, int, int, int],
    structure: Structure,
    frames: np.ndarray,
    workdir: str | Path,
    nthreads: int = 1,
    tolerance: float = DEFAULT_TOLERANCE,
) -> SelfCheckResult:
    """Verify that zeroing removed exactly the intended torsion terms.

    Args:
        binaries: resolved binaries; ``gmx`` required.
        topology: the original parsed topology.
        quad: 1-based target quadruplet.
        structure: naming for the rerun trajectory.
        frames: shape (nframes, natoms, 3) in nm.
        workdir: stage directory.
        nthreads: mdrun thread count.
        tolerance: maximum permitted absolute deviation in kJ/mol.

    Returns:
        The check result when it passes.

    Raises:
        ValidationError: when agreement is worse than ``tolerance`` anywhere.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    full_itp = workdir / "full.itp"
    full_itp.write_text(Path(topology.source).read_text())
    full_top = workdir / "full.top"
    make_system_top(full_itp, full_top, topology.name)

    zero_itp = workdir / "zeroed.itp"
    removed = write_zeroed_topology(topology, quad, zero_itp)
    zero_top = workdir / "zeroed.top"
    make_system_top(zero_itp, zero_top, topology.name)

    e_full = single_point_energies(
        binaries, full_top, structure, frames, workdir / "full", "full", nthreads
    )
    e_zero = single_point_energies(
        binaries, zero_top, structure, frames, workdir / "zeroed", "zeroed", nthreads
    )

    gromacs_difference = e_full.potential() - e_zero.potential()
    analytic_difference = torsion_set_energy(removed, frames)
    # The zeroed lines contribute k*(1+cos(...)) with k=0, i.e. nothing, so the
    # analytic sum of the removed terms is the whole expected difference.

    idx = (quad[0] - 1, quad[1] - 1, quad[2] - 1, quad[3] - 1)
    phi = dihedral_deg_frames(np.asarray(frames, float), idx)

    deviation = np.abs(gromacs_difference - analytic_difference)
    max_abs_error = float(deviation.max())
    passed = max_abs_error <= tolerance

    result = SelfCheckResult(
        max_abs_error=max_abs_error,
        passed=passed,
        gromacs_difference=gromacs_difference,
        analytic_difference=analytic_difference,
        phi_deg=phi,
        tolerance=tolerance,
    )

    if not passed:
        worst = int(np.argmax(deviation))
        raise ValidationError(
            "torsion removal self-consistency check FAILED (spec section 4.5).\n"
            f"  maximum |E_full - E_zeroed - analytic| = {max_abs_error:.6g} kJ/mol "
            f"(tolerance {tolerance:g})\n"
            f"  worst frame {worst}, phi = {phi[worst]:.2f} deg:\n"
            f"    GROMACS difference : {gromacs_difference[worst]:.6f} kJ/mol\n"
            f"    analytic reference : {analytic_difference[worst]:.6f} kJ/mol\n"
            f"  {len(removed)} dihedral entries were zeroed about the central bond "
            f"({quad[1]}, {quad[2]}).\n"
            "  The topology edit did not do what was intended -- most likely the "
            "explicit zero parameters did not suppress the [ dihedraltypes ] lookup. "
            f"  Inspect {zero_itp} and the GROMACS logs under {workdir}. "
            "Fitting is aborted."
        )

    log.info(
        "torsion removal self-check passed: max deviation %.3g kJ/mol over %d frames",
        max_abs_error, len(phi),
    )
    return result
```

- [ ] **Step 5: Run both test files against real GROMACS**

```bash
wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && PATH=/home/marzazil/miniconda3/envs/gromacs/bin:\$PATH /home/marzazil/miniconda3/envs/gromacs/bin/python -m pytest tests/test_selfcheck.py tests/test_roundtrip.py -v"
```
Expected: 5 passed.

**If `test_self_consistency_passes_on_butane` fails**, do not weaken the tolerance. It is reporting a real problem, and the diagnostic tells you which: compare `gromacs_difference` against `analytic_difference` frame by frame. A constant offset means a term was dropped that should not have been; a shape mismatch means `[ dihedraltypes ]` is still being consulted for the zeroed lines, in which case the zeroed entries need a different suppression strategy (deleting the lines entirely rather than zeroing them) and spec section 4.4 must be revisited. Report this before changing approach.

- [ ] **Step 6: Run the full suite**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && PATH=/home/marzazil/miniconda3/envs/gromacs/bin:\$PATH /home/marzazil/miniconda3/envs/gromacs/bin/python -m pytest tests/ -v"`
Expected: everything passes, nothing skipped.

- [ ] **Step 7: Commit**

```bash
git add ffrefine/mm/selfcheck.py tests/test_selfcheck.py tests/test_roundtrip.py
git commit -m "feat: mandatory torsion-removal self-check and round-trip refit test

The section 4.5 check confirms explicit zero parameters suppress the
[ dihedraltypes ] lookup on GROMACS 2025.4. Round-trip refit recovers the
original coefficients to better than 1e-6 kJ/mol.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

**Milestone 2 checkpoint:** the entire fitting machinery is now verified without any QM calculation, which is exactly what spec section 10 asks of this milestone. Stop and report before starting Task 14.

---

## MILESTONE 3 — Weighting, model selection, outputs, validation

### Task 14: Weighting schemes

**Files:**
- Create: `ffrefine/fit/weights.py`
- Test: `tests/test_weights.py`

**Interfaces:**
- Consumes: nothing beyond numpy
- Produces:
  - `GAS_CONSTANT_KJ: float = 0.0083144626` (kJ/mol/K)
  - `uniform_weights(qm_kj: np.ndarray, emax: float = 40.0) -> np.ndarray` — 1.0 below `emax` above the QM minimum, 0.0 above it (spec section 4.7)
  - `boltzmann_weights(qm_kj: np.ndarray, temperature: float = 300.0, emax: float | None = 40.0) -> np.ndarray` — `w = exp(-E/RT)`, normalised so the maximum weight is 1

- [ ] **Step 1: Write the failing test** — `tests/test_weights.py`

```python
import numpy as np
import pytest
from ffrefine.fit import weights as w


def test_uniform_keeps_low_energy_points():
    qm = np.array([0.0, 10.0, 39.0])
    assert w.uniform_weights(qm, emax=40.0) == pytest.approx([1.0, 1.0, 1.0])


def test_uniform_discards_above_emax():
    """Spec section 4.7 default: discard points above 40 kJ/mol."""
    qm = np.array([0.0, 39.0, 41.0, 100.0])
    assert w.uniform_weights(qm, emax=40.0) == pytest.approx([1.0, 1.0, 0.0, 0.0])


def test_uniform_is_relative_to_the_minimum():
    qm = np.array([500.0, 510.0, 560.0])
    assert w.uniform_weights(qm, emax=40.0) == pytest.approx([1.0, 1.0, 0.0])


def test_uniform_emax_none_keeps_everything():
    qm = np.array([0.0, 1000.0])
    assert w.uniform_weights(qm, emax=None) == pytest.approx([1.0, 1.0])


def test_boltzmann_decreases_with_energy():
    qm = np.array([0.0, 5.0, 20.0])
    out = w.boltzmann_weights(qm, temperature=300.0, emax=None)
    assert out[0] > out[1] > out[2]
    assert out[0] == pytest.approx(1.0)


def test_boltzmann_matches_formula():
    qm = np.array([0.0, 5.0])
    out = w.boltzmann_weights(qm, temperature=300.0, emax=None)
    expected = np.exp(-5.0 / (w.GAS_CONSTANT_KJ * 300.0))
    assert out[1] == pytest.approx(expected)


def test_boltzmann_still_applies_emax_cut():
    qm = np.array([0.0, 100.0])
    out = w.boltzmann_weights(qm, temperature=300.0, emax=40.0)
    assert out[1] == 0.0


def test_all_weights_non_negative():
    qm = np.linspace(0, 80, 17)
    assert np.all(w.uniform_weights(qm) >= 0)
    assert np.all(w.boltzmann_weights(qm) >= 0)
```

- [ ] **Step 2: Run to verify failure**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_weights.py -v"`
Expected: FAIL — no module `ffrefine.fit.weights`

- [ ] **Step 3: Write `ffrefine/fit/weights.py`**

```python
"""Point weighting for the dihedral fit (spec section 4.7)."""

from __future__ import annotations

import numpy as np

GAS_CONSTANT_KJ: float = 0.0083144626
"""Molar gas constant in kJ/mol/K."""

DEFAULT_EMAX: float = 40.0
"""kJ/mol above the QM minimum; points higher than this are discarded."""

DEFAULT_TEMPERATURE: float = 300.0
"""Kelvin, for Boltzmann weighting."""


def _relative(qm_kj: np.ndarray) -> np.ndarray:
    qm = np.asarray(qm_kj, dtype=float)
    return qm - qm.min()


def uniform_weights(qm_kj: np.ndarray, emax: float | None = DEFAULT_EMAX) -> np.ndarray:
    """Uniform weights, discarding points high above the QM minimum.

    Args:
        qm_kj: QM energies in kJ/mol; only differences matter.
        emax: cut-off in kJ/mol above the minimum, or None to keep everything.

    Returns:
        Weights of 1.0 (kept) or 0.0 (discarded).
    """
    rel = _relative(qm_kj)
    if emax is None:
        return np.ones_like(rel)
    return (rel <= emax).astype(float)


def boltzmann_weights(
    qm_kj: np.ndarray,
    temperature: float = DEFAULT_TEMPERATURE,
    emax: float | None = DEFAULT_EMAX,
) -> np.ndarray:
    """Boltzmann weights ``exp(-E/RT)`` relative to the QM minimum.

    Normalised so the lowest-energy point has weight 1.  The ``emax`` cut is
    still applied on top, so a Boltzmann fit never silently includes points the
    uniform scheme would have thrown away.

    Args:
        qm_kj: QM energies in kJ/mol.
        temperature: in Kelvin.
        emax: cut-off in kJ/mol above the minimum, or None.

    Returns:
        Weights in [0, 1].
    """
    rel = _relative(qm_kj)
    out = np.exp(-rel / (GAS_CONSTANT_KJ * temperature))
    if emax is not None:
        out = np.where(rel <= emax, out, 0.0)
    return out
```

- [ ] **Step 4: Run tests, then commit**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_weights.py -v"`
Expected: 8 passed

```bash
git add ffrefine/fit/weights.py tests/test_weights.py
git commit -m "feat: uniform and Boltzmann fit weighting with emax cut

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 15: BIC model selection

**Files:**
- Create: `ffrefine/fit/selection.py`
- Test: `tests/test_selection.py`

**Interfaces:**
- Consumes: `ffrefine.fit.linear.fit_cosine_series`, `FitResult`
- Produces:
  - `@dataclass Candidate`: `mults: tuple[int, ...]`, `result: FitResult`, `bic: float`, `rmse: float`, `n_params: int`
  - `bic(rmse: float, npoints: int, nparams: int) -> float`
  - `rank_candidates(phi_deg, target_kj, allowed_mults, weights=None, free_phase=False) -> list[Candidate]` — every non-empty subset, sorted by BIC ascending
  - `select_model(candidates: list[Candidate], rmse_tolerance: float = 0.5) -> Candidate` — smallest subset whose RMSE is within `rmse_tolerance` of the best (spec section 4.7)
  - `format_ranking_table(candidates: list[Candidate], limit: int = 20) -> str`

**Spec section 4.7:** "evaluate all subsets of the allowed multiplicities, rank by BIC, and by default select the smallest subset whose RMSE is within 0.5 kJ/mol of the best. Print the full ranking table." With the default 5 multiplicities that is 31 subsets — cheap.

BIC for least squares with Gaussian residuals: `n * ln(RSS/n) + k * ln(n)`, using `RSS/n = rmse^2`, so `bic = n*ln(rmse^2) + k*ln(n)`. Guard `rmse == 0` with a floor.

- [ ] **Step 1: Write the failing test** — `tests/test_selection.py`

```python
import numpy as np
import pytest
from ffrefine.fit.selection import (
    Candidate, bic, format_ranking_table, rank_candidates, select_model,
)
from ffrefine.mm.analytic import series_energy


@pytest.fixture
def threefold():
    phi = np.linspace(-180, 180, 73, endpoint=False)
    target = series_energy(np.array([2.5]), np.array([0.0]), np.array([3]), phi)
    return phi, target


def test_bic_penalises_more_parameters():
    assert bic(1.0, 100, 5) > bic(1.0, 100, 2)


def test_bic_rewards_lower_error():
    assert bic(0.5, 100, 3) < bic(2.0, 100, 3)


def test_rank_enumerates_all_subsets(threefold):
    phi, target = threefold
    cands = rank_candidates(phi, target, [1, 2, 3])
    assert len(cands) == 7  # 2^3 - 1


def test_rank_enumerates_31_for_default_multiplicities(threefold):
    phi, target = threefold
    cands = rank_candidates(phi, target, [1, 2, 3, 4, 6])
    assert len(cands) == 31


def test_rank_sorted_by_bic_ascending(threefold):
    phi, target = threefold
    cands = rank_candidates(phi, target, [1, 2, 3, 4, 6])
    assert all(a.bic <= b.bic for a, b in zip(cands, cands[1:]))


def test_selects_minimal_model_for_pure_threefold(threefold):
    """A pure n=3 profile must select exactly {3}, not a larger set."""
    phi, target = threefold
    cands = rank_candidates(phi, target, [1, 2, 3, 4, 6])
    chosen = select_model(cands)
    assert chosen.mults == (3,)


def test_selects_both_terms_for_two_term_profile():
    phi = np.linspace(-180, 180, 73, endpoint=False)
    target = series_energy(np.array([2.0, 1.0]), np.array([0.0, 180.0]),
                           np.array([3, 1]), phi)
    cands = rank_candidates(phi, target, [1, 2, 3, 4, 6])
    chosen = select_model(cands)
    assert set(chosen.mults) == {1, 3}


def test_select_prefers_fewer_terms_within_tolerance():
    """Given two models within the RMSE tolerance, the smaller wins."""
    small = Candidate(mults=(3,), result=None, bic=10.0, rmse=0.40, n_params=2)
    big = Candidate(mults=(1, 2, 3, 4), result=None, bic=9.0, rmse=0.10, n_params=5)
    assert select_model([big, small], rmse_tolerance=0.5).mults == (3,)


def test_select_rejects_too_poor_a_model():
    small = Candidate(mults=(3,), result=None, bic=10.0, rmse=5.0, n_params=2)
    big = Candidate(mults=(1, 2, 3), result=None, bic=9.0, rmse=0.10, n_params=4)
    assert select_model([big, small], rmse_tolerance=0.5).mults == (1, 2, 3)


def test_ranking_table_is_printable(threefold):
    phi, target = threefold
    text = format_ranking_table(rank_candidates(phi, target, [1, 2, 3]))
    assert "BIC" in text and "RMSE" in text
    assert "3" in text
```

- [ ] **Step 2: Run to verify failure**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_selection.py -v"`
Expected: FAIL — no module `ffrefine.fit.selection`

- [ ] **Step 3: Write `ffrefine/fit/selection.py`**

```python
"""Multiplicity subset selection by BIC (spec section 4.7)."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .linear import FitResult, fit_cosine_series

DEFAULT_MULTIPLICITIES: tuple[int, ...] = (1, 2, 3, 4, 6)
"""Spec section 4.7; overridable with --multiplicities."""

DEFAULT_RMSE_TOLERANCE: float = 0.5
"""kJ/mol; a model within this of the best RMSE counts as good enough."""

_RMSE_FLOOR = 1e-12
"""Avoids log(0) for an exact fit."""


@dataclass
class Candidate:
    """One subset of multiplicities and how well it fitted."""

    mults: tuple[int, ...]
    result: FitResult | None
    bic: float
    rmse: float
    n_params: int


def bic(rmse: float, npoints: int, nparams: int) -> float:
    """Bayesian information criterion for a least-squares fit.

    ``n * ln(RSS/n) + k * ln(n)`` with ``RSS/n`` taken as ``rmse**2``.
    Lower is better.
    """
    variance = max(float(rmse) ** 2, _RMSE_FLOOR)
    return npoints * math.log(variance) + nparams * math.log(npoints)


def rank_candidates(
    phi_deg: np.ndarray,
    target_kj: np.ndarray,
    allowed_mults: Sequence[int] = DEFAULT_MULTIPLICITIES,
    weights: np.ndarray | None = None,
    free_phase: bool = False,
) -> list[Candidate]:
    """Fit every non-empty subset of the allowed multiplicities and rank by BIC.

    Args:
        phi_deg: measured dihedral values in degrees.
        target_kj: target difference curve in kJ/mol.
        allowed_mults: multiplicities to consider.
        weights: per-point weights; None means uniform.
        free_phase: allow arbitrary phases.

    Returns:
        Candidates sorted by ascending BIC.
    """
    allowed = sorted(set(int(m) for m in allowed_mults))
    npoints = int(np.count_nonzero(weights)) if weights is not None else len(phi_deg)
    out: list[Candidate] = []
    for size in range(1, len(allowed) + 1):
        for subset in itertools.combinations(allowed, size):
            result = fit_cosine_series(
                phi_deg, target_kj, subset, weights=weights, free_phase=free_phase
            )
            out.append(
                Candidate(
                    mults=subset,
                    result=result,
                    bic=bic(result.rmse, npoints, result.n_params),
                    rmse=result.rmse,
                    n_params=result.n_params,
                )
            )
    out.sort(key=lambda c: c.bic)
    return out


def select_model(
    candidates: list[Candidate], rmse_tolerance: float = DEFAULT_RMSE_TOLERANCE
) -> Candidate:
    """Pick the smallest subset whose RMSE is within tolerance of the best.

    Args:
        candidates: as returned by :func:`rank_candidates`.
        rmse_tolerance: kJ/mol slack against the best RMSE achieved.

    Returns:
        The chosen candidate.  Ties on subset size are broken by BIC.
    """
    if not candidates:
        raise ValueError("no candidates to select from")
    best_rmse = min(c.rmse for c in candidates)
    good = [c for c in candidates if c.rmse <= best_rmse + rmse_tolerance]
    return min(good, key=lambda c: (len(c.mults), c.bic))


def format_ranking_table(candidates: list[Candidate], limit: int = 20) -> str:
    """Render the ranking as a fixed-width table (spec section 4.7)."""
    lines = [
        f"{'multiplicities':<20} {'params':>6} {'RMSE kJ/mol':>12} {'BIC':>12}",
        "-" * 53,
    ]
    for c in candidates[:limit]:
        mults = ",".join(str(m) for m in c.mults)
        lines.append(f"{mults:<20} {c.n_params:>6d} {c.rmse:>12.4f} {c.bic:>12.2f}")
    if len(candidates) > limit:
        lines.append(f"... {len(candidates) - limit} further subsets not shown")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests, then commit**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_selection.py -v"`
Expected: 10 passed. `test_selects_minimal_model_for_pure_threefold` is the one that matters — if it picks a larger subset, `select_model` is sorting by BIC rather than by subset size within the tolerance band.

```bash
git add ffrefine/fit/selection.py tests/test_selection.py
git commit -m "feat: BIC-ranked multiplicity subset selection

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 16: Asymmetry detection and free-phase fitting

**Files:**
- Modify: `ffrefine/fit/linear.py` (append; do not alter `fit_cosine_series`)
- Test: `tests/test_asymmetry.py`

**Interfaces:**
- Consumes: `ffrefine.geometry.wrap_deg`
- Produces: `linear.asymmetry_metric(phi_deg: np.ndarray, target_kj: np.ndarray) -> float` — RMS difference between the profile and its mirror about `phi = 0`, in kJ/mol; `linear.is_asymmetric(phi_deg, target_kj, tolerance: float = 0.5) -> bool`

**Spec section 4.7:** a cosine-only basis cannot fit a phase-shifted profile. Detect it, warn, and offer `--free-phase`. The mirror comparison needs interpolation because measured angles are not on an exact grid: interpolate the profile onto `-phi` using `np.interp` over a periodically extended copy.

- [ ] **Step 1: Write the failing test** — `tests/test_asymmetry.py`

```python
import numpy as np
import pytest
from ffrefine.fit.linear import asymmetry_metric, fit_cosine_series, is_asymmetric
from ffrefine.mm.analytic import series_energy


def test_symmetric_profile_has_near_zero_metric():
    phi = np.linspace(-180, 180, 73, endpoint=False)
    target = series_energy(np.array([2.0]), np.array([0.0]), np.array([3]), phi)
    assert asymmetry_metric(phi, target) < 1e-8
    assert not is_asymmetric(phi, target)


def test_phase_shifted_profile_is_flagged():
    phi = np.linspace(-180, 180, 73, endpoint=False)
    target = series_energy(np.array([3.0]), np.array([57.0]), np.array([2]), phi)
    assert asymmetry_metric(phi, target) > 0.5
    assert is_asymmetric(phi, target)


def test_phase_180_is_still_symmetric():
    """delta = 180 is representable in the cosine-only basis."""
    phi = np.linspace(-180, 180, 73, endpoint=False)
    target = series_energy(np.array([2.0]), np.array([180.0]), np.array([2]), phi)
    assert not is_asymmetric(phi, target)


def test_free_phase_fits_what_cosine_only_cannot():
    phi = np.linspace(-180, 180, 73, endpoint=False)
    target = series_energy(np.array([3.0]), np.array([57.0]), np.array([2]), phi)
    plain = fit_cosine_series(phi, target, [1, 2, 3, 4, 6])
    free = fit_cosine_series(phi, target, [1, 2, 3, 4, 6], free_phase=True)
    assert free.rmse < 1e-8
    assert plain.rmse > free.rmse


def test_metric_handles_unsorted_angles():
    rng = np.random.default_rng(6)
    phi = rng.permutation(np.linspace(-180, 180, 73, endpoint=False))
    target = series_energy(np.array([2.0]), np.array([0.0]), np.array([3]), phi)
    assert asymmetry_metric(phi, target) < 1e-6
```

- [ ] **Step 2: Run to verify failure**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_asymmetry.py -v"`
Expected: FAIL — `ImportError: cannot import name 'asymmetry_metric'`

- [ ] **Step 3: Append to `ffrefine/fit/linear.py`**

```python
DEFAULT_ASYMMETRY_TOLERANCE: float = 0.5
"""kJ/mol; above this the cosine-only basis cannot represent the profile."""


def asymmetry_metric(phi_deg: np.ndarray, target_kj: np.ndarray) -> float:
    """RMS mismatch between a torsion profile and its mirror about phi = 0.

    A cosine-only basis is even in phi, so it can only represent profiles
    symmetric about zero.  Chiral or otherwise phase-shifted torsions break
    that assumption (spec section 4.7).

    Measured angles are not on an exact grid, so the mirrored profile is
    obtained by periodic interpolation rather than by index reversal.

    Args:
        phi_deg: measured dihedral values in degrees, any order.
        target_kj: the profile in kJ/mol.

    Returns:
        RMS difference in kJ/mol between the profile and its mirror.
    """
    phi = wrap_deg(np.asarray(phi_deg, dtype=float))
    y = np.asarray(target_kj, dtype=float)
    order = np.argsort(phi)
    ps, ys = phi[order], y[order]
    # Extend periodically so interpolation at -phi never falls off the ends.
    ext_p = np.concatenate([ps - 360.0, ps, ps + 360.0])
    ext_y = np.concatenate([ys, ys, ys])
    mirrored = np.interp(-ps, ext_p, ext_y)
    return float(np.sqrt(np.mean((ys - mirrored) ** 2)))


def is_asymmetric(
    phi_deg: np.ndarray,
    target_kj: np.ndarray,
    tolerance: float = DEFAULT_ASYMMETRY_TOLERANCE,
) -> bool:
    """Whether a profile is too asymmetric for the cosine-only basis."""
    return asymmetry_metric(phi_deg, target_kj) > tolerance
```

- [ ] **Step 4: Run tests, then commit**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_asymmetry.py -v"`
Expected: 5 passed

```bash
git add ffrefine/fit/linear.py tests/test_asymmetry.py
git commit -m "feat: profile asymmetry detection for free-phase fitting

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 17: Fit report statistics

**Files:**
- Create: `ffrefine/fit/report.py`
- Test: `tests/test_fit_report.py`

**Interfaces:**
- Consumes: `ffrefine.fit.linear.FitResult`, `ffrefine.mm.analytic.series_energy`
- Produces:
  - `@dataclass ProfileStats`: `minima_deg: list[float]`, `minima_kj: list[float]`, `barriers_kj: list[float]`, `global_min_deg: float`, `max_barrier_kj: float`
  - `profile_stats(phi_deg: np.ndarray, energy_kj: np.ndarray) -> ProfileStats` — positions and depths of minima, barrier heights (spec section 4.7)
  - `fit_summary(result: FitResult, phi_deg, target_kj) -> dict` — the JSON-serialisable block that becomes `fit.json`

- [ ] **Step 1: Write the failing test** — `tests/test_fit_report.py`

```python
import numpy as np
import pytest
from ffrefine.fit.linear import fit_cosine_series
from ffrefine.fit.report import fit_summary, profile_stats
from ffrefine.mm.analytic import series_energy


def test_threefold_profile_has_three_minima():
    phi = np.linspace(-180, 180, 361)
    e = series_energy(np.array([2.0]), np.array([0.0]), np.array([3]), phi)
    s = profile_stats(phi, e)
    assert len(s.minima_deg) == 3


def test_minima_of_threefold_are_at_180_and_plus_minus_60():
    phi = np.linspace(-180, 180, 361)
    e = series_energy(np.array([2.0]), np.array([0.0]), np.array([3]), phi)
    s = profile_stats(phi, e)
    found = sorted(abs(round(m)) for m in s.minima_deg)
    assert found == [60, 60, 180]


def test_barrier_height_of_threefold():
    """A single k=2 kJ/mol n=3 term has 4 kJ/mol barriers."""
    phi = np.linspace(-180, 180, 361)
    e = series_energy(np.array([2.0]), np.array([0.0]), np.array([3]), phi)
    s = profile_stats(phi, e)
    assert s.max_barrier_kj == pytest.approx(4.0, abs=0.05)


def test_global_minimum_reported():
    phi = np.linspace(-180, 180, 361)
    e = series_energy(np.array([2.0, 1.0]), np.array([0.0, 0.0]), np.array([3, 1]), phi)
    s = profile_stats(phi, e)
    assert e[np.argmin(np.abs(phi - s.global_min_deg))] == pytest.approx(e.min(), abs=1e-6)


def test_fit_summary_is_json_serialisable():
    import json

    phi = np.linspace(-180, 180, 73, endpoint=False)
    target = series_energy(np.array([2.0]), np.array([0.0]), np.array([3]), phi)
    r = fit_cosine_series(phi, target, [1, 2, 3])
    summary = fit_summary(r, phi, target)
    json.dumps(summary)  # must not raise
    assert summary["rmse_kj_per_mol"] == pytest.approx(0.0, abs=1e-9)
    assert summary["multiplicities"] == [1, 2, 3]


def test_fit_summary_reports_kcal_alongside_kj():
    phi = np.linspace(-180, 180, 73, endpoint=False)
    target = series_energy(np.array([4.184]), np.array([0.0]), np.array([3]), phi)
    r = fit_cosine_series(phi, target, [3])
    summary = fit_summary(r, phi, target)
    assert summary["terms"][0]["k_kcal_per_mol"] == pytest.approx(1.0, abs=1e-6)
```

- [ ] **Step 2: Run to verify failure**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_fit_report.py -v"`
Expected: FAIL — no module `ffrefine.fit.report`

- [ ] **Step 3: Write `ffrefine/fit/report.py`**

```python
"""Fit statistics: minima, barriers, and the fit.json payload (spec 4.7, 4.8)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import wrap_deg
from ..units import kj_to_kcal
from .linear import FitResult, asymmetry_metric


@dataclass
class ProfileStats:
    """Shape of a torsion profile."""

    minima_deg: list[float]
    minima_kj: list[float]
    barriers_kj: list[float]
    global_min_deg: float
    max_barrier_kj: float


def profile_stats(phi_deg: np.ndarray, energy_kj: np.ndarray) -> ProfileStats:
    """Locate minima and barrier heights of a periodic torsion profile.

    The profile is treated as periodic in 360 degrees, so a minimum at the end
    of the range is compared against the other end rather than being missed.

    Args:
        phi_deg: dihedral values in degrees, any order.
        energy_kj: energies in kJ/mol.

    Returns:
        Minima positions and depths, barrier heights between consecutive
        minima, and the global minimum.
    """
    phi = wrap_deg(np.asarray(phi_deg, dtype=float))
    e = np.asarray(energy_kj, dtype=float)
    order = np.argsort(phi)
    p, y = phi[order], e[order]

    n = len(y)
    prev = np.roll(y, 1)
    nxt = np.roll(y, -1)
    is_min = (y < prev) & (y <= nxt)
    minima_idx = np.flatnonzero(is_min)

    barriers: list[float] = []
    for a, b in zip(minima_idx, np.roll(minima_idx, -1)):
        if a < b:
            segment = y[a : b + 1]
        else:  # wraps through the period boundary
            segment = np.concatenate([y[a:], y[: b + 1]])
        if segment.size:
            barriers.append(float(segment.max() - min(y[a], y[b])))

    global_idx = int(np.argmin(y))
    return ProfileStats(
        minima_deg=[float(p[i]) for i in minima_idx],
        minima_kj=[float(y[i]) for i in minima_idx],
        barriers_kj=barriers,
        global_min_deg=float(p[global_idx]),
        max_barrier_kj=float(max(barriers)) if barriers else 0.0,
    )


def fit_summary(result: FitResult, phi_deg: np.ndarray, target_kj: np.ndarray) -> dict:
    """Build the JSON-serialisable fit record written to fit.json (spec 4.8).

    Energies are reported in kJ/mol, with kcal/mol alongside for users going
    back to CHARMM.  Conversion goes through ``units`` (spec section 9.8).
    """
    stats = profile_stats(phi_deg, result.predicted)
    return {
        "multiplicities": [int(m) for m in result.mults],
        "terms": [
            {
                "multiplicity": int(m),
                "k_kj_per_mol": float(k),
                "k_kcal_per_mol": float(kj_to_kcal(k)),
                "phase_deg": float(d),
            }
            for m, k, d in zip(result.mults, result.k, result.delta_deg)
        ],
        "free_phase": bool(result.free_phase),
        "rmse_kj_per_mol": float(result.rmse),
        "max_deviation_kj_per_mol": float(result.max_dev),
        "asymmetry_kj_per_mol": float(asymmetry_metric(phi_deg, target_kj)),
        "n_points": int(len(phi_deg)),
        "n_parameters": int(result.n_params),
        "phi_deg": [float(v) for v in np.asarray(phi_deg)],
        "target_kj_per_mol": [float(v) for v in np.asarray(target_kj)],
        "fitted_kj_per_mol": [float(v) for v in result.predicted],
        "residuals_kj_per_mol": [float(v) for v in result.residuals],
        "profile": {
            "minima_deg": stats.minima_deg,
            "minima_kj_per_mol": stats.minima_kj,
            "barriers_kj_per_mol": stats.barriers_kj,
            "global_minimum_deg": stats.global_min_deg,
            "max_barrier_kj_per_mol": stats.max_barrier_kj,
        },
    }
```

- [ ] **Step 4: Run tests, then commit**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_fit_report.py -v"`
Expected: 6 passed

```bash
git add ffrefine/fit/report.py tests/test_fit_report.py
git commit -m "feat: torsion profile statistics and fit.json payload

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 18: CHARMM prm writer

**Files:**
- Create: `ffrefine/io/prm.py`
- Test: `tests/test_prm.py`

**Interfaces:**
- Consumes: `ffrefine.io.topology.Topology`, `ffrefine.units.kj_to_kcal`
- Produces: `write_prm_block(topology, quad, k, delta_deg, mult, out_path, distribute: bool) -> str` — writes and returns the `DIHEDRALS` block text

**Spec section 4.8:** the block uses **atom types**, is in **kcal/mol**, and is only physically meaningful with `--distribute`. When `distribute` is False it must still be emitted, carrying a loud warning comment.

- [ ] **Step 1: Write the failing test** — `tests/test_prm.py`

```python
import numpy as np
import pytest
from ffrefine.io.prm import write_prm_block
from ffrefine.io.topology import read_topology


def test_prm_uses_atom_types_not_indices(tmp_path, butane_itp):
    t = read_topology(butane_itp)
    text = write_prm_block(t, (1, 5, 8, 11), np.array([4.184]), np.array([0.0]),
                           np.array([3]), tmp_path / "o.prm", distribute=True)
    assert "CG331" in text and "CG321" in text


def test_prm_is_in_kcal(tmp_path, butane_itp):
    """4.184 kJ/mol must be written as 1.0 kcal/mol."""
    t = read_topology(butane_itp)
    text = write_prm_block(t, (1, 5, 8, 11), np.array([4.184]), np.array([0.0]),
                           np.array([3]), tmp_path / "o.prm", distribute=True)
    numbers = [ln.split() for ln in text.splitlines()
               if ln.strip() and not ln.strip().startswith(("!", "DIHEDRALS", "END"))]
    assert float(numbers[0][4]) == pytest.approx(1.0, abs=1e-6)


def test_prm_has_dihedrals_header(tmp_path, butane_itp):
    t = read_topology(butane_itp)
    text = write_prm_block(t, (1, 5, 8, 11), np.array([1.0]), np.array([0.0]),
                           np.array([3]), tmp_path / "o.prm", distribute=True)
    assert text.strip().startswith("DIHEDRALS") or "DIHEDRALS" in text.splitlines()[0]


def test_prm_warns_loudly_without_distribute(tmp_path, butane_itp):
    """Spec section 4.8: a type-level block is meaningless without --distribute."""
    t = read_topology(butane_itp)
    text = write_prm_block(t, (1, 5, 8, 11), np.array([1.0]), np.array([0.0]),
                           np.array([3]), tmp_path / "o.prm", distribute=False)
    assert "WARNING" in text.upper()
    assert "distribute" in text.lower()


def test_prm_no_warning_with_distribute(tmp_path, butane_itp):
    t = read_topology(butane_itp)
    text = write_prm_block(t, (1, 5, 8, 11), np.array([1.0]), np.array([0.0]),
                           np.array([3]), tmp_path / "o.prm", distribute=True)
    assert "WARNING" not in text.upper()


def test_prm_one_line_per_multiplicity(tmp_path, butane_itp):
    t = read_topology(butane_itp)
    text = write_prm_block(t, (1, 5, 8, 11), np.array([1.0, 0.5]),
                           np.array([0.0, 180.0]), np.array([3, 2]),
                           tmp_path / "o.prm", distribute=True)
    data = [ln for ln in text.splitlines()
            if ln.strip() and not ln.strip().startswith(("!", "DIHEDRALS", "END"))]
    assert len(data) == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_prm.py -v"`
Expected: FAIL — no module `ffrefine.io.prm`

- [ ] **Step 3: Write `ffrefine/io/prm.py`**

```python
"""CHARMM .prm DIHEDRALS block emission (spec section 4.8).

CHARMM parameter files are in kcal/mol and address atoms by type, so this
output is only physically meaningful when the fit was distributed over every
equivalent quadruplet.  Without --distribute the block is still written, with a
warning, because users ask for it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..units import kj_to_kcal
from .topology import Topology

Quad = tuple[int, int, int, int]

_NO_DISTRIBUTE_WARNING = """\
! ============================ WARNING ============================
! This block was generated WITHOUT --distribute.
!
! The fitted force constants were written onto a single dihedral
! quadruplet and every other torsion about the same central bond was
! set to zero.  That is exact for the one molecule it was fitted
! against, but as a CHARMM dihedral TYPE it is wrong: it will be
! applied to every quadruplet matching these atom types.
!
! Re-run with --distribute if you want a transferable dihedral type.
! ================================================================="""


def write_prm_block(
    topology: Topology,
    quad: Quad,
    k: np.ndarray,
    delta_deg: np.ndarray,
    mult: np.ndarray,
    out_path: str | Path,
    distribute: bool,
) -> str:
    """Write a CHARMM DIHEDRALS block for the fitted torsion.

    Args:
        topology: supplies the atom types for the quadruplet.
        quad: 1-based (i, j, k, l).
        k: force constants in kJ/mol; converted to kcal/mol on output.
        delta_deg: phases in degrees.
        mult: multiplicities.
        out_path: destination .prm fragment.
        distribute: whether the fit was spread over all equivalent quadruplets.

    Returns:
        The block text that was written.
    """
    types = [topology.atoms[i - 1].atype for i in quad]  # 1-based -> 0-based
    lines: list[str] = []
    if not distribute:
        lines.append(_NO_DISTRIBUTE_WARNING)
    lines.append("DIHEDRALS")
    lines.append("!atom types            Kchi     n   delta")
    for kk, dd, nn in zip(np.atleast_1d(k), np.atleast_1d(delta_deg), np.atleast_1d(mult)):
        lines.append(
            f"{types[0]:<6s}{types[1]:<6s}{types[2]:<6s}{types[3]:<6s}"
            f"{kj_to_kcal(float(kk)):>10.4f}{int(nn):>5d}{float(dd):>9.2f}"
        )
    lines.append("")
    lines.append("END")
    text = "\n".join(lines) + "\n"
    Path(out_path).write_text(text)
    return text
```

- [ ] **Step 4: Run tests, then commit**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_prm.py -v"`
Expected: 6 passed

```bash
git add ffrefine/io/prm.py tests/test_prm.py
git commit -m "feat: CHARMM prm DIHEDRALS block writer in kcal/mol

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 19: Plots

**Files:**
- Create: `ffrefine/plots.py`
- Test: `tests/test_plots.py`

**Interfaces:**
- Consumes: matplotlib (Agg backend, set at import), `ffrefine.fit.linear.FitResult`
- Produces:
  - `plot_qm_profile(phi_deg, qm_kj, path) -> None`
  - `plot_mm_zeroed(phi_deg, mm_kj, path) -> None`
  - `plot_fit(phi_deg, target_kj, result: FitResult, path) -> None` — target with the fitted series overlaid
  - `plot_residuals(phi_deg, result: FitResult, path) -> None`
  - `plot_validation(phi_deg, qm_kj, mm_before_kj, mm_after_kj, rmse_before, rmse_after, path) -> None` (spec section 4.9)

All four spec section 4.8 plots plus the validation figure. Angles are wrapped and sorted before plotting so lines do not jump across the period boundary.

- [ ] **Step 1: Write the failing test** — `tests/test_plots.py`

```python
import numpy as np
import pytest
from ffrefine import plots
from ffrefine.fit.linear import fit_cosine_series
from ffrefine.mm.analytic import series_energy


@pytest.fixture
def fitted():
    phi = np.linspace(-180, 180, 73, endpoint=False)
    target = series_energy(np.array([2.0]), np.array([0.0]), np.array([3]), phi)
    return phi, target, fit_cosine_series(phi, target, [1, 2, 3])


def test_qm_profile_writes_a_file(tmp_path, fitted):
    phi, target, _ = fitted
    p = tmp_path / "qm.png"
    plots.plot_qm_profile(phi, target, p)
    assert p.exists() and p.stat().st_size > 1000


def test_fit_plot_writes_a_file(tmp_path, fitted):
    phi, target, r = fitted
    p = tmp_path / "fit.png"
    plots.plot_fit(phi, target, r, p)
    assert p.exists() and p.stat().st_size > 1000


def test_residual_plot_writes_a_file(tmp_path, fitted):
    phi, _, r = fitted
    p = tmp_path / "res.png"
    plots.plot_residuals(phi, r, p)
    assert p.exists()


def test_mm_zeroed_plot_writes_a_file(tmp_path, fitted):
    phi, target, _ = fitted
    p = tmp_path / "mm.png"
    plots.plot_mm_zeroed(phi, target, p)
    assert p.exists()


def test_validation_plot_writes_a_file(tmp_path, fitted):
    phi, target, r = fitted
    p = tmp_path / "val.png"
    plots.plot_validation(phi, target, target + 3.0, r.predicted, 3.0, 0.001, p)
    assert p.exists()


def test_unsorted_angles_do_not_raise(tmp_path, fitted):
    phi, target, _ = fitted
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(phi))
    plots.plot_qm_profile(phi[perm], target[perm], tmp_path / "u.png")
    assert (tmp_path / "u.png").exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_plots.py -v"`
Expected: FAIL — no module `ffrefine.plots`

- [ ] **Step 3: Write `ffrefine/plots.py`**

```python
"""Diagnostic figures (spec sections 4.8 and 4.9).

The Agg backend is selected at import so plotting works with no display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .fit.linear import FitResult  # noqa: E402
from .geometry import wrap_deg  # noqa: E402

_XLABEL = "dihedral / degrees"
_YLABEL = "energy / kJ mol$^{-1}$"
_TICKS = [-180, -120, -60, 0, 60, 120, 180]


def _sorted(phi_deg: np.ndarray, *series: np.ndarray):
    """Wrap and sort by angle so plotted lines do not jump at the period edge."""
    phi = wrap_deg(np.asarray(phi_deg, dtype=float))
    order = np.argsort(phi)
    return (phi[order], *(np.asarray(s, dtype=float)[order] for s in series))


def _finish(ax, path: str | Path, title: str) -> None:
    ax.set_xlabel(_XLABEL)
    ax.set_ylabel(_YLABEL)
    ax.set_title(title)
    ax.set_xticks(_TICKS)
    ax.grid(alpha=0.3)
    ax.legend()
    fig = ax.get_figure()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_qm_profile(phi_deg: np.ndarray, qm_kj: np.ndarray, path: str | Path) -> None:
    """Relaxed QM torsion profile, shifted to its own minimum."""
    phi, qm = _sorted(phi_deg, qm_kj)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(phi, qm - qm.min(), "o-", color="#1f77b4", label="QM")
    _finish(ax, path, "QM torsion profile")


def plot_mm_zeroed(phi_deg: np.ndarray, mm_kj: np.ndarray, path: str | Path) -> None:
    """MM profile with the target torsions removed."""
    phi, mm = _sorted(phi_deg, mm_kj)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(phi, mm - mm.min(), "s-", color="#d62728", label="MM, torsion zeroed")
    _finish(ax, path, "MM profile without the target torsion")


def plot_fit(
    phi_deg: np.ndarray, target_kj: np.ndarray, result: FitResult, path: str | Path
) -> None:
    """Target difference curve with the fitted cosine series overlaid."""
    phi, target, predicted = _sorted(phi_deg, target_kj, result.predicted)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(phi, target, "o", color="#1f77b4", label="target (QM - MM$_0$)")
    ax.plot(phi, predicted, "-", color="#ff7f0e", linewidth=2, label="fitted series")
    terms = ", ".join(f"n={m}" for m in result.mults)
    _finish(ax, path, f"Dihedral fit ({terms}), RMSE {result.rmse:.3f} kJ/mol")


def plot_residuals(phi_deg: np.ndarray, result: FitResult, path: str | Path) -> None:
    """Fit residuals against dihedral angle."""
    phi, res = _sorted(phi_deg, result.residuals)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.axhline(0.0, color="#888888", linewidth=1)
    ax.plot(phi, res, "o-", color="#2ca02c", label="residual")
    _finish(ax, path, f"Residuals, max |dev| {result.max_dev:.3f} kJ/mol")


def plot_validation(
    phi_deg: np.ndarray,
    qm_kj: np.ndarray,
    mm_before_kj: np.ndarray,
    mm_after_kj: np.ndarray,
    rmse_before: float,
    rmse_after: float,
    path: str | Path,
) -> None:
    """Total MM energy against QM, before and after refitting (spec section 4.9)."""
    phi, qm, before, after = _sorted(phi_deg, qm_kj, mm_before_kj, mm_after_kj)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(phi, qm - qm.min(), "o-", color="#1f77b4", label="QM")
    ax.plot(phi, before - before.min(), "s--", color="#d62728",
            label=f"MM before (RMSE {rmse_before:.2f})")
    ax.plot(phi, after - after.min(), "^-", color="#2ca02c",
            label=f"MM after (RMSE {rmse_after:.2f})")
    _finish(ax, path, "Validation: MM against QM before and after refitting")
```

- [ ] **Step 4: Run tests, then commit**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_plots.py -v"`
Expected: 6 passed

```bash
git add ffrefine/plots.py tests/test_plots.py
git commit -m "feat: QM, MM, fit, residual and validation plots

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 20: Stage directories, manifests and resume

**Files:**
- Create: `ffrefine/stages.py`
- Test: `tests/test_stages.py`

**Interfaces:**
- Consumes: `ffrefine.errors`
- Produces:
  - `STAGE_DIRS: dict[str, str]` — the spec section 6 layout
  - `hash_file(path) -> str`, `hash_inputs(paths: list, extra: dict) -> str` (sha256, hex)
  - `stage_dir(outdir, stage) -> Path` — creates it
  - `write_manifest(outdir, stage, input_hash, metadata: dict) -> None`
  - `stage_is_current(outdir, stage, input_hash) -> bool` — the `--resume` predicate
  - `setup_logging(outdir, verbose: bool = False) -> None` — console plus `run.log`

**Spec section 6 layout** (exact directory names):

```
00_input/  01_qm_scan/  02_mm_scan/  03_fit/  04_charges/  final/  run.log
```

- [ ] **Step 1: Write the failing test** — `tests/test_stages.py`

```python
import json

import pytest
from ffrefine import stages


def test_stage_dirs_match_spec_layout():
    assert stages.STAGE_DIRS["input"] == "00_input"
    assert stages.STAGE_DIRS["qm_scan"] == "01_qm_scan"
    assert stages.STAGE_DIRS["mm_scan"] == "02_mm_scan"
    assert stages.STAGE_DIRS["fit"] == "03_fit"
    assert stages.STAGE_DIRS["charges"] == "04_charges"
    assert stages.STAGE_DIRS["final"] == "final"


def test_stage_dir_is_created(tmp_path):
    d = stages.stage_dir(tmp_path, "fit")
    assert d.exists()
    assert d.name == "03_fit"


def test_hash_file_is_stable(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("content")
    assert stages.hash_file(p) == stages.hash_file(p)


def test_hash_file_changes_with_content(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("one")
    first = stages.hash_file(p)
    p.write_text("two")
    assert stages.hash_file(p) != first


def test_hash_inputs_depends_on_extra(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("x")
    a = stages.hash_inputs([p], {"step": 15})
    b = stages.hash_inputs([p], {"step": 30})
    assert a != b


def test_hash_inputs_is_order_independent_for_extra(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("x")
    a = stages.hash_inputs([p], {"step": 15, "emax": 40})
    b = stages.hash_inputs([p], {"emax": 40, "step": 15})
    assert a == b


def test_stage_not_current_before_any_run(tmp_path):
    assert not stages.stage_is_current(tmp_path, "fit", "abc123")


def test_stage_current_after_matching_manifest(tmp_path):
    stages.stage_dir(tmp_path, "fit")
    stages.write_manifest(tmp_path, "fit", "abc123", {"note": "done"})
    assert stages.stage_is_current(tmp_path, "fit", "abc123")


def test_stage_not_current_when_inputs_changed(tmp_path):
    stages.stage_dir(tmp_path, "fit")
    stages.write_manifest(tmp_path, "fit", "abc123", {})
    assert not stages.stage_is_current(tmp_path, "fit", "different")


def test_manifest_is_readable_json(tmp_path):
    stages.stage_dir(tmp_path, "fit")
    stages.write_manifest(tmp_path, "fit", "h", {"multiplicities": [1, 3]})
    data = json.loads((tmp_path / "03_fit" / "manifest.json").read_text())
    assert data["input_hash"] == "h"
    assert data["metadata"]["multiplicities"] == [1, 3]


def test_setup_logging_creates_run_log(tmp_path):
    import logging

    stages.setup_logging(tmp_path)
    logging.getLogger("ffrefine").info("hello from the test")
    assert "hello from the test" in (tmp_path / "run.log").read_text()
```

- [ ] **Step 2: Run to verify failure**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_stages.py -v"`
Expected: FAIL — no module `ffrefine.stages`

- [ ] **Step 3: Write `ffrefine/stages.py`**

```python
"""Output layout, input hashing and resumable stages (spec sections 6 and 8)."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

STAGE_DIRS: dict[str, str] = {
    "input": "00_input",
    "qm_scan": "01_qm_scan",
    "mm_scan": "02_mm_scan",
    "fit": "03_fit",
    "charges": "04_charges",
    "final": "final",
}
"""The spec section 6 output layout."""

MANIFEST_NAME = "manifest.json"


def stage_dir(outdir: str | Path, stage: str) -> Path:
    """Return (creating if needed) the directory for a pipeline stage."""
    if stage not in STAGE_DIRS:
        raise KeyError(f"unknown stage {stage!r}; expected one of {sorted(STAGE_DIRS)}")
    path = Path(outdir) / STAGE_DIRS[stage]
    path.mkdir(parents=True, exist_ok=True)
    return path


def hash_file(path: str | Path) -> str:
    """SHA-256 of a file's contents, hex encoded."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_inputs(paths: list[str | Path], extra: dict | None = None) -> str:
    """Combined hash of input files and the settings that affect a stage.

    ``extra`` is serialised with sorted keys so equivalent settings hash
    identically regardless of dict ordering.
    """
    digest = hashlib.sha256()
    for path in paths:
        digest.update(hash_file(path).encode())
    digest.update(json.dumps(extra or {}, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def write_manifest(
    outdir: str | Path, stage: str, input_hash: str, metadata: dict | None = None
) -> None:
    """Record that a stage completed, and on what inputs."""
    path = stage_dir(outdir, stage) / MANIFEST_NAME
    path.write_text(
        json.dumps(
            {
                "stage": stage,
                "input_hash": input_hash,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {},
            },
            indent=2,
            default=str,
        )
        + "\n"
    )


def stage_is_current(outdir: str | Path, stage: str, input_hash: str) -> bool:
    """Whether a completed stage can be reused under --resume."""
    path = Path(outdir) / STAGE_DIRS[stage] / MANIFEST_NAME
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("input_hash") == input_hash
    except (OSError, json.JSONDecodeError):
        return False


def setup_logging(outdir: str | Path, verbose: bool = False) -> None:
    """Log to the console and to ``run.log`` under the output directory."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("ffrefine")
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    log.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    log.addHandler(console)

    handler = logging.FileHandler(outdir / "run.log", encoding="utf-8")
    handler.setFormatter(fmt)
    log.addHandler(handler)
```

- [ ] **Step 4: Run tests, then commit**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_stages.py -v"`
Expected: 11 passed

```bash
git add ffrefine/stages.py tests/test_stages.py
git commit -m "feat: stage layout, input hashing, resume manifests, run.log

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 21: Validation stage

**Files:**
- Create: `ffrefine/mm/validate_fit.py`
- Test: `tests/test_validate_fit.py`

**Interfaces:**
- Consumes: Tasks 10, 12, 19
- Produces:
  - `@dataclass ValidationResult`: `rmse_before: float`, `rmse_after: float`, `mm_before_kj: np.ndarray`, `mm_after_kj: np.ndarray`, `qm_kj: np.ndarray`, `phi_deg: np.ndarray`, `improved: bool`
  - `validate_refit(binaries, topology, quad, structure, frames, qm_kj, k, delta_deg, mult, workdir, distribute=False, nthreads=1) -> ValidationResult`

**Spec section 4.9:** rebuild the topology with the fitted parameters, redo the MM single points on the same geometries, and compare total MM energy to QM before and after. "This is the number that matters." Both curves are shifted to their own minimum before the RMSE is taken, since only relative energies are meaningful.

- [ ] **Step 1: Write the failing test** — `tests/test_validate_fit.py`

```python
import numpy as np
import pytest
from ffrefine.mm.validate_fit import ValidationResult, profile_rmse


def test_profile_rmse_is_zero_for_identical_shapes():
    phi = np.linspace(-180, 180, 37)
    a = np.cos(np.deg2rad(phi)) * 5.0
    assert profile_rmse(a, a) == pytest.approx(0.0, abs=1e-12)


def test_profile_rmse_ignores_constant_offset():
    """Only relative energies matter, so a rigid shift must not count."""
    phi = np.linspace(-180, 180, 37)
    a = np.cos(np.deg2rad(phi)) * 5.0
    assert profile_rmse(a, a + 1000.0) == pytest.approx(0.0, abs=1e-9)


def test_profile_rmse_detects_shape_difference():
    phi = np.linspace(-180, 180, 37)
    a = np.cos(np.deg2rad(phi)) * 5.0
    b = np.cos(2 * np.deg2rad(phi)) * 5.0
    assert profile_rmse(a, b) > 1.0


def test_validation_result_reports_improvement():
    r = ValidationResult(
        rmse_before=4.0, rmse_after=0.5,
        mm_before_kj=np.zeros(3), mm_after_kj=np.zeros(3),
        qm_kj=np.zeros(3), phi_deg=np.zeros(3), improved=True,
    )
    assert r.improved


@pytest.mark.needs_gmx
def test_refit_reproduces_the_original_profile(tmp_path, butane_pdb, butane_itp, gmx_bin):
    """Round trip through the validation stage: refitting the original
    parameters must restore the original MM profile."""
    from ffrefine.config import Binaries
    from ffrefine.geometry import dihedral_deg_frames
    from ffrefine.io.structure import read_pdb
    from ffrefine.io.topology import read_topology
    from ffrefine.io.topwriter import make_system_top
    from ffrefine.mm import gromacs
    from ffrefine.mm.validate_fit import validate_refit
    from tests.test_selfcheck import _rotated_frames

    s = read_pdb(butane_pdb)
    t = read_topology(butane_itp)
    frames = _rotated_frames(s, 24)
    b = Binaries(gmx=gmx_bin)

    full_itp = tmp_path / "full.itp"
    full_itp.write_text(butane_itp.read_text())
    full_top = tmp_path / "full.top"
    make_system_top(full_itp, full_top, "BUT")
    e_full = gromacs.single_point_energies(b, full_top, s, frames, tmp_path / "q", "q")
    qm = e_full.potential()

    # Nine equivalent terms about bond 5-8; distributing k recreates them.
    r = validate_refit(
        b, t, (1, 5, 8, 11), s, frames, qm,
        np.array([0.66944]), np.array([0.0]), np.array([3]),
        tmp_path / "val", distribute=True,
    )
    assert r.rmse_after < r.rmse_before
```

- [ ] **Step 2: Run to verify failure**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_validate_fit.py -v -m 'not needs_gmx'"`
Expected: FAIL — no module `ffrefine.mm.validate_fit`

- [ ] **Step 3: Write `ffrefine/mm/validate_fit.py`**

```python
"""Post-fit validation (spec section 4.9).

Rebuild the topology with the fitted parameters, recompute MM single points on
the same geometries, and compare the total MM energy against QM before and
after.  Spec section 4.9: this is the number that matters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Binaries
from ..geometry import dihedral_deg_frames
from ..io.structure import Structure
from ..io.topology import Topology
from ..io.topwriter import make_system_top, write_refit_topology
from .gromacs import single_point_energies

log = logging.getLogger("ffrefine")


@dataclass
class ValidationResult:
    """MM-against-QM agreement before and after refitting."""

    rmse_before: float
    rmse_after: float
    mm_before_kj: np.ndarray
    mm_after_kj: np.ndarray
    qm_kj: np.ndarray
    phi_deg: np.ndarray
    improved: bool


def profile_rmse(a_kj: np.ndarray, b_kj: np.ndarray) -> float:
    """RMSE between two energy profiles, each shifted to its own minimum.

    Absolute energies are not comparable between QM and MM, so both curves are
    referenced to their own minimum first.
    """
    a = np.asarray(a_kj, dtype=float)
    b = np.asarray(b_kj, dtype=float)
    return float(np.sqrt(np.mean(((a - a.min()) - (b - b.min())) ** 2)))


def validate_refit(
    binaries: Binaries,
    topology: Topology,
    quad: tuple[int, int, int, int],
    structure: Structure,
    frames: np.ndarray,
    qm_kj: np.ndarray,
    k: np.ndarray,
    delta_deg: np.ndarray,
    mult: np.ndarray,
    workdir: str | Path,
    distribute: bool = False,
    nthreads: int = 1,
) -> ValidationResult:
    """Compare MM against QM before and after applying the fitted parameters.

    Args:
        binaries: resolved binaries; ``gmx`` required.
        topology: the original parsed topology.
        quad: 1-based target quadruplet.
        structure: naming for the rerun trajectory.
        frames: the same geometries the QM profile was computed on.
        qm_kj: QM energies in kJ/mol.
        k, delta_deg, mult: the fitted series.
        workdir: stage directory.
        distribute: spread k over all equivalent quadruplets.
        nthreads: mdrun thread count.

    Returns:
        RMSE before and after, and the profiles behind them.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    before_itp = workdir / "before.itp"
    before_itp.write_text(Path(topology.source).read_text())
    before_top = workdir / "before.top"
    make_system_top(before_itp, before_top, topology.name)

    after_itp = workdir / "after.itp"
    write_refit_topology(topology, quad, k, delta_deg, mult, after_itp, distribute=distribute)
    after_top = workdir / "after.top"
    make_system_top(after_itp, after_top, topology.name)

    e_before = single_point_energies(
        binaries, before_top, structure, frames, workdir / "before", "before", nthreads
    )
    e_after = single_point_energies(
        binaries, after_top, structure, frames, workdir / "after", "after", nthreads
    )

    mm_before = e_before.potential()
    mm_after = e_after.potential()
    rmse_before = profile_rmse(qm_kj, mm_before)
    rmse_after = profile_rmse(qm_kj, mm_after)

    idx = (quad[0] - 1, quad[1] - 1, quad[2] - 1, quad[3] - 1)
    phi = dihedral_deg_frames(np.asarray(frames, dtype=float), idx)

    log.info(
        "validation: MM against QM RMSE %.3f kJ/mol before, %.3f kJ/mol after refitting",
        rmse_before, rmse_after,
    )
    if rmse_after > rmse_before:
        log.warning(
            "refitting made agreement WORSE (%.3f -> %.3f kJ/mol). Check the scan for "
            "conformational hops and the self-consistency diagnostics.",
            rmse_before, rmse_after,
        )

    return ValidationResult(
        rmse_before=rmse_before,
        rmse_after=rmse_after,
        mm_before_kj=mm_before,
        mm_after_kj=mm_after,
        qm_kj=np.asarray(qm_kj, dtype=float),
        phi_deg=phi,
        improved=rmse_after <= rmse_before,
    )
```

- [ ] **Step 4: Run tests, then commit**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && PATH=/home/marzazil/miniconda3/envs/gromacs/bin:\$PATH /home/marzazil/miniconda3/envs/gromacs/bin/python -m pytest tests/test_validate_fit.py -v"`
Expected: 5 passed

```bash
git add ffrefine/mm/validate_fit.py tests/test_validate_fit.py
git commit -m "feat: post-fit validation stage comparing MM to QM

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 22: CLI wiring, mmscan/fit/validate subcommands, README

**Files:**
- Create: `ffrefine/cli.py`, `README.md`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: every earlier task
- Produces:
  - `build_parser() -> argparse.ArgumentParser` — subcommands `scan`, `mmscan`, `fit`, `charges`, `validate`, `run`
  - `parse_dihedral(text: str) -> tuple[int, int, int, int]` — parses `"1,5,8,11"`, 1-based
  - `main(argv: list[str] | None = None) -> int`
  - `cmd_mmscan(args) -> int`, `cmd_fit(args) -> int`, `cmd_validate(args) -> int`

**Scope note:** `scan` and `charges` belong to milestones 4 and 5. Register them in the parser so the CLI surface matches spec section 3, but have them exit with a clear "not implemented until milestone 4/5" message rather than a stub that appears to work. `run` chains only the implemented stages and says which it skipped.

Common flags on every subcommand (spec section 3): `--config`, `--outdir`, `--nthreads`, `--resume`, `--dry-run`, plus `--gmx-bin`, `--xtb-bin`, `--orca-bin`.

- [ ] **Step 1: Write the failing test** — `tests/test_cli.py`

```python
import pytest
from ffrefine.cli import build_parser, main, parse_dihedral


def test_parse_dihedral_one_based():
    assert parse_dihedral("1,5,8,11") == (1, 5, 8, 11)


def test_parse_dihedral_tolerates_spaces():
    assert parse_dihedral(" 1 , 5 , 8 , 11 ") == (1, 5, 8, 11)


def test_parse_dihedral_rejects_wrong_count():
    with pytest.raises(SystemExit):
        parse_dihedral("1,5,8")


def test_parse_dihedral_rejects_zero_index():
    """Indices are 1-based; 0 is a common off-by-one mistake."""
    with pytest.raises(SystemExit):
        parse_dihedral("0,5,8,11")


def test_parser_has_all_six_subcommands():
    parser = build_parser()
    actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
    names = set()
    for a in actions:
        names |= set(a.choices)
    assert {"scan", "mmscan", "fit", "charges", "validate", "run"} <= names


def test_common_flags_present_on_mmscan():
    parser = build_parser()
    args = parser.parse_args(
        ["mmscan", "-s", "a.pdb", "-p", "a.itp", "--dihedral", "1,5,8,11",
         "--outdir", "out", "--nthreads", "4", "--resume", "--dry-run"]
    )
    assert args.outdir == "out"
    assert args.nthreads == 4
    assert args.resume is True
    assert args.dry_run is True


def test_dihedral_flag_is_repeatable():
    parser = build_parser()
    args = parser.parse_args(
        ["mmscan", "-s", "a.pdb", "-p", "a.itp",
         "--dihedral", "1,5,8,11", "--dihedral", "2,5,8,11"]
    )
    assert len(args.dihedral) == 2


def test_scan_reports_not_implemented(capsys):
    code = main(["scan", "-s", "a.pdb", "-p", "a.itp", "--dihedral", "1,5,8,11"])
    assert code != 0
    assert "milestone 4" in capsys.readouterr().err.lower()


def test_charges_reports_not_implemented(capsys):
    code = main(["charges", "-s", "a.pdb", "-p", "a.itp"])
    assert code != 0
    assert "milestone 5" in capsys.readouterr().err.lower()


def test_dry_run_does_not_create_output(tmp_path, butane_pdb, butane_itp):
    out = tmp_path / "out"
    code = main([
        "mmscan", "-s", str(butane_pdb), "-p", str(butane_itp),
        "--dihedral", "1,5,8,11", "--outdir", str(out), "--dry-run",
    ])
    assert code == 0
    assert not (out / "02_mm_scan" / "full.top").exists()


def test_multiplicities_flag_parses():
    parser = build_parser()
    args = parser.parse_args(
        ["fit", "-s", "a.pdb", "-p", "a.itp", "--dihedral", "1,5,8,11",
         "--multiplicities", "1,2,3"]
    )
    assert args.multiplicities == [1, 2, 3]
```

- [ ] **Step 2: Run to verify failure**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_cli.py -v"`
Expected: FAIL — no module `ffrefine.cli`

- [ ] **Step 3: Write `ffrefine/cli.py`**

```python
"""Command line interface (spec section 3).

Subcommands: scan, mmscan, fit, charges, validate, run.  ``scan`` and
``charges`` belong to milestones 4 and 5 and currently exit with a clear
message rather than pretending to work.

All atom indices on the command line are 1-based, matching the topology
[ atoms ] numbering.  Conversion to 0-based array indices happens inside
geometry and analytic evaluation, never here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .config import discover_binaries, load_config
from .errors import FFRefineError
from .fit.report import fit_summary
from .fit.selection import (
    DEFAULT_MULTIPLICITIES, format_ranking_table, rank_candidates, select_model,
)
from .fit.weights import boltzmann_weights, uniform_weights
from .geometry import dihedral_deg_frames
from .io.prm import write_prm_block
from .io.structure import read_structure
from .io.topology import read_topology
from .io.topwriter import make_system_top, write_refit_topology, write_zeroed_topology
from .io.validate import validate_structure_topology
from .mm.gromacs import single_point_energies
from .mm.selfcheck import check_torsion_removal
from .mm.torsionset import validate_quad
from .stages import hash_inputs, setup_logging, stage_dir, stage_is_current, write_manifest


def parse_dihedral(text: str) -> tuple[int, int, int, int]:
    """Parse a ``i,j,k,l`` dihedral specification of 1-based atom indices."""
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise SystemExit(f"--dihedral needs exactly 4 comma-separated indices, got {text!r}")
    try:
        idx = tuple(int(p) for p in parts)
    except ValueError:
        raise SystemExit(f"--dihedral indices must be integers, got {text!r}") from None
    if any(i < 1 for i in idx):
        raise SystemExit(f"--dihedral indices are 1-based; {text!r} contains a value below 1")
    return idx  # type: ignore[return-value]


def _parse_mults(text: str) -> list[int]:
    return [int(p.strip()) for p in text.split(",") if p.strip()]


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-s", "--structure", required=True, help="input .pdb or .gro")
    parser.add_argument("-p", "--topology", required=True, help="input .itp or .top")
    parser.add_argument("--config", help="JSON config file; CLI flags override it")
    parser.add_argument("--outdir", default="ffrefine_out", help="output directory")
    parser.add_argument("--nthreads", type=int, default=1, help="threads for external tools")
    parser.add_argument("--resume", action="store_true", help="skip stages whose inputs are unchanged")
    parser.add_argument("--dry-run", action="store_true", help="report what would run, change nothing")
    parser.add_argument("--verbose", action="store_true", help="debug-level logging")
    parser.add_argument("--gmx-bin", help="explicit path to gmx")
    parser.add_argument("--xtb-bin", help="explicit path to xtb")
    parser.add_argument("--orca-bin", help="explicit path to orca")
    parser.add_argument("--no-strict-names", action="store_true",
                        help="downgrade atom-name mismatches to warnings")


def _add_torsion(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dihedral", action="append", type=parse_dihedral, default=None,
                        help="target torsion as i,j,k,l (1-based); repeatable")
    parser.add_argument("--distribute", action="store_true",
                        help="spread the fitted k_n over all equivalent quadruplets")


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="ffrefine",
        description="Refine small-molecule force field torsions and charges.",
    )
    parser.add_argument("--version", action="version", version=f"ffrefine {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="QM relaxed torsion scan (milestone 4)")
    _add_common(scan)
    _add_torsion(scan)
    scan.add_argument("--step", type=float, default=15.0, help="scan step in degrees")

    mmscan = sub.add_parser("mmscan", help="GROMACS single points with and without the torsion")
    _add_common(mmscan)
    _add_torsion(mmscan)

    fit = sub.add_parser("fit", help="fit the dihedral series and write new parameters")
    _add_common(fit)
    _add_torsion(fit)
    fit.add_argument("--multiplicities", type=_parse_mults,
                     default=list(DEFAULT_MULTIPLICITIES), help="e.g. 1,2,3,4,6")
    fit.add_argument("--emax", type=float, default=40.0,
                     help="discard points this far above the QM minimum, kJ/mol")
    fit.add_argument("--boltzmann", nargs="?", type=float, const=300.0, default=None,
                     help="Boltzmann weighting at the given temperature (default 300 K)")
    fit.add_argument("--free-phase", action="store_true",
                     help="fit arbitrary phases rather than 0/180 only")

    charges = sub.add_parser("charges", help="ESP/RESP charges via ORCA (milestone 5)")
    _add_common(charges)

    validate = sub.add_parser("validate", help="rebuild with fitted parameters and compare to QM")
    _add_common(validate)
    _add_torsion(validate)

    run = sub.add_parser("run", help="run every implemented stage in order")
    _add_common(run)
    _add_torsion(run)

    return parser


def _load_inputs(args):
    """Read, cross-check and return (structure, topology, quads)."""
    structure = read_structure(args.structure)
    topology = read_topology(args.topology)
    warnings = validate_structure_topology(
        structure, topology, strict_names=not args.no_strict_names
    )
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    quads = args.dihedral or []
    if not quads:
        raise SystemExit(
            "no torsion selected. Pass --dihedral i,j,k,l (1-based). "
            "Automatic selection from a CGenFF .str file arrives in milestone 6."
        )
    for quad in quads:
        validate_quad(topology, quad)
    return structure, topology, quads


def cmd_not_implemented(name: str, milestone: int) -> int:
    """Report a subcommand that belongs to a later milestone."""
    print(
        f"ffrefine {name}: not implemented yet -- this subcommand arrives in "
        f"milestone {milestone}. Implemented so far: mmscan, fit, validate.",
        file=sys.stderr,
    )
    return 2


def cmd_mmscan(args) -> int:
    """Compute MM single points with and without the target torsions."""
    structure, topology, quads = _load_inputs(args)
    if args.dry_run:
        print(f"would run mmscan for {len(quads)} torsion(s) into {args.outdir}")
        return 0
    setup_logging(args.outdir, args.verbose)
    binaries = discover_binaries(
        {"gmx": args.gmx_bin, "xtb": args.xtb_bin, "orca": args.orca_bin}
    )
    workdir = stage_dir(args.outdir, "mm_scan")

    # Without a QM scan (milestone 4) the geometries come from the input
    # structure alone, so mmscan here computes the single point on that frame.
    frames = structure.coords[None, :, :]
    for quad in quads:
        result = check_torsion_removal(
            binaries, topology, quad, structure, frames, workdir, args.nthreads
        )
        print(
            f"torsion {quad}: self-consistency check passed, "
            f"max deviation {result.max_abs_error:.3g} kJ/mol"
        )
    return 0


def cmd_fit(args) -> int:
    """Fit the dihedral series from an existing scan.json and mm energies."""
    structure, topology, quads = _load_inputs(args)
    if args.dry_run:
        print(f"would fit {len(quads)} torsion(s) into {args.outdir}")
        return 0
    setup_logging(args.outdir, args.verbose)
    scan_path = Path(args.outdir) / "01_qm_scan" / "scan.json"
    if not scan_path.exists():
        raise SystemExit(
            f"no QM scan found at {scan_path}. Run 'ffrefine scan' first "
            f"(milestone 4), or supply scan.json with 'phi_deg' and 'energy_kj' arrays."
        )
    scan = json.loads(scan_path.read_text())
    phi = np.array(scan["phi_deg"], dtype=float)
    qm = np.array(scan["energy_kj"], dtype=float)

    energies_path = Path(args.outdir) / "02_mm_scan" / "energies.json"
    if not energies_path.exists():
        raise SystemExit(f"no MM energies found at {energies_path}. Run 'ffrefine mmscan' first.")
    mm0 = np.array(json.loads(energies_path.read_text())["zeroed_potential_kj"], dtype=float)

    target = (qm - qm.min()) - (mm0 - mm0.min())
    weights = (
        boltzmann_weights(qm, args.boltzmann, args.emax)
        if args.boltzmann is not None
        else uniform_weights(qm, args.emax)
    )
    candidates = rank_candidates(
        phi, target, args.multiplicities, weights=weights, free_phase=args.free_phase
    )
    print(format_ranking_table(candidates))
    chosen = select_model(candidates)
    result = chosen.result
    print(f"\nselected multiplicities {chosen.mults}, RMSE {result.rmse:.4f} kJ/mol")

    outdir = stage_dir(args.outdir, "fit")
    (outdir / "fit.json").write_text(json.dumps(fit_summary(result, phi, target), indent=2) + "\n")
    write_refit_topology(
        topology, quads[0], result.k, result.delta_deg, result.mults,
        outdir / "dihedral.itp", distribute=args.distribute,
    )
    write_prm_block(
        topology, quads[0], result.k, result.delta_deg, result.mults,
        outdir / "dihedral.prm", distribute=args.distribute,
    )
    return 0


def cmd_validate(args) -> int:
    """Rebuild with the fitted parameters and compare MM to QM."""
    print(
        "ffrefine validate: requires a QM scan from milestone 4. "
        "The validation machinery itself is implemented and covered by tests; "
        "wire it up once 'ffrefine scan' exists.",
        file=sys.stderr,
    )
    return 2


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns a process exit code."""
    args = build_parser().parse_args(argv)
    if args.config:
        load_config(args.config, {})  # validated early so a bad file fails fast
    handlers = {
        "scan": lambda a: cmd_not_implemented("scan", 4),
        "charges": lambda a: cmd_not_implemented("charges", 5),
        "mmscan": cmd_mmscan,
        "fit": cmd_fit,
        "validate": cmd_validate,
        "run": cmd_mmscan,
    }
    try:
        return handlers[args.command](args)
    except FFRefineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests**

Run: `wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && $PY -m pytest tests/test_cli.py -v"`
Expected: 11 passed

- [ ] **Step 5: Write `README.md`**

Must contain the worked butane example and a section on physical assumptions and their limits (spec section 8).

```markdown
# ffrefine

Repairs high-penalty parameters from generalized force fields (CGenFF, GAFF) by
refitting proper dihedrals against a QM torsion scan and replacing partial
charges with ESP-derived ones.

## Status

Milestones 1-3 of the build order are complete: parsers, the GROMACS energy
engine with verified torsion zeroing, and the fitter with model selection,
outputs and validation. `ffrefine scan` (xtb) and `ffrefine charges` (ORCA)
arrive in milestones 4 and 5.

## Requirements

- Python 3.10+, with numpy, scipy and matplotlib
- GROMACS 2023 or newer on PATH
- xtb and ORCA for the QM stages

## Install

    pip install -e .

## Worked example: n-butane

The C1-C2-C3-C4 torsion of n-butane is atoms 1, 5, 8, 11 in the bundled
fixture. Its central bond 5-8 carries nine dihedral entries; all nine are
zeroed to build the MM reference, which is the point of spec section 9.1.

    ffrefine mmscan \
        -s fixtures/butane/butane.pdb \
        -p fixtures/butane/butane.itp \
        --dihedral 1,5,8,11 \
        --outdir butane_out

This builds `full.top` and `zeroed.top`, runs GROMACS single points on both,
and runs the mandatory self-consistency check: `E_full - E_zeroed` must equal
the analytic sum of the removed torsion terms to better than 1e-3 kJ/mol. If it
does not, the run aborts rather than fitting nonsense.

Once a QM scan exists, fitting follows:

    ffrefine fit \
        -s fixtures/butane/butane.pdb \
        -p fixtures/butane/butane.itp \
        --dihedral 1,5,8,11 \
        --multiplicities 1,2,3,4,6 \
        --outdir butane_out

Expected for butane: a 3-fold profile with the trans global minimum, gauche
about 2.5 to 3.5 kJ/mol above trans, and a syn barrier of 14 to 20 kJ/mol.

## Physical assumptions and their limits

**The fitted torsion is convention-bound.** It absorbs whatever the 1-4
interactions do not capture, so the parameters are only valid alongside the
same nonbonded parameters and the same 1-4 conventions (`fudgeQQ`, `fudgeLJ`)
as the topology they were fitted against. CHARMM and Amber/GAFF differ here.
Moving a fitted dihedral between force fields is not meaningful.

**Vacuum throughout.** The MM reference is gas phase, so the QM scan is too. No
implicit solvent enters the torsion profile.

**Molecule-specific by default.** Without `--distribute` the fitted series is
written onto one quadruplet and the other torsions about that bond are zeroed.
That reproduces this molecule exactly but is not a transferable dihedral type.
Use `--distribute` when you want a reusable type; the CHARMM `.prm` output is
only physically meaningful in that mode.

**Relaxed scans hop conformers.** The scan driver flags discontinuities rather
than silently fitting across them. Read the warnings.

**Impropers and 1-4 pairs are never touched.** Bonds, angles, LJ parameters and
improper dihedrals are out of scope.

## Tests

    pytest

Tests needing GROMACS are marked `needs_gmx` and skip when it is absent. The
round-trip refit test exercises the whole pipeline except xtb and ORCA: it
takes MM energies with the original parameters as a pseudo-QM reference, zeroes
the torsion, refits, and requires the original coefficients back.
```

- [ ] **Step 6: Run the complete suite and format**

```bash
wsl.exe -e bash -lc "cd /mnt/c/Users/hrach/Projects/ffrefine && /home/marzazil/miniconda3/envs/gromacs/bin/python -m black ffrefine/ tests/ scripts/ && PATH=/home/marzazil/miniconda3/envs/gromacs/bin:\$PATH /home/marzazil/miniconda3/envs/gromacs/bin/python -m pytest tests/ -v"
```
Expected: black reformats cleanly and the whole suite passes with nothing skipped.

- [ ] **Step 7: Commit**

```bash
git add ffrefine/cli.py README.md tests/test_cli.py
git commit -m "feat: CLI subcommands and README with worked butane example

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

**Milestone 3 checkpoint:** stop and report. Milestones 4-6 (xtb scan driver, ORCA charges, CGenFF .str parsing and report.md) are the remaining work.

---

## Self-Review

**1. Spec coverage.** Sections mapped to tasks:

| Spec section | Task | Note |
|---|---|---|
| 1 Environment, binaries, no in-place edits | 7, 10 | `shutil.which` + overrides; writers always emit new files |
| 2 Inputs (structure, topology) | 3, 4, 5 | `--str` penalty parsing is milestone 6 |
| 3 CLI surface + common flags | 22 | `scan`/`charges` registered, exit with milestone message |
| 4.1 Topology object model | 4 | `#include`, `#ifdef`, `[ defaults ]`, fudge factors |
| 4.2 Target torsion set, `--distribute` | 8, 10 | all nine butane entries; `k_n / N_T` |
| 4.3 QM scan (xtb) | — | **milestone 4, out of scope** |
| 4.4 MM energies, mdp, fallback, name matching | 12 | both cutoff paths, one frame per frame asserted |
| 4.5 Mandatory self-consistency check | 9, 13 | independent analytic reference |
| 4.6 `--mm-relax` | — | **milestone 4, out of scope** (needs scan geometries) |
| 4.7 Fit, weighting, model selection, asymmetry | 11, 14, 15, 16, 17 | linear LSQ, BIC, `--free-phase` |
| 4.8 Outputs: itp, prm, fit.json, plots | 10, 17, 18, 19 | prm warns loudly without `--distribute` |
| 4.9 Validation | 21 | RMSE before and after |
| 5 Charges | — | **milestone 5, out of scope** |
| 6 Output layout, run.log | 20 | exact directory names |
| 7.1 Round-trip refit | 13 | |
| 7.2 Analytic torsion recovery | 9, 13 | |
| 7.3 PDB vs GRO equivalence | 6 | |
| 7.4 Unit conversions | 1 | |
| 7.5 Parser tests | 4 | funct 9 multi-line, `#ifdef`, impropers excluded |
| 7.6 Butane integration | 6, and milestone 4 for the QM half | fixture + `grompp` acceptance now |
| 8 Logging, resume, nthreads, 1-based indices | 7, 20, 22 | |
| 9.1-9.10 Pitfalls | 8, 10, 11, 12, 13, 17 | each has a named test |

Gaps are confined to milestones 4-6 and are labelled as such. `--mm-relax` (4.6) is deferred with 4.3 because it operates on scan geometries; flag this at the milestone 3 checkpoint.

**2. Placeholder scan.** No "TBD" or "add error handling" steps. Two deliberate `<RECORD ...>` markers exist in commit messages (Task 6 Step 7, Task 12 Step 6) where the implementer must write down an observed runtime fact; these are instructions to record a result, not deferred work.

**3. Type consistency.** Checked across tasks: `FitResult` fields (`mults`, `k`, `delta_deg`, `rmse`, `max_dev`, `residuals`, `predicted`, `n_params`, `free_phase`) are used identically in Tasks 15, 17, 19, 22. `EnergyResult.potential()` is used in Tasks 13, 21. `write_zeroed_topology` returns `list[Dihedral]` and is consumed as such by `torsion_set_energy` in Task 13. `Dihedral.central_bond` is defined in Task 4 and used in Task 8. `single_point_energies` has one signature across Tasks 12, 13, 21. `stage_dir` / `write_manifest` / `stage_is_current` agree between Tasks 20 and 22.

One known rough edge, called out rather than hidden: Task 12 Step 3 contains a stray `proc = run_command(...)` for `gmx energy` that `_feed_stdin_energy` supersedes; the step text instructs the implementer to delete it.
