# Milestone 4 — `ffrefine scan` (xtb relaxed torsion scan) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `ffrefine scan`, which drives a relaxed xtb torsion scan and writes `01_qm_scan/scan.json`, closing the pipeline so users no longer hand-write that file.

**Architecture:** A new `ffrefine/qm/` package splits the tool drivers (`xtb.py`, `orca.py`) from the analysis (`scan.py`), mirroring how `ffrefine/mm/` splits `gromacs.py` from `selfcheck.py`. The analysis half never touches a subprocess, so merge, hysteresis, hop and hydrogen-bond logic is tested entirely on captured and synthetic data. `cli.cmd_scan` wires them into the existing stage machinery.

**Tech Stack:** Python 3.10+, numpy, pytest, black. External: xtb 6.7.1, ORCA 6.1.1. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-14-ffrefine-m4-xtb-scan-design.md` (design), `ffrefine_spec.md` sections 4.3, 6, 8, 9, 10 (binding authority). Read both.

## Global Constraints

- **Python >= 3.10.** `pyproject.toml` pins `[tool.black] target-version = ["py310"]`, line-length 100. Run `black .` before every commit.
- **No new runtime dependencies.** numpy, scipy, matplotlib only (spec section 8).
- **All unit conversions live in `ffrefine/units.py`.** No other module may define a conversion constant (spec 9.8, and `units.py`'s own docstring). xtb is Hartree; `HARTREE_TO_KJ = 2625.499639` already exists.
- **Atom indices from the user and from topologies are 1-based**; convert to 0-based only where coordinates are indexed (`io/topology.py` module docstring).
- **Angles wrap to (-180, 180]** via `geometry.wrap_deg` (spec 9.6). Never reimplement this.
- **Every error reaching the user derives from `FFRefineError`** and prints as `error: ...` with no traceback (ruling R29). Use `ParseError` / `ValidationError` / `ExternalToolError` from `ffrefine/errors.py`.
- **`phi_deg` is always measured from coordinates, never from the requested grid** (spec 9.5). The design's section 2 probe shows xtb returning 180/120/-120/0 for a requested 0/120/240/360.
- **Write text files with `newline="\n"`** at every write site (P2-15).
- **pytest markers are declared in `pyproject.toml` only**, never also in `pytest_configure` (ruling R1).
- Verify in WSL, which is the only place the binaries exist:
  ```bash
  wsl.exe -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest -q'
  ```
- **Baseline to preserve: 327 passed, 0 skipped, 16 `needs_gmx` executing, `black --check` clean.**

## File Structure

| File | Responsibility |
|---|---|
| `ffrefine/units.py` | gains `ANGSTROM_TO_NM`, `angstrom_to_nm`, `nm_to_angstrom` (moved from `io/structure.py`) |
| `ffrefine/io/structure.py` | gains `write_xyz`; re-exports `ANGSTROM_TO_NM` for back-compat |
| `ffrefine/qm/__init__.py` | empty package marker |
| `ffrefine/qm/xtb.py` | xcontrol input generation, xtb invocation, `xtbscan.log` parsing, charge derivation |
| `ffrefine/qm/scan.py` | angle measurement, bidirectional merge, hysteresis, hop and hydrogen-bond detection |
| `ffrefine/qm/orca.py` | `--sp-refine` single points and `FINAL SINGLE POINT ENERGY` parsing |
| `ffrefine/cli.py` | `cmd_scan` replaces the not-implemented stub; new flags |
| `fixtures/butane/xtb/*` | captured real xtb output (already committed with this plan) |
| `tests/test_qm_xtb.py`, `tests/test_qm_scan.py`, `tests/test_qm_orca.py`, `tests/test_cli_scan.py` | tests |

Task order: 1 (shared helpers) → 2, 3 (xtb input/parse) → 4 (invocation) → 5, 6 (merge) → 7, 8 (discontinuity) → 9 (charge) → 10 (CLI) → 11 (`--sp-refine`) → 12 (docs).

---

### Task 1: Shared helpers — `ANGSTROM_TO_NM` in `units.py`, `write_xyz` in `structure.py`

**Files:**
- Modify: `ffrefine/units.py`
- Modify: `ffrefine/io/structure.py:16` (remove the constant, re-export it, add `write_xyz`)
- Test: `tests/test_units.py`, `tests/test_structure.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `units.ANGSTROM_TO_NM: float`, `units.angstrom_to_nm(x)`, `units.nm_to_angstrom(x)`, `io.structure.write_xyz(path, structure, coords=None, comment="") -> None`.

**Why:** `units.py`'s docstring states "No other module in ffrefine may define a conversion constant", but `io/structure.py:16` defines `ANGSTROM_TO_NM`. Task 3 converts Angstrom to nm on every scan frame, so leaving it would compound a stated-contract violation. `write_xyz` is needed by Task 4 and already exists hand-rolled in `tests/conftest.py`'s `butane_xyz` fixture.

- [ ] **Step 1: Write the failing tests**

In `tests/test_units.py`:

```python
def test_angstrom_to_nm_roundtrip():
    from ffrefine.units import ANGSTROM_TO_NM, angstrom_to_nm, nm_to_angstrom

    assert ANGSTROM_TO_NM == 0.1
    assert angstrom_to_nm(10.0) == pytest.approx(1.0)
    assert nm_to_angstrom(1.0) == pytest.approx(10.0)
    assert nm_to_angstrom(angstrom_to_nm(3.7)) == pytest.approx(3.7)


def test_structure_still_exports_angstrom_to_nm():
    """Back-compat: tests/conftest.py and io/structure.py itself import it from there."""
    from ffrefine.io.structure import ANGSTROM_TO_NM as from_structure
    from ffrefine.units import ANGSTROM_TO_NM as from_units

    assert from_structure is from_units
```

In `tests/test_structure.py`:

```python
def test_write_xyz_roundtrips_through_read_xyz(tmp_path, butane_pdb):
    from ffrefine.io.structure import read_pdb, read_xyz, write_xyz

    original = read_pdb(butane_pdb)
    out = tmp_path / "mol.xyz"
    write_xyz(out, original, comment="butane test")

    back = read_xyz(out)
    assert back.natoms == original.natoms
    assert back.elements == original.elements
    assert back.coords == pytest.approx(original.coords, abs=1e-6)


def test_write_xyz_is_in_angstrom_and_lf(tmp_path, butane_pdb):
    from ffrefine.io.structure import read_pdb, write_xyz

    s = read_pdb(butane_pdb)
    out = tmp_path / "mol.xyz"
    write_xyz(out, s, comment="c")

    raw = out.read_bytes()
    assert b"\r\n" not in raw
    lines = out.read_text().splitlines()
    assert lines[0].strip() == str(s.natoms)
    assert lines[1] == "c"
    # First atom's x in Angstrom is ten times the nm value.
    assert float(lines[2].split()[1]) == pytest.approx(s.coords[0][0] * 10.0, abs=1e-6)


def test_write_xyz_accepts_explicit_coords(tmp_path, butane_pdb):
    """Scan frames are passed as a separate array, not via structure.coords."""
    import numpy as np

    from ffrefine.io.structure import read_pdb, read_xyz, write_xyz

    s = read_pdb(butane_pdb)
    shifted = s.coords + 0.5
    out = tmp_path / "mol.xyz"
    write_xyz(out, s, coords=shifted)

    assert read_xyz(out).coords == pytest.approx(shifted, abs=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_units.py tests/test_structure.py -q -k "angstrom or write_xyz"`
Expected: FAIL with `ImportError: cannot import name 'angstrom_to_nm'` and `cannot import name 'write_xyz'`.

- [ ] **Step 3: Implement**

In `ffrefine/units.py`, after `KCAL_TO_KJ`:

```python
ANGSTROM_TO_NM: float = 0.1
"""nm per Angstrom.  xtb and ORCA speak Angstrom; GROMACS and ffrefine speak nm."""
```

and after `kj_to_kcal`:

```python
def angstrom_to_nm(x: Numeric) -> Numeric:
    """Convert Angstrom to nanometres."""
    return x * ANGSTROM_TO_NM


def nm_to_angstrom(x: Numeric) -> Numeric:
    """Convert nanometres to Angstrom."""
    return x / ANGSTROM_TO_NM
```

In `ffrefine/io/structure.py`, delete the `ANGSTROM_TO_NM: float = 0.1` definition at line 16 and its docstring, and import it instead (keeping the name importable from here, which `tests/conftest.py` relies on):

```python
from ..units import ANGSTROM_TO_NM  # re-exported: historically defined here
```

Append `write_xyz`:

```python
def write_xyz(
    path: str | Path,
    structure: Structure,
    coords: np.ndarray | None = None,
    comment: str = "",
) -> None:
    """Write one geometry as a plain .xyz in Angstrom.

    xtb and ORCA both read this format and both work in Angstrom, while
    ffrefine carries coordinates in nm, so the conversion happens here via
    :data:`ffrefine.units.ANGSTROM_TO_NM` -- the project's single conversion
    point (spec section 9.8).

    Atom order is the structure's, which is the topology's, which is what xtb
    will echo back in ``xtbscan.log``.  Nothing downstream re-derives the
    order, so it must not be permuted here.

    Args:
        path: destination file.
        structure: supplies element symbols and the atom count.
        coords: shape (natoms, 3) in nm; defaults to ``structure.coords``.
        comment: the second line of the file, free text.
    """
    xyz = structure.coords if coords is None else np.asarray(coords, float)
    if xyz.shape != (structure.natoms, 3):
        raise ValueError(f"coords shape {xyz.shape} incompatible with {structure.natoms} atoms")
    lines = [str(structure.natoms), comment]
    for element, (x, y, z) in zip(structure.elements, xyz / ANGSTROM_TO_NM):
        lines.append(f"{element:<2s} {x:>14.8f} {y:>14.8f} {z:>14.8f}")
    Path(path).write_text("\n".join(lines) + "\n", newline="\n")
```

Then simplify `tests/conftest.py`'s `butane_xyz` fixture to call `write_xyz` instead of hand-rolling the format.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_units.py tests/test_structure.py tests/test_cli.py -q`
Expected: PASS, no regressions. Then the full suite: `python -m pytest -q` — expect 327 + 5 new.

- [ ] **Step 5: Commit**

```bash
black . && git add -A
git commit -m "refactor: move ANGSTROM_TO_NM into units.py and add io.structure.write_xyz"
```

---

### Task 2: xcontrol scan-input generation

**Files:**
- Create: `ffrefine/qm/__init__.py` (empty), `ffrefine/qm/xtb.py`
- Test: `tests/test_qm_xtb.py`

**Interfaces:**
- Consumes: `ffrefine.errors.ValidationError`.
- Produces:
  - `xtb.scan_grid(step_deg: float) -> tuple[float, float, int]` returning `(start_deg, end_deg, npoints)`
  - `xtb.write_scan_input(path, quad, start_deg, end_deg, npoints, force_constant=1.0, freeze=()) -> str`

**Why:** Spec 4.3 fixes the grid as "`--step` degrees, default 15, spanning `0` to `360 - step` so endpoints are not duplicated". 0 and 360 are the same angle; including both would fit two points at identical geometry and distort any averaging.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from ffrefine.errors import ValidationError
from ffrefine.qm.xtb import scan_grid, write_scan_input


def test_scan_grid_excludes_the_duplicated_endpoint():
    """Spec 4.3: span 0 to 360-step, so 0 and 360 are not both sampled."""
    start, end, n = scan_grid(15.0)
    assert (start, end, n) == (0.0, 345.0, 24)


def test_scan_grid_rejects_a_step_that_does_not_divide_a_turn():
    with pytest.raises(ValidationError, match="360"):
        scan_grid(7.0)


def test_scan_grid_rejects_a_nonpositive_step():
    with pytest.raises(ValidationError):
        scan_grid(0.0)


def test_scan_input_matches_the_spec_block(tmp_path):
    text = write_scan_input(tmp_path / "scan.inp", (1, 5, 8, 11), 0.0, 345.0, 24)
    assert "$constrain" in text
    assert "force constant=1.0" in text
    assert "dihedral: 1,5,8,11,auto" in text
    assert "$scan" in text
    assert "1: 0.0,345.0,24" in text
    assert text.rstrip().endswith("$end")


def test_scan_input_adds_frozen_dihedrals_before_the_scan_block(tmp_path):
    """--freeze-dihedral holds other rotatable bonds; they must be constrained,
    not scanned, so they belong in $constrain and must not become scan entries."""
    text = write_scan_input(
        tmp_path / "scan.inp", (1, 5, 8, 11), 0.0, 345.0, 24, freeze=[(2, 1, 5, 8)]
    )
    constrain, scan = text.split("$scan")
    assert "dihedral: 2,1,5,8,auto" in constrain
    assert "2,1,5,8" not in scan
    # Exactly one scan entry, numbered 1, referring to the target dihedral.
    assert scan.count(":") == 1


def test_scan_input_is_written_with_lf(tmp_path):
    path = tmp_path / "scan.inp"
    write_scan_input(path, (1, 5, 8, 11), 0.0, 345.0, 24)
    assert b"\r\n" not in path.read_bytes()


def test_scan_input_rejects_a_repeated_atom(tmp_path):
    with pytest.raises(ValidationError):
        write_scan_input(tmp_path / "scan.inp", (1, 5, 5, 11), 0.0, 345.0, 24)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_qm_xtb.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ffrefine.qm'`.

- [ ] **Step 3: Implement**

Create `ffrefine/qm/__init__.py`:

```python
"""QM drivers: the xtb relaxed torsion scan and optional ORCA refinement."""
```

Create `ffrefine/qm/xtb.py`:

```python
"""xtb relaxed torsion scan driver (spec section 4.3).

xtb speaks Angstrom and Hartree; ffrefine speaks nm and kJ/mol.  Both
conversions go through :mod:`ffrefine.units`, which is the only place they
are allowed to live (spec section 9.8).

The ``$scan`` syntax this module emits was verified against xtb 6.7.1 by live
probe rather than assumed -- see the design document's section 2.  Spec 4.3
requires exactly that, and :func:`run_scan` re-verifies at runtime that the
block was actually honoured.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from ..errors import ValidationError

Quad = tuple[int, int, int, int]


def scan_grid(step_deg: float) -> tuple[float, float, int]:
    """Return ``(start_deg, end_deg, npoints)`` for a full turn at ``step_deg``.

    Spec 4.3: the grid spans 0 to ``360 - step`` so the endpoints are not
    duplicated.  0 and 360 degrees are the same angle; sampling both would put
    two points on one geometry, which biases any averaging over the profile
    and produces a duplicated abscissa in every plot.

    Raises:
        ValidationError: ``step_deg`` is not positive, or does not divide 360
            into a whole number of steps (which would leave the grid unable to
            close the turn).
    """
    if step_deg <= 0:
        raise ValidationError(f"--step must be positive, got {step_deg}")
    npoints = round(360.0 / step_deg)
    if abs(npoints * step_deg - 360.0) > 1e-9:
        raise ValidationError(
            f"--step {step_deg} does not divide 360 into whole steps "
            f"({360.0 / step_deg:.4f} of them). Use a step such as 5, 10, 15, 20 or 30."
        )
    return 0.0, 360.0 - step_deg, npoints


def _constraint_line(quad: Sequence[int]) -> str:
    if len(set(quad)) != 4:
        raise ValidationError(f"dihedral {tuple(quad)} repeats an atom index")
    return "  dihedral: " + ",".join(str(i) for i in quad) + ",auto"


def write_scan_input(
    path: str | Path,
    quad: Quad,
    start_deg: float,
    end_deg: float,
    npoints: int,
    force_constant: float = 1.0,
    freeze: Iterable[Quad] = (),
) -> str:
    """Write the xcontrol file driving a relaxed scan of ``quad``.

    ``auto`` makes xtb take each constrained dihedral's starting value from the
    input geometry, so the frozen dihedrals hold wherever the user's structure
    put them.  The scanned dihedral is listed first and is the only one given a
    ``$scan`` entry; ``freeze`` entries are constrained but never scanned,
    which is the whole point of ``--freeze-dihedral`` (spec 4.3).

    Args:
        path: destination xcontrol file.
        quad: 1-based target dihedral.
        start_deg, end_deg: scan range; ``end_deg`` may be below ``start_deg``
            for a reverse leg, which xtb accepts (verified on 6.7.1).
        npoints: number of scan points.
        force_constant: xtb constraint force constant.
        freeze: additional 1-based dihedrals to hold fixed.

    Returns:
        The text written.

    Raises:
        ValidationError: a quadruplet repeats an atom index.
    """
    lines = ["$constrain", f"  force constant={force_constant}", _constraint_line(quad)]
    for other in freeze:
        lines.append(_constraint_line(other))
    lines.append("$scan")
    lines.append(f"  1: {start_deg},{end_deg},{npoints}")
    lines.append("$end")
    text = "\n".join(lines) + "\n"
    Path(path).write_text(text, newline="\n")
    return text
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_qm_xtb.py -q`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
black . && git add -A
git commit -m "feat(qm): xcontrol scan-input generation for the xtb driver"
```

---

### Task 3: Parse `xtbscan.log`

**Files:**
- Modify: `ffrefine/qm/xtb.py`
- Test: `tests/test_qm_xtb.py`
- Fixtures (already committed): `fixtures/butane/xtb/xtbscan_forward_4pt.log`, `xtbscan_reverse_4pt.log`

**Interfaces:**
- Consumes: Task 2's module; `ffrefine.units.angstrom_to_nm`, `hartree_to_kj`; `ffrefine.errors.ParseError`.
- Produces: `xtb.parse_scan_log(path) -> tuple[np.ndarray, np.ndarray]` returning `(frames_nm, energy_kj)` with shapes `(nframes, natoms, 3)` and `(nframes,)`.

**Why:** `xtbscan.log` is a multi-frame XYZ in Angstrom whose comment lines read `energy: -13.665127531525 xtb: 6.7.1 (edcfbbe)`. The fixtures are genuine output from the design's probe, so these tests pin the real format rather than an invented one.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_qm_xtb.py`:

```python
import numpy as np

from ffrefine.errors import ParseError
from ffrefine.qm.xtb import parse_scan_log

FIXTURES = Path(__file__).parent.parent / "fixtures" / "butane" / "xtb"


@pytest.fixture
def forward_log() -> Path:
    return FIXTURES / "xtbscan_forward_4pt.log"


def test_parse_scan_log_reads_every_frame(forward_log):
    frames_nm, energy_kj = parse_scan_log(forward_log)
    assert frames_nm.shape == (4, 14, 3)
    assert energy_kj.shape == (4,)


def test_parse_scan_log_converts_hartree_to_kj(forward_log):
    """First frame's recorded energy is -13.665127531525 Eh."""
    _, energy_kj = parse_scan_log(forward_log)
    assert energy_kj[0] == pytest.approx(-13.665127531525 * 2625.499639, rel=1e-12)


def test_parse_scan_log_converts_angstrom_to_nm(forward_log):
    """The log's first atom is at x = -1.92585140500682 Angstrom."""
    frames_nm, _ = parse_scan_log(forward_log)
    assert frames_nm[0][0][0] == pytest.approx(-0.192585140500682, abs=1e-12)


def test_parse_scan_log_recovers_the_butane_profile(forward_log):
    """Sanity: relative energies must reproduce the probe's kJ/mol profile."""
    _, energy_kj = parse_scan_log(forward_log)
    relative = energy_kj - energy_kj.min()
    assert relative == pytest.approx([0.0, 10.741, 10.740, 20.691], abs=0.01)


def test_parse_scan_log_rejects_a_truncated_frame(tmp_path):
    """A killed xtb leaves a partial frame; that must not parse as valid."""
    path = tmp_path / "xtbscan.log"
    path.write_text("3\n energy: -1.0 xtb: 6.7.1\nC 0.0 0.0 0.0\nH 1.0 0.0 0.0\n", newline="\n")
    with pytest.raises(ParseError, match="truncated|incomplete"):
        parse_scan_log(path)


def test_parse_scan_log_rejects_a_missing_energy_comment(tmp_path):
    path = tmp_path / "xtbscan.log"
    path.write_text("1\nno energy here\nC 0.0 0.0 0.0\n", newline="\n")
    with pytest.raises(ParseError, match="energy"):
        parse_scan_log(path)


def test_parse_scan_log_rejects_a_varying_atom_count(tmp_path):
    """Frames must all describe the same molecule."""
    path = tmp_path / "xtbscan.log"
    path.write_text(
        "1\n energy: -1.0 xtb: x\nC 0.0 0.0 0.0\n"
        "2\n energy: -2.0 xtb: x\nC 0.0 0.0 0.0\nH 1.0 0.0 0.0\n",
        newline="\n",
    )
    with pytest.raises(ParseError, match="atom count"):
        parse_scan_log(path)


def test_parse_scan_log_rejects_an_empty_file(tmp_path):
    path = tmp_path / "xtbscan.log"
    path.write_text("", newline="\n")
    with pytest.raises(ParseError):
        parse_scan_log(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_qm_xtb.py -q -k parse_scan_log`
Expected: FAIL with `ImportError: cannot import name 'parse_scan_log'`.

- [ ] **Step 3: Implement**

Add to `ffrefine/qm/xtb.py` (and add `import numpy as np`, `from ..errors import ParseError, ValidationError`, `from ..units import angstrom_to_nm, hartree_to_kj`):

```python
def parse_scan_log(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read an ``xtbscan.log`` into coordinates and energies.

    The file is a plain multi-frame XYZ: an atom-count line, a comment line of
    the form ``energy: -13.665127531525 xtb: 6.7.1 (edcfbbe)``, then one line
    per atom with the element symbol and Angstrom coordinates.  Format
    confirmed against xtb 6.7.1 output, committed under ``fixtures/butane/xtb``.

    Every failure mode here is a *loud* one on purpose: a killed xtb leaves a
    truncated final frame, and silently returning the frames that did parse
    would hand the fit a profile with a missing grid point and no indication
    anything went wrong.

    Args:
        path: the log file.

    Returns:
        ``(frames_nm, energy_kj)`` with shapes ``(nframes, natoms, 3)`` in
        nanometres and ``(nframes,)`` in kJ/mol.  Energies are absolute, not
        referenced to a minimum.

    Raises:
        ParseError: the file is empty, a frame is truncated, a comment line
            carries no ``energy:``, or the atom count varies between frames.
    """
    path = Path(path)
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ParseError(f"could not read {path}: {exc}") from exc

    frames: list[list[list[float]]] = []
    energies: list[float] = []
    natoms: int | None = None
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        try:
            count = int(lines[i].split()[0])
        except (ValueError, IndexError) as exc:
            raise ParseError(f"{path}: expected an atom count at line {i + 1}") from exc
        if natoms is None:
            natoms = count
        elif count != natoms:
            raise ParseError(
                f"{path}: atom count changes from {natoms} to {count} at line {i + 1}; "
                "every frame must describe the same molecule"
            )
        if i + 1 >= len(lines):
            raise ParseError(f"{path}: truncated -- no comment line after line {i + 1}")
        comment = lines[i + 1]
        if "energy:" not in comment:
            raise ParseError(
                f"{path}: line {i + 2} carries no 'energy:' value: {comment.strip()!r}"
            )
        try:
            energies.append(float(comment.split("energy:")[1].split()[0]))
        except (ValueError, IndexError) as exc:
            raise ParseError(f"{path}: unreadable energy on line {i + 2}") from exc

        body = lines[i + 2 : i + 2 + count]
        if len(body) != count:
            raise ParseError(
                f"{path}: truncated -- frame {len(frames) + 1} declares {count} atoms "
                f"but only {len(body)} follow. xtb probably did not finish."
            )
        frame = []
        for offset, line in enumerate(body):
            parts = line.split()
            if len(parts) < 4:
                raise ParseError(f"{path}: incomplete atom line {i + 3 + offset}: {line!r}")
            try:
                frame.append([float(v) for v in parts[1:4]])
            except ValueError as exc:
                raise ParseError(f"{path}: unreadable coordinates on line {i + 3 + offset}") from exc
        frames.append(frame)
        i += 2 + count

    if not frames:
        raise ParseError(f"{path}: no frames found; xtb wrote no scan output")
    return angstrom_to_nm(np.array(frames, dtype=float)), hartree_to_kj(
        np.array(energies, dtype=float)
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_qm_xtb.py -q`
Expected: PASS (15 tests).

- [ ] **Step 5: Commit**

```bash
black . && git add -A
git commit -m "feat(qm): parse xtbscan.log into nm coordinates and kJ/mol energies"
```

---

### Task 4: Run the scan, and verify at runtime that `$scan` was honoured

**Files:**
- Modify: `ffrefine/qm/xtb.py`
- Test: `tests/test_qm_xtb.py`
- Modify: `pyproject.toml` (add the `needs_xtb` marker), `tests/conftest.py` (add `xtb_bin`)

**Interfaces:**
- Consumes: Tasks 2 and 3; `ffrefine.runner.run_command`; `ffrefine.config.Binaries`; `ffrefine.io.structure.write_xyz`.
- Produces: `xtb.run_scan(...) -> tuple[np.ndarray, np.ndarray]` (see signature below).

**Why:** Spec 4.3 says "Verify the exact `$scan` syntax against the installed xtb version's help output at runtime rather than assuming it; fail loudly if the scan block is rejected." `xtb --help` does not document xcontrol blocks at all (checked on 6.7.1), so the only honest verification is to confirm the relaxed scan actually ran and returned the requested number of frames.

- [ ] **Step 1: Write the failing test**

Register the marker in `pyproject.toml` (markers live here only — ruling R1):

```toml
markers = [
  "needs_gmx: requires a working gmx binary",
  "needs_xtb: requires a working xtb binary",
  "needs_orca: requires a working orca binary",
]
```

Add to `tests/conftest.py`:

```python
@pytest.fixture
def xtb_bin() -> str:
    """Path to xtb, or skip the test when xtb is unavailable."""
    exe = shutil.which("xtb")
    if exe is None:
        pytest.skip("xtb not on PATH")
    return exe
```

Add to `tests/test_qm_xtb.py`:

```python
from ffrefine.config import Binaries
from ffrefine.errors import ExternalToolError
from ffrefine.io.structure import read_pdb
from ffrefine.qm.xtb import run_scan


def test_run_scan_without_xtb_names_the_override_flag(tmp_path, butane_pdb):
    s = read_pdb(butane_pdb)
    with pytest.raises(Exception, match="xtb-bin"):
        run_scan(
            Binaries(xtb=None), s, (1, 5, 8, 11), tmp_path,
            start_deg=0.0, end_deg=345.0, npoints=24, charge=0, uhf=0,
        )


def test_run_scan_refuses_when_no_scan_log_appears(tmp_path, butane_pdb, monkeypatch):
    """If the $scan block were rejected, xtb can exit 0 having written nothing.

    That is the spec 4.3 failure this check exists for: a silent no-op must not
    be mistaken for a scan.
    """
    import ffrefine.qm.xtb as mod

    class _Proc:
        stdout = "normal termination of xtb"
        stderr = ""
        returncode = 0

    monkeypatch.setattr(mod, "run_command", lambda *a, **k: _Proc())
    s = read_pdb(butane_pdb)
    with pytest.raises(ExternalToolError, match=r"\$scan|xtbscan.log"):
        run_scan(
            Binaries(xtb="/usr/bin/true"), s, (1, 5, 8, 11), tmp_path,
            start_deg=0.0, end_deg=345.0, npoints=24, charge=0, uhf=0,
        )


def test_run_scan_refuses_a_short_scan(tmp_path, butane_pdb, monkeypatch):
    """24 points requested, 4 returned, means the scan did not do what was asked."""
    import shutil as _shutil

    import ffrefine.qm.xtb as mod

    class _Proc:
        stdout = "RELAXED SCAN\nnormal termination of xtb"
        stderr = ""
        returncode = 0

    def _fake(argv, workdir, log_stem, **kwargs):
        _shutil.copy(FIXTURES / "xtbscan_forward_4pt.log", Path(workdir) / "xtbscan.log")
        return _Proc()

    monkeypatch.setattr(mod, "run_command", _fake)
    s = read_pdb(butane_pdb)
    with pytest.raises(ExternalToolError, match="24"):
        run_scan(
            Binaries(xtb="/usr/bin/true"), s, (1, 5, 8, 11), tmp_path,
            start_deg=0.0, end_deg=345.0, npoints=24, charge=0, uhf=0,
        )


@pytest.mark.needs_xtb
def test_run_scan_drives_real_xtb(tmp_path, butane_pdb, xtb_bin):
    """Spec 4.3's runtime verification, as a test rather than as prose.

    Four points keeps this near 15 s; the default 24-point grid takes ~80 s.
    """
    s = read_pdb(butane_pdb)
    frames_nm, energy_kj = run_scan(
        Binaries(xtb=xtb_bin), s, (1, 5, 8, 11), tmp_path,
        start_deg=0.0, end_deg=360.0, npoints=4, charge=0, uhf=0, nthreads=4,
    )
    assert frames_nm.shape == (4, s.natoms, 3)
    assert energy_kj.shape == (4,)
    # Butane's syn barrier is real chemistry, not a smoke test.
    span = energy_kj.max() - energy_kj.min()
    assert 14.0 < span < 26.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_qm_xtb.py -q -k run_scan`
Expected: FAIL with `ImportError: cannot import name 'run_scan'`.

- [ ] **Step 3: Implement**

Add to `ffrefine/qm/xtb.py` (importing `run_command`, `Binaries`, `Structure`, `write_xyz`, `ExternalToolError`):

```python
SCAN_LOG_NAME = "xtbscan.log"
"""Filename xtb writes its relaxed-scan trajectory to, inside the working directory."""


def run_scan(
    binaries: Binaries,
    structure: Structure,
    quad: Quad,
    workdir: str | Path,
    *,
    start_deg: float,
    end_deg: float,
    npoints: int,
    charge: int,
    uhf: int,
    gfn: int = 2,
    nthreads: int = 1,
    freeze: Iterable[Quad] = (),
    force_constant: float = 1.0,
    coords_nm: np.ndarray | None = None,
    log_stem: str = "xtb_scan",
) -> tuple[np.ndarray, np.ndarray]:
    """Run one relaxed-scan leg and return its geometries and energies.

    Each leg gets its own ``workdir`` because xtb always writes
    ``xtbscan.log`` into the current directory: running a forward and a
    reverse leg in one directory would silently overwrite the first.

    Spec 4.3 requires the ``$scan`` syntax be verified against the installed
    xtb at runtime rather than assumed.  ``xtb --help`` documents no xcontrol
    block at all (checked on 6.7.1), so verification here is behavioural: the
    run must report a relaxed scan, must leave a log, and that log must hold
    exactly the requested number of frames.  A rejected block can otherwise
    leave xtb exiting 0 having quietly optimised the input geometry once.

    Args:
        binaries: resolved external programs; ``xtb`` must be set.
        structure: supplies elements and the default geometry.
        quad: 1-based target dihedral.
        workdir: directory to run in; created if absent.
        start_deg, end_deg, npoints: the scan range (see :func:`scan_grid`).
        charge: total molecular charge, passed as ``--chrg``.
        uhf: number of unpaired electrons, passed as ``--uhf``.
        gfn: GFN parameterisation, passed as ``--gfn``.
        nthreads: passed as ``-P`` (spec section 8).
        freeze: extra dihedrals held fixed.
        force_constant: constraint force constant.
        coords_nm: starting geometry in nm; defaults to ``structure.coords``.
            The reverse leg passes the forward leg's final frame.
        log_stem: base name for the runner's .cmd/.stdout/.stderr files.

    Returns:
        ``(frames_nm, energy_kj)`` as from :func:`parse_scan_log`.

    Raises:
        FFRefineError: xtb is not available.
        ExternalToolError: xtb failed, wrote no scan log, or returned a
            different number of frames than were requested.
    """
    exe = binaries.require("xtb")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    mol = workdir / "mol.xyz"
    write_xyz(mol, structure, coords_nm, comment=f"ffrefine scan input, torsion {quad}")
    inp = workdir / "scan.inp"
    write_scan_input(inp, quad, start_deg, end_deg, npoints, force_constant, freeze)

    proc = run_command(
        [
            exe, mol.name, "--opt", "tight", "--input", inp.name,
            "--chrg", str(charge), "--uhf", str(uhf), "--gfn", str(gfn),
            "-P", str(nthreads),
        ],
        workdir,
        log_stem,
    )

    log_path = workdir / SCAN_LOG_NAME
    if not log_path.exists():
        raise ExternalToolError(
            f"xtb exited 0 but wrote no {SCAN_LOG_NAME} in {workdir}, so no relaxed scan "
            f"ran. The $scan block in {inp.name} was most likely rejected by this xtb "
            f"build. Check {log_stem}.stdout for what it made of the input."
        )
    if "RELAXED SCAN" not in (proc.stdout or ""):
        raise ExternalToolError(
            f"xtb wrote {SCAN_LOG_NAME} but never reported 'RELAXED SCAN'; the $scan block "
            f"in {inp.name} was not honoured. See {log_stem}.stdout."
        )

    frames_nm, energy_kj = parse_scan_log(log_path)
    if len(energy_kj) != npoints:
        raise ExternalToolError(
            f"requested {npoints} scan points but {log_path} holds {len(energy_kj)}. "
            "The scan did not complete, so the profile would have a hole in it."
        )
    if frames_nm.shape[1] != structure.natoms:
        raise ExternalToolError(
            f"{log_path} has {frames_nm.shape[1]} atoms per frame, but the structure has "
            f"{structure.natoms}"
        )
    return frames_nm, energy_kj
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_qm_xtb.py -q`
Expected: PASS. Then confirm the live one really ran:
`wsl.exe -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_qm_xtb.py -q -m needs_xtb -v'`
Expected: 1 passed, not skipped.

- [ ] **Step 5: Commit**

```bash
black . && git add -A
git commit -m "feat(qm): drive xtb relaxed scans, verifying the \$scan block was honoured"
```

---

### Task 5: Measure angles and model a scan leg

**Files:**
- Create: `ffrefine/qm/scan.py`
- Test: `tests/test_qm_scan.py`

**Interfaces:**
- Consumes: `ffrefine.geometry.dihedral_deg_frames`, `wrap_deg`.
- Produces: `scan.ScanLeg` dataclass with fields `frames_nm`, `energy_kj`, `phi_deg`, `direction`; `scan.measure_angles(frames_nm, quad) -> np.ndarray`.

**Why:** Spec 9.5 — angles are measured from coordinates, never taken from the requested grid. The design's probe shows a requested 0/120/240/360 returning measured 180/120/-120/0.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest

from ffrefine.qm.scan import ScanLeg, measure_angles
from ffrefine.qm.xtb import parse_scan_log

FIXTURES = Path(__file__).parent.parent / "fixtures" / "butane" / "xtb"


def test_measured_angles_are_not_the_requested_grid():
    """The regression that justifies spec 9.5.

    The fixture was produced by requesting 0, 120, 240, 360 degrees.  xtb
    returned 180, 120, -120, 0.  Any code deriving phi from the request would
    mislabel the anti minimum as the syn barrier.
    """
    frames_nm, _ = parse_scan_log(FIXTURES / "xtbscan_forward_4pt.log")
    phi = measure_angles(frames_nm, (1, 5, 8, 11))
    assert phi == pytest.approx([180.0, 120.0, -119.99, -0.01], abs=0.05)
    requested = np.array([0.0, 120.0, 240.0, 360.0])
    assert not np.allclose(phi, requested, atol=1.0)


def test_measure_angles_converts_from_one_based_indices():
    """quad is 1-based, coords are 0-based; off-by-one here is silent and fatal."""
    frames_nm, _ = parse_scan_log(FIXTURES / "xtbscan_forward_4pt.log")
    from ffrefine.geometry import dihedral_deg_frames

    assert measure_angles(frames_nm, (1, 5, 8, 11)) == pytest.approx(
        dihedral_deg_frames(frames_nm, (0, 4, 7, 10))
    )


def test_measure_angles_wraps_to_the_project_convention():
    frames_nm, _ = parse_scan_log(FIXTURES / "xtbscan_forward_4pt.log")
    phi = measure_angles(frames_nm, (1, 5, 8, 11))
    assert np.all(phi > -180.0) and np.all(phi <= 180.0)


def test_scan_leg_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        ScanLeg(
            frames_nm=np.zeros((3, 4, 3)),
            energy_kj=np.zeros(2),
            phi_deg=np.zeros(3),
            direction="forward",
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_qm_scan.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ffrefine.qm.scan'`.

- [ ] **Step 3: Implement**

Create `ffrefine/qm/scan.py`:

```python
"""Analysis of xtb scan output: angles, bidirectional merge, discontinuities.

Nothing in this module runs a subprocess.  That is deliberate: every rule it
implements -- which forward point pairs with which reverse point, what counts
as a conformational hop, when a hydrogen bond has formed -- is a judgement
that decides whether a published torsion profile is trustworthy, and those
judgements are testable only if they can be exercised on captured and
synthetic data without an external tool in the loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import dihedral_deg_frames, wrap_deg

Quad = tuple[int, int, int, int]


def measure_angles(frames_nm: np.ndarray, quad: Quad) -> np.ndarray:
    """Measure the target dihedral in every frame, in degrees.

    Spec 9.5: "Measure the actual dihedral value from each frame's
    coordinates. Do not trust the requested grid."  This is not a formality --
    on the committed 4-point butane fixture, a scan requested at 0/120/240/360
    came back at 180/120/-120/0, so code that labelled points by the request
    would report the anti minimum as the syn barrier.

    Args:
        frames_nm: shape (nframes, natoms, 3).
        quad: 1-based atom indices, as the user and the topology write them.

    Returns:
        Shape (nframes,) in degrees, wrapped to (-180, 180].
    """
    zero_based = tuple(i - 1 for i in quad)
    return dihedral_deg_frames(np.asarray(frames_nm, float), zero_based)


@dataclass
class ScanLeg:
    """One direction of a (possibly bidirectional) relaxed scan.

    Attributes:
        frames_nm: shape (nframes, natoms, 3) in nanometres.
        energy_kj: shape (nframes,) absolute energies in kJ/mol.
        phi_deg: shape (nframes,) measured target dihedral, degrees.
        direction: ``"forward"`` or ``"reverse"``, for messages and provenance.
    """

    frames_nm: np.ndarray
    energy_kj: np.ndarray
    phi_deg: np.ndarray
    direction: str

    def __post_init__(self) -> None:
        n = len(self.energy_kj)
        if len(self.phi_deg) != n or len(self.frames_nm) != n:
            raise ValueError(
                f"ScanLeg arrays disagree: {len(self.frames_nm)} frames, {n} energies, "
                f"{len(self.phi_deg)} angles"
            )

    def __len__(self) -> int:
        return len(self.energy_kj)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_qm_scan.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
black . && git add -A
git commit -m "feat(qm): measure scan angles from coordinates, never from the grid"
```

---

### Task 6: Bidirectional merge and hysteresis

**Files:**
- Modify: `ffrefine/qm/scan.py`
- Test: `tests/test_qm_scan.py`

**Interfaces:**
- Consumes: Task 5's `ScanLeg`, `measure_angles`.
- Produces: `scan.MergeResult` dataclass (`frames_nm`, `energy_kj`, `phi_deg`, `source`, `max_hysteresis_kj`, `worst_angle_deg`, `unpaired_forward_deg`, `unpaired_reverse_deg`); `scan.merge_bidirectional(forward, reverse, tol_deg, ) -> MergeResult`.

**Why — read this before writing code.** Spec 4.3: "Take the lower energy at each grid point." The two legs' measured angles from the design's probe were:

```
forward:  180.00   120.00  -119.99   -0.01
reverse:   -0.01  -119.99   119.99    0.01
```

Pairing by array index pairs **180 with 0** — the anti minimum against the syn barrier — and taking the lower energy then writes the minimum's energy onto the syn point, erasing the barrier the scan exists to measure. Three of four index pairings are wrong. Reversed-index pairing fails too: the reverse leg visits 0 twice and never reaches 180. **Pair by measured angle.**

- [ ] **Step 1: Write the failing test**

```python
from ffrefine.errors import ValidationError
from ffrefine.qm.scan import ScanLeg, measure_angles, merge_bidirectional
from ffrefine.qm.xtb import parse_scan_log


def _leg(path: Path, direction: str) -> ScanLeg:
    frames_nm, energy_kj = parse_scan_log(path)
    return ScanLeg(frames_nm, energy_kj, measure_angles(frames_nm, (1, 5, 8, 11)), direction)


def test_merge_pairs_by_angle_not_by_index():
    """The regression this whole design decision exists for.

    Index pairing would pair forward 180 deg with reverse 0 deg and take the
    lower energy, writing the anti minimum onto the syn point and flattening a
    20 kJ/mol barrier to nothing.
    """
    fwd = _leg(FIXTURES / "xtbscan_forward_4pt.log", "forward")
    rev = _leg(FIXTURES / "xtbscan_reverse_4pt.log", "reverse")

    merged = merge_bidirectional(fwd, rev, tol_deg=7.5)

    # Output keeps the forward grid's angles.
    assert merged.phi_deg == pytest.approx(fwd.phi_deg, abs=1e-9)
    # The barrier survives: syn is still ~20.7 kJ/mol above anti.
    relative = merged.energy_kj - merged.energy_kj.min()
    assert relative[0] == pytest.approx(0.0, abs=1e-9)
    assert relative[3] == pytest.approx(20.69, abs=0.1)


def test_merge_takes_the_lower_energy_at_each_paired_point():
    frames = np.zeros((2, 4, 3))
    fwd = ScanLeg(frames, np.array([10.0, 20.0]), np.array([0.0, 90.0]), "forward")
    rev = ScanLeg(frames, np.array([5.0, 25.0]), np.array([0.2, 90.1]), "reverse")

    merged = merge_bidirectional(fwd, rev, tol_deg=5.0)

    assert merged.energy_kj == pytest.approx([5.0, 20.0])
    assert merged.source == ["reverse", "forward"]


def test_merge_reports_hysteresis_and_its_worst_angle():
    frames = np.zeros((2, 4, 3))
    fwd = ScanLeg(frames, np.array([10.0, 20.0]), np.array([0.0, 90.0]), "forward")
    rev = ScanLeg(frames, np.array([10.5, 27.0]), np.array([0.0, 90.0]), "reverse")

    merged = merge_bidirectional(fwd, rev, tol_deg=5.0)

    assert merged.max_hysteresis_kj == pytest.approx(7.0)
    assert merged.worst_angle_deg == pytest.approx(90.0)


def test_merge_pairs_across_the_wrap_boundary():
    """-179 and +179 are two degrees apart, not 358."""
    frames = np.zeros((1, 4, 3))
    fwd = ScanLeg(frames, np.array([10.0]), np.array([179.0]), "forward")
    rev = ScanLeg(frames, np.array([4.0]), np.array([-179.0]), "reverse")

    merged = merge_bidirectional(fwd, rev, tol_deg=5.0)

    assert merged.energy_kj == pytest.approx([4.0])
    assert merged.unpaired_forward_deg == []


def test_merge_reports_unpaired_points_rather_than_dropping_them():
    frames = np.zeros((2, 4, 3))
    fwd = ScanLeg(frames, np.array([10.0, 20.0]), np.array([0.0, 90.0]), "forward")
    rev = ScanLeg(frames, np.array([5.0, 1.0]), np.array([0.0, 45.0]), "reverse")

    merged = merge_bidirectional(fwd, rev, tol_deg=5.0)

    # The forward point at 90 has no partner, so it keeps its own energy.
    assert merged.energy_kj == pytest.approx([5.0, 20.0])
    assert merged.unpaired_forward_deg == pytest.approx([90.0])
    assert merged.unpaired_reverse_deg == pytest.approx([45.0])


def test_merge_refuses_when_two_forward_points_claim_one_reverse_point():
    """A degenerate grid; guessing which pairing was meant would be a coin flip."""
    frames = np.zeros((2, 4, 3))
    fwd = ScanLeg(frames, np.array([10.0, 20.0]), np.array([0.0, 1.0]), "forward")
    rev = ScanLeg(frames, np.array([5.0]), np.array([0.5]), "reverse")

    with pytest.raises(ValidationError, match="both|ambiguous"):
        merge_bidirectional(fwd, rev, tol_deg=5.0)


def test_merge_keeps_the_frame_belonging_to_the_chosen_energy():
    """Taking the reverse energy but the forward geometry would be incoherent."""
    fwd_frames = np.zeros((1, 4, 3))
    rev_frames = np.ones((1, 4, 3))
    fwd = ScanLeg(fwd_frames, np.array([10.0]), np.array([0.0]), "forward")
    rev = ScanLeg(rev_frames, np.array([5.0]), np.array([0.0]), "reverse")

    merged = merge_bidirectional(fwd, rev, tol_deg=5.0)

    assert merged.frames_nm[0] == pytest.approx(rev_frames[0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_qm_scan.py -q -k merge`
Expected: FAIL with `ImportError: cannot import name 'merge_bidirectional'`.

- [ ] **Step 3: Implement**

Add to `ffrefine/qm/scan.py` (import `ValidationError`):

```python
@dataclass
class MergeResult:
    """The profile produced by combining a forward and a reverse leg.

    Attributes:
        frames_nm: chosen geometry per grid point, shape (npoints, natoms, 3).
        energy_kj: chosen energy per grid point, shape (npoints,).
        phi_deg: the forward leg's measured angles, shape (npoints,).
        source: ``"forward"`` or ``"reverse"`` per point, saying which leg won.
        max_hysteresis_kj: largest |E_forward - E_reverse| over paired points.
        worst_angle_deg: the angle where that maximum occurred, or nan.
        unpaired_forward_deg: forward angles with no reverse partner in tolerance.
        unpaired_reverse_deg: reverse angles that partnered nothing.
    """

    frames_nm: np.ndarray
    energy_kj: np.ndarray
    phi_deg: np.ndarray
    source: list[str]
    max_hysteresis_kj: float
    worst_angle_deg: float
    unpaired_forward_deg: list[float]
    unpaired_reverse_deg: list[float]


def angular_distance_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest absolute separation between two angles in degrees.

    Wrap-aware: -179 and +179 are two degrees apart, not 358.
    """
    return np.abs(wrap_deg(np.asarray(a, float) - np.asarray(b, float)))


def merge_bidirectional(
    forward: ScanLeg, reverse: ScanLeg, tol_deg: float
) -> MergeResult:
    """Combine two legs, taking the lower energy at each grid point (spec 4.3).

    **Points pair by measured angle, never by array index.**  Two independent
    relaxed scans do not land on the same angles, and the committed 4-point
    butane fixtures show why this matters: the forward leg measured
    180/120/-120/0 while the reverse measured 0/-120/120/0.  Index pairing
    would pair 180 with 0 -- the anti minimum against the syn barrier -- and
    "take the lower energy" would then write the minimum onto the syn point,
    erasing the 20.7 kJ/mol barrier the scan exists to measure.  Three of the
    four index pairings are wrong, and reversed-index pairing fares no better:
    the reverse leg visits 0 twice and never reaches 180 at all.

    The output grid is the forward leg's, because that is the grid the user
    asked for.  A reverse point matching nothing is *reported*, not merged in,
    which would otherwise produce duplicated abscissae and an uneven grid.

    Args:
        forward: the leg run from the input geometry.
        reverse: the leg run back from the forward leg's final structure.
        tol_deg: largest angular separation still considered the same grid
            point.  Half a step is the sensible value.

    Returns:
        A :class:`MergeResult`.

    Raises:
        ValidationError: two forward points claim the same reverse point,
            which means the grid is degenerate and the intended pairing is
            genuinely ambiguous.
    """
    frames = np.array(forward.frames_nm, dtype=float, copy=True)
    energies = np.array(forward.energy_kj, dtype=float, copy=True)
    source = ["forward"] * len(forward)
    claimed: dict[int, int] = {}
    hysteresis = 0.0
    worst_angle = float("nan")
    unpaired_forward: list[float] = []

    for i, phi in enumerate(forward.phi_deg):
        if len(reverse) == 0:
            unpaired_forward.append(float(phi))
            continue
        separation = angular_distance_deg(reverse.phi_deg, phi)
        j = int(np.argmin(separation))
        if separation[j] > tol_deg:
            unpaired_forward.append(float(phi))
            continue
        if j in claimed:
            raise ValidationError(
                f"forward points at {forward.phi_deg[claimed[j]]:.2f} and {phi:.2f} degrees "
                f"both pair with the reverse point at {reverse.phi_deg[j]:.2f} degrees "
                f"within the {tol_deg:g} degree tolerance. The grid is degenerate, so "
                "which pairing was intended is ambiguous; re-run with a coarser --step "
                "or a tighter tolerance."
            )
        claimed[j] = i

        gap = abs(float(forward.energy_kj[i] - reverse.energy_kj[j]))
        if gap > hysteresis:
            hysteresis = gap
            worst_angle = float(phi)
        if reverse.energy_kj[j] < forward.energy_kj[i]:
            energies[i] = reverse.energy_kj[j]
            frames[i] = reverse.frames_nm[j]
            source[i] = "reverse"

    unpaired_reverse = [
        float(reverse.phi_deg[j]) for j in range(len(reverse)) if j not in claimed
    ]
    return MergeResult(
        frames_nm=frames,
        energy_kj=energies,
        phi_deg=np.array(forward.phi_deg, dtype=float, copy=True),
        source=source,
        max_hysteresis_kj=hysteresis,
        worst_angle_deg=worst_angle,
        unpaired_forward_deg=unpaired_forward,
        unpaired_reverse_deg=unpaired_reverse,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_qm_scan.py -q`
Expected: PASS (11 tests).

- [ ] **Step 5: Commit**

```bash
black . && git add -A
git commit -m "feat(qm): merge bidirectional legs by measured angle, with hysteresis"
```

---

### Task 7: Conformational-hop detection

**Files:**
- Modify: `ffrefine/qm/scan.py`
- Test: `tests/test_qm_scan.py`

**Interfaces:**
- Consumes: `ffrefine.geometry.kabsch_rmsd`; `ffrefine.io.structure.guess_element`.
- Produces: `scan.Hop` dataclass (`index`, `phi_deg`, `rmsd_nm`, `threshold_nm`); `scan.find_hops(frames_nm, heavy_indices, phi_deg, k_mad=5.0, cyclic=True) -> list[Hop]`.

**Why:** Adjacent frames differ by the scanned dihedral **by construction**, so adjacent-frame RMSD is never zero and an absolute threshold is meaningless. A genuine hop is a *spike* against the typical step, so the test must be a robust outlier test. MAD is scaled by 1.4826 to make `median + k*MAD` comparable to a standard deviation for normally distributed steps.

- [ ] **Step 1: Write the failing test**

```python
from ffrefine.qm.scan import Hop, find_hops


def _smooth_rotation(n: int = 12, natoms: int = 8) -> np.ndarray:
    """Frames whose adjacent RMSDs are near-uniform: a clean scan, no hops."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(natoms, 3)) * 0.1
    out = []
    for step in range(n):
        frame = base.copy()
        frame[natoms // 2 :, 0] += 0.01 * step  # steady drift, same size each step
        out.append(frame)
    return np.stack(out)


def test_find_hops_is_silent_on_a_smooth_scan():
    frames = _smooth_rotation()
    phi = np.linspace(-180.0, 150.0, len(frames))
    assert find_hops(frames, list(range(frames.shape[1])), phi, cyclic=False) == []


def test_find_hops_flags_a_planted_spike():
    frames = _smooth_rotation()
    frames[7:] += 0.5  # a sudden rearrangement between frames 6 and 7
    phi = np.linspace(-180.0, 150.0, len(frames))

    hops = find_hops(frames, list(range(frames.shape[1])), phi, cyclic=False)

    assert [h.index for h in hops] == [7]
    assert hops[0].phi_deg == pytest.approx(phi[7])
    assert hops[0].rmsd_nm > hops[0].threshold_nm


def test_find_hops_ignores_hydrogens():
    """Spec 4.3 says heavy-atom RMSD; a spinning methyl is not a conformational hop."""
    frames = _smooth_rotation(natoms=6)
    heavy = [0, 1, 2]
    frames[5:, 3:] += 2.0  # only the 'hydrogens' move
    phi = np.linspace(-180.0, 150.0, len(frames))

    assert find_hops(frames, heavy, phi, cyclic=False) == []


def test_find_hops_checks_the_wrap_pair_when_cyclic():
    """A full-turn scan's last and first frames are adjacent in angle."""
    frames = _smooth_rotation()
    frames[-1] += 0.5
    phi = np.linspace(-180.0, 150.0, len(frames))

    indices = [h.index for h in find_hops(frames, list(range(frames.shape[1])), phi, cyclic=True)]

    assert 0 in indices or len(frames) - 1 in indices


def test_find_hops_survives_identical_frames():
    """MAD is zero here; the threshold must not become nan or divide by zero."""
    frames = np.zeros((5, 4, 3))
    phi = np.linspace(-180.0, 108.0, 5)
    assert find_hops(frames, [0, 1, 2, 3], phi, cyclic=False) == []


def test_find_hops_needs_enough_points_to_judge_an_outlier():
    """Two frames give one gap; calling it an outlier against itself is nonsense."""
    frames = np.zeros((2, 4, 3))
    frames[1] += 5.0
    assert find_hops(frames, [0, 1, 2, 3], np.array([0.0, 90.0]), cyclic=False) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_qm_scan.py -q -k hops`
Expected: FAIL with `ImportError: cannot import name 'find_hops'`.

- [ ] **Step 3: Implement**

Add to `ffrefine/qm/scan.py` (import `kabsch_rmsd`):

```python
MAD_TO_SIGMA = 1.4826
"""Scales a median absolute deviation to a standard-deviation-equivalent for normal data."""

MIN_POINTS_FOR_OUTLIER = 4
"""Fewest adjacent gaps from which calling one an outlier is meaningful."""


@dataclass
class Hop:
    """A suspected conformational hop between two adjacent scan frames.

    Attributes:
        index: the later frame of the pair (so the hop is between
            ``index - 1`` and ``index``, or between the last frame and frame 0
            when ``index`` is 0 and the scan is cyclic).
        phi_deg: that frame's measured dihedral.
        rmsd_nm: the aligned heavy-atom RMSD across the pair.
        threshold_nm: the outlier threshold it exceeded.
    """

    index: int
    phi_deg: float
    rmsd_nm: float
    threshold_nm: float


def find_hops(
    frames_nm: np.ndarray,
    heavy_indices: list[int],
    phi_deg: np.ndarray,
    k_mad: float = 5.0,
    cyclic: bool = True,
) -> list[Hop]:
    """Flag adjacent frames that look like a conformational hop (spec 4.3).

    **Why this is an outlier test and not a threshold.**  In a relaxed torsion
    scan, adjacent frames differ by the scanned dihedral *by construction*, so
    the aligned heavy-atom RMSD between them is never zero and no absolute
    cutoff carries meaning across molecules -- a large flexible molecule's
    normal step dwarfs a small rigid one's genuine hop.  What a hop actually
    looks like is a *spike* against the run of typical steps, so the statistic
    has to be relative to those steps and robust to the very outliers it is
    hunting.  Hence ``median + k * 1.4826 * MAD``.

    Hydrogens are excluded because a rotating methyl produces a large RMSD
    every step without any conformational change; spec 4.3 says heavy-atom
    RMSD for exactly this reason.

    The tool reports and continues (spec 4.3): a hop makes the profile
    unfittable by a smooth periodic function, but whether to re-scan, freeze
    another dihedral or accept it is the user's call, not this function's.

    Args:
        frames_nm: shape (nframes, natoms, 3).
        heavy_indices: 0-based indices of the non-hydrogen atoms.
        phi_deg: measured angles, shape (nframes,), for reporting.
        k_mad: how many scaled MADs above the median counts as a hop.
        cyclic: also compare the last frame with the first.  Correct for the
            default full-turn grid, where they are one step apart in angle;
            pass False for a partial scan, where they are not adjacent at all.

    Returns:
        The flagged hops, in frame order.  Empty when there are too few gaps
        to judge an outlier, or when every gap is identical.
    """
    frames = np.asarray(frames_nm, float)
    if len(frames) < 2 or not heavy_indices:
        return []

    pairs = [(i - 1, i) for i in range(1, len(frames))]
    if cyclic and len(frames) > 2:
        pairs.append((len(frames) - 1, 0))

    rmsds = np.array(
        [kabsch_rmsd(frames[a][heavy_indices], frames[b][heavy_indices]) for a, b in pairs]
    )
    if len(rmsds) < MIN_POINTS_FOR_OUTLIER:
        return []

    median = float(np.median(rmsds))
    mad = float(np.median(np.abs(rmsds - median)))
    if mad <= 0.0:
        return []
    threshold = median + k_mad * MAD_TO_SIGMA * mad

    hops = []
    for (_, later), rmsd in zip(pairs, rmsds):
        if rmsd > threshold:
            hops.append(
                Hop(
                    index=int(later),
                    phi_deg=float(phi_deg[later]),
                    rmsd_nm=float(rmsd),
                    threshold_nm=float(threshold),
                )
            )
    return sorted(hops, key=lambda h: h.index)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_qm_scan.py -q`
Expected: PASS (17 tests).

- [ ] **Step 5: Commit**

```bash
black . && git add -A
git commit -m "feat(qm): flag conformational hops by robust RMSD outlier test"
```

---

### Task 8: Hydrogen-bond formation detection

**Files:**
- Modify: `ffrefine/qm/scan.py`
- Test: `tests/test_qm_scan.py`

**Interfaces:**
- Consumes: `ffrefine.io.topology.Topology`; `ffrefine.io.structure.guess_element`; **`ffrefine.io.validate.element_from_mass`, promoted from the existing private `_element_from_mass` as part of this task.**
- Produces: `scan.HBond` (NamedTuple `donor`, `hydrogen`, `acceptor`, all 1-based); `scan.hbond_donors_and_acceptors(topology) -> tuple[list[tuple[int,int]], list[int]]`; `scan.find_hydrogen_bonds(frame_nm, donors, acceptors, topology, max_h_acceptor_nm=0.25, min_angle_deg=120.0) -> set[HBond]`; `scan.hbond_changes(frames_nm, topology, phi_deg, ...) -> list[HBondChange]`.

**Element determination — do not shortcut this.** `guess_element(name, mass)` **ignores** its
`mass` argument and infers purely from the atom name (its docstring says so explicitly, after
the P2-13 fix). That is not good enough here: a topology atom named `NA` matches the
two-letter table and comes back as sodium when it is a nitrogen, which would silently drop a
real donor or invent a false acceptor. Topologies carry masses, and
`validate_structure_topology` already treats the mass as the authority, cross-checking the
name-derived answer against it. Follow that precedent: **prefer the mass, fall back to the
name only when the topology has no mass.** Promote `io/validate.py`'s `_element_from_mass` to
a public `element_from_mass` (keeping `_element_from_mass = element_from_mass` if any internal
caller reads better that way) so this module can use it without reaching into a private name.

**Why:** The scan is *relaxed*, so everything but the scanned dihedral rearranges freely. In any molecule with a donor and an acceptor, a hydroxyl can rotate mid-scan into an intramolecular hydrogen bond — 1,2-ethanediol scanning O–C–C–O is the canonical case, where gauche gains ~10–20 kJ/mol that anti does not have. That well is **not** the torsional potential; fitting through it bakes an electrostatic interaction into the dihedral term, which the nonbonded terms then count a second time. Butane has no donors or acceptors and correctly reports none, which is why these tests are synthetic.

- [ ] **Step 1: Write the failing test**

```python
from ffrefine.qm.scan import (
    HBond,
    find_hydrogen_bonds,
    hbond_changes,
    hbond_donors_and_acceptors,
)


def _glycol_topology():
    """A minimal 1,2-ethanediol-like topology: O1-C2-C3-O4, with H5 on O1, H6 on O4."""
    from pathlib import Path

    from ffrefine.io.topology import Atom, Topology

    atoms = [
        Atom(1, "OG311", 1, "LIG", "O1", 1, -0.65, 15.999),
        Atom(2, "CG321", 1, "LIG", "C2", 1, 0.05, 12.011),
        Atom(3, "CG321", 1, "LIG", "C3", 1, 0.05, 12.011),
        Atom(4, "OG311", 1, "LIG", "O4", 1, -0.65, 15.999),
        Atom(5, "HGP1", 1, "LIG", "H5", 1, 0.42, 1.008),
        Atom(6, "HGP1", 1, "LIG", "H6", 1, 0.42, 1.008),
    ]
    return Topology(
        name="GLY", nrexcl=3, atoms=atoms,
        bonds=[(1, 2), (2, 3), (3, 4), (1, 5), (4, 6)],
        pairs=[], angles=[], propers=[], impropers=[], defaults=None,
        raw_lines=[], source=Path("synthetic.itp"),
    )


def test_donors_and_acceptors_come_from_the_topology_bonds():
    donors, acceptors = hbond_donors_and_acceptors(_glycol_topology())
    assert set(donors) == {(1, 5), (4, 6)}   # (heavy donor, its hydrogen)
    assert set(acceptors) == {1, 4}          # both oxygens


def test_element_comes_from_the_mass_not_the_ambiguous_name():
    """An atom named 'NA' with nitrogen's mass is nitrogen, not sodium.

    guess_element reads the name only and would call this sodium, silently
    dropping a donor or inventing an acceptor.
    """
    top = _glycol_topology()
    top.atoms[0].name = "NA"
    top.atoms[0].mass = 14.007

    donors, acceptors = hbond_donors_and_acceptors(top)

    assert 1 in acceptors          # still recognised as N, via the mass
    assert (1, 5) in donors


def test_butane_has_no_donors_or_acceptors(butane_itp):
    """The only committed fixture cannot exercise this path -- assert that, don't assume it."""
    from ffrefine.io.topology import read_topology

    donors, acceptors = hbond_donors_and_acceptors(read_topology(butane_itp))
    assert donors == [] and acceptors == []


def test_finds_a_hydrogen_bond_when_the_hydroxyl_points_at_the_far_oxygen():
    """O4-H6 aimed at O1 at 0.19 nm and 170 degrees is a textbook H-bond."""
    top = _glycol_topology()
    donors, acceptors = hbond_donors_and_acceptors(top)
    frame = np.array([
        [0.000, 0.000, 0.0],   # O1
        [0.143, 0.000, 0.0],   # C2
        [0.200, 0.140, 0.0],   # C3
        [0.350, 0.140, 0.0],   # O4
        [-0.040, 0.090, 0.0],  # H5 on O1, pointing away
        [0.190, 0.010, 0.0],   # H6 on O4, aimed back at O1
    ])
    found = find_hydrogen_bonds(frame, donors, acceptors, top)
    assert HBond(donor=4, hydrogen=6, acceptor=1) in found


def test_no_hydrogen_bond_when_the_hydroxyl_is_rotated_away():
    top = _glycol_topology()
    donors, acceptors = hbond_donors_and_acceptors(top)
    frame = np.array([
        [0.000, 0.000, 0.0],
        [0.143, 0.000, 0.0],
        [0.200, 0.140, 0.0],
        [0.350, 0.140, 0.0],
        [-0.040, 0.090, 0.0],
        [0.440, 0.180, 0.0],   # H6 now points away from O1
    ])
    assert find_hydrogen_bonds(frame, donors, acceptors, top) == set()


def test_a_hydroxyl_does_not_hydrogen_bond_to_its_own_oxygen():
    """O4-H6...O4 is 1-2 bonded, not a hydrogen bond; the exclusion must hold."""
    top = _glycol_topology()
    donors, acceptors = hbond_donors_and_acceptors(top)
    frame = np.array([
        [0.000, 0.000, 0.0], [0.143, 0.000, 0.0], [0.200, 0.140, 0.0],
        [0.350, 0.140, 0.0], [-0.040, 0.090, 0.0], [0.446, 0.140, 0.0],
    ])
    assert all(b.acceptor != 4 for b in find_hydrogen_bonds(frame, donors, acceptors, top))


def test_hbond_changes_reports_formation_between_frames():
    """The event a user needs told about: a bond that was not there, now is."""
    top = _glycol_topology()
    away = np.array([
        [0.000, 0.000, 0.0], [0.143, 0.000, 0.0], [0.200, 0.140, 0.0],
        [0.350, 0.140, 0.0], [-0.040, 0.090, 0.0], [0.440, 0.180, 0.0],
    ])
    bonded = away.copy()
    bonded[5] = [0.190, 0.010, 0.0]

    changes = hbond_changes(np.stack([away, bonded]), top, np.array([180.0, 60.0]))

    assert len(changes) == 1
    assert changes[0].index == 1
    assert changes[0].formed == {HBond(donor=4, hydrogen=6, acceptor=1)}
    assert changes[0].broken == set()
    assert changes[0].phi_deg == pytest.approx(60.0)


def test_hbond_changes_is_silent_on_butane(butane_pdb, butane_itp):
    """No donors, no acceptors, no findings -- and no crash on the empty case."""
    from ffrefine.io.structure import read_pdb
    from ffrefine.io.topology import read_topology

    s = read_pdb(butane_pdb)
    frames = np.stack([s.coords, s.coords + 0.01])
    assert hbond_changes(frames, read_topology(butane_itp), np.array([0.0, 15.0])) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_qm_scan.py -q -k hbond or hydrogen`
Expected: FAIL with `ImportError: cannot import name 'HBond'`.

- [ ] **Step 3: Implement**

Add to `ffrefine/qm/scan.py`:

```python
from typing import NamedTuple

from ..io.structure import guess_element
from ..io.topology import Topology
from ..io.validate import element_from_mass

HBOND_ELEMENTS = frozenset({"N", "O", "F"})
"""Elements treated as hydrogen-bond donors (when carrying an H) and acceptors."""

HBOND_MAX_H_ACCEPTOR_NM = 0.25
"""Longest H...acceptor separation still counted as a hydrogen bond."""

HBOND_MIN_ANGLE_DEG = 120.0
"""Smallest donor-H...acceptor angle still counted as a hydrogen bond."""


class HBond(NamedTuple):
    """One intramolecular hydrogen bond, by 1-based topology atom index."""

    donor: int
    hydrogen: int
    acceptor: int


@dataclass
class HBondChange:
    """Hydrogen bonds that appeared or vanished between adjacent frames.

    Attributes:
        index: the later frame of the pair.
        phi_deg: that frame's measured dihedral.
        formed: bonds present at ``index`` but not at ``index - 1``.
        broken: bonds present at ``index - 1`` but not at ``index``.
    """

    index: int
    phi_deg: float
    formed: set[HBond]
    broken: set[HBond]


def _element_of(topology: Topology, nr: int) -> str:
    """Element of a topology atom, preferring its mass over its name.

    ``guess_element`` reads the name only, and names are ambiguous in exactly
    the way that matters here: ``NA`` matches the two-letter element table and
    returns sodium for what is often a nitrogen, which would drop a real donor
    or invent an acceptor that does not exist.  The mass is unambiguous, and
    :func:`ffrefine.io.validate.validate_structure_topology` already treats it
    as the authority, so this follows that precedent and falls back to the name
    only when the topology carries no mass.
    """
    atom = topology.atoms[nr - 1]
    if atom.mass is not None:
        from_mass = element_from_mass(atom.mass)
        if from_mass:
            return from_mass
    return guess_element(atom.name) or ""


def hbond_donors_and_acceptors(
    topology: Topology,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Find hydrogen-bond donors and acceptors from the topology's bonds.

    Bonds come from the topology rather than from interatomic distances,
    because distance-guessed connectivity would mistake a close contact for a
    covalent bond in exactly the crowded geometries this detector cares about.

    Args:
        topology: the parsed molecule.

    Returns:
        ``(donors, acceptors)`` where each donor is ``(heavy_atom_nr,
        hydrogen_nr)`` and acceptors are heavy-atom numbers; all 1-based.
    """
    donors: list[tuple[int, int]] = []
    acceptors: list[int] = []
    for atom in topology.atoms:
        element = _element_of(topology, atom.nr)
        if element in HBOND_ELEMENTS:
            acceptors.append(atom.nr)
        elif element == "H":
            for neighbour in topology.bonded_neighbours(atom.nr):
                if _element_of(topology, neighbour) in HBOND_ELEMENTS:
                    donors.append((neighbour, atom.nr))
    return donors, acceptors


def _excluded_acceptors(topology: Topology, donor: int) -> set[int]:
    """Acceptors too close in the bond graph to count (1-2 and 1-3 neighbours)."""
    out = {donor}
    for first in topology.bonded_neighbours(donor):
        out.add(first)
        out.update(topology.bonded_neighbours(first))
    return out


def find_hydrogen_bonds(
    frame_nm: np.ndarray,
    donors: list[tuple[int, int]],
    acceptors: list[int],
    topology: Topology,
    max_h_acceptor_nm: float = HBOND_MAX_H_ACCEPTOR_NM,
    min_angle_deg: float = HBOND_MIN_ANGLE_DEG,
) -> set[HBond]:
    """Hydrogen bonds present in one geometry, by the usual geometric criterion.

    A bond counts when the H...acceptor separation is under
    ``max_h_acceptor_nm`` and the donor-H...acceptor angle is above
    ``min_angle_deg``.  The angle matters as much as the distance: a hydrogen
    merely passing near an oxygen sideways is not donating to it.

    1-2 and 1-3 bonded acceptors are excluded -- a hydroxyl cannot hydrogen
    bond to its own oxygen or to the carbon's other substituent, and the
    geometry would otherwise trip the distance test constantly.

    Args:
        frame_nm: shape (natoms, 3) in nanometres.
        donors: ``(heavy, hydrogen)`` pairs from
            :func:`hbond_donors_and_acceptors`.
        acceptors: acceptor atom numbers from the same.
        topology: supplies the bond graph for the exclusions.
        max_h_acceptor_nm: distance cutoff.
        min_angle_deg: angle cutoff.

    Returns:
        The hydrogen bonds found, as 1-based index triples.
    """
    frame = np.asarray(frame_nm, float)
    found: set[HBond] = set()
    for heavy, hydrogen in donors:
        excluded = _excluded_acceptors(topology, heavy)
        h_pos = frame[hydrogen - 1]
        d_pos = frame[heavy - 1]
        for acceptor in acceptors:
            if acceptor in excluded:
                continue
            a_pos = frame[acceptor - 1]
            if float(np.linalg.norm(a_pos - h_pos)) > max_h_acceptor_nm:
                continue
            v1 = d_pos - h_pos
            v2 = a_pos - h_pos
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 == 0.0 or n2 == 0.0:
                continue
            cosine = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
            if np.degrees(np.arccos(cosine)) >= min_angle_deg:
                found.add(HBond(donor=heavy, hydrogen=hydrogen, acceptor=acceptor))
    return found


def hbond_changes(
    frames_nm: np.ndarray,
    topology: Topology,
    phi_deg: np.ndarray,
    max_h_acceptor_nm: float = HBOND_MAX_H_ACCEPTOR_NM,
    min_angle_deg: float = HBOND_MIN_ANGLE_DEG,
) -> list[HBondChange]:
    """Report intramolecular hydrogen bonds forming or breaking across a scan.

    Why this belongs in a torsion tool: the scan is *relaxed*, so at every grid
    point everything except the scanned dihedral is free to rearrange.  In a
    molecule carrying both a donor and an acceptor, a hydroxyl can rotate into
    an internal hydrogen bond partway through -- 1,2-ethanediol scanning
    O-C-C-O is the standard case, where the gauche region gains roughly 10-20
    kJ/mol of stabilisation the anti region has no access to.  That well is not
    the torsional potential.  Fitting a cosine series through it writes an
    electrostatic interaction into the dihedral term, and the nonbonded terms
    then count the same interaction again in every simulation that uses the
    result.

    Per spec 4.3 this reports and continues: re-scanning with
    ``--freeze-dihedral``, choosing another starting conformer, or knowingly
    accepting the profile are all the user's decisions.

    Args:
        frames_nm: shape (nframes, natoms, 3).
        topology: supplies elements and the bond graph.
        phi_deg: measured angles, for reporting.
        max_h_acceptor_nm: distance cutoff.
        min_angle_deg: angle cutoff.

    Returns:
        One :class:`HBondChange` per adjacent pair that differs.  Empty for a
        molecule with no donors or acceptors, which includes the butane fixture.
    """
    donors, acceptors = hbond_donors_and_acceptors(topology)
    if not donors or not acceptors:
        return []

    per_frame = [
        find_hydrogen_bonds(frame, donors, acceptors, topology, max_h_acceptor_nm, min_angle_deg)
        for frame in np.asarray(frames_nm, float)
    ]
    changes = []
    for i in range(1, len(per_frame)):
        formed = per_frame[i] - per_frame[i - 1]
        broken = per_frame[i - 1] - per_frame[i]
        if formed or broken:
            changes.append(
                HBondChange(
                    index=i, phi_deg=float(phi_deg[i]), formed=formed, broken=broken
                )
            )
    return changes
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_qm_scan.py -q`
Expected: PASS (25 tests).

- [ ] **Step 5: Commit**

```bash
black . && git add -A
git commit -m "feat(qm): detect intramolecular hydrogen bonds forming during a scan"
```

---

### Task 9: Derive the molecular charge from the topology

**Files:**
- Modify: `ffrefine/qm/xtb.py`
- Test: `tests/test_qm_xtb.py`

**Interfaces:**
- Consumes: `ffrefine.io.topology.Topology`, `ffrefine.errors.ValidationError`.
- Produces: `xtb.derive_charge(topology, override=None, tol=0.05) -> int`.

**Why:** Spec 4.3 runs xtb with `--chrg Q` but never says where Q comes from. The topology already knows: `[ atoms ]` charges sum to the molecular charge. Requiring the user to retype it invites a silently wrong answer on an ionic molecule. Multiplicity genuinely cannot be derived (an electron-count parity check cannot tell a triplet from a singlet diradical), so `--uhf` stays an explicit flag defaulting to 0.

- [ ] **Step 1: Write the failing test**

```python
from ffrefine.qm.xtb import derive_charge


def test_derive_charge_reads_the_topology_sum(butane_itp):
    from ffrefine.io.topology import read_topology

    assert derive_charge(read_topology(butane_itp)) == 0


def test_derive_charge_rounds_a_near_integer_sum(butane_itp):
    """CGenFF charges print to 3-4 decimals, so the sum is rarely exact."""
    from ffrefine.io.topology import read_topology

    top = read_topology(butane_itp)
    top.atoms[0].charge += 0.999
    assert derive_charge(top) == 1


def test_derive_charge_refuses_a_sum_that_is_not_near_an_integer(butane_itp):
    from ffrefine.io.topology import read_topology

    top = read_topology(butane_itp)
    top.atoms[0].charge += 0.4
    with pytest.raises(ValidationError, match="0.4"):
        derive_charge(top)


def test_derive_charge_override_wins(butane_itp):
    from ffrefine.io.topology import read_topology

    assert derive_charge(read_topology(butane_itp), override=-1) == -1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_qm_xtb.py -q -k derive_charge`
Expected: FAIL with `ImportError: cannot import name 'derive_charge'`.

- [ ] **Step 3: Implement**

Add to `ffrefine/qm/xtb.py`:

```python
CHARGE_TOLERANCE = 0.05
"""How far the topology's charge sum may sit from an integer before it is refused."""


def derive_charge(
    topology: "Topology", override: int | None = None, tol: float = CHARGE_TOLERANCE
) -> int:
    """Determine the total molecular charge for xtb's ``--chrg``.

    Spec 4.3 requires the flag but does not say where the value comes from.
    The topology knows: ``[ atoms ]`` charges sum to the molecular charge by
    construction.  Deriving it beats asking, because a user who forgets
    ``--charge`` on an ionic molecule gets a silently wrong electronic
    structure and therefore a silently wrong torsion profile -- there is no
    error, just a bad answer.

    Force field charges print to three or four decimals, so the sum is rarely
    exact; it is rounded.  A sum far from any integer means the topology is
    incomplete or corrupt, and guessing a charge for it would be worse than
    stopping.

    Multiplicity is deliberately NOT derived.  Electron-count parity cannot
    distinguish a triplet from a singlet diradical, so ``--uhf`` stays explicit
    and defaults to 0.

    Args:
        topology: the parsed molecule.
        override: an explicit ``--charge`` from the user; wins outright.
        tol: largest allowed departure from an integer.

    Returns:
        The integer charge.

    Raises:
        ValidationError: the charge sum is not within ``tol`` of an integer.
    """
    if override is not None:
        return int(override)
    total = sum(atom.charge for atom in topology.atoms)
    nearest = round(total)
    if abs(total - nearest) > tol:
        raise ValidationError(
            f"the topology's [ atoms ] charges sum to {total:.4f}, which is "
            f"{abs(total - nearest):.4f} from the nearest integer ({nearest}) and outside "
            f"the {tol} tolerance. That usually means the topology is incomplete. Pass "
            "--charge to state the molecular charge explicitly."
        )
    return int(nearest)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_qm_xtb.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
black . && git add -A
git commit -m "feat(qm): derive the molecular charge from the topology for xtb"
```

---

### Task 10: Wire `cmd_scan` into the CLI

**Files:**
- Modify: `ffrefine/cli.py` (replace the `scan` stub at line ~300, `cmd_not_implemented` dispatch, `_STAGE_OUTPUTS`, `_SCAN_REMEDY`)
- Test: `tests/test_cli_scan.py`

**Interfaces:**
- Consumes: every prior task; `stages.stage_dir`, `hash_inputs`, `write_manifest`, `stage_is_current`, `hash_array`; `cli._prepare`, `_binaries`, `_stage_outputs`, `_warn`, `_scan_path`.
- Produces: `cli.cmd_scan(args) -> int`; `01_qm_scan/scan.json`.

**Why:** This is the task that closes the pipeline. `scan.json` must match the schema `cmd_mmscan`/`cmd_fit`/`cmd_validate` already consume, plus `quad` and `provenance`.

- [ ] **Step 1: Write the failing test**

```python
import json

import numpy as np
import pytest

from ffrefine.cli import main


def test_scan_is_no_longer_reported_as_unimplemented(butane_pdb, butane_itp, tmp_path, capsys):
    rc = main([
        "scan", "-s", str(butane_pdb), "-p", str(butane_itp),
        "--dihedral", "1,5,8,11", "--outdir", str(tmp_path / "out"), "--dry-run",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "NOT IMPLEMENTED" not in out.upper()


def test_scan_dry_run_writes_nothing(butane_pdb, butane_itp, tmp_path):
    outdir = tmp_path / "out"
    main([
        "scan", "-s", str(butane_pdb), "-p", str(butane_itp),
        "--dihedral", "1,5,8,11", "--outdir", str(outdir), "--dry-run",
    ])
    assert not outdir.exists()


def test_scan_rejects_a_step_that_does_not_divide_360(butane_pdb, butane_itp, tmp_path, capsys):
    rc = main([
        "scan", "-s", str(butane_pdb), "-p", str(butane_itp),
        "--dihedral", "1,5,8,11", "--outdir", str(tmp_path / "o"), "--step", "7",
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert "Traceback" not in err


@pytest.mark.needs_xtb
def test_scan_writes_a_scan_json_the_fit_pipeline_accepts(
    butane_pdb, butane_itp, tmp_path, xtb_bin
):
    """End to end on real xtb, at 4 points to stay near 30 s (24 would be ~3 min)."""
    outdir = tmp_path / "out"
    rc = main([
        "scan", "-s", str(butane_pdb), "-p", str(butane_itp),
        "--dihedral", "1,5,8,11", "--outdir", str(outdir),
        "--step", "90", "--no-bidirectional", "--nthreads", "4",
    ])
    assert rc == 0

    payload = json.loads((outdir / "01_qm_scan" / "scan.json").read_text())
    frames = np.array(payload["frames_nm"])
    assert frames.shape == (4, 14, 3)
    assert len(payload["energy_kj"]) == 4
    assert len(payload["phi_deg"]) == 4
    assert payload["quad"] == [1, 5, 8, 11]
    # Coordinates are nm, so every butane atom sits within a couple of nm of the origin.
    assert np.abs(frames).max() < 5.0

    prov = payload["provenance"]
    assert prov["charge"] == 0
    assert prov["uhf"] == 0
    assert prov["step_deg"] == 90.0
    assert prov["bidirectional"] is False
    assert "6.7.1" in prov["xtb_version"] or prov["xtb_version"]
    assert (outdir / "01_qm_scan" / "scan.pdb").exists()


@pytest.mark.needs_xtb
def test_scan_output_feeds_mmscan_without_hand_editing(
    butane_pdb, butane_itp, tmp_path, xtb_bin, gmx_bin
):
    """The point of the milestone: no hand-written scan.json anywhere."""
    outdir = tmp_path / "out"
    assert main([
        "scan", "-s", str(butane_pdb), "-p", str(butane_itp),
        "--dihedral", "1,5,8,11", "--outdir", str(outdir),
        "--step", "90", "--no-bidirectional", "--nthreads", "4",
    ]) == 0
    assert main([
        "mmscan", "-s", str(butane_pdb), "-p", str(butane_itp),
        "--dihedral", "1,5,8,11", "--outdir", str(outdir), "--nthreads", "2",
    ]) == 0

    energies = json.loads((outdir / "02_mm_scan" / "energies.json").read_text())
    assert energies["quad"] == [1, 5, 8, 11]
    assert len(energies["phi_deg"]) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli_scan.py -q`
Expected: FAIL — the stub prints `NOT IMPLEMENTED` and returns 2.

- [ ] **Step 3: Implement**

In `ffrefine/cli.py`, extend the `scan` subparser:

```python
    scan = sub.add_parser("scan", help="QM relaxed torsion scan (xtb)")
    _add_common(scan)
    _add_torsion(scan)
    scan.add_argument("--step", type=float, help="scan step in degrees (default: 15)")
    scan.add_argument("--charge", type=int, help="molecular charge (default: from the topology)")
    scan.add_argument("--uhf", type=int, default=0, help="unpaired electrons (default: 0)")
    scan.add_argument("--gfn", type=int, default=2, help="xtb GFN parameterisation (default: 2)")
    scan.add_argument(
        "--bidirectional", dest="bidirectional", action="store_true", default=True,
        help="scan forward then back, taking the lower energy (default)",
    )
    scan.add_argument(
        "--no-bidirectional", dest="bidirectional", action="store_false",
        help="scan in one direction only",
    )
    scan.add_argument(
        "--hysteresis-tol", type=float, default=4.0,
        help="warn when the two directions differ by more than this, kJ/mol (default: 4)",
    )
    scan.add_argument(
        "--freeze-dihedral", action="append", type=_dihedral_arg, default=None,
        metavar="i,j,k,l", help="hold another dihedral fixed; repeatable",
    )
```

Add `"qm_scan": ["scan.json"]` to `_STAGE_OUTPUTS`, and rewrite `_SCAN_REMEDY` to stop calling the stage unimplemented:

```python
_SCAN_REMEDY = (
    "Run 'ffrefine scan' first to produce it, or supply the file yourself: a JSON object "
    "with 'energy_kj' (QM energies in kJ/mol, one per frame) and 'frames_nm' (an "
    "(nframes, natoms, 3) array of coordinates in nanometres, in topology atom order)."
)
```

Add `cmd_scan`, and dispatch `scan` to it instead of `cmd_not_implemented`:

```python
def cmd_scan(args: argparse.Namespace) -> int:
    """Run the xtb relaxed torsion scan and write ``01_qm_scan/scan.json``.

    This is the stage that closes the pipeline: before it existed, the README
    told users to hand-write ``scan.json``.

    Bidirectional scanning runs a second leg backwards from the first leg's
    final geometry and takes the lower energy at each grid point (spec 4.3).
    The two legs are paired by *measured angle*, never by array index -- see
    :func:`ffrefine.qm.scan.merge_bidirectional` for why that distinction
    decides whether a torsion barrier survives.
    """
    step = args.step if args.step is not None else 15.0
    if args.dry_run:
        structure, topology, quad = _load_inputs(args)
        start, end, npoints = scan_grid(step)
        legs = 2 if args.bidirectional else 1
        print(
            f"would scan torsion {quad} of {topology.name} ({structure.natoms} atoms) "
            f"over {npoints} points from {start} to {end} degrees, {legs} leg(s), "
            f"into {args.outdir}"
        )
        return 0

    structure, topology, quad = _prepare(args)
    start, end, npoints = scan_grid(step)
    charge = derive_charge(topology, args.charge)
    freeze = tuple(args.freeze_dihedral or ())

    settings = {
        "quad": list(quad),
        "step_deg": step,
        "charge": charge,
        "uhf": args.uhf,
        "gfn": args.gfn,
        "bidirectional": bool(args.bidirectional),
        "freeze": [list(f) for f in freeze],
    }
    digest = hash_inputs([args.topology, args.structure], settings)
    if args.resume and stage_is_current(
        args.outdir, "qm_scan", digest, _stage_outputs(args, "qm_scan")
    ):
        print(f"scan: skipping, {_scan_path(args)} is current (--resume)")
        return 0

    binaries = _binaries(args)
    workdir = stage_dir(args.outdir, "qm_scan")

    forward_frames, forward_energy = run_scan(
        binaries, structure, quad, workdir / "forward",
        start_deg=start, end_deg=end, npoints=npoints, charge=charge, uhf=args.uhf,
        gfn=args.gfn, nthreads=args.nthreads, freeze=freeze, log_stem="xtb_forward",
    )
    forward = ScanLeg(
        forward_frames, forward_energy, measure_angles(forward_frames, quad), "forward"
    )

    merged = None
    if args.bidirectional:
        reverse_frames, reverse_energy = run_scan(
            binaries, structure, quad, workdir / "reverse",
            start_deg=end, end_deg=start, npoints=npoints, charge=charge, uhf=args.uhf,
            gfn=args.gfn, nthreads=args.nthreads, freeze=freeze,
            coords_nm=forward_frames[-1], log_stem="xtb_reverse",
        )
        reverse = ScanLeg(
            reverse_frames, reverse_energy, measure_angles(reverse_frames, quad), "reverse"
        )
        merged = merge_bidirectional(forward, reverse, tol_deg=step / 2.0)
        frames_nm, energy_kj, phi_deg = merged.frames_nm, merged.energy_kj, merged.phi_deg
        if merged.max_hysteresis_kj > args.hysteresis_tol:
            _warn(
                f"forward and reverse scans differ by up to "
                f"{merged.max_hysteresis_kj:.2f} kJ/mol (worst at "
                f"{merged.worst_angle_deg:.1f} degrees), above the "
                f"{args.hysteresis_tol:g} kJ/mol tolerance. The scan is not reversible "
                "here, so the profile depends on the direction it was walked; treat the "
                "barrier heights with suspicion."
            )
        if merged.unpaired_forward_deg or merged.unpaired_reverse_deg:
            _warn(
                f"{len(merged.unpaired_forward_deg)} forward and "
                f"{len(merged.unpaired_reverse_deg)} reverse points had no partner within "
                f"{step / 2.0:g} degrees, so those grid points come from one direction only."
            )
    else:
        frames_nm, energy_kj, phi_deg = forward.frames_nm, forward.energy_kj, forward.phi_deg

    heavy = [
        i for i, atom in enumerate(topology.atoms)
        if guess_element(atom.name, atom.mass) != "H"
    ]
    hops = find_hops(frames_nm, heavy, phi_deg)
    for hop in hops:
        _warn(
            f"possible conformational hop at {hop.phi_deg:.1f} degrees (frame {hop.index}): "
            f"heavy-atom RMSD {hop.rmsd_nm:.4f} nm against a {hop.threshold_nm:.4f} nm "
            "outlier threshold. A smooth cosine series fitted through a hop is meaningless."
        )
    hbonds = hbond_changes(frames_nm, topology, phi_deg)
    for change in hbonds:
        _warn(
            f"hydrogen bonding changed at {change.phi_deg:.1f} degrees (frame "
            f"{change.index}): formed {sorted(change.formed)}, broke {sorted(change.broken)}. "
            "That stabilisation is not the torsional potential; fitting through it writes "
            "an electrostatic interaction into the dihedral term."
        )

    write_multimodel_pdb(workdir / "scan.pdb", structure, frames_nm)
    payload = {
        "quad": list(quad),
        "nframes": int(frames_nm.shape[0]),
        "phi_deg": [float(v) for v in phi_deg],
        "energy_kj": [float(v) for v in energy_kj],
        "frames_nm": [[[float(c) for c in atom] for atom in frame] for frame in frames_nm],
        "frames_sha256": hash_array(frames_nm),
        "provenance": {
            "xtb_version": _xtb_version(binaries),
            "charge": charge,
            "uhf": int(args.uhf),
            "gfn": int(args.gfn),
            "step_deg": float(step),
            "npoints": int(npoints),
            "bidirectional": bool(args.bidirectional),
            "hysteresis_tol_kj": float(args.hysteresis_tol),
            "max_hysteresis_kj": float(merged.max_hysteresis_kj) if merged else None,
            "worst_hysteresis_angle_deg": float(merged.worst_angle_deg) if merged else None,
            "source_per_point": merged.source if merged else ["forward"] * len(energy_kj),
            "freeze": [list(f) for f in freeze],
            "hops": [
                {"frame": h.index, "phi_deg": h.phi_deg, "rmsd_nm": h.rmsd_nm} for h in hops
            ],
            "hbond_changes": [
                {
                    "frame": c.index,
                    "phi_deg": c.phi_deg,
                    "formed": [list(b) for b in sorted(c.formed)],
                    "broken": [list(b) for b in sorted(c.broken)],
                }
                for c in hbonds
            ],
        },
    }
    _scan_path(args).write_text(json.dumps(payload, indent=2) + "\n", newline="\n")
    write_manifest(args.outdir, "qm_scan", digest, {"quad": list(quad), "npoints": npoints})

    relative = energy_kj - energy_kj.min()
    print(
        f"torsion {quad}: {npoints}-point relaxed scan complete, span "
        f"{relative.max():.2f} kJ/mol, written to {_scan_path(args)}"
    )
    return 0
```

Add a small helper next to `_binaries`:

```python
def _xtb_version(binaries: Binaries) -> str:
    """Best-effort xtb version string for the provenance block; never fatal."""
    try:
        proc = subprocess.run(
            [binaries.require("xtb"), "--version"], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError, FFRefineError):
        return "unknown"
    for line in (proc.stdout or "").splitlines():
        if "version" in line.lower():
            return line.strip()
    return "unknown"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cli_scan.py -q`
Then the full suite in WSL, confirming the live tests ran:
```bash
wsl.exe -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && \
  python -m pytest -q && python -m pytest -q -m "needs_xtb or needs_gmx" -v'
```
Expected: all pass, 0 skipped, `needs_gmx` still 16, `needs_xtb` executing.

- [ ] **Step 5: Commit**

```bash
black . && git add -A
git commit -m "feat(cli): wire ffrefine scan, closing the pipeline end to end"
```

---

### Task 11: `--sp-refine` — ORCA single points on the xtb geometries

**Files:**
- Create: `ffrefine/qm/orca.py`
- Modify: `ffrefine/cli.py` (flags, call site, provenance), `ffrefine/config.py` (`orca_libdir` default)
- Test: `tests/test_qm_orca.py`

**Interfaces:**
- Consumes: `run_command`, `Binaries`, `write_xyz`, `hartree_to_kj`, `ExternalToolError`.
- Produces: `orca.write_sp_input(path, structure, coords_nm, keywords, charge, mult, nprocs=1) -> str`; `orca.parse_final_energy(text) -> float`; `orca.refine_energies(binaries, structure, frames_nm, workdir, keywords, charge, mult, nprocs=1) -> np.ndarray`; `orca.orca_env(binaries) -> dict[str, str]`.

**Why:** Spec 4.3's optional refinement. ORCA 6.1.1 is a **shared** build that will not start unless its own directory is on `LD_LIBRARY_PATH` — the reason `Binaries.orca_libdir` exists. It is on no PATH, so `--orca-bin` is how users reach it. The default `! wB97X-D3 def2-TZVP TightSCF` is far too slow for tests; they override it.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest

from ffrefine.config import Binaries
from ffrefine.errors import ExternalToolError
from ffrefine.io.structure import read_pdb
from ffrefine.qm.orca import (
    orca_env,
    parse_final_energy,
    refine_energies,
    write_sp_input,
)

SAMPLE_OUT = """
------------------- FINAL SINGLE POINT ENERGY -------------------
FINAL SINGLE POINT ENERGY        -1.116759307204
"""


def test_parse_final_energy_reads_the_hartree_value():
    assert parse_final_energy(SAMPLE_OUT) == pytest.approx(-1.116759307204)


def test_parse_final_energy_takes_the_last_when_several_appear():
    """A geometry optimisation prints one per cycle; the converged one is last."""
    text = "FINAL SINGLE POINT ENERGY -1.0\nFINAL SINGLE POINT ENERGY -2.0\n"
    assert parse_final_energy(text) == pytest.approx(-2.0)


def test_parse_final_energy_refuses_output_without_one():
    with pytest.raises(ExternalToolError, match="FINAL SINGLE POINT ENERGY"):
        parse_final_energy("SCF NOT CONVERGED\n")


def test_sp_input_carries_keywords_charge_and_multiplicity(tmp_path, butane_pdb):
    s = read_pdb(butane_pdb)
    text = write_sp_input(
        tmp_path / "sp.inp", s, s.coords, "! HF STO-3G", charge=0, mult=1, nprocs=2
    )
    assert text.startswith("! HF STO-3G")
    assert "%pal nprocs 2" in text
    assert "*xyz 0 1" in text
    assert text.rstrip().endswith("*")


def test_sp_input_is_in_angstrom(tmp_path, butane_pdb):
    s = read_pdb(butane_pdb)
    text = write_sp_input(tmp_path / "sp.inp", s, s.coords, "! HF STO-3G", 0, 1)
    first = [ln for ln in text.splitlines() if ln.startswith(("C ", "H "))][0]
    assert float(first.split()[1]) == pytest.approx(s.coords[0][0] * 10.0, abs=1e-5)


def test_orca_env_puts_the_libdir_on_ld_library_path(tmp_path):
    """A shared ORCA build will not start otherwise."""
    env = orca_env(Binaries(orca=str(tmp_path / "orca"), orca_libdir=str(tmp_path)))
    assert str(tmp_path) in env["LD_LIBRARY_PATH"]


def test_orca_env_defaults_the_libdir_to_the_binarys_own_directory(tmp_path):
    env = orca_env(Binaries(orca=str(tmp_path / "orca")))
    assert str(tmp_path) in env["LD_LIBRARY_PATH"]


@pytest.mark.needs_orca
def test_refine_energies_runs_real_orca(tmp_path, butane_pdb, orca_bin):
    """Cheap keywords on purpose: the real default is 24 frames of hybrid DFT."""
    s = read_pdb(butane_pdb)
    frames = np.stack([s.coords, s.coords])
    energies = refine_energies(
        Binaries(orca=orca_bin), s, frames, tmp_path, "! HF STO-3G", charge=0, mult=1
    )
    assert energies.shape == (2,)
    # Same geometry twice, so the two energies must agree exactly.
    assert energies[0] == pytest.approx(energies[1], rel=1e-10)
    # Butane at HF/STO-3G is around -155 Hartree, i.e. very negative in kJ/mol.
    assert energies[0] < -300000.0
```

Add to `tests/conftest.py`:

```python
@pytest.fixture
def orca_bin() -> str:
    """Path to orca, or skip the test when ORCA is unavailable."""
    exe = shutil.which("orca")
    if exe is None:
        pytest.skip("orca not on PATH")
    return exe
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_qm_orca.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ffrefine.qm.orca'`.

- [ ] **Step 3: Implement**

Create `ffrefine/qm/orca.py`:

```python
"""ORCA single-point driver for --sp-refine (spec section 4.3).

ORCA reports energies in Hartree, converted here through
:mod:`ffrefine.units`.  Geometries go out in Angstrom.

The installed ORCA is a *shared* build: it will not start unless its own
directory is on ``LD_LIBRARY_PATH``, which is what
:attr:`ffrefine.config.Binaries.orca_libdir` is for.  It is also on no PATH by
default, so ``--orca-bin`` is the normal way to reach it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..config import Binaries
from ..errors import ExternalToolError
from ..io.structure import Structure, write_xyz
from ..runner import run_command
from ..units import ANGSTROM_TO_NM, hartree_to_kj

DEFAULT_SP_KEYWORDS = "! wB97X-D3 def2-TZVP TightSCF"
"""Spec 4.3's default single-point level; overridable with --orca-sp-keywords."""


def orca_env(binaries: Binaries) -> dict[str, str]:
    """Environment additions a shared ORCA build needs to start.

    Defaults ``orca_libdir`` to the ORCA binary's own directory, which is where
    a stock unpacked ORCA keeps its shared objects.
    """
    libdir = binaries.orca_libdir or str(Path(binaries.require("orca")).parent)
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    return {"LD_LIBRARY_PATH": f"{libdir}:{existing}" if existing else libdir}


def write_sp_input(
    path: str | Path,
    structure: Structure,
    coords_nm: np.ndarray,
    keywords: str,
    charge: int,
    mult: int,
    nprocs: int = 1,
) -> str:
    """Write an ORCA single-point input for one geometry."""
    lines = [keywords]
    if nprocs > 1:
        lines.append(f"%pal nprocs {nprocs} end")
    lines.append("")
    lines.append(f"*xyz {charge} {mult}")
    for element, (x, y, z) in zip(structure.elements, np.asarray(coords_nm, float) / ANGSTROM_TO_NM):
        lines.append(f"{element:<2s} {x:>14.8f} {y:>14.8f} {z:>14.8f}")
    lines.append("*")
    text = "\n".join(lines) + "\n"
    Path(path).write_text(text, newline="\n")
    return text


def parse_final_energy(text: str) -> float:
    """Extract ``FINAL SINGLE POINT ENERGY`` in Hartree from ORCA output.

    Takes the last occurrence: an optimisation prints one per cycle and only
    the converged one is meaningful.

    Raises:
        ExternalToolError: no such line, which means the job did not converge
            or died before reporting.
    """
    value = None
    for line in text.splitlines():
        if "FINAL SINGLE POINT ENERGY" in line:
            parts = line.split()
            try:
                value = float(parts[-1])
            except ValueError:
                continue
    if value is None:
        raise ExternalToolError(
            "ORCA produced no 'FINAL SINGLE POINT ENERGY' line, so the calculation did not "
            "reach a converged energy. Check the .stdout log in the stage directory."
        )
    return value


def refine_energies(
    binaries: Binaries,
    structure: Structure,
    frames_nm: np.ndarray,
    workdir: str | Path,
    keywords: str = DEFAULT_SP_KEYWORDS,
    charge: int = 0,
    mult: int = 1,
    nprocs: int = 1,
) -> np.ndarray:
    """Run one ORCA single point per frame, returning energies in kJ/mol.

    Spec 4.3: "Use these energies in place of the xtb ones when present."  The
    geometries are xtb's; only the energies change, so the fit target improves
    without the scan being redone.

    Raises:
        ExternalToolError: any frame's job fails or reports no final energy.
    """
    exe = binaries.require("orca")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    env = orca_env(binaries)

    out = []
    for index, frame in enumerate(np.asarray(frames_nm, float)):
        stem = f"sp_{index:03d}"
        inp = workdir / f"{stem}.inp"
        write_sp_input(inp, structure, frame, keywords, charge, mult, nprocs)
        proc = run_command([exe, inp.name], workdir, stem, extra_env=env)
        try:
            out.append(parse_final_energy(proc.stdout or ""))
        except ExternalToolError as exc:
            raise ExternalToolError(f"frame {index}: {exc}") from exc
    return hartree_to_kj(np.array(out, dtype=float))
```

In `ffrefine/cli.py`, add to the `scan` subparser and call it after the merge, before the discontinuity checks:

```python
    scan.add_argument(
        "--sp-refine", action="store_true",
        help="refine the xtb energies with ORCA single points on the same geometries",
    )
    scan.add_argument(
        "--orca-sp-keywords", default=None,
        help=f"ORCA keyword line for --sp-refine (default: {DEFAULT_SP_KEYWORDS!r})",
    )
```

```python
    sp_keywords = None
    if args.sp_refine:
        sp_keywords = args.orca_sp_keywords or DEFAULT_SP_KEYWORDS
        energy_kj = refine_energies(
            binaries, structure, frames_nm, workdir / "orca_sp", sp_keywords,
            charge=charge, mult=args.uhf + 1, nprocs=args.nthreads,
        )
        print(f"refined {len(energy_kj)} energies with ORCA ({sp_keywords})")
```

Record `"sp_refine_keywords": sp_keywords` in the provenance block, and add `sp_refine` plus the keywords to the `settings` dict feeding `hash_inputs`, so `--resume` reruns when they change.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_qm_orca.py -q`, then in WSL confirm the live ORCA test executes:
```bash
wsl.exe -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && \
  python -m pytest -q -m needs_orca -v'
```
Expected: passes, not skipped.

- [ ] **Step 5: Commit**

```bash
black . && git add -A
git commit -m "feat(qm): --sp-refine via ORCA single points on the xtb geometries"
```

---

### Task 12: Documentation

**Files:**
- Modify: `README.md`

**Interfaces:** none.

**Why:** The README currently instructs users to hand-write `scan.json` and lists `scan` as unimplemented. Both are now false, and a README that misdescribes the tool is the same class of defect as ruling R40's comment.

- [ ] **Step 1: Update the subcommand table**

Change the `ffrefine scan` row from **not implemented** to a description of what it does: relaxed xtb scan, bidirectional by default, discontinuity reporting, optional `--sp-refine`.

- [ ] **Step 2: Rewrite "Supplying a QM scan"**

Retitle to "Running the QM scan". Lead with the `ffrefine scan` invocation. Keep the hand-written `scan.json` schema as a documented fallback for users bringing their own QM, since `mmscan` still accepts it. Keep every existing warning about the `frames_nm` digest and re-running `mmscan`.

- [ ] **Step 3: Update "What this build cannot do"**

Remove `scan` from the not-implemented list, leaving `charges` (milestone 5), `.str` parsing and `final/report.md` (milestone 6), and `--mm-relax`. Add a short subsection stating what the scan does not do: it reports hops and hydrogen bonds but does not correct or reject a contaminated profile, and `--sp-refine`'s default level is expensive.

- [ ] **Step 4: Document the new flags**

`--step`, `--charge`, `--uhf`, `--gfn`, `--bidirectional/--no-bidirectional`, `--hysteresis-tol`, `--freeze-dihedral`, `--sp-refine`, `--orca-sp-keywords`. Note that ORCA is typically not on PATH and needs `--orca-bin`, and that its shared build needs `orca_libdir` (defaulted from the binary's directory).

- [ ] **Step 5: Verify and commit**

```bash
wsl.exe -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && \
  python -m pytest -q && python -m pytest -q -m "needs_gmx or needs_xtb or needs_orca" && \
  python -m black --check .'
git add -A
git commit -m "docs: ffrefine scan is implemented; document its flags and limits"
```

---

## Self-Review

**Spec coverage** (design doc → task):

| Design requirement | Task |
|---|---|
| xcontrol `$constrain`/`$scan` generation | 2 |
| Runtime verification the block was honoured | 4 |
| `xtbscan.log` parsing, Angstrom→nm, Hartree→kJ | 1, 3 |
| Angles measured from coordinates (spec 9.5) | 5 |
| Bidirectional scanning, angle-based pairing | 6 |
| Hysteresis reporting | 6, 10 |
| Hop detection (robust outlier) | 7 |
| Hydrogen-bond detection | 8 |
| `--freeze-dihedral` | 2, 10 |
| Charge from topology, `--uhf` flag | 9 |
| `scan.json` + `scan.pdb` + manifest + `--resume` | 10 |
| `--sp-refine` with ORCA | 11 |
| `ANGSTROM_TO_NM` → `units.py`; `write_xyz` | 1 |
| `needs_xtb` / `needs_orca` markers in pyproject only | 4, 11 |
| README | 12 |

**Placeholder scan:** none — every code step carries real code, every test step real assertions.

**Type consistency:** `ScanLeg(frames_nm, energy_kj, phi_deg, direction)` is constructed identically in Tasks 6 and 10. `run_scan` is keyword-only after `workdir` in Tasks 4, 10 and 11. `parse_scan_log` returns `(frames_nm, energy_kj)` in that order at all three call sites. `measure_angles(frames_nm, quad)` takes a **1-based** quad everywhere. `derive_charge(topology, override)` matches Task 10's `derive_charge(topology, args.charge)`. `find_hops(frames_nm, heavy_indices, phi_deg, ...)` and `hbond_changes(frames_nm, topology, phi_deg)` match their Task 10 call sites.

**Open question carried from the design:** whether `--nthreads` should drive ORCA's `%pal nprocs`. Task 11 passes `nprocs=args.nthreads`, but the installed ORCA targets OpenMPI 4.1.8 against the system's 4.1.6 and only serial execution has been verified. **If the `needs_orca` test fails under `nprocs > 1`, hard-code `nprocs=1` and record the reason** rather than debugging the MPI stack — parallel ORCA is milestone 5's problem, and a serial single point on the test's cheap keywords costs nothing.
