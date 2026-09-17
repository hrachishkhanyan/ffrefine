# Milestone 5 — `ffrefine charges` (ESP-derived partial charges) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `ffrefine charges` stub with a working stage that fits partial charges to an ORCA electrostatic potential, writes a charge-refined topology, and chains into `run` ahead of the torsion stages.

**Architecture:** A new pure-numpy `ffrefine/charges/` package decides things (grid, fit, equivalence, bookkeeping) and runs no subprocess; `ffrefine/qm/orca.py` and `ffrefine/qm/xtb.py` drive the external programs. CHELPG reaches end-to-end working state at Task 10; RESP builds on the same plumbing in Tasks 11-14.

**Tech Stack:** Python 3.10+, numpy, pytest, black (line-length 100). ORCA 6.1.1 and xtb 6.7.1 via WSL. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-17-ffrefine-m5-orca-charges-design.md`

## Global Constraints

- All user-facing atom indices are **1-based**; internal numpy arrays are 0-based; convert at the boundary and say so in the docstring (spec section 8).
- Unit conversions live **only** in `ffrefine/units.py` (spec section 9.8). `orca_vpot` speaks bohr and Hartree/e.
- Type hints throughout, docstrings on public functions, `black` formatting at line-length 100, target py310.
- Every external command goes through `ffrefine.runner.run_command`, which logs the full command line and captures stdout/stderr into the stage directory (spec section 8).
- Nothing in `ffrefine/charges/` may import `subprocess` or call an external binary. That is what makes the numerics testable with no ORCA installed.
- Tests must be run in an **interactive** WSL shell: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest -q -rs'`. A login shell (`bash -lc`) does not source `~/.bashrc` and so cannot see ORCA, and the `needs_orca` tests then skip while reporting green. **Require zero skipped.**
- Branch: `ffrefine/milestone-5-orca-charges`. Do not work on `master`.

## One refinement of the design's module table

The design (section 3) lists four modules in `ffrefine/charges/`. This plan adds a fifth, `esp.py`, holding the point-charge potential evaluation and the RRMS. Both schemes need it — design section 4 reports RRMS for CHELPG as well as RESP — and putting it in `resp.py` would make the CHELPG path import the RESP module for something that is not RESP.

## File structure

| File | Responsibility | Task |
|---|---|---|
| `ffrefine/units.py` | add bohr conversions | 1 |
| `ffrefine/cli.py` | split `_load_inputs`; flags; `cmd_charges`; `run` chaining | 2, 9, 10, 14 |
| `ffrefine/charges/__init__.py` | package marker | 3 |
| `ffrefine/charges/esp.py` | potential from point charges; RRMS | 3 |
| `ffrefine/charges/equivalence.py` | equivalent-hydrogen groups from the bond graph | 4 |
| `ffrefine/charges/apply.py` | `selected`/`all`, redistribution, rounding, exact total | 5 |
| `ffrefine/io/topwriter.py` | `write_charged_topology`, line-preserving | 6 |
| `ffrefine/qm/orca.py` | ESP job, CHELPG parsing, `orca_vpot` driver | 7, 12 |
| `ffrefine/qm/xtb.py` | `optimize_geometry` | 8 |
| `ffrefine/charges/grid.py` | Merz-Singh-Kollman shell grid | 11 |
| `ffrefine/charges/resp.py` | two-stage restrained fit | 13 |
| `fixtures/butane/orca/` | committed CHELPG and vpot fixtures | 7, 12 |
| `README.md` | subcommand table, worked example, flags | 15 |

**CHELPG is end-to-end usable after Task 10.** Tasks 11-14 add RESP. Task 15 documents both.

---

### Task 1: Bohr conversions in `units.py`

**Files:**
- Modify: `ffrefine/units.py`
- Test: `tests/test_units.py`

**Interfaces:**
- Consumes: the existing `ANGSTROM_TO_NM` and `Numeric` alias.
- Produces: `units.BOHR_TO_ANGSTROM: float`, `units.bohr_to_nm(x: Numeric) -> Numeric`, `units.nm_to_bohr(x: Numeric) -> Numeric`.

**Why:** `orca_vpot` reads grid coordinates and writes potentials in atomic units. Spec section 9.8 allows conversions in exactly one module, and the grid builder (Task 11) and the vpot parser (Task 12) both need this.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_units.py`:

```python
def test_bohr_to_nm_uses_the_codata_value():
    from ffrefine.units import bohr_to_nm

    assert bohr_to_nm(1.0) == pytest.approx(0.0529177210903)


def test_bohr_round_trips_through_nm():
    from ffrefine.units import bohr_to_nm, nm_to_bohr

    assert nm_to_bohr(bohr_to_nm(2.5)) == pytest.approx(2.5, rel=1e-12)


def test_bohr_conversion_is_consistent_with_the_angstrom_one():
    """One bohr is 0.529177 Angstrom, and Angstrom already has a conversion.

    Two independent constants for the same physical relation is how a unit bug
    hides: each looks right on its own.
    """
    from ffrefine.units import angstrom_to_nm, bohr_to_nm

    assert bohr_to_nm(1.0) == pytest.approx(angstrom_to_nm(0.529177210903), rel=1e-12)


def test_bohr_conversions_work_on_arrays():
    from ffrefine.units import nm_to_bohr

    out = nm_to_bohr(np.array([0.0, 0.0529177210903]))
    assert out == pytest.approx([0.0, 1.0])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_units.py -q -k bohr`
Expected: FAIL with `ImportError: cannot import name 'bohr_to_nm'`.

- [ ] **Step 3: Implement**

Add to `ffrefine/units.py`, next to `ANGSTROM_TO_NM`:

```python
BOHR_TO_ANGSTROM: float = 0.529177210903
"""Angstrom per bohr (CODATA 2018).

ORCA's ``orca_vpot`` reads grid coordinates and writes potentials in atomic
units, so the ESP path crosses this boundary twice.
"""
```

and next to the other converters:

```python
def bohr_to_nm(x: Numeric) -> Numeric:
    """Convert bohr to nanometres."""
    return x * BOHR_TO_ANGSTROM * ANGSTROM_TO_NM


def nm_to_bohr(x: Numeric) -> Numeric:
    """Convert nanometres to bohr."""
    return x / (BOHR_TO_ANGSTROM * ANGSTROM_TO_NM)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_units.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
python -m black ffrefine tests -q
git add ffrefine/units.py tests/test_units.py
git commit -m "feat(units): bohr conversions for the ORCA ESP path"
```

---

### Task 2: Let a stage load inputs without demanding a torsion

**Files:**
- Modify: `ffrefine/cli.py` (`_load_inputs`, `_write_input_stage`, `_prepare`)
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `cli._load_structure_and_topology(args) -> tuple[Structure, Topology]`; `cli._resolve_quad(args, topology) -> Quad`; `_load_inputs` unchanged in signature; `_write_input_stage(args, structure, topology, quad: Quad | None)`; `_prepare(args, *, need_quad: bool = True) -> tuple[Structure, Topology, Quad | None]`.
- Consumed by: Task 9's `cmd_charges`, which calls `_prepare(args, need_quad=False)`.

**Why:** Design section 9. `_load_inputs` refuses without `--dihedral` and `_write_input_stage` needs a quad, so `charges` would demand a flag it never uses. Split the quad resolution out rather than building a second path around it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli.py`:

```python
def test_inputs_load_without_a_dihedral(butane_pdb, butane_itp):
    """charges needs no torsion, so loading must not require one."""
    from ffrefine.cli import _load_structure_and_topology

    parser = build_parser()
    args = parser.parse_args(["charges", "-s", str(butane_pdb), "-p", str(butane_itp)])
    structure, topology = _load_structure_and_topology(args)

    assert topology.natoms == 14
    assert structure.natoms == 14


def test_resolving_a_quad_still_refuses_when_none_was_given(butane_pdb, butane_itp):
    """The refusal must survive the split, not leak away with it."""
    from ffrefine.cli import _load_structure_and_topology, _resolve_quad

    parser = build_parser()
    args = parser.parse_args(["mmscan", "-s", str(butane_pdb), "-p", str(butane_itp)])
    _, topology = _load_structure_and_topology(args)

    with pytest.raises(ValidationError, match="--dihedral"):
        _resolve_quad(args, topology)


def test_input_stage_omits_the_torsion_set_when_there_is_no_torsion(
    tmp_path, butane_pdb, butane_itp
):
    """model.json must not claim a target torsion that was never selected."""
    from ffrefine.cli import _load_structure_and_topology, _write_input_stage

    parser = build_parser()
    args = parser.parse_args(
        ["charges", "-s", str(butane_pdb), "-p", str(butane_itp), "--outdir", str(tmp_path)]
    )
    structure, topology = _load_structure_and_topology(args)
    _write_input_stage(args, structure, topology, None)

    model = json.loads((tmp_path / "00_input" / "model.json").read_text())
    assert model["natoms"] == 14
    assert "target_quad_1_based" not in model
    assert "torsion_set_1_based" not in model
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_cli.py -q -k "without_a_dihedral or resolving_a_quad or omits_the_torsion"`
Expected: FAIL with `ImportError: cannot import name '_load_structure_and_topology'`.

- [ ] **Step 3: Implement**

Replace `_load_inputs` in `ffrefine/cli.py` with three functions, keeping the existing body verbatim except where noted:

```python
def _load_structure_and_topology(args: argparse.Namespace) -> tuple[Structure, Topology]:
    """Read and cross-check the structure and topology, with no torsion involved.

    Split out of :func:`_load_inputs` so ``charges``, which refines charges and
    touches no dihedral, does not have to supply a ``--dihedral`` it never uses.

    Raises:
        ParseError: a structure or topology file could not be parsed.
    """
    structure = read_structure(args.structure)
    topology = read_topology(args.topology)
    for warning in validate_structure_topology(
        structure, topology, strict_names=not args.no_strict_names
    ):
        _warn(warning)
    # XYZ carries no atom names, so the reader's placeholders would reach
    # grompp and never match the topology. Order is already checked above, so
    # the topology's own naming is the right one to use from here on.
    return adopt_topology_naming(structure, topology), topology


def _resolve_quad(args: argparse.Namespace, topology: Topology) -> Quad:
    """Resolve the single target torsion from ``--dihedral``.

    Raises:
        ValidationError: no ``--dihedral`` was given, more than one was given,
            or the quadruplet is unusable with this topology.
    """
    quads = getattr(args, "dihedral", None) or []
    if not quads:
        raise ValidationError(
            "no torsion selected. Pass --dihedral i,j,k,l using 1-based atom "
            "indices. Automatic selection from a CGenFF .str file arrives in "
            "milestone 6 and is not available in this build."
        )
    if len(quads) > 1:
        raise ValidationError(
            f"{len(quads)} torsions requested, but this build fits one torsion per run: "
            "spec section 6's output layout holds a single QM scan and a single fit per "
            "outdir. Re-run with one --dihedral and its own --outdir for each torsion."
        )
    quad = quads[0]
    validate_quad(topology, quad)
    return quad


def _load_inputs(args: argparse.Namespace) -> tuple[Structure, Topology, Quad]:
    """Read, cross-check and return the structure, topology and target quad."""
    structure, topology = _load_structure_and_topology(args)
    return structure, topology, _resolve_quad(args, topology)
```

- [ ] **Step 4: Make `_write_input_stage` and `_prepare` accept no quad**

In `_write_input_stage`, change the signature to `quad: Quad | None` and replace the three torsion keys with a conditional block:

```python
    if quad is not None:
        torsion_set = collect_torsion_set(topology, quad)
        model["target_quad_1_based"] = list(quad)
        model["central_bond_1_based"] = [quad[1], quad[2]]
        model["torsion_set_size"] = len(torsion_set)
        model["torsion_set_1_based"] = [[d.ai, d.aj, d.ak, d.al] for d in torsion_set]
```

(build `model` without those four keys, then add them only when a quad exists). Then widen `_prepare`:

```python
def _prepare(
    args: argparse.Namespace, *, need_quad: bool = True
) -> tuple[Structure, Topology, Quad | None]:
    """Configure logging, read the inputs and write ``00_input/``.

    ``need_quad=False`` is for stages that refine something other than a
    torsion; they get ``None`` and ``00_input/model.json`` omits the torsion
    set rather than recording one that was never chosen.
    """
    setup_logging(args.outdir, args.verbose)
    structure, topology = _load_structure_and_topology(args)
    quad = _resolve_quad(args, topology) if need_quad else None
    _write_input_stage(args, structure, topology, quad)
    return structure, topology, quad
```

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q -rs`
Expected: PASS, zero skipped, no existing test changed behaviour. The other four stages still call `_prepare(args)` and still get a quad.

- [ ] **Step 6: Commit**

```bash
python -m black ffrefine tests -q
git add ffrefine/cli.py tests/test_cli.py
git commit -m "refactor(cli): separate loading inputs from resolving the target torsion"
```

---

### Task 3: `charges/esp.py` — potential from point charges, and RRMS

**Files:**
- Create: `ffrefine/charges/__init__.py`, `ffrefine/charges/esp.py`
- Test: `tests/test_charges_esp.py`

**Interfaces:**
- Produces: `esp.inverse_distances(atom_xyz_bohr, grid_bohr) -> np.ndarray` shape `(npoints, natoms)`; `esp.potential_from_charges(inv_r, charges_e) -> np.ndarray` shape `(npoints,)`; `esp.normal_equations(inv_r, v_ref) -> tuple[np.ndarray, np.ndarray]` shapes `(natoms, natoms)` and `(natoms,)`; `esp.rrms(v_ref, v_fit) -> float`.
- Consumed by: Task 9 (CHELPG RRMS), Task 13 (the RESP fit).

**Why:** Design sections 4 and 6.3. Every other piece of the ESP maths is one of these four. `inverse_distances` is computed once per fit and reused by both the potential evaluation and the normal equations, so a transposed axis shows up in every test at once rather than in one of them.

- [ ] **Step 1: Write the failing test**

Create `tests/test_charges_esp.py`:

```python
import numpy as np
import pytest

from ffrefine.charges.esp import (
    inverse_distances,
    normal_equations,
    potential_from_charges,
    rrms,
)
from ffrefine.errors import ValidationError


def test_a_single_charge_gives_the_coulomb_potential():
    """V = q/r in atomic units, which is the whole physical content here."""
    atoms = np.array([[0.0, 0.0, 0.0]])
    grid = np.array([[2.0, 0.0, 0.0], [0.0, 4.0, 0.0]])

    v = potential_from_charges(inverse_distances(atoms, grid), np.array([0.5]))

    assert v == pytest.approx([0.25, 0.125])


def test_charges_superpose():
    atoms = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    grid = np.array([[1.0, 0.0, 0.0]])
    inv_r = inverse_distances(atoms, grid)

    assert potential_from_charges(inv_r, np.array([1.0, 1.0])) == pytest.approx([2.0])
    assert potential_from_charges(inv_r, np.array([1.0, -1.0])) == pytest.approx([0.0])


def test_inverse_distances_is_points_by_atoms_not_the_transpose():
    """The shape is the whole bug class: a transpose still multiplies cleanly."""
    atoms = np.zeros((3, 3))
    grid = np.zeros((7, 3))
    grid[:, 0] = np.arange(1.0, 8.0)

    assert inverse_distances(atoms, grid).shape == (7, 3)


def test_a_grid_point_sitting_on_an_atom_is_refused():
    """1/0 would poison the fit with inf and give charges that look finite."""
    atoms = np.array([[0.0, 0.0, 0.0]])
    grid = np.array([[0.0, 0.0, 0.0]])

    with pytest.raises(ValidationError, match="coincides"):
        inverse_distances(atoms, grid)


def test_normal_equations_are_symmetric_and_solve_the_exact_case():
    """Fitting a potential generated BY charges must return those charges."""
    rng = np.random.default_rng(0)
    atoms = rng.normal(size=(4, 3)) * 2.0
    grid = rng.normal(size=(200, 3)) * 8.0
    truth = np.array([0.3, -0.4, 0.2, -0.1])
    inv_r = inverse_distances(atoms, grid)
    v_ref = potential_from_charges(inv_r, truth)

    a, b = normal_equations(inv_r, v_ref)

    assert a == pytest.approx(a.T)
    assert np.linalg.solve(a, b) == pytest.approx(truth, abs=1e-9)


def test_rrms_is_zero_for_a_perfect_fit_and_one_when_nothing_is_reproduced():
    v = np.array([1.0, -2.0, 3.0])

    assert rrms(v, v) == pytest.approx(0.0)
    assert rrms(v, np.zeros(3)) == pytest.approx(1.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_charges_esp.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ffrefine.charges'`.

- [ ] **Step 3: Implement**

Create `ffrefine/charges/__init__.py`:

```python
"""Charge refinement: ESP grids, restrained fitting, and topology bookkeeping."""
```

Create `ffrefine/charges/esp.py`:

```python
"""Electrostatic potential arithmetic shared by both charge schemes.

Everything here is in atomic units -- coordinates in bohr, potentials in
Hartree per electron -- because that is what ``orca_vpot`` reads and writes,
and converting once at the process boundary beats converting in every formula.

Nothing in this module runs a subprocess, so the numerics can be tested
without ORCA installed.
"""

from __future__ import annotations

import numpy as np

from ..errors import ValidationError


def inverse_distances(atom_xyz_bohr: np.ndarray, grid_bohr: np.ndarray) -> np.ndarray:
    """``1 / |r_i - R_j|`` for every grid point i and atom j.

    Computed once and reused by :func:`potential_from_charges` and
    :func:`normal_equations`, so the two cannot disagree about which axis is
    which.

    Args:
        atom_xyz_bohr: shape (natoms, 3).
        grid_bohr: shape (npoints, 3).

    Returns:
        Shape ``(npoints, natoms)`` -- points first, matching the grid.

    Raises:
        ValidationError: a grid point coincides with an atom, which would make
            the entry infinite and hand the fit charges that still look finite.
    """
    atoms = np.asarray(atom_xyz_bohr, float)
    grid = np.asarray(grid_bohr, float)
    delta = grid[:, None, :] - atoms[None, :, :]
    r = np.sqrt((delta**2).sum(axis=-1))
    if not np.all(r > 1e-12):
        point, atom = np.unravel_index(int(np.argmin(r)), r.shape)
        raise ValidationError(
            f"grid point {point} coincides with atom {atom + 1} (separation "
            f"{r[point, atom]:.3e} bohr). The potential there is infinite, so the grid "
            "cannot be used; this usually means a vdW radius far too small for the element."
        )
    return 1.0 / r


def potential_from_charges(inv_r: np.ndarray, charges_e: np.ndarray) -> np.ndarray:
    """Potential at each grid point from a set of point charges."""
    return np.asarray(inv_r, float) @ np.asarray(charges_e, float)


def normal_equations(inv_r: np.ndarray, v_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The unrestrained least-squares system for charges against a potential.

    ``A_jk = sum_i 1/(r_ij r_ik)`` and ``B_j = sum_i V_i / r_ij``, which is the
    normal-equation form of minimising ``sum_i (V_i - sum_j q_j / r_ij)^2``.

    Returns:
        ``(A, B)`` with shapes ``(natoms, natoms)`` and ``(natoms,)``.
    """
    inv_r = np.asarray(inv_r, float)
    return inv_r.T @ inv_r, inv_r.T @ np.asarray(v_ref, float)


def rrms(v_ref: np.ndarray, v_fit: np.ndarray) -> float:
    """Relative root-mean-square deviation of a fitted potential.

    Relative rather than absolute because the potential's scale is set by the
    molecule and the grid, so an absolute RMS says nothing across systems.
    """
    ref = np.asarray(v_ref, float)
    denominator = float((ref**2).sum())
    if denominator <= 0.0:
        raise ValidationError("the reference potential is identically zero; RRMS is undefined")
    return float(np.sqrt(((ref - np.asarray(v_fit, float)) ** 2).sum() / denominator))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_charges_esp.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
python -m black ffrefine tests -q
git add ffrefine/charges tests/test_charges_esp.py
git commit -m "feat(charges): ESP potential arithmetic shared by both schemes"
```

---

### Task 4: `charges/equivalence.py` — equivalent hydrogens

**Files:**
- Modify: `ffrefine/io/validate.py` (promote the element helper), `ffrefine/qm/scan.py` (use it)
- Create: `ffrefine/charges/equivalence.py`
- Test: `tests/test_charges_equivalence.py`

**Interfaces:**
- Produces: `validate.element_of(topology, nr) -> str`; `equivalence.equivalent_hydrogen_groups(topology) -> list[list[int]]` (1-based, only groups of two or more, ascending); `equivalence.average_within_groups(charges_e, groups) -> np.ndarray` (charges 0-based by atom order, groups 1-based).
- Consumed by: Task 9 (CHELPG averaging), Task 13 (RESP stage-2 constraints).

**Why:** Design section 6.1. Spec 5.3's wording admits several readings, so this fixes one: **equivalent means bonded to the same heavy atom and sharing an atom type.** Averaging preserves each group's sum, so it can never disturb the molecular total.

`qm/scan.py` already has a private `_element_of` doing mass-first element determination. Promoting it to `io/validate.py`, which already owns `element_from_mass`, stops a second copy drifting from the first.

- [ ] **Step 1: Write the failing test**

Create `tests/test_charges_equivalence.py`:

```python
import numpy as np
import pytest

from ffrefine.charges.equivalence import average_within_groups, equivalent_hydrogen_groups
from ffrefine.io.topology import read_topology


def test_butane_gives_two_methyl_and_two_methylene_groups(butane_itp):
    groups = equivalent_hydrogen_groups(read_topology(butane_itp))

    assert [len(g) for g in groups] == [3, 2, 2, 3]
    assert groups[0] == [2, 3, 4]  # the H on C1


def test_a_hydrogen_with_a_different_atom_type_is_not_grouped(butane_itp):
    """Same parent is not enough: the atom type is part of the definition."""
    top = read_topology(butane_itp)
    top.atoms[1].atype = "HGA1"  # H11 no longer matches its siblings

    groups = equivalent_hydrogen_groups(top)

    assert [3, 4] in groups
    assert not any(2 in g for g in groups)


def test_a_lone_hydrogen_forms_no_group(butane_itp):
    """A group of one constrains nothing, so it must not be reported as one."""
    top = read_topology(butane_itp)
    top.atoms[2].atype = "HGA1"
    top.atoms[3].atype = "HGA2"

    assert not any(2 in g or 3 in g or 4 in g for g in equivalent_hydrogen_groups(top))


def test_the_element_comes_from_the_mass_not_the_name(butane_itp):
    """An atom named NA with nitrogen's mass is nitrogen; guess_element disagrees."""
    top = read_topology(butane_itp)
    top.atoms[0].name = "NA"
    top.atoms[0].mass = 14.007

    groups = equivalent_hydrogen_groups(top)

    assert groups[0] == [2, 3, 4]  # still a heavy parent carrying three H


def test_averaging_makes_a_group_equal_and_preserves_its_sum():
    """Preserving the sum is why averaging cannot disturb the molecular total."""
    charges = np.array([-0.3, 0.09, 0.11, 0.10, 0.0])
    before = charges.sum()

    out = average_within_groups(charges, [[2, 3, 4]])

    assert out[1] == pytest.approx(0.10)
    assert out[2] == pytest.approx(0.10)
    assert out[3] == pytest.approx(0.10)
    assert out.sum() == pytest.approx(before)
    assert out[0] == pytest.approx(-0.3)  # untouched


def test_averaging_leaves_everything_alone_when_there_are_no_groups():
    charges = np.array([0.1, -0.1])
    assert average_within_groups(charges, []) == pytest.approx(charges)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_charges_equivalence.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ffrefine.charges.equivalence'`.

- [ ] **Step 3: Promote the element helper**

Move the body of `_element_of` out of `ffrefine/qm/scan.py` into `ffrefine/io/validate.py`, renamed and made public:

```python
def element_of(topology: Topology, nr: int) -> str:
    """Element of a topology atom, preferring its mass over its name.

    ``guess_element`` reads the name only, and names are ambiguous in exactly
    the way that matters: ``NA`` matches the two-letter element table and comes
    back as sodium for what is often a nitrogen.  The mass is unambiguous, and
    :func:`validate_structure_topology` already treats it as the authority, so
    this follows that precedent and falls back to the name only when the
    topology carries no mass.

    Args:
        topology: the parsed molecule.
        nr: 1-based atom number, as written in ``[ atoms ]``.
    """
    atom = topology.atoms[nr - 1]
    if atom.mass is not None:
        from_mass = element_from_mass(atom.mass)
        if from_mass:
            return from_mass
    return guess_element(atom.name) or ""
```

In `ffrefine/qm/scan.py`, delete `_element_of`, import `element_of` from `..io.validate`, and replace both call sites (`hbond_donors_and_acceptors`, `heavy_atom_indices`). `validate.py` already imports `guess_element` and `Topology`, so it needs no new import.

- [ ] **Step 4: Implement the module**

Create `ffrefine/charges/equivalence.py`:

```python
"""Topologically equivalent hydrogens, and how to enforce their equality.

Spec 5.3 asks for hydrogens that are "the same" to carry the same charge.  Its
wording -- same heavy-atom parent, same atom type, same connectivity depth --
can be read several ways, so this module fixes one reading: **two hydrogens are
equivalent exactly when they are bonded to the same heavy atom and share an
atom type.**  That covers the methyl and methylene groups the spec names.

Molecule-wide symmetry is deliberately out of scope: butane's two terminal
carbons are topologically identical, and a real ESP fit gives them charges a
few thousandths apart, but recognising that needs automorphism detection over
the bond graph.  The difference is reported rather than averaged away.

Nothing here runs a subprocess.
"""

from __future__ import annotations

import numpy as np

from ..io.topology import Topology
from ..io.validate import element_of


def equivalent_hydrogen_groups(topology: Topology) -> list[list[int]]:
    """Group hydrogens that must share a charge.

    Args:
        topology: the parsed molecule.  Connectivity comes from ``[ bonds ]``,
            never from interatomic distances, which would mistake a close
            contact for a covalent bond.

    Returns:
        Groups of **1-based** atom numbers, each of length two or more, sorted
        within a group and ordered by the group's first atom.  A hydrogen with
        no equivalent partner appears in no group, because a group of one
        constrains nothing.
    """
    buckets: dict[tuple[int, str], list[int]] = {}
    for atom in topology.atoms:
        if element_of(topology, atom.nr) != "H":
            continue
        heavy = [n for n in topology.bonded_neighbours(atom.nr) if element_of(topology, n) != "H"]
        if len(heavy) != 1:
            continue
        buckets.setdefault((heavy[0], atom.atype), []).append(atom.nr)
    groups = [sorted(members) for members in buckets.values() if len(members) > 1]
    return sorted(groups, key=lambda g: g[0])


def average_within_groups(charges_e: np.ndarray, groups: list[list[int]]) -> np.ndarray:
    """Set every member of each group to the group's mean charge.

    Used by the CHELPG path, where ORCA has already done the fitting and the
    equality can only be imposed afterwards.  Averaging preserves each group's
    sum, so it cannot move the molecular total -- which matters, because the
    total is a hard constraint rather than a preference.

    Args:
        charges_e: shape (natoms,), indexed 0-based in topology atom order.
        groups: 1-based groups from :func:`equivalent_hydrogen_groups`.

    Returns:
        A new array; the input is not modified.
    """
    out = np.array(charges_e, dtype=float, copy=True)
    for group in groups:
        index = [nr - 1 for nr in group]
        out[index] = out[index].mean()
    return out
```

- [ ] **Step 5: Run the tests and the full suite**

Run: `python -m pytest tests/test_charges_equivalence.py -q` then `python -m pytest -q -rs`
Expected: PASS, zero skipped. The `qm/scan.py` change must leave every hop and hydrogen-bond test passing; if `heavy_atom_indices` breaks, the element promotion changed behaviour and needs looking at rather than patching around.

- [ ] **Step 6: Commit**

```bash
python -m black ffrefine tests -q
git add ffrefine/charges/equivalence.py ffrefine/io/validate.py ffrefine/qm/scan.py tests/test_charges_equivalence.py
git commit -m "feat(charges): detect equivalent hydrogens from the bond graph"
```

---

### Task 5: `charges/apply.py` — selected/all, redistribution, exact total

**Files:**
- Create: `ffrefine/charges/apply.py`
- Test: `tests/test_charges_apply.py`

**Interfaces:**
- Produces: `apply.CHARGE_DECIMALS: int`; `apply.ChargeAssignment` dataclass with fields `nr: int`, `name: str`, `before: float`, `after: float`, `source: str` (one of `"fitted"`, `"redistributed"`, `"unchanged"`); `apply.apply_charges(topology, fitted_e, selected, target_total, decimals=CHARGE_DECIMALS) -> list[ChargeAssignment]` where `selected` is a list of 1-based atom numbers, or `None` meaning every atom.
- Consumed by: Task 9 (`cmd_charges`), Task 6 (the writer takes the assignments).

**Why:** Design sections 6.4 and 6.5, from spec 5.3. The molecular total is a hard constraint: a topology whose charges sum to 0.0003 is not a neutral molecule, and GROMACS will happily simulate it.

Butane's charge groups, used by the tests below: atoms 1-4 are `cgnr` 1, atoms 5-7 are 2, atoms 8-10 are 3, atoms 11-14 are 4.

- [ ] **Step 1: Write the failing test**

Create `tests/test_charges_apply.py`:

```python
import numpy as np
import pytest

from ffrefine.charges.apply import apply_charges
from ffrefine.errors import ValidationError
from ffrefine.io.topology import read_topology


def _fitted(natoms: int = 14) -> np.ndarray:
    """A fitted set that differs from butane's and still sums to zero."""
    q = np.full(natoms, 0.05)
    q[[0, 10]] = -0.35
    return q


def test_all_mode_replaces_every_charge_and_keeps_the_total_exact(butane_itp):
    top = read_topology(butane_itp)

    out = apply_charges(top, _fitted(), selected=None, target_total=0)

    assert {a.source for a in out} == {"fitted"}
    assert sum(a.after for a in out) == pytest.approx(0.0, abs=1e-6)


def test_selected_mode_changes_only_the_named_atoms(butane_itp):
    top = read_topology(butane_itp)

    out = apply_charges(top, _fitted(), selected=[1], target_total=0)

    assert out[0].source == "fitted"
    assert out[0].after == pytest.approx(-0.35)
    # Everything outside charge group 1 is untouched.
    assert all(a.source == "unchanged" for a in out[4:])


def test_the_residual_lands_in_the_selected_atoms_charge_group(butane_itp):
    """Spec 5.3: spread over the atoms in the same charge group."""
    top = read_topology(butane_itp)

    out = apply_charges(top, _fitted(), selected=[1], target_total=0)

    recipients = [a.nr for a in out if a.source == "redistributed"]
    assert recipients == [2, 3, 4]  # the rest of cgnr 1
    assert sum(a.after for a in out) == pytest.approx(0.0, abs=1e-6)


def test_the_residual_falls_back_to_bonded_heavy_atoms(butane_itp):
    """A charge group with nothing else in it must not dead-end."""
    top = read_topology(butane_itp)
    for atom in top.atoms:  # every atom in its own charge group
        atom.cgnr = atom.nr

    out = apply_charges(top, _fitted(), selected=[1], target_total=0)

    recipients = [a.nr for a in out if a.source == "redistributed"]
    assert recipients == [5]  # C2, the only heavy atom bonded to C1
    assert sum(a.after for a in out) == pytest.approx(0.0, abs=1e-6)


def test_no_recipient_anywhere_is_refused(butane_itp):
    """Silently leaving the total wrong would be the worst option."""
    top = read_topology(butane_itp)
    for atom in top.atoms:
        atom.cgnr = atom.nr

    with pytest.raises(ValidationError, match="redistribute"):
        apply_charges(top, _fitted(), selected=list(range(1, 15)), target_total=1)


def test_the_total_is_exact_after_rounding(butane_itp):
    """Rounding 14 charges to 4 decimals leaves a residual; it must be absorbed."""
    rng = np.random.default_rng(0)
    top = read_topology(butane_itp)
    q = rng.normal(size=14) * 0.2
    q -= q.sum() / 14.0  # sums to zero, but not to any multiple of 1e-4

    out = apply_charges(top, q, selected=None, target_total=0)

    assert sum(a.after for a in out) == pytest.approx(0.0, abs=1e-6)
    assert all(a.after == pytest.approx(round(a.after, 4)) for a in out)


def test_a_nonzero_target_total_is_honoured(butane_itp):
    top = read_topology(butane_itp)

    out = apply_charges(top, _fitted(), selected=[1], target_total=-1)

    assert sum(a.after for a in out) == pytest.approx(-1.0, abs=1e-6)


def test_an_out_of_range_atom_is_refused(butane_itp):
    top = read_topology(butane_itp)

    with pytest.raises(ValidationError, match="15"):
        apply_charges(top, _fitted(), selected=[15], target_total=0)


def test_the_before_column_is_the_topology_not_the_fit(butane_itp):
    """charges.json reports what changed, so 'before' must be the input value."""
    top = read_topology(butane_itp)

    out = apply_charges(top, _fitted(), selected=[1], target_total=0)

    assert out[0].before == pytest.approx(-0.270)
    assert out[0].name == "C1"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_charges_apply.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ffrefine.charges.apply'`.

- [ ] **Step 3: Implement**

Create `ffrefine/charges/apply.py`:

```python
"""Turning fitted charges into topology charges, with the total kept exact.

Spec 5.3's bookkeeping.  The molecular total is a hard constraint rather than a
preference: a topology whose charges sum to 0.0003 is not a neutral molecule,
and GROMACS will simulate it anyway without complaint.

Nothing here runs a subprocess.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import ValidationError
from ..io.topology import Topology
from ..io.validate import element_of

CHARGE_DECIMALS = 4
"""Decimal places written into ``[ atoms ]``.

Every written charge is then a multiple of 1e-4, so compensating the rounding
residual on one atom makes the total exactly the target rather than merely
close to it.
"""


@dataclass
class ChargeAssignment:
    """What happened to one atom's charge.

    Attributes:
        nr: 1-based atom number.
        name: atom name, for the log and ``charges.json``.
        before: the topology's charge before this stage.
        after: the charge written out.
        source: ``"fitted"`` (taken from the ESP fit), ``"redistributed"``
            (adjusted to keep the total exact) or ``"unchanged"``.
    """

    nr: int
    name: str
    before: float
    after: float
    source: str


def _recipients(topology: Topology, selected: set[int]) -> list[int]:
    """Atoms that may absorb the residual, as 0-based indices.

    Spec 5.3: the atoms in the same charge group as the selected ones, falling
    back to bonded heavy atoms.  Hydrogens are excluded from the fallback
    because pushing a correction onto a hydrogen distorts the group it belongs
    to more than putting it on the heavy atom it hangs off.
    """
    groups = {topology.atoms[nr - 1].cgnr for nr in selected}
    same_group = [a.nr for a in topology.atoms if a.cgnr in groups and a.nr not in selected]
    if same_group:
        return [nr - 1 for nr in same_group]
    bonded = {
        n
        for nr in selected
        for n in topology.bonded_neighbours(nr)
        if n not in selected and element_of(topology, n) != "H"
    }
    return [nr - 1 for nr in sorted(bonded)]


def apply_charges(
    topology: Topology,
    fitted_e: np.ndarray,
    selected: list[int] | None,
    target_total: int,
    decimals: int = CHARGE_DECIMALS,
) -> list[ChargeAssignment]:
    """Combine fitted charges with the topology's, keeping the total exact.

    Args:
        topology: the parsed molecule, supplying the existing charges, the
            charge groups and the bond graph.
        fitted_e: shape (natoms,), 0-based in topology atom order.
        selected: 1-based atom numbers to replace, or ``None`` for every atom
            (``--charge-mode all``).
        target_total: the molecular charge the result must sum to, from
            ``derive_charge``.
        decimals: output precision.

    Returns:
        One :class:`ChargeAssignment` per atom, in topology order.

    Raises:
        ValidationError: a selected atom number is out of range, or the
            residual has nowhere to go.
    """
    natoms = topology.natoms
    fitted = np.asarray(fitted_e, float)
    if fitted.shape != (natoms,):
        raise ValidationError(f"fitted charges have shape {fitted.shape}, expected ({natoms},)")

    if selected is None:
        chosen = set(range(1, natoms + 1))
    else:
        chosen = set(selected)
        bad = sorted(nr for nr in chosen if not 1 <= nr <= natoms)
        if bad:
            raise ValidationError(
                f"--charge-atoms names atom(s) {bad}, but the topology has {natoms} atoms "
                "(indices are 1-based, matching [ atoms ])."
            )

    before = np.array([a.charge for a in topology.atoms], dtype=float)
    after = before.copy()
    index = sorted(nr - 1 for nr in chosen)
    after[index] = fitted[index]

    residual = float(target_total) - float(after.sum())
    if selected is not None and abs(residual) > 1e-12:
        recipients = _recipients(topology, chosen)
        if not recipients:
            raise ValidationError(
                f"nothing left to redistribute the {residual:+.6f} e residual onto: every "
                "atom in the selected atoms' charge groups was itself selected, and no "
                "unselected heavy atom is bonded to one. Select fewer atoms, or use "
                "--charge-mode all."
            )
        after[recipients] += residual / len(recipients)

    after = np.round(after, decimals)
    drift = float(target_total) - float(after.sum())
    if abs(drift) > 0.0:
        largest = int(np.argmax(np.abs(after)))
        after[largest] = round(after[largest] + drift, decimals)
    final_total = float(after.sum())
    if abs(final_total - target_total) > 1e-6:
        raise ValidationError(
            f"charges sum to {final_total:.8f} but the molecular charge is {target_total}; "
            "the rounding compensation did not close the gap."
        )

    out = []
    for i, atom in enumerate(topology.atoms):
        if atom.nr in chosen:
            source = "fitted"
        elif after[i] != before[i]:
            source = "redistributed"
        else:
            source = "unchanged"
        out.append(
            ChargeAssignment(
                nr=atom.nr,
                name=atom.name,
                before=float(before[i]),
                after=float(after[i]),
                source=source,
            )
        )
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_charges_apply.py -q`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
python -m black ffrefine tests -q
git add ffrefine/charges/apply.py tests/test_charges_apply.py
git commit -m "feat(charges): apply fitted charges with an exact molecular total"
```

---

### Task 6: `write_charged_topology` — rewrite only the charge column

**Files:**
- Modify: `ffrefine/io/topwriter.py`
- Test: `tests/test_topwriter.py`

**Interfaces:**
- Consumes: `apply.ChargeAssignment` from Task 5.
- Produces: `topwriter.write_charged_topology(source, assignments, out_path) -> list[int]`, returning the 1-based atom numbers whose lines were rewritten.

**Why:** Design section 4. The existing writers rewrite `[ dihedrals ]` while leaving every other line untouched, and the charge writer must be equally conservative: it reads the source *text* rather than `raw_lines`, so `#ifdef` branches this build did not take still reach grompp.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_topwriter.py`:

```python
def test_charged_topology_changes_only_the_charge_column(tmp_path, butane_itp):
    """Everything but the edited charges must survive byte-for-byte."""
    from ffrefine.charges.apply import ChargeAssignment
    from ffrefine.io.topwriter import write_charged_topology

    assignments = [
        ChargeAssignment(nr=1, name="C1", before=-0.270, after=-0.4058, source="fitted")
    ]
    out = tmp_path / "charged.itp"

    changed = write_charged_topology(butane_itp, assignments, out)

    assert changed == [1]
    before = butane_itp.read_text().splitlines()
    after = out.read_text().splitlines()
    assert len(before) == len(after)
    differing = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert len(differing) == 1
    assert "-0.4058" in after[differing[0]]
    assert "CG331" in after[differing[0]]  # the rest of the line is intact


def test_charged_topology_rewrites_nothing_when_no_charge_moved(tmp_path, butane_itp):
    from ffrefine.charges.apply import ChargeAssignment
    from ffrefine.io.topwriter import write_charged_topology

    assignments = [
        ChargeAssignment(nr=1, name="C1", before=-0.270, after=-0.270, source="unchanged")
    ]
    out = tmp_path / "charged.itp"

    assert write_charged_topology(butane_itp, assignments, out) == []
    assert out.read_text() == butane_itp.read_text()


def test_charged_topology_reads_back_with_the_new_charges(tmp_path, butane_itp):
    """The point of the file: the parser must see what was written."""
    from ffrefine.charges.apply import ChargeAssignment
    from ffrefine.io.topwriter import write_charged_topology
    from ffrefine.io.topology import read_topology

    assignments = [
        ChargeAssignment(nr=1, name="C1", before=-0.270, after=-0.4058, source="fitted"),
        ChargeAssignment(nr=11, name="C4", before=-0.270, after=-0.4091, source="fitted"),
    ]
    out = tmp_path / "charged.itp"
    write_charged_topology(butane_itp, assignments, out)

    top = read_topology(out)
    assert top.atoms[0].charge == pytest.approx(-0.4058)
    assert top.atoms[10].charge == pytest.approx(-0.4091)
    assert top.atoms[1].charge == pytest.approx(0.090)  # untouched


def test_charged_topology_refuses_an_atom_it_cannot_find(tmp_path, butane_itp):
    """A silently skipped atom would give a topology with the wrong total."""
    from ffrefine.charges.apply import ChargeAssignment
    from ffrefine.io.topwriter import write_charged_topology
    from ffrefine.errors import ValidationError

    assignments = [
        ChargeAssignment(nr=99, name="ZZ", before=0.0, after=0.5, source="fitted")
    ]

    with pytest.raises(ValidationError, match="99"):
        write_charged_topology(butane_itp, assignments, tmp_path / "charged.itp")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_topwriter.py -q -k charged`
Expected: FAIL with `ImportError: cannot import name 'write_charged_topology'`.

- [ ] **Step 3: Implement**

Add to `ffrefine/io/topwriter.py` (it already imports `Path`, `ValidationError` and `log`; add `ChargeAssignment` under `TYPE_CHECKING` or import it directly from `..charges.apply`):

```python
def write_charged_topology(
    source: str | Path,
    assignments: list["ChargeAssignment"],
    out_path: str | Path,
) -> list[int]:
    """Copy a topology, rewriting only the charge column of changed atoms.

    Reads the source *text* rather than the parsed ``raw_lines``, for the same
    reason :func:`write_molecule_copy` does: ``#ifdef`` branches this build did
    not take must still reach grompp, which runs its own preprocessor.

    Only the charge field is touched.  Atom type, residue, charge group and
    mass are left exactly as written, including their spacing, so a diff
    against the input shows the charges and nothing else.

    Args:
        source: the user's topology; never modified.
        assignments: from :func:`ffrefine.charges.apply.apply_charges`.  Entries
            whose charge did not move are skipped.
        out_path: destination.

    Returns:
        The 1-based atom numbers whose lines were rewritten, in file order.

    Raises:
        ValidationError: an assignment names an atom that has no line in
            ``[ atoms ]``.  Skipping it silently would leave a topology whose
            charges no longer sum to the molecular charge.
    """
    wanted = {a.nr: a for a in assignments if a.after != a.before}
    lines = Path(source).read_text().splitlines()
    out: list[str] = []
    changed: list[int] = []
    section = ""
    for line in lines:
        body = line.split(";", 1)[0].strip()
        if body.startswith("[") and body.endswith("]"):
            section = body.strip("[] ").strip()
            out.append(line)
            continue
        if section != "atoms" or not body:
            out.append(line)
            continue
        fields = body.split()
        nr = int(fields[0]) if fields[0].isdigit() else None
        if nr is None or nr not in wanted:
            out.append(line)
            continue
        out.append(_replace_charge_field(line, wanted[nr].after))
        changed.append(nr)

    missing = sorted(set(wanted) - set(changed))
    if missing:
        raise ValidationError(
            f"{source}: no [ atoms ] line for atom(s) {missing}, so their new charges "
            "could not be written. The topology and the fitted charges disagree about "
            "how many atoms this molecule has."
        )
    Path(out_path).write_text("\n".join(out) + "\n", newline="\n")
    log.info("wrote %s with %d charge(s) changed", Path(out_path).name, len(changed))
    return changed


def _replace_charge_field(line: str, charge: float) -> str:
    """Rewrite the seventh whitespace field of an ``[ atoms ]`` line in place.

    Works on the character span of the existing field rather than reformatting
    the line, so the surrounding columns keep their original spacing and any
    trailing comment survives.
    """
    body, comment = (line.split(";", 1) + [""])[:2]
    fields = body.split()
    if len(fields) < 7:
        raise ValidationError(f"[ atoms ] line has {len(fields)} fields, expected at least 7: {line!r}")
    old = fields[6]
    start = body.index(old, sum(len(f) for f in fields[:6]))
    new = f"{charge:.{CHARGE_DECIMALS}f}".rjust(len(old))
    rebuilt = body[:start] + new + body[start + len(old) :]
    return rebuilt + (";" + comment if comment else "")
```

Import `CHARGE_DECIMALS` from `..charges.apply` at the top of the module.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_topwriter.py -q`
Expected: PASS, including the existing dihedral-writer tests.

- [ ] **Step 5: Commit**

```bash
python -m black ffrefine tests -q
git add ffrefine/io/topwriter.py tests/test_topwriter.py
git commit -m "feat(io): write a charge-refined topology, touching only the charge column"
```

---

### Task 7: ORCA ESP job and CHELPG parsing

**Files:**
- Modify: `ffrefine/qm/orca.py`
- Create: `fixtures/butane/orca/esp.chelpg.xyz`, `fixtures/butane/orca/esp_head.vpot`
- Test: `tests/test_qm_orca.py`

**Interfaces:**
- Consumes: `orca.write_sp_input`, `orca.orca_env`, `runner.run_command`, `units.bohr_to_nm` (Task 1).
- Produces: `orca.ESP_DEFAULT_KEYWORDS: str`; `orca.run_esp_job(binaries, structure, coords_nm, workdir, keywords, charge, mult, nprocs=1, chelpg=True) -> Path` returning the job's base path (so `<base>.gbw`, `<base>.chelpg.xyz` and `<base>.vpot` are derivable); `orca.parse_chelpg_charges(path) -> np.ndarray` (shape `(natoms,)`, in e); `orca.read_chelpg_vpot(path) -> tuple[np.ndarray, np.ndarray]` returning grid coordinates in **nm** shape `(npoints, 3)` and potentials in Hartree/e shape `(npoints,)`.

**Why:** Design section 2. `<base>.chelpg.xyz` carries eight decimals against the stdout table's six, and has no banner to drift between ORCA versions. `read_chelpg_vpot` exists so the CHELPG path can report an RRMS against ORCA's own grid (design section 4).

**The trap this task must not fall into:** the CHELPG run writes `<base>.vpot` with columns **`V x y z`**, while `orca_vpot` (Task 12) writes **`x y z V`**. Same extension, reversed order. Each parser is separate and named for its producer.

- [ ] **Step 1: Create the fixtures**

Create `fixtures/butane/orca/esp.chelpg.xyz` with exactly this content (captured from ORCA 6.1.1 on the butane fixture at `! HF 6-31G* CHELPG`):

```
14
# ORCA CHELPG point charges xyz file
C       -1.926000     0.265000     0.000000  -0.40576138
H       -2.032000     0.897000    -0.884000   0.08817793
H       -2.032000     0.897000     0.884000   0.08746497
H       -2.744000    -0.457000     0.000000   0.09559476
C       -0.595000    -0.475000     0.000000   0.17720002
H       -0.536000    -1.124000     0.879000  -0.02207280
H       -0.536000    -1.124000    -0.879000  -0.02279233
C        0.595000     0.475000     0.000000   0.18790252
H        0.536000     1.124000     0.879000  -0.02471356
H        0.536000     1.124000    -0.879000  -0.02429518
C        1.926000    -0.265000     0.000000  -0.40905194
H        2.744000     0.457000     0.000000   0.09569030
H        2.032000    -0.897000     0.884000   0.08822003
H        2.032000    -0.897000    -0.884000   0.08843667
```

Create `fixtures/butane/orca/esp_head.vpot` — the first five grid rows of the same run's `.vpot`, keeping its header and all fourteen atom rows, so the parser is exercised on the real layout:

```
   14  5
                   -3.6396125E+00   5.0077743E-01   0.0000000E+00
                   -3.8399235E+00   1.6950843E+00  -1.6705179E+00
                   -3.8399235E+00   1.6950843E+00   1.6705179E+00
                   -5.1854085E+00  -8.6360484E-01   0.0000000E+00
                   -1.1243870E+00  -8.9761991E-01   0.0000000E+00
                   -1.0128932E+00  -2.1240522E+00   1.6610693E+00
                   -1.0128932E+00  -2.1240522E+00  -1.6610693E+00
                    1.1243870E+00   8.9761991E-01   0.0000000E+00
                    1.0128932E+00   2.1240522E+00   1.6610693E+00
                    1.0128932E+00   2.1240522E+00  -1.6610693E+00
                    3.6396125E+00  -5.0077743E-01   0.0000000E+00
                    5.1854085E+00   8.6360484E-01   0.0000000E+00
                    3.8399235E+00  -1.6950843E+00   1.6705179E+00
                    3.8399235E+00  -1.6950843E+00  -1.6705179E+00
    1.9701622E-04  -9.9097238E+00  -2.8799426E+00  -7.2565484E-01
    2.3633834E-04  -9.9097238E+00  -2.8799426E+00  -1.5873700E-01
    2.2492757E-04  -9.9097238E+00  -2.8799426E+00   4.0818084E-01
    1.6610901E-04  -9.9097238E+00  -2.8799426E+00   9.7509869E-01
   -1.2312459E-04  -9.9097238E+00  -2.3130248E+00  -1.8594905E+00
```

Note the header count was edited from `9972` to `5` to match the truncation; the parser must trust the header, so a mismatch here would be a test of the wrong thing.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_qm_orca.py`:

```python
ORCA_FIXTURES = Path(__file__).parent.parent / "fixtures" / "butane" / "orca"


def test_parse_chelpg_charges_reads_all_fourteen():
    from ffrefine.qm.orca import parse_chelpg_charges

    q = parse_chelpg_charges(ORCA_FIXTURES / "esp.chelpg.xyz")

    assert q.shape == (14,)
    assert q[0] == pytest.approx(-0.40576138)
    assert q[10] == pytest.approx(-0.40905194)
    assert q.sum() == pytest.approx(0.0, abs=1e-6)


def test_parse_chelpg_charges_refuses_a_file_with_no_charge_column(tmp_path):
    from ffrefine.qm.orca import parse_chelpg_charges

    path = tmp_path / "bad.xyz"
    path.write_text("1\ncomment\nC 0.0 0.0 0.0\n")

    with pytest.raises(ExternalToolError, match="charge"):
        parse_chelpg_charges(path)


def test_read_chelpg_vpot_takes_potential_from_the_first_column():
    """The CHELPG .vpot is V x y z. Reading it as x y z V silently swaps them."""
    from ffrefine.qm.orca import read_chelpg_vpot

    grid_nm, v = read_chelpg_vpot(ORCA_FIXTURES / "esp_head.vpot")

    assert grid_nm.shape == (5, 3)
    assert v.shape == (5,)
    assert v[0] == pytest.approx(1.9701622e-04)
    assert v[4] == pytest.approx(-1.2312459e-04)
    # -9.9097238 bohr is about -0.5244 nm; if the columns were swapped this
    # would be the 1.97e-04 potential instead.
    assert grid_nm[0][0] == pytest.approx(-0.52440, abs=1e-4)


def test_read_chelpg_vpot_refuses_a_truncated_file(tmp_path):
    from ffrefine.qm.orca import read_chelpg_vpot

    path = tmp_path / "short.vpot"
    path.write_text("   2  3\n 0.0 0.0 0.0\n 0.0 0.0 0.0\n 1.0 2.0 3.0 4.0\n")

    with pytest.raises(ExternalToolError, match="3"):
        read_chelpg_vpot(path)


def test_esp_input_adds_chelpg_to_the_keyword_line(tmp_path, butane_pdb):
    """--orca-esp-keywords carries the level only; the scheme adds CHELPG."""
    from ffrefine.io.structure import read_pdb
    from ffrefine.qm.orca import esp_keyword_line

    assert esp_keyword_line("! HF 6-31G*", chelpg=True) == "! HF 6-31G* CHELPG"
    assert esp_keyword_line("! HF 6-31G*", chelpg=False) == "! HF 6-31G*"


def test_esp_keyword_line_does_not_double_chelpg():
    """A user who passes CHELPG themselves must not get it twice."""
    from ffrefine.qm.orca import esp_keyword_line

    assert esp_keyword_line("! HF 6-31G* CHELPG", chelpg=True) == "! HF 6-31G* CHELPG"


@pytest.mark.needs_orca
def test_run_esp_job_produces_the_files_the_schemes_need(tmp_path, butane_pdb, orca_bin):
    """Cheap keywords on purpose; the default is hybrid DFT."""
    from ffrefine.config import Binaries
    from ffrefine.io.structure import read_pdb
    from ffrefine.qm.orca import parse_chelpg_charges, run_esp_job

    s = read_pdb(butane_pdb)
    base = run_esp_job(
        Binaries(orca=orca_bin), s, s.coords, tmp_path, "! HF STO-3G", charge=0, mult=1
    )

    assert base.with_suffix(".gbw").exists()
    assert Path(str(base) + ".chelpg.xyz").exists()
    assert Path(str(base) + ".vpot").exists()
    q = parse_chelpg_charges(Path(str(base) + ".chelpg.xyz"))
    assert q.shape == (14,)
    assert q.sum() == pytest.approx(0.0, abs=1e-4)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_qm_orca.py -q`
Expected: FAIL with `ImportError: cannot import name 'parse_chelpg_charges'`.

- [ ] **Step 4: Implement**

Add to `ffrefine/qm/orca.py` (it already imports `Path`, `np`, `Binaries`, `ExternalToolError`, `run_command`, `write_sp_input`, `orca_env`; add `from ..units import bohr_to_nm`):

```python
ESP_DEFAULT_KEYWORDS = "! HF 6-31G*"
"""Spec 5.2's ESP level of theory, without the scheme keyword.

HF/6-31G* is deliberate rather than dated: its systematic overpolarisation
approximates condensed-phase charges, which is why both Amber and the
CHARMM-adjacent workflows fit charges at this level.

The scheme appends ``CHELPG`` itself (see :func:`esp_keyword_line`), so that
overriding the level with --orca-esp-keywords cannot silently switch the
scheme off.
"""


def esp_keyword_line(keywords: str, chelpg: bool) -> str:
    """The ORCA keyword line for an ESP job, adding CHELPG when the scheme needs it."""
    if chelpg and "chelpg" not in keywords.lower():
        return f"{keywords} CHELPG"
    return keywords


def run_esp_job(
    binaries: Binaries,
    structure: Structure,
    coords_nm: np.ndarray,
    workdir: str | Path,
    keywords: str = ESP_DEFAULT_KEYWORDS,
    charge: int = 0,
    mult: int = 1,
    nprocs: int = 1,
    chelpg: bool = True,
) -> Path:
    """Run the single ESP job both charge schemes are built on.

    One job serves both: RESP needs the converged density for ``orca_vpot``,
    and CHELPG costs nothing on top of it.  The run also leaves ``<base>.vpot``
    -- ORCA's own grid carrying the QM potential -- which is what lets the
    CHELPG path report a fit RRMS comparable with RESP's.

    Returns:
        The job's base path.  ``<base>.gbw``, ``<base>.scfp``,
        ``<base>.chelpg.xyz`` and ``<base>.vpot`` hang off it.

    Raises:
        ExternalToolError: ORCA failed, or produced no CHELPG file when asked.
    """
    exe = binaries.require("orca")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    base = workdir / "esp"
    write_sp_input(
        base.with_suffix(".inp"),
        structure,
        coords_nm,
        esp_keyword_line(keywords, chelpg),
        charge,
        mult,
        nprocs,
    )
    run_command([exe, base.with_suffix(".inp").name], workdir, "esp", extra_env=orca_env(binaries))
    if chelpg and not Path(str(base) + ".chelpg.xyz").exists():
        raise ExternalToolError(
            f"ORCA finished but wrote no {base.name}.chelpg.xyz, so the CHELPG charges were "
            "never produced. Check esp.stdout in the stage directory; the usual cause is a "
            "keyword line that does not reach the CHELPG module."
        )
    return base


def parse_chelpg_charges(path: str | Path) -> np.ndarray:
    """Read ORCA's ``<base>.chelpg.xyz`` into an array of charges in e.

    Preferred over the stdout table: this file carries eight decimals to the
    table's six, and no surrounding banner that can change between versions.

    Raises:
        ExternalToolError: the file has no charge column, which means the run
            did not produce CHELPG charges.
    """
    lines = Path(path).read_text().splitlines()
    if len(lines) < 3:
        raise ExternalToolError(f"{path}: too short to be a CHELPG point-charge file")
    count = int(lines[0].split()[0])
    charges = []
    for line in lines[2 : 2 + count]:
        fields = line.split()
        if len(fields) < 5:
            raise ExternalToolError(
                f"{path}: line {line!r} has no charge column. ORCA writes element, x, y, z "
                "and charge; a four-column file is a plain xyz, not a CHELPG result."
            )
        charges.append(float(fields[4]))
    if len(charges) != count:
        raise ExternalToolError(f"{path}: header says {count} atoms, found {len(charges)}")
    return np.array(charges, dtype=float)


def read_chelpg_vpot(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the ``.vpot`` written by a CHELPG run.

    **Column order matters.** This file is ``V x y z``; the file ``orca_vpot``
    writes is ``x y z V``.  Same extension, reversed order, and reading either
    with the other's parser yields coordinates that look like potentials while
    the fit carries on producing numbers.

    Returns:
        ``(grid_nm, v_hartree_per_e)`` with shapes ``(npoints, 3)`` and
        ``(npoints,)``.  Coordinates are converted from bohr here, at the
        process boundary (spec section 9.8).

    Raises:
        ExternalToolError: the file holds fewer points than its header claims.
    """
    lines = Path(path).read_text().splitlines()
    natoms, npoints = (int(v) for v in lines[0].split()[:2])
    rows = [ln.split() for ln in lines[1 + natoms : 1 + natoms + npoints] if ln.strip()]
    if len(rows) != npoints:
        raise ExternalToolError(
            f"{path}: header claims {npoints} grid points, found {len(rows)}. The file is "
            "truncated, which usually means ORCA was killed part-way through."
        )
    data = np.array([[float(x) for x in row[:4]] for row in rows], dtype=float)
    return bohr_to_nm(data[:, 1:4]), data[:, 0]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_qm_orca.py -q -rs` in an interactive WSL shell.
Expected: PASS, zero skipped — the `needs_orca` test must execute.

- [ ] **Step 6: Commit**

```bash
python -m black ffrefine tests -q
git add ffrefine/qm/orca.py tests/test_qm_orca.py fixtures/butane/orca
git commit -m "feat(qm): ORCA ESP job and CHELPG parsing"
```

---

### Task 8: `optimize_geometry` for `--charge-geom xtb`

**Files:**
- Modify: `ffrefine/qm/xtb.py`
- Test: `tests/test_qm_xtb.py`

**Interfaces:**
- Consumes: `write_xyz`, `run_command`, `read_xyz`, `Binaries`.
- Produces: `xtb.optimize_geometry(binaries, structure, coords_nm, workdir, charge, uhf=0, gfn=2, nthreads=1) -> np.ndarray` shape `(natoms, 3)` in nm.

**Why:** Spec 5.1's default geometry source. Charges fitted to a strained input geometry describe a molecule nobody will simulate.

`run_scan` already refuses to trust exit 0 and verifies xtb *reports* what was asked; this does the same, checking for the converged banner and that the atom count came back unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_qm_xtb.py`:

```python
@pytest.mark.needs_xtb
def test_optimize_geometry_moves_the_structure_and_keeps_the_atom_order(
    tmp_path, butane_pdb, xtb_bin
):
    from ffrefine.qm.xtb import optimize_geometry

    s = read_pdb(butane_pdb)
    out = optimize_geometry(
        Binaries(xtb=xtb_bin), s, s.coords, tmp_path, charge=0, uhf=0, nthreads=1
    )

    assert out.shape == s.coords.shape
    # The hand-authored fixture is not at an xtb minimum, so it must move,
    # but it must stay the same molecule rather than fly apart.
    assert np.abs(out - s.coords).max() > 1e-4
    assert np.abs(out - s.coords).max() < 0.1


def test_optimize_geometry_refuses_output_that_never_converged(tmp_path, butane_pdb, monkeypatch):
    """xtb can exit 0 having given up; the banner is the real signal."""
    import ffrefine.qm.xtb as xtb_module
    from ffrefine.errors import ExternalToolError

    class _Proc:
        stdout = "some output without the magic words\n"
        stderr = ""

    monkeypatch.setattr(xtb_module, "run_command", lambda *a, **k: _Proc())
    s = read_pdb(butane_pdb)

    with pytest.raises(ExternalToolError, match="converge"):
        xtb_module.optimize_geometry(
            Binaries(xtb="/nonexistent/xtb"), s, s.coords, tmp_path, charge=0
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_qm_xtb.py -q -k optimize_geometry`
Expected: FAIL with `ImportError: cannot import name 'optimize_geometry'`.

- [ ] **Step 3: Implement**

Add to `ffrefine/qm/xtb.py`:

```python
OPT_CONVERGED_MARKER = "GEOMETRY OPTIMIZATION CONVERGED"
"""What xtb prints when the optimiser actually finished.

Checked rather than trusting exit 0, for the same reason run_scan verifies its
$scan block: xtb can stop early and still exit cleanly, and a geometry that
never converged produces charges for a molecule that does not exist.
"""


def optimize_geometry(
    binaries: Binaries,
    structure: Structure,
    coords_nm: np.ndarray,
    workdir: str | Path,
    charge: int,
    uhf: int = 0,
    gfn: int = 2,
    nthreads: int = 1,
) -> np.ndarray:
    """Optimise a geometry with ``xtb --opt tight`` and return it in nm.

    Spec 5.1's default for ``--charge-geom``: charges fitted to a strained
    input geometry describe a molecule nobody is going to simulate.

    Args:
        binaries: resolved external programs.
        structure: supplies the element symbols; its coordinates are ignored in
            favour of ``coords_nm``.
        coords_nm: starting geometry, shape (natoms, 3).
        workdir: directory to run in; xtb writes ``xtbopt.xyz`` into it.
        charge: molecular charge for ``--chrg``.
        uhf: unpaired electrons for ``--uhf``.
        gfn: GFN parameterisation.
        nthreads: ``-P``.

    Returns:
        The optimised coordinates in nm, in the input atom order.

    Raises:
        ExternalToolError: xtb did not report convergence, wrote no
            ``xtbopt.xyz``, or returned a different number of atoms.
    """
    exe = binaries.require("xtb")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    write_xyz(workdir / "mol.xyz", structure, coords_nm)
    argv = [exe, "mol.xyz", "--opt", "tight", "--chrg", str(charge), "--uhf", str(uhf)]
    argv += ["--gfn", str(gfn), "-P", str(nthreads)]
    proc = run_command(argv, workdir, "xtb_opt")

    if OPT_CONVERGED_MARKER not in (proc.stdout or ""):
        raise ExternalToolError(
            f"xtb did not report '{OPT_CONVERGED_MARKER}', so the geometry was never "
            "optimised even though the run finished. See xtb_opt.stdout in the stage "
            "directory. Charges fitted to an unconverged geometry are meaningless."
        )
    optimised = workdir / "xtbopt.xyz"
    if not optimised.exists():
        raise ExternalToolError(f"xtb reported convergence but wrote no {optimised}")
    out = read_xyz(optimised)
    if out.natoms != len(coords_nm):
        raise ExternalToolError(
            f"{optimised} has {out.natoms} atoms, the input had {len(coords_nm)}"
        )
    return out.coords
```

`read_xyz` needs importing from `..io.structure` alongside the existing `Structure, write_xyz`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_qm_xtb.py -q -rs` in an interactive WSL shell.
Expected: PASS, zero skipped.

- [ ] **Step 5: Commit**

```bash
python -m black ffrefine tests -q
git add ffrefine/qm/xtb.py tests/test_qm_xtb.py
git commit -m "feat(qm): xtb geometry optimisation for the charge stage"
```

---

### Task 9: `cmd_charges` — CHELPG end to end

**Files:**
- Modify: `ffrefine/cli.py`
- Test: `tests/test_cli_charges.py`

**Interfaces:**
- Consumes: everything from Tasks 1-8.
- Produces: `cli.cmd_charges(args) -> int`; `cli._add_charge_flags(parser)`; `cli._atom_list(text) -> tuple[int, ...]`; `04_charges/charges.json`; `04_charges/molecule_charged.itp`.

**Why:** Design sections 1 (CLI surface), 4 (data flow) and 7 (error handling). After this task `ffrefine charges --charge-scheme chelpg` works end to end; Task 10 chains it into `run`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli_charges.py`:

```python
import json

import numpy as np
import pytest

from ffrefine.cli import build_parser, main
from ffrefine.io.topology import read_topology


def _args(butane_pdb, butane_itp, outdir, *extra):
    return [
        "charges",
        "-s",
        str(butane_pdb),
        "-p",
        str(butane_itp),
        "--outdir",
        str(outdir),
        *extra,
    ]


def test_atom_list_parses_one_based_indices():
    from ffrefine.cli import _atom_list

    assert _atom_list("1,5,8") == (1, 5, 8)
    assert _atom_list(" 2 , 3 ") == (2, 3)


def test_atom_list_rejects_zero_and_negatives():
    from ffrefine.cli import _atom_list
    from ffrefine.errors import ValidationError

    with pytest.raises(ValidationError, match="1-based"):
        _atom_list("0,3")


def test_charges_defaults_to_selected_and_chelpg():
    args = build_parser().parse_args(["charges", "-s", "a.pdb", "-p", "a.itp"])

    assert args.charge_scheme == "chelpg"
    assert args.charge_mode == "selected"
    assert args.charge_geom == "xtb"
    assert args.charge_atoms is None


def test_selected_mode_without_atoms_is_refused(butane_pdb, butane_itp, tmp_path, capsys):
    """Nothing is replaced silently, and the message names both ways forward."""
    rc = main(_args(butane_pdb, butane_itp, tmp_path / "out"))

    assert rc == 1
    err = capsys.readouterr().err
    assert "--charge-atoms" in err
    assert "--charge-mode all" in err
    assert "Traceback" not in err


def test_charges_dry_run_writes_nothing(butane_pdb, butane_itp, tmp_path):
    outdir = tmp_path / "out"
    rc = main(_args(butane_pdb, butane_itp, outdir, "--charge-mode", "all", "--dry-run"))

    assert rc == 0
    assert not outdir.exists()


def test_charges_needs_no_dihedral(butane_pdb, butane_itp, tmp_path, capsys):
    """Task 2's split exists for this: charges touches no torsion."""
    rc = main(_args(butane_pdb, butane_itp, tmp_path / "out", "--charge-mode", "all", "--dry-run"))

    assert rc == 0
    assert "--dihedral" not in capsys.readouterr().err


@pytest.mark.needs_orca
def test_chelpg_end_to_end_writes_a_usable_topology(
    butane_pdb, butane_itp, tmp_path, orca_bin, xtb_bin
):
    """Cheap keywords: the default level is hybrid DFT on a real molecule."""
    outdir = tmp_path / "out"
    rc = main(
        _args(
            butane_pdb,
            butane_itp,
            outdir,
            "--charge-mode",
            "all",
            "--charge-geom",
            "input",
            "--orca-esp-keywords",
            "! HF STO-3G",
            "--nthreads",
            "1",
        )
    )
    assert rc == 0

    payload = json.loads((outdir / "04_charges" / "charges.json").read_text())
    assert len(payload["atoms"]) == 14
    assert payload["total_after"] == pytest.approx(0.0, abs=1e-6)
    assert payload["rrms"] > 0.0
    assert payload["provenance"]["scheme"] == "chelpg"

    charged = read_topology(outdir / "04_charges" / "molecule_charged.itp")
    assert sum(a.charge for a in charged.atoms) == pytest.approx(0.0, abs=1e-6)
    # Equivalent hydrogens were averaged, so a methyl's three must be identical.
    assert charged.atoms[1].charge == charged.atoms[2].charge == charged.atoms[3].charge


@pytest.mark.needs_orca
def test_selected_mode_leaves_unselected_atoms_alone(
    butane_pdb, butane_itp, tmp_path, orca_bin
):
    outdir = tmp_path / "out"
    assert (
        main(
            _args(
                butane_pdb,
                butane_itp,
                outdir,
                "--charge-atoms",
                "1",
                "--charge-geom",
                "input",
                "--orca-esp-keywords",
                "! HF STO-3G",
                "--nthreads",
                "1",
            )
        )
        == 0
    )

    charged = read_topology(outdir / "04_charges" / "molecule_charged.itp")
    original = read_topology(butane_itp)
    assert charged.atoms[0].charge != original.atoms[0].charge
    assert charged.atoms[7].charge == original.atoms[7].charge  # C3, far from the change
    assert sum(a.charge for a in charged.atoms) == pytest.approx(0.0, abs=1e-6)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_cli_charges.py -q`
Expected: FAIL — `_atom_list` does not exist, and `charges` still exits 2 with the milestone-5 message.

- [ ] **Step 3: Add the flags**

In `ffrefine/cli.py`, next to `_add_scan_flags`:

```python
def _atom_list(text: str) -> tuple[int, ...]:
    """Parse ``1,5,8`` into 1-based atom numbers.

    Raises:
        ValidationError: a field is not a positive integer.  Ruling R29 makes
            every validation failure a ValidationError, which the parser adapts
            to argparse's own error channel.
    """
    out = []
    for field in text.split(","):
        field = field.strip()
        try:
            value = int(field)
        except ValueError:
            raise ValidationError(f"--charge-atoms: {field!r} is not an integer") from None
        if value < 1:
            raise ValidationError(
                f"--charge-atoms: {value} is not a valid atom number; indices are 1-based, "
                "matching the topology's [ atoms ] numbering"
            )
        out.append(value)
    if not out:
        raise ValidationError("--charge-atoms: no atoms given")
    return tuple(out)


def _add_charge_flags(parser: argparse.ArgumentParser) -> None:
    """Add the charge-refinement flags (also needed by ``run``, which chains them)."""
    parser.add_argument(
        "--charge-scheme",
        choices=("chelpg", "resp"),
        default="chelpg",
        help="ESP charge scheme (default: chelpg)",
    )
    parser.add_argument(
        "--charge-geom",
        choices=("input", "xtb"),
        default="xtb",
        help="geometry for the ESP job (default: xtb, an xtb --opt tight optimisation)",
    )
    parser.add_argument(
        "--charge-mode",
        choices=("selected", "all"),
        default="selected",
        help="replace only the named atoms, or every charge (default: selected)",
    )
    parser.add_argument(
        "--charge-atoms",
        action="append",
        type=_atom_list,
        default=None,
        metavar="i,j,k",
        help="1-based atoms whose charges to replace; repeatable",
    )
    parser.add_argument(
        "--orca-esp-keywords",
        default=ESP_DEFAULT_KEYWORDS,
        help=f"ORCA level of theory for the ESP job (default: {ESP_DEFAULT_KEYWORDS!r})",
    )
    parser.add_argument(
        "--esp-density",
        type=float,
        default=1.0,
        help="MK grid points per square Angstrom, --charge-scheme resp only (default: 1.0)",
    )
```

Apply it to both subparsers:

```python
    charges = sub.add_parser("charges", help="ESP/RESP charges via ORCA")
    _add_common(charges)
    _add_charge_flags(charges)
```

and add `_add_charge_flags(run)` alongside the existing `_add_scan_flags(run)`.

Also in this step:

- Add `"charges": ("charges.json", "molecule_charged.itp"),` to `_STAGE_OUTPUTS`.
- Change the dispatch entry from `lambda a: cmd_not_implemented("charges", 5)` to `cmd_charges`.
- Delete the `if args.command == "charges": return handlers[args.command](args)` bypass in `main`, so the stage goes through config handling like every other implemented one. The comment above it, about unimplemented stages not being able to half-run, goes with it.

Imports to add at the top of `cli.py`:

```python
import os
import re

from .charges.apply import apply_charges
from .charges.equivalence import average_within_groups, equivalent_hydrogen_groups
from .charges.esp import inverse_distances, potential_from_charges, rrms
from .qm.orca import (
    ESP_DEFAULT_KEYWORDS,
    orca_env,
    parse_chelpg_charges,
    read_chelpg_vpot,
    run_esp_job,
)
from .units import nm_to_bohr
```

plus `write_charged_topology` added to the existing `.io.topwriter` import and `optimize_geometry` to the existing `.qm.xtb` import.

- [ ] **Step 4: Implement `cmd_charges`**

Add three helpers next to `_scan_path`:

```python
def _charges_path(args: argparse.Namespace) -> Path:
    return Path(args.outdir) / STAGE_DIRS["charges"] / "charges.json"


def _orca_version(binaries: Binaries) -> str:
    """Best-effort ORCA version for the provenance block; never fatal."""
    try:
        proc = subprocess.run(
            [binaries.require("orca"), "--version"],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, **orca_env(binaries)},
        )
    except (OSError, subprocess.SubprocessError, FFRefineError):
        return "unknown"
    for line in (proc.stdout or "").splitlines():
        if "version" in line.lower():
            return line.strip()
    return "unknown"


def _looks_like_cgenff(topology: Topology) -> bool:
    """True when the atom types look like CGenFF (CG331, HGA3, OG311).

    Decides only whether spec 5.3's warning applies.  A false positive prints a
    warning that is merely irrelevant; a false negative withholds one that
    matters, so the test is deliberately loose.
    """
    return any(re.match(r"^[A-Z]{1,2}G\d", atom.atype or "") for atom in topology.atoms)
```

Then add `cmd_charges` next to `cmd_scan`:

```python
def cmd_charges(args: argparse.Namespace) -> int:
    """Fit partial charges to an ORCA electrostatic potential (spec section 5).

    Charges are refined *before* the torsion stages, never after: the fitted
    torsion absorbs whatever the 1-4 electrostatics do not account for, and
    those depend on these charges.  Fitting the torsion first and then changing
    the charges leaves a topology whose torsion was fitted against charges it
    no longer carries.
    """
    selected_atoms = None
    if args.charge_mode == "selected":
        if not args.charge_atoms:
            raise ValidationError(
                "--charge-mode selected replaces only the atoms you name, and none were "
                "named. Pass --charge-atoms i,j,k (1-based), or --charge-mode all to "
                "replace every charge -- which on a CGenFF topology changes the derivation "
                "scheme, so read the warning it prints. Choosing atoms automatically from "
                "a CGenFF .str file arrives in milestone 6."
            )
        selected_atoms = sorted({nr for group in args.charge_atoms for nr in group})

    if args.dry_run:
        structure, topology = _load_structure_and_topology(args)
        what = "every charge" if selected_atoms is None else f"atoms {selected_atoms}"
        print(
            f"would fit {args.charge_scheme} charges for {what} of {topology.name} "
            f"({structure.natoms} atoms), geometry from {args.charge_geom}, "
            f"into {args.outdir}"
        )
        return 0

    structure, topology, _ = _prepare(args, need_quad=False)
    target_total = derive_charge(topology, args.charge)
    settings = {
        "scheme": args.charge_scheme,
        "geom": args.charge_geom,
        "mode": args.charge_mode,
        "atoms": selected_atoms,
        "keywords": args.orca_esp_keywords,
        "esp_density": float(args.esp_density),
        "charge": target_total,
        "uhf": int(args.uhf),
    }
    digest = hash_inputs([args.topology, args.structure], settings)
    if args.resume and stage_is_current(
        args.outdir, "charges", digest, _stage_outputs(args, "charges")
    ):
        print(f"charges: skipping, {_charges_path(args)} is current (--resume)")
        return 0

    binaries = _binaries(args)
    workdir = stage_dir(args.outdir, "charges")

    coords_nm = structure.coords
    if args.charge_geom == "xtb":
        coords_nm = optimize_geometry(
            binaries,
            structure,
            coords_nm,
            workdir / "opt",
            charge=target_total,
            uhf=args.uhf,
            nthreads=args.nthreads,
        )

    base = run_esp_job(
        binaries,
        structure,
        coords_nm,
        workdir,
        args.orca_esp_keywords,
        charge=target_total,
        mult=args.uhf + 1,
        nprocs=args.nthreads,
        chelpg=True,
    )
    raw = parse_chelpg_charges(Path(str(base) + ".chelpg.xyz"))

    groups = equivalent_hydrogen_groups(topology)
    raw = average_within_groups(raw, groups)

    grid_nm, v_ref = read_chelpg_vpot(Path(str(base) + ".vpot"))
    inv_r = inverse_distances(nm_to_bohr(coords_nm), nm_to_bohr(grid_nm))
    fit_rrms = rrms(v_ref, potential_from_charges(inv_r, raw))
    if fit_rrms > 0.5:
        _warn(
            f"the ESP fit reproduces the QM potential poorly (RRMS {fit_rrms:.3f}). Point "
            "charges may simply be a poor model for this molecule; treat the result with "
            "suspicion rather than assuming a bug."
        )

    if selected_atoms is None and _looks_like_cgenff(topology):
        _warn(
            "--charge-mode all replaces every charge on what looks like a CGenFF topology. "
            "CGenFF charges are fitted to water interaction energies and dipoles, not to "
            "the gas-phase ESP, so replacing them wholesale is a change of derivation "
            "scheme and can degrade condensed-phase behaviour. --charge-mode selected with "
            "--charge-atoms is the safer choice."
        )

    assignments = apply_charges(topology, raw, selected_atoms, target_total)
    for a in assignments:
        if a.source == "redistributed":
            log.info(
                "charges: atom %d (%s) %+.4f -> %+.4f, absorbing the redistribution residual",
                a.nr,
                a.name,
                a.before,
                a.after,
            )
    charged_itp = workdir / "molecule_charged.itp"
    write_charged_topology(args.topology, assignments, charged_itp)

    payload = {
        "molecule": topology.name,
        "natoms": topology.natoms,
        "total_before": float(sum(a.before for a in assignments)),
        "total_after": float(sum(a.after for a in assignments)),
        "target_total": int(target_total),
        "rrms": float(fit_rrms),
        "atoms": [
            {
                "nr": a.nr,
                "name": a.name,
                "charge_before": a.before,
                "charge_after": a.after,
                "source": a.source,
            }
            for a in assignments
        ],
        "provenance": {
            "scheme": args.charge_scheme,
            "geometry": args.charge_geom,
            "mode": args.charge_mode,
            "selected_atoms": selected_atoms,
            "orca_esp_keywords": args.orca_esp_keywords,
            "orca_version": _orca_version(binaries),
            "charge": int(target_total),
            "uhf": int(args.uhf),
            "equivalence_groups": groups,
            "npoints": int(len(v_ref)),
        },
    }
    _charges_path(args).write_text(json.dumps(payload, indent=2) + "\n", newline="\n")
    write_manifest(args.outdir, "charges", digest, {"scheme": args.charge_scheme})

    changed = sum(1 for a in assignments if a.source != "unchanged")
    print(
        f"{args.charge_scheme} charges: {changed} of {topology.natoms} atoms changed, "
        f"ESP fit RRMS {fit_rrms:.4f}, written to {charged_itp}"
    )
    print(f"to use them, pass -p {charged_itp} to the remaining stages")
    return 0
```

- [ ] **Step 5: Run the tests to verify they pass**

Run `python -m pytest tests/test_cli_charges.py -q -rs`, then `python -m pytest -q -rs`, both in an interactive WSL shell.
Expected: PASS, zero skipped.

- [ ] **Step 6: Commit**

```bash
python -m black ffrefine tests -q
git add ffrefine/cli.py tests/test_cli_charges.py
git commit -m "feat(cli): ffrefine charges, CHELPG end to end"
```

---

### Task 10: Chain `charges` into `run`, ahead of the torsion stages

**Files:**
- Modify: `ffrefine/cli.py` (`cmd_run`)
- Test: `tests/test_cli_charges.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `cmd_charges` from Task 9, `_charges_path`.
- Produces: no new names; `cmd_run` gains a `charges` stage and rewrites `args.topology`.

**Why:** Design section 5. This is the task that makes the ordering real. After it, **CHELPG is complete end to end** and Tasks 11-14 add RESP on the same plumbing.

Two behaviours matter and are easy to get subtly wrong:

- `run` chains `charges` **only when charge work was asked for** (`--charge-atoms`, or `--charge-mode all`). Otherwise today's `ffrefine run` on butane would start refusing, because `selected` with no atoms is an error.
- After `charges` succeeds, `args.topology` is repointed at `04_charges/molecule_charged.itp` so scan, mmscan, fit and validate use the refined charges. `--resume` needs nothing new: every stage already hashes `args.topology`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli_charges.py`:

```python
def test_run_skips_charges_when_none_was_asked_for(butane_pdb, butane_itp, tmp_path, capsys):
    """Plain `ffrefine run` must keep working exactly as it did."""
    rc = main(
        [
            "run",
            "-s",
            str(butane_pdb),
            "-p",
            str(butane_itp),
            "--dihedral",
            "1,5,8,11",
            "--outdir",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "=== charges ===" not in out
    assert "charges" in out.lower()  # it still says it skipped them


def test_run_chains_charges_first_when_asked(butane_pdb, butane_itp, tmp_path, capsys):
    """Charges must come before mmscan, not after the fit."""
    rc = main(
        [
            "run",
            "-s",
            str(butane_pdb),
            "-p",
            str(butane_itp),
            "--dihedral",
            "1,5,8,11",
            "--outdir",
            str(tmp_path / "out"),
            "--charge-mode",
            "all",
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert out.index("=== charges ===") < out.index("=== mmscan ===")
    assert out.index("=== charges ===") < out.index("=== scan ===")


def test_run_points_later_stages_at_the_charged_topology(
    butane_pdb, butane_itp, tmp_path, monkeypatch
):
    """The swap is the whole mechanism; assert it rather than trusting it."""
    import ffrefine.cli as cli

    outdir = tmp_path / "out"
    charged = outdir / "04_charges" / "molecule_charged.itp"
    charged.parent.mkdir(parents=True)
    charged.write_text(butane_itp.read_text())
    seen = []

    monkeypatch.setattr(cli, "cmd_charges", lambda a: 0)
    monkeypatch.setattr(cli, "cmd_scan", lambda a: seen.append(a.topology) or 0)
    monkeypatch.setattr(cli, "cmd_mmscan", lambda a: seen.append(a.topology) or 0)
    monkeypatch.setattr(cli, "cmd_fit", lambda a: seen.append(a.topology) or 0)
    monkeypatch.setattr(cli, "cmd_validate", lambda a: seen.append(a.topology) or 0)

    args = cli.build_parser().parse_args(
        [
            "run",
            "-s",
            str(butane_pdb),
            "-p",
            str(butane_itp),
            "--dihedral",
            "1,5,8,11",
            "--outdir",
            str(outdir),
            "--charge-mode",
            "all",
        ]
    )
    assert cli.cmd_run(args) == 0
    assert seen == [str(charged)] * 4
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_cli_charges.py -q -k run_`
Expected: FAIL — `run` does not chain charges and never rewrites `args.topology`.

- [ ] **Step 3: Implement**

Replace `cmd_run`'s banner and stage list:

```python
def cmd_run(args: argparse.Namespace) -> int:
    """Run every implemented stage in order, naming the ones it cannot run.

    Charges lead, when they were asked for: the fitted torsion absorbs whatever
    the 1-4 electrostatics do not account for, so refining charges afterwards
    would leave a torsion fitted against charges the topology no longer has.
    """
    wants_charges = bool(args.charge_atoms) or args.charge_mode == "all"
    if wants_charges:
        print(
            "ffrefine run: chaining charges -> scan -> mmscan -> fit -> validate. The "
            "remaining stages use the charge-refined topology, because a torsion fitted "
            "against different charges than it ships with is not transferable."
        )
    else:
        print(
            "ffrefine run: chaining scan -> mmscan -> fit -> validate and SKIPPING "
            "charges, which needs --charge-atoms or --charge-mode all. The topology keeps "
            "the charges it came with."
        )

    stages: list[tuple[str, object]] = []
    if wants_charges:
        stages.append(("charges", cmd_charges))
    stages += [
        ("scan", cmd_scan),
        ("mmscan", cmd_mmscan),
        ("fit", cmd_fit),
        ("validate", cmd_validate),
    ]

    for name, handler in stages:
        print(f"\n=== {name} ===")
        code = handler(args)
        if code != 0:
            print(f"ffrefine run: stopping, stage {name} failed", file=sys.stderr)
            return code
        if name == "charges" and not args.dry_run:
            charged = Path(args.outdir) / STAGE_DIRS["charges"] / "molecule_charged.itp"
            if not charged.exists():
                raise ValidationError(
                    f"the charges stage reported success but wrote no {charged}, so the "
                    "remaining stages have no refined topology to use."
                )
            args.topology = str(charged)
            print(f"run: remaining stages will use {charged}")
    return 0
```

- [ ] **Step 4: Update the existing skip test**

`tests/test_cli.py::test_run_says_which_stages_it_skipped` asserts the old banner. It must now assert that a plain `run` skips charges and still names milestone 5 nowhere -- charges is implemented, it is simply not requested. Change its body to assert `"SKIPPING charges"` appears and `"=== charges ==="` does not.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q -rs` in an interactive WSL shell.
Expected: PASS, zero skipped.

- [ ] **Step 6: Commit**

```bash
python -m black ffrefine tests -q
git add ffrefine/cli.py tests/test_cli_charges.py tests/test_cli.py
git commit -m "feat(cli): run chains charges ahead of the torsion stages"
```

**CHELPG is now complete end to end.** A good point to stop and check the milestone against real use before starting RESP.

---

### Task 11: `charges/grid.py` — the Merz-Singh-Kollman grid

**Files:**
- Create: `ffrefine/charges/grid.py`
- Test: `tests/test_charges_grid.py`

**Interfaces:**
- Produces: `grid.MK_SHELL_SCALES: tuple[float, ...]`; `grid.VDW_RADII_ANGSTROM: dict[str, float]`; `grid.vdw_radius_nm(element) -> float`; `grid.mk_grid(coords_nm, elements, density_per_angstrom2=1.0, scales=MK_SHELL_SCALES) -> np.ndarray` shape `(npoints, 3)` in nm.
- Consumed by: Task 14 (the RESP path in `cmd_charges`).

**Why:** Design section 6.2, from spec 5.2. Two properties are load-bearing: the grid is **deterministic**, so two runs on one input give the same charges and the tests can be pinned; and an element with **no radius is refused**, because a guessed radius moves the fitting surface and biases every charge with nothing in the output to show it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_charges_grid.py`:

```python
import numpy as np
import pytest

from ffrefine.charges.grid import MK_SHELL_SCALES, mk_grid, vdw_radius_nm
from ffrefine.errors import ValidationError


def test_an_element_with_no_radius_is_refused():
    """Defaulting a radius biases every charge and leaves no trace in the output."""
    with pytest.raises(ValidationError, match="Xe"):
        vdw_radius_nm("Xe")


def test_radii_are_in_nanometres():
    assert vdw_radius_nm("C") == pytest.approx(0.15)
    assert vdw_radius_nm("H") == pytest.approx(0.12)


def test_one_atom_puts_every_point_on_one_of_the_four_shells():
    coords = np.zeros((1, 3))
    points = mk_grid(coords, ["C"])

    radii = np.linalg.norm(points, axis=1)
    expected = sorted(s * 0.15 for s in MK_SHELL_SCALES)
    for r in radii:
        assert min(abs(r - e) for e in expected) < 1e-9


def test_the_grid_is_deterministic():
    """Random placement would make charges differ between identical runs."""
    coords = np.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]])

    first = mk_grid(coords, ["C", "O"])
    second = mk_grid(coords, ["C", "O"])

    assert first.shape == second.shape
    assert np.array_equal(first, second)


def test_point_count_grows_with_density():
    coords = np.zeros((1, 3))

    sparse = mk_grid(coords, ["C"], density_per_angstrom2=1.0)
    dense = mk_grid(coords, ["C"], density_per_angstrom2=4.0)

    assert len(dense) > 3.5 * len(sparse)


def test_point_count_grows_with_the_square_of_the_radius():
    """Points per unit area is the spec's unit, so count scales with area."""
    small = mk_grid(np.zeros((1, 3)), ["H"])   # 1.20 A
    large = mk_grid(np.zeros((1, 3)), ["S"])   # 1.75 A

    ratio = len(large) / len(small)
    assert ratio == pytest.approx((1.75 / 1.20) ** 2, rel=0.1)


def test_no_point_survives_inside_another_atoms_shell():
    """The overlap rule is what stops the inner shells burying points in atoms."""
    coords = np.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]])
    elements = ["C", "C"]

    points = mk_grid(coords, elements)

    inner = MK_SHELL_SCALES[0] * 0.15
    for atom in coords:
        assert np.linalg.norm(points - atom, axis=1).min() >= inner - 1e-9


def test_two_atoms_give_fewer_points_than_two_isolated_ones():
    """Because the overlap removal actually removed something."""
    lone = len(mk_grid(np.zeros((1, 3)), ["C"]))
    pair = len(mk_grid(np.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]]), ["C", "C"]))

    assert pair < 2 * lone


def test_a_mismatched_element_list_is_refused():
    with pytest.raises(ValidationError, match="3"):
        mk_grid(np.zeros((3, 3)), ["C", "H"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_charges_grid.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ffrefine.charges.grid'`.

- [ ] **Step 3: Implement**

Create `ffrefine/charges/grid.py`:

```python
"""The Merz-Singh-Kollman shell grid the RESP fit is evaluated on.

Spec 5.2: four shells at 1.4, 1.6, 1.8 and 2.0 times the van der Waals radius,
about one point per square Angstrom, with points lying inside another atom's
shell discarded.

Two properties here are load-bearing rather than incidental.  The construction
is **deterministic** -- a golden spiral, never a random draw -- so two runs on
the same molecule give the same charges and these tests can assert exact
counts.  And an element with no tabulated radius is **refused** rather than
defaulted, because the radius sets where the fitting surface lies, so a guessed
one biases every charge in the molecule with nothing in the output to show it.

Nothing here runs a subprocess.
"""

from __future__ import annotations

import numpy as np

from ..errors import ValidationError
from ..units import ANGSTROM_TO_NM

MK_SHELL_SCALES: tuple[float, ...] = (1.4, 1.6, 1.8, 2.0)
"""Shell radii as multiples of the van der Waals radius (spec 5.2)."""

VDW_RADII_ANGSTROM: dict[str, float] = {
    "H": 1.20,
    "C": 1.50,
    "N": 1.50,
    "O": 1.40,
    "F": 1.35,
    "P": 1.80,
    "S": 1.75,
    "Cl": 1.70,
    "Br": 1.85,
    "I": 1.98,
}
"""Merz-Kollman radii, with Bondi values for Br and I, which MK does not define.

Recorded in ``charges.json`` so a later reader can tell which surface the
charges were fitted on.
"""


def vdw_radius_nm(element: str) -> float:
    """Van der Waals radius in nm.

    Raises:
        ValidationError: the element has no tabulated radius.  Substituting a
            neighbour's value would move the fitting surface and change every
            charge, silently, so this refuses instead.
    """
    symbol = element.strip().capitalize()
    if symbol not in VDW_RADII_ANGSTROM:
        raise ValidationError(
            f"no van der Waals radius for element {element!r}. The ESP grid is built from "
            f"these radii, so guessing one would bias every fitted charge. Known elements: "
            f"{', '.join(sorted(VDW_RADII_ANGSTROM))}."
        )
    return VDW_RADII_ANGSTROM[symbol] * ANGSTROM_TO_NM


def _unit_sphere(n: int) -> np.ndarray:
    """``n`` near-uniform points on the unit sphere, by the golden spiral.

    Deterministic by construction.  The usual alternative -- drawing random
    directions until the count is met -- gives a different grid every run, so
    the same molecule would yield different charges each time and no test could
    pin the result.
    """
    if n < 1:
        return np.zeros((0, 3))
    index = np.arange(n, dtype=float) + 0.5
    z = 1.0 - 2.0 * index / n
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    golden_angle = np.pi * (1.0 + 5.0**0.5)
    theta = golden_angle * index
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta), z))


def mk_grid(
    coords_nm: np.ndarray,
    elements: list[str],
    density_per_angstrom2: float = 1.0,
    scales: tuple[float, ...] = MK_SHELL_SCALES,
) -> np.ndarray:
    """Build the MK grid for one geometry.

    Args:
        coords_nm: shape (natoms, 3).
        elements: one symbol per atom, in the same order.
        density_per_angstrom2: points per square Angstrom of shell surface.
        scales: shell radii as multiples of the vdW radius.

    Returns:
        Shape ``(npoints, 3)`` in nm, in a deterministic order.

    Raises:
        ValidationError: the element list does not match the coordinates, an
            element has no radius, or every point was discarded.
    """
    coords = np.asarray(coords_nm, float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValidationError(f"coordinates have shape {coords.shape}, expected (natoms, 3)")
    if len(elements) != len(coords):
        raise ValidationError(
            f"{len(coords)} atoms but {len(elements)} element symbols; they must correspond"
        )
    if density_per_angstrom2 <= 0.0:
        raise ValidationError(f"--esp-density must be positive, got {density_per_angstrom2}")

    radii = np.array([vdw_radius_nm(e) for e in elements])
    kept: list[np.ndarray] = []
    for scale in scales:
        shell = radii * scale
        for i, centre in enumerate(coords):
            area_angstrom2 = 4.0 * np.pi * (shell[i] / ANGSTROM_TO_NM) ** 2
            n = int(round(area_angstrom2 * density_per_angstrom2))
            if n < 1:
                continue
            points = centre + shell[i] * _unit_sphere(n)
            # Discard points that fall inside any other atom's shell at this
            # scale: they sit in a region the potential is not being fitted on.
            distance = np.linalg.norm(points[:, None, :] - coords[None, :, :], axis=-1)
            inside = (distance < shell[None, :] - 1e-12).any(axis=1)
            kept.append(points[~inside])
    grid = np.vstack(kept) if kept else np.zeros((0, 3))
    if len(grid) == 0:
        raise ValidationError(
            "the ESP grid is empty: every shell point fell inside another atom. Check the "
            "geometry for overlapping atoms."
        )
    return grid
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_charges_grid.py -q`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
python -m black ffrefine tests -q
git add ffrefine/charges/grid.py tests/test_charges_grid.py
git commit -m "feat(charges): deterministic Merz-Singh-Kollman ESP grid"
```

---

### Task 12: the `orca_vpot` driver

**Files:**
- Modify: `ffrefine/qm/orca.py`
- Create: `fixtures/butane/orca/vpot_out.vpot`
- Test: `tests/test_qm_orca.py`

**Interfaces:**
- Consumes: `orca_env`, `run_command`, `Binaries`, `units.nm_to_bohr`, `units.bohr_to_nm`.
- Produces: `orca.write_vpot_grid(path, grid_nm) -> None`; `orca.read_vpot_output(path, expected_points) -> np.ndarray` shape `(npoints,)` in Hartree/e; `orca.run_vpot(binaries, base, grid_nm, workdir) -> np.ndarray`.

**Why:** Design section 2. Spec 5.2 requires this invocation be verified rather than assumed, and it was: the density argument is the **full `<base>.scfp`**, and a wrong name **segfaults** (exit 139) leaving no output, so the driver checks the output exists and carries the right number of points rather than trusting the exit code.

**Column order:** the file `orca_vpot` writes is `x y z V`. The file a CHELPG run writes is `V x y z` (Task 7). Two parsers, each named for its producer, because reading one with the other still yields numbers.

- [ ] **Step 1: Create the fixture**

Create `fixtures/butane/orca/vpot_out.vpot` with exactly this content, captured from the probe:

```
5
     -9.9097238000000001      -2.8799426000000001      -0.7256548400000000       0.0001970162336823
     -9.9097238000000001      -2.8799426000000001      -0.1587370000000000       0.0002363383587169
     -9.9097238000000001      -2.8799426000000001       0.4081808400000000       0.0002249275852303
     -9.9097238000000001      -2.8799426000000001       0.9750986900000000       0.0001661090260327
     -9.9097238000000001      -2.3130248000000000      -1.8594904999999999      -0.0001231245707903
```

These are the same five points as `esp_head.vpot`, with the same potentials, in the opposite column order. That is the point of committing both.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_qm_orca.py`:

```python
def test_read_vpot_output_takes_potential_from_the_last_column():
    """orca_vpot writes x y z V, the reverse of the CHELPG .vpot layout."""
    from ffrefine.qm.orca import read_vpot_output

    v = read_vpot_output(ORCA_FIXTURES / "vpot_out.vpot", expected_points=5)

    assert v.shape == (5,)
    assert v[0] == pytest.approx(1.9701622336823e-04)
    assert v[4] == pytest.approx(-1.231245707903e-04)


def test_the_two_vpot_layouts_agree_on_the_potential():
    """Same five points from the same run, written in opposite column orders.

    If either parser reads the wrong column this test fails, which is the whole
    reason both fixtures are committed.
    """
    from ffrefine.qm.orca import read_chelpg_vpot, read_vpot_output

    _, from_chelpg = read_chelpg_vpot(ORCA_FIXTURES / "esp_head.vpot")
    from_vpot = read_vpot_output(ORCA_FIXTURES / "vpot_out.vpot", expected_points=5)

    assert from_chelpg == pytest.approx(from_vpot, rel=1e-6)


def test_read_vpot_output_refuses_a_short_file(tmp_path):
    """A wrong density name segfaults orca_vpot; the output is the only witness."""
    from ffrefine.qm.orca import read_vpot_output

    path = tmp_path / "short.vpot"
    path.write_text("2\n 0.0 0.0 0.0 1.0\n")

    with pytest.raises(ExternalToolError, match="2"):
        read_vpot_output(path, expected_points=2)


def test_write_vpot_grid_writes_bohr_with_a_count_header(tmp_path):
    from ffrefine.qm.orca import write_vpot_grid

    path = tmp_path / "grid.xyz"
    write_vpot_grid(path, np.array([[0.0529177210903, 0.0, 0.0]]))

    lines = path.read_text().splitlines()
    assert lines[0].strip() == "1"
    assert float(lines[1].split()[0]) == pytest.approx(1.0, rel=1e-9)


@pytest.mark.needs_orca
def test_run_vpot_reproduces_orcas_own_potential(tmp_path, butane_pdb, orca_bin):
    """Checked against ORCA's own number, not merely 'it ran'."""
    from ffrefine.config import Binaries
    from ffrefine.io.structure import read_pdb
    from ffrefine.qm.orca import read_chelpg_vpot, run_esp_job, run_vpot

    s = read_pdb(butane_pdb)
    binaries = Binaries(orca=orca_bin)
    base = run_esp_job(binaries, s, s.coords, tmp_path, "! HF STO-3G", charge=0, mult=1)

    grid_nm, reference = read_chelpg_vpot(Path(str(base) + ".vpot"))
    subset = grid_nm[:20]
    v = run_vpot(binaries, base, subset, tmp_path)

    assert v.shape == (20,)
    assert v == pytest.approx(reference[:20], rel=1e-6)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_qm_orca.py -q -k vpot`
Expected: FAIL with `ImportError: cannot import name 'read_vpot_output'`.

- [ ] **Step 4: Implement**

Add to `ffrefine/qm/orca.py` (add `nm_to_bohr` to the existing `..units` import):

```python
def write_vpot_grid(path: str | Path, grid_nm: np.ndarray) -> None:
    """Write grid points for ``orca_vpot``: a count line, then bohr coordinates."""
    grid = nm_to_bohr(np.asarray(grid_nm, float))
    lines = [str(len(grid))]
    lines += [f"{x:>22.14f} {y:>22.14f} {z:>22.14f}" for x, y, z in grid]
    Path(path).write_text("\n".join(lines) + "\n", newline="\n")


def read_vpot_output(path: str | Path, expected_points: int) -> np.ndarray:
    """Read the potential from the file ``orca_vpot`` writes.

    **Columns are ``x y z V``** here -- the reverse of the ``.vpot`` a CHELPG
    run writes, which is ``V x y z``.  Two functions rather than one flag,
    because the wrong choice produces a plausible array either way.

    Args:
        path: the output file.
        expected_points: how many points were sent.  Checked, because a wrong
            density name makes ``orca_vpot`` segfault, and a truncated or
            missing file is the only evidence of that.

    Returns:
        Potentials in Hartree/e, shape ``(expected_points,)``.

    Raises:
        ExternalToolError: the file is missing, short, or malformed.
    """
    path = Path(path)
    if not path.exists():
        raise ExternalToolError(
            f"orca_vpot wrote no {path}. It exits by segfault when it cannot read the "
            "density, so an absent output usually means the density name was wrong."
        )
    rows = [ln.split() for ln in path.read_text().splitlines()[1:] if ln.strip()]
    if len(rows) != expected_points:
        raise ExternalToolError(
            f"{path}: {len(rows)} potential values for {expected_points} grid points. "
            "orca_vpot did not finish; check vpot.stdout in the stage directory."
        )
    return np.array([float(row[3]) for row in rows], dtype=float)


def run_vpot(
    binaries: Binaries,
    base: Path,
    grid_nm: np.ndarray,
    workdir: str | Path,
) -> np.ndarray:
    """Evaluate the QM potential at ``grid_nm`` against a finished ESP job.

    The invocation was established by probe against ORCA 6.1.1 rather than
    taken from the usage text, as spec 5.2 requires:

        orca_vpot <base>.gbw <base>.scfp grid.xyz out.vpot

    The density argument is the **full** ``<base>.scfp``.  Passing the bare
    ``scfp`` that the usage text suggests segfaults (exit 139), and the optional
    fifth argument is unnecessary.

    Returns:
        Potentials in Hartree/e, one per grid point.

    Raises:
        ExternalToolError: orca_vpot failed, or produced the wrong point count.
    """
    exe = binaries.require("orca_vpot")
    workdir = Path(workdir)
    grid_path = workdir / "grid.xyz"
    out_path = workdir / "grid_out.vpot"
    write_vpot_grid(grid_path, grid_nm)
    run_command(
        [exe, f"{base.name}.gbw", f"{base.name}.scfp", grid_path.name, out_path.name],
        workdir,
        "vpot",
        extra_env=orca_env(binaries),
        check=False,
    )
    return read_vpot_output(out_path, expected_points=len(grid_nm))
```

`check=False` is deliberate: `orca_vpot` dies by signal on a bad density, and `read_vpot_output`'s message about the density is more use than a generic nonzero-exit report.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_qm_orca.py -q -rs` in an interactive WSL shell.
Expected: PASS, zero skipped.

- [ ] **Step 6: Commit**

```bash
python -m black ffrefine tests -q
git add ffrefine/qm/orca.py tests/test_qm_orca.py fixtures/butane/orca/vpot_out.vpot
git commit -m "feat(qm): orca_vpot driver with a verified invocation"
```

---

### Task 13: `charges/resp.py` — the two-stage restrained fit

**Files:**
- Create: `ffrefine/charges/resp.py`
- Test: `tests/test_charges_resp.py`

**Interfaces:**
- Consumes: `charges.esp.normal_equations`, `charges.esp.potential_from_charges`, `charges.esp.rrms`.
- Produces: `resp.RESP_A_STAGE1 = 0.0005`, `resp.RESP_A_STAGE2 = 0.001`, `resp.RESP_B = 0.1`; `resp.RespResult` dataclass with `charges: np.ndarray`, `rrms: float`, `iterations: tuple[int, int]`; `resp.resp_fit(inv_r, v_ref, *, restrain, groups, stage2_free, total_charge, a_stage1=..., a_stage2=..., b=..., max_iterations=100, tol=1e-8) -> RespResult`. `groups` and `stage2_free` are **0-based**.

**Why:** Design section 6.3, from spec 5.2. This is the piece most likely to be wrong in a way that still produces plausible numbers, which is why the first test fits a potential generated *from* known charges and demands them back.

Constraints are all handled the same way, as extra rows in a bordered system: one row for the total charge, one per equivalence pair, and one per frozen atom in stage 2. Uniform beats three mechanisms.

- [ ] **Step 1: Write the failing test**

Create `tests/test_charges_resp.py`:

```python
import numpy as np
import pytest

from ffrefine.charges.esp import inverse_distances, potential_from_charges
from ffrefine.charges.resp import resp_fit
from ffrefine.errors import ValidationError


def _case(seed: int = 0, natoms: int = 5):
    rng = np.random.default_rng(seed)
    atoms = rng.normal(size=(natoms, 3)) * 2.0
    grid = rng.normal(size=(900, 3)) * 9.0
    return atoms, grid


def test_an_unrestrained_fit_recovers_the_charges_that_made_the_potential():
    """The load-bearing test: a transposed matrix or a wrong sign dies here."""
    atoms, grid = _case()
    truth = np.array([0.4, -0.3, 0.25, -0.2, -0.15])
    inv_r = inverse_distances(atoms, grid)
    v = potential_from_charges(inv_r, truth)

    out = resp_fit(
        inv_r,
        v,
        restrain=np.zeros(5, dtype=bool),
        groups=[],
        stage2_free=[],
        total_charge=float(truth.sum()),
        a_stage1=0.0,
        a_stage2=0.0,
    )

    assert out.charges == pytest.approx(truth, abs=1e-8)
    assert out.rrms == pytest.approx(0.0, abs=1e-10)


def test_the_total_charge_constraint_is_exact():
    atoms, grid = _case(seed=1)
    truth = np.array([0.4, -0.3, 0.25, -0.2, -0.15])
    inv_r = inverse_distances(atoms, grid)
    v = potential_from_charges(inv_r, truth)

    out = resp_fit(
        inv_r,
        v,
        restrain=np.ones(5, dtype=bool),
        groups=[],
        stage2_free=[],
        total_charge=-1.0,
    )

    assert out.charges.sum() == pytest.approx(-1.0, abs=1e-10)


def test_the_restraint_pulls_heavy_atom_charges_toward_zero():
    """That is the whole purpose of RESP over a plain ESP fit."""
    atoms, grid = _case(seed=2)
    truth = np.array([0.9, -0.8, 0.5, -0.4, -0.2])
    inv_r = inverse_distances(atoms, grid)
    v = potential_from_charges(inv_r, truth)
    common = dict(groups=[], stage2_free=[], total_charge=float(truth.sum()))

    free = resp_fit(inv_r, v, restrain=np.zeros(5, dtype=bool), a_stage1=0.0, **common)
    tight = resp_fit(inv_r, v, restrain=np.ones(5, dtype=bool), a_stage1=0.01, **common)

    assert np.abs(tight.charges).sum() < np.abs(free.charges).sum()


def test_stage_two_makes_an_equivalence_group_exactly_equal():
    atoms, grid = _case(seed=3)
    truth = np.array([-0.4, 0.1, 0.12, 0.09, 0.09])
    inv_r = inverse_distances(atoms, grid)
    v = potential_from_charges(inv_r, truth)

    out = resp_fit(
        inv_r,
        v,
        restrain=np.array([True, False, False, False, False]),
        groups=[[1, 2, 3]],
        stage2_free=[0, 1, 2, 3],
        total_charge=float(truth.sum()),
    )

    assert out.charges[1] == pytest.approx(out.charges[2], abs=1e-10)
    assert out.charges[2] == pytest.approx(out.charges[3], abs=1e-10)
    assert out.charges.sum() == pytest.approx(truth.sum(), abs=1e-10)


def test_atoms_outside_stage_two_keep_their_stage_one_values():
    atoms, grid = _case(seed=4)
    truth = np.array([-0.4, 0.1, 0.12, 0.09, 0.09])
    inv_r = inverse_distances(atoms, grid)
    v = potential_from_charges(inv_r, truth)
    common = dict(restrain=np.ones(5, dtype=bool), total_charge=float(truth.sum()))

    stage1_only = resp_fit(inv_r, v, groups=[], stage2_free=[], **common)
    both = resp_fit(inv_r, v, groups=[[1, 2]], stage2_free=[1, 2], **common)

    assert both.charges[4] == pytest.approx(stage1_only.charges[4], abs=1e-10)


def test_a_fit_that_will_not_converge_is_refused():
    """Returning an unconverged answer would look like a converged one."""
    atoms, grid = _case(seed=5)
    inv_r = inverse_distances(atoms, grid)
    v = potential_from_charges(inv_r, np.array([0.4, -0.3, 0.25, -0.2, -0.15]))

    with pytest.raises(ValidationError, match="converge"):
        resp_fit(
            inv_r,
            v,
            restrain=np.ones(5, dtype=bool),
            groups=[],
            stage2_free=[],
            total_charge=0.0,
            max_iterations=1,
            tol=1e-15,
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_charges_resp.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ffrefine.charges.resp'`.

- [ ] **Step 3: Implement**

Create `ffrefine/charges/resp.py`:

```python
"""The two-stage restrained ESP fit (spec 5.2).

A plain least-squares fit of charges to an electrostatic potential is badly
conditioned: buried atoms barely affect the potential outside the molecule, so
the fit assigns them large, arbitrary charges that happen to cancel.  RESP adds
a hyperbolic restraint pulling heavy-atom charges toward zero, which costs
almost nothing in fit quality and removes most of that arbitrariness.

Stage 1 fits every atom.  Stage 2 refits only the atoms in equivalence groups
and their parents, with a stronger restraint and the group members constrained
equal, leaving everything else at its stage 1 value -- so imposing methyl
symmetry cannot quietly rearrange the rest of the molecule.

Every constraint is an extra row in a bordered system: one for the total
charge, one per equivalence pair, one per frozen atom.  One mechanism rather
than three.

Nothing here runs a subprocess.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import ValidationError
from .esp import normal_equations, potential_from_charges, rrms

RESP_A_STAGE1 = 0.0005
"""Stage 1 restraint strength (spec 5.2)."""

RESP_A_STAGE2 = 0.001
"""Stage 2 restraint strength: stronger, over far fewer atoms (spec 5.2)."""

RESP_B = 0.1
"""Hyperbolic restraint width, in e (spec 5.2)."""


@dataclass
class RespResult:
    """Outcome of a two-stage fit.

    Attributes:
        charges: shape (natoms,) in e.
        rrms: relative RMS deviation of the fitted potential from the QM one.
        iterations: iterations used by stage 1 and stage 2.
    """

    charges: np.ndarray
    rrms: float
    iterations: tuple[int, int]


def _constraint_rows(
    natoms: int,
    total_charge: float,
    groups: list[list[int]],
    frozen: dict[int, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``C`` and ``d`` for the constraints ``C q = d``."""
    rows: list[np.ndarray] = []
    values: list[float] = []

    total = np.ones(natoms)
    rows.append(total)
    values.append(float(total_charge))

    for group in groups:
        for other in group[1:]:
            row = np.zeros(natoms)
            row[group[0]] = 1.0
            row[other] = -1.0
            rows.append(row)
            values.append(0.0)

    for index, value in sorted(frozen.items()):
        row = np.zeros(natoms)
        row[index] = 1.0
        rows.append(row)
        values.append(float(value))

    return np.array(rows), np.array(values)


def _solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Solve the bordered system ``[[A, C^T], [C, 0]] [q; lambda] = [b; d]``."""
    n = len(b)
    k = len(d)
    matrix = np.zeros((n + k, n + k))
    matrix[:n, :n] = a
    matrix[:n, n:] = c.T
    matrix[n:, :n] = c
    rhs = np.concatenate([b, d])
    try:
        solution = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValidationError(
            f"the ESP fit is singular ({exc}). This usually means two atoms share a "
            "position, or the grid has too few points to determine every charge."
        ) from None
    return solution[:n]
```

Continue the same file:

```python
def _run_stage(
    a_matrix: np.ndarray,
    b_vector: np.ndarray,
    constraints: tuple[np.ndarray, np.ndarray],
    restrain: np.ndarray,
    a_strength: float,
    b_width: float,
    max_iterations: int,
    tol: float,
    stage: int,
) -> tuple[np.ndarray, int]:
    """Iterate one restrained stage to convergence.

    The hyperbolic penalty ``a * sum(sqrt(q^2 + b^2) - b)`` is not quadratic, so
    its second derivative depends on the charges.  Linearising it as a diagonal
    ``a / sqrt(q^2 + b^2)`` built from the previous iteration's charges and
    re-solving is the standard treatment, and it converges in a handful of
    passes.
    """
    c, d = constraints
    charges = _solve(a_matrix, b_vector, c, d)
    for iteration in range(1, max_iterations + 1):
        diagonal = np.where(
            restrain, a_strength / np.sqrt(charges**2 + b_width**2), 0.0
        )
        updated = _solve(a_matrix + np.diag(diagonal), b_vector, c, d)
        movement = float(np.abs(updated - charges).max())
        charges = updated
        if movement < tol:
            return charges, iteration
    raise ValidationError(
        f"the RESP stage {stage} fit did not converge in {max_iterations} iterations; the "
        f"charges were still moving by {movement:.2e} e (tolerance {tol:.0e}). Returning an "
        "unconverged fit would be indistinguishable from a converged one."
    )


def resp_fit(
    inv_r: np.ndarray,
    v_ref: np.ndarray,
    *,
    restrain: np.ndarray,
    groups: list[list[int]],
    stage2_free: list[int],
    total_charge: float,
    a_stage1: float = RESP_A_STAGE1,
    a_stage2: float = RESP_A_STAGE2,
    b: float = RESP_B,
    max_iterations: int = 100,
    tol: float = 1e-8,
) -> RespResult:
    """Fit charges to a potential with the two-stage restrained procedure.

    Args:
        inv_r: from :func:`ffrefine.charges.esp.inverse_distances`, shape
            (npoints, natoms).
        v_ref: QM potential at those points, Hartree/e.
        restrain: per-atom mask, True where the hyperbolic restraint applies.
            Spec 5.2 restrains non-hydrogen atoms.
        groups: **0-based** equivalence groups, constrained equal in stage 2.
        stage2_free: **0-based** atoms refit in stage 2; everything else is
            frozen at its stage 1 charge.  Empty means stage 2 does nothing.
        total_charge: the molecular charge, enforced exactly.
        a_stage1, a_stage2: restraint strengths.
        b: restraint width.
        max_iterations: cap per stage.
        tol: convergence threshold on the largest charge movement, in e.

    Returns:
        A :class:`RespResult`.

    Raises:
        ValidationError: either stage failed to converge, or the system is
            singular.
    """
    inv_r = np.asarray(inv_r, float)
    natoms = inv_r.shape[1]
    restrain = np.asarray(restrain, bool)
    if restrain.shape != (natoms,):
        raise ValidationError(
            f"restrain mask has shape {restrain.shape}, expected ({natoms},)"
        )

    a_matrix, b_vector = normal_equations(inv_r, v_ref)

    stage1, iter1 = _run_stage(
        a_matrix,
        b_vector,
        _constraint_rows(natoms, total_charge, [], {}),
        restrain,
        a_stage1,
        b,
        max_iterations,
        tol,
        stage=1,
    )

    charges, iter2 = stage1, 0
    if stage2_free:
        free = set(stage2_free)
        frozen = {i: float(stage1[i]) for i in range(natoms) if i not in free}
        active_groups = [g for g in groups if all(i in free for i in g)]
        charges, iter2 = _run_stage(
            a_matrix,
            b_vector,
            _constraint_rows(natoms, total_charge, active_groups, frozen),
            restrain,
            a_stage2,
            b,
            max_iterations,
            tol,
            stage=2,
        )

    return RespResult(
        charges=charges,
        rrms=rrms(v_ref, potential_from_charges(inv_r, charges)),
        iterations=(iter1, iter2),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_charges_resp.py -q`
Expected: PASS (6 tests). If the round-trip test fails by a sign, check `normal_equations` before touching this module — the bug is more likely there.

- [ ] **Step 5: Commit**

```bash
python -m black ffrefine tests -q
git add ffrefine/charges/resp.py tests/test_charges_resp.py
git commit -m "feat(charges): two-stage restrained ESP fit"
```

---

### Task 14: Wire `--charge-scheme resp` into `cmd_charges`

**Files:**
- Modify: `ffrefine/cli.py` (`cmd_charges`)
- Test: `tests/test_cli_charges.py`

**Interfaces:**
- Consumes: `grid.mk_grid`, `orca.run_vpot`, `resp.resp_fit`, `validate.element_of`, `equivalence.equivalent_hydrogen_groups`.
- Produces: no new names; `charges.json` gains `grid` and `restraint` provenance.

**Why:** Design section 4. Everything the RESP branch needs now exists; this replaces the single `parse_chelpg_charges` line with a branch.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli_charges.py`:

```python
@pytest.mark.needs_orca
def test_resp_end_to_end_gives_equal_methyl_hydrogens(
    butane_pdb, butane_itp, tmp_path, orca_bin
):
    """Stage 2 constrains the group exactly equal, not merely close."""
    outdir = tmp_path / "out"
    rc = main(
        _args(
            butane_pdb,
            butane_itp,
            outdir,
            "--charge-scheme",
            "resp",
            "--charge-mode",
            "all",
            "--charge-geom",
            "input",
            "--orca-esp-keywords",
            "! HF STO-3G",
            "--esp-density",
            "1.0",
            "--nthreads",
            "1",
        )
    )
    assert rc == 0

    payload = json.loads((outdir / "04_charges" / "charges.json").read_text())
    assert payload["provenance"]["scheme"] == "resp"
    assert payload["provenance"]["restraint"]["a_stage1"] == pytest.approx(0.0005)
    assert payload["provenance"]["grid"]["npoints"] > 500
    assert payload["total_after"] == pytest.approx(0.0, abs=1e-6)

    charged = read_topology(outdir / "04_charges" / "molecule_charged.itp")
    assert charged.atoms[1].charge == charged.atoms[2].charge == charged.atoms[3].charge


@pytest.mark.needs_orca
def test_resp_and_chelpg_broadly_agree_on_butane(butane_pdb, butane_itp, tmp_path, orca_bin):
    """Two schemes on one molecule should not disagree about its chemistry."""
    charges = {}
    for scheme in ("chelpg", "resp"):
        outdir = tmp_path / scheme
        assert (
            main(
                _args(
                    butane_pdb,
                    butane_itp,
                    outdir,
                    "--charge-scheme",
                    scheme,
                    "--charge-mode",
                    "all",
                    "--charge-geom",
                    "input",
                    "--orca-esp-keywords",
                    "! HF STO-3G",
                    "--nthreads",
                    "1",
                )
            )
            == 0
        )
        payload = json.loads((outdir / "04_charges" / "charges.json").read_text())
        charges[scheme] = np.array([a["charge_after"] for a in payload["atoms"]])

    # RESP restrains the buried carbons, so the two differ; they must still
    # agree on the sign of every terminal carbon and be within 0.2 e overall.
    assert np.abs(charges["chelpg"] - charges["resp"]).max() < 0.2
    assert np.sign(charges["chelpg"][0]) == np.sign(charges["resp"][0])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_cli_charges.py -q -k resp`
Expected: FAIL — `--charge-scheme resp` currently runs the CHELPG path and writes no `restraint` provenance.

- [ ] **Step 3: Implement**

In `cmd_charges`, replace these three lines from Task 9

```python
    raw = parse_chelpg_charges(Path(str(base) + ".chelpg.xyz"))

    groups = equivalent_hydrogen_groups(topology)
    raw = average_within_groups(raw, groups)
```

with the branch below. `groups` moves above the branch because RESP needs it
too, and `average_within_groups` moves inside the CHELPG arm: RESP enforces the
same equality as a stage-2 constraint, so averaging its output afterwards would
apply the constraint twice and quietly undo the fit's own answer.

```python
    groups = equivalent_hydrogen_groups(topology)
    grid_provenance: dict[str, object] | None = None
    restraint_provenance: dict[str, object] | None = None

    if args.charge_scheme == "chelpg":
        raw = parse_chelpg_charges(Path(str(base) + ".chelpg.xyz"))
        raw = average_within_groups(raw, groups)
    else:
        elements = [element_of(topology, atom.nr) for atom in topology.atoms]
        grid_nm = mk_grid(coords_nm, elements, density_per_angstrom2=args.esp_density)
        v_grid = run_vpot(binaries, base, grid_nm, workdir)
        inv_r_fit = inverse_distances(nm_to_bohr(coords_nm), nm_to_bohr(grid_nm))
        zero_based = [[nr - 1 for nr in group] for group in groups]
        parents = {
            n - 1
            for group in groups
            for n in topology.bonded_neighbours(group[0])
            if element_of(topology, n) != "H"
        }
        stage2_free = sorted(parents.union(i for g in zero_based for i in g))
        result = resp_fit(
            inv_r_fit,
            v_grid,
            restrain=np.array([e != "H" for e in elements]),
            groups=zero_based,
            stage2_free=stage2_free,
            total_charge=float(target_total),
        )
        raw = result.charges
        grid_provenance = {
            "npoints": int(len(grid_nm)),
            "density_per_angstrom2": float(args.esp_density),
            "shell_scales": list(MK_SHELL_SCALES),
            "radii_angstrom": {e: VDW_RADII_ANGSTROM[e] for e in sorted(set(elements))},
        }
        restraint_provenance = {
            "a_stage1": RESP_A_STAGE1,
            "a_stage2": RESP_A_STAGE2,
            "b": RESP_B,
            "iterations": list(result.iterations),
        }
```

The existing RRMS block stays exactly as it is: it measures both schemes against ORCA's own CHELPG grid, so the two numbers are comparable (design section 4). Then add the two blocks to the provenance dict:

```python
            "grid": grid_provenance,
            "restraint": restraint_provenance,
```

New imports in `cli.py`:

```python
from .charges.grid import MK_SHELL_SCALES, VDW_RADII_ANGSTROM, mk_grid
from .charges.resp import RESP_A_STAGE1, RESP_A_STAGE2, RESP_B, resp_fit
from .io.validate import element_of  # add to the existing io.validate import
from .qm.orca import run_vpot  # add to the existing qm.orca import
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_cli_charges.py -q -rs`, then the full suite, in an interactive WSL shell.
Expected: PASS, zero skipped.

- [ ] **Step 5: Commit**

```bash
python -m black ffrefine tests -q
git add ffrefine/cli.py tests/test_cli_charges.py
git commit -m "feat(cli): --charge-scheme resp, fitting on our own MK grid"
```

---

### Task 15: Documentation

**Files:**
- Modify: `README.md`

**Interfaces:** none.

**Why:** The README lists `ffrefine charges` as not implemented and describes `run` as chaining four stages in an order this milestone changes. A README that misdescribes the tool is the same class of defect as ruling R40's comment.

- [ ] **Step 1: Update the status table**

Change the `ffrefine charges` row from **not implemented** to a description: ESP-derived charges via ORCA, CHELPG or RESP, equivalent hydrogens constrained, `selected` by default. Update the `ffrefine run` row: it now chains five stages, and skips charges unless asked. Change the heading from "milestones 1-4 of 6" to "milestones 1-5 of 6", and remove milestone 5 from the "Also not implemented" paragraph, leaving the milestone 6 items (`--str` parsing and `final/report.md`) and `--mm-relax`.

- [ ] **Step 2: Add a worked charges section to the butane example**

Insert before "1. MM energies", since charges now come first:

```markdown
### 1. Charges (optional, but first if you want them)

    ffrefine charges \
        -s fixtures/butane/butane.pdb \
        -p fixtures/butane/butane.itp \
        --charge-atoms 1,11 \
        --outdir butane_out

This optimises the geometry with xtb, runs one ORCA ESP job, fits charges to
it, and writes `butane_out/04_charges/molecule_charged.itp` along with
`charges.json` recording every atom's charge before and after and why it
changed.

**Charges come before the torsion, not after.** The fitted torsion absorbs
whatever the 1-4 electrostatics do not already account for, so a torsion fitted
against one set of charges does not belong in a topology carrying another. Pass
the charge-refined topology to the later stages:

    ffrefine mmscan -s ... -p butane_out/04_charges/molecule_charged.itp ...

`ffrefine run` does that swap for you.
```

Renumber the existing sections that follow.

- [ ] **Step 3: Document the flags**

Add a "Charge flags" subsection next to the scan flags, covering `--charge-scheme`, `--charge-geom`, `--charge-mode`, `--charge-atoms`, `--orca-esp-keywords` and `--esp-density`, each with its default. State plainly that:

- `--charge-mode selected` is the default and **refuses to run without `--charge-atoms`**, because replacing every charge on a CGenFF topology changes the derivation scheme. Automatic selection from CGenFF penalties arrives in milestone 6.
- `--orca-esp-keywords` carries the level of theory only; the scheme adds `CHELPG` itself.
- ORCA is usually on no `PATH`, so `--orca-bin` is the normal way to reach it, and a shared build needs its own directory on `LD_LIBRARY_PATH`, which ffrefine sets from the binary's location.

- [ ] **Step 4: State what the charges do not do**

Extend "What the scan does not do" into a section covering both. For charges:

- Equivalence is **local**: hydrogens on the same heavy atom with the same type. Molecule-wide symmetry is not detected, so butane's two terminal carbons come back a few thousandths of an electron apart. That is ESP fitting noise and is reported, not smoothed away.
- Gas-phase ESP charges are not condensed-phase charges. HF/6-31G* is chosen because its overpolarisation partly compensates, but `--charge-mode all` on a CGenFF topology is still a change of derivation scheme.
- `--conformers` (multi-conformer fitting) and `--charge-geom orca` are not implemented.

- [ ] **Step 5: Update the output layout**

`04_charges/` is no longer "part of the layout but not produced". Describe it: `charges.json`, `molecule_charged.itp`, the ORCA ESP inputs and outputs, `opt/` when `--charge-geom xtb`, and `grid.xyz` plus `grid_out.vpot` under `--charge-scheme resp`.

- [ ] **Step 6: Verify and commit**

Run, in an interactive WSL shell:

```bash
python -m pytest -q -rs
python -m pytest -q -rs -m "needs_gmx or needs_xtb or needs_orca"
python -m black --check .
```

Expected: all pass, **zero skipped**, and the live-marker counts should now be 16 `needs_gmx`, 3 `needs_xtb` and 6 or more `needs_orca`.

```bash
git add README.md
git commit -m "docs: ffrefine charges is implemented; document its flags and limits"
```

---

## Self-review

**Spec coverage.** Spec 5.1 geometry: Tasks 8 and 9 (`input` and `xtb`; `orca` and `--conformers` are out of scope per design section 1). Spec 5.2 CHELPG: Tasks 7 and 9. Spec 5.2 RESP grid, `orca_vpot`, two-stage fit and RRMS: Tasks 11, 12, 13, 14. Spec 5.3 equivalence: Task 4. Spec 5.3 `selected`/`all`, redistribution, rounding, CGenFF warning: Tasks 5 and 9. Spec section 6's `04_charges/`: Task 9. Spec section 8 resumability: Task 9's `hash_inputs`/`write_manifest`. Design section 5's ordering: Task 10. Design section 9's cleanup: Task 2. Documentation: Task 15.

**Type consistency.** `apply_charges(topology, fitted_e, selected, target_total)` is called with those names in Task 9. `ChargeAssignment` fields (`nr`, `name`, `before`, `after`, `source`) are used by Task 6's writer and Task 9's payload. `inverse_distances` returns `(npoints, natoms)` in Task 3 and is consumed that way in Tasks 9, 13 and 14. `equivalent_hydrogen_groups` returns 1-based groups throughout; Task 14 converts to 0-based explicitly before calling `resp_fit`, which documents its inputs as 0-based. `run_esp_job` returns a base `Path` in Task 7 and is used as `str(base) + ".chelpg.xyz"` in Task 9 and as `base.name` in Task 12's `run_vpot`.

**Known follow-ups**, carried from design section 10: frozen-atom fitting for RESP `selected` mode; molecule-wide symmetry; whether the ESP job should run in parallel.
