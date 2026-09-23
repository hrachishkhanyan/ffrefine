# Milestone 6b — CGenFF penalties, automatic selection, multi-bond runs, `report.md` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `ffrefine run --str mol.str` reads CGenFF's penalties, refines the flagged charges, refits every flagged torsion in sequence (one subfolder per bond), checks the combined result, and writes one topology carrying every change plus a standalone `final/report.md`.

**Architecture:** Three new pure modules — `ffrefine/io/cgenff.py` (parse the `.str`), `ffrefine/selection.py` (map it onto the topology and decide), `ffrefine/report.py` (write the report from the JSON payloads on disk). `run` re-invokes the existing stage commands once per bond with a derived outdir and a chained topology, then runs a combined check. Every stage payload keeps its own warnings, collected at the logging layer. Six small prerequisite fixes come first.

**Tech Stack:** Python 3.10+, numpy, matplotlib, pytest, black (line-length 100). GROMACS 2025.4, xtb 6.7.1, ORCA 6.1.1 via WSL. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-23-ffrefine-m6b-cgenff-report-design.md`

## Global Constraints

- All user-facing atom indices are **1-based**; internal numpy arrays are 0-based; convert at the boundary and say so in the docstring (spec section 8).
- Type hints throughout, docstrings on public functions, `black` formatting at line-length 100, target py310.
- Every external command goes through `ffrefine.runner.run_command` or the existing `_xtb_version` / `_orca_version` helpers; every grompp and gmx invocation lives in `ffrefine/mm/gromacs.py`.
- `ffrefine/io/` and `ffrefine/charges/` never start a process. `ffrefine/io/cgenff.py` imports nothing from `ffrefine` except `ffrefine.errors`. `ffrefine/selection.py` and `ffrefine/report.py` start no process.
- Penalty thresholds default to **charge 10** and **dihedral 50**, and a penalty is flagged when it is **>=** the threshold (inclusive, both kinds).
- Bond subfolders are named `torsion_i-j-k-l` after the bond's quadruplet, e.g. `torsion_1-3-4-5`.
- New files: `00_input/selection.json` (plus a copy of the `.str` beside it), `final/run_status.json`, `final/combined.json`, `final/report.md`, `final/plots/combined_<subfolder>.png`.
- One torsion keeps today's layout in `--outdir` itself; two or more get one subfolder each.
- Input files are never modified.
- Tests must be run in an **interactive** WSL shell: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest -q -rs'`. A login shell (`bash -lc`) hides ORCA and a Windows `python` cannot see gmx or xtb, so the `needs_*` tests then skip while reporting green. **Require zero skipped.**
- WSL's `/tmp` is wiped between separate `wsl` invocations. Scratch that must survive between commands goes under `$HOME/`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Branch: `ffrefine/milestone-6b-cgenff-report`. Do not work on `master`.

## Review Focus

The inputs and conditions the spec implies but no task's main tests exercise, most likely first. Each has a test in the task named.

1. **`--resume` on a finished multi-bond run** must skip every bond and the combined check, and the report must come out with the same content (Task 9: `test_combined_check_resumes`).
2. **Payloads written by an older build** (no `warnings`, no `torsion_parameters`, no `charges_json`, no `defaults`) must give a report with "not recorded" where the data is missing, never a `KeyError` (Task 11: `test_report_tolerates_payloads_from_an_older_build`).
3. **A `.str` whose atoms are reordered among atoms of the same type**, so the type check passes, must still be caught by the charge comparison (Task 7: `test_a_reordering_within_one_type_is_caught_by_the_charges`).
4. **A `.str` saved with Windows line endings** must parse the same as the original (Task 6: `test_crlf_line_endings_parse_the_same`).
5. **A `--str` path that does not exist** must give a clean one-line error, not a traceback (Task 6: `test_a_missing_file_is_a_parse_error`; Task 8: `test_a_missing_str_is_a_clean_error`).

## Refinements of the design, fixed here

1. **Warnings are collected at the logging layer, not only through `_warn`.** Today only `cli._warn` feeds `_collect_warnings`. GROMACS's echoed warnings, the self-check's structure warnings and the fit's own warnings are plain `log.warning` calls inside modules and never reached a payload. A `WarningCollector` logging handler (in `stages.py`, which owns logging) now captures every WARNING record on the `ffrefine` logger while a stage runs; `setup_logging` leaves collectors attached.
2. **`model.json` gets no `warnings` list.** Every stage rewrites `model.json` and every stage's own payload already carries the warnings its input step raised (they are emitted inside the same collector). Adding them to `model.json` too would only duplicate them in the report.
3. **The xtb version is already recorded** (`scan.json` provenance `xtb_version`), and `charges.json` already records `orca_version`. The only missing version is ORCA's under `--sp-refine`; Task 3 adds it.
4. **Design open question 1 (combined-check plots):** `plot_validation` gains a keyword `title`; the combined check reuses it with "Combined check: original topology against the final one". No new plotting function.
5. **Design open question 2 (status of a failed bond):** `run` writes `final/run_status.json` as it goes; `report.py` reads it when present and infers status from the payloads when not.
6. **`selection.py` and `report.py` live at the package top level**, beside `stages.py`, as the design's architecture section says.
7. **Standalone `charges --str` with no flagged atom** prints that nothing is flagged and exits 0 without running ORCA (no `charges.json`), rather than an error: nothing was asked for that could fail.
8. **Every bond subfolder has its own `run.log`.** `setup_logging` attaches `run.log` to whichever `--outdir` the stage runs in; the top-level `run.log` holds selection, charges and the combined check.

## Facts established while planning

Measured on the committed fixtures with the prototype parser and selection (the code in Tasks 6 and 7):

| | PFOA | capecitabine |
|---|---|---|
| `.str` atoms | 26 | 47 |
| parameters (bond, angle, dihedral, improper) | 2, 8, 13, 1 | 0, 2, 12, 1 |
| CGenFF versions (program, parameters) | 4.0, 5.0 | 4.0, 5.0 |
| `RESI` param / charge penalty | 69.0 / 41.068 | 50.0 / 5.422 |
| flagged atoms at charge 10 | 23 (not 13, 14, 26) | none |
| bonds at dihedral 50, in run order | `(3,4)` 69 quad `(1,3,4,5)`; `(1,3)` 67 quad `(25,1,3,4)` | `(7,9)` 50 quad `(6,7,9,10)` |
| bonds at dihedral 51 | the same two | none |
| ring bonds excluded at dihedral 0.5 | none | `(10,16)`, `(10,11)`, `(14,15)`, `(15,16)` |

ORCA 6.1.1 (probed 2026-09-23): a converged job prints `*           SCF CONVERGED AFTER   1 CYCLES          *`; with `%scf IgnoreConv true MaxIter 1 end` an unconverged job exits 0, prints `****ORCA TERMINATED NORMALLY****` and `Warning (ORCA_LEANSCF): the SCF did not converge, but SCFConvIgnored is TRUE.`, and prints neither `SCF CONVERGED AFTER` nor `SCF NOT CONVERGED`. `orca --version` prints `Program Version 6.1.1  -  RELEASE   -`.

## File structure

| File | Responsibility | Task |
|---|---|---|
| `ffrefine/io/topology.py` | `CHARGE_TOLERANCE`, `derive_charge` (moved from `qm/xtb.py`) | 1 |
| `ffrefine/qm/xtb.py`, `ffrefine/charges/apply.py` | re-export / import from `io.topology` | 1 |
| `tests/test_layering.py` | `charges/` and `io/` start no process | 1 |
| `ffrefine/cli.py` | R27 write order | 1 |
| `ffrefine/io/topwriter.py` | charge provenance line; strip system sections from the charged topology | 2 |
| `ffrefine/cli.py` | `validation.json` `charges_refined` / `charges_json` | 2 |
| `ffrefine/qm/orca.py` | require `SCF CONVERGED AFTER` | 3 |
| `ffrefine/cli.py` | ORCA version under `--sp-refine` | 3 |
| `ffrefine/mm/tprparams.py` | R9 warning | 4 |
| `ffrefine/stages.py` | `WarningCollector`; `setup_logging` keeps collectors | 5 |
| `ffrefine/cli.py` | `_warn`, `_collect_warnings`, `_current_warnings`, warnings in every payload | 5 |
| `ffrefine/io/cgenff.py` | `.str` parser | 6 |
| `ffrefine/selection.py` | mapping and selection; `selection_to_json` | 7 |
| `ffrefine/cli.py` | `--str`, `--penalty-threshold`, `--no-charges`; `selection.json`; `charges --str`; torsion stages with `--str` | 8 |
| `ffrefine/mm/validate_fit.py`, `ffrefine/plots.py`, `ffrefine/cli.py` | combined check, `combined.json`, `_bond_dirname` | 9 |
| `ffrefine/cli.py` | `run`: charges from `--str`, per-bond loop, `run_status.json`, final topology | 10 |
| `ffrefine/report.py` | `write_report` | 11 |
| `ffrefine/cli.py`, tests | report at the end of `run` and `validate`; live end-to-end tests | 12 |
| `README.md` | flags, layout, `--str` example, R11/R12 corrections | 13 |

---

### Task 1: Charges start no process (R26), and write their topology last (R27)

**Files:**
- Modify: `ffrefine/io/topology.py` (add `CHARGE_TOLERANCE`, `derive_charge`)
- Modify: `ffrefine/qm/xtb.py` (remove them; re-export)
- Modify: `ffrefine/charges/apply.py:19` (import from `io.topology`)
- Modify: `ffrefine/cli.py` (`_charges_stage`: score before writing)
- Create: `tests/test_layering.py`
- Test: `tests/test_cli_charges.py` (append)

**Interfaces:**
- Consumes: the existing `derive_charge` body and docstring in `ffrefine/qm/xtb.py` (lines ~338-388).
- Produces: `ffrefine.io.topology.CHARGE_TOLERANCE: float = 0.05` and `ffrefine.io.topology.derive_charge(topology, override=None, tol=CHARGE_TOLERANCE) -> int`, unchanged in behaviour; `ffrefine.qm.xtb` still exports both names.

- [ ] **Step 1: Write the failing layering test**

Create `tests/test_layering.py`:

```python
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
        m for m in _imported_modules(path) if any(m == f or m.startswith(f + ".") for f in forbidden)
    ]
    assert not bad, f"ffrefine/{subpackage}/{path.name} imports {bad}"


def test_the_checker_resolves_relative_imports():
    """Guard against a checker that silently passes everything."""
    apply = PACKAGE / "charges" / "apply.py"
    assert "ffrefine.io.topology" in _imported_modules(apply)
```

- [ ] **Step 2: Write the failing write-order test**

Append to `tests/test_cli_charges.py`:

```python
def test_a_failed_rrms_read_leaves_no_charged_topology(
    butane_pdb, butane_itp, tmp_path, fake_esp, monkeypatch, capsys
):
    """Milestone 5 ruling R27: the topology is written only once everything it
    is reported with exists, so a failure cannot leave an orphan behind."""
    import ffrefine.cli as cli
    from ffrefine.errors import ExternalToolError

    def unreadable(*args, **kwargs):
        raise ExternalToolError("esp.vpot could not be read")

    monkeypatch.setattr(cli, "_rrms_on_orca_grid", unreadable)
    outdir = tmp_path / "out"
    argv = _args(butane_pdb, butane_itp, outdir, "--charge-atoms", "1,11", "--charge-geom", "input")
    assert main(argv) == 1
    assert "esp.vpot could not be read" in capsys.readouterr().err
    assert not (outdir / "04_charges" / "molecule_charged.itp").exists()
    assert not (outdir / "04_charges" / "charges.json").exists()
```

- [ ] **Step 3: Run them to verify they fail**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_layering.py tests/test_cli_charges.py::test_a_failed_rrms_read_leaves_no_charged_topology -q'`
Expected: `test_pure_subpackages_start_no_process[charges-apply.py]` fails naming `ffrefine.qm.xtb`; the write-order test fails on `assert not (...molecule_charged.itp).exists()`.

- [ ] **Step 4: Move `CHARGE_TOLERANCE` and `derive_charge` to `io/topology.py`**

In `ffrefine/io/topology.py`, change `from ..errors import ParseError` to `from ..errors import ParseError, ValidationError`, and append at the end of the module:

```python
CHARGE_TOLERANCE = 0.05
"""How far the topology's charge sum may sit from an integer before it is refused."""


def derive_charge(
    topology: Topology, override: int | None = None, tol: float = CHARGE_TOLERANCE
) -> int:
    """Determine the total molecular charge for xtb's ``--chrg``.
```

followed by the **existing docstring body and code of `derive_charge` from `ffrefine/qm/xtb.py`, moved verbatim** (everything from "Spec 4.3 requires the flag" to `return int(nearest)`).

In `ffrefine/qm/xtb.py`, delete the `CHARGE_TOLERANCE` constant and the `derive_charge` function, and change `from ..io.topology import Topology` to:

```python
from ..io.topology import CHARGE_TOLERANCE, Topology, derive_charge  # noqa: F401 - re-exported
```

In `ffrefine/charges/apply.py`, replace

```python
from ..io.topology import Topology
from ..io.validate import element_of
from ..qm.xtb import CHARGE_TOLERANCE
```

with

```python
from ..io.topology import CHARGE_TOLERANCE, Topology
from ..io.validate import element_of
```

and update any docstring in that file that names `ffrefine.qm.xtb.CHARGE_TOLERANCE` to `ffrefine.io.topology.CHARGE_TOLERANCE`.

- [ ] **Step 5: Score before writing, in `_charges_stage`**

In `ffrefine/cli.py`, in `_charges_stage`, move the `write_charged_topology(...)` call so it comes **after** the `_rrms_on_orca_grid(...)` call and the RRMS warning block, directly before `payload = {`. The `charged_itp = workdir / "molecule_charged.itp"` assignment stays where it is (the warning's text uses its name). Add above the moved call:

```python
    # Milestone 5 ruling R27: written only once everything reported with it
    # exists, so a failure above cannot leave an orphaned charged topology.
```

- [ ] **Step 6: Run the tests**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_layering.py tests/test_cli_charges.py tests/test_charges_apply.py tests/test_qm_xtb.py -q -rs && python -m pytest -q -rs'`
Expected: all pass, 0 skipped.

- [ ] **Step 7: black and commit**

```bash
wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m black ffrefine tests'
cd /c/Users/hrach/Projects/ffrefine
git add ffrefine/io/topology.py ffrefine/qm/xtb.py ffrefine/charges/apply.py ffrefine/cli.py tests/test_layering.py tests/test_cli_charges.py
git commit -m "fix(charges): charges/ starts no process; write the charged topology last

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: The charged topology says where its charges came from (R28), and is a molecule include (M7)

**Files:**
- Modify: `ffrefine/io/topwriter.py` (`CHARGES_PROVENANCE_PREFIX`, `charges_provenance`, `write_charged_topology`)
- Modify: `ffrefine/cli.py` (`_charges_stage` passes `charges_json`; `cmd_validate` computes `charges_refined` / `charges_json`)
- Test: `tests/test_topwriter.py` (append), `tests/test_cli_charges.py` (update `_fake_charges_stage`; append)

**Interfaces:**
- Consumes: `strip_system_sections`, `mark_include_base`, `include_base` (all in or imported by `topwriter.py`).
- Produces:
  - `ffrefine.io.topwriter.CHARGES_PROVENANCE_PREFIX = "; ffrefine charges: "`
  - `ffrefine.io.topwriter.charges_provenance(path: str | Path) -> Path | None`
  - `write_charged_topology(source, assignments, out_path, decimals=4, charges_json: str | Path | None = None) -> list[int]`
  - `validation.json` keys `"charges_refined": bool` and `"charges_json": str | None`.

- [ ] **Step 1: Write the failing writer tests**

Append to `tests/test_topwriter.py` (it already imports `write_charged_topology`, `write_refit_topology`, `read_topology`, `np`, `Path`, `PFOA`, `MARKER_NOTE`, `MARKER_PREFIX`; add the two new names to its `ffrefine.io.topwriter` import):

```python
from ffrefine.io.topwriter import CHARGES_PROVENANCE_PREFIX, charges_provenance


def test_charged_topology_records_where_its_charges_came_from(tmp_path, butane_itp):
    charges_json = tmp_path / "04_charges" / "charges.json"
    out = tmp_path / "charged.itp"
    write_charged_topology(butane_itp, [], out, charges_json=charges_json)
    assert charges_provenance(out) == charges_json.resolve()
    assert out.read_text().count(CHARGES_PROVENANCE_PREFIX) == 1


def test_charged_topology_replaces_an_earlier_provenance_line(tmp_path, butane_itp):
    first = tmp_path / "a.itp"
    write_charged_topology(butane_itp, [], first, charges_json=tmp_path / "old" / "charges.json")
    second = tmp_path / "b.itp"
    write_charged_topology(first, [], second, charges_json=tmp_path / "new" / "charges.json")
    assert charges_provenance(second) == (tmp_path / "new" / "charges.json").resolve()
    assert second.read_text().count(CHARGES_PROVENANCE_PREFIX) == 1


def test_provenance_survives_the_refit_writer(tmp_path, butane_itp):
    """Every later topology in a chain inherits it, bond after bond."""
    charges_json = tmp_path / "charges.json"
    charged = tmp_path / "charged.itp"
    write_charged_topology(butane_itp, [], charged, charges_json=charges_json)
    t = read_topology(charged)
    refined = tmp_path / "refined.itp"
    args = (t, (1, 5, 8, 11), np.array([1.0]), np.array([0.0]), np.array([3]))
    write_refit_topology(*args, refined, includes="verbatim")
    assert charges_provenance(refined) == charges_json.resolve()


def test_a_topology_without_the_line_has_no_provenance(butane_itp):
    assert charges_provenance(butane_itp) is None


def test_charged_topology_is_a_molecule_include(tmp_path):
    """Milestone 5 finding M7: a charged .top kept [ system ] and [ molecules ]."""
    out = tmp_path / "molecule_charged.itp"
    write_charged_topology(PFOA / "pfoa_gmx.top", [], out)
    text = out.read_text()
    assert "[ system ]" not in text and "[ molecules ]" not in text
    assert "[ moleculetype ]" in text
```

- [ ] **Step 2: Update the fake charges stage and write the failing validation test**

In `tests/test_cli_charges.py`, replace the body of `fake` inside `_fake_charges_stage` so the fake writes the provenance line the real stage writes:

```python
    def fake(a):
        from ffrefine.io.topwriter import CHARGES_PROVENANCE_PREFIX

        cli._prepare(a, need_quad=False)
        charged = cli.stage_dir(a.outdir, "charges") / "molecule_charged.itp"
        charges_json = (charged.parent / "charges.json").resolve()
        charged.write_text(
            f"{CHARGES_PROVENANCE_PREFIX}{charges_json}\n"
            + butane_itp.read_text().replace("-0.270", "-0.300", 1).replace("0.090", "0.100", 1)
        )
        return 0
```

and in `test_validation_json_names_the_charge_refined_topology_it_measured_before_against`, after the `charges_refined is True` assertion, add:

```python
    assert validation["charges_json"] == str((outdir / "04_charges" / "charges.json").resolve())
```

and in `test_validation_json_names_the_users_topology_when_no_charges_ran`, after its `charges_refined is False` assertion, add:

```python
    assert validation["charges_json"] is None
```

Append:

```python
def test_charges_refined_holds_for_a_charged_topology_from_another_outdir(
    butane_pdb, butane_itp, tmp_path, monkeypatch
):
    """Milestone 5 ruling R28: the flag used to compare paths inside one outdir only."""
    import ffrefine.cli as cli
    from ffrefine.io.topwriter import write_charged_topology

    elsewhere = tmp_path / "earlier_run" / "04_charges"
    elsewhere.mkdir(parents=True)
    charged = elsewhere / "molecule_charged.itp"
    write_charged_topology(butane_itp, [], charged, charges_json=elsewhere / "charges.json")

    outdir = tmp_path / "out"
    _fake_validation_inputs(outdir, butane_pdb)
    monkeypatch.setattr(cli, "validate_refit", _fake_validate_refit([]))
    monkeypatch.setattr(cli, "plot_validation", lambda *a, **k: None)
    args = cli.build_parser().parse_args(
        ["validate", "-s", str(butane_pdb), "-p", str(charged), "--dihedral", "1,5,8,11"]
        + ["--outdir", str(outdir)]
    )
    cli._apply_config(args)
    assert cli.cmd_validate(args) == 0
    validation = json.loads((outdir / "final" / "validation.json").read_text())
    assert validation["charges_refined"] is True
    assert validation["charges_json"] == str((elsewhere / "charges.json").resolve())
```

- [ ] **Step 3: Run them to verify they fail**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_topwriter.py tests/test_cli_charges.py -q'`
Expected: `ImportError: cannot import name 'CHARGES_PROVENANCE_PREFIX'`.

- [ ] **Step 4: Implement in `ffrefine/io/topwriter.py`**

Add below `_WRAPPER_SECTIONS`:

```python
CHARGES_PROVENANCE_PREFIX = "; ffrefine charges: "
"""The comment line a charge-refined topology carries, naming its charges.json.

Written by :func:`write_charged_topology` and copied verbatim by every later
writer, so any topology in a chain -- a bond subfolder's input, the final
refined topology -- says which charges its torsions were fitted against
(milestone 5 ruling R28)."""


def charges_provenance(path: str | Path) -> Path | None:
    """The ``charges.json`` a topology's charges came from, or ``None``."""
    for line in Path(path).read_text().splitlines():
        if line.startswith(CHARGES_PROVENANCE_PREFIX):
            return Path(line[len(CHARGES_PROVENANCE_PREFIX) :].strip())
    return None
```

Change `write_charged_topology`'s signature to

```python
def write_charged_topology(
    source: str | Path,
    assignments: list["ChargeAssignment"],
    out_path: str | Path,
    decimals: int = 4,
    charges_json: str | Path | None = None,
) -> list[int]:
```

document the new argument in its Args list (`charges_json: the stage's charges.json; recorded in a CHARGES_PROVENANCE_PREFIX comment line, replacing any earlier one. None writes no such line.`), add to its docstring "Like the other writers it drops ``[ system ]`` and ``[ molecules ]`` (milestone 5 finding M7): every topology ffrefine writes is a molecule include file.", and replace its final two lines before `log.info(...)`:

```python
    out = mark_include_base(out, include_base(source))
    Path(out_path).write_text("\n".join(out) + "\n", newline="\n")
```

with

```python
    out = [line for line in out if not line.startswith(CHARGES_PROVENANCE_PREFIX)]
    if charges_json is not None:
        out.insert(0, f"{CHARGES_PROVENANCE_PREFIX}{Path(charges_json).resolve()}")
    out, dropped = strip_system_sections(out)
    if dropped:
        log.info("%s: dropped [ %s ] from the charged copy", Path(source).name, " ] [ ".join(dropped))
    out = mark_include_base(out, include_base(source))
    Path(out_path).write_text("\n".join(out) + "\n", newline="\n")
```

- [ ] **Step 5: Wire it into the CLI**

In `ffrefine/cli.py`, change the import `from .io.topwriter import write_charged_topology, write_refit_topology` to `from .io.topwriter import charges_provenance, write_charged_topology, write_refit_topology`.

In `_charges_stage`, pass the provenance:

```python
    write_charged_topology(
        args.topology, assignments, charged_itp, decimals=decimals, charges_json=_charges_path(args)
    )
```

In `cmd_validate`, replace

```python
    before_topology = Path(args.topology).resolve()
    charged_output = Path(args.outdir) / STAGE_DIRS["charges"] / "molecule_charged.itp"
    charges_refined = before_topology == charged_output.resolve()
```

with

```python
    before_topology = Path(args.topology).resolve()
    # Milestone 5 ruling R28: read from the topology itself, so the answer is
    # right for a charged topology from another outdir and for every bond of a
    # chained run, not only for this outdir's own 04_charges/.
    charges_json = charges_provenance(before_topology)
```

and in the `validation.json` dict replace `"charges_refined": bool(charges_refined),` with

```python
                "charges_refined": charges_json is not None,
                "charges_json": None if charges_json is None else str(charges_json),
```

Update `cmd_validate`'s docstring sentence about `charges_refined` to: "whether its charges were refined by ffrefine (``charges_refined``, read from the provenance line the charges stage writes) and which ``charges.json`` they came from (``charges_json``)".

- [ ] **Step 6: Run the tests**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_topwriter.py tests/test_cli_charges.py tests/test_cli_typed.py -q -rs && python -m pytest -q -rs'`
Expected: all pass, 0 skipped.

- [ ] **Step 7: black and commit**

```bash
wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m black ffrefine tests'
cd /c/Users/hrach/Projects/ffrefine
git add ffrefine/io/topwriter.py ffrefine/cli.py tests/test_topwriter.py tests/test_cli_charges.py
git commit -m "fix(charges): charged topologies carry their provenance and are molecule includes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: An SCF that did not converge is refused, `IgnoreConv` or not (R29)

**Files:**
- Modify: `ffrefine/qm/orca.py` (`ORCA_SCF_CONVERGED_MARKER`; `run_esp_job`; `refine_energies`)
- Modify: `ffrefine/cli.py` (`scan.json` provenance `orca_version` under `--sp-refine`)
- Test: `tests/test_qm_orca.py` (update `ORCA_OK_STDOUT`; append)

**Interfaces:**
- Produces: `ffrefine.qm.orca.ORCA_SCF_CONVERGED_MARKER = "SCF CONVERGED AFTER"`; `run_esp_job` and `refine_energies` raise `ExternalToolError` when a job's stdout lacks it; `scan.json` provenance key `"orca_version": str | None`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_qm_orca.py`, replace

```python
ORCA_OK_STDOUT = "...\n                             ****ORCA TERMINATED NORMALLY****\n"
```

with

```python
ORCA_OK_STDOUT = (
    "...\n"
    "               *           SCF CONVERGED AFTER  12 CYCLES          *\n"
    "...\n"
    "                             ****ORCA TERMINATED NORMALLY****\n"
)

# What ORCA 6.1.1 printed for water at HF/6-31G* with
# `%scf IgnoreConv true MaxIter 1 end` (probed 2026-09-23). Exit status 0, a
# normal termination, and neither "SCF NOT CONVERGED" nor "SCF CONVERGED AFTER".
IGNORECONV_STDOUT = """\
WARNING: Found SCFConvIgnored == true
Warning (ORCA_LEANSCF): the SCF did not converge, but SCFConvIgnored is TRUE.
                             ****ORCA TERMINATED NORMALLY****
"""
```

Append:

```python
def test_run_esp_job_refuses_an_scf_ignoreconv_let_through(tmp_path, butane_pdb, monkeypatch):
    """Milestone 5 ruling R29: IgnoreConv makes ORCA exit 0 with no failure marker."""
    import ffrefine.qm.orca as orca_mod

    writes = ("esp.chelpg.xyz",)
    monkeypatch.setattr(orca_mod, "run_command", _fake_orca(IGNORECONV_STDOUT, writes=writes))
    s = read_pdb(butane_pdb)
    with pytest.raises(ExternalToolError, match="IgnoreConv"):
        orca_mod.run_esp_job(Binaries(orca="/nonexistent/orca"), s, s.coords, tmp_path)


def test_refine_energies_refuses_a_frame_whose_scf_did_not_converge(
    tmp_path, butane_pdb, monkeypatch
):
    import ffrefine.qm.orca as orca_mod

    stdout = IGNORECONV_STDOUT + "FINAL SINGLE POINT ENERGY      -157.123456\n"
    monkeypatch.setattr(orca_mod, "run_command", _fake_orca(stdout))
    s = read_pdb(butane_pdb)
    with pytest.raises(ExternalToolError, match="frame 0: .*SCF CONVERGED AFTER"):
        orca_mod.refine_energies(
            Binaries(orca="/nonexistent/orca"), s, s.coords[None, :, :], tmp_path
        )


def test_refine_energies_accepts_a_converged_frame(tmp_path, butane_pdb, monkeypatch):
    import ffrefine.qm.orca as orca_mod

    stdout = ORCA_OK_STDOUT + "FINAL SINGLE POINT ENERGY      -157.123456\n"
    monkeypatch.setattr(orca_mod, "run_command", _fake_orca(stdout))
    s = read_pdb(butane_pdb)
    energies = orca_mod.refine_energies(
        Binaries(orca="/nonexistent/orca"), s, s.coords[None, :, :], tmp_path
    )
    assert energies.shape == (1,)


@pytest.mark.needs_orca
def test_run_esp_job_refuses_real_orca_under_ignoreconv(tmp_path, orca_bin):
    """The live counterpart: ORCA 6.1.1 itself, one SCF iteration, IgnoreConv."""
    from ffrefine.io.structure import read_xyz

    xyz = tmp_path / "water.xyz"
    xyz.write_text("3\nwater\nO 0.000 0.000 0.000\nH 0.000 0.757 0.587\nH 0.000 -0.757 0.587\n")
    s = read_xyz(xyz)
    keywords = "! HF 6-31G* CHELPG\n%scf IgnoreConv true MaxIter 1 end"
    with pytest.raises(ExternalToolError, match="IgnoreConv"):
        run_esp_job(Binaries(orca=orca_bin), s, s.coords, tmp_path / "esp", keywords)
```

(`run_esp_job` and `read_pdb`, `Binaries`, `ExternalToolError`, `pytest` are already imported at the top of the file; add `run_esp_job` to its `ffrefine.qm.orca` import if missing.)

- [ ] **Step 2: Run them to verify they fail**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_qm_orca.py -q -rs'`
Expected: the three unit tests about IgnoreConv / convergence fail with "DID NOT RAISE"; the live test fails the same way.

- [ ] **Step 3: Implement in `ffrefine/qm/orca.py`**

Below `ORCA_SCF_FAILED_MARKER`, add:

```python
ORCA_SCF_CONVERGED_MARKER = "SCF CONVERGED AFTER"
"""What ORCA prints when its SCF really converged. Required, not merely the
absence of a failure: with ``%scf IgnoreConv true end`` ORCA 6.1.1 exits 0,
terminates normally and prints no failure marker for an unconverged SCF
(probed 2026-09-23; milestone 5 ruling R29)."""

_IGNORECONV_HINT = (
    "ORCA prints 'SCF CONVERGED AFTER N CYCLES' when its SCF converges, and this output "
    "does not. With '%scf IgnoreConv true end' ORCA exits 0 and terminates normally even "
    "so, which is why the exit code proves nothing."
)
```

In `run_esp_job`, after the `ORCA_NORMAL_TERMINATION_MARKER` check, add:

```python
    if ORCA_SCF_CONVERGED_MARKER not in stdout:
        raise ExternalToolError(
            f"ORCA's ESP job has no converged SCF. {_IGNORECONV_HINT} There is no "
            f"density to take charges or a potential from. See {log_path}."
        )
```

and add "or ORCA never printed that its SCF converged" to the function's Raises line.

In `refine_energies`, replace the `try: out.append(parse_final_energy(...))` block with:

```python
        stdout = proc.stdout or ""
        if ORCA_SCF_CONVERGED_MARKER not in stdout:
            raise ExternalToolError(
                f"frame {index}: no '{ORCA_SCF_CONVERGED_MARKER}' line. {_IGNORECONV_HINT} "
                f"See {workdir / (stem + '.stdout')}."
            )
        try:
            out.append(parse_final_energy(stdout))
        except ExternalToolError as exc:
            raise ExternalToolError(f"frame {index}: {exc}") from exc
```

(`ORCA_SCF_CONVERGED_MARKER` and `_IGNORECONV_HINT` are defined further down the module than `refine_energies`; Python resolves them at call time, so no reordering is needed.)

- [ ] **Step 4: Record ORCA's version under `--sp-refine`**

In `ffrefine/cli.py`, in `cmd_scan`'s payload `"provenance"` dict, after `"sp_refine_keywords": sp_keywords,` add:

```python
            "orca_version": _orca_version(binaries) if sp_keywords else None,
```

- [ ] **Step 5: Run the tests**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_qm_orca.py tests/test_cli_scan.py -q -rs && python -m pytest -q -rs'`
Expected: all pass, 0 skipped (the `needs_orca` tests run for real).

- [ ] **Step 6: black and commit**

```bash
wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m black ffrefine tests'
cd /c/Users/hrach/Projects/ffrefine
git add ffrefine/qm/orca.py ffrefine/cli.py tests/test_qm_orca.py
git commit -m "fix(qm): require ORCA's SCF CONVERGED line; IgnoreConv no longer slips through

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: A no-op refit under `#ifdef` is a warning (R9)

**Files:**
- Modify: `ffrefine/mm/tprparams.py` (`_dihedrals_have_conditionals`; `resolve_torsion_set`)
- Test: `tests/test_tprparams.py` (append)

**Interfaces:**
- Consumes: `resolve_torsion_set`, `collect_torsion_set`, `SOURCE_DIHEDRALTYPES` (existing).
- Produces: `_dihedrals_have_conditionals(path: str | Path) -> bool` (private); `resolve_torsion_set` issues `log.warning(...)` for each typed quad that resolved to no terms when the source topology's `[ dihedrals ]` section holds an `#ifdef`, `#ifndef` or `#else`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tprparams.py`:

```python
from ffrefine.mm.tprparams import _dihedrals_have_conditionals


def test_conditionals_are_detected_only_inside_dihedrals(tmp_path):
    inside = tmp_path / "inside.itp"
    inside.write_text("[ dihedrals ]\n#ifdef FOO\n1 2 3 4 9\n#endif\n")
    elsewhere = tmp_path / "elsewhere.itp"
    elsewhere.write_text("[ atoms ]\n#ifdef FOO\n#endif\n[ dihedrals ]\n1 2 3 4 9\n")
    assert _dihedrals_have_conditionals(inside)
    assert not _dihedrals_have_conditionals(elsewhere)
    assert not _dihedrals_have_conditionals(CAPECITABINE / "capecitabine_gmx.top")


@contextlib.contextmanager
def _ffrefine_warnings():
    """Every WARNING record on the ``ffrefine`` logger while the block runs.

    Attached to that logger directly: ``setup_logging`` turns propagation off,
    so pytest's ``caplog`` would miss these records after any CLI test ran.
    """
    records: list[logging.LogRecord] = []
    handler = logging.Handler(level=logging.WARNING)
    handler.emit = records.append
    logger = logging.getLogger("ffrefine")
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


@pytest.mark.needs_gmx
def test_an_empty_typed_line_under_ifdef_is_a_warning(tmp_path, gmx_bin):
    """Capecitabine's 6 7 9 37 resolves to no terms. With an #ifdef in
    [ dihedrals ] that is no longer provably the all-k=0 case, so it warns."""
    work = tmp_path / "cap"
    shutil.copytree(CAPECITABINE, work)
    top = work / "capecitabine_gmx.top"
    text = top.read_text()
    marker = "    6     7     9    37     9\n"
    assert marker in text, "fixture layout changed"
    top.write_text(text.replace(marker, f"#ifndef NEVER_DEFINED\n{marker}#endif\n", 1))
    t = read_topology(top)
    s = read_pdb(work / "capecitabine_gmx.pdb")
    with _ffrefine_warnings() as records:
        resolve_torsion_set(Binaries(gmx=gmx_bin), t, (6, 7, 9, 10), s, tmp_path / "wd")
    messages = [r.getMessage() for r in records]
    assert any("6 7 9 37" in m and "#ifdef" in m for m in messages), messages


@pytest.mark.needs_gmx
def test_an_empty_typed_line_without_ifdef_stays_info(tmp_path, gmx_bin):
    t = read_topology(CAPECITABINE / "capecitabine_gmx.top")
    s = read_pdb(CAPECITABINE / "capecitabine_gmx.pdb")
    with _ffrefine_warnings() as records:
        resolve_torsion_set(Binaries(gmx=gmx_bin), t, (6, 7, 9, 10), s, tmp_path)
    assert not [r.getMessage() for r in records if "resolved to no terms" in r.getMessage()]
```

(Add `import contextlib` and `import logging` to the file's imports.)

- [ ] **Step 2: Run them to verify they fail**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_tprparams.py -q -rs'`
Expected: `ImportError: cannot import name '_dihedrals_have_conditionals'`.

- [ ] **Step 3: Implement in `ffrefine/mm/tprparams.py`**

Add, above `resolve_torsion_set`:

```python
_CONDITIONAL_RE = re.compile(r"^\s*#\s*(ifdef|ifndef|else)\b")


def _dihedrals_have_conditionals(path: str | Path) -> bool:
    """Whether a ``[ dihedrals ]`` section of ``path`` holds an ``#ifdef``/``#ifndef``/``#else``.

    ffrefine's parser resolves such branches with no defines; grompp may take
    a different branch when an included force-field file defines something.
    """
    section = ""
    for raw in Path(path).read_text().splitlines():
        body = raw.split(";", 1)[0].strip()
        if body.startswith("[") and body.endswith("]"):
            section = body[1:-1].strip().lower()
        elif section == "dihedrals" and _CONDITIONAL_RE.match(raw):
            return True
    return False
```

In `resolve_torsion_set`, after `terms = cross_check(...)`, add:

```python
    resolved_quads = {
        (t.dihedral.ai, t.dihedral.aj, t.dihedral.ak, t.dihedral.al) for t in terms
    }
    empty = [
        (d.ai, d.aj, d.ak, d.al)
        for d in collect_torsion_set(topology, quad)
        if not d.params and (d.ai, d.aj, d.ak, d.al) not in resolved_quads
    ]
    if empty and _dihedrals_have_conditionals(topology.source):
        for q in dict.fromkeys(empty):
            log.warning(
                "torsion %s resolved to no terms, and %s has an #ifdef in [ dihedrals ]: "
                "grompp may have read a different branch than ffrefine, in which case a "
                "refit written onto this line would have no effect (milestone 6a ruling R9).",
                " ".join(map(str, q)),
                Path(topology.source).name,
            )
```

- [ ] **Step 4: Run the tests**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_tprparams.py -q -rs && python -m pytest -q -rs'`
Expected: all pass, 0 skipped.

- [ ] **Step 5: black and commit**

```bash
wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m black ffrefine tests'
cd /c/Users/hrach/Projects/ffrefine
git add ffrefine/mm/tprparams.py tests/test_tprparams.py
git commit -m "fix(mm): warn when a typed torsion under #ifdef resolves to nothing

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Every stage payload keeps its warnings

**Files:**
- Modify: `ffrefine/stages.py` (`WarningCollector`; `setup_logging` keeps collectors)
- Modify: `ffrefine/cli.py` (`_warn`, `_collect_warnings`, `_current_warnings`, `_collects_warnings`, `_reset_logging`; `warnings` in `scan.json`, `energies.json`, `fit.json`, `validation.json`; the `--distribute` warning moves before `fit.json` is written)
- Create: `tests/test_warnings.py`
- Test: `tests/test_cli.py` (append), `tests/test_cli_typed.py` (append)

**Interfaces:**
- Produces:
  - `ffrefine.stages.WarningCollector(logging.Handler)` with attribute `messages: list[str]`.
  - `cli._collect_warnings() -> ContextManager[list[str]]` (same contract as today: yields the list the block's warnings accumulate in).
  - `cli._current_warnings() -> list[str]`: a copy of the innermost active collector's messages, `[]` outside any.
  - `cli._collects_warnings(func)`: decorator running a stage command inside `_collect_warnings()`; applied to `cmd_scan`, `cmd_mmscan`, `cmd_fit`, `cmd_validate`.
  - A top-level `"warnings": list[str]` key in `scan.json`, `energies.json`, `fit.json`, `validation.json` (charges.json already has one).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_warnings.py`:

```python
"""Every warning a stage raises reaches its payload (milestone 6b design, section 6).

Refinement 1: warnings are collected at the logging layer, so a module's own
``log.warning`` (GROMACS's echoed warnings, the self-check's) is kept as well
as the CLI's ``_warn``.
"""

from __future__ import annotations

import logging

from ffrefine import cli
from ffrefine.stages import WarningCollector, setup_logging

log = logging.getLogger("ffrefine")


def test_a_collector_keeps_module_and_cli_warnings_in_order():
    with cli._collect_warnings() as got:
        log.warning("from a module")
        cli._warn("from the cli")
        log.info("not a warning")
    assert got == ["from a module", "from the cli"]


def test_setup_logging_leaves_a_collector_attached(tmp_path):
    with cli._collect_warnings() as got:
        setup_logging(tmp_path)
        cli._warn("after setup_logging")
    assert got == ["after setup_logging"]
    assert "after setup_logging" in (tmp_path / "run.log").read_text()
    assert not any(isinstance(h, WarningCollector) for h in log.handlers)


def test_current_warnings_reads_the_innermost_collector():
    assert cli._current_warnings() == []
    with cli._collect_warnings():
        cli._warn("outer")
        with cli._collect_warnings() as inner:
            cli._warn("inner")
            assert cli._current_warnings() == ["inner"]
        assert inner == ["inner"]


def test_a_warning_with_no_log_handler_is_printed_and_collected(capsys):
    cli._reset_logging()
    with cli._collect_warnings() as got:
        cli._warn("before logging exists")
    assert got == ["before logging exists"]
    assert "warning: before logging exists" in capsys.readouterr().err
```

Append to `tests/test_cli.py`:

```python
def test_fit_json_keeps_the_warnings_the_fit_raised(tmp_path, butane_pdb, butane_itp):
    """Fitting without --distribute always warns; the warning is in fit.json."""
    out = tmp_path / "out"
    phi = _grid()
    _write_stage_inputs(out, phi, 5.0 * (1.0 + np.cos(np.radians(3.0 * phi))))
    assert main(["fit"] + _base_args(butane_pdb, butane_itp, out)) == 0
    fit = json.loads((out / "03_fit" / "fit.json").read_text())
    assert any("without --distribute" in w for w in fit["warnings"])
```

Append to `tests/test_cli_typed.py`:

```python
def test_energies_json_keeps_the_warnings_mmscan_raised(
    tmp_path, butane_pdb, butane_itp, monkeypatch
):
    """No scan.json: mmscan warns that it covers one geometry, and keeps the warning."""
    out = tmp_path / "out"
    monkeypatch.setattr(cli, "check_torsion_removal", _fake_check([]))
    monkeypatch.setattr(cli, "resolve_torsion_set", _fake_resolve("a" * 64))
    argv = ["mmscan", "-s", str(butane_pdb), "-p", str(butane_itp), "--dihedral", "1,5,8,11"]
    assert main(argv + ["--outdir", str(out)]) == 0
    energies = json.loads((out / "02_mm_scan" / "energies.json").read_text())
    assert any("no QM scan" in w for w in energies["warnings"])


def test_validation_json_keeps_the_did_not_improve_warning(tmp_path, monkeypatch):
    out = tmp_path / "out"
    _write_pfoa_stage_inputs(out)
    assert main(["fit"] + _pfoa_args(out)) == 0

    def worse(*args, **kwargs):
        import dataclasses

        result = _fake_validate_refit(*args, **kwargs)
        return dataclasses.replace(result, rmse_after=2.0, improved=False)

    _stub_validate(monkeypatch)
    monkeypatch.setattr(cli, "validate_refit", worse)
    assert main(["validate"] + _pfoa_args(out)) == 0
    validation = json.loads((out / "final" / "validation.json").read_text())
    assert any("did not improve" in w for w in validation["warnings"])
```

- [ ] **Step 2: Run them to verify they fail**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_warnings.py tests/test_cli.py::test_fit_json_keeps_the_warnings_the_fit_raised tests/test_cli_typed.py -q'`
Expected: `ImportError: cannot import name 'WarningCollector'`; the payload tests fail with `KeyError: 'warnings'`.

- [ ] **Step 3: Add `WarningCollector` to `ffrefine/stages.py`**

Above `setup_logging`:

```python
class WarningCollector(logging.Handler):
    """Collects the text of every WARNING-or-worse record while attached.

    ``cli._collect_warnings`` attaches one around each stage, so the stage's
    JSON payload can carry every warning it raised -- including the ones a
    module logs itself (GROMACS's echoed warnings, the self-check's structure
    warnings), not only the CLI's own. :func:`setup_logging` leaves these
    attached; they belong to whoever opened them.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())
```

In `setup_logging`, change the handler-removal loop to:

```python
    for handler in list(log.handlers):
        if isinstance(handler, WarningCollector):
            continue
        log.removeHandler(handler)
        handler.close()
```

and add to its docstring: "Handlers that are :class:`WarningCollector` instances are left in place."

- [ ] **Step 4: Rework the collection in `ffrefine/cli.py`**

Add `import functools` to the imports, and `WarningCollector` to the `from .stages import (...)` list.

Replace `_WARNING_SINKS`, `_collect_warnings` and `_warn` with:

```python
def _output_handlers() -> list[logging.Handler]:
    """The ``ffrefine`` logger's handlers that actually show a message somewhere."""
    return [h for h in log.handlers if not isinstance(h, WarningCollector)]


@contextmanager
def _collect_warnings() -> Iterator[list[str]]:
    """Collect every warning issued inside the block, from any module.

    A :class:`~ffrefine.stages.WarningCollector` is attached to the
    ``ffrefine`` logger for the block's duration, so the text of every
    WARNING record lands in the yielded list: the CLI's own :func:`_warn`
    and a module's ``log.warning`` alike (milestone 6b, refinement 1). Each
    stage writes the list into its JSON payload, which is what
    ``report.md``'s "every warning raised" is built from.
    """
    collector = WarningCollector()
    log.addHandler(collector)
    try:
        yield collector.messages
    finally:
        log.removeHandler(collector)


def _current_warnings() -> list[str]:
    """A copy of the innermost active collector's warnings, or ``[]`` outside any."""
    collectors = [h for h in log.handlers if isinstance(h, WarningCollector)]
    return list(collectors[-1].messages) if collectors else []


def _collects_warnings(func):
    """Run a stage command inside :func:`_collect_warnings`."""

    @functools.wraps(func)
    def wrapper(args: argparse.Namespace) -> int:
        with _collect_warnings():
            return func(args)

    return wrapper


def _warn(message: str) -> None:
    """Emit a user-facing warning exactly once.

    Once logging is configured, ``setup_logging``'s console handler puts the
    message on stderr *and* into ``run.log``, and every active collector
    receives it through the logger. Before that (dry runs, argument handling)
    nothing would show it, so it is printed to stderr and handed to the
    collectors directly.
    """
    if _output_handlers():
        log.warning(message)
        return
    print(f"warning: {message}", file=sys.stderr)
    for handler in log.handlers:
        if isinstance(handler, WarningCollector):
            handler.messages.append(message)
```

In `_reset_logging`, skip collectors:

```python
    for handler in list(log.handlers):
        if isinstance(handler, WarningCollector):
            continue
        log.removeHandler(handler)
        handler.close()
```

Put `@_collects_warnings` on the line above `def cmd_scan`, `def cmd_mmscan`, `def cmd_fit` and `def cmd_validate`. (`cmd_charges` already wraps its body in `_collect_warnings()`; leave it.)

- [ ] **Step 5: Write the warnings into the payloads**

- `cmd_scan`: in `payload`, after `"frames_sha256": hash_array(frames_nm),` add `"warnings": _current_warnings(),`.
- `cmd_mmscan`: in `payload`, after `"preprocessed_topology_sha256": resolved.pp_sha256,` add `"warnings": _current_warnings(),`.
- `cmd_fit`: move the block

  ```python
      if not args.distribute:
          _warn(
              "fitted without --distribute: ...
          )
  ```

  so it sits directly before the `(outdir / "fit.json").write_text(` call (it reads only `args.distribute`), then replace that call with:

  ```python
      summary = fit_summary(result, phi, target, provenance, qm_kj=qm)
      summary["warnings"] = _current_warnings()
      (outdir / "fit.json").write_text(json.dumps(summary, indent=2) + "\n")
  ```

- `cmd_validate`: in the `validation.json` dict, after `"input_topology": ...,` add `"warnings": _current_warnings(),`.

- [ ] **Step 6: Run the tests**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_warnings.py tests/test_cli.py tests/test_cli_typed.py tests/test_cli_charges.py -q -rs && python -m pytest -q -rs'`
Expected: all pass, 0 skipped.

- [ ] **Step 7: black and commit**

```bash
wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m black ffrefine tests'
cd /c/Users/hrach/Projects/ffrefine
git add ffrefine/stages.py ffrefine/cli.py tests/test_warnings.py tests/test_cli.py tests/test_cli_typed.py
git commit -m "feat(cli): every stage payload keeps the warnings it raised, from any module

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: `io/cgenff.py`, the `.str` parser

**Files:**
- Create: `ffrefine/io/cgenff.py`
- Create: `tests/test_cgenff.py`

**Interfaces:**
- Consumes: `ffrefine.errors.ParseError` only.
- Produces (all in `ffrefine.io.cgenff`):
  - `PARAMETER_KINDS: dict[str, tuple[str, int]]`
  - `StrAtom(index: int, name: str, atype: str, charge: float, penalty: float)`, frozen; `index` 1-based
  - `StrParameter(kind: str, types: tuple[str, ...], penalty: float)`, frozen; `kind` in `bond`, `angle`, `dihedral`, `improper`
  - `StrFile(atoms, parameters, program_version, parameter_version, param_penalty, charge_penalty, source)` with method `of_kind(kind: str) -> list[StrParameter]`
  - `read_str(path: str | Path) -> StrFile`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cgenff.py`:

```python
"""The CGenFF .str parser, on the two committed CGenFF outputs (design section 4)."""

from __future__ import annotations

from pathlib import Path

import pytest
from ffrefine.errors import ParseError
from ffrefine.io.cgenff import StrAtom, read_str

FIXTURES = Path(__file__).parent.parent / "fixtures"
PFOA_STR = FIXTURES / "pfoa_gromacs" / "pfoa.str"
CAP_STR = FIXTURES / "capecitabine_gromacs" / "capecitabine.str"


def _penalty(strfile, kind, types):
    [match] = [p for p in strfile.of_kind(kind) if p.types == tuple(types)]
    return match.penalty


def test_pfoa_atoms_in_file_order():
    s = read_str(PFOA_STR)
    assert len(s.atoms) == 26
    assert s.atoms[0] == StrAtom(1, "C1", "CG2O2", 0.789, 25.023)
    assert s.atoms[1] == StrAtom(2, "O1", "OG2D1", -0.537, 17.721)
    assert s.atoms[-1] == StrAtom(26, "H", "HGP1", 0.429, 6.830)


def test_pfoa_parameters_and_header():
    s = read_str(PFOA_STR)
    counts = {kind: len(s.of_kind(kind)) for kind in ("bond", "angle", "dihedral", "improper")}
    assert counts == {"bond": 2, "angle": 8, "dihedral": 13, "improper": 1}
    assert (s.program_version, s.parameter_version) == ("4.0", "5.0")
    assert (s.param_penalty, s.charge_penalty) == (69.0, 41.068)
    assert s.source == PFOA_STR


def test_a_dihedral_over_several_multiplicity_lines_is_one_entry():
    s = read_str(PFOA_STR)
    assert _penalty(s, "dihedral", ["OG311", "CG2O2", "CG312", "CG312"]) == 67.0
    assert _penalty(s, "dihedral", ["CG2O2", "CG312", "CG312", "FGA2"]) == 67.2
    assert _penalty(s, "dihedral", ["OG2D1", "CG2O2", "CG312", "CG312"]) == 22.5
    assert _penalty(s, "angle", ["CG312", "CG2O2", "OG2D1"]) == 10.5
    assert _penalty(s, "bond", ["CG2O2", "CG312"]) == 10.0
    assert _penalty(s, "improper", ["CG2O2", "CG312", "OG2D1", "OG311"]) == 5.5


def test_capecitabine_skips_ring_lines_and_reads_urey_bradley_angles():
    s = read_str(CAP_STR)
    assert len(s.atoms) == 47
    assert s.atoms[9] == StrAtom(10, "C7", "CG2R64", 0.389, 5.422)
    assert _penalty(s, "angle", ["CG2R66", "CG2R64", "NG2S1"]) == 0.5
    assert _penalty(s, "dihedral", ["OG302", "CG2O6", "NG2S1", "CG2R64"]) == 50.0
    assert (s.param_penalty, s.charge_penalty) == (50.0, 5.422)


def test_crlf_line_endings_parse_the_same(tmp_path):
    """Review focus 4: a .str saved on Windows."""
    crlf = tmp_path / "pfoa.str"
    crlf.write_bytes(PFOA_STR.read_bytes().replace(b"\n", b"\r\n"))
    a, b = read_str(PFOA_STR), read_str(crlf)
    assert (a.atoms, a.parameters) == (b.atoms, b.parameters)


def test_a_missing_file_is_a_parse_error(tmp_path):
    """Review focus 5."""
    with pytest.raises(ParseError, match="could not be read"):
        read_str(tmp_path / "nope.str")


def test_a_file_with_no_atom_is_refused(tmp_path):
    bad = tmp_path / "empty.str"
    bad.write_text("* nothing here\nread rtf card append\nEND\n")
    with pytest.raises(ParseError, match="no ATOM line"):
        read_str(bad)


def test_a_malformed_atom_line_names_its_line(tmp_path):
    bad = tmp_path / "bad.str"
    bad.write_text(PFOA_STR.read_text().replace("ATOM C1     CG2O2   0.789", "ATOM C1     CG2O2", 1))
    with pytest.raises(ParseError, match=r"bad\.str:20: malformed ATOM"):
        read_str(bad)


def test_a_non_numeric_penalty_is_refused(tmp_path):
    bad = tmp_path / "bad.str"
    bad.write_text(PFOA_STR.read_text().replace("penalty= 67", "penalty= sixty-seven", 1))
    with pytest.raises(ParseError, match="penalty 'sixty-seven' is not a number"):
        read_str(bad)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_cgenff.py -q'`
Expected: `ModuleNotFoundError: No module named 'ffrefine.io.cgenff'`.

- [ ] **Step 3: Write `ffrefine/io/cgenff.py`**

This code was run on both committed `.str` files while planning (with `_PENALTY_RE` then capturing only numerals; the `\S+` form below makes a malformed penalty an error instead of a silent 0).

```python
"""Read a CGenFF stream file (``.str``): atom charges, parameters, penalties.

Milestone 6b design, section 4. CGenFF annotates every atom it assigned a
charge to with a *charge penalty* (the first number after ``!`` on its
``ATOM`` line) and every parameter it had to guess by analogy with
``penalty= X`` in the line's comment. Higher means less trustworthy; the
file's own header gives the bands (below 10 fair, 10 to 50 validate, above
50 poor).

Pure text: this module never starts a process.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..errors import ParseError

PARAMETER_KINDS: dict[str, tuple[str, int]] = {
    "BONDS": ("bond", 2),
    "ANGLES": ("angle", 3),
    "DIHEDRALS": ("dihedral", 4),
    "IMPROPERS": ("improper", 4),
}
"""Parameter section header -> (kind, number of leading atom-type fields)."""

_OTHER_PARAMETER_SECTIONS = frozenset({"NONBONDED", "NBFIX", "CMAP", "HBOND"})
_PENALTY_RE = re.compile(r"penalty=\s*(\S+)")
_RESI_RE = re.compile(r"param penalty=\s*([-+0-9.eE]+)\s*;\s*charge penalty=\s*([-+0-9.eE]+)")
_PROGRAM_RE = re.compile(r"CGenFF\)\s+program version\s+(\S+)")
_PARAMETER_VERSION_RE = re.compile(r"topology and parameter files version\s+(\S+)")


@dataclass(frozen=True)
class StrAtom:
    """One ``ATOM`` line. ``index`` is 1-based, in file order."""

    index: int
    name: str
    atype: str
    charge: float
    penalty: float


@dataclass(frozen=True)
class StrParameter:
    """One bonded parameter CGenFF assigned by analogy.

    ``kind`` is ``bond``, ``angle``, ``dihedral`` or ``improper``. A dihedral
    written over several multiplicity lines is one entry carrying the highest
    of their penalties.
    """

    kind: str
    types: tuple[str, ...]
    penalty: float


@dataclass
class StrFile:
    """A parsed ``.str`` file."""

    atoms: list[StrAtom]
    parameters: list[StrParameter]
    program_version: str | None
    parameter_version: str | None
    param_penalty: float | None
    charge_penalty: float | None
    source: Path

    def of_kind(self, kind: str) -> list[StrParameter]:
        """The parameters of one kind, in file order."""
        return [p for p in self.parameters if p.kind == kind]


def _number(text: str, path: Path, lineno: int, what: str) -> float:
    try:
        return float(text)
    except ValueError as exc:
        raise ParseError(f"{path}:{lineno}: {what} {text!r} is not a number") from exc


def read_str(path: str | Path) -> StrFile:
    """Parse a CGenFF ``.str`` file.

    The topology block runs from ``read rtf card`` to its ``END``; each
    ``ATOM name type charge ! penalty ...`` line is one atom. The parameter
    block runs from ``read param card`` to its ``END``; inside ``BONDS``,
    ``ANGLES``, ``DIHEDRALS`` and ``IMPROPERS`` each line gives its leading
    atom types and the number after ``penalty=`` (0 when absent).

    Raises:
        ParseError: the file cannot be read, holds no ``ATOM`` line, or has an
            ``ATOM`` or parameter line that does not parse.
    """
    path = Path(path)
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ParseError(f"{path}: could not be read: {exc}") from exc

    atoms: list[StrAtom] = []
    parameters: dict[tuple[str, tuple[str, ...]], float] = {}
    program_version = parameter_version = None
    param_penalty = charge_penalty = None
    block = ""
    section = ""
    for lineno, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if program_version is None and (m := _PROGRAM_RE.search(stripped)):
            program_version = m.group(1)
        if parameter_version is None and (m := _PARAMETER_VERSION_RE.search(stripped)):
            parameter_version = m.group(1)
        lowered = stripped.lower()
        if lowered.startswith("read rtf card"):
            block, section = "rtf", ""
            continue
        if lowered.startswith("read param card"):
            block, section = "param", ""
            continue
        if stripped.upper() == "END":
            block, section = "", ""
            continue
        if not stripped or stripped.startswith(("!", "*")):
            continue
        body = stripped.split("!", 1)[0].split()
        if block == "rtf":
            keyword = body[0].upper() if body else ""
            if keyword == "RESI" and (m := _RESI_RE.search(stripped)):
                param_penalty, charge_penalty = float(m.group(1)), float(m.group(2))
            elif keyword == "ATOM":
                comment = stripped.split("!", 1)[1].split() if "!" in stripped else []
                if len(body) < 4 or not comment:
                    raise ParseError(f"{path}:{lineno}: malformed ATOM line {raw!r}")
                atoms.append(
                    StrAtom(
                        index=len(atoms) + 1,
                        name=body[1],
                        atype=body[2],
                        charge=_number(body[3], path, lineno, "charge"),
                        penalty=_number(comment[0], path, lineno, "charge penalty"),
                    )
                )
        elif block == "param":
            header = stripped.split()[0].upper()
            if header in PARAMETER_KINDS or header in _OTHER_PARAMETER_SECTIONS:
                section = header
                continue
            if section not in PARAMETER_KINDS:
                continue
            kind, ntypes = PARAMETER_KINDS[section]
            if len(body) <= ntypes:
                raise ParseError(f"{path}:{lineno}: malformed {section} line {raw!r}")
            m = _PENALTY_RE.search(stripped)
            penalty = _number(m.group(1), path, lineno, "penalty") if m else 0.0
            key = (kind, tuple(body[:ntypes]))
            parameters[key] = max(penalty, parameters.get(key, penalty))

    if not atoms:
        raise ParseError(f"{path}: no ATOM line found; is this a CGenFF .str file?")
    return StrFile(
        atoms=atoms,
        parameters=[StrParameter(kind, types, pen) for (kind, types), pen in parameters.items()],
        program_version=program_version,
        parameter_version=parameter_version,
        param_penalty=param_penalty,
        charge_penalty=charge_penalty,
        source=path,
    )
```

- [ ] **Step 4: Run the tests**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_cgenff.py tests/test_layering.py -q'`
Expected: all pass (the layering test now also checks `io/cgenff.py`).

- [ ] **Step 5: black and commit**

```bash
wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m black ffrefine/io/cgenff.py tests/test_cgenff.py'
cd /c/Users/hrach/Projects/ffrefine
git add ffrefine/io/cgenff.py tests/test_cgenff.py
git commit -m "feat(io): parse CGenFF .str files, charges and parameter penalties

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: `selection.py`, from penalties to atoms and bonds

**Files:**
- Create: `ffrefine/selection.py`
- Create: `tests/test_selection.py`

**Interfaces:**
- Consumes: Task 6's `StrFile`, `StrAtom`, `StrParameter`, `read_str`; `ffrefine.charges.equivalence.equivalent_hydrogen_groups(topology) -> list[list[int]]`; `ffrefine.io.validate.element_of(topology, nr) -> str`; `Topology.bonds`, `Topology.propers`, `Topology.atoms`.
- Produces (all in `ffrefine.selection`):
  - `Thresholds(charge: float = 10.0, dihedral: float = 50.0)`, frozen
  - `AtomFlag`, `DihedralTypeFlag`, `BondChoice(bond: tuple[int, int], penalty: float, quad: Quad, types: list[tuple[str, str, str, str]])`
  - `Selection` with fields `thresholds, atoms, dihedral_types, other_parameters, bonds, ring_bonds, unmatched_types, warnings` and properties `flagged_atoms -> list[int]`, `quads -> list[Quad]`, `highest_dihedral_penalty -> float`
  - `check_mapping(topology, strfile) -> list[str]` (raises `ValidationError`)
  - `select(topology, strfile, thresholds) -> Selection`
  - `selection_to_json(selection: Selection, strfile: StrFile) -> dict`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_selection.py`:

```python
"""Selection from CGenFF penalties (design section 4), on the committed fixtures."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from ffrefine.errors import ValidationError
from ffrefine.io.cgenff import StrAtom, StrFile, StrParameter, read_str
from ffrefine.io.topology import read_topology
from ffrefine.selection import Thresholds, select, selection_to_json

FIXTURES = Path(__file__).parent.parent / "fixtures"
PFOA = FIXTURES / "pfoa_gromacs"
CAP = FIXTURES / "capecitabine_gromacs"


def _pfoa():
    return read_topology(PFOA / "pfoa_gmx.top"), read_str(PFOA / "pfoa.str")


def _cap():
    return read_topology(CAP / "capecitabine_gmx.top"), read_str(CAP / "capecitabine.str")


def _str_for(topology, penalties=None, parameters=()):
    """A StrFile describing ``topology`` exactly, with the given charge penalties."""
    penalties = penalties or {}
    atoms = [
        StrAtom(a.nr, a.name, a.atype, a.charge, float(penalties.get(a.nr, 0.0)))
        for a in topology.atoms
    ]
    return StrFile(atoms, list(parameters), None, None, None, None, Path("test.str"))


def test_pfoa_flags_23_atoms_and_two_bonds_in_penalty_order():
    sel = select(*_pfoa(), Thresholds())
    assert len(sel.flagged_atoms) == 23
    assert [a.nr for a in sel.atoms if not a.flagged] == [13, 14, 26]
    assert [(b.bond, b.penalty, b.quad) for b in sel.bonds] == [
        ((3, 4), 69.0, (1, 3, 4, 5)),
        ((1, 3), 67.0, (25, 1, 3, 4)),
    ]
    assert sel.quads == [(1, 3, 4, 5), (25, 1, 3, 4)]
    assert sel.ring_bonds == [] and sel.warnings == []


def test_capecitabine_flags_its_carbamate_at_exactly_the_threshold():
    sel = select(*_cap(), Thresholds())
    assert sel.flagged_atoms == []
    assert [(b.bond, b.penalty, b.quad) for b in sel.bonds] == [((7, 9), 50.0, (6, 7, 9, 10))]
    assert select(*_cap(), Thresholds(dihedral=51.0)).bonds == []
    assert sel.highest_dihedral_penalty == 50.0


def test_ring_bonds_are_never_selected():
    sel = select(*_cap(), Thresholds(dihedral=0.5))
    rings = {b.bond for b in sel.ring_bonds}
    assert rings == {(10, 16), (10, 11), (14, 15), (15, 16)}
    assert not rings & {b.bond for b in sel.bonds}


def test_a_different_atom_count_is_refused():
    t, _ = _cap()
    with pytest.raises(ValidationError, match="describes 26 atoms but the topology"):
        select(t, read_str(PFOA / "pfoa.str"), Thresholds())


def test_a_different_type_is_refused(tmp_path):
    bad = tmp_path / "pfoa.str"
    bad.write_text(
        (PFOA / "pfoa.str").read_text().replace("ATOM C1     CG2O2", "ATOM C1     CG331", 1)
    )
    with pytest.raises(ValidationError, match="atom 1 is CG2O2 in the topology, CG331 in the .str"):
        select(read_topology(PFOA / "pfoa_gmx.top"), read_str(bad), Thresholds())


def test_a_charge_difference_is_a_warning(tmp_path):
    edited = tmp_path / "pfoa.str"
    edited.write_text(
        (PFOA / "pfoa.str").read_text().replace("CG2O2   0.789", "CG2O2   0.800", 1)
    )
    sel = select(read_topology(PFOA / "pfoa_gmx.top"), read_str(edited), Thresholds())
    assert len(sel.warnings) == 1 and "differ" in sel.warnings[0] and "1 (+0.789" in sel.warnings[0]


def test_a_reordering_within_one_type_is_caught_by_the_charges(tmp_path):
    """Review focus 3: swapping two FGA2 atoms passes the type check."""
    lines = (PFOA / "pfoa.str").read_text().splitlines(keepends=True)
    i4 = next(i for i, line in enumerate(lines) if line.startswith("ATOM F4 "))
    i6 = next(i for i, line in enumerate(lines) if line.startswith("ATOM F6 "))
    lines[i4], lines[i6] = lines[i6], lines[i4]  # .str atoms 13 and 15, both FGA2
    swapped = tmp_path / "pfoa.str"
    swapped.write_text("".join(lines))
    sel = select(read_topology(PFOA / "pfoa_gmx.top"), read_str(swapped), Thresholds())
    assert sel.warnings and "13 (" in sel.warnings[0] and "15 (" in sel.warnings[0]


def test_flagged_hydrogens_extend_to_their_equivalence_group(butane_itp):
    t = read_topology(butane_itp)
    first_h = next(a.nr for a in t.atoms if a.atype == "HGA3")
    sel = select(t, _str_for(t, {first_h: 20.0}), Thresholds())
    methyl = [a for a in sel.atoms if a.flagged]
    assert len(methyl) == 3 and all(a.atype == "HGA3" for a in methyl)
    assert [a.by_equivalence for a in methyl] == [False, True, True]


def test_types_match_in_either_direction():
    t, s = _pfoa()
    reversed_types = StrParameter("dihedral", ("CG312", "CG312", "CG2O2", "OG311"), 67.0)
    sel = select(t, replace(s, parameters=[reversed_types]), Thresholds())
    assert [b.quad for b in sel.bonds] == [(25, 1, 3, 4)]


def test_an_unmatched_flagged_type_warns():
    t, s = _pfoa()
    alien = StrParameter("dihedral", ("NG2S1", "CG2O6", "NG2S1", "CG2R64"), 80.0)
    sel = select(t, replace(s, parameters=[alien]), Thresholds())
    assert sel.unmatched_types == [alien.types]
    assert any("another molecule" in w for w in sel.warnings)


def test_heavy_outer_atoms_win_a_penalty_tie(butane_itp):
    t = read_topology(butane_itp)
    types_of = {a.nr: a.atype for a in t.atoms}
    heavy = next(d for d in t.propers if (d.ai, d.aj, d.ak, d.al) == (1, 5, 8, 11))
    with_h = next(
        d for d in t.propers if (d.aj, d.ak) == (5, 8) and types_of[d.ai].startswith("H")
    )
    params = [
        StrParameter("dihedral", tuple(types_of[n] for n in (d.ai, d.aj, d.ak, d.al)), 60.0)
        for d in (with_h, heavy)
    ]
    sel = select(t, _str_for(t, parameters=params), Thresholds())
    assert [b.quad for b in sel.bonds] == [(1, 5, 8, 11)]


def test_selection_json_records_every_decision():
    t, s = _pfoa()
    data = selection_to_json(select(t, s, Thresholds()), s)
    json.dumps(data)  # serialisable
    assert data["thresholds"] == {"charge": 10.0, "dihedral": 50.0}
    assert data["cgenff_program_version"] == "4.0"
    assert data["resi_param_penalty"] == 69.0
    assert [b["quad_1_based"] for b in data["bonds"]] == [[1, 3, 4, 5], [25, 1, 3, 4]]
    atom1 = data["atoms"][0]
    assert atom1["top_name"] == "C1" and atom1["str_name"] == "C1" and atom1["flagged"] is True
    atom2 = data["atoms"][1]
    assert (atom2["top_name"], atom2["str_name"]) == ("O2", "O1")
    assert len(data["dihedral_types"]) == 13
    assert len(data["other_parameters"]) == 11
```

- [ ] **Step 2: Run them to verify they fail**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_selection.py -q'`
Expected: `ModuleNotFoundError: No module named 'ffrefine.selection'`.

- [ ] **Step 3: Write `ffrefine/selection.py`**

The mapping, flagging, matching, ring and tie-break logic below was run on both fixtures while planning and gave the facts table's results.

```python
"""Decide from CGenFF penalties which charges and torsions to refine.

Milestone 6b design, section 4. Pure: a parsed topology and a parsed ``.str``
in, a :class:`Selection` out. Nothing here reads a file or starts a process.

All atom numbers are 1-based, matching the topology's ``[ atoms ]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .charges.equivalence import equivalent_hydrogen_groups
from .errors import ValidationError
from .io.cgenff import StrFile, StrParameter
from .io.topology import Topology
from .io.validate import element_of

Quad = tuple[int, int, int, int]
Bond = tuple[int, int]
Types4 = tuple[str, str, str, str]

STR_CHARGE_TOLERANCE = 0.0015
"""How far a topology charge may sit from the ``.str``'s before it is reported.
The ``.str`` prints three decimals; half a unit in the last place plus a
little is all rounding can explain."""

MAX_LISTED = 5
"""How many mismatches a message names before it summarises the rest."""


@dataclass(frozen=True)
class Thresholds:
    """Penalty thresholds; a penalty is flagged when it is >= the threshold."""

    charge: float = 10.0
    dihedral: float = 50.0


@dataclass
class AtomFlag:
    """One atom, as the ``.str`` and the topology describe it."""

    nr: int
    top_name: str
    str_name: str
    atype: str
    charge_top: float
    charge_str: float
    penalty: float
    flagged: bool
    by_equivalence: bool = False


@dataclass
class DihedralTypeFlag:
    """One dihedral type CGenFF assigned, and the topology quads it matches."""

    types: Types4
    penalty: float
    flagged: bool
    quads: list[Quad]


@dataclass
class BondChoice:
    """One central bond selected (or, in a ring, excluded) for refitting."""

    bond: Bond
    penalty: float
    quad: Quad
    types: list[Types4]


@dataclass
class Selection:
    """Everything decided from a ``.str``; ``00_input/selection.json`` records it."""

    thresholds: Thresholds
    atoms: list[AtomFlag]
    dihedral_types: list[DihedralTypeFlag]
    other_parameters: list[StrParameter]
    bonds: list[BondChoice]
    ring_bonds: list[BondChoice]
    unmatched_types: list[Types4]
    warnings: list[str] = field(default_factory=list)

    @property
    def flagged_atoms(self) -> list[int]:
        """1-based atom numbers to refine, ascending."""
        return [a.nr for a in self.atoms if a.flagged]

    @property
    def quads(self) -> list[Quad]:
        """The selected bonds' quadruplets, in run order."""
        return [b.quad for b in self.bonds]

    @property
    def highest_dihedral_penalty(self) -> float:
        """The largest dihedral penalty in the ``.str``; 0 when it has none."""
        return max((d.penalty for d in self.dihedral_types), default=0.0)


def check_mapping(topology: Topology, strfile: StrFile) -> list[str]:
    """Check the ``.str`` describes this topology, atom for atom.

    Atoms correspond by index: the CGenFF export renames atoms, so names are
    never compared. Types must agree exactly; charges may differ (the
    topology may have been edited on purpose) and are only reported.

    Returns:
        Warnings about charge differences.

    Raises:
        ValidationError: the atom counts differ, or a type differs at some index.
    """
    if len(strfile.atoms) != topology.natoms:
        raise ValidationError(
            f"{strfile.source} describes {len(strfile.atoms)} atoms but the topology "
            f"{topology.source} has {topology.natoms}; the .str belongs to another molecule."
        )
    mismatches = [
        f"atom {a.nr} is {a.atype} in the topology, {s.atype} in the .str ({s.name})"
        for a, s in zip(topology.atoms, strfile.atoms)
        if a.atype != s.atype
    ]
    if mismatches:
        more = len(mismatches) - MAX_LISTED
        raise ValidationError(
            f"{strfile.source} does not describe {topology.source} atom for atom: "
            + "; ".join(mismatches[:MAX_LISTED])
            + (f"; and {more} more" if more > 0 else "")
            + ". Atoms are matched by position, so the .str probably belongs to another "
            "molecule or another atom order."
        )
    differ = [
        f"{a.nr} ({a.charge:+.3f} in the topology, {s.charge:+.3f} in the .str)"
        for a, s in zip(topology.atoms, strfile.atoms)
        if abs(a.charge - s.charge) > STR_CHARGE_TOLERANCE
    ]
    if not differ:
        return []
    return [
        f"{len(differ)} atom charge(s) in {topology.source.name} differ from "
        f"{strfile.source.name}: {', '.join(differ[:MAX_LISTED])}"
        + (" ..." if len(differ) > MAX_LISTED else "")
        + ". The penalties describe the .str's charges; check the topology was not edited "
        "by mistake, and that the two list their atoms in the same order."
    ]


def _in_ring(topology: Topology, bond: Bond) -> bool:
    """Whether the two atoms stay connected once their bond is removed."""
    j, k = bond
    adjacency: dict[int, set[int]] = {}
    for a, b in topology.bonds:
        if {a, b} == {j, k}:
            continue
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    seen, stack = {j}, [j]
    while stack:
        for n in adjacency.get(stack.pop(), ()):
            if n == k:
                return True
            if n not in seen:
                seen.add(n)
                stack.append(n)
    return False


def select(topology: Topology, strfile: StrFile, thresholds: Thresholds) -> Selection:
    """Map the ``.str`` onto the topology and decide what to refine.

    Atoms are flagged at charge penalty >= ``thresholds.charge`` and extended
    to whole equivalent-hydrogen groups. Dihedral types are flagged at
    penalty >= ``thresholds.dihedral`` and matched to proper ``[ dihedrals ]``
    lines by atom type, forward or reversed; each matched central bond is a
    candidate carrying the highest penalty among its flagged types. Ring
    bonds cannot be scanned and are set aside. Each bond's quadruplet is the
    matched line of highest penalty, then one with two heavy outer atoms,
    then the lowest outer indices. Bonds are ordered by penalty, highest
    first, ties by atom number.

    Raises:
        ValidationError: see :func:`check_mapping`.
    """
    warnings = check_mapping(topology, strfile)

    atoms = [
        AtomFlag(
            nr=a.nr,
            top_name=a.name,
            str_name=s.name,
            atype=a.atype,
            charge_top=a.charge,
            charge_str=s.charge,
            penalty=s.penalty,
            flagged=s.penalty >= thresholds.charge,
        )
        for a, s in zip(topology.atoms, strfile.atoms)
    ]
    flagged = {a.nr for a in atoms if a.flagged}
    for group in equivalent_hydrogen_groups(topology):
        if flagged & set(group) and not set(group) <= flagged:
            for nr in group:
                if nr not in flagged:
                    atoms[nr - 1].flagged = True
                    atoms[nr - 1].by_equivalence = True
            flagged |= set(group)

    types_of = {a.nr: a.atype for a in topology.atoms}
    dihedral_types: list[DihedralTypeFlag] = []
    candidates: dict[Bond, list[tuple[float, Quad, Types4]]] = {}
    unmatched: list[Types4] = []
    for param in strfile.of_kind("dihedral"):
        types: Types4 = tuple(param.types)  # type: ignore[assignment]
        quads: list[Quad] = []
        for d in topology.propers:
            quad = (d.ai, d.aj, d.ak, d.al)
            line_types = tuple(types_of[n] for n in quad)
            if (line_types == types or line_types[::-1] == types) and quad not in quads:
                quads.append(quad)
        is_flagged = param.penalty >= thresholds.dihedral
        dihedral_types.append(DihedralTypeFlag(types, param.penalty, is_flagged, quads))
        if not is_flagged:
            continue
        if not quads:
            unmatched.append(types)
            warnings.append(
                f"CGenFF penalises the dihedral type {' '.join(types)} at {param.penalty:g}, "
                "but no [ dihedrals ] line in the topology has those types; the .str may "
                "belong to another molecule."
            )
        for quad in quads:
            bond = (min(quad[1], quad[2]), max(quad[1], quad[2]))
            candidates.setdefault(bond, []).append((param.penalty, quad, types))

    def heavy_ends(quad: Quad) -> bool:
        return element_of(topology, quad[0]) != "H" and element_of(topology, quad[3]) != "H"

    bonds: list[BondChoice] = []
    rings: list[BondChoice] = []
    for bond, matches in candidates.items():
        penalty = max(p for p, _, _ in matches)
        best = min(
            matches,
            key=lambda m: (
                -m[0],
                not heavy_ends(m[1]),
                min(m[1][0], m[1][3]),
                max(m[1][0], m[1][3]),
            ),
        )
        matched_types = list(dict.fromkeys(t for _, _, t in matches))
        choice = BondChoice(bond, penalty, best[1], matched_types)
        (rings if _in_ring(topology, bond) else bonds).append(choice)
    bonds.sort(key=lambda b: (-b.penalty, b.bond))
    rings.sort(key=lambda b: (-b.penalty, b.bond))

    return Selection(
        thresholds=thresholds,
        atoms=atoms,
        dihedral_types=dihedral_types,
        other_parameters=[p for p in strfile.parameters if p.kind != "dihedral"],
        bonds=bonds,
        ring_bonds=rings,
        unmatched_types=unmatched,
        warnings=warnings,
    )


def _bond_json(b: BondChoice) -> dict:
    return {
        "bond_1_based": list(b.bond),
        "penalty": b.penalty,
        "quad_1_based": list(b.quad),
        "types": [list(t) for t in b.types],
    }


def selection_to_json(selection: Selection, strfile: StrFile) -> dict:
    """What ``00_input/selection.json`` records (design section 4)."""
    return {
        "str": str(strfile.source.resolve()),
        "cgenff_program_version": strfile.program_version,
        "cgenff_parameter_version": strfile.parameter_version,
        "resi_param_penalty": strfile.param_penalty,
        "resi_charge_penalty": strfile.charge_penalty,
        "thresholds": {
            "charge": selection.thresholds.charge,
            "dihedral": selection.thresholds.dihedral,
        },
        "atoms": [
            {
                "nr": a.nr,
                "top_name": a.top_name,
                "str_name": a.str_name,
                "type": a.atype,
                "charge_topology": a.charge_top,
                "charge_str": a.charge_str,
                "penalty": a.penalty,
                "flagged": a.flagged,
                "by_equivalence": a.by_equivalence,
            }
            for a in selection.atoms
        ],
        "dihedral_types": [
            {
                "types": list(d.types),
                "penalty": d.penalty,
                "flagged": d.flagged,
                "quads_1_based": [list(q) for q in d.quads],
            }
            for d in selection.dihedral_types
        ],
        "other_parameters": [
            {"kind": p.kind, "types": list(p.types), "penalty": p.penalty}
            for p in selection.other_parameters
        ],
        "bonds": [_bond_json(b) for b in selection.bonds],
        "ring_bonds": [_bond_json(b) for b in selection.ring_bonds],
        "unmatched_types": [list(t) for t in selection.unmatched_types],
        "warnings": list(selection.warnings),
    }
```

- [ ] **Step 4: Run the tests**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_selection.py -q'`
Expected: all pass.

- [ ] **Step 5: black and commit**

```bash
wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m black ffrefine/selection.py tests/test_selection.py'
cd /c/Users/hrach/Projects/ffrefine
git add ffrefine/selection.py tests/test_selection.py
git commit -m "feat: choose atoms and bonds to refine from CGenFF penalties

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: `--str` on the command line; `selection.json`; `charges --str`

**Files:**
- Modify: `ffrefine/cli.py` (flags; `_penalty_threshold`, `_penalty_threshold_arg`, `_thresholds`, `_selection`; `_resolve_quad`; `_charges_stage`)
- Create: `tests/test_cli_str.py`

**Interfaces:**
- Consumes: Task 6's `read_str`; Task 7's `Thresholds`, `Selection`, `select`, `selection_to_json`.
- Produces:
  - Flags `--str PATH` and `--penalty-threshold KIND=VALUE` (repeatable) on `scan`, `mmscan`, `fit`, `charges`, `validate`, `run`; `--no-charges` on `run`. Parser defaults: `str=None`, `penalty_threshold=None`, `no_charges=False`.
  - `cli._thresholds(args) -> Thresholds`
  - `cli._selection(args, topology) -> Selection`: parses, selects, emits the selection's warnings, writes `00_input/selection.json` and a copy of the `.str` into `00_input/` (not under `--dry-run`), and caches the result on `args.selection`; a second call returns the cached object.
  - `_resolve_quad` uses the one flagged bond when `--str` is given without `--dihedral`.
  - `charges --str` refines the flagged atoms; nothing flagged prints a message and returns 0.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli_str.py`:

```python
"""--str on the command line (milestone 6b design, sections 5 and 7)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ffrefine import cli
from ffrefine.cli import build_parser, main

FIXTURES = Path(__file__).parent.parent / "fixtures"
PFOA = FIXTURES / "pfoa_gromacs"
CAP = FIXTURES / "capecitabine_gromacs"


def _inputs(mol_dir: Path, stem: str) -> list[str]:
    return ["-s", str(mol_dir / f"{stem}_gmx.pdb"), "-p", str(mol_dir / f"{stem}_gmx.top")]


def _butane_str(path: Path, butane_itp: Path, penalties: dict[int, float]) -> Path:
    """A minimal CGenFF-style .str describing the butane fixture atom for atom."""
    from ffrefine.io.topology import read_topology

    t = read_topology(butane_itp)
    lines = ["* test stream", "read rtf card append", "RESI BUT 0.000", "GROUP"]
    for a in t.atoms:
        lines.append(f"ATOM {a.name} {a.atype} {a.charge:.3f} ! {penalties.get(a.nr, 0.0):.3f}")
    lines += ["END", "read param card flex append", "DIHEDRALS", "END", "RETURN"]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_penalty_threshold_parses_kind_and_value():
    args = build_parser().parse_args(
        ["scan", "-s", "a.pdb", "-p", "a.top", "--penalty-threshold", "dihedral=40"]
        + ["--penalty-threshold", "charge=12.5"]
    )
    assert cli._thresholds(args).dihedral == 40.0
    assert cli._thresholds(args).charge == 12.5
    defaults = build_parser().parse_args(["scan", "-s", "a.pdb", "-p", "a.top"])
    assert (cli._thresholds(defaults).charge, cli._thresholds(defaults).dihedral) == (10.0, 50.0)


@pytest.mark.parametrize("bad", ["dihedral", "angle=3", "charge=-1", "dihedral=x"])
def test_a_malformed_penalty_threshold_is_a_usage_error(bad, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["scan", "-s", "a.pdb", "-p", "a.top", "--penalty-threshold", bad])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--penalty-threshold" in err and "Traceback" not in err


def test_str_picks_the_one_flagged_bond_for_a_standalone_stage(tmp_path, capsys):
    argv = ["scan"] + _inputs(CAP, "capecitabine") + ["--str", str(CAP / "capecitabine.str")]
    assert main(argv + ["--outdir", str(tmp_path / "out"), "--dry-run"]) == 0
    assert "(6, 7, 9, 10)" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()  # a dry run writes nothing, selection.json included


def test_several_flagged_bonds_point_a_standalone_stage_to_run(tmp_path, capsys):
    argv = ["scan"] + _inputs(PFOA, "pfoa") + ["--str", str(PFOA / "pfoa.str")]
    assert main(argv + ["--outdir", str(tmp_path / "out"), "--dry-run"]) == 1
    err = capsys.readouterr().err
    assert "2 bonds" in err and "ffrefine run" in err and "1,3,4,5" in err


def test_no_flagged_bond_names_the_highest_penalty(tmp_path, capsys):
    argv = ["scan"] + _inputs(CAP, "capecitabine") + ["--str", str(CAP / "capecitabine.str")]
    argv += ["--penalty-threshold", "dihedral=51", "--outdir", str(tmp_path / "o"), "--dry-run"]
    assert main(argv) == 1
    err = capsys.readouterr().err
    assert "highest dihedral penalty" in err and "50" in err and "--penalty-threshold" in err


def test_a_missing_str_is_a_clean_error(tmp_path, capsys):
    """Review focus 5."""
    argv = ["scan"] + _inputs(CAP, "capecitabine") + ["--str", str(tmp_path / "nope.str")]
    assert main(argv + ["--outdir", str(tmp_path / "o"), "--dry-run"]) == 1
    err = capsys.readouterr().err
    assert "could not be read" in err and "Traceback" not in err


def test_two_dihedrals_on_a_standalone_stage_point_to_run(tmp_path, capsys):
    argv = ["scan"] + _inputs(PFOA, "pfoa") + ["--dihedral", "1,3,4,5", "--dihedral", "25,1,3,4"]
    assert main(argv + ["--outdir", str(tmp_path / "o"), "--dry-run"]) == 1
    assert "ffrefine run" in capsys.readouterr().err


def test_selection_is_computed_once_and_recorded(tmp_path):
    args = build_parser().parse_args(
        ["scan"] + _inputs(PFOA, "pfoa") + ["--str", str(PFOA / "pfoa.str"), "--outdir"]
        + [str(tmp_path / "out")]
    )
    cli._apply_config(args)
    from ffrefine.io.topology import read_topology

    t = read_topology(PFOA / "pfoa_gmx.top")
    first = cli._selection(args, t)
    assert cli._selection(args, t) is first
    data = json.loads((tmp_path / "out" / "00_input" / "selection.json").read_text())
    assert [b["quad_1_based"] for b in data["bonds"]] == [[1, 3, 4, 5], [25, 1, 3, 4]]
    assert (tmp_path / "out" / "00_input" / "pfoa.str").read_text() == (
        PFOA / "pfoa.str"
    ).read_text()


def test_charges_str_refines_the_flagged_atoms_and_their_group(
    tmp_path, butane_pdb, butane_itp, fake_esp
):
    """A flagged hydrogen brings its whole methyl group (design section 4)."""
    from ffrefine.io.topology import read_topology

    t = read_topology(butane_itp)
    first_h = next(a.nr for a in t.atoms if a.atype == "HGA3")
    strfile = _butane_str(tmp_path / "but.str", butane_itp, {first_h: 20.0})
    out = tmp_path / "out"
    argv = ["charges", "-s", str(butane_pdb), "-p", str(butane_itp), "--str", str(strfile)]
    assert main(argv + ["--charge-geom", "input", "--outdir", str(out)]) == 0
    charges = json.loads((out / "04_charges" / "charges.json").read_text())
    selected = charges["provenance"]["selected_atoms"]
    assert len(selected) == 3 and first_h in selected


def test_charges_str_with_nothing_flagged_runs_no_orca(
    tmp_path, butane_pdb, butane_itp, fake_esp, capsys
):
    strfile = _butane_str(tmp_path / "but.str", butane_itp, {})
    out = tmp_path / "out"
    argv = ["charges", "-s", str(butane_pdb), "-p", str(butane_itp), "--str", str(strfile)]
    assert main(argv + ["--charge-geom", "input", "--outdir", str(out)]) == 0
    assert "nothing to refine" in capsys.readouterr().out
    assert fake_esp.calls == 0
    assert not (out / "04_charges" / "charges.json").exists()


def test_replacing_most_of_the_molecule_warns(
    tmp_path, butane_pdb, butane_itp, fake_esp, capsys
):
    from ffrefine.io.topology import read_topology

    t = read_topology(butane_itp)
    strfile = _butane_str(tmp_path / "but.str", butane_itp, {a.nr: 20.0 for a in t.atoms})
    out = tmp_path / "out"
    argv = ["charges", "-s", str(butane_pdb), "-p", str(butane_itp), "--str", str(strfile)]
    assert main(argv + ["--charge-geom", "input", "--outdir", str(out)]) == 0
    charges = json.loads((out / "04_charges" / "charges.json").read_text())
    assert any("close to replacing every charge" in w for w in charges["warnings"])
```

These tests need the fake ESP job that `tests/test_cli_charges.py` defines as its `fake_esp` fixture. Test modules cannot import each other here (no `__init__.py`), so move that fixture and its helpers (`FakeEsp`, `BUTANE_ESP`, `BOHR_IN_NM`, `_inverse_distances_bohr`, `_shell_points_bohr`, `_write_chelpg_vpot`, and any constant they use) **verbatim** from `tests/test_cli_charges.py` into `tests/conftest.py`, keeping the name `fake_esp`, then delete the moved definitions from `tests/test_cli_charges.py`; its tests keep using `fake_esp` from `conftest.py` unchanged.

- [ ] **Step 2: Run them to verify they fail**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_cli_str.py tests/test_cli_charges.py -q'`
Expected: `test_cli_charges.py` still passes (fixture moved); the new tests fail with `unrecognized arguments: --str` (exit 2) or `AttributeError: module 'ffrefine.cli' has no attribute '_thresholds'`.

- [ ] **Step 3: Add the flags and helpers in `ffrefine/cli.py`**

Imports:

```python
from .io.cgenff import read_str
from .selection import Selection, Thresholds, select, selection_to_json
```

Below `_atom_list_arg`, add:

```python
PENALTY_KINDS = ("charge", "dihedral")


def _penalty_threshold(text: str) -> tuple[str, float]:
    """Parse ``KIND=VALUE`` for ``--penalty-threshold``.

    Raises:
        ValidationError: not ``charge=`` or ``dihedral=`` followed by a
            non-negative number.
    """
    kind, sep, value = text.partition("=")
    kind = kind.strip().lower()
    if not sep or kind not in PENALTY_KINDS:
        raise ValidationError(
            f"{text!r} is not KIND=VALUE with KIND one of {', '.join(PENALTY_KINDS)} "
            "(e.g. dihedral=40)"
        )
    try:
        number = float(value)
    except ValueError as exc:
        raise ValidationError(f"{value.strip()!r} in {text!r} is not a number") from exc
    if number < 0:
        raise ValidationError(f"a penalty threshold cannot be negative ({text!r})")
    return kind, number


def _penalty_threshold_arg(text: str) -> tuple[str, float]:
    """argparse ``type=`` adapter for :func:`_penalty_threshold` (see :func:`_dihedral_arg`)."""
    try:
        return _penalty_threshold(text)
    except ValidationError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _add_str_flags(parser: argparse.ArgumentParser) -> None:
    """Add the CGenFF stream-file flags (milestone 6b)."""
    parser.add_argument(
        "--str",
        default=None,
        metavar="PATH",
        help="CGenFF .str file: choose the charges and torsions to refine from its penalties",
    )
    parser.add_argument(
        "--penalty-threshold",
        action="append",
        type=_penalty_threshold_arg,
        default=None,
        metavar="KIND=VALUE",
        help="flag penalties >= VALUE; KIND is charge (default 10) or dihedral (default 50); "
        "repeatable",
    )
```

In `build_parser`, call `_add_str_flags(...)` for `scan`, `mmscan`, `fit`, `charges`, `validate` and `run`, and add to `run` only:

```python
    run.add_argument(
        "--no-charges",
        action="store_true",
        help="with --str, skip refining the flagged charges and refit torsions only",
    )
```

Below `_input_topology`, add:

```python
def _thresholds(args: argparse.Namespace) -> Thresholds:
    """The penalty thresholds, defaults overridden by ``--penalty-threshold``."""
    values = {"charge": Thresholds.charge, "dihedral": Thresholds.dihedral}
    for kind, value in getattr(args, "penalty_threshold", None) or []:
        values[kind] = value
    return Thresholds(**values)


def _selection(args: argparse.Namespace, topology: Topology) -> Selection:
    """What ``--str`` selects for this topology; computed and recorded once.

    The first call parses the ``.str``, selects, emits the selection's
    warnings and -- unless ``--dry-run`` -- writes ``00_input/selection.json``
    and a copy of the ``.str`` beside it. The result is cached on ``args`` so
    ``run`` and the stages it calls agree without recomputing.

    Raises:
        ParseError: the ``.str`` could not be read or parsed.
        ValidationError: it does not describe this topology.
    """
    cached = getattr(args, "selection", None)
    if cached is not None:
        return cached
    strfile = read_str(args.str)
    selection = select(topology, strfile, _thresholds(args))
    for warning in selection.warnings:
        _warn(warning)
    if not args.dry_run:
        target = stage_dir(args.outdir, "input")
        (target / "selection.json").write_text(
            json.dumps(selection_to_json(selection, strfile), indent=2) + "\n", newline="\n"
        )
        if (target / strfile.source.name).resolve() != strfile.source.resolve():
            shutil.copyfile(strfile.source, target / strfile.source.name)
    args.selection = selection
    return selection
```

- [ ] **Step 4: Use the selection in `_resolve_quad`**

Replace `_resolve_quad`'s body with:

```python
    quads = list(getattr(args, "dihedral", None) or [])
    if not quads and getattr(args, "str", None):
        selection = _selection(args, topology)
        if not selection.bonds:
            raise ValidationError(
                f"no torsion in {args.str} has a dihedral penalty >= "
                f"{selection.thresholds.dihedral:g}; the highest dihedral penalty is "
                f"{selection.highest_dihedral_penalty:g}. Pass --dihedral i,j,k,l, or a lower "
                "--penalty-threshold dihedral=VALUE."
            )
        if len(selection.bonds) > 1:
            listed = ", ".join(
                f"{','.join(map(str, b.quad))} (penalty {b.penalty:g})" for b in selection.bonds
            )
            raise ValidationError(
                f"{len(selection.bonds)} bonds are flagged in {args.str}: {listed}. A single "
                "stage fits one torsion per outdir, and a later bond must be fitted against "
                "the earlier ones' refits; 'ffrefine run' does that in sequence, one subfolder "
                "per bond. Or pass --dihedral to pick one here."
            )
        quads = [selection.bonds[0].quad]
    if not quads:
        raise ValidationError(
            "no torsion selected. Pass --dihedral i,j,k,l using 1-based atom indices, or "
            "--str mol.str to choose it from CGenFF's penalties."
        )
    if len(quads) > 1:
        raise ValidationError(
            f"{len(quads)} torsions requested, but a single stage fits one torsion per "
            "outdir, and a later torsion must be fitted against the earlier ones' refits. "
            "'ffrefine run' fits several in sequence, one subfolder per bond."
        )
    quad = quads[0]
    validate_quad(topology, quad)
    return quad
```

and update its docstring's Raises: "no ``--dihedral`` and no usable ``--str`` selection, more than one torsion, or the quadruplet is unusable with this topology." (No existing test asserts on the old refusal texts; checked while planning.)

- [ ] **Step 5: Use the selection in `_charges_stage`**

Replace the opening block of `_charges_stage` (from `selected_atoms = None` through the end of its `if args.charge_mode == "selected":` block, stopping before `elif args.charge_atoms:`) with:

```python
    selected_atoms = None
    from_str = False
    if args.charge_mode == "selected":
        if args.charge_atoms:
            selected_atoms = sorted({nr for group in args.charge_atoms for nr in group})
        elif getattr(args, "str", None):
            from_str = True
        else:
            raise ValidationError(
                "--charge-mode selected replaces only the atoms you name, and none were "
                "named. Pass --charge-atoms i,j,k (1-based), --str mol.str to take them from "
                "CGenFF's charge penalties, or --charge-mode all to replace every charge -- "
                "which on a CGenFF topology changes the derivation scheme, so read the "
                "warning it prints."
            )
```

The `elif args.charge_atoms: raise ValidationError("--charge-atoms names the atoms to replace under --charge-mode selected, but ...")` branch that follows stays exactly as it is.

In the dry-run branch, directly after `structure, topology = _load_structure_and_topology(args)`, and in the real branch, directly after `structure, topology, _ = _prepare(args, need_quad=False)`, insert the same block:

```python
    if from_str:
        selection = _selection(args, topology)
        selected_atoms = selection.flagged_atoms
        if not selected_atoms:
            print(
                f"charges: no atom in {args.str} has a charge penalty >= "
                f"{selection.thresholds.charge:g}; nothing to refine"
            )
            return 0
```

After `check_selection(topology, selected_atoms, target_total)` in the real branch (and after the dry run's `check_selection` call), add:

```python
    if selected_atoms is not None and len(selected_atoms) > topology.natoms / 2:
        _warn(
            f"--charge-mode selected replaces {len(selected_atoms)} of the {topology.natoms} "
            "atoms here, so this is close to replacing every charge with gas-phase ESP "
            "charges."
            + (
                " CGenFF charges are fitted to water interaction energies and dipoles, not "
                "to the gas-phase ESP, so this is a change of derivation scheme and can "
                "degrade condensed-phase behaviour."
                if _looks_like_cgenff(topology)
                else ""
            )
        )
```

- [ ] **Step 6: Run the tests**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_cli_str.py tests/test_cli_charges.py tests/test_cli.py -q -rs && python -m pytest -q -rs'`
Expected: all pass, 0 skipped.

- [ ] **Step 7: black and commit**

```bash
wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m black ffrefine tests'
cd /c/Users/hrach/Projects/ffrefine
git add ffrefine/cli.py tests/conftest.py tests/test_cli_str.py tests/test_cli_charges.py tests/test_cli.py
git commit -m "feat(cli): --str and --penalty-threshold; selection.json; charges from CGenFF penalties

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: The combined check

**Files:**
- Modify: `ffrefine/mm/validate_fit.py` (extract `_energy_comparison`; add `compare_topologies`)
- Modify: `ffrefine/plots.py` (`plot_validation` gains `title` and `labels`)
- Modify: `ffrefine/cli.py` (`_combined_check`)
- Create: `tests/test_combined.py`
- Test: `tests/test_validate_fit.py` (append)

**Interfaces:**
- Consumes: Task 5's `_collect_warnings`, `_current_warnings`; M6a's `preprocessed_sha256`, `write_molecule_copy`, `make_system_top`, `single_point_energies`; `profile_rmse`, `_is_improved`, `ValidationResult` in `validate_fit.py`.
- Produces:
  - `ffrefine.mm.validate_fit.compare_topologies(binaries, before: Topology, after: Topology, structure, frames, qm_kj, quad, workdir, nthreads=1, strict_names=True, ff_includes=None) -> ValidationResult` — "before"/"after" are two whole topologies, energies on the same frames.
  - `plot_validation(..., path, title="Validation: MM against QM before and after refitting", labels=("MM before", "MM after"))`
  - `cli._combined_check(args, bonds: list[tuple[Quad, Path]], original: Path, final: Path) -> None`, writing `final/combined.json` (keys `original_topology`, `final_topology`, `bonds`, `warnings`; each bond row `quad_1_based`, `dir`, `rmse_original`, `rmse_final`, `rmse_own_before`, `rmse_own_after`, `improved`, `phi_deg`, `qm_kj_per_mol`, `mm_original_kj_per_mol`, `mm_final_kj_per_mol`) and `final/plots/combined_<dir>.png`; resumable under the `final` stage manifest.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_validate_fit.py`:

```python
@pytest.mark.needs_gmx
def test_compare_topologies_of_one_topology_with_itself_agrees(
    tmp_path, butane_pdb, butane_itp, gmx_bin, rotated_frames
):
    from ffrefine.config import Binaries
    from ffrefine.io.structure import read_pdb
    from ffrefine.io.topology import read_topology
    from ffrefine.mm.validate_fit import compare_topologies

    s = read_pdb(butane_pdb)
    t = read_topology(butane_itp)
    frames = rotated_frames(s, 6)
    qm = np.linspace(0.0, 5.0, 6)
    r = compare_topologies(Binaries(gmx=gmx_bin), t, t, s, frames, qm, (1, 5, 8, 11), tmp_path)
    assert np.array_equal(r.mm_before_kj, r.mm_after_kj)
    assert r.rmse_before == r.rmse_after
    assert r.phi_deg.shape == (6,)
```

Create `tests/test_combined.py`:

```python
"""The combined check: every bond's scan, original topology against final (design section 5)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ffrefine import cli
from ffrefine.cli import build_parser
from ffrefine.mm.validate_fit import ValidationResult

QUADS = [(1, 5, 8, 11), (2, 1, 5, 8)]


def _bond_tree(outdir: Path, quad, rmse_before: float, rmse_after: float) -> Path:
    """A finished bond subfolder: its scan and its own validation."""
    bond = outdir / cli._bond_dirname(quad)
    (bond / "01_qm_scan").mkdir(parents=True)
    frames = np.arange(4 * 14 * 3, dtype=float).reshape(4, 14, 3) * 0.01
    (bond / "01_qm_scan" / "scan.json").write_text(
        json.dumps({"quad": list(quad), "energy_kj": [0.0, 1.0, 2.0, 1.0], "frames_nm": frames.tolist()})
    )
    (bond / "final").mkdir()
    (bond / "final" / "validation.json").write_text(
        json.dumps({"rmse_before_kj_per_mol": rmse_before, "rmse_after_kj_per_mol": rmse_after})
    )
    return bond


def _args(outdir, butane_pdb, butane_itp, *extra):
    args = build_parser().parse_args(
        ["run", "-s", str(butane_pdb), "-p", str(butane_itp), "--outdir", str(outdir), *extra]
    )
    cli._apply_config(args)
    return args


def _stub(monkeypatch, calls, improved=(True, True)):
    outcomes = iter(improved)

    def fake_compare(binaries, before, after, structure, frames, qm, quad, workdir, **kwargs):
        calls.append(quad)
        good = next(outcomes)
        return ValidationResult(
            rmse_before=2.0,
            rmse_after=1.0 if good else 3.0,
            mm_before_kj=np.zeros(4),
            mm_after_kj=np.ones(4),
            qm_kj=np.asarray(qm, float),
            phi_deg=np.array([0.0, 90.0, 180.0, 270.0]),
            improved=good,
        )

    monkeypatch.setattr(cli, "compare_topologies", fake_compare)
    monkeypatch.setattr(cli, "preprocessed_sha256", lambda *a, **k: "d" * 64)
    monkeypatch.setattr(cli, "plot_validation", lambda *a, **k: None)


def test_combined_check_records_every_bond_in_run_order(
    tmp_path, butane_pdb, butane_itp, monkeypatch
):
    out = tmp_path / "out"
    bonds = [(q, _bond_tree(out, q, 3.0, 1.5)) for q in QUADS]
    calls: list = []
    _stub(monkeypatch, calls, improved=(True, False))
    cli._combined_check(_args(out, butane_pdb, butane_itp), bonds, butane_itp, butane_itp)
    assert calls == QUADS
    data = json.loads((out / "final" / "combined.json").read_text())
    assert [row["quad_1_based"] for row in data["bonds"]] == [list(q) for q in QUADS]
    first = data["bonds"][0]
    assert (first["rmse_original"], first["rmse_final"]) == (2.0, 1.0)
    assert (first["rmse_own_before"], first["rmse_own_after"]) == (3.0, 1.5)
    assert first["dir"] == "torsion_1-5-8-11"
    assert data["bonds"][1]["improved"] is False
    assert any("no better than the original" in w for w in data["warnings"])


def test_combined_check_resumes(tmp_path, butane_pdb, butane_itp, monkeypatch):
    """Review focus 1: a finished combined check is skipped under --resume,
    and recomputed when the final topology changes."""
    out = tmp_path / "out"
    bonds = [(q, _bond_tree(out, q, 3.0, 1.5)) for q in QUADS]
    final = tmp_path / "final.itp"
    final.write_text(butane_itp.read_text())
    calls: list = []
    _stub(monkeypatch, calls, improved=(True,) * 6)
    cli._combined_check(_args(out, butane_pdb, butane_itp), bonds, butane_itp, final)
    assert len(calls) == 2
    cli._combined_check(_args(out, butane_pdb, butane_itp, "--resume"), bonds, butane_itp, final)
    assert len(calls) == 2
    final.write_text(butane_itp.read_text() + "; edited\n")
    cli._combined_check(_args(out, butane_pdb, butane_itp, "--resume"), bonds, butane_itp, final)
    assert len(calls) == 4
```

- [ ] **Step 2: Run them to verify they fail**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_combined.py tests/test_validate_fit.py -q -rs'`
Expected: `AttributeError: module 'ffrefine.cli' has no attribute '_bond_dirname'` (Task 10 adds it; add it now as part of this task — see Step 5) and `ImportError: cannot import name 'compare_topologies'`.

- [ ] **Step 3: Split `validate_refit` and add `compare_topologies` in `ffrefine/mm/validate_fit.py`**

Add, above `validate_refit`:

```python
def _energy_comparison(
    binaries: Binaries,
    before_top: Path,
    after_top: Path,
    structure: Structure,
    frames: np.ndarray,
    qm_kj: np.ndarray,
    quad: Quad,
    workdir: Path,
    nthreads: int,
) -> ValidationResult:
    """MM single points for two wrapper ``.top`` files on the same frames, scored against QM."""
    e_before = single_point_energies(
        binaries, before_top, structure, frames, workdir / "before", "before", nthreads
    )
    e_after = single_point_energies(
        binaries, after_top, structure, frames, workdir / "after", "after", nthreads
    )
    mm_before = e_before.potential()
    mm_after = e_after.potential()
    qm = np.asarray(qm_kj, dtype=float)
    rmse_before = profile_rmse(qm, mm_before)
    rmse_after = profile_rmse(qm, mm_after)
    # quad is 1-based (user-facing, topology-file convention); frames is
    # indexed 0-based, so convert here at the boundary.
    idx = (quad[0] - 1, quad[1] - 1, quad[2] - 1, quad[3] - 1)
    phi = dihedral_deg_frames(np.asarray(frames, dtype=float), idx)
    return ValidationResult(
        rmse_before=rmse_before,
        rmse_after=rmse_after,
        mm_before_kj=mm_before,
        mm_after_kj=mm_after,
        qm_kj=qm,
        phi_deg=phi,
        improved=_is_improved(rmse_before, rmse_after),
    )
```

In `validate_refit`, replace everything from `e_before = single_point_energies(` down to (not including) its closing `log.info("validation: MM against QM RMSE ...")` with:

```python
    result = _energy_comparison(
        binaries, before_top, after_top, structure, frames, qm_kj, quad, workdir, nthreads
    )
```

and rewrite its tail to use `result`:

```python
    log.info(
        "validation: MM against QM RMSE %.3f kJ/mol before, %.3f kJ/mol after refitting",
        result.rmse_before,
        result.rmse_after,
    )
    if not result.improved:
        log.warning(
            "refitting did not improve agreement (%.3f -> %.3f kJ/mol). Check the scan "
            "for conformational hops and the self-consistency diagnostics.",
            result.rmse_before,
            result.rmse_after,
        )
    return result
```

Append:

```python
def compare_topologies(
    binaries: Binaries,
    before: Topology,
    after: Topology,
    structure: Structure,
    frames: np.ndarray,
    qm_kj: np.ndarray,
    quad: Quad,
    workdir: str | Path,
    nthreads: int = 1,
    strict_names: bool = True,
    ff_includes: list[str] | None = None,
) -> ValidationResult:
    """MM against QM on one scan's frames, for two whole topologies.

    The milestone 6b combined check: ``before`` is the user's original
    topology, ``after`` the final one carrying every refit (and any refined
    charges). Neither is modified; both are copied (absolute includes) and
    re-run through the same single-point engine as :func:`validate_refit`.

    Args:
        quad: 1-based; used only to measure the scan angle for plotting.
        Others: as for :func:`validate_refit`.

    Raises:
        ValidationError: ``quad`` is unusable with either topology, or the
            structure and a topology disagree.
    """
    for topology in (before, after):
        validate_quad(topology, quad)
        for warning in validate_structure_topology(structure, topology, strict_names=strict_names):
            log.warning("structure/topology validation: %s", warning)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    tops = []
    for stem, topology in (("original", before), ("final", after)):
        itp = workdir / f"{stem}.itp"
        write_molecule_copy(topology.source, itp)
        top = workdir / f"{stem}.top"
        make_system_top(itp, top, topology.name, ff_includes)
        tops.append(top)
    return _energy_comparison(
        binaries, tops[0], tops[1], structure, frames, qm_kj, quad, workdir, nthreads
    )
```

- [ ] **Step 4: Title and labels for `plot_validation` in `ffrefine/plots.py`**

Change its signature to

```python
def plot_validation(
    phi_deg: np.ndarray,
    qm_kj: np.ndarray,
    mm_before_kj: np.ndarray,
    mm_after_kj: np.ndarray,
    rmse_before: float,
    rmse_after: float,
    path: str | Path,
    title: str = "Validation: MM against QM before and after refitting",
    labels: tuple[str, str] = ("MM before", "MM after"),
) -> None:
```

and use `label=f"{labels[0]} (RMSE {rmse_before:.2f})"`, `label=f"{labels[1]} (RMSE {rmse_after:.2f})"` and `_finish(ax, path, title)`.

- [ ] **Step 5: `_bond_dirname` and `_combined_check` in `ffrefine/cli.py`**

Imports: add `compare_topologies` to the `from .mm.validate_fit import ...` line; add `preprocessed_sha256` if Task 8 of milestone 6a's import does not already name it (it does).

Add below `_input_topology`:

```python
def _bond_dirname(quad: Quad) -> str:
    """The subfolder a bond's stages run in when a run refits several: ``torsion_i-j-k-l``."""
    return "torsion_" + "-".join(str(n) for n in quad)
```

Add above `cmd_run`:

```python
def _combined_check(
    args: argparse.Namespace, bonds: list[tuple[Quad, Path]], original: Path, final: Path
) -> None:
    """Re-run every bond's scan with the original and the final topology (design section 5).

    Each bond was fitted against the topology as it stood before the later
    bonds were refit, so their refits can shift its profile. This measures
    that: per bond, RMSE against QM with the user's original topology and
    with the final one, beside the bond's own before/after from its
    ``validation.json``. Written to ``final/combined.json`` and one plot per
    bond; resumable under the ``final`` stage manifest.
    """
    with _collect_warnings():
        setup_logging(args.outdir, args.verbose)
        before = read_topology(original)
        after = read_topology(final)
        structure = adopt_topology_naming(read_structure(args.structure), before)
        binaries = _binaries(args)
        final_dir = stage_dir(args.outdir, "final")
        work = final_dir / "combined"
        ff_includes = _ff_includes(args)
        scan_paths = [bond_dir / STAGE_DIRS["qm_scan"] / "scan.json" for _, bond_dir in bonds]
        settings = {
            "quads": [list(q) for q, _ in bonds],
            "ff_includes": ff_includes,
            "original_preprocessed_sha256": preprocessed_sha256(
                binaries, before, structure, work, "original", ff_includes
            ),
            "final_preprocessed_sha256": preprocessed_sha256(
                binaries, after, structure, work, "final", ff_includes
            ),
        }
        digest = hash_inputs([original, final, *scan_paths], settings)
        combined_json = final_dir / "combined.json"
        if args.resume and stage_is_current(args.outdir, "final", digest, [combined_json]):
            print(f"combined check: skipping, {combined_json} is current (--resume)")
            return

        rows = []
        for (quad, bond_dir), scan_path in zip(bonds, scan_paths):
            scan = _read_json(scan_path, "QM scan (scan.json)", _SCAN_REMEDY)
            frames = _require_array(scan, "frames_nm", scan_path)
            qm = _require_array(scan, "energy_kj", scan_path)
            own_path = bond_dir / STAGE_DIRS["final"] / "validation.json"
            own = _read_json(own_path, "that bond's validation.json", "Its validate stage did not finish.")
            result = compare_topologies(
                binaries,
                before,
                after,
                structure,
                frames,
                qm,
                quad,
                work / bond_dir.name,
                nthreads=args.nthreads,
                strict_names=not args.no_strict_names,
                ff_includes=ff_includes,
            )
            plot_validation(
                result.phi_deg,
                result.qm_kj,
                result.mm_before_kj,
                result.mm_after_kj,
                result.rmse_before,
                result.rmse_after,
                final_dir / "plots" / f"combined_{bond_dir.name}.png",
                title=f"Combined check, torsion {'-'.join(map(str, quad))}: original against final",
                labels=("original topology", "final topology"),
            )
            if not result.improved:
                _warn(
                    f"torsion {'-'.join(map(str, quad))}: with every refit applied, the final "
                    f"topology agrees with QM no better than the original (RMSE "
                    f"{result.rmse_before:.3f} -> {result.rmse_after:.3f} kJ/mol). A later "
                    "bond's refit may have undone this one's; see "
                    f"final/plots/combined_{bond_dir.name}.png."
                )
            rows.append(
                {
                    "quad_1_based": list(quad),
                    "dir": bond_dir.name,
                    "rmse_original": float(result.rmse_before),
                    "rmse_final": float(result.rmse_after),
                    "rmse_own_before": own.get("rmse_before_kj_per_mol"),
                    "rmse_own_after": own.get("rmse_after_kj_per_mol"),
                    "improved": bool(result.improved),
                    "phi_deg": [float(v) for v in result.phi_deg],
                    "qm_kj_per_mol": [float(v) for v in result.qm_kj],
                    "mm_original_kj_per_mol": [float(v) for v in result.mm_before_kj],
                    "mm_final_kj_per_mol": [float(v) for v in result.mm_after_kj],
                }
            )
        payload = {
            "original_topology": str(Path(original).resolve()),
            "final_topology": str(Path(final).resolve()),
            "bonds": rows,
            "warnings": _current_warnings(),
        }
        combined_json.write_text(json.dumps(payload, indent=2) + "\n", newline="\n")
        write_manifest(args.outdir, "final", digest, {"bonds": len(rows)})
        print(f"combined check: wrote {combined_json}")
```

- [ ] **Step 6: Run the tests**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_combined.py tests/test_validate_fit.py tests/test_plots.py -q -rs && python -m pytest -q -rs'`
Expected: all pass, 0 skipped.

- [ ] **Step 7: black and commit**

```bash
wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m black ffrefine tests'
cd /c/Users/hrach/Projects/ffrefine
git add ffrefine/mm/validate_fit.py ffrefine/plots.py ffrefine/cli.py tests/test_combined.py tests/test_validate_fit.py
git commit -m "feat: combined check of every bond's scan against the final topology

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: `run` drives every bond in sequence

**Files:**
- Modify: `ffrefine/cli.py` (`_bond_args`, `_RunStatus`, `cmd_run`)
- Create: `tests/test_cli_run.py`

**Interfaces:**
- Consumes: Task 8's `_selection`, `_thresholds`; Task 9's `_bond_dirname`, `_combined_check`; the stage commands `cmd_charges`, `cmd_scan`, `cmd_mmscan`, `cmd_fit`, `cmd_validate` (looked up at call time, so tests can replace them).
- Produces:
  - `cli._bond_args(args, quad, topology: str) -> argparse.Namespace`
  - `cli._RunStatus` writing `final/run_status.json`: `{"charges": "not run"|"skipped"|"complete"|"failed", "torsions": [{"quad_1_based", "dir", "status": "not reached"|"running"|"complete"|"failed", "failed_stage"}], "combined": "not applicable"|"not run"|"complete"|"failed"}`; `"dir"` is `"."` for a single torsion.
  - `cmd_run` behaviour per design section 5.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli_run.py`:

```python
"""run: charges once, then every bond in sequence (milestone 6b design, section 5).

The stage commands are replaced by stand-ins that record what they were
given; the stand-in validate writes the refined topology the next bond reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ffrefine import cli
from ffrefine.cli import main
from ffrefine.errors import ValidationError

FIXTURES = Path(__file__).parent.parent / "fixtures"
PFOA = FIXTURES / "pfoa_gromacs"
CAP = FIXTURES / "capecitabine_gromacs"


def _pfoa(outdir: Path, *extra: str) -> list[str]:
    return [
        "run",
        "-s",
        str(PFOA / "pfoa_gmx.pdb"),
        "-p",
        str(PFOA / "pfoa_gmx.top"),
        "--str",
        str(PFOA / "pfoa.str"),
        "--outdir",
        str(outdir),
        *extra,
    ]


def _cap(outdir: Path, *extra: str) -> list[str]:
    return [
        "run",
        "-s",
        str(CAP / "capecitabine_gmx.pdb"),
        "-p",
        str(CAP / "capecitabine_gmx.top"),
        "--str",
        str(CAP / "capecitabine.str"),
        "--outdir",
        str(outdir),
        *extra,
    ]


@pytest.fixture
def stages(monkeypatch):
    """Stand-in stage commands; returns the list of (stage, outdir, topology, dihedral)."""
    calls: list[tuple] = []

    def record(name):
        def stage(a):
            calls.append((name, Path(a.outdir), a.topology, a.dihedral and list(a.dihedral)))
            if name == "validate":
                refined = Path(a.outdir) / "final" / "molecule_refined.itp"
                refined.parent.mkdir(parents=True, exist_ok=True)
                refined.write_text(f"; refined {a.dihedral[0]} from {a.topology}\n")
            if name == "charges":
                charged = Path(a.outdir) / "04_charges" / "molecule_charged.itp"
                charged.parent.mkdir(parents=True, exist_ok=True)
                charged.write_text(Path(a.topology).read_text())
            return 0

        return stage

    for name in ("charges", "scan", "mmscan", "fit", "validate"):
        monkeypatch.setattr(cli, f"cmd_{name}", record(name))
    combined: list[tuple] = []
    monkeypatch.setattr(
        cli, "_combined_check", lambda a, bonds, original, final: combined.append((bonds, original, final))
    )
    calls_and_combined = (calls, combined)
    return calls_and_combined


def test_bonds_run_in_penalty_order_each_on_the_previous_ones_topology(tmp_path, stages):
    calls, combined = stages
    out = tmp_path / "out"
    assert main(_pfoa(out, "--no-charges")) == 0
    torsion_calls = [c for c in calls if c[0] != "charges"]
    assert [c[0] for c in torsion_calls] == ["scan", "mmscan", "fit", "validate"] * 2
    first, second = torsion_calls[:4], torsion_calls[4:]
    assert {c[1] for c in first} == {out / "torsion_1-3-4-5"}
    assert {c[1] for c in second} == {out / "torsion_25-1-3-4"}
    assert {c[3][0] for c in first} == {(1, 3, 4, 5)}
    assert {str(c[2]) for c in first} == {str(PFOA / "pfoa_gmx.top")}
    bond1_refined = out / "torsion_1-3-4-5" / "final" / "molecule_refined.itp"
    assert {str(c[2]) for c in second} == {str(bond1_refined)}
    assert not [c for c in calls if c[0] == "charges"]

    [(bonds, original, final)] = combined
    assert [q for q, _ in bonds] == [(1, 3, 4, 5), (25, 1, 3, 4)]
    assert Path(original) == PFOA / "pfoa_gmx.top"
    last = out / "torsion_25-1-3-4" / "final" / "molecule_refined.itp"
    assert Path(final) == last
    assert (out / "final" / "molecule_refined.itp").read_bytes() == last.read_bytes()

    status = json.loads((out / "final" / "run_status.json").read_text())
    assert status["charges"] == "skipped"
    assert [t["status"] for t in status["torsions"]] == ["complete", "complete"]
    assert [t["dir"] for t in status["torsions"]] == ["torsion_1-3-4-5", "torsion_25-1-3-4"]
    assert status["combined"] == "complete"


def test_flagged_charges_run_first_and_feed_the_first_bond(tmp_path, stages):
    calls, _ = stages
    out = tmp_path / "out"
    assert main(_pfoa(out)) == 0
    assert calls[0][0] == "charges"
    charged = out / "04_charges" / "molecule_charged.itp"
    first_scan = next(c for c in calls if c[0] == "scan")
    assert str(first_scan[2]) == str(charged)


def test_one_flagged_torsion_keeps_todays_layout(tmp_path, stages):
    calls, combined = stages
    out = tmp_path / "out"
    assert main(_cap(out)) == 0
    assert [c[0] for c in calls] == ["scan", "mmscan", "fit", "validate"]  # nothing flagged to charge
    assert {c[1] for c in calls} == {out}
    assert {c[3][0] for c in calls} == {(6, 7, 9, 10)}
    assert combined == []
    status = json.loads((out / "final" / "run_status.json").read_text())
    assert status["torsions"][0]["dir"] == "." and status["combined"] == "not applicable"


def test_a_failing_bond_stops_the_run_and_records_where(tmp_path, stages, monkeypatch, capsys):
    calls, combined = stages
    out = tmp_path / "out"
    fit = cli.cmd_fit

    def failing_fit(a):
        if a.dihedral[0] == (25, 1, 3, 4):
            raise ValidationError("the fit fell over")
        return fit(a)

    monkeypatch.setattr(cli, "cmd_fit", failing_fit)
    assert main(_pfoa(out, "--no-charges")) == 1
    assert "the fit fell over" in capsys.readouterr().err
    status = json.loads((out / "final" / "run_status.json").read_text())
    assert status["torsions"][0]["status"] == "complete"
    assert status["torsions"][1] == {
        "quad_1_based": [25, 1, 3, 4],
        "dir": "torsion_25-1-3-4",
        "status": "failed",
        "failed_stage": "fit",
    }
    assert status["combined"] == "not run" and combined == []
    assert not (out / "final" / "molecule_refined.itp").exists()


def test_no_flagged_torsion_stops_cleanly(tmp_path, stages, capsys):
    calls, _ = stages
    out = tmp_path / "out"
    assert main(_cap(out, "--penalty-threshold", "dihedral=51")) == 0
    assert calls == []
    out_text = capsys.readouterr().out
    assert "no torsion" in out_text and "highest dihedral penalty is 50" in out_text


def test_no_charges_conflicts_with_asking_for_charges(tmp_path, stages, capsys):
    out = tmp_path / "out"
    assert main(_pfoa(out, "--no-charges", "--charge-mode", "all")) == 1
    assert "--no-charges" in capsys.readouterr().err


def test_a_multi_bond_dry_run_prints_the_plan_and_writes_nothing(tmp_path, stages, capsys):
    calls, _ = stages
    out = tmp_path / "out"
    assert main(_pfoa(out, "--dry-run")) == 0
    text = capsys.readouterr().out
    assert "torsion_1-3-4-5" in text and "torsion_25-1-3-4" in text and "23 flagged" in text
    assert calls == [] and not out.exists()


def test_repeated_dihedrals_run_in_the_order_given(tmp_path, stages):
    calls, _ = stages
    out = tmp_path / "out"
    argv = [
        "run", "-s", str(PFOA / "pfoa_gmx.pdb"), "-p", str(PFOA / "pfoa_gmx.top"),
        "--dihedral", "25,1,3,4", "--dihedral", "1,3,4,5", "--outdir", str(out),
    ]
    assert main(argv) == 0
    scans = [c for c in calls if c[0] == "scan"]
    assert [c[1].name for c in scans] == ["torsion_25-1-3-4", "torsion_1-3-4-5"]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_cli_run.py -q'`
Expected: failures such as `unrecognized arguments` gone (flags exist since Task 8) but `run_status.json` missing, stages called with the top-level outdir, `_combined_check` never called.

- [ ] **Step 3: Add `_bond_args` and `_RunStatus` in `ffrefine/cli.py`**

Add `import copy` to the imports. Above `cmd_run`:

```python
def _bond_args(args: argparse.Namespace, quad: Quad, topology: str) -> argparse.Namespace:
    """The arguments one bond's stages run with in a multi-bond run.

    Its own outdir, one ``--dihedral``, and the topology the previous bond
    left (the charge-refined or user's topology for the first). ``--str`` is
    dropped: the bond is already chosen, and its input topology's charges may
    legitimately differ from the ``.str``'s. ``model.json`` still names the
    user's original input (``input_topology``).
    """
    sub = copy.copy(args)
    sub.outdir = str(Path(args.outdir) / _bond_dirname(quad))
    sub.dihedral = [quad]
    sub.topology = topology
    sub.input_topology = _input_topology(args)
    sub.str = None
    sub.selection = None
    return sub


class _RunStatus:
    """``final/run_status.json``, rewritten after every change (plan refinement 5).

    ``report.md`` reads it to say which bond failed at which stage even when
    that stage wrote no payload.
    """

    def __init__(self, args: argparse.Namespace, quads: list[Quad], multi: bool) -> None:
        self.path = Path(args.outdir) / STAGE_DIRS["final"] / "run_status.json"
        self.data: dict = {
            "charges": "not run",
            "torsions": [
                {
                    "quad_1_based": list(q),
                    "dir": _bond_dirname(q) if multi else ".",
                    "status": "not reached",
                    "failed_stage": None,
                }
                for q in quads
            ],
            "combined": "not run" if multi else "not applicable",
        }
        self.write()

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2) + "\n", newline="\n")

    def torsion(self, index: int, status: str, failed_stage: str | None = None) -> None:
        self.data["torsions"][index].update(status=status, failed_stage=failed_stage)
        self.write()

    def set(self, key: str, value: str) -> None:
        self.data[key] = value
        self.write()
```

- [ ] **Step 4: Rewrite `cmd_run`**

Replace `cmd_run` entirely with:

```python
def cmd_run(args: argparse.Namespace) -> int:
    """Run every stage in order: selection, charges, then each torsion (design section 5).

    Charges lead, when they are asked for or ``--str`` flags any: the fitted
    torsion absorbs whatever the 1-4 electrostatics do not account for, so
    refining charges afterwards would leave a torsion fitted against charges
    the topology no longer has. Several torsions run in sequence, one
    subfolder each, every bond fitted against the topology the previous bond
    left; a combined check then re-runs every bond's scan with the final
    topology.
    """
    if args.no_charges and (args.charge_atoms or args.charge_mode == "all"):
        raise ValidationError(
            "--no-charges skips the charge refit, but --charge-atoms / --charge-mode all asks "
            "for one. Drop one of them."
        )
    selection = None
    if args.str:
        if not args.dry_run:
            setup_logging(args.outdir, args.verbose)
        _, topology = _load_structure_and_topology(args)
        selection = _selection(args, topology)
    flagged = selection.flagged_atoms if selection else []
    wants_charges = not args.no_charges and bool(
        args.charge_atoms or args.charge_mode == "all" or flagged
    )
    quads = list(args.dihedral or []) or (selection.quads if selection else [])
    if not quads and selection is None:
        raise ValidationError(
            "no torsion selected. Pass --dihedral i,j,k,l (repeatable), or --str mol.str to "
            "choose from CGenFF's penalties."
        )
    multi = len(quads) > 1
    torsion_stages = [
        ("scan", cmd_scan),
        ("mmscan", cmd_mmscan),
        ("fit", cmd_fit),
        ("validate", cmd_validate),
    ]

    # Two existing tests (test_cli.py::test_run_says_which_stages_it_skipped and
    # test_cli_scan.py) pin "SKIPPING charges"; keep the announcement.
    if wants_charges:
        print(
            "ffrefine run: chaining charges -> scan -> mmscan -> fit -> validate. The torsion "
            "stages use the charge-refined topology, because a torsion fitted against "
            "different charges than it ships with is not transferable."
        )
    else:
        print(
            "ffrefine run: chaining scan -> mmscan -> fit -> validate and SKIPPING charges, "
            "which needs --charge-atoms, --charge-mode all, or a --str that flags atoms. The "
            "topology keeps the charges it came with."
        )

    if args.dry_run:
        if selection is not None:
            print(
                f"--str: {len(flagged)} flagged atom(s) at charge penalty >= "
                f"{selection.thresholds.charge:g}, {len(selection.bonds)} flagged bond(s) at "
                f"dihedral penalty >= {selection.thresholds.dihedral:g}"
            )
        if multi:
            for n, quad in enumerate(quads, start=1):
                print(f"torsion {n}: {','.join(map(str, quad))} in {_bond_dirname(quad)}/")
            print("then a combined check of every bond against the final topology")
            return 0
        if not quads:
            return 0
        args.dihedral = [quads[0]]
        for name, handler in ([("charges", cmd_charges)] if wants_charges else []) + torsion_stages:
            print(f"\n=== {name} ===")
            code = handler(args)
            if code != 0:
                return code
        return 0

    status = _RunStatus(args, quads, multi)
    if wants_charges:
        print("\n=== charges ===")
        try:
            code = cmd_charges(args)
        except FFRefineError:
            status.set("charges", "failed")
            raise
        if code != 0:
            status.set("charges", "failed")
            print("ffrefine run: stopping, stage charges failed", file=sys.stderr)
            return code
        charged = Path(args.outdir) / STAGE_DIRS["charges"] / "molecule_charged.itp"
        if not charged.exists():
            status.set("charges", "failed")
            raise ValidationError(
                f"the charges stage reported success but wrote no {charged}, so the remaining "
                "stages have no refined topology to use."
            )
        status.set("charges", "complete")
        args.input_topology = _input_topology(args)
        args.topology = str(charged)
        print(f"run: the torsion stages will use {charged}")
    else:
        status.set("charges", "skipped")

    if not quads:
        print(
            f"run: no torsion in {args.str} has a dihedral penalty >= "
            f"{selection.thresholds.dihedral:g}; the highest dihedral penalty is "
            f"{selection.highest_dihedral_penalty:g}. Pass --dihedral, or a lower "
            "--penalty-threshold dihedral=VALUE, to refit one."
        )
        return 0

    original = Path(_input_topology(args))
    topology_in = args.topology
    bonds: list[tuple[Quad, Path]] = []
    for index, quad in enumerate(quads):
        stage_args = _bond_args(args, quad, topology_in) if multi else args
        if not multi:
            args.dihedral = [quad]
        status.torsion(index, "running")
        where = f" ({_bond_dirname(quad)})" if multi else ""
        for name, handler in torsion_stages:
            print(f"\n=== {name}{where} ===")
            try:
                code = handler(stage_args)
            except FFRefineError:
                status.torsion(index, "failed", name)
                raise
            if code != 0:
                status.torsion(index, "failed", name)
                print(f"ffrefine run: stopping, stage {name} failed", file=sys.stderr)
                return code
        status.torsion(index, "complete")
        bond_dir = Path(stage_args.outdir)
        bonds.append((quad, bond_dir))
        topology_in = str(bond_dir / STAGE_DIRS["final"] / "molecule_refined.itp")

    if multi:
        print("\n=== combined check ===")
        try:
            _combined_check(args, bonds, original, Path(topology_in))
        except FFRefineError:
            status.set("combined", "failed")
            raise
        status.set("combined", "complete")
        final_topology = Path(args.outdir) / STAGE_DIRS["final"] / "molecule_refined.itp"
        shutil.copyfile(topology_in, final_topology)
        print(f"run: {final_topology} carries every refit")
    return 0
```

- [ ] **Step 5: Run the tests**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_cli_run.py tests/test_cli_charges.py tests/test_cli.py tests/test_cli_typed.py -q -rs && python -m pytest -q -rs'`
Expected: all pass, 0 skipped (the existing single-torsion run tests are unchanged in behaviour).

- [ ] **Step 6: black and commit**

```bash
wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m black ffrefine tests'
cd /c/Users/hrach/Projects/ffrefine
git add ffrefine/cli.py tests/test_cli_run.py
git commit -m "feat(cli): run refits every flagged bond in sequence, one subfolder each

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: `report.py`

**Files:**
- Create: `ffrefine/report.py`
- Create: `tests/test_report.py`

**Interfaces:**
- Consumes (files only): `00_input/model.json`, `00_input/selection.json`, `04_charges/charges.json`, `final/combined.json`, `final/run_status.json`, and per bond `01_qm_scan/scan.json`, `02_mm_scan/energies.json`, `03_fit/fit.json`, `final/validation.json`, with the keys named in Tasks 2, 3, 5, 7, 9 and 10 and the existing ones. Every key is read with `.get`, so an older payload degrades to "not recorded".
- Produces: `ffrefine.report.write_report(outdir: str | Path) -> Path`, writing `<outdir>/final/report.md` (UTF-8) and returning its path.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_report.py`:

```python
"""report.md, built from payload trees on disk (milestone 6b design, section 6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ffrefine.errors import ValidationError
from ffrefine.report import write_report


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _selection(bonds):
    atoms = [
        {"nr": n, "top_name": name, "type": atype, "penalty": pen, "flagged": pen >= 10}
        for n, name, atype, pen in [
            (1, "C1", "CG2O2", 25.0),
            (3, "C3", "CG312", 30.5),
            (4, "C4", "CG312", 28.1),
            (5, "C5", "CG312", 30.8),
            (6, "O6", "OG302", 2.5),
            (7, "C7", "CG2O6", 2.6),
            (9, "N9", "NG2S1", 5.1),
            (10, "C10", "CG2R64", 5.4),
            (25, "O25", "OG311", 10.7),
        ]
    ]
    return {
        "str": "/data/mol.str",
        "cgenff_program_version": "4.0",
        "cgenff_parameter_version": "5.0",
        "resi_param_penalty": 69.0,
        "resi_charge_penalty": 41.068,
        "thresholds": {"charge": 10.0, "dihedral": 50.0},
        "atoms": atoms,
        "dihedral_types": [],
        "other_parameters": [{"kind": "angle", "types": ["CG312", "CG312", "CG312"], "penalty": 55.0}],
        "bonds": [{"quad_1_based": list(q), "penalty": p, "bond_1_based": list(q[1:3])} for q, p in bonds],
        "ring_bonds": [{"bond_1_based": [10, 16], "penalty": 60.0, "quad_1_based": [9, 10, 16, 15]}],
        "unmatched_types": [],
        "warnings": ["a selection warning"],
    }


def _bond(root: Path, quad, rmse=(3.0, 1.5), warnings=None, defaults=True, stages=4):
    warnings = warnings or {}
    _write(
        root / "01_qm_scan" / "scan.json",
        {
            "quad": list(quad),
            "provenance": {
                "gfn": 2,
                "xtb_version": "* xtb version 6.7.1 (edcfbbe)",
                "step_deg": 15.0,
                "npoints": 24,
                "bidirectional": True,
                "charge": 0,
                "uhf": 0,
                "hops": [],
                "hbond_changes": [],
            },
            "warnings": warnings.get("scan", []),
        },
    )
    if stages < 2:
        return
    energies = {
        "quad": list(quad),
        "torsion_parameters": [
            {"quad_1_based": list(quad), "multiplicity": 3, "k_kj_per_mol": 0.5858, "phi0_deg": 180.0, "source": "dihedraltypes", "funct": 9}
        ],
        "selfcheck_max_abs_error_kj": 4.9e-05,
        "selfcheck_tolerance_kj": 0.001,
        "warnings": warnings.get("mmscan", []),
    }
    if defaults:
        energies["defaults"] = {"fudgeQQ": 1.0, "fudgeLJ": 1.0, "gen_pairs": True, "source": "grompp -pp"}
    _write(root / "02_mm_scan" / "energies.json", energies)
    if stages < 3:
        return
    _write(
        root / "03_fit" / "fit.json",
        {
            "provenance": {"quad_1_based": list(quad), "distribute": False},
            "terms": [{"multiplicity": 3, "k_kj_per_mol": 1.25, "k_kcal_per_mol": 0.2988, "phase_deg": 0.0}],
            "rmse_kj_per_mol": 0.4,
            "warnings": warnings.get("fit", []),
        },
    )
    if stages < 4:
        return
    _write(
        root / "final" / "validation.json",
        {
            "quad_1_based": list(quad),
            "rmse_before_kj_per_mol": rmse[0],
            "rmse_after_kj_per_mol": rmse[1],
            "improved": rmse[1] < rmse[0],
            "charges_refined": False,
            "charges_json": None,
            "warnings": warnings.get("validate", []),
        },
    )


def _model(outdir: Path, **extra):
    _write(
        outdir / "00_input" / "model.json",
        {"molecule": "Other", "structure": "/data/mol.pdb", "topology": "/data/mol.top", **extra},
    )


def test_a_single_bond_report_has_every_required_part(tmp_path):
    _model(tmp_path)
    _write(tmp_path / "00_input" / "selection.json", _selection([((6, 7, 9, 10), 50.0)]))
    _bond(tmp_path, (6, 7, 9, 10))
    text = write_report(tmp_path).read_text(encoding="utf-8")
    for heading in ("# ffrefine report: Other", "## Summary", "## Torsion 6-7-9-10", "## Warnings", "## Validity", "## Files"):
        assert heading in text
    assert "O6-C7-N9-C10 (OG302 CG2O6 NG2S1 CG2R64)" in text
    assert "| 50 |" in text
    assert "| 3.000 | 1.500 |" in text
    assert "GFN2-xTB (* xtb version 6.7.1 (edcfbbe))" in text
    assert "CGenFF program 4.0, parameters 5.0" in text
    assert "valid only with the same nonbonded parameters and the same 1-4 conventions" in text
    assert "fudgeQQ = 1, fudgeLJ = 1, gen-pairs = yes (as grompp read them)" in text
    assert "| 3 | 1.2500 | 0.2988 | 0.0 |" in text
    assert "| 6-7-9-10 | 3 | 0.5858 | 180.0 | dihedraltypes |" in text


def test_two_bonds_follow_the_run_status_order_with_the_combined_rmse(tmp_path):
    _model(tmp_path)
    _write(tmp_path / "00_input" / "selection.json", _selection([((1, 3, 4, 5), 69.0), ((25, 1, 3, 4), 67.0)]))
    _bond(tmp_path / "torsion_1-3-4-5", (1, 3, 4, 5))
    _bond(tmp_path / "torsion_25-1-3-4", (25, 1, 3, 4), rmse=(2.0, 1.0))
    _write(
        tmp_path / "final" / "run_status.json",
        {
            "charges": "skipped",
            "torsions": [
                {"quad_1_based": [1, 3, 4, 5], "dir": "torsion_1-3-4-5", "status": "complete", "failed_stage": None},
                {"quad_1_based": [25, 1, 3, 4], "dir": "torsion_25-1-3-4", "status": "complete", "failed_stage": None},
            ],
            "combined": "complete",
        },
    )
    _write(
        tmp_path / "final" / "combined.json",
        {"bonds": [{"quad_1_based": [1, 3, 4, 5], "rmse_final": 1.75}, {"quad_1_based": [25, 1, 3, 4], "rmse_final": 0.9}], "warnings": []},
    )
    text = write_report(tmp_path).read_text(encoding="utf-8")
    assert text.index("## Torsion 1-3-4-5") < text.index("## Torsion 25-1-3-4")
    assert "(`torsion_1-3-4-5/`)" in text
    assert "| 3.000 | 1.500 | 1.750 | complete |" in text
    assert "| 2.000 | 1.000 | 0.900 | complete |" in text


def test_warnings_are_grouped_by_torsion_and_collapsed(tmp_path):
    _model(tmp_path)
    same = "atom name mismatch at atom 3"
    _bond(tmp_path, (6, 7, 9, 10), warnings={"scan": [same], "mmscan": [same, "grompp: note"], "fit": [same]})
    text = write_report(tmp_path).read_text(encoding="utf-8")
    section = text[text.index("## Warnings") : text.index("## Validity")]
    assert section.count(same) == 1
    assert f"(scan, mmscan, fit) {same}" in section
    assert "(mmscan) grompp: note" in section


def test_a_failed_bond_says_where_it_stopped(tmp_path):
    _model(tmp_path)
    _bond(tmp_path / "torsion_1-3-4-5", (1, 3, 4, 5))
    _bond(tmp_path / "torsion_25-1-3-4", (25, 1, 3, 4), stages=2)
    _write(
        tmp_path / "final" / "run_status.json",
        {
            "charges": "skipped",
            "torsions": [
                {"quad_1_based": [1, 3, 4, 5], "dir": "torsion_1-3-4-5", "status": "complete", "failed_stage": None},
                {"quad_1_based": [25, 1, 3, 4], "dir": "torsion_25-1-3-4", "status": "failed", "failed_stage": "fit"},
            ],
            "combined": "not run",
        },
    )
    text = write_report(tmp_path).read_text(encoding="utf-8")
    assert "failed (fit)" in text
    assert "- Fit: not run" in text


def test_without_run_status_the_status_is_inferred(tmp_path):
    _model(tmp_path)
    _bond(tmp_path, (6, 7, 9, 10), stages=1)
    text = write_report(tmp_path).read_text(encoding="utf-8")
    assert "stopped after scan" in text
    assert "- MM energies: not run" in text


def test_a_malformed_payload_is_an_error(tmp_path):
    _model(tmp_path)
    _bond(tmp_path, (6, 7, 9, 10))
    (tmp_path / "03_fit" / "fit.json").write_text("{ not json")
    with pytest.raises(ValidationError, match=r"fit\.json is not valid JSON"):
        write_report(tmp_path)


def test_report_tolerates_payloads_from_an_older_build(tmp_path):
    """Review focus 2: no warnings, torsion_parameters, defaults or charges_json keys."""
    _model(tmp_path, fudgeQQ=None, fudgeLJ=None)
    _bond(tmp_path, (6, 7, 9, 10), defaults=False)
    for rel in ("01_qm_scan/scan.json", "02_mm_scan/energies.json", "03_fit/fit.json", "final/validation.json"):
        path = tmp_path / rel
        data = json.loads(path.read_text())
        for key in ("warnings", "torsion_parameters", "charges_json", "defaults"):
            data.pop(key, None)
        path.write_text(json.dumps(data))
    text = write_report(tmp_path).read_text(encoding="utf-8")
    assert "not recorded" in text
    assert "fudgeQQ and fudgeLJ not recorded" in text


def test_the_charges_section_names_changed_atoms_penalties_and_provenance(tmp_path):
    _model(tmp_path)
    _write(tmp_path / "00_input" / "selection.json", _selection([((6, 7, 9, 10), 50.0)]))
    _write(
        tmp_path / "04_charges" / "charges.json",
        {
            "natoms": 26,
            "target_total": 0,
            "rrms_written": 0.21,
            "rrms_fitted": 0.18,
            "rrms_before": 0.35,
            "atoms": [
                {"nr": 1, "name": "C1", "charge_before": 0.789, "charge_after": 0.701, "source": "fitted"},
                {"nr": 2, "name": "O2", "charge_before": -0.537, "charge_after": -0.537, "source": "unchanged"},
            ],
            "provenance": {
                "scheme": "chelpg",
                "geometry": "xtb",
                "mode": "selected",
                "selected_atoms": list(range(1, 24)),
                "orca_esp_keywords": "! HF 6-31G*",
                "orca_version": "Program Version 6.1.1  -  RELEASE   -",
            },
            "warnings": [],
        },
    )
    _bond(tmp_path, (6, 7, 9, 10))
    text = write_report(tmp_path).read_text(encoding="utf-8")
    assert "## Charges" in text
    assert "| 1 | C1 | +0.7890 | +0.7010 | fitted | 25.000 |" in text
    assert "1 of 26 atoms changed" in text
    assert "Program Version 6.1.1" in text
    assert "**Caveat.**" in text  # 23 of 26 selected: most of the molecule
    assert "fitted against the refined charges recorded in `04_charges/charges.json`" in text


def test_penalised_but_not_refit_lists_rings_and_other_parameters(tmp_path):
    _model(tmp_path)
    _write(tmp_path / "00_input" / "selection.json", _selection([((6, 7, 9, 10), 50.0)]))
    _bond(tmp_path, (6, 7, 9, 10))
    text = write_report(tmp_path).read_text(encoding="utf-8")
    assert "## Penalised by CGenFF but not refit" in text
    assert "Ring bond 10-16 (penalty 60)" in text
    assert "angle CG312 CG312 CG312: penalty 55" in text
```

- [ ] **Step 2: Run them to verify they fail**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_report.py -q'`
Expected: `ModuleNotFoundError: No module named 'ffrefine.report'`.

- [ ] **Step 3: Write `ffrefine/report.py`**

```python
"""``final/report.md``: what a run changed, readable on its own (design section 6).

Built only from the JSON payloads the stages leave on disk -- never from a
run's memory -- so a fresh run, a resumed one and a standalone ``validate``
all produce the same report. A payload that does not exist yet reads "not
run"; one that exists but does not parse is an error. Every key is read with
``.get``: a payload written by an older ffrefine degrades to "not recorded".

Pure: reads files, writes one file, starts no process.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .errors import FFRefineError, ValidationError
from .io.topology import read_topology

NOT_RECORDED = "not recorded"
NA = "n/a"

STAGES: tuple[tuple[str, str], ...] = (
    ("scan", "01_qm_scan/scan.json"),
    ("mmscan", "02_mm_scan/energies.json"),
    ("fit", "03_fit/fit.json"),
    ("validate", "final/validation.json"),
)

VALIDITY = (
    "The fitted torsion parameters are valid only with the same nonbonded parameters and the "
    "same 1-4 conventions as the topology they were fitted against: {conventions}. Change "
    "either and the torsion no longer reproduces the scan it was fitted to."
)

CHARGE_CAVEAT = (
    "Most of the molecule's charges were replaced. CGenFF charges are fitted to water "
    "interaction energies and dipoles, not to the gas-phase electrostatic potential, so "
    "replacing them with ESP charges is a change of derivation scheme and can degrade "
    "condensed-phase behaviour (spec section 5.3)."
)


@dataclass
class _Bond:
    label: str
    root: Path
    quad: list[int] | None
    payloads: dict[str, dict | None]
    status: str
    failed_stage: str | None


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValidationError(
            f"{path} is not valid JSON ({exc}); report.md cannot be written from it"
        ) from exc
    if not isinstance(data, dict):
        raise ValidationError(f"{path} does not hold a JSON object; report.md cannot use it")
    return data


def _num(value, digits: int = 3) -> str:
    return NA if value is None else f"{float(value):.{digits}f}"


def _dash(quad) -> str:
    return "-".join(str(n) for n in quad)


def _dir_order(path: Path):
    try:
        return (0, [int(n) for n in path.name.split("_", 1)[1].split("-")])
    except (IndexError, ValueError):
        return (1, [path.name])


def _quad_of(payloads: dict) -> list[int] | None:
    candidates = (
        (payloads["validate"] or {}).get("quad_1_based"),
        ((payloads["fit"] or {}).get("provenance") or {}).get("quad_1_based"),
        (payloads["mmscan"] or {}).get("quad"),
        (payloads["scan"] or {}).get("quad"),
    )
    return next((list(q) for q in candidates if q), None)


def _infer_status(payloads: dict) -> tuple[str, str | None]:
    if payloads["validate"] is not None:
        return "complete", None
    done = [stage for stage, _ in STAGES if payloads[stage] is not None]
    return ("not run", None) if not done else (f"stopped after {done[-1]}", None)


def _bonds(outdir: Path, status: dict | None) -> list[_Bond]:
    rows: list[tuple] = []
    if status and status.get("torsions"):
        for t in status["torsions"]:
            label = t.get("dir") or "."
            root = outdir if label == "." else outdir / label
            rows.append((label, root, t.get("quad_1_based"), t.get("status"), t.get("failed_stage")))
    else:
        subdirs = sorted(outdir.glob("torsion_*"), key=_dir_order)
        if subdirs:
            rows = [(d.name, d, None, None, None) for d in subdirs]
        elif any((outdir / rel).exists() for _, rel in STAGES):
            rows = [(".", outdir, None, None, None)]
    bonds = []
    for label, root, quad, state, failed in rows:
        payloads = {stage: _load(root / rel) for stage, rel in STAGES}
        if state is None:
            state, failed = _infer_status(payloads)
        bonds.append(_Bond(label, root, list(quad) if quad else _quad_of(payloads), payloads, state, failed))
    return bonds


def _atom_labels(model: dict | None, selection: dict | None) -> dict[int, tuple[str, str]]:
    if selection and selection.get("atoms"):
        return {a["nr"]: (a.get("top_name", ""), a.get("type", "")) for a in selection["atoms"]}
    path = (model or {}).get("topology")
    if not path:
        return {}
    try:
        topology = read_topology(path)
    except FFRefineError:
        return {}
    return {a.nr: (a.name, a.atype) for a in topology.atoms}


def _describe(quad, labels) -> str:
    if quad and all(n in labels for n in quad):
        names = "-".join(labels[n][0] for n in quad)
        types = " ".join(labels[n][1] for n in quad)
        return f"{names} ({types})"
    return NA


def _penalty(quad, selection: dict | None) -> str:
    if not quad or not selection:
        return NA
    for bond in selection.get("bonds", []):
        if list(bond.get("quad_1_based", [])) == list(quad):
            return f"{bond['penalty']:g}"
    return NA


def _status_text(bond: _Bond) -> str:
    return bond.status + (f" ({bond.failed_stage})" if bond.failed_stage else "")


def _rel(path: Path, start: Path) -> str:
    return Path(os.path.relpath(path, start)).as_posix()


def _header(model, selection) -> list[str]:
    lines = [
        f"# ffrefine report: {(model or {}).get('molecule', NA)}",
        "",
        f"Written {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by ffrefine "
        f"{__version__}.",
        "",
    ]
    if model:
        lines.append(f"- Structure: `{model.get('structure', NA)}`")
        lines.append(f"- Topology: `{model.get('topology', NA)}`")
    if selection:
        lines.append(
            f"- CGenFF stream file: `{selection.get('str', NA)}` (CGenFF program "
            f"{selection.get('cgenff_program_version') or NA}, parameters "
            f"{selection.get('cgenff_parameter_version') or NA}; highest parameter penalty "
            f"{_num(selection.get('resi_param_penalty'), 1)}, highest charge penalty "
            f"{_num(selection.get('resi_charge_penalty'), 3)})"
        )
        th = selection.get("thresholds") or {}
        lines.append(
            f"- Penalty thresholds: charge >= {th.get('charge', NA):g}, dihedral >= "
            f"{th.get('dihedral', NA):g}"
            if th
            else f"- Penalty thresholds: {NOT_RECORDED}"
        )
    lines.append("")
    return lines


def _summary(bonds, labels, selection, combined) -> list[str]:
    lines = ["## Summary", ""]
    if not bonds:
        reason = " No dihedral reached the penalty threshold." if selection else ""
        return lines + [f"No torsion was refit.{reason}", ""]
    final_rmse = {
        tuple(row.get("quad_1_based", [])): row.get("rmse_final")
        for row in (combined or {}).get("bonds", [])
    }
    lines += [
        "| Torsion | Atoms (types) | CGenFF penalty | RMSE before (kJ/mol) | RMSE after (kJ/mol) "
        "| RMSE with every change (kJ/mol) | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    for bond in bonds:
        validation = bond.payloads["validate"] or {}
        quad = bond.quad
        lines.append(
            f"| {_dash(quad) if quad else NA} | {_describe(quad, labels)} | "
            f"{_penalty(quad, selection)} | "
            f"{_num(validation.get('rmse_before_kj_per_mol'))} | "
            f"{_num(validation.get('rmse_after_kj_per_mol'))} | "
            f"{_num(final_rmse.get(tuple(quad)) if quad else None)} | {_status_text(bond)} |"
        )
    lines += [
        "",
        "RMSE is MM against QM over the torsion scan, each profile shifted to its own "
        "minimum. \"Before\" is the topology the torsion was fitted against; \"after\" is that "
        "topology with this torsion refit. The next column, filled for runs that refit "
        "several torsions, re-runs the scan with the final topology carrying every change.",
        "",
    ]
    return lines


def _charges_section(charges, selection, status) -> list[str]:
    state = (status or {}).get("charges")
    if not charges:
        if state == "failed":
            return ["## Charges", "", "The charge refit failed; see `04_charges/` and `run.log`.", ""]
        flagged = [a for a in (selection or {}).get("atoms", []) if a.get("flagged")]
        if flagged and state == "skipped":
            return [
                "## Charges",
                "",
                f"Not refined (`--no-charges`), although CGenFF flags {len(flagged)} atom(s).",
                "",
            ]
        return []
    prov = charges.get("provenance") or {}
    lines = [
        "## Charges",
        "",
        f"- Scheme: {prov.get('scheme', NA)}, geometry: {prov.get('geometry', NA)}, mode: "
        f"{prov.get('mode', NA)}",
        f"- ORCA level: `{prov.get('orca_esp_keywords') or NA}` "
        f"({prov.get('orca_version') or NOT_RECORDED})",
        f"- Total charge {charges.get('target_total', NA)}; ESP RRMS of the written charges "
        f"{_num(charges.get('rrms_written'), 4)}, of the fully fitted set "
        f"{_num(charges.get('rrms_fitted'), 4)}, of the input charges "
        f"{_num(charges.get('rrms_before'), 4)}",
        "",
    ]
    penalties = {a["nr"]: a.get("penalty") for a in (selection or {}).get("atoms", [])}
    changed = [a for a in charges.get("atoms", []) if a.get("source") != "unchanged"]
    lines += [
        f"{len(changed)} of {charges.get('natoms', NA)} atoms changed:",
        "",
        "| Atom | Name | Before | After | Why | CGenFF charge penalty |",
        "|---|---|---|---|---|---|",
    ]
    for a in changed:
        lines.append(
            f"| {a['nr']} | {a.get('name', '')} | {a['charge_before']:+.4f} | "
            f"{a['charge_after']:+.4f} | {a.get('source', '')} | {_num(penalties.get(a['nr']))} |"
        )
    natoms = charges.get("natoms") or 0
    selected = prov.get("selected_atoms") or []
    if prov.get("mode") == "all" or (natoms and len(selected) > natoms / 2):
        lines += ["", f"**Caveat.** {CHARGE_CAVEAT}"]
    lines.append("")
    return lines


def _bond_section(bond: _Bond, labels, report_dir: Path) -> list[str]:
    quad = bond.quad
    title = f"## Torsion {_dash(quad) if quad else bond.label}"
    if bond.label != ".":
        title += f" (`{bond.label}/`)"
    lines = [title, "", f"{_describe(quad, labels)}; status: {_status_text(bond)}.", ""]
    scan, energies, fit, validation = (bond.payloads[s] for s, _ in STAGES)

    if scan is None:
        lines.append("- QM scan: not run")
    else:
        prov = scan.get("provenance") or {}
        lines.append(
            f"- QM scan: GFN{prov.get('gfn', '?')}-xTB ({prov.get('xtb_version') or NOT_RECORDED}), "
            f"step {prov.get('step_deg', NA)} degrees, {prov.get('npoints', NA)} points, "
            f"{'bidirectional' if prov.get('bidirectional') else 'one direction'}, total charge "
            f"{prov.get('charge', NA)}, unpaired electrons {prov.get('uhf', NA)}"
        )
        if prov.get("sp_refine_keywords"):
            lines.append(
                f"- Energies refined with ORCA `{prov['sp_refine_keywords']}` "
                f"({prov.get('orca_version') or NOT_RECORDED})"
            )
        hops, hbonds = prov.get("hops") or [], prov.get("hbond_changes") or []
        lines.append(
            f"- Conformational hops: {len(hops)}; hydrogen-bond changes: {len(hbonds)}"
            + (" (see Warnings)" if hops or hbonds else "")
        )

    if energies is None:
        lines.append("- MM energies: not run")
    else:
        lines.append(
            f"- Self-check residual: {energies.get('selfcheck_max_abs_error_kj', NA)} kJ/mol "
            f"(tolerance {energies.get('selfcheck_tolerance_kj', NA)})"
        )
        params = energies.get("torsion_parameters")
        if params is None:
            lines.append(f"- Force-field terms removed about the central bond: {NOT_RECORDED}")
        else:
            lines += [
                "",
                "Force-field terms about the central bond, all zeroed before fitting:",
                "",
                "| Quad | n | k (kJ/mol) | phase (deg) | from |",
                "|---|---|---|---|---|",
            ]
            for p in params:
                lines.append(
                    f"| {_dash(p['quad_1_based'])} | {p['multiplicity']} | "
                    f"{p['k_kj_per_mol']:.4f} | {p['phi0_deg']:.1f} | {p.get('source', NA)} |"
                )
            lines.append("")

    if fit is None:
        lines.append("- Fit: not run")
    else:
        where = (
            "spread over every quadruplet about the bond (--distribute)"
            if (fit.get("provenance") or {}).get("distribute")
            else "on the target quadruplet, the others zeroed"
        )
        lines += [
            "",
            f"Fitted series, {where}; fit RMSE {_num(fit.get('rmse_kj_per_mol'))} kJ/mol:",
            "",
            "| n | k (kJ/mol) | k (kcal/mol) | phase (deg) |",
            "|---|---|---|---|",
        ]
        for t in fit.get("terms", []):
            lines.append(
                f"| {t['multiplicity']} | {t['k_kj_per_mol']:.4f} | "
                f"{t.get('k_kcal_per_mol', 0.0):.4f} | {t['phase_deg']:.1f} |"
            )
        lines.append("")

    if validation is None:
        lines.append("- Validation: not run")
    else:
        verdict = "improved" if validation.get("improved") else "NOT improved"
        lines.append(
            f"- Validation: RMSE against QM {_num(validation.get('rmse_before_kj_per_mol'))} -> "
            f"{_num(validation.get('rmse_after_kj_per_mol'))} kJ/mol ({verdict})"
        )

    plots = sorted((bond.root / "03_fit" / "plots").glob("*.png")) + sorted(
        (bond.root / "final" / "plots").glob("*.png")
    )
    if plots:
        lines.append("- Plots: " + ", ".join(f"[{p.stem}]({_rel(p, report_dir)})" for p in plots))
    lines.append("")
    return lines


def _not_refit(selection) -> list[str]:
    if not selection:
        return []
    threshold = (selection.get("thresholds") or {}).get("dihedral", 50.0)
    rings = selection.get("ring_bonds") or []
    unmatched = selection.get("unmatched_types") or []
    others = [p for p in selection.get("other_parameters") or [] if p["penalty"] >= threshold]
    if not (rings or unmatched or others):
        return []
    lines = ["## Penalised by CGenFF but not refit", ""]
    for r in rings:
        lines.append(
            f"- Ring bond {_dash(r['bond_1_based'])} (penalty {r['penalty']:g}): a relaxed scan "
            "cannot rotate a bond in a ring, so it was not refit."
        )
    for types in unmatched:
        lines.append(f"- Dihedral type {' '.join(types)}: matched no [ dihedrals ] line.")
    for p in others:
        lines.append(
            f"- {p['kind']} {' '.join(p['types'])}: penalty {p['penalty']:g}; ffrefine refits "
            "torsions only."
        )
    lines.append("")
    return lines


def _warnings_section(selection, charges, bonds, combined) -> list[str]:
    sources: list[tuple[str, list[tuple[str, dict]]]] = []
    if selection is not None:
        sources.append(("Selection", [("selection", selection)]))
    if charges is not None:
        sources.append(("Charges", [("charges", charges)]))
    for bond in bonds:
        name = f"Torsion {_dash(bond.quad)}" if bond.quad else f"Torsion {bond.label}"
        sources.append(
            (name, [(stage, bond.payloads[stage]) for stage, _ in STAGES if bond.payloads[stage] is not None])
        )
    if combined is not None:
        sources.append(("Combined check", [("combined", combined)]))

    lines = ["## Warnings", ""]
    written = False
    for title, payloads in sources:
        seen: dict[str, list[str]] = {}
        missing: list[str] = []
        for stage, payload in payloads:
            if "warnings" not in payload:
                missing.append(stage)
                continue
            for message in payload["warnings"]:
                seen.setdefault(message, []).append(stage)
        if not seen and not missing:
            continue
        lines += [f"### {title}", ""]
        for message, stages in seen.items():
            where = ", ".join(
                f"{stage} x{count}" if count > 1 else stage for stage, count in Counter(stages).items()
            )
            lines.append(f"- ({where}) {message}")
        if missing:
            lines.append(
                f"- Warnings {NOT_RECORDED} by: {', '.join(missing)} (written by an older ffrefine)."
            )
        lines.append("")
        written = True
    if not written:
        lines += ["No warnings were raised.", ""]
    return lines


def _validity(bonds, model, charges) -> list[str]:
    defaults = next(
        (
            b.payloads["mmscan"]["defaults"]
            for b in bonds
            if b.payloads["mmscan"] and b.payloads["mmscan"].get("defaults")
        ),
        None,
    )
    if defaults:
        conventions = (
            f"fudgeQQ = {defaults['fudgeQQ']:g}, fudgeLJ = {defaults['fudgeLJ']:g}, gen-pairs = "
            f"{'yes' if defaults.get('gen_pairs') else 'no'} (as grompp read them)"
        )
    elif model and model.get("fudgeQQ") is not None and model.get("fudgeLJ") is not None:
        conventions = (
            f"fudgeQQ = {model['fudgeQQ']:g}, fudgeLJ = {model['fudgeLJ']:g}, gen-pairs = "
            f"{'yes' if model.get('gen_pairs') else 'no'} (from the topology's own [ defaults ])"
        )
    else:
        conventions = (
            f"fudgeQQ and fudgeLJ {NOT_RECORDED}; read your force field's [ defaults ] before "
            "reusing these parameters"
        )
    lines = ["## Validity", "", VALIDITY.format(conventions=conventions)]
    charges_json = next(
        (
            b.payloads["validate"]["charges_json"]
            for b in bonds
            if b.payloads["validate"] and b.payloads["validate"].get("charges_json")
        ),
        None,
    )
    if charges_json or charges is not None:
        source = charges_json or "04_charges/charges.json"
        lines += [
            "",
            f"The torsions were fitted against the refined charges recorded in `{source}`; ship "
            "them together.",
        ]
    lines.append("")
    return lines


def _files(outdir: Path, bonds, report_dir: Path) -> list[str]:
    lines = ["## Files", ""]
    final_topology = outdir / "final" / "molecule_refined.itp"
    if final_topology.exists():
        lines.append(
            f"- Refined topology: `{_rel(final_topology, report_dir)}` -- a molecule include file; "
            "`#include` it from your own `.top` beside the force field it names."
        )
    for bond in bonds:
        prm = bond.root / "03_fit" / "dihedral.prm"
        if prm.exists():
            lines.append(f"- CHARMM parameter block: `{_rel(prm, report_dir)}`")
    for name in ("combined.json", "run_status.json"):
        path = outdir / "final" / name
        if path.exists():
            lines.append(f"- `{_rel(path, report_dir)}`")
    for path in sorted((outdir / "final" / "plots").glob("combined_*.png")):
        lines.append(f"- [{path.stem}]({_rel(path, report_dir)})")
    lines.append(f"- Every command and message: `{_rel(outdir / 'run.log', report_dir)}`")
    lines.append("")
    return lines


def write_report(outdir: str | Path) -> Path:
    """Write ``<outdir>/final/report.md`` from the payloads under ``outdir``.

    ``outdir`` is either a run with one torsion (its stage directories at
    the top level) or a run with several (``torsion_*`` subfolders, ordered
    by ``final/run_status.json`` when present).

    Raises:
        ValidationError: a payload exists but is not a valid JSON object.
    """
    outdir = Path(outdir)
    model = _load(outdir / "00_input" / "model.json")
    selection = _load(outdir / "00_input" / "selection.json")
    charges = _load(outdir / "04_charges" / "charges.json")
    combined = _load(outdir / "final" / "combined.json")
    status = _load(outdir / "final" / "run_status.json")
    bonds = _bonds(outdir, status)
    labels = _atom_labels(model, selection)
    report_dir = outdir / "final"

    lines: list[str] = []
    lines += _header(model, selection)
    lines += _summary(bonds, labels, selection, combined)
    lines += _charges_section(charges, selection, status)
    for bond in bonds:
        lines += _bond_section(bond, labels, report_dir)
    lines += _not_refit(selection)
    lines += _warnings_section(selection, charges, bonds, combined)
    lines += _validity(bonds, model, charges)
    lines += _files(outdir, bonds, report_dir)

    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path
```

- [ ] **Step 4: Run the tests**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_report.py tests/test_layering.py -q'`
Expected: all pass.

- [ ] **Step 5: black and commit**

```bash
wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m black ffrefine/report.py tests/test_report.py'
cd /c/Users/hrach/Projects/ffrefine
git add ffrefine/report.py tests/test_report.py
git commit -m "feat: report.md, a standalone account of what a run changed

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Write the report from `run` and `validate`; live end-to-end runs

**Files:**
- Modify: `ffrefine/cli.py` (`_write_report_quietly`; `cmd_validate` and `cmd_run` call it)
- Create: `tests/test_end_to_end.py`
- Test: `tests/test_cli_run.py` (append)

**Interfaces:**
- Consumes: Task 11's `write_report`; everything before.
- Produces: `final/report.md` at the end of every non-dry `run` (success, failure, or nothing flagged) and every `validate`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli_run.py`:

```python
def test_run_writes_the_report_even_when_a_bond_fails(tmp_path, stages, monkeypatch):
    calls, _ = stages

    def failing(a):
        raise ValidationError("scan fell over")

    monkeypatch.setattr(cli, "cmd_scan", failing)
    out = tmp_path / "out"
    assert main(_pfoa(out, "--no-charges")) == 1
    report = (out / "final" / "report.md").read_text(encoding="utf-8")
    assert "failed (scan)" in report and "not reached" in report


def test_run_with_nothing_flagged_still_writes_a_report(tmp_path, stages):
    out = tmp_path / "out"
    assert main(_cap(out, "--penalty-threshold", "dihedral=51")) == 0
    assert "No torsion was refit" in (out / "final" / "report.md").read_text(encoding="utf-8")
```

Create `tests/test_end_to_end.py`:

```python
"""Milestone 6b end to end, on the real CGenFF fixtures and the real programs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ffrefine.cli import main
from ffrefine.io.topwriter import charges_provenance

FIXTURES = Path(__file__).parent.parent / "fixtures"
PFOA = FIXTURES / "pfoa_gromacs"
CAP = FIXTURES / "capecitabine_gromacs"
PAYLOADS = ("01_qm_scan/scan.json", "02_mm_scan/energies.json", "03_fit/fit.json", "final/validation.json")


@pytest.mark.needs_gmx
@pytest.mark.needs_xtb
def test_capecitabine_run_with_str_refits_its_one_flagged_torsion(tmp_path, gmx_bin, xtb_bin):
    out = tmp_path / "out"
    argv = ["run", "-s", str(CAP / "capecitabine_gmx.pdb"), "-p", str(CAP / "capecitabine_gmx.top")]
    assert main(argv + ["--str", str(CAP / "capecitabine.str"), "--outdir", str(out)]) == 0
    validation = json.loads((out / "final" / "validation.json").read_text())
    assert validation["quad_1_based"] == [6, 7, 9, 10]
    assert not (out / "04_charges").exists()
    for rel in PAYLOADS:
        assert isinstance(json.loads((out / rel).read_text())["warnings"], list), rel
    report = (out / "final" / "report.md").read_text(encoding="utf-8")
    assert "| 6-7-9-10 |" in report and "(OG302 CG2O6 NG2S1 CG2R64)" in report
    assert "| 50 |" in report
    assert "valid only with the same nonbonded parameters" in report


@pytest.mark.needs_gmx
@pytest.mark.needs_xtb
def test_pfoa_run_with_str_refits_both_bonds_in_sequence(tmp_path, gmx_bin, xtb_bin):
    out = tmp_path / "out"
    argv = ["run", "-s", str(PFOA / "pfoa_gmx.pdb"), "-p", str(PFOA / "pfoa_gmx.top")]
    argv += ["--str", str(PFOA / "pfoa.str"), "--no-charges", "--outdir", str(out)]
    assert main(argv) == 0

    status = json.loads((out / "final" / "run_status.json").read_text())
    assert [t["dir"] for t in status["torsions"]] == ["torsion_1-3-4-5", "torsion_25-1-3-4"]
    assert {t["status"] for t in status["torsions"]} == {"complete"}

    bond1 = out / "torsion_1-3-4-5" / "final" / "molecule_refined.itp"
    second = json.loads((out / "torsion_25-1-3-4" / "final" / "validation.json").read_text())
    assert second["before_topology"] == str(bond1.resolve())

    final_lines = (out / "final" / "molecule_refined.itp").read_text().splitlines()

    def fitted(quad):
        head = [str(n) for n in quad]
        return [line for line in final_lines if line.split()[:4] == head and "ffrefine fit" in line]

    assert fitted((1, 3, 4, 5)) and fitted((25, 1, 3, 4))
    combined = json.loads((out / "final" / "combined.json").read_text())
    assert [row["quad_1_based"] for row in combined["bonds"]] == [[1, 3, 4, 5], [25, 1, 3, 4]]
    report = (out / "final" / "report.md").read_text(encoding="utf-8")
    assert "## Torsion 1-3-4-5 (`torsion_1-3-4-5/`)" in report
    assert "## Torsion 25-1-3-4 (`torsion_25-1-3-4/`)" in report
    assert "RMSE with every change" in report


@pytest.mark.needs_orca
def test_pfoa_charges_from_str(tmp_path, orca_bin):
    out = tmp_path / "out"
    argv = ["charges", "-s", str(PFOA / "pfoa_gmx.pdb"), "-p", str(PFOA / "pfoa_gmx.top")]
    argv += ["--str", str(PFOA / "pfoa.str"), "--charge-geom", "input", "--outdir", str(out)]
    assert main(argv) == 0
    charges = json.loads((out / "04_charges" / "charges.json").read_text())
    assert len(charges["provenance"]["selected_atoms"]) == 23
    assert any("close to replacing every charge" in w for w in charges["warnings"])
    assert charges["provenance"]["orca_version"].startswith("Program Version")
    charged = out / "04_charges" / "molecule_charged.itp"
    assert charges_provenance(charged) == (out / "04_charges" / "charges.json").resolve()
    assert (out / "00_input" / "selection.json").exists()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_cli_run.py tests/test_end_to_end.py -q -rs'`
Expected: the two `test_cli_run.py` additions and the two `needs_xtb` end-to-end tests fail on the missing `report.md`; the ORCA test may already pass (it pins the goal).

- [ ] **Step 3: Wire the report in `ffrefine/cli.py`**

Import `from .report import write_report`. Add above `cmd_run`:

```python
def _write_report_quietly(outdir: str | Path) -> None:
    """Write ``final/report.md``; a failure here warns but never masks the run's own outcome."""
    try:
        path = write_report(outdir)
    except FFRefineError as exc:
        _warn(f"report.md could not be written: {exc}")
        return
    print(f"report: wrote {path}")
```

In `cmd_validate`, directly before its final `print(f"validate: wrote ...")`, add `_write_report_quietly(args.outdir)`.

In `cmd_run`, wrap everything from `status = _RunStatus(args, quads, multi)` to the function's final `return 0` in:

```python
    status = _RunStatus(args, quads, multi)
    try:
        ...  # the existing body, unchanged, indented one level
    finally:
        _write_report_quietly(args.outdir)
```

so the report is written whether the run succeeds, stops on a failing stage, raises, or finds nothing flagged.

- [ ] **Step 4: Run the tests, then the whole suite**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_cli_run.py tests/test_end_to_end.py -q -rs && python -m pytest -q -rs'`
Expected: all pass, 0 skipped. The end-to-end tests run real xtb scans (a few minutes each) and a real ORCA ESP job. If a live run fails for a reason this plan does not anticipate, stop and report the stage and its output; do not relax an assertion.

- [ ] **Step 5: black and commit**

```bash
wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m black ffrefine tests'
cd /c/Users/hrach/Projects/ffrefine
git add ffrefine/cli.py tests/test_cli_run.py tests/test_end_to_end.py
git commit -m "feat(cli): run and validate write report.md; CGenFF runs end to end

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: README

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the behaviour Tasks 1-12 built; the final test counts.
- Produces: user-facing documentation only.

- [ ] **Step 1: Status**

Change `## Status: milestones 1-5 and 6a of 6` to `## Status: milestones 1-6 of 6`. In the table, change the `ffrefine run` row to:

```
works -- chains charges and every torsion, charges first when asked for or flagged by `--str`; several torsions run in sequence, one subfolder each, then a combined check; writes `final/report.md`
```

Replace the "Also not implemented: ..." paragraph with:

```
Not implemented: `--mm-relax` (spec 4.6).
```

- [ ] **Step 2: Inputs**

Add, after the `--dihedral` bullet in the Inputs section:

```
- `--str mol.str` (optional): the CGenFF stream file for the molecule. ffrefine
  reads its penalties and chooses the charges and torsions to refine; see
  "Letting CGenFF's penalties choose" below. Atoms are matched to the topology
  by position and checked by atom type, because the CGenFF GROMACS export
  renames them.
```

- [ ] **Step 3: A section on `--str`**

Insert, directly before `## Output layout`, replacing the last paragraph of the PFOA worked example ("Choosing the torsion from `pfoa.str`'s penalties automatically, ... are milestone 6b."):

```markdown
### Letting CGenFF's penalties choose

    ffrefine run \
        -s fixtures/pfoa_gromacs/pfoa_gmx.pdb \
        -p fixtures/pfoa_gromacs/pfoa_gmx.top \
        --str fixtures/pfoa_gromacs/pfoa.str \
        --outdir pfoa_out

With `--str` and no `--dihedral`, ffrefine flags every atom whose CGenFF charge
penalty is at least 10 and every torsion whose dihedral penalty is at least 50
(`--penalty-threshold charge=VALUE` / `dihedral=VALUE` change either; a penalty
equal to the threshold counts). For PFOA that is 23 of 26 atoms and two bonds:
C3-C4 (penalty 69, fitted as `1,3,4,5`) and C1-C3 (67, fitted as `25,1,3,4`).

- The flagged charges are refined first (`--no-charges` skips them). When they
  are most of the molecule, as here, the run warns that this is close to
  replacing CGenFF's charges wholesale.
- Each bond then runs `scan -> mmscan -> fit -> validate` in its own subfolder,
  `torsion_1-3-4-5/` then `torsion_25-1-3-4/`, highest penalty first. Each is
  fitted against the topology the previous one left, so the second bond's fit
  already sees the first bond's refit.
- A combined check re-runs every bond's scan with your original topology and
  with the final one, and records both RMSEs in `final/combined.json`.
- `final/molecule_refined.itp` carries every refit; `final/report.md` says what
  changed, how well it fits, every warning raised, the QM levels used, and the
  1-4 conventions the parameters depend on.

A ring bond is never refit, because a relaxed scan cannot rotate it; the report
lists it. Capecitabine's `.str` flags one torsion at exactly 50 and no charges,
so `ffrefine run ... --str capecitabine.str` refits `6,7,9,10` alone, in the
usual single-torsion layout. A single stage (`scan`, `mmscan`, `fit`,
`validate`) also accepts `--str` when exactly one bond is flagged.
```

- [ ] **Step 4: Flags**

Add, after the "Charge flags" subsection:

```markdown
### CGenFF flags

- `--str PATH`: the CGenFF stream file; its penalties choose the charges
  (`charges`, `run`) and the torsions (every torsion stage) to refine when
  `--charge-atoms` / `--dihedral` are not given. `00_input/selection.json`
  records every penalty and every decision; a copy of the `.str` sits beside it.
- `--penalty-threshold KIND=VALUE`: `KIND` is `charge` (default 10) or
  `dihedral` (default 50); repeatable. A penalty at or above the threshold is
  flagged.
- `--no-charges` (`run` only): refit torsions against the topology's own
  charges even when `--str` flags some.
```

- [ ] **Step 5: Output layout**

Below the existing layout block, add:

```markdown
A run that refits several torsions puts each in its own subfolder, with the
layout above inside it, and keeps the shared pieces at the top:

    pfoa_out/
      00_input/          input copies, model.json, selection.json, the .str copy
      04_charges/        when charges run
      torsion_1-3-4-5/   00_input/ 01_qm_scan/ 02_mm_scan/ 03_fit/ final/ run.log
      torsion_25-1-3-4/  ...
      final/             molecule_refined.itp (every refit), combined.json,
                         plots/combined_*.png, run_status.json, report.md
      run.log            selection, charges and the combined check
```

and change "`final/report.md` is part of the layout but belongs to milestone 6b and is not produced." to "`final/report.md` is written by `run` and by `validate`; every stage's JSON also keeps its own `warnings`, which is where the report's warning list comes from."

- [ ] **Step 6: The `--resume` bullet (milestone 6a ruling R12)**

In the `--resume` bullet, replace "Re-run those three without `--resume` after changing an included file." with:

```
None of them reads an included file -- `scan` and `charges` use only the
molecule's own atoms and charges, and `fit` reruns by itself once `mmscan`'s
energies change -- so nothing more is needed.
```

- [ ] **Step 7: Test counts**

```bash
wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest --collect-only -q | tail -1 && for m in needs_gmx needs_xtb needs_orca; do printf "%s " $m; python -m pytest --collect-only -q -m $m | tail -1; done'
```

Replace the "As of this milestone the suite holds ..." sentence's four numbers with the ones just printed.

- [ ] **Step 8: Final verification, then commit**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest -q -rs && python -m pytest -q -rs -m "needs_gmx or needs_xtb or needs_orca" && python -m black --check ffrefine tests scripts'`
Expected: every test passes, **0 skipped** in both runs, and black is clean.

```bash
cd /c/Users/hrach/Projects/ffrefine
git add README.md
git commit -m "docs(readme): --str, multi-bond runs, report.md; milestone 6 complete

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
