# Milestone 6a — force-field-typed torsions and `#include`d topologies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a real CGenFF `.top` (PFOA, capecitabine) run through `ffrefine run --dihedral ...` end to end: typed torsions resolved from GROMACS's own processed topology, relative `#include`s that survive being copied, and `[ defaults ]` read even when it lives in an include.

**Architecture:** A pure-text `ffrefine/io/includes.py` decides how every written topology spells its includes: absolute in the files grompp reads, verbatim plus an include-base marker in the files the user keeps. `ffrefine/mm/tprparams.py` asks grompp for the answer instead of reimplementing `[ dihedraltypes ]` matching: `grompp -pp` flattens the topology, `gmx dump` prints the parameters in the `.tpr`, and a strict parser and cross-check pair each torsion-set line with the terms grompp actually built. The self-check, `mmscan`, `fit` and `validate` consume the resolved parameters and the new include modes.

**Tech Stack:** Python 3.10+, numpy, pytest, black (line-length 100). GROMACS 2025.4 and xtb 6.7.1 via WSL. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-22-ffrefine-m6a-typed-torsions-design.md`

## Global Constraints

- All user-facing atom indices are **1-based**; internal numpy arrays are 0-based; convert at the boundary and say so in the docstring (spec section 8).
- Type hints throughout, docstrings on public functions, `black` formatting at line-length 100, target py310.
- Every external command goes through `ffrefine.runner.run_command`, which logs the full command line and captures stdout/stderr into the stage directory (spec section 8). Every grompp and gmx invocation lives in `ffrefine/mm/gromacs.py`.
- `ffrefine/io/` never starts a process. `ffrefine/io/includes.py` imports nothing from `ffrefine` except `ffrefine.errors`.
- Input files are never modified. `00_input/` copies stay byte-for-byte copies of what the user passed.
- butane's written topologies must stay byte-identical to what master writes: butane has no `#include`, so it gets no marker and no rewritten line.
- Tests must be run in an **interactive** WSL shell: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest -q -rs'`. A login shell (`bash -lc`) does not source `~/.bashrc`, and a Windows `python` cannot see `gmx` at all, so the `needs_*` tests then skip while reporting green. **Require zero skipped.**
- WSL's `/tmp` is wiped between separate `wsl` invocations. Scratch work that must survive between commands goes under `$HOME/`.
- Commit messages end with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Branch: `ffrefine/milestone-6a-typed-torsions`. Do not work on `master`.

## Refinements of the design, fixed here

Each is a decision the design left to planning, or a detail the design named loosely.

1. **The dump file's name.** The design calls it `resolve_dump.txt`. `run_command` already writes every command's stdout to `<stem>.stdout`, so the dump is `02_mm_scan/full_dump.stdout`, not a second copy of the same text. Messages name that file.
2. **Which writers take an include mode.** The design says every writer takes an explicit mode. Three of the four writers only ever serve one purpose, so their mode is fixed, not a parameter:
   - `write_molecule_copy` is always `absolute`: it builds grompp inputs;
   - `write_zeroed_topology` is always `absolute`: same;
   - `write_charged_topology` is always `verbatim`: it writes a file the user keeps;
   - `write_refit_topology` serves both (`final/after.itp` goes to grompp; `03_fit/dihedral.itp` and `final/molecule_refined.itp` are kept), so it takes a **required** keyword `includes: IncludeMode`. No default, so no call site can pick one by accident.
3. **Design open question 1 (the committed dump).** Resolved: commit the whole dump. Captured from the *trimmed* PFOA fixture it is 45 KB, because the 549 KB of the full-force-field dump is CHARMM's protein CMAP grid, which the trim removes.
4. **Design open question 2 (sharing grompp).** Resolved: `resolve_torsion_set` runs its own grompp. It must run before the `--resume` check, before any rerun, and it costs about a second.
5. **The trim equivalence check compares physics, not type indices.** Measured while planning: on capecitabine the full force field gives `atnr=21`, the trimmed one `atnr=19`. grompp merges atom types whose nonbonded rows are identical, judged against every type in the force field; the full field's NBFIX pairs with types capecitabine never uses (e.g. `NG2S1`-`CLGR1`) keep two of its types apart. So `scripts/compare_tpr_dumps.py` compares each atom's charge and mass, the LJ c6/c12 of every atom pair, and every interaction with its parameters written out. Measured identical: 590 lines on PFOA, 1570 on capecitabine.
6. **`nr:` in a dump block counts integers, not entries.** Each iatoms entry is five integers (type plus four atoms): PFOA's `Proper Dih.` block says `nr: 380` for 76 entries. The design's "nr must equal the number of lines read" is `nr == 5 * entries`.
7. **grompp drops zero-`k` terms whether typed or explicit.** Measured: an explicit `25 1 3 4 9 0.0 0.0 1` line gets no iatoms entry either. So the cross-check expects every `k = 0` term to be absent, typed or explicit.
8. **One pre-existing defect is fixed in Task 3, because 6a makes it matter.** `_rewrite_dihedrals` copies `topology.raw_lines`, which drops every line inside an `#ifdef` branch the parser did not take. Every CGenFF `.top` ends with `#ifdef POSRES_WATER` / `[ position_restraints ]` / `1 1 1000 1000 1000` / `#endif`, so the "drop-in" `molecule_refined.itp` would silently lose the water position restraints. `write_molecule_copy` and `write_charged_topology` already read the source *text* for exactly this reason (their docstrings say so); `_rewrite_dihedrals` is the only reader of `raw_lines` in `ffrefine/`, and it now reads the source text too.

## Facts established while planning (GROMACS 2025.4, trimmed fixtures)

| | PFOA | capecitabine |
|---|---|---|
| target `--dihedral` | `25,1,3,4` (`OG311 CG2O2 CG312 CG312`, penalty 67) | `6,7,9,10` (`OG302 CG2O6 NG2S1 CG2R64`, penalty 50) |
| `[ dihedrals ]` lines about the central bond | 6, all typed | 4, all typed |
| dumped `PDIHS` terms about it | 20 | 3 |
| target quad's terms (phi0, k kJ/mol, n) | (0, 0.66944, 1), (0, 0.46024, 2), (0, 0.33472, 3), (0, 1.046, 6) | (0, 6.6944, 1), (180, 10.46, 2) |
| total `PDIHS` terms | 76 | 140 |
| atoms | 26 | 47 |

On capecitabine, `6 7 9 37` (`OG302 CG2O6 NG2S1 HGP1`) and `8 7 9 37` (`OG2D1 CG2O6 NG2S1 HGP1`) are typed lines whose CGenFF terms all have `k = 0`, so they resolve to **no terms**. That is the design's legitimate-empty case, on real data. The `[ moleculetype ]` name in both `.top` files is `Other`. Both force fields' `[ defaults ]` line is `1 2 yes 1.0 1.0`.

## File structure

| File | Responsibility | Task |
|---|---|---|
| `scripts/trim_forcefield.py` | cut a CHARMM36 force field to one molecule's types | 1 |
| `scripts/compare_tpr_dumps.py` | index-free comparison of two `gmx dump` outputs | 1 |
| `fixtures/pfoa_gromacs/`, `fixtures/capecitabine_gromacs/` | verbatim `.str`/`.top`/`.pdb`, trimmed `charmm36.ff/`, `PROVENANCE.md` | 1 |
| `fixtures/pfoa_gromacs/gromacs/pfoa_dump_2025.4.txt` | captured `gmx dump` for the parser tests | 1 |
| `.gitignore` | `fixtures/_full_ff/` | 1 |
| `ffrefine/io/includes.py` | include rewriting, the marker, `include_base` | 2 |
| `ffrefine/io/topology.py` | `parse_defaults_fields`, `Topology.include_marker`/`include_base` | 2 |
| `ffrefine/io/topwriter.py` | include modes in every writer | 3 |
| `ffrefine/mm/validate_fit.py` | `after.itp` written `absolute` | 3 |
| `ffrefine/cli.py` | fit's `dihedral.itp` `verbatim` | 3 |
| `ffrefine/mm/tprparams.py` | dump parser, `[ defaults ]` parser, cross-check; then `resolve_torsion_set`, `preprocessed_sha256` | 4, 5 |
| `ffrefine/mm/gromacs.py` | `preprocess_topology` (grompp `-pp`), `dump_tpr` | 5 |
| `ffrefine/mm/selfcheck.py` | resolved parameters in; `_reject_typed_terms` out | 6 |
| `tests/conftest.py` | `rotate_side` fixture | 6 |
| `tests/test_cli_typed.py` | CLI on PFOA: include modes, deferred warning, resume keys, end to end | 3, 7, 8 |
| `ffrefine/cli.py` | mmscan resolve step, payload, resume key; deferred defaults warning | 7 |
| `ffrefine/cli.py` | validate resume key, `molecule_refined.itp` written not copied | 8 |
| `README.md` | limits rewritten, PFOA worked example, layout, test counts | 9 |

---

### Task 1: Trimmed PFOA and capecitabine fixtures

**Files:**
- Create: `scripts/trim_forcefield.py`, `scripts/compare_tpr_dumps.py`
- Create: `tests/test_trim_forcefield.py`
- Create: `fixtures/pfoa_gromacs/charmm36.ff/*`, `fixtures/capecitabine_gromacs/charmm36.ff/*` (script output)
- Create: `fixtures/pfoa_gromacs/PROVENANCE.md`, `fixtures/capecitabine_gromacs/PROVENANCE.md`
- Create: `fixtures/pfoa_gromacs/gromacs/pfoa_dump_2025.4.txt`
- Modify: `.gitignore`
- Commit verbatim, unchanged: the `.str`, `_gmx.top` and `_gmx.pdb` of both molecules

**Interfaces:**
- Consumes: nothing from `ffrefine`.
- Produces: the committed fixtures every later task's tests read, and the captured dump Task 4 parses. Later tasks rely on these exact paths: `fixtures/pfoa_gromacs/pfoa_gmx.top`, `fixtures/pfoa_gromacs/pfoa_gmx.pdb`, `fixtures/pfoa_gromacs/charmm36.ff/pfo_ffbonded.itp`, `fixtures/capecitabine_gromacs/capecitabine_gmx.top`, `fixtures/capecitabine_gromacs/capecitabine_gmx.pdb`, `fixtures/pfoa_gromacs/gromacs/pfoa_dump_2025.4.txt`.

**Starting state:** `fixtures/pfoa_gromacs/` and `fixtures/capecitabine_gromacs/` are **untracked** and hold the user's full `charmm36.ff/` (about 14 MB each, identical except for `pfo.rtp`/`pfo_ffbonded.itp` and `cap.rtp`/`cap_ffbonded.itp`). They are the user's only copies in the repo. **Move, never delete.**

- [ ] **Step 1: Move the full force fields aside and ignore them**

```bash
cd /c/Users/hrach/Projects/ffrefine
mkdir -p fixtures/_full_ff/pfoa fixtures/_full_ff/capecitabine
mv fixtures/pfoa_gromacs/charmm36.ff fixtures/_full_ff/pfoa/charmm36.ff
mv fixtures/capecitabine_gromacs/charmm36.ff fixtures/_full_ff/capecitabine/charmm36.ff
printf 'fixtures/_full_ff/\n' >> .gitignore
git status --short
```

Expected: `.gitignore` modified; `fixtures/pfoa_gromacs/` and `fixtures/capecitabine_gromacs/` still untracked; nothing under `fixtures/_full_ff/` listed.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_trim_forcefield.py`:

```python
"""scripts/trim_forcefield.py -- the fixture force-field trimmer (design section 8).

The script is not a package module, so it is loaded from its path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "trim_forcefield.py"
PFOA = ROOT / "fixtures" / "pfoa_gromacs"
CAPECITABINE = ROOT / "fixtures" / "capecitabine_gromacs"


@pytest.fixture(scope="module")
def trim():
    spec = importlib.util.spec_from_file_location("trim_forcefield", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_keeps_entries_whose_types_are_all_used_or_wildcard(trim):
    lines = [
        "[ dihedraltypes ]",
        "; i j k l",
        "X CG321 CG321 X 9 0.0 1.0 3",
        "CG321 CG321 CG321 CG321 9 0.0 1.0 3",
        "CG321 CG321 CG321 HGA2 9 0.0 1.0 3",
    ]
    assert trim.trim_lines(lines, {"CG321"}) == lines[:4]


def test_counts_type_fields_per_section(trim):
    """A pairtypes line has two type fields; its numbers are never read as types."""
    lines = ["[ pairtypes ]", "CG321 CG321 1 0.3 0.2", "CG321 HGA2 1 0.3 0.2"]
    assert trim.trim_lines(lines, {"CG321"}) == lines[:2]


def test_drops_a_continued_cmap_entry_whole(trim):
    lines = [
        "[ cmaptypes ]",
        "",
        "C NH1 CT1 C NH1 1 2 2\\",
        "1.0 2.0\\",
        "3.0 4.0",
        "",
        "CG321 CG321 CG321 CG321 CG321 1 2 2\\",
        "5.0 6.0\\",
        "7.0 8.0",
    ]
    assert trim.trim_lines(lines, {"CG321"}) == ["[ cmaptypes ]", "", "", *lines[6:]]


def test_copies_directives_comments_and_other_sections(trim):
    lines = [
        "[ atomtypes ]",
        "#ifdef HEAVY_H",
        "HT 1 4.032 0.000 A 0.0 0.0",
        "#else",
        "HT 1 1.008 0.000 A 0.0 0.0",
        "#endif",
        "; a comment",
        "[ moleculetype ]",
        "SOL 2",
    ]
    assert trim.trim_lines(lines, set()) == [lines[0], lines[1], lines[3], *lines[5:]]


def test_used_types_include_water_and_ions(trim):
    used = trim.used_types(
        PFOA / "pfoa_gmx.top", PFOA / "charmm36.ff", "charmm36.ff"
    )
    assert {"CG2O2", "OG311", "FGA2", "HGP1"} <= used
    assert {"OT", "HT", "SOD", "CLA"} <= used


@pytest.mark.parametrize("mol, top", [(PFOA, "pfoa_gmx.top"), (CAPECITABINE, "capecitabine_gmx.top")])
def test_trimming_a_committed_fixture_changes_nothing(trim, tmp_path, mol, top):
    """The committed trees are this script's output, and trimming is idempotent."""
    out = tmp_path / "charmm36.ff"
    trim.trim_forcefield(mol / "charmm36.ff", mol / top, out)
    committed = sorted(p.name for p in (mol / "charmm36.ff").iterdir())
    assert sorted(p.name for p in out.iterdir()) == committed
    for name in committed:
        assert (out / name).read_bytes() == (mol / "charmm36.ff" / name).read_bytes(), name
```

- [ ] **Step 3: Run them to verify they fail**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_trim_forcefield.py -q'`
Expected: every test errors, with `FileNotFoundError` for `scripts/trim_forcefield.py`.

- [ ] **Step 4: Write `scripts/trim_forcefield.py`**

This exact code was run on both molecules while planning.

```python
"""Trim a CHARMM36 GROMACS force field to what one molecule's .top needs.

Builds the committed fixtures under ``fixtures/<mol>_gromacs/charmm36.ff/``
(milestone 6a design, section 8). An entry in a type-keyed section is kept only
when every atom-type field names a type that occurs in some ``[ atoms ]`` the
``.top`` pulls in, or is the wildcard ``X``. An entry naming any other type can
match none of this molecule's interactions, so dropping it cannot change which
entry grompp picks. ``scripts/compare_tpr_dumps.py`` checks that claim.

Usage:
    python scripts/trim_forcefield.py FULL_FF_DIR MOLECULE_TOP OUT_FF_DIR
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path, PurePosixPath

PARAMETER_FILES = (
    "ffnonbonded.itp",
    "ffbonded.itp",
    "gb.itp",
    "cmap.itp",
    "old_c36_cmap.itp",
    "nbfix.itp",
    "silcs.itp",
)
"""Every file charmm36.ff/forcefield.itp can include, in either #ifdef branch."""

VERBATIM_FILES = frozenset({"forcefield.itp", "tip3p.itp", "ions.itp"})
"""Copied byte for byte: they hold no type-keyed parameter tables."""

TYPE_FIELDS = {
    "atomtypes": 1,
    "bondtypes": 2,
    "constrainttypes": 2,
    "angletypes": 3,
    "dihedraltypes": 4,
    "pairtypes": 2,
    "nonbond_params": 2,
    "cmaptypes": 5,
    "implicit_genborn_params": 1,
}
"""Leading atom-type fields of each type-keyed section."""

WILDCARD = "X"

_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"')
_SECTION_RE = re.compile(r"^\[\s*([A-Za-z_0-9]+)\s*\]")


def include_names(path: Path, ff_dirname: str) -> list[str]:
    """Names of the files ``path`` includes from a directory called ``ff_dirname``.

    ``ff_dirname=""`` selects bare names, which is how ``forcefield.itp``
    includes its siblings.
    """
    names: list[str] = []
    for raw in path.read_text().splitlines():
        m = _INCLUDE_RE.match(raw)
        if m:
            p = PurePosixPath(m.group(1))
            if p.parent.name == ff_dirname and p.name not in names:
                names.append(p.name)
    return names


def atom_types_in(path: Path) -> set[str]:
    """Atom types named in every ``[ atoms ]`` section of one file."""
    types: set[str] = set()
    section = ""
    for raw in path.read_text().splitlines():
        body = raw.split(";", 1)[0].strip()
        if not body or body.startswith("#"):
            continue
        m = _SECTION_RE.match(body)
        if m:
            section = m.group(1).lower()
            continue
        fields = body.split()
        if section == "atoms" and len(fields) >= 2 and fields[0].isdigit():
            types.add(fields[1])
    return types


def used_types(top: Path, full_ff: Path, ff_dirname: str) -> set[str]:
    """Types of the molecule and of every moleculetype the .top includes.

    grompp resolves the atom types of TIP3P and the ions while parsing them,
    even though ``[ molecules ]`` never lists them, so their types must survive.
    The includes are looked up in ``full_ff``, not beside the .top: while the
    fixture is being built, the .top's own ``charmm36.ff/`` is the output.
    """
    types = atom_types_in(top)
    for name in include_names(top, ff_dirname):
        types |= atom_types_in(full_ff / name)
    return types


def trim_lines(lines: list[str], used: set[str]) -> list[str]:
    """Drop type-keyed entries naming a type outside ``used``; copy the rest.

    A ``[ cmaptypes ]`` entry continues over lines ending in a backslash; it
    is kept or dropped whole.
    """
    out: list[str] = []
    section = ""
    in_entry = False
    keep_entry = True
    for raw in lines:
        if in_entry:
            if keep_entry:
                out.append(raw)
            in_entry = raw.rstrip().endswith("\\")
            continue
        body = raw.split(";", 1)[0].strip()
        m = _SECTION_RE.match(body)
        if m:
            section = m.group(1).lower()
            out.append(raw)
            continue
        if not body or body.startswith("#") or section not in TYPE_FIELDS:
            out.append(raw)
            continue
        n = TYPE_FIELDS[section]
        fields = body.split()
        keep_entry = len(fields) >= n and all(f in used or f == WILDCARD for f in fields[:n])
        if keep_entry:
            out.append(raw)
        in_entry = raw.rstrip().endswith("\\")
    return out


def trim_forcefield(full_ff: Path, top: Path, out_ff: Path) -> dict[str, tuple[int, int]]:
    """Write the trimmed force field; return ``{file: (lines_in, lines_out)}``."""
    if out_ff.resolve() == full_ff.resolve():
        raise SystemExit(f"refusing to trim {full_ff} in place")
    declared = include_names(full_ff / "forcefield.itp", "")
    unknown = [n for n in declared if n not in PARAMETER_FILES]
    if unknown:
        raise SystemExit(
            f"{full_ff / 'forcefield.itp'} includes {unknown}, which this script does not "
            "know how to trim; extend PARAMETER_FILES"
        )
    ff_dirname = out_ff.name
    used = used_types(top, full_ff, ff_dirname)
    names = ["forcefield.itp", *PARAMETER_FILES]
    names += [n for n in include_names(top, ff_dirname) if n not in names]
    out_ff.mkdir(parents=True, exist_ok=True)
    report: dict[str, tuple[int, int]] = {}
    for name in names:
        src = full_ff / name
        if not src.is_file():
            raise SystemExit(f"{src} is missing")
        lines = src.read_text().splitlines()
        if name in VERBATIM_FILES:
            shutil.copyfile(src, out_ff / name)
            report[name] = (len(lines), len(lines))
            continue
        kept = trim_lines(lines, used)
        (out_ff / name).write_text("\n".join(kept) + "\n", newline="\n")
        report[name] = (len(lines), len(kept))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("full_ff", type=Path)
    parser.add_argument("top", type=Path)
    parser.add_argument("out_ff", type=Path)
    args = parser.parse_args(argv)
    report = trim_forcefield(args.full_ff, args.top, args.out_ff)
    for name, (before, after) in report.items():
        print(f"{name:24s} {before:8d} -> {after:6d} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Write `scripts/compare_tpr_dumps.py`**

Also run while planning: identical on both molecules, and a one-digit change to one force constant produces a diff.

```python
"""Compare what two ``gmx dump -s`` outputs make the molecule feel.

Used once per fixture to show that a trimmed force field (see
``scripts/trim_forcefield.py``) gives grompp exactly the physics the full one
does. The comparison is per atom and per interaction, never per type index:
grompp merges atom types whose nonbonded rows are identical, judged against
every type in the force field, so dropping types the molecule never uses can
legitimately renumber (and merge) the molecule's own types without changing a
single parameter.

Compared, for every moltype: each atom's mass, charge, particle type and
element; the Lennard-Jones c6/c12 of every atom pair; every interaction list
entry with its parameters written out; plus ``fudgeQQ`` and ``reppow``.

Usage:
    python scripts/compare_tpr_dumps.py FULL_DUMP TRIMMED_DUMP
Exit status 0 when identical, 1 with a unified diff otherwise.
"""

from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

_FUNCTYPE_RE = re.compile(r"^\s*functype\[(\d+)\]=(\w+),\s*(.*?)\s*$")
_ATNR_RE = re.compile(r"^\s*atnr=(\d+)\s*$")
_MOLTYPE_RE = re.compile(r"^\s*moltype \((\d+)\):\s*$")
_ATOM_RE = re.compile(
    r"^\s*atom\[\s*(\d+)\]=\{type=\s*(\d+),.*?ptype=\s*(\w+), m=\s*(\S+), q=\s*(\S+),"
    r".*atomnumber=\s*(-?\d+)\}"
)
_HEADING_RE = re.compile(r"^ {6}(\S.*):\s*$")
_IATOM_RE = re.compile(r"^\s*\d+ type=(\d+) \((\w+)\)\s+([\d\s]+)$")
_SCALAR_RE = re.compile(r"^\s*(fudgeQQ|reppow)\s*=\s*(\S+)")


def canonical(text: str) -> list[str]:
    """Index-free description of every moltype's parameters."""
    lines = text.splitlines()
    functypes: dict[int, tuple[str, str]] = {}
    atnr = None
    out: list[str] = []
    for line in lines:
        m = _FUNCTYPE_RE.match(line)
        if m:
            functypes[int(m.group(1))] = (m.group(2), m.group(3))
            continue
        m = _ATNR_RE.match(line)
        if m:
            atnr = int(m.group(1))
            continue
        m = _SCALAR_RE.match(line)
        if m:
            out.append(f"{m.group(1)} = {m.group(2)}")
    if atnr is None:
        raise SystemExit("no atnr= line: not a gmx dump -s output")

    moltype = None
    atom_types: list[int] = []
    heading = ""
    for line in lines:
        m = _MOLTYPE_RE.match(line)
        if m:
            if moltype is not None:
                out += _pair_lines(moltype, atom_types, atnr, functypes)
            moltype, atom_types, heading = m.group(1), [], ""
            out.append(f"moltype {moltype}")
            continue
        if moltype is None:
            continue
        m = _ATOM_RE.match(line)
        if m and heading in ("", "atoms"):
            atom_types.append(int(m.group(2)))
            out.append(
                f"atom {m.group(1)} ptype={m.group(3)} m={m.group(4)} q={m.group(5)} "
                f"element={m.group(6)}"
            )
            continue
        m = _HEADING_RE.match(line)
        if m:
            heading = m.group(1)
            continue
        m = _IATOM_RE.match(line)
        if m:
            name, params = functypes[int(m.group(1))]
            out.append(f"{heading}: {m.group(3).split()} {name} {params}")
    if moltype is not None:
        out += _pair_lines(moltype, atom_types, atnr, functypes)
    return out


def _pair_lines(
    moltype: str, atom_types: list[int], atnr: int, functypes: dict[int, tuple[str, str]]
) -> list[str]:
    """LJ parameters of every atom pair, looked up through the type matrix."""
    out = []
    for i, ti in enumerate(atom_types):
        for j in range(i, len(atom_types)):
            name, params = functypes[ti * atnr + atom_types[j]]
            out.append(f"moltype {moltype} pair {i} {j} {name} {params}")
    return out


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    full, trimmed = (canonical(Path(p).read_text()) for p in argv)
    if full == trimmed:
        print(f"identical: {len(full)} per-atom and per-interaction lines compared")
        return 0
    sys.stdout.writelines(
        line + "\n" for line in difflib.unified_diff(full, trimmed, argv[0], argv[1], lineterm="")
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Trim both force fields**

```bash
wsl -e bash -ic 'cd /mnt/c/Users/hrach/Projects/ffrefine && conda activate gromacs && \
  python scripts/trim_forcefield.py fixtures/_full_ff/pfoa/charmm36.ff fixtures/pfoa_gromacs/pfoa_gmx.top fixtures/pfoa_gromacs/charmm36.ff && \
  python scripts/trim_forcefield.py fixtures/_full_ff/capecitabine/charmm36.ff fixtures/capecitabine_gromacs/capecitabine_gmx.top fixtures/capecitabine_gromacs/charmm36.ff'
```

Expected, PFOA: `ffnonbonded.itp 42461 -> 74`, `ffbonded.itp 47881 -> 53`, `cmap.itp 1445 -> 29`, and `pfo_ffbonded.itp`, `tip3p.itp` and `ions.itp` unchanged (55, 49 and 84 lines). Capecitabine: `ffnonbonded.itp 42461 -> 238`, `ffbonded.itp 47881 -> 823`. Each `charmm36.ff/` is now under 100 KB (`du -sh`).

- [ ] **Step 7: Prove the trim is physics-neutral**

Build a tpr from the full and the trimmed tree of each molecule, and compare the dumps:

```bash
wsl -e bash -ic 'conda activate gromacs && set -e && R=/mnt/c/Users/hrach/Projects/ffrefine && W=$HOME/m6a_equiv && rm -rf $W && mkdir -p $W && \
for m in pfoa capecitabine; do
  for v in full trim; do
    mkdir -p $W/$m/$v && cp $R/fixtures/${m}_gromacs/${m}_gmx.top $R/fixtures/${m}_gromacs/${m}_gmx.pdb $W/$m/$v/
  done
  cp -r $R/fixtures/_full_ff/$m/charmm36.ff $W/$m/full/
  cp -r $R/fixtures/${m}_gromacs/charmm36.ff $W/$m/trim/
  for v in full trim; do
    cd $W/$m/$v
    printf "integrator=md\nnsteps=0\ncutoff-scheme=Verlet\npbc=xyz\n" > t.mdp
    gmx editconf -f ${m}_gmx.pdb -o box.gro -box 6 6 6 -c > editconf.log 2>&1
    gmx grompp -f t.mdp -c box.gro -p ${m}_gmx.top -o t.tpr -maxwarn 5 > grompp.log 2>&1
    gmx dump -s t.tpr > dump.txt 2> dump.err
  done
  python $R/scripts/compare_tpr_dumps.py $W/$m/full/dump.txt $W/$m/trim/dump.txt
done'
```

Expected: `identical: 590 per-atom and per-interaction lines compared` (PFOA), then `identical: 1570 per-atom and per-interaction lines compared` (capecitabine). **If either prints a diff, stop and report it; do not commit the fixtures.**

- [ ] **Step 8: Capture the PFOA dump for the parser tests**

```bash
wsl -e bash -ic 'mkdir -p /mnt/c/Users/hrach/Projects/ffrefine/fixtures/pfoa_gromacs/gromacs && cp $HOME/m6a_equiv/pfoa/trim/dump.txt /mnt/c/Users/hrach/Projects/ffrefine/fixtures/pfoa_gromacs/gromacs/pfoa_dump_2025.4.txt && head -1 $HOME/m6a_equiv/pfoa/trim/dump.err && wc -c /mnt/c/Users/hrach/Projects/ffrefine/fixtures/pfoa_gromacs/gromacs/pfoa_dump_2025.4.txt'
```

Expected: `:-) GROMACS - gmx dump, 2025.4-conda_forge (-:` and a size of about 45 000 bytes.

- [ ] **Step 9: Write the provenance notes**

Create `fixtures/pfoa_gromacs/PROVENANCE.md`:

```markdown
# PFOA fixture

Perfluorooctanoic acid, neutral (COOH), as the CGenFF server returned it.
Supplied by the project owner, 2026-09-22.

| File | Origin |
|---|---|
| `pfoa.str` | CGenFF program 4.0, parameters 5.0; verbatim |
| `pfoa_gmx.top`, `pfoa_gmx.pdb` | the CGenFF server's GROMACS export (generated 2026-05-27); verbatim |
| `charmm36.ff/` | CHARMM36 November 2018 GROMACS port (`forcefield.doc`) plus the export's `pfo_ffbonded.itp`, **trimmed** |
| `gromacs/pfoa_dump_2025.4.txt` | `gmx dump -s` of a tpr built from this directory with GROMACS 2025.4 |

## The trim

    python scripts/trim_forcefield.py <full charmm36.ff> fixtures/pfoa_gromacs/pfoa_gmx.top fixtures/pfoa_gromacs/charmm36.ff

keeps every file grompp opens and, in each type-keyed section, only the entries whose atom
types all occur in the molecule, TIP3P or the ions (or are `X`). The pdb2gmx-only files are
not kept. `tests/test_trim_forcefield.py` checks that trimming this tree again changes nothing.

Equivalence, checked 2026-09-22 with GROMACS 2025.4: tprs built from the full and the trimmed
force field, compared with `scripts/compare_tpr_dumps.py` (per atom, per atom pair and per
interaction, never per type index): **identical, 590 lines**.

## Notes for milestone 6b

- The export renamed the atoms: `.str` `O1` is `.top` atom 2 (`O2`), `.str` `C2` is atom 3
  (`C3`), and so on. Order, types and charges agree index for index.
- Dihedral penalties above 50 sit on the `.str` bonds C1-C2 and C2-C3, i.e. `.top` atoms 1-3
  and 3-4. 23 of the 26 atoms carry a charge penalty of 10 or more.
```

Create `fixtures/capecitabine_gromacs/PROVENANCE.md`:

```markdown
# Capecitabine fixture

Capecitabine, as the CGenFF server returned it. Supplied by the project owner, 2026-09-22.

| File | Origin |
|---|---|
| `capecitabine.str` | CGenFF program 4.0, parameters 5.0; verbatim |
| `capecitabine_gmx.top`, `capecitabine_gmx.pdb` | the CGenFF server's GROMACS export (generated 2026-06-01); verbatim |
| `charmm36.ff/` | CHARMM36 November 2018 GROMACS port (`forcefield.doc`) plus the export's `cap_ffbonded.itp`, **trimmed** |

## The trim

    python scripts/trim_forcefield.py <full charmm36.ff> fixtures/capecitabine_gromacs/capecitabine_gmx.top fixtures/capecitabine_gromacs/charmm36.ff

Equivalence, checked 2026-09-22 with GROMACS 2025.4, with `scripts/compare_tpr_dumps.py`:
**identical, 1570 lines**. grompp numbers the atom types differently (`atnr` 21 with the full
force field, 19 trimmed): it merges types whose nonbonded rows are identical, judged against
every type present, and the full field's NBFIX pairs with types this molecule never uses keep
two of its types apart. No parameter any atom or interaction sees changes, which is what the
comparison checks.

## Notes for milestone 6b

- The only dihedral at the default penalty threshold is `OG302 CG2O6 NG2S1 CG2R64`, at
  exactly 50: `.top` quad `6 7 9 10`. Every charge penalty is below 10.
- `6 7 9 37` and `8 7 9 37` resolve to CGenFF terms that all have `k = 0`; grompp builds no
  term for them.
```

- [ ] **Step 10: Run the trim tests and black**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_trim_forcefield.py -q && python -m black --check scripts tests/test_trim_forcefield.py'`
Expected: 7 passed; black reports nothing to reformat.

- [ ] **Step 11: Commit**

```bash
cd /c/Users/hrach/Projects/ffrefine
git add .gitignore scripts/trim_forcefield.py scripts/compare_tpr_dumps.py tests/test_trim_forcefield.py fixtures/pfoa_gromacs fixtures/capecitabine_gromacs
git status --short   # nothing under fixtures/_full_ff/ may appear
git commit -m "test(fixtures): PFOA and capecitabine CGenFF fixtures with a trimmed CHARMM36

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: `io/includes.py` and the topology's include base

**Files:**
- Create: `ffrefine/io/includes.py`
- Modify: `ffrefine/io/topology.py` (add `parse_defaults_fields`; `Topology.include_marker`, `Topology.include_base`; use both in `read_topology`)
- Test: `tests/test_includes.py` (new), `tests/test_topology.py` (append)

**Interfaces:**
- Consumes: `ffrefine.errors.ParseError`, `ffrefine.errors.ValidationError`.
- Produces:
  - `ffrefine.io.includes.IncludeMode = Literal["absolute", "verbatim"]`
  - `MARKER_NOTE: str`, `MARKER_PREFIX: str`
  - `has_includes(lines: list[str]) -> bool`
  - `find_marker(lines: list[str], source: str | Path) -> Path | None`, raising `ParseError` on two different markers
  - `resolve_include_base(source: str | Path, marker: Path | None) -> Path`, raising `ValidationError` when the marked directory is missing
  - `include_base(path: str | Path) -> Path`
  - `strip_marker(lines: list[str]) -> list[str]`
  - `absolutise_includes(lines: list[str], base: Path) -> list[str]`
  - `mark_include_base(lines: list[str], base: Path) -> list[str]`
  - `apply_include_mode(lines: list[str], base: Path, mode: IncludeMode) -> list[str]`
  - `ffrefine.io.topology.parse_defaults_fields(fields: list[str]) -> Defaults`
  - `Topology.include_marker: Path | None` (new last field, default `None`); `Topology.include_base -> Path` (property; may raise `ValidationError`)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_includes.py`:

```python
"""Include handling for the topologies ffrefine writes (milestone 6a design, section 5)."""

from __future__ import annotations

from pathlib import Path

import pytest
from ffrefine.errors import ParseError, ValidationError
from ffrefine.io import includes
from ffrefine.io.includes import MARKER_NOTE, MARKER_PREFIX

LINES = [
    "; molecule",
    '#include "./charmm36.ff/forcefield.itp"',
    '# include "./charmm36.ff/mol_ffbonded.itp"',
    '#include "posre.itp"',
    "#include <gmxlib.itp>",
    "[ moleculetype ]",
]


def _user_dir(tmp_path: Path) -> Path:
    """A user directory holding a force-field folder the LINES include."""
    user = tmp_path / "user"
    (user / "charmm36.ff").mkdir(parents=True)
    (user / "charmm36.ff" / "forcefield.itp").write_text("[ defaults ]\n1 2 yes 1.0 1.0\n")
    (user / "charmm36.ff" / "mol_ffbonded.itp").write_text("")
    return user


def test_absolutise_rewrites_only_relative_includes_that_exist(tmp_path):
    user = _user_dir(tmp_path)
    ff = (user / "charmm36.ff").resolve()
    out = includes.absolutise_includes(LINES, user)
    assert out[1] == f'#include "{ff / "forcefield.itp"}"'
    assert out[2] == f'# include "{ff / "mol_ffbonded.itp"}"'
    assert out[3] == '#include "posre.itp"'  # not under the base: grompp searches for it
    assert out[4] == "#include <gmxlib.itp>"  # angle brackets are never rewritten
    assert out[0] == LINES[0] and out[5] == LINES[5]


def test_absolutise_leaves_an_absolute_include_alone(tmp_path):
    user = _user_dir(tmp_path)
    line = f'#include "{(user / "charmm36.ff" / "forcefield.itp").resolve()}"'
    assert includes.absolutise_includes([line], tmp_path / "elsewhere") == [line]


def test_absolutise_keeps_a_trailing_comment(tmp_path):
    user = _user_dir(tmp_path)
    out = includes.absolutise_includes(['#include "./charmm36.ff/forcefield.itp" ; ff'], user)
    assert out[0].endswith('forcefield.itp" ; ff')


def test_mark_prepends_the_base_when_a_relative_include_resolved(tmp_path):
    user = _user_dir(tmp_path)
    out = includes.mark_include_base(LINES, user)
    assert out[:2] == [MARKER_NOTE, f"{MARKER_PREFIX}{user.resolve()}"]
    assert out[2:] == LINES  # the include lines themselves are untouched


def test_mark_writes_nothing_without_a_resolvable_relative_include(tmp_path):
    lines = ["[ moleculetype ]", '#include "posre.itp"']
    assert includes.mark_include_base(lines, tmp_path) == lines


def test_mark_replaces_an_existing_marker_instead_of_adding_a_second(tmp_path):
    user = _user_dir(tmp_path)
    once = includes.mark_include_base(LINES, user)
    assert includes.mark_include_base(once, user) == once


def test_absolutise_strips_a_marker(tmp_path):
    user = _user_dir(tmp_path)
    out = includes.absolutise_includes(includes.mark_include_base(LINES, user), user)
    assert not any(line == MARKER_NOTE or line.startswith(MARKER_PREFIX) for line in out)


def test_apply_include_mode_dispatches(tmp_path):
    user = _user_dir(tmp_path)
    assert includes.apply_include_mode(LINES, user, "absolute") == includes.absolutise_includes(
        LINES, user
    )
    assert includes.apply_include_mode(LINES, user, "verbatim") == includes.mark_include_base(
        LINES, user
    )
    with pytest.raises(ValueError):
        includes.apply_include_mode(LINES, user, "relative")


def test_include_base_is_the_file_directory_without_a_marker(tmp_path):
    user = _user_dir(tmp_path)
    top = user / "mol.top"
    top.write_text("\n".join(LINES) + "\n")
    assert includes.include_base(top) == user.resolve()


def test_include_base_follows_the_marker(tmp_path):
    user = _user_dir(tmp_path)
    moved = tmp_path / "out" / "04_charges"
    moved.mkdir(parents=True)
    copy = moved / "molecule_charged.itp"
    copy.write_text("\n".join(includes.mark_include_base(LINES, user)) + "\n")
    assert includes.include_base(copy) == user.resolve()


def test_a_marker_naming_a_missing_directory_is_refused(tmp_path):
    copy = tmp_path / "x.itp"
    copy.write_text(f"{MARKER_PREFIX}{tmp_path / 'gone'}\n[ moleculetype ]\n")
    with pytest.raises(ValidationError, match="does not exist"):
        includes.include_base(copy)


def test_two_different_markers_are_refused(tmp_path):
    copy = tmp_path / "x.itp"
    copy.write_text(f"{MARKER_PREFIX}/a\n{MARKER_PREFIX}/b\n")
    with pytest.raises(ParseError, match="two ffrefine include-base markers"):
        includes.include_base(copy)


def test_has_includes():
    assert includes.has_includes(['# include "a.itp"'])
    assert not includes.has_includes(["; #include in a comment", "[ atoms ]"])
```

Append to `tests/test_topology.py` (it already imports `read_topology`; add the other names to its imports):

```python
from ffrefine.io.includes import MARKER_NOTE, MARKER_PREFIX
from ffrefine.io.topology import Defaults, parse_defaults_fields


def test_topology_include_base_defaults_to_its_directory(butane_itp):
    assert read_topology(butane_itp).include_base == butane_itp.resolve().parent


def test_topology_include_base_reads_the_marker(tmp_path, butane_itp):
    base = butane_itp.resolve().parent
    copy = tmp_path / "copy.itp"
    copy.write_text(f"{MARKER_NOTE}\n{MARKER_PREFIX}{base}\n" + butane_itp.read_text())
    t = read_topology(copy)
    assert t.include_marker == base
    assert t.include_base == base


def test_parse_defaults_fields():
    assert parse_defaults_fields(["1", "2", "yes", "0.5", "0.8333"]) == Defaults(
        nbfunc=1, comb_rule=2, gen_pairs=True, fudge_lj=0.5, fudge_qq=0.8333
    )
    assert parse_defaults_fields(["1", "1"]) == Defaults(1, 1, False, 1.0, 1.0)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_includes.py tests/test_topology.py -q'`
Expected: collection errors, `ModuleNotFoundError: No module named 'ffrefine.io.includes'`.

- [ ] **Step 3: Write `ffrefine/io/includes.py`**

```python
"""How the topologies ffrefine writes spell their ``#include`` lines.

Milestone 6a design, section 5. Two modes:

``absolute``
    For the copies grompp reads (``02_mm_scan/full.itp``, ``zeroed.itp``,
    ``final/before.itp``, ``after.itp``). They live in stage directories,
    where a relative include such as ``./charmm36.ff/forcefield.itp`` would
    no longer resolve, so every relative include that names an existing file
    under the include base is rewritten to its absolute path.

``verbatim``
    For the files the user keeps (``03_fit/dihedral.itp``,
    ``04_charges/molecule_charged.itp``, ``final/molecule_refined.itp``). The
    user's own include lines are kept, so each file is a drop-in replacement
    beside the original, and two comment lines at the top record the
    directory those lines were written for. That marker is how a later stage,
    or a later standalone command, finds the force field again.

The include base of a file is the directory its marker names, or the file's
own directory when it carries no marker. Only the lines of the file being
written are rewritten; includes nested inside included files resolve against
their own directories, as they always did.

Pure text: this module never starts a process.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

from ..errors import ParseError, ValidationError

log = logging.getLogger("ffrefine")

IncludeMode = Literal["absolute", "verbatim"]

MARKER_NOTE = (
    "; ffrefine: relative #include paths in this file are written for the directory below"
)
MARKER_PREFIX = "; ffrefine include-base: "

_INCLUDE_RE = re.compile(r'^(?P<lead>\s*#\s*include\s*)"(?P<path>[^"]+)"(?P<tail>.*)$')
_ANY_INCLUDE_RE = re.compile(r"^\s*#\s*include\b")


def has_includes(lines: list[str]) -> bool:
    """Whether any line is an ``#include`` directive (quoted or angle-bracketed)."""
    return any(_ANY_INCLUDE_RE.match(line) for line in lines)


def find_marker(lines: list[str], source: str | Path) -> Path | None:
    """The include base a file's marker records, or ``None`` without one.

    Raises:
        ParseError: the file carries two markers naming different directories.
    """
    values = {line[len(MARKER_PREFIX) :].strip() for line in lines if line.startswith(MARKER_PREFIX)}
    if len(values) > 1:
        raise ParseError(
            f"{source}: two ffrefine include-base markers name different directories "
            f"({', '.join(sorted(values))}); keep the one the relative #include lines were "
            "written for and delete the other"
        )
    return Path(values.pop()) if values else None


def resolve_include_base(source: str | Path, marker: Path | None) -> Path:
    """The directory ``source``'s relative includes resolve against.

    Raises:
        ValidationError: the marker names a directory that does not exist.
            Falling back to the file's own directory could resolve a
            different force field, so this is never done silently.
    """
    if marker is None:
        return Path(source).resolve().parent
    if not marker.is_dir():
        raise ValidationError(
            f"{source} says its relative #include paths are written for {marker}, which "
            "does not exist. Re-run from the original topology, or move the force field "
            f"back to {marker}."
        )
    return marker


def include_base(path: str | Path) -> Path:
    """Read ``path`` and return the directory its relative includes resolve against."""
    lines = Path(path).read_text().splitlines()
    return resolve_include_base(path, find_marker(lines, path))


def strip_marker(lines: list[str]) -> list[str]:
    """``lines`` without ffrefine's include-base marker."""
    return [
        line
        for line in lines
        if line.rstrip() != MARKER_NOTE and not line.startswith(MARKER_PREFIX)
    ]


def _resolved_target(line: str, base: Path) -> Path | None:
    """The absolute file a relative quoted include names under ``base``, if it exists."""
    m = _INCLUDE_RE.match(line)
    if not m or Path(m.group("path")).is_absolute():
        return None
    candidate = Path(base) / m.group("path")
    return candidate.resolve() if candidate.is_file() else None


def absolutise_includes(lines: list[str], base: Path) -> list[str]:
    """Mode ``absolute``: rewrite every relative include that exists under ``base``.

    Angle-bracketed includes, absolute paths and relative paths with no file
    under ``base`` are copied verbatim; grompp then searches for them exactly
    as it would have for the original (the including file's directory, then
    GMXLIB). A marker is dropped: an absolute copy no longer needs one.
    """
    out: list[str] = []
    for line in strip_marker(lines):
        m = _INCLUDE_RE.match(line)
        target = _resolved_target(line, base)
        if target is not None:
            line = f'{m.group("lead")}"{target}"{m.group("tail")}'
        elif m and not Path(m.group("path")).is_absolute():
            log.info(
                "%s: no such file under %s; left for grompp's own include search",
                m.group("path"),
                base,
            )
        out.append(line)
    return out


def mark_include_base(lines: list[str], base: Path) -> list[str]:
    """Mode ``verbatim``: keep the user's include lines and record their base.

    The marker is written only when at least one relative include resolves
    under ``base``, so a topology without includes (the butane fixture) comes
    out unchanged. An existing marker is replaced, never duplicated.
    """
    body = strip_marker(lines)
    if not any(_resolved_target(line, base) is not None for line in body):
        return body
    return [MARKER_NOTE, f"{MARKER_PREFIX}{Path(base).resolve()}", *body]


def apply_include_mode(lines: list[str], base: Path, mode: IncludeMode) -> list[str]:
    """Dispatch to :func:`absolutise_includes` or :func:`mark_include_base`."""
    if mode == "absolute":
        return absolutise_includes(lines, base)
    if mode == "verbatim":
        return mark_include_base(lines, base)
    raise ValueError(f"unknown include mode {mode!r}; expected 'absolute' or 'verbatim'")
```

- [ ] **Step 4: Extend `ffrefine/io/topology.py`**

Add the import below the existing `from ..errors import ParseError`:

```python
from .includes import find_marker, resolve_include_base
```

Add the field and property to `Topology`, after `source: Path` (document the field in the class docstring's Attributes list too: "include_marker: the directory an ffrefine include-base marker in the file names, or None"):

```python
    include_marker: Path | None = None

    @property
    def include_base(self) -> Path:
        """Directory the file's relative ``#include`` lines resolve against.

        The directory an ffrefine marker names, else the file's own directory
        (milestone 6a design, section 5).

        Raises:
            ValidationError: the marker names a directory that does not exist.
        """
        return resolve_include_base(self.source, self.include_marker)
```

Add, above `read_topology`:

```python
def parse_defaults_fields(fields: list[str]) -> Defaults:
    """Build a :class:`Defaults` from the fields of one ``[ defaults ]`` line.

    Shared by :func:`read_topology` and the reader of grompp's preprocessed
    topology, so the two cannot disagree about the directive's layout.

    Raises:
        IndexError, ValueError: the line is malformed.
    """
    gen = fields[2].lower() in ("yes", "y") if len(fields) > 2 else False
    return Defaults(
        nbfunc=int(fields[0]),
        comb_rule=int(fields[1]),
        gen_pairs=gen,
        fudge_lj=float(fields[3]) if len(fields) > 3 else 1.0,
        fudge_qq=float(fields[4]) if len(fields) > 4 else 1.0,
    )
```

In `read_topology`, replace the body of the `if section == "defaults":` branch with:

```python
            if section == "defaults":
                defaults = parse_defaults_fields(fields)
```

and pass the marker when building the result (last keyword of the `Topology(...)` call):

```python
        source=path,
        include_marker=find_marker(all_lines, path),
    )
```

`find_marker` raises `ParseError` for two conflicting markers, which `read_topology`'s docstring already promises for malformed input; add "or two conflicting ffrefine include-base markers" to its Raises line.

- [ ] **Step 5: Run the tests**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_includes.py tests/test_topology.py -q && python -m pytest -q -x'`
Expected: the new tests pass, and the whole suite still passes (the `needs_*` tests included).

- [ ] **Step 6: black and commit**

```bash
wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m black ffrefine/io/includes.py ffrefine/io/topology.py tests/test_includes.py tests/test_topology.py'
cd /c/Users/hrach/Projects/ffrefine
git add ffrefine/io/includes.py ffrefine/io/topology.py tests/test_includes.py tests/test_topology.py
git commit -m "feat(io): include modes and the include-base marker for written topologies

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Include modes in every topology writer

**Files:**
- Modify: `ffrefine/io/topwriter.py` (`write_molecule_copy`, `write_zeroed_topology`, `write_refit_topology`, `write_charged_topology`; `_rewrite_dihedrals` reads the source text, refinement 8)
- Modify: `ffrefine/mm/validate_fit.py:196` (`after.itp` written `absolute`)
- Modify: `ffrefine/cli.py` (fit's `dihedral.itp` written `verbatim`; validate writes `molecule_refined.itp` instead of copying `after.itp`)
- Modify: `tests/test_topwriter.py:110,135,224,257`, `tests/test_prm.py:197` (pass `includes=`)
- Test: `tests/test_topwriter.py` (append), `tests/test_cli_typed.py` (new)

**Interfaces:**
- Consumes: from Task 2, `IncludeMode`, `absolutise_includes`, `mark_include_base`, `apply_include_mode`, `include_base`, `MARKER_NOTE`, `MARKER_PREFIX`, `Topology.include_base`.
- Produces:
  - `write_refit_topology(topology, quad, k, delta_deg, mult, out_path, distribute=False, *, includes: IncludeMode) -> None`: `includes` is keyword-only and **required**.
  - `write_molecule_copy(source, out_path) -> list[str]` and `write_zeroed_topology(topology, quad, out_path) -> list[Dihedral]`: unchanged signatures, now always `absolute`.
  - `write_charged_topology(source, assignments, out_path, decimals=4) -> list[int]`: unchanged signature, now always `verbatim`.
  - `tests/test_cli_typed.py` helpers later tasks extend: `PFOA`, `PFOA_QUAD`, `_pfoa_args(outdir, top=None)`, `_include_lines(path)`, `_write_pfoa_stage_inputs(outdir, n=24)`, `_fake_validate_refit(...)`, `_stub_validate(monkeypatch)`.

- [ ] **Step 1: Write the failing writer tests**

Append to `tests/test_topwriter.py` (add `import re`, `from pathlib import Path`, `import pytest` and these imports if missing):

```python
from ffrefine.io.includes import MARKER_NOTE, MARKER_PREFIX
from ffrefine.io.topwriter import write_charged_topology, write_molecule_copy

PFOA = Path(__file__).parent.parent / "fixtures" / "pfoa_gromacs"
PFOA_FF = (PFOA / "charmm36.ff").resolve()


def _include_lines(path) -> list[str]:
    return [line for line in Path(path).read_text().splitlines() if re.match(r"^\s*#\s*include", line)]


def _refit_args(t, quad):
    return (t, quad, np.array([1.0]), np.array([0.0]), np.array([3]))


def test_grompp_facing_copies_carry_absolute_includes(tmp_path):
    """Design section 5: a stage-directory copy must still find charmm36.ff/."""
    t = read_topology(PFOA / "pfoa_gmx.top")
    write_molecule_copy(t.source, tmp_path / "full.itp")
    write_zeroed_topology(t, (25, 1, 3, 4), tmp_path / "zeroed.itp")
    write_refit_topology(*_refit_args(t, (25, 1, 3, 4)), tmp_path / "after.itp", includes="absolute")
    for name in ("full.itp", "zeroed.itp", "after.itp"):
        lines = _include_lines(tmp_path / name)
        assert f'#include "{PFOA_FF / "forcefield.itp"}"' in lines, name
        assert f'# include "{PFOA_FF / "pfo_ffbonded.itp"}"' in lines, name
        assert '#include "posre.itp"' in lines, name  # not beside the .top: left for grompp
        assert MARKER_PREFIX not in (tmp_path / name).read_text(), name


def test_kept_copies_carry_the_users_includes_and_the_marker(tmp_path):
    """Design section 5: the files the user keeps are drop-in replacements."""
    t = read_topology(PFOA / "pfoa_gmx.top")
    write_refit_topology(
        *_refit_args(t, (25, 1, 3, 4)), tmp_path / "molecule_refined.itp", includes="verbatim"
    )
    write_charged_topology(PFOA / "pfoa_gmx.top", [], tmp_path / "molecule_charged.itp")
    for name in ("molecule_refined.itp", "molecule_charged.itp"):
        path = tmp_path / name
        assert _include_lines(path) == _include_lines(PFOA / "pfoa_gmx.top"), name
        assert path.read_text().splitlines()[:2] == [
            MARKER_NOTE,
            f"{MARKER_PREFIX}{PFOA.resolve()}",
        ], name


def test_a_marked_copy_resolves_its_includes_against_the_marked_directory(tmp_path):
    """The charged topology moves to 04_charges/, yet later copies find the force field."""
    moved = tmp_path / "out" / "04_charges"
    moved.mkdir(parents=True)
    write_charged_topology(PFOA / "pfoa_gmx.top", [], moved / "molecule_charged.itp")
    t = read_topology(moved / "molecule_charged.itp")
    write_zeroed_topology(t, (25, 1, 3, 4), tmp_path / "zeroed.itp")
    write_molecule_copy(moved / "molecule_charged.itp", tmp_path / "full.itp")
    for name in ("zeroed.itp", "full.itp"):
        assert f'#include "{PFOA_FF / "forcefield.itp"}"' in _include_lines(tmp_path / name)


def test_rewritten_copies_keep_ifdef_branches_the_parser_did_not_take(tmp_path):
    """Refinement 8: CGenFF's POSRES_WATER block must survive in every rewritten copy."""
    t = read_topology(PFOA / "pfoa_gmx.top")
    write_zeroed_topology(t, (25, 1, 3, 4), tmp_path / "zeroed.itp")
    write_refit_topology(
        *_refit_args(t, (25, 1, 3, 4)), tmp_path / "molecule_refined.itp", includes="verbatim"
    )
    for name in ("zeroed.itp", "molecule_refined.itp"):
        lines = [line.strip() for line in (tmp_path / name).read_text().splitlines()]
        start = lines.index("#ifdef POSRES_WATER")
        assert lines[start : start + 5] == [
            "#ifdef POSRES_WATER",
            "; Position restraint for each water oxygen",
            "[ position_restraints ]",
            ";  i funct       fcx        fcy        fcz",
            "1    1       1000       1000       1000",
        ], name


def test_butane_outputs_are_identical_in_both_modes(tmp_path, butane_itp):
    """No #include, so no marker and no rewritten line: master's bytes exactly."""
    t = read_topology(butane_itp)
    write_refit_topology(*_refit_args(t, (1, 5, 8, 11)), tmp_path / "a.itp", includes="absolute")
    write_refit_topology(*_refit_args(t, (1, 5, 8, 11)), tmp_path / "v.itp", includes="verbatim")
    assert (tmp_path / "a.itp").read_bytes() == (tmp_path / "v.itp").read_bytes()
    assert MARKER_PREFIX not in (tmp_path / "a.itp").read_text()


def test_write_refit_topology_requires_an_include_mode(tmp_path, butane_itp):
    t = read_topology(butane_itp)
    with pytest.raises(TypeError, match="includes"):
        write_refit_topology(*_refit_args(t, (1, 5, 8, 11)), tmp_path / "x.itp")
```

Update the five existing `write_refit_topology` calls so they still run: `tests/test_topwriter.py` lines 110, 135, 224 and 257, and `tests/test_prm.py` line 197. Add `includes="absolute"` as the last argument of each (butane has no includes, so either mode writes the same bytes).

- [ ] **Step 2: Write the failing CLI tests**

Create `tests/test_cli_typed.py`:

```python
"""The CLI on a real CGenFF topology: typed torsions and #include'd force fields.

Milestone 6a. The unit tests here stub the GROMACS-facing calls, so they need
no binary; the live tests are marked.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from ffrefine import cli
from ffrefine.cli import main
from ffrefine.io.includes import MARKER_NOTE, MARKER_PREFIX
from ffrefine.io.topwriter import write_refit_topology
from ffrefine.mm.validate_fit import ValidationResult
from ffrefine.stages import hash_array

FIXTURES = Path(__file__).parent.parent / "fixtures"
PFOA = FIXTURES / "pfoa_gromacs"
PFOA_QUAD = (25, 1, 3, 4)
PFOA_FF = (PFOA / "charmm36.ff").resolve()


def _pfoa_args(outdir: Path, top: Path | None = None) -> list[str]:
    return [
        "-s",
        str(PFOA / "pfoa_gmx.pdb"),
        "-p",
        str(top or PFOA / "pfoa_gmx.top"),
        "--dihedral",
        ",".join(map(str, PFOA_QUAD)),
        "--outdir",
        str(outdir),
    ]


def _include_lines(path: Path) -> list[str]:
    return [line for line in Path(path).read_text().splitlines() if re.match(r"^\s*#\s*include", line)]


def _marker(path: Path) -> list[str]:
    return Path(path).read_text().splitlines()[:2]


def _write_pfoa_stage_inputs(outdir: Path, n: int = 24) -> None:
    """Hand-built 01_qm_scan and 02_mm_scan payloads for PFOA (see test_cli.py).

    ``fit`` only hashes the frames, so placeholder coordinates of the right
    shape are enough, and they must be the same in both payloads.
    """
    phi = np.linspace(-180.0, 180.0, n, endpoint=False)
    qm = 5.0 * (1.0 + np.cos(np.radians(3.0 * phi)))
    frames = np.arange(n * 26 * 3, dtype=float).reshape(n, 26, 3) * 0.01
    (outdir / "01_qm_scan").mkdir(parents=True, exist_ok=True)
    (outdir / "01_qm_scan" / "scan.json").write_text(
        json.dumps({"phi_deg": phi.tolist(), "energy_kj": qm.tolist(), "frames_nm": frames.tolist()})
    )
    (outdir / "02_mm_scan").mkdir(parents=True, exist_ok=True)
    (outdir / "02_mm_scan" / "energies.json").write_text(
        json.dumps(
            {
                "quad": list(PFOA_QUAD),
                "frames_sha256": hash_array(frames),
                "phi_deg": phi.tolist(),
                "zeroed_potential_kj": [0.0] * n,
                "full_potential_kj": [0.0] * n,
            }
        )
    )


def _fake_validate_refit(
    binaries, topology, quad, structure, frames, qm, k, delta, mult, workdir, **kwargs
):
    """validate_refit without GROMACS: writes after.itp the way the real one does."""
    write_refit_topology(
        topology,
        quad,
        k,
        delta,
        mult,
        Path(workdir) / "after.itp",
        distribute=kwargs.get("distribute", False),
        includes="absolute",
    )
    n = len(qm)
    return ValidationResult(
        rmse_before=1.0,
        rmse_after=0.5,
        mm_before_kj=np.zeros(n),
        mm_after_kj=np.zeros(n),
        qm_kj=np.asarray(qm, dtype=float),
        phi_deg=np.linspace(-180.0, 180.0, n, endpoint=False),
        improved=True,
    )


def _stub_validate(monkeypatch) -> None:
    """Replace every GROMACS-facing call cmd_validate makes."""
    monkeypatch.setattr(cli, "validate_refit", _fake_validate_refit)


def test_fit_writes_dihedral_itp_with_the_users_includes(tmp_path):
    out = tmp_path / "out"
    _write_pfoa_stage_inputs(out)
    assert main(["fit"] + _pfoa_args(out)) == 0
    itp = out / "03_fit" / "dihedral.itp"
    assert _include_lines(itp) == _include_lines(PFOA / "pfoa_gmx.top")
    assert _marker(itp) == [MARKER_NOTE, f"{MARKER_PREFIX}{PFOA.resolve()}"]


def test_validate_writes_molecule_refined_as_a_drop_in(tmp_path, monkeypatch):
    """final/after.itp goes to grompp (absolute); molecule_refined.itp is kept (verbatim)."""
    out = tmp_path / "out"
    _write_pfoa_stage_inputs(out)
    assert main(["fit"] + _pfoa_args(out)) == 0
    _stub_validate(monkeypatch)
    assert main(["validate"] + _pfoa_args(out)) == 0
    final = out / "final"
    refined = final / "molecule_refined.itp"
    assert _include_lines(refined) == _include_lines(PFOA / "pfoa_gmx.top")
    assert _marker(refined) == [MARKER_NOTE, f"{MARKER_PREFIX}{PFOA.resolve()}"]
    assert f'#include "{PFOA_FF / "forcefield.itp"}"' in _include_lines(final / "after.itp")
```

- [ ] **Step 3: Run them to verify they fail**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_topwriter.py tests/test_prm.py tests/test_cli_typed.py -q'`
Expected: the five updated calls and the new writer tests fail with `TypeError: write_refit_topology() got an unexpected keyword argument 'includes'`; the includes assertions fail.

- [ ] **Step 4: Apply the modes in `ffrefine/io/topwriter.py`**

Add to the imports:

```python
from .includes import (
    IncludeMode,
    absolutise_includes,
    apply_include_mode,
    include_base,
    mark_include_base,
)
```

Add one paragraph to the module docstring, after the first:

```
Every writer settles how the copy spells its ``#include`` lines (milestone 6a
design, section 5): the copies grompp reads get absolute paths, because they
live in stage directories; the files the user keeps get the user's own lines
plus an include-base marker, so they are drop-in replacements beside the
original. See :mod:`ffrefine.io.includes`.
```

In `_rewrite_dihedrals`, iterate over the source text instead of the parsed lines (refinement 8):

```python
    for raw in Path(topology.source).read_text().splitlines():
```

replacing `for raw in topology.raw_lines:`, and change its docstring's Args line to "topology: parsed source; its file is re-read, so ``#ifdef`` branches the parser did not take are copied too (a CGenFF ``.top``'s ``POSRES_WATER`` block), exactly as :func:`write_molecule_copy` does." The target set is still computed from the parsed propers, i.e. from the branches the parser took.

In `write_molecule_copy`, make the includes absolute before writing:

```python
    lines, dropped = strip_system_sections(Path(source).read_text().splitlines())
    lines = absolutise_includes(lines, include_base(source))
    Path(out_path).write_text("\n".join(lines) + "\n", newline="\n")
```

and add to its docstring: "Relative ``#include`` lines are made absolute against the source's include base: the copy lives in a stage directory, where they would no longer resolve."

In `write_zeroed_topology`, replace the last two lines of the body with:

```python
    lines, _ = strip_system_sections(_rewrite_dihedrals(topology, targets, replacement))
    lines = absolutise_includes(lines, topology.include_base)
    Path(out_path).write_text("\n".join(lines) + "\n", newline="\n")
    return tset
```

and add to its docstring: "Always a grompp input, so relative ``#include`` lines are made absolute."

Change `write_refit_topology`'s signature and its final lines:

```python
def write_refit_topology(
    topology: Topology,
    quad: Quad,
    k: np.ndarray,
    delta_deg: np.ndarray,
    mult: np.ndarray,
    out_path: str | Path,
    distribute: bool = False,
    *,
    includes: IncludeMode,
) -> None:
```

```python
    lines, _ = strip_system_sections(_rewrite_dihedrals(topology, targets, replacement))
    lines = apply_include_mode(lines, topology.include_base, includes)
    Path(out_path).write_text("\n".join(lines) + "\n", newline="\n")
```

and document the argument in its Args list:

```
        includes: ``"absolute"`` for a grompp input (``final/after.itp``),
            ``"verbatim"`` for a file the user keeps (``03_fit/dihedral.itp``,
            ``final/molecule_refined.itp``). Required: this writer serves both,
            and neither is a safe default.
```

In `write_charged_topology`, mark the base just before the write:

```python
    out = mark_include_base(out, include_base(source))
    Path(out_path).write_text("\n".join(out) + "\n", newline="\n")
```

and add to its docstring: "The file is kept by the user and read by later stages, so its ``#include`` lines stay as written and an include-base marker records where they point."

- [ ] **Step 5: Update the three production call sites**

`ffrefine/mm/validate_fit.py`, in `validate_refit`:

```python
    write_refit_topology(
        topology, quad, k, delta_deg, mult, after_itp, distribute=distribute, includes="absolute"
    )
```

`ffrefine/cli.py`, in `cmd_fit`, the `write_refit_topology(...)` call that writes `outdir / "dihedral.itp"`: add `includes="verbatim",` after `distribute=args.distribute,`.

`ffrefine/cli.py`, in `cmd_validate`, replace

```python
    shutil.copyfile(final / "after.itp", final / "molecule_refined.itp")
```

with

```python
    # Written, not copied from after.itp: after.itp is a grompp input and so
    # carries absolute #include paths, while the file the user keeps carries
    # their own (milestone 6a design, section 4). Same bytes for butane.
    write_refit_topology(
        topology,
        quad,
        k,
        delta,
        mult,
        final / "molecule_refined.itp",
        distribute=args.distribute,
        includes="verbatim",
    )
```

- [ ] **Step 6: Run the tests**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_topwriter.py tests/test_prm.py tests/test_cli_typed.py -q && python -m pytest -q'`
Expected: all pass, the `needs_*` tests included (the butane round trip still grompps its absolute copies, and they are unchanged for butane).

- [ ] **Step 7: black and commit**

```bash
wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m black ffrefine tests'
cd /c/Users/hrach/Projects/ffrefine
git add ffrefine/io/topwriter.py ffrefine/mm/validate_fit.py ffrefine/cli.py tests/test_topwriter.py tests/test_prm.py tests/test_cli_typed.py
git commit -m "feat(io): absolute includes in grompp inputs, the user's own in kept files

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: `mm/tprparams.py`, the parsers and the cross-check

**Files:**
- Create: `ffrefine/mm/tprparams.py`
- Test: `tests/test_tprparams.py` (new)

**Interfaces:**
- Consumes: from Task 1, `fixtures/pfoa_gromacs/gromacs/pfoa_dump_2025.4.txt`; from Task 2, `parse_defaults_fields`; the existing `Dihedral`, `Defaults`.
- Produces (all in `ffrefine.mm.tprparams`):
  - `SOURCE_TOPOLOGY = "topology"`, `SOURCE_DIHEDRALTYPES = "dihedraltypes"`, `PHI_TOLERANCE_DEG = 1e-4`, `K_RELATIVE_TOLERANCE = 1e-6`
  - `DumpedTerm(quad: tuple[int, int, int, int], phi0_deg: float, k_kj_per_mol: float, multiplicity: int)`, frozen; `quad` 1-based
  - `ResolvedTerm(dihedral: Dihedral, source: str)`, frozen
  - `ResolvedTorsionSet(terms, defaults, pp_sha256, full_top, dump_path, gromacs_version)` with property `dihedrals -> list[Dihedral]`
  - `parse_dump_torsions(text: str, molname: str, natoms: int, source: Path) -> list[DumpedTerm]`
  - `parse_defaults(text: str, source: Path) -> Defaults | None`
  - `cross_check(tset: list[Dihedral], dumped: list[DumpedTerm], dump_path: Path) -> list[ResolvedTerm]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tprparams.py`:

```python
"""Reading grompp's own torsion parameters back (milestone 6a design, section 6).

The parser tests read a real `gmx dump` of the trimmed PFOA fixture, captured
with GROMACS 2025.4 (fixtures/pfoa_gromacs/PROVENANCE.md), plus text-edited
copies of it for every refusal.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from ffrefine.errors import ValidationError
from ffrefine.io.topology import Defaults, Dihedral
from ffrefine.mm.tprparams import (
    SOURCE_DIHEDRALTYPES,
    SOURCE_TOPOLOGY,
    DumpedTerm,
    ResolvedTerm,
    cross_check,
    parse_defaults,
    parse_dump_torsions,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"
PFOA = FIXTURES / "pfoa_gromacs"
DUMP = PFOA / "gromacs" / "pfoa_dump_2025.4.txt"
SRC = Path("dump.txt")


@pytest.fixture(scope="module")
def dump_text() -> str:
    return DUMP.read_text()


def _edit(text: str, pattern: str, repl: str) -> str:
    """Apply one regex edit, and fail loudly if it matched nothing."""
    out, n = re.subn(pattern, repl, text, count=1)
    assert n == 1, f"fixture layout changed: {pattern!r} not found"
    return out


# --------------------------------------------------------------------------
# parse_dump_torsions
# --------------------------------------------------------------------------


def test_reads_every_proper_term_of_the_pfoa_dump(dump_text):
    terms = parse_dump_torsions(dump_text, "Other", 26, DUMP)
    assert len(terms) == 76
    about = [t for t in terms if sorted(t.quad[1:3]) == [1, 3]]
    assert len(about) == 20
    target = [t for t in about if t.quad == (25, 1, 3, 4)]
    assert [t.multiplicity for t in target] == [1, 2, 3, 6]
    assert [t.phi0_deg for t in target] == [0.0, 0.0, 0.0, 0.0]
    # pfo_ffbonded.itp: OG311 CG2O2 CG312 CG312, stored by grompp as float32.
    assert [t.k_kj_per_mol for t in target] == pytest.approx(
        [0.66944, 0.46024, 0.33472, 1.046], rel=1e-6
    )


def test_keeps_the_written_atom_order(dump_text):
    """The dump keeps each [ dihedrals ] line's order: 2 1 3 4, never 4 3 1 2."""
    quads = {t.quad for t in parse_dump_torsions(dump_text, "Other", 26, DUMP)}
    assert (2, 1, 3, 4) in quads and (4, 3, 1, 2) not in quads


def test_refuses_a_missing_moltype(dump_text):
    with pytest.raises(ValidationError, match="exactly one moltype named 'Nope'"):
        parse_dump_torsions(dump_text, "Nope", 26, DUMP)


def test_refuses_an_atom_count_mismatch(dump_text):
    with pytest.raises(ValidationError, match="has 26 atoms, the topology 25"):
        parse_dump_torsions(dump_text, "Other", 25, DUMP)


def test_refuses_a_free_energy_topology(dump_text):
    text = _edit(dump_text, r"cpB= 2\.88696003e\+00", "cpB= 3.00000000e+00")
    with pytest.raises(ValidationError, match="different A and B states"):
        parse_dump_torsions(text, "Other", 26, DUMP)


def test_refuses_a_count_mismatch(dump_text):
    text = _edit(dump_text, r"(Proper Dih\.:\n\s*nr: )380", r"\g<1>385")
    with pytest.raises(ValidationError, match="nr: 385"):
        parse_dump_torsions(text, "Other", 26, DUMP)


def test_refuses_a_missing_proper_block(dump_text):
    text = _edit(dump_text, r"Proper Dih\.:", "Proper Dih.X:")
    with pytest.raises(ValidationError, match="no 'Proper Dih.' block"):
        parse_dump_torsions(text, "Other", 26, DUMP)


def test_refuses_an_entry_naming_a_non_pdihs_functype(dump_text):
    text = _edit(dump_text, r"(Proper Dih\.:\n\s*nr: \d+\n\s*iatoms:\n\s*0 type=)\d+", r"\g<1>0")
    with pytest.raises(ValidationError, match="not PDIHS"):
        parse_dump_torsions(text, "Other", 26, DUMP)


def test_refuses_an_unrecognised_pdihs_line(dump_text):
    text = _edit(dump_text, r"(functype\[\d+\]=PDIHS, .*?mult=)\d+", r"\g<1>x")
    with pytest.raises(ValidationError, match="unrecognised PDIHS functype line"):
        parse_dump_torsions(text, "Other", 26, DUMP)


def test_refusals_blame_the_reader_not_the_topology(dump_text):
    with pytest.raises(ValidationError) as excinfo:
        parse_dump_torsions(dump_text, "Nope", 26, DUMP)
    assert "not a problem with your topology" in str(excinfo.value)
    assert str(DUMP) in str(excinfo.value)


# --------------------------------------------------------------------------
# parse_defaults
# --------------------------------------------------------------------------


def test_parse_defaults_reads_the_defaults_line():
    text = "; header\n[ defaults ]\n; nbfunc comb-rule\n1  2  yes  1.0  1.0\n[ atomtypes ]\n"
    assert parse_defaults(text, SRC) == Defaults(1, 2, True, 1.0, 1.0)


def test_parse_defaults_without_the_directive():
    assert parse_defaults("[ atomtypes ]\nCG321 6 12.011 0 A 0.3 0.2\n", SRC) is None


def test_parse_defaults_refuses_a_malformed_line():
    with pytest.raises(ValidationError, match="malformed"):
        parse_defaults("[ defaults ]\none two\n", SRC)


# --------------------------------------------------------------------------
# cross_check (design section 6.3)
# --------------------------------------------------------------------------

Q = (1, 2, 3, 4)
R = (5, 2, 3, 4)


def _line(quad, params=(), funct=9) -> Dihedral:
    return Dihedral(*quad, funct=funct, params=[float(p) for p in params])


def _term(quad, phi, k, n) -> DumpedTerm:
    return DumpedTerm(quad, phi, k, n)


def test_a_typed_line_takes_the_dumped_terms():
    out = cross_check([_line(Q)], [_term(Q, 0.0, 1.5, 1), _term(Q, 180.0, 0.5, 2)], SRC)
    assert [(r.dihedral.params, r.source) for r in out] == [
        ([0.0, 1.5, 1.0], SOURCE_DIHEDRALTYPES),
        ([180.0, 0.5, 2.0], SOURCE_DIHEDRALTYPES),
    ]
    assert all((r.dihedral.ai, r.dihedral.aj, r.dihedral.ak, r.dihedral.al) == Q for r in out)
    assert all(r.dihedral.funct == 9 for r in out)


def test_a_typed_line_with_no_dumped_terms_contributes_nothing():
    """Every CGenFF term k = 0: grompp builds nothing (capecitabine's 6 7 9 37)."""
    explicit = _line(R, (0.0, 2.0, 3))
    assert cross_check([_line(Q), explicit], [_term(R, 0.0, 2.0, 3)], SRC) == [
        ResolvedTerm(explicit, SOURCE_TOPOLOGY)
    ]


def test_explicit_lines_keep_their_own_values_within_float32_rounding():
    line = _line(Q, (0.0, 2.0, 3))
    out = cross_check([line], [_term(Q, 0.0, 2.0000001, 3)], SRC)
    assert out == [ResolvedTerm(line, SOURCE_TOPOLOGY)]


def test_an_explicit_line_that_disagrees_with_the_dump_is_refused():
    with pytest.raises(ValidationError, match="disagree"):
        cross_check([_line(Q, (0.0, 2.0, 3))], [_term(Q, 0.0, 2.5, 3)], SRC)


def test_an_explicit_zero_k_line_is_expected_absent_from_the_dump():
    line = _line(Q, (0.0, 0.0, 1))
    assert cross_check([line], [], SRC) == [ResolvedTerm(line, SOURCE_TOPOLOGY)]


def test_a_quad_both_typed_and_explicit_is_refused():
    with pytest.raises(ValidationError, match="ambiguous"):
        cross_check([_line(Q), _line(Q, (0.0, 1.0, 3))], [_term(Q, 0.0, 1.0, 3)], SRC)


def test_a_quad_typed_twice_is_refused():
    with pytest.raises(ValidationError, match="ambiguous"):
        cross_check([_line(Q), _line(Q)], [_term(Q, 0.0, 1.0, 3)] * 2, SRC)


def test_a_dumped_term_no_line_accounts_for_is_refused():
    with pytest.raises(ValidationError, match=r"no \[ dihedrals \] line"):
        cross_check(
            [_line(Q, (0.0, 1.0, 3))],
            [_term(Q, 0.0, 1.0, 3), _term((6, 3, 2, 7), 0.0, 1.0, 3)],
            SRC,
        )


def test_terms_about_other_bonds_are_ignored():
    out = cross_check([_line(Q)], [_term(Q, 0.0, 1.0, 3), _term((2, 3, 4, 8), 0.0, 9.0, 1)], SRC)
    assert len(out) == 1


def test_an_empty_torsion_set_resolves_to_nothing():
    assert cross_check([], [_term(Q, 0.0, 1.0, 3)], SRC) == []
```

- [ ] **Step 2: Run them to verify they fail**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_tprparams.py -q'`
Expected: collection error, `ModuleNotFoundError: No module named 'ffrefine.mm.tprparams'`.

- [ ] **Step 3: Write `ffrefine/mm/tprparams.py`**

The parser below was run against both molecules' real dumps while planning.

```python
"""The torsion parameters grompp actually built (milestone 6a design, section 6).

A CGenFF or GAFF molecule topology writes its torsions as bare ``i j k l 9``
lines; the force constants come from ``[ dihedraltypes ]`` in the force-field
files it includes. This build does not reimplement grompp's matching rules
(most non-wildcard matches, first in file order, both orientations,
consecutive-line blocks for funct 9, later definitions overriding earlier
ones, bonded types, two-type entries). It asks grompp: ``grompp -pp`` writes
the fully preprocessed topology, ``gmx dump`` prints the parameters stored in
the ``.tpr``, and :func:`cross_check` pairs each line of the torsion set T
with the terms grompp built for it.

Two facts measured on GROMACS 2025.4 shape the cross-check:

* A typed funct-9 line expands into one term per multiplicity, and the dump
  keeps the atom order the line was written in.
* grompp builds **no** term for a ``k = 0`` term, typed or explicit. A ``k = 0``
  term is therefore expected to be absent, and a typed line whose terms all
  have ``k = 0`` legitimately resolves to nothing.

Indices are 1-based throughout, like the topology file; the dump's 0-based
atom indices are converted as they are read.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from ..errors import ValidationError
from ..io.topology import Defaults, Dihedral, parse_defaults_fields

log = logging.getLogger("ffrefine")

Quad = tuple[int, int, int, int]

SOURCE_TOPOLOGY = "topology"
"""The term's parameters are written on its own ``[ dihedrals ]`` line."""

SOURCE_DIHEDRALTYPES = "dihedraltypes"
"""The term came from ``[ dihedraltypes ]``, read back from grompp's ``.tpr``."""

PHI_TOLERANCE_DEG = 1e-4
K_RELATIVE_TOLERANCE = 1e-6
"""Explicit lines against the dump: grompp stores float32, printed to 9 digits."""

_FUNCTYPE_RE = re.compile(r"^\s*functype\[(\d+)\]=(\w+),\s*(.*?)\s*$")
_PDIHS_RE = re.compile(
    r"^phiA=\s*(\S+),\s*cpA=\s*(\S+),\s*phiB=\s*(\S+),\s*cpB=\s*(\S+),\s*mult=\s*(-?\d+)$"
)
_MOLTYPE_RE = re.compile(r"^\s*moltype \((\d+)\):\s*$")
_NAME_RE = re.compile(r'^\s*name="(.*)"\s*$')
_ATOM_COUNT_RE = re.compile(r"^\s*atom \((\d+)\):\s*$")
_PROPER_HEADING = "Proper Dih.:"
_NR_RE = re.compile(r"^\s*nr:\s*(\d+)\s*$")
_IATOM_RE = re.compile(r"^\s*(\d+)\s+type=(\d+)\s+\((\w+)\)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$")


@dataclass(frozen=True)
class DumpedTerm:
    """One ``Proper Dih.`` entry of the dump.  ``quad`` is 1-based, as written."""

    quad: Quad
    phi0_deg: float
    k_kj_per_mol: float
    multiplicity: int


@dataclass(frozen=True)
class ResolvedTerm:
    """One term of T with explicit parameters, and where they came from.

    ``dihedral.params`` is always ``[phi0_deg, k_kj_per_mol, multiplicity]``.
    ``source`` is :data:`SOURCE_TOPOLOGY` or :data:`SOURCE_DIHEDRALTYPES`.
    """

    dihedral: Dihedral
    source: str


@dataclass
class ResolvedTorsionSet:
    """T with every term's parameters, as grompp built them.

    Attributes:
        terms: one entry per term, in topology order.
        defaults: ``[ defaults ]`` from grompp's preprocessed topology, or
            ``None`` if it has none.
        pp_sha256: SHA-256 of that preprocessed topology: a digest of
            everything grompp read, used as part of the ``--resume`` key.
        full_top: the wrapper ``.top`` grompp was given for the unmodified
            topology; the self-check's "full" energies use the same file.
        dump_path: where the ``gmx dump`` text is on disk.
        gromacs_version: from the dump's banner, for messages.
    """

    terms: list[ResolvedTerm]
    defaults: Defaults | None
    pp_sha256: str
    full_top: Path
    dump_path: Path
    gromacs_version: str

    @property
    def dihedrals(self) -> list[Dihedral]:
        """The terms as dihedral entries, for :func:`ffrefine.mm.analytic.torsion_set_energy`."""
        return [t.dihedral for t in self.terms]


def _dump_error(source: Path, what: str) -> ValidationError:
    return ValidationError(
        f"could not read the torsion parameters from {source}: {what}. This is a "
        "limit of ffrefine's reader of 'gmx dump' output, not a problem with your "
        "topology."
    )


def _pdihs_functypes(lines: list[str], source: Path) -> dict[int, tuple[float, float, int]]:
    """``functype[N]=PDIHS`` entries: ``N -> (phi0_deg, k, multiplicity)``."""
    out: dict[int, tuple[float, float, int]] = {}
    for line in lines:
        m = _FUNCTYPE_RE.match(line)
        if not m or m.group(2) != "PDIHS":
            continue
        p = _PDIHS_RE.match(m.group(3))
        if not p:
            raise _dump_error(source, f"unrecognised PDIHS functype line {line.strip()!r}")
        phi_a, k_a, phi_b, k_b = (float(p.group(i)) for i in range(1, 5))
        if phi_a != phi_b or k_a != k_b:
            raise ValidationError(
                f"{source}: functype[{m.group(1)}] has different A and B states "
                f"(phi {phi_a} / {phi_b}, k {k_a} / {k_b}). Free-energy (B-state) "
                "topologies are not supported."
            )
        out[int(m.group(1))] = (phi_a, k_a, int(p.group(5)))
    return out


def parse_dump_torsions(text: str, molname: str, natoms: int, source: Path) -> list[DumpedTerm]:
    """Every proper-dihedral term of one moltype in a ``gmx dump -s`` output.

    Args:
        text: the dump.
        molname: the topology's ``[ moleculetype ]`` name.
        natoms: the topology's atom count; the moltype must agree.
        source: the dump's path, for messages.

    Returns:
        One entry per ``Proper Dih.`` iatoms line, in order, 1-based.

    Raises:
        ValidationError: the layout is not what this reader knows, the
            moltype is missing or ambiguous or has the wrong atom count, or a
            term has different A and B states.
    """
    lines = text.splitlines()
    functypes = _pdihs_functypes(lines, source)

    starts = [i for i, line in enumerate(lines) if _MOLTYPE_RE.match(line)]
    blocks: list[tuple[str | None, list[str]]] = []
    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        name = next(
            (m.group(1) for m in map(_NAME_RE.match, lines[start + 1 : start + 3]) if m), None
        )
        blocks.append((name, lines[start + 1 : end]))
    matching = [block for name, block in blocks if name == molname]
    if len(matching) != 1:
        names = [name for name, _ in blocks]
        raise _dump_error(
            source, f"expected exactly one moltype named {molname!r}, found moltypes {names}"
        )
    block = matching[0]

    count = next((int(m.group(1)) for m in map(_ATOM_COUNT_RE.match, block) if m), None)
    if count != natoms:
        raise _dump_error(
            source, f"moltype {molname!r} has {count} atoms, the topology {natoms}"
        )

    try:
        head = next(i for i, line in enumerate(block) if line.strip() == _PROPER_HEADING)
    except StopIteration:
        raise _dump_error(source, f"moltype {molname!r} has no 'Proper Dih.' block") from None
    nr = _NR_RE.match(block[head + 1]) if head + 1 < len(block) else None
    if not nr:
        raise _dump_error(source, "the 'Proper Dih.' block has no 'nr:' line")
    expected = int(nr.group(1))
    terms: list[DumpedTerm] = []
    if expected:
        if head + 2 >= len(block) or block[head + 2].strip() != "iatoms:":
            raise _dump_error(source, "the 'Proper Dih.' block has no 'iatoms:' line")
        for line in block[head + 3 :]:
            m = _IATOM_RE.match(line)
            if not m:
                break
            if m.group(3) != "PDIHS":
                raise _dump_error(source, f"'Proper Dih.' entry of type {m.group(3)}: {line!r}")
            index = int(m.group(2))
            if index not in functypes:
                raise _dump_error(source, f"an entry names functype[{index}], which is not PDIHS")
            phi0, k, mult = functypes[index]
            quad = (int(m.group(4)) + 1, int(m.group(5)) + 1, int(m.group(6)) + 1, int(m.group(7)) + 1)
            terms.append(DumpedTerm(quad, phi0, k, mult))
    # nr counts integers: five per entry (the functype index and four atoms).
    if 5 * len(terms) != expected:
        raise _dump_error(
            source, f"'Proper Dih.' says nr: {expected} but {len(terms)} entries were read"
        )
    return terms


def parse_defaults(text: str, source: Path) -> Defaults | None:
    """``[ defaults ]`` from grompp's preprocessed topology, or ``None``.

    Raises:
        ValidationError: the directive's line is malformed.
    """
    section = ""
    for lineno, raw in enumerate(text.splitlines(), start=1):
        body = raw.split(";", 1)[0].strip()
        if not body or body.startswith("#"):
            continue
        if body.startswith("[") and body.endswith("]"):
            section = body[1:-1].strip().lower()
            continue
        if section == "defaults":
            try:
                return parse_defaults_fields(body.split())
            except (IndexError, ValueError) as exc:
                raise ValidationError(
                    f"{source}:{lineno}: malformed [ defaults ] line {raw!r}"
                ) from exc
    return None


def _central(quad: Quad) -> tuple[int, int]:
    return (min(quad[1], quad[2]), max(quad[1], quad[2]))


def _fmt(terms: list[tuple[float, float, int]]) -> str:
    return ", ".join(f"(phi0 {p:g}, k {k:g}, n {n})" for p, k, n in terms) or "none"


def _same_terms(expected: list[tuple[float, float, int]], got: list[DumpedTerm]) -> bool:
    """Multiset equality within the float32 tolerances."""
    if len(expected) != len(got):
        return False
    exp = sorted(expected, key=lambda t: (t[2], t[0], t[1]))
    obs = sorted(
        ((g.phi0_deg, g.k_kj_per_mol, g.multiplicity) for g in got),
        key=lambda t: (t[2], t[0], t[1]),
    )
    for (p1, k1, n1), (p2, k2, n2) in zip(exp, obs):
        if n1 != n2 or abs(p1 - p2) > PHI_TOLERANCE_DEG:
            return False
        if abs(k1 - k2) > K_RELATIVE_TOLERANCE * max(1.0, abs(k1)):
            return False
    return True


_IFDEF_HINT = (
    "Either ffrefine misread the dump, or grompp did not read the [ dihedrals ] lines as "
    "ffrefine did -- for example an #ifdef in the molecule file driven by a define that an "
    "included force-field file sets, which ffrefine cannot see."
)


def cross_check(
    tset: list[Dihedral], dumped: list[DumpedTerm], dump_path: Path
) -> list[ResolvedTerm]:
    """Pair every member of T with the terms grompp built for it (design section 6.3).

    Args:
        tset: :func:`ffrefine.mm.torsionset.collect_torsion_set`, one entry
            per term-line.
        dumped: :func:`parse_dump_torsions` for the whole molecule.
        dump_path: for messages.

    Returns:
        T with explicit parameters on every term, in topology order. An
        explicit line keeps its own values (the analytic reference for an
        explicit-only set is unchanged); a typed line takes grompp's.

    Raises:
        ValidationError: a quad written both typed and explicit, or typed
            twice; an explicit line whose non-zero terms differ from grompp's;
            a term grompp built about the central bond that no line accounts
            for.
    """
    if not tset:
        return []
    bond = tset[0].central_bond
    groups: dict[Quad, list[Dihedral]] = {}
    for d in tset:
        groups.setdefault((d.ai, d.aj, d.ak, d.al), []).append(d)
    by_quad: dict[Quad, list[DumpedTerm]] = {}
    for t in dumped:
        if _central(t.quad) == bond:
            by_quad.setdefault(t.quad, []).append(t)

    resolved: list[ResolvedTerm] = []
    for quad, lines in groups.items():
        where = " ".join(map(str, quad))
        typed = [d for d in lines if not d.params]
        explicit = [d for d in lines if d.params]
        got = by_quad.pop(quad, [])
        if (typed and explicit) or len(typed) > 1:
            raise ValidationError(
                f"the torsion {where} is written {len(lines)} times in [ dihedrals ] "
                f"({len(typed)} without parameters, {len(explicit)} with), and is refused as "
                "ambiguous: grompp builds terms for every line, and there is no unambiguous "
                "way to say which of them belong to which line. Merge them into one line, or "
                "give every line explicit parameters."
            )
        if typed:
            for t in got:
                resolved.append(
                    ResolvedTerm(
                        Dihedral(
                            *quad,
                            funct=typed[0].funct,
                            params=[t.phi0_deg, t.k_kj_per_mol, float(t.multiplicity)],
                            comment="from [ dihedraltypes ], read back with gmx dump",
                        ),
                        SOURCE_DIHEDRALTYPES,
                    )
                )
            if not got:
                log.info(
                    "torsion %s: every [ dihedraltypes ] term grompp matched has k = 0, so "
                    "it contributes nothing",
                    where,
                )
            continue
        for d in explicit:
            if len(d.params) < 3:
                raise ValidationError(
                    f"the torsion {where} has {len(d.params)} parameter(s) on its line; a "
                    "funct 1 or 9 dihedral needs phi0, k and a multiplicity"
                )
        expected = [
            (float(d.params[0]), float(d.params[1]), int(round(float(d.params[2]))))
            for d in explicit
            if float(d.params[1]) != 0.0
        ]
        if not _same_terms(expected, got):
            observed = [(g.phi0_deg, g.k_kj_per_mol, g.multiplicity) for g in got]
            raise ValidationError(
                f"the torsion {where} carries explicit parameters, but the terms grompp "
                "built for it disagree with them.\n"
                f"  in the topology (k != 0): {_fmt(expected)}\n"
                f"  in {dump_path}: {_fmt(observed)}\n  {_IFDEF_HINT}"
            )
        resolved += [ResolvedTerm(d, SOURCE_TOPOLOGY) for d in explicit]

    if by_quad:
        extra = [
            f"{' '.join(map(str, q))} (phi0 {t.phi0_deg:g}, k {t.k_kj_per_mol:g}, n {t.multiplicity})"
            for q, ts in by_quad.items()
            for t in ts
        ]
        raise ValidationError(
            f"grompp built {len(extra)} term(s) about the central bond {bond} that no "
            f"[ dihedrals ] line ffrefine read accounts for: {'; '.join(extra)} "
            f"(from {dump_path}).\n  {_IFDEF_HINT}"
        )
    return resolved
```

- [ ] **Step 4: Run the tests**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_tprparams.py -q'`
Expected: all pass.

- [ ] **Step 5: black and commit**

```bash
wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m black ffrefine/mm/tprparams.py tests/test_tprparams.py'
cd /c/Users/hrach/Projects/ffrefine
git add ffrefine/mm/tprparams.py tests/test_tprparams.py
git commit -m "feat(mm): read grompp's torsion parameters back from gmx dump, strictly

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Ask grompp, `resolve_torsion_set` and `preprocessed_sha256`

**Files:**
- Modify: `ffrefine/mm/gromacs.py` (add `TprDump`, `preprocess_topology`, `dump_tpr`)
- Modify: `ffrefine/mm/tprparams.py` (add `write_full_system`, `resolve_torsion_set`, `preprocessed_sha256`)
- Test: `tests/test_tprparams.py` (append live tests)

**Interfaces:**
- Consumes: Task 3's `write_molecule_copy` (absolute), `make_system_top`; Task 4's parsers and `cross_check`; the existing `write_mdp`, `_write_conf_gro`, `_box_line`, `_grompp_log_text`, `_log_grompp_warnings`, `collect_torsion_set`.
- Produces:
  - `ffrefine.mm.gromacs.TprDump(text: str, path: Path, gromacs_version: str)`
  - `preprocess_topology(binaries, topology_path, structure, workdir, stem) -> tuple[Path, Path]`: returns `(tpr, flattened_top)`, written as `<stem>_resolve.tpr` and `<stem>_pp.top`
  - `dump_tpr(binaries, tpr, workdir, stem) -> TprDump`: stdout kept as `<stem>.stdout`
  - `ffrefine.mm.tprparams.write_full_system(topology, workdir, stem, ff_includes=None) -> Path`: writes `<stem>.itp` and `<stem>.top`, returns the `.top`
  - `resolve_torsion_set(binaries, topology, quad, structure, workdir, ff_includes=None) -> ResolvedTorsionSet`: files `full.itp`, `full.top`, `full_pp.top`, `full_resolve.tpr`, `full_dump.stdout` in `workdir`
  - `preprocessed_sha256(binaries, topology, structure, workdir, stem, ff_includes=None) -> str`

- [ ] **Step 1: Write the failing live tests**

Append to `tests/test_tprparams.py` (add `import shutil`, and the imports below):

```python
from ffrefine.config import Binaries
from ffrefine.io.structure import read_pdb
from ffrefine.io.topology import read_topology
from ffrefine.mm.torsionset import collect_torsion_set
from ffrefine.mm.tprparams import preprocessed_sha256, resolve_torsion_set

CAPECITABINE = FIXTURES / "capecitabine_gromacs"


def _quad_of(d: Dihedral) -> tuple[int, int, int, int]:
    return (d.ai, d.aj, d.ak, d.al)


@pytest.mark.needs_gmx
def test_pfoa_typed_torsion_set_resolves_from_dihedraltypes(tmp_path, gmx_bin):
    t = read_topology(PFOA / "pfoa_gmx.top")
    s = read_pdb(PFOA / "pfoa_gmx.pdb")
    r = resolve_torsion_set(Binaries(gmx=gmx_bin), t, (25, 1, 3, 4), s, tmp_path)
    assert len(r.terms) == 20
    assert {x.source for x in r.terms} == {SOURCE_DIHEDRALTYPES}
    target = [x.dihedral for x in r.terms if _quad_of(x.dihedral) == (25, 1, 3, 4)]
    assert [d.params[2] for d in target] == [1.0, 2.0, 3.0, 6.0]
    assert [d.params[1] for d in target] == pytest.approx(
        [0.66944, 0.46024, 0.33472, 1.046], rel=1e-6
    )
    assert r.defaults == Defaults(1, 2, True, 1.0, 1.0)
    assert re.fullmatch(r"[0-9a-f]{64}", r.pp_sha256)
    assert r.gromacs_version != "unknown"
    assert r.full_top == tmp_path / "full.top"
    assert r.dump_path == tmp_path / "full_dump.stdout" and r.dump_path.is_file()
    ff = (PFOA / "charmm36.ff").resolve()
    assert f'#include "{ff / "forcefield.itp"}"' in (tmp_path / "full.itp").read_text()


@pytest.mark.needs_gmx
def test_capecitabine_zero_k_lines_resolve_to_no_terms(tmp_path, gmx_bin):
    """6 7 9 37 and 8 7 9 37 are typed, and every CGenFF term for them has k = 0."""
    t = read_topology(CAPECITABINE / "capecitabine_gmx.top")
    s = read_pdb(CAPECITABINE / "capecitabine_gmx.pdb")
    r = resolve_torsion_set(Binaries(gmx=gmx_bin), t, (6, 7, 9, 10), s, tmp_path)
    assert len(collect_torsion_set(t, (6, 7, 9, 10))) == 4
    assert [(_quad_of(x.dihedral), x.dihedral.params[2]) for x in r.terms] == [
        ((6, 7, 9, 10), 1.0),
        ((6, 7, 9, 10), 2.0),
        ((8, 7, 9, 10), 2.0),
    ]
    assert [x.dihedral.params[1] for x in r.terms] == pytest.approx(
        [6.6944, 10.46, 8.368], rel=1e-6
    )


@pytest.mark.needs_gmx
def test_explicit_butane_lines_pass_the_dump_cross_check(tmp_path, butane_itp, butane_pdb, gmx_bin):
    """Every butane term is explicit: the cross-check runs and changes nothing."""
    t = read_topology(butane_itp)
    r = resolve_torsion_set(Binaries(gmx=gmx_bin), t, (1, 5, 8, 11), read_pdb(butane_pdb), tmp_path)
    assert {x.source for x in r.terms} == {SOURCE_TOPOLOGY}
    assert r.dihedrals == collect_torsion_set(t, (1, 5, 8, 11))


@pytest.mark.needs_gmx
def test_preprocessed_sha256_tracks_included_files(tmp_path, gmx_bin):
    """Editing a file only the .top's includes reach must change the digest."""
    work = tmp_path / "pfoa"
    shutil.copytree(PFOA, work)
    t = read_topology(work / "pfoa_gmx.top")
    s = read_pdb(work / "pfoa_gmx.pdb")
    binaries = Binaries(gmx=gmx_bin)
    first = preprocessed_sha256(binaries, t, s, tmp_path / "a", "before")
    assert preprocessed_sha256(binaries, t, s, tmp_path / "b", "before") == first
    ffbonded = work / "charmm36.ff" / "pfo_ffbonded.itp"
    ffbonded.write_text(ffbonded.read_text().replace("0.669440", "0.700000", 1))
    assert preprocessed_sha256(binaries, t, s, tmp_path / "c", "before") != first


@pytest.mark.needs_gmx
def test_a_missing_include_names_the_base_it_was_resolved_against(tmp_path, gmx_bin):
    """Design section 7: grompp's error, plus where ffrefine looked and why there."""
    from ffrefine.errors import ExternalToolError

    lonely = tmp_path / "lonely"
    lonely.mkdir()
    shutil.copyfile(PFOA / "pfoa_gmx.top", lonely / "pfoa_gmx.top")  # no charmm36.ff/ beside it
    t = read_topology(lonely / "pfoa_gmx.top")
    s = read_pdb(PFOA / "pfoa_gmx.pdb")
    with pytest.raises(ExternalToolError) as excinfo:
        resolve_torsion_set(Binaries(gmx=gmx_bin), t, (25, 1, 3, 4), s, tmp_path / "wd")
    msg = str(excinfo.value)
    assert 'Topology include file "./charmm36.ff/forcefield.itp" not found' in msg
    assert f"against {lonely.resolve()} (the directory of" in msg
```

- [ ] **Step 2: Run them to verify they fail**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_tprparams.py -q -rs'`
Expected: `ImportError: cannot import name 'preprocessed_sha256'`.

- [ ] **Step 3: Add the grompp and dump runners to `ffrefine/mm/gromacs.py`**

Add `import re` to the imports. Then, below `single_point_energies`:

```python
@dataclass
class TprDump:
    """``gmx dump -s`` output, where it is on disk, and the GROMACS that wrote it."""

    text: str
    path: Path
    gromacs_version: str


_GMX_BANNER_RE = re.compile(r":-\)\s+GROMACS - gmx \w+, (\S+)\s+\(-:")


def preprocess_topology(
    binaries: Binaries,
    topology_path: str | Path,
    structure: Structure,
    workdir: str | Path,
    stem: str,
) -> tuple[Path, Path]:
    """Run grompp once for GROMACS's own view of a topology (milestone 6a, section 4).

    Writes ``<stem>_pp.top`` (``grompp -pp``: every include expanded, every
    ``#ifdef`` resolved, exactly as grompp read it) and ``<stem>_resolve.tpr``.
    Uses the large-box fallback mdp: bonded parameters do not depend on the
    mdp, and every supported GROMACS accepts that one.

    Args:
        binaries: resolved binaries; ``gmx`` required.
        topology_path: a ``.top`` grompp can process.
        structure: coordinates and naming for the conf file.
        workdir: directory for every file this writes, logs included.
        stem: file-name prefix.

    Returns:
        ``(tpr, flattened_topology)``.

    Raises:
        ExternalToolError: grompp failed; its logs are named.
    """
    gmx = binaries.require("gmx")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    mdp = workdir / f"{stem}_resolve.mdp"
    write_mdp(mdp, fallback=True)
    conf = workdir / f"{stem}_resolve_conf.gro"
    _write_conf_gro(conf, structure, structure.coords, box=_box_line(True))
    tpr = workdir / f"{stem}_resolve.tpr"
    flat = workdir / f"{stem}_pp.top"
    grompp_stem = f"{stem}_resolve_grompp"
    run_command(
        [
            gmx,
            "grompp",
            "-f",
            mdp.name,
            "-c",
            conf.name,
            "-p",
            str(Path(topology_path).resolve()),
            "-pp",
            flat.name,
            "-po",
            f"{stem}_resolve_mdout.mdp",
            "-o",
            tpr.name,
            "-maxwarn",
            "5",
        ],
        workdir,
        grompp_stem,
    )
    _log_grompp_warnings(_grompp_log_text(workdir, grompp_stem))
    return tpr, flat


def dump_tpr(binaries: Binaries, tpr: str | Path, workdir: str | Path, stem: str) -> TprDump:
    """``gmx dump -s tpr``; the text is also kept on disk as ``<stem>.stdout``.

    Raises:
        ExternalToolError: gmx dump failed.
    """
    gmx = binaries.require("gmx")
    proc = run_command([gmx, "dump", "-s", str(Path(tpr).resolve())], workdir, stem)
    banner = _GMX_BANNER_RE.search((proc.stderr or "") + (proc.stdout or ""))
    return TprDump(
        text=proc.stdout or "",
        path=Path(workdir) / f"{stem}.stdout",
        gromacs_version=banner.group(1) if banner else "unknown",
    )
```

- [ ] **Step 4: Add the orchestration to `ffrefine/mm/tprparams.py`**

Add to the imports:

```python
import hashlib

from ..config import Binaries
from ..io.structure import Structure
from ..io.topology import Topology
from ..io.topwriter import make_system_top, write_molecule_copy
from .gromacs import dump_tpr, preprocess_topology
from .torsionset import collect_torsion_set
```

(merge `Topology` into the existing `from ..io.topology import ...` line, and change `from ..errors import ValidationError` to `from ..errors import ExternalToolError, ValidationError`.) Then append:

```python
def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _preprocess(
    binaries: Binaries, topology: Topology, top: Path, structure: Structure, workdir, stem: str
) -> tuple[Path, Path]:
    """:func:`preprocess_topology`, naming the include base when an include is missing.

    grompp's "Topology include file ... not found" says what it could not
    open, not where ffrefine pointed it; design section 7 asks for both.
    """
    try:
        return preprocess_topology(binaries, top, structure, workdir, stem)
    except ExternalToolError as exc:
        if "include file" not in str(exc):
            raise
        how = (
            f"named by the ffrefine include-base marker in {topology.source}"
            if topology.include_marker is not None
            else f"the directory of {topology.source}"
        )
        raise ExternalToolError(
            f"{exc}\n  ffrefine resolved the relative #include lines of "
            f"{Path(topology.source).name} against {topology.include_base} ({how}). An "
            "include with no file there is left for grompp's own search, which is what "
            "failed above."
        ) from exc


def write_full_system(
    topology: Topology, workdir: str | Path, stem: str, ff_includes: list[str] | None = None
) -> Path:
    """Write ``<stem>.itp`` (the unmodified topology, includes absolute) and ``<stem>.top``.

    Returns:
        The wrapper ``.top`` to hand to grompp.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    itp = workdir / f"{stem}.itp"
    write_molecule_copy(topology.source, itp)
    top = workdir / f"{stem}.top"
    make_system_top(itp, top, topology.name, ff_includes)
    return top


def resolve_torsion_set(
    binaries: Binaries,
    topology: Topology,
    quad: Quad,
    structure: Structure,
    workdir: str | Path,
    ff_includes: list[str] | None = None,
) -> ResolvedTorsionSet:
    """Give every term of T explicit parameters, as grompp built them.

    Writes ``full.itp``/``full.top`` (the unmodified topology), runs
    ``grompp -pp`` and ``gmx dump`` on it, and cross-checks the result
    against T (design section 6.3). About a second on PFOA with CHARMM36.

    Args:
        binaries: ``gmx`` required.
        topology: the unmodified topology.
        quad: 1-based target quadruplet; only its central bond matters.
        structure: for the conf file.
        workdir: stage directory.
        ff_includes: ``--ff-include`` values for the wrapper.

    Raises:
        ExternalToolError: grompp or gmx dump failed.
        ValidationError: the dump could not be read, or the cross-check
            refused (see :func:`cross_check`).
    """
    full_top = write_full_system(topology, workdir, "full", ff_includes)
    tpr, flat = _preprocess(binaries, topology, full_top, structure, workdir, "full")
    dump = dump_tpr(binaries, tpr, workdir, "full_dump")
    try:
        dumped = parse_dump_torsions(dump.text, topology.name, topology.natoms, dump.path)
    except ValidationError as exc:
        raise ValidationError(f"{exc} (GROMACS {dump.gromacs_version})") from exc
    terms = cross_check(collect_torsion_set(topology, quad), dumped, dump.path)
    n_typed = sum(t.source == SOURCE_DIHEDRALTYPES for t in terms)
    log.info(
        "torsion set about bond (%d, %d): %d term(s), %d of them from [ dihedraltypes ] "
        "as grompp built them (%s)",
        quad[1],
        quad[2],
        len(terms),
        n_typed,
        dump.path.name,
    )
    return ResolvedTorsionSet(
        terms=terms,
        defaults=parse_defaults(flat.read_text(), flat),
        pp_sha256=_sha256(flat),
        full_top=full_top,
        dump_path=dump.path,
        gromacs_version=dump.gromacs_version,
    )


def preprocessed_sha256(
    binaries: Binaries,
    topology: Topology,
    structure: Structure,
    workdir: str | Path,
    stem: str,
    ff_includes: list[str] | None = None,
) -> str:
    """SHA-256 of grompp's preprocessed view of ``topology``.

    Covers every file grompp opens -- force-field includes too -- which the
    ``--resume`` key of a stage that runs grompp must (design section 4).
    Writes ``<stem>.itp``, ``<stem>.top`` and ``<stem>_pp.top`` into ``workdir``.
    """
    top = write_full_system(topology, workdir, stem, ff_includes)
    _, flat = _preprocess(binaries, topology, top, structure, workdir, stem)
    return _sha256(flat)
```

- [ ] **Step 5: Run the tests**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_tprparams.py -q -rs'`
Expected: all pass, 0 skipped. If `test_preprocessed_sha256_tracks_included_files` fails on its *first* assertion (two runs differing), grompp embedded something run-specific in `_pp.top`: diff `a/before_pp.top` against `b/before_pp.top` and report it; do not weaken the test.

- [ ] **Step 6: black and commit**

```bash
wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m black ffrefine/mm/gromacs.py ffrefine/mm/tprparams.py tests/test_tprparams.py'
cd /c/Users/hrach/Projects/ffrefine
git add ffrefine/mm/gromacs.py ffrefine/mm/tprparams.py tests/test_tprparams.py
git commit -m "feat(mm): resolve a torsion set through grompp -pp and gmx dump

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: The self-check takes resolved parameters

**Files:**
- Modify: `ffrefine/mm/selfcheck.py` (module docstring; delete `_reject_typed_terms`; `check_torsion_removal` gains `resolved`; failure message)
- Modify: `tests/conftest.py` (add the `rotate_side` fixture)
- Modify: `tests/test_selfcheck.py` (delete two refusal tests, extend one, add four live tests)

**Interfaces:**
- Consumes: Task 5's `ResolvedTorsionSet`, `resolve_torsion_set`, `SOURCE_DIHEDRALTYPES`; Task 3's `write_zeroed_topology` (absolute).
- Produces: `check_torsion_removal(binaries, topology, quad, structure, frames, workdir, nthreads=1, tolerance=DEFAULT_TOLERANCE, strict_names=True, ff_includes=None, resolved: ResolvedTorsionSet | None = None) -> SelfCheckResult`. With `resolved=None` it resolves in `workdir` itself; `mmscan` (Task 7) passes its own. The "full" energies come from `resolved.full_top`.

- [ ] **Step 1: Add the `rotate_side` fixture to `tests/conftest.py`**

```python
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
```

- [ ] **Step 2: Write the failing tests**

In `tests/test_selfcheck.py`, **delete** `test_typed_dihedrals_are_refused_before_gromacs_runs` and `test_typed_dihedral_message_counts_only_the_untyped_ones`: the refusal they pin is what this task removes. Keep `TARGET_BLOCK`, `TYPED_BLOCK`, `_typed_topology` and `DIHEDRALTYPES_SECTION`; the new tests use them.

Replace the body of `test_self_check_raises_when_tolerance_impossible` from the `with pytest.raises` line on with:

```python
    with pytest.raises(ValidationError, match="self-consistency") as excinfo:
        selfcheck.check_torsion_removal(
            Binaries(gmx=gmx_bin),
            t,
            (1, 5, 8, 11),
            s,
            frames,
            tmp_path,
            tolerance=1e-18,
        )
    msg = str(excinfo.value)
    assert "10 term(s): 10 written in the topology and 0 resolved from [ dihedraltypes ]" in msg
    assert "full_dump.stdout" in msg
    # The old claim is gone: typed terms are no longer refused before the run.
    assert "checked before the run" not in msg
```

Add near the top, after the imports:

```python
from pathlib import Path

FIXTURES = Path(__file__).parent.parent / "fixtures"
PFOA = FIXTURES / "pfoa_gromacs"
CAPECITABINE = FIXTURES / "capecitabine_gromacs"
```

Append the live tests:

```python
@pytest.mark.needs_gmx
def test_typed_butane_resolves_through_a_dihedraltypes_entry(
    tmp_path, butane_pdb, butane_itp, gmx_bin, rotated_frames
):
    """Every term of T typed, matched by one wildcard entry: 9 quadruplets x 20 kJ/mol."""
    from ffrefine.config import Binaries

    text = _typed_topology(tmp_path, butane_itp).read_text()
    itp = tmp_path / "typed_with_types.itp"
    itp.write_text(text.replace("[ moleculetype ]", DIHEDRALTYPES_SECTION, 1), newline="\n")
    s = read_pdb(butane_pdb)
    t = read_topology(itp)
    r = selfcheck.check_torsion_removal(
        Binaries(gmx=gmx_bin), t, (1, 5, 8, 11), s, rotated_frames(s, 8), tmp_path / "wd"
    )
    assert r.passed and r.max_abs_error < 1e-3
    assert np.abs(r.analytic_difference).max() > 50.0  # the 20 kJ/mol type is really felt


@pytest.mark.needs_gmx
def test_typed_butane_without_matching_types_fails_in_grompp(tmp_path, butane_pdb, butane_itp, gmx_bin):
    """No [ dihedraltypes ] entry for the typed lines: grompp's own error, before any rerun."""
    from ffrefine.config import Binaries
    from ffrefine.errors import ExternalToolError

    s = read_pdb(butane_pdb)
    t = read_topology(_typed_topology(tmp_path, butane_itp))
    with pytest.raises(ExternalToolError):
        selfcheck.check_torsion_removal(
            Binaries(gmx=gmx_bin), t, (1, 5, 8, 11), s, s.coords[None, :, :], tmp_path / "wd"
        )
    assert not (tmp_path / "wd" / "full").exists()  # no rerun was started


@pytest.mark.needs_gmx
def test_self_check_passes_on_pfoa_typed_torsions(tmp_path, gmx_bin, rotate_side):
    """The real CGenFF case: 20 terms about C1-C3, all from [ dihedraltypes ]."""
    from ffrefine.config import Binaries

    s = read_pdb(PFOA / "pfoa_gmx.pdb")
    t = read_topology(PFOA / "pfoa_gmx.top")
    # Turn the carboxyl group (atoms 1, 2, 25, 26) a full circle about C1-C3.
    frames = rotate_side(s.coords, t.bonds, fixed=3, moving=1, angles_deg=np.arange(0.0, 360.0, 30.0))
    r = selfcheck.check_torsion_removal(Binaries(gmx=gmx_bin), t, (25, 1, 3, 4), s, frames, tmp_path)
    assert r.passed and r.max_abs_error < 1e-3
    assert np.ptp(r.analytic_difference) > 1.0


@pytest.mark.needs_gmx
def test_self_check_passes_on_capecitabine_typed_torsions(tmp_path, gmx_bin, rotate_side):
    """Carbamate C7-N9, where two of the four typed lines resolve to no terms."""
    from ffrefine.config import Binaries

    s = read_pdb(CAPECITABINE / "capecitabine_gmx.pdb")
    t = read_topology(CAPECITABINE / "capecitabine_gmx.top")
    # Swing the pentyl carbamate side +-30 degrees; wider sweeps drive it into the rings.
    frames = rotate_side(s.coords, t.bonds, fixed=9, moving=7, angles_deg=np.linspace(-30.0, 30.0, 7))
    r = selfcheck.check_torsion_removal(Binaries(gmx=gmx_bin), t, (6, 7, 9, 10), s, frames, tmp_path)
    assert r.passed and r.max_abs_error < 1e-3
    assert np.ptp(r.analytic_difference) > 1.0
```

- [ ] **Step 3: Run them to verify they fail**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_selfcheck.py -q -rs'`
Expected: the four new tests fail with `ValidationError: cannot run the spec 4.5 self-consistency check ... carry no explicit parameters` (the old refusal), and the tolerance test fails on its new message assertions.

- [ ] **Step 4: Rewire `ffrefine/mm/selfcheck.py`**

Imports: drop `write_molecule_copy` from the `..io.topwriter` import and drop `from .torsionset import collect_torsion_set` (both become unused); add

```python
from .tprparams import SOURCE_DIHEDRALTYPES, ResolvedTorsionSet, resolve_torsion_set
```

In the module docstring, replace the paragraph that begins "One case that used to reach that failure" with:

```
Typed torsions -- terms whose parameters live in [ dihedraltypes ], the normal
shape of a CGenFF or GAFF molecule topology -- get their parameters from
:func:`ffrefine.mm.tprparams.resolve_torsion_set`, which reads back the terms
grompp itself built. The check then verifies what it always has: that the
zeroed copy removed exactly those terms and nothing else. Explicit parameters
on a zeroed line do suppress the [ dihedraltypes ] lookup; that is verified,
not assumed, by
``tests/test_selfcheck.py::test_dihedraltypes_entry_is_suppressed_by_explicit_params``.
```

Delete `_reject_typed_terms` entirely.

In `check_torsion_removal`: add the parameter after `ff_includes` and document it:

```python
    ff_includes: list[str] | None = None,
    resolved: ResolvedTorsionSet | None = None,
) -> SelfCheckResult:
```

```
        resolved: T with every term's parameters as grompp built them
            (:func:`ffrefine.mm.tprparams.resolve_torsion_set`). ``mmscan``
            resolves before its ``--resume`` check and passes the result in;
            when omitted it is resolved here, in ``workdir``. Its
            ``full_top`` is the "full" system this check runs.
```

Replace everything from the `_reject_typed_terms(topology, quad)` call (and its three-line comment above) down to and including the `e_zero = single_point_energies(...)` call with:

```python
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if resolved is None:
        resolved = resolve_torsion_set(binaries, topology, quad, structure, workdir, ff_includes)

    zero_itp = workdir / "zeroed.itp"
    removed = write_zeroed_topology(topology, quad, zero_itp)
    zero_top = workdir / "zeroed.top"
    make_system_top(zero_itp, zero_top, topology.name, ff_includes)

    e_full = single_point_energies(
        binaries, resolved.full_top, structure, frames, workdir / "full", "full", nthreads
    )
    e_zero = single_point_energies(
        binaries, zero_top, structure, frames, workdir / "zeroed", "zeroed", nthreads
    )
```

Change the analytic line to use the resolved terms:

```python
    analytic_difference = torsion_set_energy(resolved.dihedrals, gmx_frames)
```

Replace the failure message (the `raise ValidationError(...)` inside `if not passed:`) with:

```python
        n_typed = sum(t.source == SOURCE_DIHEDRALTYPES for t in resolved.terms)
        raise ValidationError(
            "torsion removal self-consistency check FAILED (spec section 4.5).\n"
            f"  maximum |E_full - E_zeroed - analytic| = {max_abs_error:.6g} kJ/mol "
            f"(tolerance {tolerance:g})\n"
            f"  worst frame {worst}, phi = {phi[worst]:.2f} deg:\n"
            f"    GROMACS difference : {gromacs_difference[worst]:.6f} kJ/mol\n"
            f"    analytic reference : {analytic_difference[worst]:.6f} kJ/mol\n"
            f"  {len(removed)} dihedral term-line(s) were zeroed about the central bond "
            f"({quad[1]}, {quad[2]}). The analytic reference sums {len(resolved.terms)} "
            f"term(s): {len(resolved.terms) - n_typed} written in the topology and {n_typed} "
            "resolved from [ dihedraltypes ], as grompp built them (read back from "
            f"{resolved.dump_path}).\n"
            "  The topology edit did not do what was intended.\n"
            f"  Inspect {zero_itp}, {resolved.dump_path} and the GROMACS logs under "
            f"{workdir}. Fitting is aborted."
        )
```

- [ ] **Step 5: Run the tests**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_selfcheck.py tests/test_roundtrip.py tests/test_gromacs.py -q -rs'`
Expected: all pass, 0 skipped. If a PFOA or capecitabine self-check fails with a residual above 1e-3, **do not raise the tolerance**: report the residual, the worst frame and `full_dump.stdout`.

- [ ] **Step 6: black and commit**

```bash
wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m black ffrefine/mm/selfcheck.py tests/conftest.py tests/test_selfcheck.py'
cd /c/Users/hrach/Projects/ffrefine
git add ffrefine/mm/selfcheck.py tests/conftest.py tests/test_selfcheck.py
git commit -m "feat(mm): self-check typed torsions with the parameters grompp built

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: `mmscan` resolves first, keys on it, and records it

**Files:**
- Modify: `ffrefine/cli.py` (imports; `_write_input_stage`; `cmd_mmscan`; new helpers `_torsion_record`, `_defaults_record`)
- Test: `tests/test_cli_typed.py` (append)

**Interfaces:**
- Consumes: Task 2's `has_includes`; Task 5's `resolve_torsion_set`, `ResolvedTerm`; Task 6's `check_torsion_removal(..., resolved=...)`.
- Produces, in `02_mm_scan/energies.json`:
  - `"torsion_parameters"`: list of `{"quad_1_based": [i, j, k, l], "funct": int, "phi0_deg": float, "k_kj_per_mol": float, "multiplicity": int, "source": "topology" | "dihedraltypes"}`, one per term, in topology order;
  - `"defaults"`: `{"fudgeQQ": float, "fudgeLJ": float, "gen_pairs": bool, "source": "grompp -pp"}` or `null`;
  - `"preprocessed_topology_sha256"`: str.
  - The mmscan `--resume` key gains `"preprocessed_topology_sha256"` in its settings.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli_typed.py` (add `import shutil`, `import pytest`, and these imports):

```python
from ffrefine.io.topwriter import write_charged_topology
from ffrefine.mm.selfcheck import SelfCheckResult
from ffrefine.mm.torsionset import collect_torsion_set
from ffrefine.mm.tprparams import SOURCE_TOPOLOGY, ResolvedTerm, ResolvedTorsionSet

NO_DEFAULTS_WARNING = "has no [ defaults ] directive"


def _without_defaults(butane_itp: Path, tmp_path: Path) -> Path:
    """butane.itp minus its [ defaults ] block, and with no #include either."""
    stripped, n = re.subn(r"\[ defaults \]\n;[^\n]*\n[^\n]*\n", "", butane_itp.read_text())
    assert n == 1, "butane fixture layout changed"
    out = tmp_path / "no_defaults.itp"
    out.write_text(stripped, newline="\n")
    return out


def test_model_json_defers_the_defaults_warning_for_an_included_force_field(tmp_path, capsys):
    """PFOA's [ defaults ] is in charmm36.ff/forcefield.itp: mmscan reads it, not 00_input."""
    out = tmp_path / "out"
    assert main(["fit"] + _pfoa_args(out)) == 1  # no scan yet; 00_input is written first
    assert NO_DEFAULTS_WARNING not in capsys.readouterr().err
    assert json.loads((out / "00_input" / "model.json").read_text())["fudgeQQ"] is None


def test_model_json_still_warns_with_neither_defaults_nor_includes(
    tmp_path, butane_pdb, butane_itp, capsys
):
    itp = _without_defaults(butane_itp, tmp_path)
    argv = ["fit", "-s", str(butane_pdb), "-p", str(itp), "--dihedral", "1,5,8,11"]
    assert main(argv + ["--outdir", str(tmp_path / "out")]) == 1
    assert NO_DEFAULTS_WARNING in capsys.readouterr().err


def test_model_json_defers_the_warning_for_ff_include(tmp_path, butane_pdb, butane_itp, capsys):
    itp = _without_defaults(butane_itp, tmp_path)
    ff = tmp_path / "ff.itp"
    ff.write_text("[ defaults ]\n1 2 yes 1.0 1.0\n")
    argv = ["fit", "-s", str(butane_pdb), "-p", str(itp), "--dihedral", "1,5,8,11"]
    argv += ["--ff-include", str(ff), "--outdir", str(tmp_path / "out")]
    assert main(argv) == 1
    assert NO_DEFAULTS_WARNING not in capsys.readouterr().err


def _fake_resolve(sha: str):
    def fake(binaries, topology, quad, structure, workdir, ff_includes=None):
        return ResolvedTorsionSet(
            terms=[ResolvedTerm(d, SOURCE_TOPOLOGY) for d in collect_torsion_set(topology, quad)],
            defaults=topology.defaults,
            pp_sha256=sha,
            full_top=Path(workdir) / "full.top",
            dump_path=Path(workdir) / "full_dump.stdout",
            gromacs_version="test",
        )

    return fake


def _fake_check(calls: list):
    def fake(*args, **kwargs):
        calls.append(kwargs.get("resolved"))
        one = np.zeros(1)
        return SelfCheckResult(
            max_abs_error=0.0,
            passed=True,
            gromacs_difference=one,
            analytic_difference=one,
            phi_deg=one,
            tolerance=1e-3,
            full_potential_kj=one,
            zeroed_potential_kj=one,
        )

    return fake


def test_mmscan_resume_key_covers_the_preprocessed_topology(
    tmp_path, butane_pdb, butane_itp, monkeypatch
):
    out = tmp_path / "out"
    argv = ["mmscan", "-s", str(butane_pdb), "-p", str(butane_itp), "--dihedral", "1,5,8,11"]
    argv += ["--outdir", str(out)]
    calls: list = []
    monkeypatch.setattr(cli, "check_torsion_removal", _fake_check(calls))
    monkeypatch.setattr(cli, "resolve_torsion_set", _fake_resolve("a" * 64))
    assert main(argv) == 0 and len(calls) == 1
    assert calls[0].pp_sha256 == "a" * 64  # the check gets mmscan's resolution, not its own
    assert main(argv + ["--resume"]) == 0 and len(calls) == 1  # same key: skipped
    monkeypatch.setattr(cli, "resolve_torsion_set", _fake_resolve("b" * 64))
    assert main(argv + ["--resume"]) == 0 and len(calls) == 2  # an included file changed

    payload = json.loads((out / "02_mm_scan" / "energies.json").read_text())
    assert payload["preprocessed_topology_sha256"] == "b" * 64
    assert len(payload["torsion_parameters"]) == 10
    assert payload["torsion_parameters"][0] == {
        "quad_1_based": [1, 5, 8, 11],
        "funct": 9,
        "phi0_deg": 0.0,
        "k_kj_per_mol": 0.58576,
        "multiplicity": 3,
        "source": "topology",
    }
    assert payload["defaults"] == {
        "fudgeQQ": 1.0,
        "fudgeLJ": 1.0,
        "gen_pairs": True,
        "source": "grompp -pp",
    }


@pytest.mark.needs_gmx
def test_mmscan_on_pfoa_records_grompps_parameters(tmp_path, gmx_bin, capsys):
    out = tmp_path / "out"
    assert main(["mmscan"] + _pfoa_args(out)) == 0
    assert NO_DEFAULTS_WARNING not in capsys.readouterr().err
    payload = json.loads((out / "02_mm_scan" / "energies.json").read_text())
    params = payload["torsion_parameters"]
    assert len(params) == 20
    assert {p["source"] for p in params} == {"dihedraltypes"}
    target = [p for p in params if p["quad_1_based"] == list(PFOA_QUAD)]
    assert [p["multiplicity"] for p in target] == [1, 2, 3, 6]
    assert payload["defaults"] == {
        "fudgeQQ": 1.0,
        "fudgeLJ": 1.0,
        "gen_pairs": True,
        "source": "grompp -pp",
    }
    assert payload["selfcheck_max_abs_error_kj"] < 1e-3


@pytest.mark.needs_gmx
def test_mmscan_on_a_marked_copy_finds_the_force_field(tmp_path, gmx_bin):
    """A charge-refined topology written elsewhere still finds charmm36.ff/ (R28's scenario)."""
    moved = tmp_path / "elsewhere"
    moved.mkdir()
    write_charged_topology(PFOA / "pfoa_gmx.top", [], moved / "molecule_charged.itp")
    out = tmp_path / "out"
    assert main(["mmscan"] + _pfoa_args(out, top=moved / "molecule_charged.itp")) == 0
    assert f'#include "{PFOA_FF / "forcefield.itp"}"' in _include_lines(
        out / "02_mm_scan" / "full.itp"
    )


@pytest.mark.needs_gmx
def test_mmscan_resume_reruns_when_an_included_file_changes(tmp_path, gmx_bin, capsys):
    work = tmp_path / "pfoa"
    shutil.copytree(PFOA, work)
    out = tmp_path / "out"
    argv = ["mmscan", "-s", str(work / "pfoa_gmx.pdb"), "-p", str(work / "pfoa_gmx.top")]
    argv += ["--dihedral", "25,1,3,4", "--outdir", str(out)]
    assert main(argv) == 0
    capsys.readouterr()
    assert main(argv + ["--resume"]) == 0
    assert "skipping" in capsys.readouterr().out

    # The first 0.669440 in pfo_ffbonded.itp is OG2D1 CG2O2 CG312 CG312, multiplicity 2:
    # quad 2 1 3 4, which is in the torsion set about C1-C3.
    ffbonded = work / "charmm36.ff" / "pfo_ffbonded.itp"
    ffbonded.write_text(ffbonded.read_text().replace("0.669440", "0.700000", 1))
    assert main(argv + ["--resume"]) == 0
    assert "skipping" not in capsys.readouterr().out
    payload = json.loads((out / "02_mm_scan" / "energies.json").read_text())
    ks = [p["k_kj_per_mol"] for p in payload["torsion_parameters"] if p["quad_1_based"] == [2, 1, 3, 4]]
    assert ks[1] == pytest.approx(0.7, rel=1e-6)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_cli_typed.py -q -rs'`
Expected: `test_model_json_defers_*` fail (the warning is still printed); `test_mmscan_resume_key_*` fails with `AttributeError: <module 'ffrefine.cli'> does not have the attribute 'resolve_torsion_set'`; the live tests fail on the missing payload keys.

- [ ] **Step 3: Wire it into `ffrefine/cli.py`**

Imports: add `from .io.includes import has_includes`; change `from .io.topology import Topology, read_topology` to `from .io.topology import Defaults, Topology, read_topology`; add `from .mm.tprparams import ResolvedTerm, resolve_torsion_set`.

In `_write_input_stage`, replace the `if defaults is None:` block with:

```python
    if defaults is None:
        if has_includes(topology.raw_lines) or _ff_includes(args):
            # grompp reads the includes; mmscan asks it (milestone 6a design, section 4).
            log.info(
                "%s: [ defaults ] is not in this file itself; mmscan reads it from grompp's "
                "preprocessed topology and records it in 02_mm_scan/energies.json",
                input_topology,
            )
        else:
            _warn(
                f"{input_topology} has no [ defaults ] directive, so fudgeQQ/fudgeLJ could not "
                "be read. The fitted torsion is only valid with the 1-4 conventions it was "
                "fitted against (spec section 9.3); record them yourself."
            )
```

Add two helpers above `cmd_mmscan`:

```python
def _torsion_record(term: ResolvedTerm) -> dict:
    """One ``energies.json`` ``torsion_parameters`` entry (milestone 6a design, section 4)."""
    d = term.dihedral
    return {
        "quad_1_based": [d.ai, d.aj, d.ak, d.al],
        "funct": d.funct,
        "phi0_deg": float(d.params[0]),
        "k_kj_per_mol": float(d.params[1]),
        "multiplicity": int(round(float(d.params[2]))),
        "source": term.source,
    }


def _defaults_record(defaults: Defaults | None) -> dict | None:
    """``[ defaults ]`` as grompp read it, for ``energies.json``."""
    if defaults is None:
        return None
    return {
        "fudgeQQ": defaults.fudge_qq,
        "fudgeLJ": defaults.fudge_lj,
        "gen_pairs": defaults.gen_pairs,
        "source": "grompp -pp",
    }
```

In `cmd_mmscan`, replace everything from `ff_includes = _ff_includes(args)` down to and including the `result = check_torsion_removal(...)` call with:

```python
    ff_includes = _ff_includes(args)
    binaries = _binaries(args)
    workdir = stage_dir(args.outdir, "mm_scan")
    # Resolved before the --resume check: the key must cover every file grompp
    # reads, and only grompp knows which files those are (milestone 6a design,
    # section 4). About a second, even when the stage is then skipped.
    resolved = resolve_torsion_set(binaries, topology, quad, structure, workdir, ff_includes)
    settings = {
        "quad": list(quad),
        "nframes": int(frames.shape[0]),
        "ff_includes": ff_includes,
        "preprocessed_topology_sha256": resolved.pp_sha256,
    }
    digest = hash_inputs(inputs, settings)
    if args.resume and stage_is_current(
        args.outdir, "mm_scan", digest, _stage_outputs(args, "mm_scan")
    ):
        print(f"mmscan: skipping, {_energies_path(args)} is current (--resume)")
        return 0

    result = check_torsion_removal(
        binaries,
        topology,
        quad,
        structure,
        frames,
        workdir,
        args.nthreads,
        strict_names=not args.no_strict_names,
        ff_includes=ff_includes,
        resolved=resolved,
    )
```

After the `print(... self-consistency check passed ...)` call, add:

```python
    if resolved.defaults is None:
        _warn(
            f"grompp's preprocessed topology {workdir / 'full_pp.top'} has no [ defaults ] "
            "directive, so fudgeQQ/fudgeLJ could not be recorded. The fitted torsion is only "
            "valid with the 1-4 conventions it was fitted against (spec section 9.3); record "
            "them yourself."
        )
```

and add three keys at the end of the `payload` dict, after `"selfcheck_tolerance_kj"`:

```python
        # Milestone 6a: the parameters the self-check used, as grompp built them,
        # with where each came from; milestone 6b's report reads its "before"
        # values here. And [ defaults ] as grompp read it, include or not.
        "torsion_parameters": [_torsion_record(t) for t in resolved.terms],
        "defaults": _defaults_record(resolved.defaults),
        "preprocessed_topology_sha256": resolved.pp_sha256,
```

- [ ] **Step 4: Run the tests**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_cli_typed.py tests/test_cli.py -q -rs'`
Expected: all pass, 0 skipped.

- [ ] **Step 5: black and commit**

```bash
wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m black ffrefine/cli.py tests/test_cli_typed.py'
cd /c/Users/hrach/Projects/ffrefine
git add ffrefine/cli.py tests/test_cli_typed.py
git commit -m "feat(cli): mmscan resolves typed torsions first, keys --resume on it, records it

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: `validate` keys on the force field, and PFOA end to end

**Files:**
- Modify: `ffrefine/cli.py` (`cmd_validate`: `binaries`/`final` before the resume check, the preprocessed digest in its key)
- Test: `tests/test_cli_typed.py` (extend `_stub_validate`; append one unit and one live test)

**Interfaces:**
- Consumes: Task 5's `preprocessed_sha256`; Task 3's `molecule_refined.itp` write.
- Produces: the validate `--resume` key gains `"preprocessed_topology_sha256"`, the digest of `final/before_pp.top`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_cli_typed.py`, replace `_stub_validate` so it also stubs the digest (cmd_validate is about to run grompp for it):

```python
def _stub_validate(monkeypatch, sha: str = "a" * 64) -> None:
    """Replace every GROMACS-facing call cmd_validate makes."""
    monkeypatch.setattr(cli, "validate_refit", _fake_validate_refit)
    monkeypatch.setattr(cli, "preprocessed_sha256", lambda *args, **kwargs: sha)
```

Append (add `from ffrefine.config import Binaries`, `from ffrefine.io.structure import read_pdb`, `from ffrefine.io.topwriter import make_system_top`, `from ffrefine.mm.gromacs import single_point_energies`):

```python
def test_validate_resume_key_covers_the_preprocessed_topology(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    _write_pfoa_stage_inputs(out)
    assert main(["fit"] + _pfoa_args(out)) == 0
    _stub_validate(monkeypatch, "a" * 64)
    assert main(["validate"] + _pfoa_args(out)) == 0
    capsys.readouterr()
    assert main(["validate"] + _pfoa_args(out) + ["--resume"]) == 0
    assert "skipping" in capsys.readouterr().out
    _stub_validate(monkeypatch, "b" * 64)
    assert main(["validate"] + _pfoa_args(out) + ["--resume"]) == 0
    assert "skipping" not in capsys.readouterr().out


@pytest.mark.needs_gmx
@pytest.mark.needs_xtb
def test_run_on_pfoa_end_to_end_gives_a_drop_in_topology(tmp_path, gmx_bin, xtb_bin):
    """The milestone's goal: a real CGenFF .top, no hand edits, a usable result."""
    out = tmp_path / "out"
    assert main(["run"] + _pfoa_args(out)) == 0
    validation = json.loads((out / "final" / "validation.json").read_text())
    assert validation["quad_1_based"] == list(PFOA_QUAD)
    refined = out / "final" / "molecule_refined.itp"
    assert _include_lines(refined) == _include_lines(PFOA / "pfoa_gmx.top")

    # Drop-in: beside the original charmm36.ff/, the refined file runs as is.
    dropin = tmp_path / "dropin"
    dropin.mkdir()
    shutil.copytree(PFOA / "charmm36.ff", dropin / "charmm36.ff")
    shutil.copyfile(refined, dropin / "molecule_refined.itp")
    make_system_top(dropin / "molecule_refined.itp", dropin / "system.top", "Other")
    s = read_pdb(PFOA / "pfoa_gmx.pdb")
    energies = single_point_energies(
        Binaries(gmx=gmx_bin), dropin / "system.top", s, s.coords[None, :, :], dropin / "sp", "dropin"
    )
    assert np.isfinite(energies.potential()).all()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_cli_typed.py -q -rs'`
Expected: `AttributeError: <module 'ffrefine.cli'> does not have the attribute 'preprocessed_sha256'` for every test using `_stub_validate`. The end-to-end test takes a few minutes (xtb scan); it may already pass, which is fine: it pins the goal.

- [ ] **Step 3: Rework `cmd_validate`'s key in `ffrefine/cli.py`**

Change the tprparams import to `from .mm.tprparams import ResolvedTerm, preprocessed_sha256, resolve_torsion_set`.

In `cmd_validate`, replace the block from `ff_includes = _ff_includes(args)` down to and including `final = stage_dir(args.outdir, "final")` with:

```python
    ff_includes = _ff_includes(args)
    binaries = _binaries(args)
    final = stage_dir(args.outdir, "final")
    # The "before" energies depend on every force-field file grompp reads, so
    # the key covers grompp's preprocessed view of the topology (milestone 6a
    # design, section 4). About a second, even when the stage is then skipped.
    pp_sha256 = preprocessed_sha256(binaries, topology, structure, final, "before", ff_includes)
    settings = {
        "quad": list(quad),
        "distribute": bool(args.distribute),
        "ff_includes": ff_includes,
        "preprocessed_topology_sha256": pp_sha256,
    }
    digest = hash_inputs([args.topology, scan_path, fit_path], settings)
    if args.resume and stage_is_current(
        args.outdir, "final", digest, _stage_outputs(args, "final")
    ):
        print(f"validate: skipping, {Path(args.outdir) / STAGE_DIRS['final']} is current")
        return 0
```

(The old `binaries = _binaries(args)` and `final = stage_dir(...)` lines that followed the resume check are gone; `validate_refit(binaries, ..., final, ...)` below uses these.)

- [ ] **Step 4: Run the tests, then the whole suite**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest tests/test_cli_typed.py -q -rs && python -m pytest -q -rs'`
Expected: all pass, 0 skipped. `test_run_on_pfoa_end_to_end_gives_a_drop_in_topology` runs the real xtb scan.

- [ ] **Step 5: black and commit**

```bash
wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m black ffrefine/cli.py tests/test_cli_typed.py'
cd /c/Users/hrach/Projects/ffrefine
git add ffrefine/cli.py tests/test_cli_typed.py
git commit -m "feat(cli): validate keys --resume on the force field; PFOA runs end to end

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: README

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the behaviour Tasks 1-8 built; the final test counts.
- Produces: user-facing documentation only.

- [ ] **Step 1: Status**

Change the heading `## Status: milestones 1-5 of 6` to `## Status: milestones 1-5 and 6a of 6`. In the table, change the `ffrefine mmscan` row's text to:

```
works -- MM single points with and without the torsion, plus the mandatory self-check; typed (`[ dihedraltypes ]`) torsions get the parameters grompp itself built
```

Replace the paragraph that begins "Also not implemented:" with:

```
Also not implemented: CGenFF `.str` penalty parsing and the automatic torsion
and charge-atom selection it would drive (`--str`), and the `final/report.md`
summary document -- both milestone 6b -- and `--mm-relax` (spec 4.6).
```

- [ ] **Step 2: Inputs**

In the `-p/--topology` bullet, replace its last sentence ("Read "What this build cannot read" below before pointing this at a real force field.") with:

```
  The CGenFF server's GROMACS export (`<mol>_gmx.top` beside its
  `charmm36.ff/`) works as is: see "Worked example: PFOA" below.
```

- [ ] **Step 3: Rewrite "What this build cannot read"**

Replace the whole section (from `## What this build cannot read` up to, not including, `## Worked example: n-butane`) with:

```markdown
## What this build cannot read

The topology parser reads **one file**: the one `-p` names. grompp reads
everything that file includes, and ffrefine asks grompp for what it needs from
there instead of reading the force field itself. One common layout still does
not work, and it fails with a clear message rather than a wrong number.

**`[ atoms ]` inside an `#include`d file.** A `.top` whose molecule lives in an
included `.itp` parses to "no `[ atoms ]` section found". Point `-p` at the file
that actually contains the `[ moleculetype ]` you want to refit. The same
limitation applies to `--resume`: see the `--resume` bullet under Common flags.

What used to be on this list, and now works:

- **Torsions whose parameters come from `[ dihedraltypes ]`**, the normal shape
  of a CGenFF or GAFF molecule topology, where the `[ dihedrals ]` lines carry
  `i j k l 9` and nothing else. `mmscan` runs `grompp -pp` and `gmx dump` on your
  unmodified topology and reads back the terms grompp itself built
  (`02_mm_scan/full_dump.stdout`), cross-checking them line by line against the
  torsion set. Those are the numbers the mandatory self-check uses, and
  `energies.json` records every one with its source. ffrefine does not
  reimplement grompp's type matching, so it cannot disagree with it. grompp
  builds no term for a `k = 0` entry; a typed line whose terms are all zero
  simply contributes nothing.
- **Relative `#include`s.** The copies ffrefine hands to grompp live in stage
  directories, so their relative includes are rewritten to absolute paths. The
  files you keep -- `03_fit/dihedral.itp`, `04_charges/molecule_charged.itp`,
  `final/molecule_refined.itp` -- keep your include lines exactly as written, so
  each is a drop-in replacement beside your original, and two comment lines at
  the top name the directory those lines were written for. ffrefine reads that
  marker when `-p` later points at such a file.
- **`[ defaults ]` only in an include.** `fudgeQQ`, `fudgeLJ` and gen-pairs are
  read from grompp's preprocessed topology and recorded in `energies.json`.
- **`--resume` after editing the force field.** `mmscan` and `validate` key on a
  digest of grompp's preprocessed topology, so editing any file grompp reads
  makes them rerun.

A molecule `.itp` with no `[ atomtypes ]` and no `#include` of its own still
needs `--ff-include` to name the force field; it is emitted ahead of the
molecule in the generated wrapper `.top`, and a path to a file ffrefine can see
is made absolute.
```

- [ ] **Step 4: Add the PFOA worked example**

Insert immediately before `## Output layout`:

```markdown
## Worked example: PFOA, a real CGenFF topology

`fixtures/pfoa_gromacs/` holds perfluorooctanoic acid exactly as the CGenFF
server exported it for GROMACS -- `pfoa.str`, `pfoa_gmx.top`, `pfoa_gmx.pdb` --
beside a `charmm36.ff/` trimmed to the types it uses (see its `PROVENANCE.md`).
Every torsion in `pfoa_gmx.top` is typed, and the `.top` includes
`./charmm36.ff/forcefield.itp` and `pfo_ffbonded.itp`.

`pfoa.str` gives the torsion `OG311 CG2O2 CG312 CG312` a penalty of 67: in
`.top` numbering that is atoms **25, 1, 3, 4** (the hydroxyl oxygen, the
carboxyl carbon and the first two chain carbons). The export renamed the atoms,
so the `.str` names do not match the `.top`'s; the indices are the `.top`'s.

    ffrefine run \
        -s fixtures/pfoa_gromacs/pfoa_gmx.pdb \
        -p fixtures/pfoa_gromacs/pfoa_gmx.top \
        --dihedral 25,1,3,4 \
        --outdir pfoa_out

`pfoa_out/02_mm_scan/energies.json` lists the 20 terms about the C1-C3 bond,
each with `"source": "dihedraltypes"` -- the target's four CGenFF
multiplicities (1, 2, 3 and 6) among them -- and the CHARMM36 defaults,
`fudgeQQ` and `fudgeLJ` both 1.0. `pfoa_out/final/molecule_refined.itp`
carries the fitted series on the target torsion; copied next to `pfoa_gmx.top`
and its `charmm36.ff/`, it replaces the original.

Choosing the torsion from `pfoa.str`'s penalties automatically, and the
`report.md` that summarises a run, are milestone 6b.
```

- [ ] **Step 5: Output layout**

In the layout block, replace the `02_mm_scan/` entry with:

```
      02_mm_scan/   mdp, tpr, edr, xvg, full.top, zeroed.top, GROMACS logs,
                    full_pp.top and full_dump.stdout (grompp's preprocessed
                    topology, and the parameters it built, read back), and
                    energies.json (measured phi, full and zeroed potentials, the
                    self-check residual, every removed term's parameters and
                    where they came from, fudgeQQ/fudgeLJ as grompp read them)
```

and the `final/` entry with:

```
      final/        molecule_refined.itp (your #include lines, drop-in),
                    validation.json, plots/validation.png, plus the before/after
                    topologies grompp ran (absolute includes), before_pp.top
                    and GROMACS logs
```

Change the sentence "`final/report.md` is part of the layout but belongs to milestone 6 and is not produced." to say "milestone 6b".

- [ ] **Step 6: Test counts**

Count the suite:

```bash
wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest --collect-only -q | tail -1 && for m in needs_gmx needs_xtb needs_orca; do printf "%s " $m; python -m pytest --collect-only -q -m $m | tail -1; done'
```

Replace the paragraph "As of this milestone the suite holds 547 tests: 16 marked `needs_gmx`, 4 `needs_xtb` and 11 `needs_orca`." with the same sentence carrying the four numbers just printed. (A test marked both `needs_gmx` and `needs_xtb`, like the PFOA end-to-end run, counts under each.)

- [ ] **Step 7: Final verification, then commit**

Run: `wsl -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && python -m pytest -q -rs && python -m pytest -q -rs -m "needs_gmx or needs_xtb or needs_orca" && python -m black --check ffrefine tests scripts'`
Expected: every test passes, **0 skipped** in both runs, and black is clean.

```bash
cd /c/Users/hrach/Projects/ffrefine
git add README.md
git commit -m "docs(readme): typed torsions and included force fields work; PFOA worked example

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```
