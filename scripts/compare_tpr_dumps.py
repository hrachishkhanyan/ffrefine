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
