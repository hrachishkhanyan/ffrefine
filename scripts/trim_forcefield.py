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
