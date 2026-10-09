"""Convert a CGenFF stream file (``.str``) and its mol2 into a GROMACS topology.

What the CGenFF server's GROMACS export does, for when that export fails:

* ``[ atoms ]`` (types, charges) and the bond graph from the ``.str``'s RTF
  block; impropers from its ``IMPR`` lines, in CHARMM's own atom order;
* angles, proper dihedrals and 1-4 pairs generated from the bond graph;
* the ``.str``'s parameter block (the parameters CGenFF had to guess, which no
  force-field release carries) written as ``<resi>_ffbonded.itp`` with the
  CHARMM -> GROMACS unit conversions;
* every other parameter is looked up by grompp in the CHARMM36 port, which is
  copied next to the topology like the server's export does.

Coordinates come from the mol2 CGenFF wrote (``*.cgenff.mol2``), whose atom
names and order match the ``.str``. A parameter the port lacks makes grompp
fail with "No default ... types" rather than pass silently, so check the
output with ``gmx grompp`` before using it.

Usage:
    python scripts/cgenff_to_gmx.py MOL.str MOL.cgenff.mol2 CHARMM36_FF_DIR OUTDIR
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from itertools import combinations
from pathlib import Path

KCAL = 4.184
"""kJ per kcal."""

_BOND_KEYWORDS = {"BOND", "DOUB", "DOUBLE", "TRIP", "TRIPLE", "AROM", "AROMATIC"}
_SECTIONS = {"BONDS", "ANGLES", "DIHEDRALS", "IMPROPERS", "IMPROPER"}
_UNSUPPORTED = {"NONBONDED", "NBFIX", "CMAP", "LONEPAIR", "LONE", "ANISOTROPY"}


class ConversionError(Exception):
    """The input cannot be converted faithfully."""


def _body(line: str) -> list[str]:
    return line.split("!", 1)[0].split()


def parse_str(path: Path) -> dict:
    """Atoms, bonds, impropers and the parameter block of a single-residue ``.str``."""
    resi = None
    charge = 0.0
    atoms: list[tuple[str, str, float]] = []
    bonds: list[tuple[str, str]] = []
    imprs: list[tuple[str, str, str, str]] = []
    params: dict[str, list[list[str]]] = {"BONDS": [], "ANGLES": [], "DIHEDRALS": [], "IMPROPERS": []}
    block = section = ""
    for raw in path.read_text().splitlines():
        stripped = raw.strip()
        low = stripped.lower()
        if low.startswith("read rtf"):
            block, section = "rtf", ""
            continue
        if low.startswith("read para"):
            block, section = "param", ""
            continue
        if stripped.upper() == "END":
            block = section = ""
            continue
        body = _body(stripped)
        if not body or stripped.startswith(("*", "!")):
            continue
        key = body[0].upper()
        if block == "rtf":
            if key == "RESI":
                if resi is not None:
                    raise ConversionError(f"{path}: more than one RESI; convert one residue at a time")
                resi, charge = body[1], float(body[2])
            elif key == "ATOM":
                atoms.append((body[1], body[2], float(body[3])))
            elif key in _BOND_KEYWORDS:
                names = body[1:]
                if len(names) % 2:
                    raise ConversionError(f"{path}: odd number of atoms on {raw!r}")
                bonds += [(names[i], names[i + 1]) for i in range(0, len(names), 2)]
            elif key in ("IMPR", "IMPH"):
                names = body[1:]
                if len(names) % 4:
                    raise ConversionError(f"{path}: improper line {raw!r} is not a multiple of 4 atoms")
                imprs += [tuple(names[i : i + 4]) for i in range(0, len(names), 4)]
            elif key in ("LONEPAIR", "LONE", "ANISOTROPY", "CMAP"):
                raise ConversionError(f"{path}: {key} is not supported")
        elif block == "param":
            if key in _SECTIONS:
                section = "IMPROPERS" if key.startswith("IMPROPER") else key
                continue
            if key in _UNSUPPORTED:
                raise ConversionError(
                    f"{path}: the .str has a {key} section, which this converter does not translate"
                )
            if section:
                params[section].append(body)
    if resi is None or not atoms:
        raise ConversionError(f"{path}: no RESI/ATOM block")
    return {"resi": resi, "charge": charge, "atoms": atoms, "bonds": bonds, "imprs": imprs,
            "params": params}


def read_mol2_coords(path: Path) -> list[tuple[str, float, float, float]]:
    text = path.read_text()
    block = re.search(r"@<TRIPOS>ATOM\n(.*?)(?:@<TRIPOS>|\Z)", text, re.S)
    if not block:
        raise ConversionError(f"{path}: no @<TRIPOS>ATOM block")
    out = []
    for line in block.group(1).splitlines():
        f = line.split()
        if f:
            out.append((f[1], float(f[2]), float(f[3]), float(f[4])))
    return out


def read_masses(ff: Path) -> dict[str, float]:
    masses = {}
    for line in (ff / "atomtypes.atp").read_text().splitlines():
        f = line.split(";", 1)[0].split()
        if len(f) >= 2:
            masses[f[0]] = float(f[1])
    return masses


def topology_terms(n: int, bonds: list[tuple[int, int]]):
    """Angles, proper dihedrals and 1-4 pairs from a bond graph (0-based indices)."""
    nb = {i: set() for i in range(n)}
    for i, j in bonds:
        nb[i].add(j)
        nb[j].add(i)
    angles = []
    for j in range(n):
        for i, k in combinations(sorted(nb[j]), 2):
            angles.append((i, j, k))
    dihedrals = []
    for j, k in sorted({tuple(sorted(b)) for b in bonds}):
        for i in sorted(nb[j] - {k}):
            for l in sorted(nb[k] - {j}):
                if i != l:
                    dihedrals.append((i, j, k, l))
    close = {tuple(sorted(b)) for b in bonds} | {tuple(sorted((a[0], a[2]))) for a in angles}
    pairs = sorted({tuple(sorted((d[0], d[3]))) for d in dihedrals} - close)
    return angles, dihedrals, pairs


def ffbonded_text(params: dict[str, list[list[str]]], resi: str) -> str:
    out = ["; Parameters CGenFF assigned by analogy, from the .str",
           f"; converted by scripts/cgenff_to_gmx.py for {resi}", ""]
    if params["BONDS"]:
        out += ["[ bondtypes ]", ";      i        j  func           b0           kb"]
        for f in params["BONDS"]:
            kb, b0 = float(f[2]), float(f[3])
            out.append(f"{f[0]:>8s} {f[1]:>8s}     1 {b0 / 10:12.8f} {kb * 2 * KCAL * 100:12.2f}")
        out.append("")
    if params["ANGLES"]:
        out += ["[ angletypes ]",
                ";      i        j        k  func       theta0       ktheta          ub0          kub"]
        for f in params["ANGLES"]:
            kt, t0 = float(f[3]), float(f[4])
            kub, s0 = (float(f[5]), float(f[6])) if len(f) >= 7 else (0.0, 0.0)
            out.append(f"{f[0]:>8s} {f[1]:>8s} {f[2]:>8s}     5 {t0:12.6f} {kt * 2 * KCAL:12.6f} "
                       f"{s0 / 10:12.8f} {kub * 2 * KCAL * 100:12.2f}")
        out.append("")
    if params["DIHEDRALS"]:
        out += ["[ dihedraltypes ]",
                ";      i        j        k        l  func         phi0         kphi  mult"]
        for f in params["DIHEDRALS"]:
            k, mult, phi0 = float(f[4]), int(f[5]), float(f[6])
            out.append(f"{f[0]:>8s} {f[1]:>8s} {f[2]:>8s} {f[3]:>8s}     9 {phi0:12.6f} "
                       f"{k * KCAL:12.6f} {mult:5d}")
        out.append("")
    if params["IMPROPERS"]:
        out += ["[ dihedraltypes ]", "; 'improper' dihedrals",
                ";      i        j        k        l  func         phi0         kphi"]
        for f in params["IMPROPERS"]:
            k, psi0 = float(f[4]), float(f[6])
            out.append(f"{f[0]:>8s} {f[1]:>8s} {f[2]:>8s} {f[3]:>8s}     2 {psi0:12.6f} "
                       f"{k * 2 * KCAL:12.6f}")
        out.append("")
    return "\n".join(out) + "\n"


def convert(str_path: Path, mol2_path: Path, ff: Path, outdir: Path) -> tuple[Path, Path]:
    data = parse_str(str_path)
    resi = data["resi"]
    names = [a[0] for a in data["atoms"]]
    index = {name: i for i, name in enumerate(names)}
    if len(index) != len(names):
        raise ConversionError(f"{str_path}: atom names are not unique")
    coords = read_mol2_coords(mol2_path)
    if [c[0] for c in coords] != names:
        raise ConversionError(
            f"{mol2_path}: atom names/order differ from {str_path}; use the mol2 CGenFF wrote "
            "(*.cgenff.mol2)"
        )
    masses = read_masses(ff)
    missing = sorted({a[1] for a in data["atoms"]} - masses.keys())
    if missing:
        raise ConversionError(f"{ff}: atom type(s) {missing} are not in atomtypes.atp")
    try:
        bonds = [(index[a], index[b]) for a, b in data["bonds"]]
        imprs = [tuple(index[x] for x in q) for q in data["imprs"]]
    except KeyError as exc:
        raise ConversionError(f"{str_path}: bond/improper names unknown atom {exc}") from exc
    total = sum(a[2] for a in data["atoms"])
    if abs(total - data["charge"]) > 1e-3:
        raise ConversionError(f"{str_path}: charges sum to {total:+.4f}, RESI says {data['charge']:+.3f}")
    angles, dihedrals, pairs = topology_terms(len(names), bonds)

    outdir.mkdir(parents=True, exist_ok=True)
    ff_out = outdir / "charmm36.ff"
    if ff_out.exists():
        shutil.rmtree(ff_out)
    shutil.copytree(ff, ff_out)
    ffbonded = f"{resi.lower()}_ffbonded.itp"
    (ff_out / ffbonded).write_text(ffbonded_text(data["params"], resi))

    stem = str_path.stem
    top = outdir / f"{stem}_gmx.top"
    lines = [
        "; Topology converted from a CGenFF .str by scripts/cgenff_to_gmx.py",
        f"; source: {str_path.name} + {mol2_path.name}",
        ";",
        "; Include forcefield parameters",
        '#include "./charmm36.ff/forcefield.itp"',
        ";",
        "; Include ligand specific parameters",
        f'#include "./charmm36.ff/{ffbonded}"',
        "",
        "[ moleculetype ]",
        "; Name            nrexcl",
        f"{resi:<18s}3",
        "",
        "[ atoms ]",
        ";   nr       type  resnr residue  atom   cgnr     charge       mass",
    ]
    qtot = 0.0
    for i, (name, atype, q) in enumerate(data["atoms"], start=1):
        qtot += q
        lines.append(f"{i:6d} {atype:>10s} {1:6d} {resi:>7s} {name:>6s} {i:6d} {q:10.3f} "
                     f"{masses[atype]:10.4f}   ; qtot {qtot:+.3f}")
    lines += ["", "[ bonds ]", ";  ai    aj funct"]
    lines += [f"{i + 1:5d} {j + 1:5d}     1" for i, j in bonds]
    lines += ["", "[ pairs ]", ";  ai    aj funct"]
    lines += [f"{i + 1:5d} {j + 1:5d}     1" for i, j in pairs]
    lines += ["", "[ angles ]", ";  ai    aj    ak funct"]
    lines += [f"{i + 1:5d} {j + 1:5d} {k + 1:5d}     5" for i, j, k in angles]
    lines += ["", "[ dihedrals ]", ";  ai    aj    ak    al funct"]
    lines += [f"{i + 1:5d} {j + 1:5d} {k + 1:5d} {l + 1:5d}     9" for i, j, k, l in dihedrals]
    if imprs:
        lines += ["", "[ dihedrals ]", ";  ai    aj    ak    al funct"]
        lines += [f"{i + 1:5d} {j + 1:5d} {k + 1:5d} {l + 1:5d}     2" for i, j, k, l in imprs]
    lines += ["", "[ system ]", f"{resi}", "", "[ molecules ]", f"{resi:<18s}1", ""]
    top.write_text("\n".join(lines))

    pdb = outdir / f"{stem}_gmx.pdb"
    rows = ["REMARK   converted by scripts/cgenff_to_gmx.py"]
    for i, (name, x, y, z) in enumerate(coords, start=1):
        pdbname = f" {name:<3s}" if len(name) < 4 else name[:4]
        element = re.match(r"[A-Za-z]", name).group(0).upper()
        rows.append(f"ATOM  {i:5d} {pdbname} {resi[:4]:>4s}A   1    {x:8.3f}{y:8.3f}{z:8.3f}"
                    f"  1.00  0.00          {element:>2s}")
    rows.append("END")
    pdb.write_text("\n".join(rows) + "\n")
    return top, pdb


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("str_file", type=Path)
    parser.add_argument("mol2", type=Path)
    parser.add_argument("ff", type=Path, help="a CHARMM36 GROMACS port (the charmm36.ff directory)")
    parser.add_argument("outdir", type=Path)
    args = parser.parse_args(argv)
    try:
        top, pdb = convert(args.str_file, args.mol2, args.ff, args.outdir)
    except ConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {top} and {pdb}; check with gmx grompp before use")
    return 0


if __name__ == "__main__":
    sys.exit(main())
