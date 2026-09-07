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
        names_are_synthetic: the source format carries no atom names and
            ``names`` was manufactured from the element symbols (XYZ).  Such
            names can never match a topology, so
            :func:`ffrefine.io.validate.validate_structure_topology` skips the
            name comparison for them and
            :func:`ffrefine.io.validate.adopt_topology_naming` replaces them
            with the topology's own.  Elements and atom *order* are still
            checked, which is what actually has to be right.
    """

    names: list[str]
    resnames: list[str]
    resids: list[int]
    elements: list[str]
    coords: np.ndarray
    title: str = ""
    box: np.ndarray | None = field(default=None)
    names_are_synthetic: bool = False

    @property
    def natoms(self) -> int:
        """Number of atoms."""
        return len(self.names)


def guess_element(atom_name: str, mass: float | None = None) -> str:
    """Infer an element symbol from an atom name.

    Args:
        atom_name: the name as written in the structure file. Leading digits
            are stripped, then a two-letter prefix is matched against the
            known two-letter elements before falling back to the first letter.
        mass: **ignored.** Accepted only so callers that have a mass to hand
            can pass it without a conditional. Deciding an element from a mass
            is a separate check and lives in
            :func:`ffrefine.io.validate._element_from_mass`, which
            :func:`ffrefine.io.validate.validate_structure_topology` uses to
            cross-check this function's answer. The docstring previously
            claimed the mass was used here; it never was.

    Returns:
        The element symbol, capitalised (e.g. "C", "Cl"), or "X" for a name
        with nothing left after stripping digits.
    """
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
    try:
        text = Path(path).read_text()
    except OSError as exc:
        raise ParseError(f"{path}: could not be read: {exc}") from exc
    for line in text.splitlines():
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
    try:
        lines = Path(path).read_text().splitlines()
    except OSError as exc:
        raise ParseError(f"{path}: could not be read: {exc}") from exc
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
    """Read an XYZ file (Angstrom) into nm.

    XYZ carries element symbols and coordinates and nothing else: no atom
    names, no residues.  The ``names`` this builds (``C1``, ``H2``, ...) are
    therefore placeholders and are flagged as such via
    ``Structure.names_are_synthetic`` -- they could never match a topology,
    and before that flag existed the default strict-name check made every
    ``.xyz`` input fatal.
    """
    try:
        lines = Path(path).read_text().splitlines()
    except OSError as exc:
        raise ParseError(f"{path}: could not be read: {exc}") from exc
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
        names_are_synthetic=True,
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

    Column layout matches ``read_pdb`` exactly (serial 6-10, name 12-15,
    altLoc 16, resName 17-19, resSeq 22-25, x/y/z 30-37/38-45/46-53,
    element 76-77) so that a structure written here round-trips through
    ``read_pdb`` unchanged.

    Args:
        path: output file.
        structure: supplies names, resnames, resids and elements.
        frames: shape (nframes, natoms, 3) in nm.
    """
    frames = np.asarray(frames, float)
    if frames.ndim != 3 or frames.shape[1] != structure.natoms:
        raise ValueError(f"frames shape {frames.shape} incompatible with {structure.natoms} atoms")
    out = [f"TITLE     {structure.title or 'ffrefine scan'}"]
    for m, frame in enumerate(frames, start=1):
        out.append(f"MODEL     {m:>4d}")
        for i in range(structure.natoms):
            x, y, z = frame[i] / ANGSTROM_TO_NM
            name = structure.names[i]
            # PDB puts one-to-three character names at column 14, four at 13.
            name_field = f"{name:<4s}" if len(name) >= 4 else f" {name:<3s}"
            out.append(
                f"ATOM  {i + 1:>5d} {name_field} {structure.resnames[i]:>3s}  "
                f"{structure.resids[i]:>4d}    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          "
                f"{structure.elements[i]:>2s}"
            )
        out.append("ENDMDL")
    out.append("END")
    # newline="\n" for consistency with every other file ffrefine hands to
    # GROMACS (see ffrefine.mm.gromacs and ffrefine.io.topwriter).
    Path(path).write_text("\n".join(out) + "\n", newline="\n")
