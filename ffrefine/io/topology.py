"""GROMACS topology parser and object model (spec section 4.1).

Atom indices are stored exactly as they appear in the file, i.e. **1-based**.
Conversion to 0-based array indices happens where coordinates are indexed,
never here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..errors import ParseError, ValidationError
from .includes import find_marker, resolve_include_base

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
    """Parsed molecule topology.

    Attributes:
        name: molecule name from [ moleculetype ].
        nrexcl: exclusion range from [ moleculetype ].
        atoms: [ atoms ] entries. ``Atom.nr`` is 1-based, matching the file.
        bonds: [ bonds ] pairs, 1-based atom indices.
        pairs: [ pairs ] 1-4 pairs, 1-based atom indices.
        angles: [ angles ] triples, 1-based atom indices.
        propers: [ dihedrals ] entries classified as proper (funct 1, 9).
        impropers: [ dihedrals ] entries classified as improper (funct 2, 4);
            never modified (spec section 9.4).
        defaults: the [ defaults ] directive, or None if absent.
        raw_lines: the file's lines, verbatim, after resolving
            ``#ifdef``/``#ifndef``/``#else``/``#endif`` against the
            ``defines`` passed to :func:`read_topology` -- lines inside a
            branch that was not taken are dropped, everything else
            (including preprocessor directive lines) is kept unmodified, for
            round-tripping and downstream rewriting.
        source: path the topology was read from.
        include_marker: the directory an ffrefine include-base marker in the
            file names, or None.
    """

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
    include_marker: Path | None = None

    @property
    def natoms(self) -> int:
        """Number of atoms."""
        return len(self.atoms)

    @property
    def include_base(self) -> Path:
        """Directory the file's relative ``#include`` lines resolve against.

        The directory an ffrefine marker names, else the file's own directory
        (milestone 6a design, section 5).

        Raises:
            ValidationError: the marker names a directory that does not exist.
        """
        return resolve_include_base(self.source, self.include_marker)

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
        ParseError: the file could not be read (missing, a directory, or
            unreadable), on malformed input, a missing [ atoms ] section,
            or two conflicting ffrefine include-base markers.
    """
    path = Path(path)
    defines = set(defines or ())
    try:
        all_lines = path.read_text().splitlines()
    except OSError as exc:
        raise ParseError(f"{path}: could not be read: {exc}") from exc
    raw_lines: list[str] = []

    section = ""
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

    for lineno, raw in enumerate(all_lines, start=1):
        stripped = raw.strip()
        if stripped.startswith("#"):
            raw_lines.append(raw)
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
        raw_lines.append(raw)

        body = _strip(raw)
        if not body:
            continue
        if body.startswith("[") and body.endswith("]"):
            section = body[1:-1].strip().lower()
            if section == "atoms":
                seen_atoms_section = True
            continue

        fields = body.split()
        try:
            if section == "defaults":
                defaults = parse_defaults_fields(fields)
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
                    raise ParseError(f"{path}:{lineno}: unsupported dihedral funct {funct}")
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
        include_marker=find_marker(all_lines, path),
    )


CHARGE_TOLERANCE = 0.05
"""How far the topology's charge sum may sit from an integer before it is refused."""


def derive_charge(
    topology: Topology, override: int | None = None, tol: float = CHARGE_TOLERANCE
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
        override: an explicit ``--charge`` from the user; wins outright, and is
            not re-checked against the sum -- it is the escape hatch the
            refusal below advertises, so it has to work on the topology that
            triggered the refusal.
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
