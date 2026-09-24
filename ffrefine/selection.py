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
