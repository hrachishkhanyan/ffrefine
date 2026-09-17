"""Cross-checks between a structure and its topology (spec section 4.1)."""

from __future__ import annotations

from dataclasses import replace

from ..errors import ValidationError
from .structure import Structure, guess_element
from .topology import Topology

_ELEMENT_MASSES = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "Br": 79.904,
    "I": 126.904,
}


def element_from_mass(mass: float) -> str | None:
    """Nearest element by atomic mass, or None if nothing is within 0.5 amu.

    The mass is the unambiguous authority on an atom's element.  Names are not:
    ``NA`` matches the two-letter element table and reads as sodium when it is
    very often a nitrogen, so callers that have a mass to hand should prefer
    this over :func:`ffrefine.io.structure.guess_element`.
    """
    best, best_diff = None, 0.5
    for sym, ref in _ELEMENT_MASSES.items():
        diff = abs(mass - ref)
        if diff < best_diff:
            best, best_diff = sym, diff
    return best


_element_from_mass = element_from_mass
"""Kept so this module's existing internal callers read unchanged."""


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


def validate_structure_topology(
    structure: Structure, topology: Topology, strict_names: bool = True
) -> list[str]:
    """Verify that a structure and topology describe the same molecule.

    Checks atom count, element ordering (via topology masses where available)
    and atom names.  Atom order must be identical between the two files.

    Args:
        structure: parsed coordinates.
        topology: parsed topology.
        strict_names: treat an atom-name mismatch as fatal rather than a warning.
            Ignored for a structure whose names are synthetic (see
            ``Structure.names_are_synthetic``): those names come from the
            reader, not the user's file, so failing on them would only ever
            report the reader's own placeholder against the topology.

    Returns:
        Non-fatal warning strings.

    Raises:
        ValidationError: on atom-count or element mismatch, or on a name
            mismatch when ``strict_names`` is set.
    """
    warnings: list[str] = []
    if structure.natoms != topology.natoms:
        raise ValidationError(
            f"atom count mismatch: structure has {structure.natoms} atoms, "
            f"topology has {topology.natoms}. The two files must describe the "
            f"same molecule in the same order."
        )

    name_mismatches = []
    for i, (satom_name, atom) in enumerate(zip(structure.names, topology.atoms), start=1):
        if atom.mass is not None:
            expected = _element_from_mass(atom.mass)
            actual = structure.elements[i - 1] or guess_element(satom_name)
            if expected is not None and expected != actual:
                raise ValidationError(
                    f"element mismatch at atom {i} ({satom_name}/{atom.name}): "
                    f"structure says {actual}, topology mass {atom.mass} implies "
                    f"{expected}. Atom order must be identical in both files."
                )
        if not structure.names_are_synthetic and satom_name != atom.name:
            name_mismatches.append(f"atom {i}: structure {satom_name!r} vs topology {atom.name!r}")

    if structure.names_are_synthetic:
        warnings.append(
            "the structure format carries no atom names, so the names compared here "
            "were manufactured from element symbols and the name check was skipped. "
            "Atom count, element ordering and atom ORDER are still enforced, and they "
            "are what has to be right: a name check could not have caught a reordered "
            "file either. The topology's own atom naming is adopted for the files "
            "handed to GROMACS."
        )

    if name_mismatches:
        detail = "; ".join(name_mismatches[:10])
        message = f"atom name mismatch ({len(name_mismatches)} atoms): {detail}"
        if strict_names:
            raise ValidationError(
                message + ". Pass --no-strict-names to downgrade this to a warning."
            )
        warnings.append(message)
    return warnings


def adopt_topology_naming(structure: Structure, topology: Topology) -> Structure:
    """Return a copy of ``structure`` carrying the topology's atom naming.

    For a format with no atom names of its own (XYZ), the reader manufactures
    ``C1``, ``H2``, ... .  Those reach ``grompp`` in the generated coordinate
    and rerun-trajectory files, where they can never match the topology.  The
    two files are required to be in the same atom order anyway (checked by
    :func:`validate_structure_topology`, which must be called first), so the
    topology's names, residue names and residue ids are simply the right ones
    to use.

    A structure whose names are real is returned unchanged, so this is safe to
    call unconditionally.

    Args:
        structure: parsed coordinates.
        topology: parsed topology; supplies the naming.

    Returns:
        Either ``structure`` itself, or a copy with topology naming and
        ``names_are_synthetic`` cleared.

    Raises:
        ValidationError: the two disagree on atom count -- call
            :func:`validate_structure_topology` first, which reports that
            properly.
    """
    if not structure.names_are_synthetic:
        return structure
    if structure.natoms != topology.natoms:
        raise ValidationError(
            f"cannot adopt topology naming: structure has {structure.natoms} atoms, "
            f"topology has {topology.natoms}"
        )
    return replace(
        structure,
        names=[a.name for a in topology.atoms],
        resnames=[a.resname for a in topology.atoms],
        resids=[a.resnr for a in topology.atoms],
        names_are_synthetic=False,
    )
