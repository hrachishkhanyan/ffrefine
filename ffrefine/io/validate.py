"""Cross-checks between a structure and its topology (spec section 4.1)."""

from __future__ import annotations

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


def _element_from_mass(mass: float) -> str | None:
    """Nearest element by atomic mass, or None if nothing is within 0.5 amu."""
    best, best_diff = None, 0.5
    for sym, ref in _ELEMENT_MASSES.items():
        diff = abs(mass - ref)
        if diff < best_diff:
            best, best_diff = sym, diff
    return best


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
        if satom_name != atom.name:
            name_mismatches.append(f"atom {i}: structure {satom_name!r} vs topology {atom.name!r}")

    if name_mismatches:
        detail = "; ".join(name_mismatches[:10])
        message = f"atom name mismatch ({len(name_mismatches)} atoms): {detail}"
        if strict_names:
            raise ValidationError(
                message + ". Pass --no-strict-names to downgrade this to a warning."
            )
        warnings.append(message)
    return warnings
