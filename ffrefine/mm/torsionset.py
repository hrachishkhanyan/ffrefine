"""Selection of the dihedral terms to remove (spec sections 4.2 and 9.1).

All indices in this module are 1-based, matching both the topology file and
the user-facing --dihedral flag.
"""

from __future__ import annotations

from ..errors import ValidationError
from ..io.topology import Dihedral, Topology

Quad = tuple[int, int, int, int]


def _normalise_bond(j: int, k: int) -> tuple[int, int]:
    """Order a bond pair so it can be compared regardless of direction."""
    return (min(j, k), max(j, k))


def validate_quad(topology: Topology, quad: Quad) -> None:
    """Check that a user-supplied dihedral quadruplet is usable.

    Args:
        topology: parsed topology.
        quad: 1-based (i, j, k, l).

    Raises:
        ValidationError: index out of range, central pair not bonded, or no
            proper dihedral in the topology uses that central bond.
    """
    if len(quad) != 4:
        raise ValidationError(f"dihedral must have 4 atom indices, got {len(quad)}")
    for idx in quad:
        if not 1 <= idx <= topology.natoms:
            raise ValidationError(
                f"atom index {idx} out of range; topology has {topology.natoms} atoms "
                f"(indices are 1-based)"
            )
    if len(set(quad)) != 4:
        raise ValidationError(f"dihedral {quad} repeats an atom index")

    j, k = quad[1], quad[2]
    bonds = {_normalise_bond(a, b) for a, b in topology.bonds}
    if _normalise_bond(j, k) not in bonds:
        raise ValidationError(
            f"atoms {j} and {k} are not bonded; the central pair of a proper "
            f"dihedral must be a bond"
        )
    if not collect_torsion_set(topology, quad):
        raise ValidationError(
            f"no proper dihedral in the topology has central bond ({j}, {k}); nothing to refit"
        )


def collect_torsion_set(topology: Topology, quad: Quad) -> list[Dihedral]:
    """Every proper dihedral sharing the central bond of ``quad`` (set T).

    Spec section 9.1: all of these are zeroed for the MM reference, not only
    the selected quadruplet.  Impropers are excluded by construction because
    the topology model keeps them in a separate list.

    Args:
        topology: parsed topology.
        quad: 1-based (i, j, k, l); only j and k matter here.

    Returns:
        Matching proper dihedral entries, in topology order.  A funct-9
        torsion written as several lines (multiple multiplicities) contributes
        one entry per line -- callers must not collapse or deduplicate by
        quadruplet.
    """
    target = _normalise_bond(quad[1], quad[2])
    return [d for d in topology.propers if d.central_bond == target]


def find_target_entries(topology: Topology, quad: Quad) -> list[Dihedral]:
    """The members of T matching the exact quadruplet, in either direction."""
    forward = tuple(quad)
    reverse = tuple(reversed(quad))
    return [
        d
        for d in collect_torsion_set(topology, quad)
        if (d.ai, d.aj, d.ak, d.al) in (forward, reverse)
    ]
