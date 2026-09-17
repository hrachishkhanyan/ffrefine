"""Topologically equivalent hydrogens, and how to enforce their equality.

Spec 5.3 asks for hydrogens that are "the same" to carry the same charge.  Its
wording -- same heavy-atom parent, same atom type, same connectivity depth --
can be read several ways, so this module fixes one reading: **two hydrogens are
equivalent exactly when they are bonded to the same heavy atom and share an
atom type.**  That covers the methyl and methylene groups the spec names.

Molecule-wide symmetry is deliberately out of scope: butane's two terminal
carbons are topologically identical, and a real ESP fit gives them charges a
few thousandths apart, but recognising that needs automorphism detection over
the bond graph.  The difference is reported rather than averaged away.

Nothing here runs a subprocess.
"""

from __future__ import annotations

import numpy as np

from ..io.topology import Topology
from ..io.validate import element_of


def equivalent_hydrogen_groups(topology: Topology) -> list[list[int]]:
    """Group hydrogens that must share a charge.

    Args:
        topology: the parsed molecule.  Connectivity comes from ``[ bonds ]``,
            never from interatomic distances, which would mistake a close
            contact for a covalent bond.

    Returns:
        Groups of **1-based** atom numbers, each of length two or more, sorted
        within a group and ordered by the group's first atom.  A hydrogen with
        no equivalent partner appears in no group, because a group of one
        constrains nothing.
    """
    buckets: dict[tuple[int, str], list[int]] = {}
    for atom in topology.atoms:
        if element_of(topology, atom.nr) != "H":
            continue
        heavy = [n for n in topology.bonded_neighbours(atom.nr) if element_of(topology, n) != "H"]
        if len(heavy) != 1:
            continue
        buckets.setdefault((heavy[0], atom.atype), []).append(atom.nr)
    groups = [sorted(members) for members in buckets.values() if len(members) > 1]
    return sorted(groups, key=lambda g: g[0])


def average_within_groups(charges_e: np.ndarray, groups: list[list[int]]) -> np.ndarray:
    """Set every member of each group to the group's mean charge.

    Used by the CHELPG path, where ORCA has already done the fitting and the
    equality can only be imposed afterwards.  Averaging preserves each group's
    sum, so it cannot move the molecular total -- which matters, because the
    total is a hard constraint rather than a preference.

    Args:
        charges_e: shape (natoms,), indexed 0-based in topology atom order.
        groups: 1-based groups from :func:`equivalent_hydrogen_groups`.

    Returns:
        A new array; the input is not modified.
    """
    out = np.array(charges_e, dtype=float, copy=True)
    for group in groups:
        index = [nr - 1 for nr in group]
        out[index] = out[index].mean()
    return out
