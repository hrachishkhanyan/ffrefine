"""Turning fitted charges into topology charges, with the total kept exact.

Spec 5.3's bookkeeping.  The molecular total is a hard constraint rather than a
preference: a topology whose charges sum to 0.0003 is not a neutral molecule,
and GROMACS will simulate it anyway without complaint.

Nothing here runs a subprocess.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import ValidationError
from ..io.topology import Topology
from ..io.validate import element_of

CHARGE_DECIMALS = 4
"""Decimal places written into ``[ atoms ]``.

Every written charge is then a multiple of 1e-4, so compensating the rounding
residual on one atom makes the total exactly the target rather than merely
close to it.
"""


@dataclass
class ChargeAssignment:
    """What happened to one atom's charge.

    Attributes:
        nr: 1-based atom number.
        name: atom name, for the log and ``charges.json``.
        before: the topology's charge before this stage.
        after: the charge written out.
        source: ``"fitted"`` (taken from the ESP fit), ``"redistributed"``
            (adjusted to keep the total exact) or ``"unchanged"``.
    """

    nr: int
    name: str
    before: float
    after: float
    source: str


def _recipients(topology: Topology, selected: set[int]) -> list[int]:
    """Atoms that may absorb the residual, as 0-based indices.

    Spec 5.3: the atoms in the same charge group as the selected ones, falling
    back to bonded heavy atoms.  Hydrogens are excluded from the fallback
    because pushing a correction onto a hydrogen distorts the group it belongs
    to more than putting it on the heavy atom it hangs off.
    """
    groups = {topology.atoms[nr - 1].cgnr for nr in selected}
    same_group = [a.nr for a in topology.atoms if a.cgnr in groups and a.nr not in selected]
    if same_group:
        return [nr - 1 for nr in same_group]
    bonded = {
        n
        for nr in selected
        for n in topology.bonded_neighbours(nr)
        if n not in selected and element_of(topology, n) != "H"
    }
    return [nr - 1 for nr in sorted(bonded)]


def apply_charges(
    topology: Topology,
    fitted_e: np.ndarray,
    selected: list[int] | None,
    target_total: int,
    decimals: int = CHARGE_DECIMALS,
) -> list[ChargeAssignment]:
    """Combine fitted charges with the topology's, keeping the total exact.

    In ``selected`` mode, the fitted charges replace only the named atoms; the
    residual is redistributed to unselected atoms in the same charge group
    (falling back to bonded heavy atoms). After rounding to the output
    precision, any remaining drift is absorbed by the largest-magnitude atom
    among the redistribution recipients (or, in ``all`` mode, any atom). This
    ensures an atom reported as ``source="fitted"`` carries its fitted value
    unchanged: drift is kept off fitted atoms when there is anywhere else to
    put it. The molecular total is then exact rather than merely close.

    Args:
        topology: the parsed molecule, supplying the existing charges, the
            charge groups and the bond graph.
        fitted_e: shape (natoms,), 0-based in topology atom order.
        selected: 1-based atom numbers to replace, or ``None`` for every atom
            (``--charge-mode all``).
        target_total: the molecular charge the result must sum to, from
            ``derive_charge``.
        decimals: output precision.

    Returns:
        One :class:`ChargeAssignment` per atom, in topology order.

    Raises:
        ValidationError: a selected atom number is out of range, or the
            residual has nowhere to go.
    """
    natoms = topology.natoms
    fitted = np.asarray(fitted_e, float)
    if fitted.shape != (natoms,):
        raise ValidationError(f"fitted charges have shape {fitted.shape}, expected ({natoms},)")

    if selected is None:
        chosen = set(range(1, natoms + 1))
    else:
        chosen = set(selected)
        bad = sorted(nr for nr in chosen if not 1 <= nr <= natoms)
        if bad:
            raise ValidationError(
                f"--charge-atoms names atom(s) {bad}, but the topology has {natoms} atoms "
                "(indices are 1-based, matching [ atoms ])."
            )

    before = np.array([a.charge for a in topology.atoms], dtype=float)
    after = before.copy()
    index = sorted(nr - 1 for nr in chosen)
    after[index] = fitted[index]

    recipients = []
    residual = float(target_total) - float(after.sum())
    if selected is not None and abs(residual) > 1e-12:
        recipients = _recipients(topology, chosen)
        if not recipients:
            raise ValidationError(
                f"nothing left to redistribute the {residual:+.6f} e residual onto: every "
                "atom in the selected atoms' charge groups was itself selected, and no "
                "unselected heavy atom is bonded to one. Select fewer atoms, or use "
                "--charge-mode all."
            )
        after[recipients] += residual / len(recipients)

    after = np.round(after, decimals)
    drift = float(target_total) - float(after.sum())
    if abs(drift) > 0.0:
        # Apply drift to recipients first if available, otherwise largest magnitude atom
        if recipients:
            largest = recipients[int(np.argmax(np.abs(after[recipients])))]
        else:
            largest = int(np.argmax(np.abs(after)))
        after[largest] = round(after[largest] + drift, decimals)
    final_total = float(after.sum())
    if abs(final_total - target_total) > 1e-6:
        raise ValidationError(
            f"charges sum to {final_total:.8f} but the molecular charge is {target_total}; "
            "the rounding compensation did not close the gap."
        )

    out = []
    for i, atom in enumerate(topology.atoms):
        if atom.nr in chosen:
            source = "fitted"
        elif after[i] != before[i]:
            source = "redistributed"
        else:
            source = "unchanged"
        out.append(
            ChargeAssignment(
                nr=atom.nr,
                name=atom.name,
                before=float(before[i]),
                after=float(after[i]),
                source=source,
            )
        )
    return out
