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
from ..qm.xtb import CHARGE_TOLERANCE

CHARGE_DECIMALS = 4
"""Decimal places written into ``[ atoms ]``.

Every written charge is then a multiple of 1e-4, so compensating the rounding
residual on one atom makes the total exactly the target rather than merely
close to it.
"""


def check_selection(topology: Topology, selected: list[int] | None, target_total: int) -> None:
    """Refuse a request :func:`apply_charges` could only answer with nonsense.

    Cheap -- it needs only the topology -- so a caller can run it before any
    QM job as well; :func:`apply_charges` runs it again regardless.

    Two refusals, both in ``selected`` mode only:

    * a selected atom number that does not exist in the topology; and
    * a target total that disagrees with the topology's own charge sum by
      more than :data:`~ffrefine.qm.xtb.CHARGE_TOLERANCE` (ruling R21,
      amending R10/R11).  ``selected`` mode replaces only the named atoms,
      so the whole disagreement would have to be absorbed by the two or
      three atoms sharing their charge groups: ``--charge -1`` on neutral
      butane with C1 selected took its three hydrogens from +0.090 to
      -0.198 each.  An offset that size is not rounding; it means the
      declared charge and the topology describe different molecules.
      ``all`` mode has no such problem, because the ESP job runs at the
      declared charge and every atom is refitted.

    Args:
        topology: the parsed molecule.
        selected: 1-based atom numbers, or ``None`` for ``--charge-mode all``.
        target_total: the molecular charge the result must sum to.

    Raises:
        ValidationError: either condition above.
    """
    if selected is None:
        return
    natoms = topology.natoms
    bad = sorted(nr for nr in set(selected) if not 1 <= nr <= natoms)
    if bad:
        raise ValidationError(
            f"--charge-atoms names atom(s) {bad}, but the topology has {natoms} atoms "
            "(indices are 1-based, matching [ atoms ])."
        )
    total = float(sum(atom.charge for atom in topology.atoms))
    offset = float(target_total) - total
    if abs(offset) > CHARGE_TOLERANCE:
        raise ValidationError(
            f"the molecular charge is {target_total}, but the topology's [ atoms ] charges "
            f"sum to {total:+.4f}: a {offset:+.4f} e disagreement, beyond the "
            f"{CHARGE_TOLERANCE} e that rounding can explain. --charge-mode selected replaces "
            "only the atoms you named, so the whole difference would be pushed onto the few "
            "atoms sharing their charge groups -- a chemically meaningless result, not a "
            "rounding correction. Use --charge-mode all to refit every charge at the declared "
            f"charge, or correct the topology's charges so they sum to {target_total}."
        )


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


def apply_charges(
    topology: Topology,
    fitted_e: np.ndarray,
    selected: list[int] | None,
    target_total: int,
    decimals: int = CHARGE_DECIMALS,
) -> list[ChargeAssignment]:
    """Combine fitted charges with the topology's, keeping the total exact.

    In ``selected`` mode, the fitted charges replace only the named atoms. Per
    CHARMM conventions, each charge group absorbs its own residual separately:
    for each group containing a selected atom, that group's residual (the
    charge change the fit made within it) is redistributed to unselected atoms
    in that same group, falling back to bonded heavy atoms if no in-group
    recipients exist. After all per-group redistributions, any remaining
    global offset (an explicit ``--charge`` differing from the topology's sum
    by no more than :data:`~ffrefine.qm.xtb.CHARGE_TOLERANCE`) is spread over
    the pooled recipients; a larger offset is refused before anything is
    changed (see :func:`check_selection`). After rounding to
    the output precision, any remaining drift is absorbed by the largest-
    magnitude atom among the redistribution recipients (or, in ``all`` mode,
    any atom). This ensures an atom reported as ``source="fitted"`` carries
    its fitted value unchanged: drift is kept off fitted atoms when there is
    anywhere else to put it. The molecular total is then exact rather than
    merely close.

    That rounding-drift step is only allowed to absorb a rounding-sized
    drift (at most about ``natoms * 0.5 * 10**-decimals``), in every mode,
    including ``all``. A larger drift means an offset was never placed onto
    a recipient upstream, and it is refused rather than silently folded in.
    There is no legitimate large drift in ``all`` mode: the ESP job runs at
    ``charge=target_total``, so the fitted charges already sum to it.

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
        ValidationError: a selected atom number is out of range, the target
            total disagrees with the topology by more than
            ``CHARGE_TOLERANCE`` in ``selected`` mode, or the residual has
            nowhere to go.
    """
    natoms = topology.natoms
    fitted = np.asarray(fitted_e, float)
    if fitted.shape != (natoms,):
        raise ValidationError(f"fitted charges have shape {fitted.shape}, expected ({natoms},)")

    check_selection(topology, selected, target_total)
    chosen = set(range(1, natoms + 1)) if selected is None else set(selected)

    before = np.array([a.charge for a in topology.atoms], dtype=float)
    after = before.copy()
    index = sorted(nr - 1 for nr in chosen)
    after[index] = fitted[index]

    recipients = []
    if selected is not None:
        # Per-group redistribution: each charge group absorbs its own residual
        groups_with_selected = {topology.atoms[nr - 1].cgnr for nr in chosen}
        for cgnr in groups_with_selected:
            # Selected atoms in this group
            selected_in_group = [nr - 1 for nr in chosen if topology.atoms[nr - 1].cgnr == cgnr]
            # Residual for this group: how much the fit changed it
            residual_g = sum(before[i] - fitted[i] for i in selected_in_group)

            # Try to find recipients within the same group
            recipients_in_group = [
                i
                for i in range(len(topology.atoms))
                if topology.atoms[i].cgnr == cgnr and i not in selected_in_group
            ]

            if recipients_in_group:
                group_recipients = recipients_in_group
            else:
                # Fall back to bonded heavy atoms from this group
                bonded = {
                    n
                    for nr in [topology.atoms[i].nr for i in selected_in_group]
                    for n in topology.bonded_neighbours(nr)
                    if n not in chosen and element_of(topology, n) != "H"
                }
                if not bonded:
                    # Only refuse if this group has a nonzero residual
                    if abs(residual_g) > 1e-12:
                        raise ValidationError(
                            f"nothing left to redistribute the {residual_g:+.6f} e residual onto: every "
                            "atom in the selected atoms' charge groups was itself selected, and no "
                            "unselected heavy atom is bonded to one. Select fewer atoms, or use "
                            "--charge-mode all."
                        )
                    continue  # Skip this group if no recipients and no residual
                group_recipients = [nr - 1 for nr in sorted(bonded)]

            # Collect recipients regardless of whether residual is zero
            recipients.extend(group_recipients)

            # Spread residual only if nonzero
            if abs(residual_g) > 1e-12:
                after[group_recipients] += residual_g / len(group_recipients)

        # After per-group redistribution, handle any remaining global offset
        global_offset = float(target_total) - float(after.sum())
        if abs(global_offset) > 1e-12:
            if not recipients:
                raise ValidationError(
                    f"nothing left to redistribute the {global_offset:+.6f} e global offset onto: "
                    "all selected atoms have residuals of zero and no charge groups have recipients. "
                    "Select fewer atoms, or use --charge-mode all."
                )
            recipients_set = sorted(set(recipients))
            after[recipients_set] += global_offset / len(recipients_set)

    after = np.round(after, decimals)
    drift = float(target_total) - float(after.sum())
    if abs(drift) > 0.0:
        # After rounding, the drift can only be rounding-sized: each atom's own
        # rounding moves it by at most half a unit in the last decimal place, so
        # natoms atoms can drift the total by at most natoms * 0.5 * 10**-decimals
        # (plus a little float slack). This check runs in every mode, including
        # ``all``: there is no legitimate large drift there either, since the ESP
        # job runs at charge=target_total, so the fitted charges already sum to
        # the target. A larger drift means an offset was never placed onto a
        # recipient upstream and must not be silently absorbed here.
        max_rounding_drift = natoms * 0.5 * (10**-decimals) + 1e-9
        if abs(drift) > max_rounding_drift:
            raise ValidationError(
                f"charges drift by {drift:+.8f} e after rounding, more than rounding alone can "
                f"cause (up to ~{max_rounding_drift:.8f} e for {natoms} atoms); an offset was "
                "never placed onto a recipient. This is an internal error."
            )
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
