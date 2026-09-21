"""The two-stage restrained ESP fit (spec 5.2).

A plain least-squares fit of charges to an electrostatic potential is badly
conditioned: buried atoms barely affect the potential outside the molecule, so
the fit assigns them large, arbitrary charges that happen to cancel.  RESP adds
a hyperbolic restraint pulling heavy-atom charges toward zero, which costs
almost nothing in fit quality and removes most of that arbitrariness.

Stage 1 fits every atom.  Stage 2 refits only the atoms in equivalence groups
and their parents, with a stronger restraint and the group members constrained
equal, leaving everything else at its stage 1 value -- so imposing methyl
symmetry cannot quietly rearrange the rest of the molecule.

Every constraint is an extra row in a bordered system: one for the total
charge, one per equivalence pair, one per frozen atom.  One mechanism rather
than three.

Nothing here runs a subprocess.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import ValidationError
from .esp import normal_equations, potential_from_charges, rrms

RESP_A_STAGE1 = 0.0005
"""Stage 1 restraint strength (spec 5.2)."""

RESP_A_STAGE2 = 0.001
"""Stage 2 restraint strength: stronger, over far fewer atoms (spec 5.2)."""

RESP_B = 0.1
"""Hyperbolic restraint width, in e (spec 5.2)."""


@dataclass
class RespResult:
    """Outcome of a two-stage fit.

    Attributes:
        charges: shape (natoms,) in e.
        rrms: relative RMS deviation of the fitted potential from the QM one.
        iterations: iterations used by stage 1 and stage 2.
    """

    charges: np.ndarray
    rrms: float
    iterations: tuple[int, int]


def _constraint_rows(
    natoms: int,
    total_charge: float,
    groups: list[list[int]],
    frozen: dict[int, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``C`` and ``d`` for the constraints ``C q = d``."""
    rows: list[np.ndarray] = []
    values: list[float] = []

    total = np.ones(natoms)
    rows.append(total)
    values.append(float(total_charge))

    for group in groups:
        for other in group[1:]:
            row = np.zeros(natoms)
            row[group[0]] = 1.0
            row[other] = -1.0
            rows.append(row)
            values.append(0.0)

    for index, value in sorted(frozen.items()):
        row = np.zeros(natoms)
        row[index] = 1.0
        rows.append(row)
        values.append(float(value))

    return np.array(rows), np.array(values)


def _solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Solve the bordered system ``[[A, C^T], [C, 0]] [q; lambda] = [b; d]``."""
    n = len(b)
    k = len(d)
    matrix = np.zeros((n + k, n + k))
    matrix[:n, :n] = a
    matrix[:n, n:] = c.T
    matrix[n:, :n] = c
    rhs = np.concatenate([b, d])
    try:
        solution = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValidationError(
            f"the ESP fit is singular ({exc}). This usually means two atoms share a "
            "position, or the grid has too few points to determine every charge."
        ) from None
    return solution[:n]


def _run_stage(
    a_matrix: np.ndarray,
    b_vector: np.ndarray,
    constraints: tuple[np.ndarray, np.ndarray],
    restrain: np.ndarray,
    a_strength: float,
    b_width: float,
    max_iterations: int,
    tol: float,
    stage: int,
) -> tuple[np.ndarray, int]:
    """Iterate one restrained stage to convergence.

    The hyperbolic penalty ``a * sum(sqrt(q^2 + b^2) - b)`` is not quadratic, so
    its second derivative depends on the charges.  Linearising it as a diagonal
    ``a / sqrt(q^2 + b^2)`` built from the previous iteration's charges and
    re-solving is the standard treatment, and it converges in a handful of
    passes.
    """
    c, d = constraints
    charges = _solve(a_matrix, b_vector, c, d)
    for iteration in range(1, max_iterations + 1):
        diagonal = np.where(restrain, a_strength / np.sqrt(charges**2 + b_width**2), 0.0)
        updated = _solve(a_matrix + np.diag(diagonal), b_vector, c, d)
        movement = float(np.abs(updated - charges).max())
        charges = updated
        if movement < tol:
            return charges, iteration
    raise ValidationError(
        f"the RESP stage {stage} fit did not converge in {max_iterations} iterations; the "
        f"charges were still moving by {movement:.2e} e (tolerance {tol:.0e}). Returning an "
        "unconverged fit would be indistinguishable from a converged one."
    )


def resp_fit(
    inv_r: np.ndarray,
    v_ref: np.ndarray,
    *,
    restrain: np.ndarray,
    groups: list[list[int]],
    stage2_free: list[int],
    total_charge: float,
    a_stage1: float = RESP_A_STAGE1,
    a_stage2: float = RESP_A_STAGE2,
    b: float = RESP_B,
    max_iterations: int = 100,
    tol: float = 1e-8,
) -> RespResult:
    """Fit charges to a potential with the two-stage restrained procedure.

    Args:
        inv_r: from :func:`ffrefine.charges.esp.inverse_distances`, shape
            (npoints, natoms).
        v_ref: QM potential at those points, Hartree/e.
        restrain: per-atom mask, True where the hyperbolic restraint applies.
            Spec 5.2 restrains non-hydrogen atoms.
        groups: **0-based** equivalence groups, constrained equal in stage 2.
        stage2_free: **0-based** atoms refit in stage 2; everything else is
            frozen at its stage 1 charge.  Empty means stage 2 does nothing.
        total_charge: the molecular charge, enforced exactly.
        a_stage1, a_stage2: restraint strengths.
        b: restraint width.
        max_iterations: cap per stage.
        tol: convergence threshold on the largest charge movement, in e.

    Returns:
        A :class:`RespResult`.

    Raises:
        ValidationError: either stage failed to converge, or the system is
            singular.
    """
    inv_r = np.asarray(inv_r, float)
    natoms = inv_r.shape[1]
    restrain = np.asarray(restrain, bool)
    if restrain.shape != (natoms,):
        raise ValidationError(f"restrain mask has shape {restrain.shape}, expected ({natoms},)")

    a_matrix, b_vector = normal_equations(inv_r, v_ref)

    stage1, iter1 = _run_stage(
        a_matrix,
        b_vector,
        _constraint_rows(natoms, total_charge, [], {}),
        restrain,
        a_stage1,
        b,
        max_iterations,
        tol,
        stage=1,
    )

    charges, iter2 = stage1, 0
    if stage2_free:
        free = set(stage2_free)
        frozen = {i: float(stage1[i]) for i in range(natoms) if i not in free}
        active_groups = [g for g in groups if all(i in free for i in g)]
        charges, iter2 = _run_stage(
            a_matrix,
            b_vector,
            _constraint_rows(natoms, total_charge, active_groups, frozen),
            restrain,
            a_stage2,
            b,
            max_iterations,
            tol,
            stage=2,
        )

    return RespResult(
        charges=charges,
        rrms=rrms(v_ref, potential_from_charges(inv_r, charges)),
        iterations=(iter1, iter2),
    )
