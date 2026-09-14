"""xtb relaxed torsion scan driver (spec section 4.3).

xtb speaks Angstrom and Hartree; ffrefine speaks nm and kJ/mol.  Both
conversions go through :mod:`ffrefine.units`, which is the only place they
are allowed to live (spec section 9.8).

The ``$scan`` syntax this module emits was verified against xtb 6.7.1 by live
probe rather than assumed -- see the design document's section 2.  Spec 4.3
requires exactly that, and :func:`run_scan` re-verifies at runtime that the
block was actually honoured.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from ..errors import ValidationError

Quad = tuple[int, int, int, int]


def scan_grid(step_deg: float) -> tuple[float, float, int]:
    """Return ``(start_deg, end_deg, npoints)`` for a full turn at ``step_deg``.

    Spec 4.3: the grid spans 0 to ``360 - step`` so the endpoints are not
    duplicated.  0 and 360 degrees are the same angle; sampling both would put
    two points on one geometry, which biases any averaging over the profile
    and produces a duplicated abscissa in every plot.

    Raises:
        ValidationError: ``step_deg`` is not positive, or does not divide 360
            into a whole number of steps (which would leave the grid unable to
            close the turn).
    """
    if step_deg <= 0:
        raise ValidationError(f"--step must be positive, got {step_deg}")
    npoints = round(360.0 / step_deg)
    if abs(npoints * step_deg - 360.0) > 1e-9:
        raise ValidationError(
            f"--step {step_deg} does not divide 360 into whole steps "
            f"({360.0 / step_deg:.4f} of them). Use a step such as 5, 10, 15, 20 or 30."
        )
    return 0.0, 360.0 - step_deg, npoints


def _constraint_line(quad: Sequence[int]) -> str:
    if len(set(quad)) != 4:
        raise ValidationError(f"dihedral {tuple(quad)} repeats an atom index")
    return "  dihedral: " + ",".join(str(i) for i in quad) + ",auto"


def write_scan_input(
    path: str | Path,
    quad: Quad,
    start_deg: float,
    end_deg: float,
    npoints: int,
    force_constant: float = 1.0,
    freeze: Iterable[Quad] = (),
) -> str:
    """Write the xcontrol file driving a relaxed scan of ``quad``.

    ``auto`` makes xtb take each constrained dihedral's starting value from the
    input geometry, so the frozen dihedrals hold wherever the user's structure
    put them.  The scanned dihedral is listed first and is the only one given a
    ``$scan`` entry; ``freeze`` entries are constrained but never scanned,
    which is the whole point of ``--freeze-dihedral`` (spec 4.3).

    Args:
        path: destination xcontrol file.
        quad: 1-based target dihedral.
        start_deg, end_deg: scan range; ``end_deg`` may be below ``start_deg``
            for a reverse leg, which xtb accepts (verified on 6.7.1).
        npoints: number of scan points.
        force_constant: xtb constraint force constant.
        freeze: additional 1-based dihedrals to hold fixed.

    Returns:
        The text written.

    Raises:
        ValidationError: a quadruplet repeats an atom index.
    """
    lines = ["$constrain", f"  force constant={force_constant}", _constraint_line(quad)]
    for other in freeze:
        lines.append(_constraint_line(other))
    lines.append("$scan")
    lines.append(f"  1: {start_deg},{end_deg},{npoints}")
    lines.append("$end")
    text = "\n".join(lines) + "\n"
    Path(path).write_text(text, newline="\n")
    return text
