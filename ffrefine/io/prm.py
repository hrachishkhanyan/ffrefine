"""CHARMM .prm DIHEDRALS block emission (spec section 4.8).

CHARMM parameter files are in kcal/mol and address atoms by type, so this
output is only physically meaningful when the fit was distributed over every
equivalent quadruplet.  Without --distribute the block is still written, with a
warning, because users ask for it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..mm.torsionset import validate_quad
from ..units import kj_to_kcal
from .topology import Topology

Quad = tuple[int, int, int, int]

_NO_DISTRIBUTE_WARNING = """\
! ============================ WARNING ============================
! This block was generated WITHOUT --distribute.
!
! The fitted force constants were written onto a single dihedral
! quadruplet and every other torsion about the same central bond was
! set to zero.  That is exact for the one molecule it was fitted
! against, but as a CHARMM dihedral TYPE it is wrong: it will be
! applied to every quadruplet matching these atom types.
!
! Re-run with --distribute if you want a transferable dihedral type.
! ================================================================="""


def write_prm_block(
    topology: Topology,
    quad: Quad,
    k: np.ndarray,
    delta_deg: np.ndarray,
    mult: np.ndarray,
    out_path: str | Path,
    distribute: bool,
) -> str:
    """Write a CHARMM DIHEDRALS block for the fitted torsion.

    Args:
        topology: supplies the atom types for the quadruplet.
        quad: 1-based (i, j, k, l), matching the topology's atom numbering.
        k: force constants in kJ/mol; converted to kcal/mol on output via
            :func:`ffrefine.units.kj_to_kcal` (the project's single unit
            conversion point).
        delta_deg: phases in degrees.
        mult: multiplicities.
        out_path: destination .prm fragment; the input topology is never
            modified.
        distribute: whether the fit was spread over all equivalent
            quadruplets (spec section 9.7). When False the block is still
            emitted, with a loud warning comment, because a type-level
            CHARMM entry built from a molecule-specific fit is not
            transferable.

    Returns:
        The block text that was written to ``out_path``.

    Raises:
        ValidationError: ``quad`` is malformed -- wrong length, an atom
            index outside ``1..topology.natoms``, a repeated index, a
            central pair that is not a bond, or no proper dihedral in the
            topology actually uses that central bond. Delegated to
            :func:`ffrefine.mm.torsionset.validate_quad`, the single place
            this check is implemented; a raw ``atoms[i - 1]`` lookup would
            otherwise accept an out-of-range or wraparound index (e.g. ``0``
            silently indexing the last atom) and emit a plausible-looking
            but wrong block.
    """
    validate_quad(topology, quad)
    types = [topology.atoms[i - 1].atype for i in quad]  # 1-based -> 0-based
    lines: list[str] = []
    if not distribute:
        lines.append(_NO_DISTRIBUTE_WARNING)
    lines.append("DIHEDRALS")
    lines.append("!atom types                          Kchi     n   delta")
    # Each type field is padded to 8 chars -- 2 wider than CGenFF's 6-char
    # type-name limit -- and explicitly space-joined, so two fields can never
    # fuse into one unparseable token even if a type reaches or exceeds that
    # width; padding alone (e.g. ``f"{t:<6s}"`` with no separator) would fuse
    # silently once a type is >= the field width.
    type_field = " ".join(f"{t:<8s}" for t in types)
    for kk, dd, nn in zip(np.atleast_1d(k), np.atleast_1d(delta_deg), np.atleast_1d(mult)):
        lines.append(f"{type_field} {kj_to_kcal(float(kk)):>10.4f}{int(nn):>5d}{float(dd):>9.2f}")
    lines.append("")
    lines.append("END")
    text = "\n".join(lines) + "\n"
    Path(out_path).write_text(text)
    return text
