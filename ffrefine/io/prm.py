"""CHARMM .prm DIHEDRALS block emission (spec section 4.8).

CHARMM parameter files are in kcal/mol and address atoms by type, so this
output is only physically meaningful when the fit was distributed over every
equivalent quadruplet.  Without --distribute the block is still written, with a
warning, because users ask for it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..mm.torsionset import distinct_entry_keys, validate_quad
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


def _distribute_note(quads: list[tuple[int, int, int, int, int]], central: tuple[int, int]) -> str:
    """Comment block naming the quadruplets a distributed Kchi is spread over.

    A CHARMM ``Kchi`` is a *per-quadruplet* constant and CHARMM sums the type
    over every matching quadruplet, exactly as GROMACS sums its own
    ``[ dihedrals ]`` lines.  So the value here must be the same
    per-quadruplet constant written into the refit ``.itp``, i.e. the fitted
    ``k_n`` divided by N_T -- not the undivided total.  Saying so in the file
    is what lets a reader check it against ``03_fit/dihedral.itp`` by eye.
    """
    n_t = len(quads)
    listed = "  ".join("-".join(str(i) for i in key[:4]) for key in quads)
    wrapped: list[str] = []
    row: list[str] = []
    for item in listed.split("  "):
        row.append(item)
        if len(row) == 5:
            wrapped.append("!   " + "  ".join(row))
            row = []
    if row:
        wrapped.append("!   " + "  ".join(row))
    return "\n".join(
        [
            "! Generated with --distribute.",
            f"! The fitted force constants were spread evenly over the {n_t} distinct",
            f"! quadruplets about the central bond ({central[0]}, {central[1]}), so each Kchi",
            f"! below is the fitted k_n divided by {n_t} -- the same per-quadruplet value",
            "! written to the refit dihedral.itp.  CHARMM sums this type over every",
            "! matching quadruplet, as GROMACS does, so the two must agree.",
            "! Quadruplets covered (1-based topology atom indices):",
            *wrapped,
        ]
    )


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
        k: force constants in kJ/mol, as fitted -- i.e. the *total* over the
            torsion set, exactly as handed to
            :func:`ffrefine.io.topwriter.write_refit_topology`.  Converted to
            kcal/mol on output via :func:`ffrefine.units.kj_to_kcal` (the
            project's single unit conversion point), and divided by N_T when
            ``distribute`` is set, so the emitted ``Kchi`` is the same
            per-quadruplet constant the ``.itp`` carries.
        delta_deg: phases in degrees.
        mult: multiplicities.
        out_path: destination .prm fragment; the input topology is never
            modified.
        distribute: whether the fit was spread over all equivalent
            quadruplets (spec section 9.7). When True each ``Kchi`` is scaled
            by ``1 / N_T`` using the same denominator ``topwriter`` uses
            (spec line 242, test 7: "divide the fitted k_n by the number of
            equivalent quadruplets" -- unqualified, so it governs this file
            too). When False the block is still emitted unscaled, with a loud
            warning comment, because a type-level CHARMM entry built from a
            molecule-specific fit is not transferable.

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
    if distribute:
        # Same denominator as topwriter.write_refit_topology, by construction:
        # both call distinct_entry_keys. Distinct quadruplets (9 on butane),
        # NOT term-lines (10) -- rulings R14/R17.
        keys = distinct_entry_keys(topology, quad)
        scale = 1.0 / len(keys)
        lines.append(_distribute_note(keys, (quad[1], quad[2])))
    else:
        scale = 1.0
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
        lines.append(
            f"{type_field} {kj_to_kcal(float(kk) * scale):>10.4f}" f"{int(nn):>5d}{float(dd):>9.2f}"
        )
    lines.append("")
    lines.append("END")
    text = "\n".join(lines) + "\n"
    Path(out_path).write_text(text)
    return text
