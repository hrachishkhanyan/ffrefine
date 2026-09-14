"""CHARMM .prm DIHEDRALS block emission (spec section 4.8).

CHARMM parameter files are in kcal/mol and address atoms by type, so this
output is only physically meaningful when the fit was distributed over every
equivalent quadruplet.  Without --distribute the block is still written, with a
warning, because users ask for it.

Scope, and it is narrower than it looks (ruling R40): the block carries ONE
dihedral type -- the target quadruplet's -- while the N_T quadruplets about the
bond generally span several.  So even under --distribute this file is not a
transcription of the refit ``.itp``; it covers the share belonging to that one
type.  :func:`_distribute_note` states the coverage and names the combinations
that are omitted, because the alternative -- emitting every signature -- would
redefine generic CGenFF types for every molecule containing the motif, which is
a spec decision for a later milestone rather than a bug fix.
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


def _charmm_type_key(types: tuple[str, ...]) -> tuple[str, ...]:
    """Canonical form of a CHARMM dihedral type: ``A-B-C-D`` and ``D-C-B-A`` are one type."""
    return min(types, tuple(reversed(types)))


def _quad_rows(labels: list[str], indent: str) -> list[str]:
    """Wrap quadruplet labels five to a line so a long list stays readable."""
    return [indent + "  ".join(labels[i : i + 5]) for i in range(0, len(labels), 5)]


def _distribute_note(
    topology: Topology,
    quads: list[tuple[int, int, int, int, int]],
    central: tuple[int, int],
    emitted: tuple[str, ...],
) -> str:
    """Comment block stating what a distributed Kchi is, and what it does not cover.

    Two separate facts, and ruling R40 is that the block must carry both.

    The VALUE is a *per-quadruplet* constant.  CHARMM sums a dihedral type over
    every matching quadruplet exactly as GROMACS sums its ``[ dihedrals ]``
    lines, so the number here must be the fitted ``k_n`` divided by N_T -- the
    same value written into the refit ``.itp``.  Saying so is what lets a reader
    check it against ``03_fit/dihedral.itp`` by eye.

    The COVERAGE is not the whole torsion set.  :func:`write_prm_block` emits a
    single type signature -- the target quadruplet's -- while the N_T
    quadruplets about the bond generally span several.  On the butane fixture
    the emitted ``CG331-CG321-CG321-CG331`` matches 1 of 9; the other 8 are
    ``CG331-CG321-CG321-HGA2`` and ``HGA2-CG321-CG321-HGA2``.  An earlier
    version of this note claimed the block and the ``.itp`` "must agree", which
    was wrong in the direction that matters: it invited a user to paste a block
    carrying one ninth of the correction and believe it equivalent.

    Emitting the other signatures is deliberately NOT the remedy (ruling R40):
    ``HGA2-CG321-CG321-HGA2`` is a generic CGenFF type whose redefinition would
    perturb every molecule containing that motif, not just this one.  That is a
    spec decision for a later milestone, so the note states the limitation
    instead of the tool silently implying there is none.
    """
    n_t = len(quads)
    groups: dict[tuple[str, ...], list[str]] = {}
    for key in quads:
        types = tuple(topology.atoms[i - 1].atype for i in key[:4])
        groups.setdefault(_charmm_type_key(types), []).append("-".join(str(i) for i in key[:4]))

    emitted_key = _charmm_type_key(emitted)
    covered = groups.get(emitted_key, [])
    omitted = sorted((k for k in groups if k != emitted_key))

    lines = [
        "! Generated with --distribute.",
        f"! The fitted force constants were spread evenly over the {n_t} distinct",
        f"! quadruplets about the central bond ({central[0]}, {central[1]}), so each Kchi",
        f"! below is the fitted k_n divided by {n_t} -- the same per-quadruplet value",
        "! written to the refit dihedral.itp.",
        "!",
        "! Read this before pasting the block into a CHARMM parameter file.",
        f"! This block defines ONE dihedral type, {'-'.join(emitted)}, covering",
        f"! {len(covered)} of the {n_t} quadruplets (1-based topology atom indices):",
        *_quad_rows(covered, "!     "),
    ]
    if omitted:
        n_omitted = sum(len(groups[k]) for k in omitted)
        lines += [
            f"! The other {n_omitted} carry different type combinations, which ffrefine",
            "! does not emit:",
        ]
        for key in omitted:
            labels = groups[key]
            lines.append(f"!   {'-'.join(key)}  -- {len(labels)} of the {n_t}")
            lines += _quad_rows(labels, "!     ")
        lines += [
            "! So this block alone does not reproduce dihedral.itp: within this",
            f"! molecule it carries {len(covered)}/{n_t} of the fitted torsional correction.",
            "! Use dihedral.itp for GROMACS.  Treat this block as the starting point",
            "! for a CHARMM parameterisation, not as a drop-in equivalent.",
        ]
    else:
        lines += [
            "! Every quadruplet about this bond carries that one type, so within this",
            "! molecule the block does reproduce dihedral.itp's torsional correction.",
        ]
    return "\n".join(lines)


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
        lines.append(_distribute_note(topology, keys, (quad[1], quad[2]), tuple(types)))
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
