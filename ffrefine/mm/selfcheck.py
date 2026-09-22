"""The mandatory torsion-removal self-consistency check (spec section 4.5).

E_full(phi) - E_zeroed(phi) computed by GROMACS must equal the analytic sum of
the removed torsion terms, evaluated at the measured dihedral values with the
original parameters, to better than 1e-3 kJ/mol at every point. The analytic
side is evaluated at the coordinates GROMACS actually received (see
``_as_gromacs_saw_it``): the rerun trajectory format's own coordinate
precision (see ``ffrefine.mm.gromacs._write_g96_trajectory``) must be matched
exactly, or the comparison measures the transport's precision rather than the
correctness of the topology edit.

Failing this means the topology edit did not do what was intended.  On failure
the pipeline aborts; it never proceeds to fitting.

Typed torsions -- terms whose parameters live in [ dihedraltypes ], the normal
shape of a CGenFF or GAFF molecule topology -- get their parameters from
:func:`ffrefine.mm.tprparams.resolve_torsion_set`, which reads back the terms
grompp itself built. The check then verifies what it always has: that the
zeroed copy removed exactly those terms and nothing else. Explicit parameters
on a zeroed line do suppress the [ dihedraltypes ] lookup; that is verified,
not assumed, by
``tests/test_selfcheck.py::test_dihedraltypes_entry_is_suppressed_by_explicit_params``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Binaries
from ..errors import ValidationError
from ..geometry import dihedral_deg_frames
from ..io.structure import Structure
from ..io.topology import Topology
from ..io.topwriter import make_system_top, write_zeroed_topology
from ..io.validate import validate_structure_topology
from .analytic import torsion_set_energy
from .gromacs import single_point_energies
from .tprparams import SOURCE_DIHEDRALTYPES, ResolvedTorsionSet, resolve_torsion_set

log = logging.getLogger("ffrefine")

DEFAULT_TOLERANCE = 1e-3
"""kJ/mol; spec section 4.5."""

_G96_COORD_DECIMALS = 9
"""Decimal places ``_write_g96_trajectory`` keeps, in nm (G96 %15.9f)."""


def _as_gromacs_saw_it(frames: np.ndarray) -> np.ndarray:
    """Quantize coordinates to what GROMACS actually received.

    ``single_point_energies`` drives ``mdrun -rerun`` from a multi-frame
    GROMOS-96 (.g96) trajectory (see
    :func:`ffrefine.mm.gromacs._write_g96_trajectory`), whose fixed-width
    fields keep nine decimal places *in nanometres directly* -- about 1e-9 nm
    per atom. Evaluating the analytic reference at the caller's full
    double-precision ``frames`` without matching this rounding would compare
    two different inputs, not two evaluations of the same one.

    Task 13 fix round 1 (ruling R22): this function previously matched a PDB
    rerun trajectory's much coarser 3-decimal-Angstrom precision (~1e-4 nm).
    On the butane fixture that quantization alone was large enough
    (~1.6e-3 kJ/mol) to fail the section 4.5 tolerance regardless of whether
    the topology edit was correct. Moving the rerun trajectory to .g96
    (~1e-9 nm) reduces this function's effect to a no-op in practice -- the
    measured residual after the switch is ~4e-5 kJ/mol either way, consistent
    with GROMACS's own mixed-precision numerical noise floor (confirmed via
    ``gmx --version``: "Precision: mixed") rather than any remaining
    coordinate quantization. It is kept (rather than
    removed) so the comparison stays exactly matched to whatever precision
    the trajectory writer actually uses, by construction, instead of
    silently drifting out of sync with it again.

    Args:
        frames: shape (nframes, natoms, 3) in nm.

    Returns:
        The same coordinates, rounded to the g96 trajectory's precision.
    """
    return np.round(np.asarray(frames, dtype=float), _G96_COORD_DECIMALS)


@dataclass
class SelfCheckResult:
    """Outcome of the section 4.5 check.

    ``full_potential_kj`` and ``zeroed_potential_kj`` are the two total
    potential-energy profiles the check itself had to compute (spec section
    4.4: "single-point energies with and without the target torsions"). They
    are carried out of here rather than discarded so the ``mmscan`` stage can
    write them to ``02_mm_scan/energies.json`` -- which the fit consumes as
    ``E_MM0`` -- without paying for a third GROMACS rerun over the same
    frames. ``gromacs_difference`` is exactly their difference.
    """

    max_abs_error: float
    passed: bool
    gromacs_difference: np.ndarray
    analytic_difference: np.ndarray
    phi_deg: np.ndarray
    tolerance: float
    full_potential_kj: np.ndarray
    zeroed_potential_kj: np.ndarray


def check_torsion_removal(
    binaries: Binaries,
    topology: Topology,
    quad: tuple[int, int, int, int],
    structure: Structure,
    frames: np.ndarray,
    workdir: str | Path,
    nthreads: int = 1,
    tolerance: float = DEFAULT_TOLERANCE,
    strict_names: bool = True,
    ff_includes: list[str] | None = None,
    resolved: ResolvedTorsionSet | None = None,
) -> SelfCheckResult:
    """Verify that zeroing removed exactly the intended torsion terms.

    Args:
        binaries: resolved binaries; ``gmx`` required.
        topology: the original parsed topology.
        quad: 1-based target quadruplet.
        structure: naming for the rerun trajectory.
        frames: shape (nframes, natoms, 3) in nm.
        workdir: stage directory.
        nthreads: mdrun thread count.
        tolerance: maximum permitted absolute deviation in kJ/mol.
        strict_names: treat an atom-name mismatch between ``structure`` and
            ``topology`` as fatal rather than as a warning. Element ordering
            and atom count are always fatal regardless; only *names* are
            relaxed, and only because a topology and a coordinate file can
            legitimately disagree on naming while describing the same
            molecule in the same order. Exposed so the CLI's
            ``--no-strict-names`` -- which
            :func:`ffrefine.io.validate.validate_structure_topology` names in
            its own error message -- actually reaches this call, instead of
            being overridden by a strict re-check here.
        ff_includes: force-field ``#include`` lines to emit ahead of the
            molecule in the generated wrapper ``.top`` (the CLI's
            ``--ff-include``). Needed when the molecule topology has no
            ``[ atomtypes ]`` of its own.
        resolved: T with every term's parameters as grompp built them
            (:func:`ffrefine.mm.tprparams.resolve_torsion_set`). ``mmscan``
            resolves before its ``--resume`` check and passes the result in;
            when omitted it is resolved here, in ``workdir``. Its
            ``full_top`` is the "full" system this check runs.

    Returns:
        The check result when it passes.

    Raises:
        ValidationError: when ``structure`` and ``topology`` disagree (see
            :func:`ffrefine.io.validate.validate_structure_topology`), or when
            agreement is worse than ``tolerance`` anywhere.
    """
    # grompp is invoked with -maxwarn (see single_point_energies), which
    # suppresses its atom-name-mismatch warning -- the only in-band guard
    # against a structure whose atom order disagrees with the topology. With
    # that guard suppressed, grompp would silently proceed using topology
    # order and every frame's energy would be wrong. Check explicitly here,
    # before any energy is computed, and surface non-fatal warnings.
    for warning in validate_structure_topology(structure, topology, strict_names=strict_names):
        log.warning("structure/topology validation: %s", warning)

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if resolved is None:
        resolved = resolve_torsion_set(binaries, topology, quad, structure, workdir, ff_includes)

    zero_itp = workdir / "zeroed.itp"
    removed = write_zeroed_topology(topology, quad, zero_itp)
    zero_top = workdir / "zeroed.top"
    make_system_top(zero_itp, zero_top, topology.name, ff_includes)

    e_full = single_point_energies(
        binaries, resolved.full_top, structure, frames, workdir / "full", "full", nthreads
    )
    e_zero = single_point_energies(
        binaries, zero_top, structure, frames, workdir / "zeroed", "zeroed", nthreads
    )

    full_potential = e_full.potential()
    zeroed_potential = e_zero.potential()
    gromacs_difference = full_potential - zeroed_potential
    # Evaluate the analytic reference at the same quantized coordinates
    # GROMACS actually saw (see _as_gromacs_saw_it), not the caller's raw
    # frames -- otherwise the comparison measures PDB coordinate precision
    # rather than the correctness of the topology edit.
    gmx_frames = _as_gromacs_saw_it(frames)
    analytic_difference = torsion_set_energy(resolved.dihedrals, gmx_frames)
    # The zeroed lines contribute k*(1+cos(...)) with k=0, i.e. nothing, so the
    # analytic sum of the removed terms is the whole expected difference.

    idx = (quad[0] - 1, quad[1] - 1, quad[2] - 1, quad[3] - 1)
    phi = dihedral_deg_frames(gmx_frames, idx)

    deviation = np.abs(gromacs_difference - analytic_difference)
    max_abs_error = float(deviation.max())
    passed = max_abs_error <= tolerance

    result = SelfCheckResult(
        max_abs_error=max_abs_error,
        passed=passed,
        gromacs_difference=gromacs_difference,
        analytic_difference=analytic_difference,
        phi_deg=phi,
        tolerance=tolerance,
        full_potential_kj=full_potential,
        zeroed_potential_kj=zeroed_potential,
    )

    if not passed:
        worst = int(np.argmax(deviation))
        n_typed = sum(t.source == SOURCE_DIHEDRALTYPES for t in resolved.terms)
        raise ValidationError(
            "torsion removal self-consistency check FAILED (spec section 4.5).\n"
            f"  maximum |E_full - E_zeroed - analytic| = {max_abs_error:.6g} kJ/mol "
            f"(tolerance {tolerance:g})\n"
            f"  worst frame {worst}, phi = {phi[worst]:.2f} deg:\n"
            f"    GROMACS difference : {gromacs_difference[worst]:.6f} kJ/mol\n"
            f"    analytic reference : {analytic_difference[worst]:.6f} kJ/mol\n"
            f"  {len(removed)} dihedral term-line(s) were zeroed about the central bond "
            f"({quad[1]}, {quad[2]}). The analytic reference sums {len(resolved.terms)} "
            f"term(s): {len(resolved.terms) - n_typed} written in the topology and {n_typed} "
            "resolved from [ dihedraltypes ], as grompp built them (read back from "
            f"{resolved.dump_path}).\n"
            "  The topology edit did not do what was intended.\n"
            f"  Inspect {zero_itp}, {resolved.dump_path} and the GROMACS logs under "
            f"{workdir}. Fitting is aborted."
        )

    log.info(
        "torsion removal self-check passed: max deviation %.3g kJ/mol over %d frames",
        max_abs_error,
        len(phi),
    )
    return result
