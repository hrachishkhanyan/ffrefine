"""The mandatory torsion-removal self-consistency check (spec section 4.5).

E_full(phi) - E_zeroed(phi) computed by GROMACS must equal the analytic sum of
the removed torsion terms, evaluated at the measured dihedral values with the
original parameters, to better than 1e-3 kJ/mol at every point. The analytic
side is evaluated at the coordinates GROMACS actually received (see
``_as_gromacs_saw_it``): the multi-model PDB used to drive ``mdrun -rerun``
keeps only three decimal places of Angstrom precision, and comparing against
the caller's full-precision input instead would measure that format's
coordinate quantization rather than the correctness of the topology edit.

Failing this means the topology edit did not do what was intended -- most
often that explicit parameters did not suppress the [ dihedraltypes ] lookup as
assumed.  On failure the pipeline aborts; it never proceeds to fitting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Binaries
from ..errors import ValidationError
from ..geometry import dihedral_deg_frames
from ..io.structure import ANGSTROM_TO_NM, Structure
from ..io.topology import Topology
from ..io.topwriter import make_system_top, write_zeroed_topology
from ..io.validate import validate_structure_topology
from .analytic import torsion_set_energy
from .gromacs import single_point_energies

log = logging.getLogger("ffrefine")

DEFAULT_TOLERANCE = 1e-3
"""kJ/mol; spec section 4.5."""

_PDB_COORD_DECIMALS = 3
"""Decimal places ``write_multimodel_pdb`` keeps, in Angstrom (PDB %8.3f)."""


def _as_gromacs_saw_it(frames: np.ndarray) -> np.ndarray:
    """Quantize coordinates to what GROMACS actually received.

    ``single_point_energies`` drives ``mdrun -rerun`` from a multi-model PDB
    (:func:`ffrefine.io.structure.write_multimodel_pdb`), whose fixed-width
    Angstrom fields keep only three decimal places -- about 1e-4 nm per atom.
    Evaluating the analytic reference at the caller's full double-precision
    ``frames`` instead of this quantized geometry compares two different
    inputs, not two evaluations of the same one: the residual it measures is
    the PDB format's coordinate precision, not whether the topology edit is
    correct. Confirmed empirically (Task 13): rounding here reduces the
    observed discrepancy on the butane fixture from ~1.6e-3 kJ/mol -- enough
    to spuriously fail the section 4.5 tolerance -- to ~3e-5 kJ/mol, in line
    with GROMACS's own single-precision .edr/.xvg output noise.

    Args:
        frames: shape (nframes, natoms, 3) in nm.

    Returns:
        The same coordinates, rounded to the PDB trajectory's precision.
    """
    angstrom = np.asarray(frames, dtype=float) / ANGSTROM_TO_NM
    return np.round(angstrom, _PDB_COORD_DECIMALS) * ANGSTROM_TO_NM


@dataclass
class SelfCheckResult:
    """Outcome of the section 4.5 check."""

    max_abs_error: float
    passed: bool
    gromacs_difference: np.ndarray
    analytic_difference: np.ndarray
    phi_deg: np.ndarray
    tolerance: float


def check_torsion_removal(
    binaries: Binaries,
    topology: Topology,
    quad: tuple[int, int, int, int],
    structure: Structure,
    frames: np.ndarray,
    workdir: str | Path,
    nthreads: int = 1,
    tolerance: float = DEFAULT_TOLERANCE,
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
    for warning in validate_structure_topology(structure, topology):
        log.warning("structure/topology validation: %s", warning)

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    full_itp = workdir / "full.itp"
    full_itp.write_text(Path(topology.source).read_text())
    full_top = workdir / "full.top"
    make_system_top(full_itp, full_top, topology.name)

    zero_itp = workdir / "zeroed.itp"
    removed = write_zeroed_topology(topology, quad, zero_itp)
    zero_top = workdir / "zeroed.top"
    make_system_top(zero_itp, zero_top, topology.name)

    e_full = single_point_energies(
        binaries, full_top, structure, frames, workdir / "full", "full", nthreads
    )
    e_zero = single_point_energies(
        binaries, zero_top, structure, frames, workdir / "zeroed", "zeroed", nthreads
    )

    gromacs_difference = e_full.potential() - e_zero.potential()
    # Evaluate the analytic reference at the same quantized coordinates
    # GROMACS actually saw (see _as_gromacs_saw_it), not the caller's raw
    # frames -- otherwise the comparison measures PDB coordinate precision
    # rather than the correctness of the topology edit.
    gmx_frames = _as_gromacs_saw_it(frames)
    analytic_difference = torsion_set_energy(removed, gmx_frames)
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
    )

    if not passed:
        worst = int(np.argmax(deviation))
        raise ValidationError(
            "torsion removal self-consistency check FAILED (spec section 4.5).\n"
            f"  maximum |E_full - E_zeroed - analytic| = {max_abs_error:.6g} kJ/mol "
            f"(tolerance {tolerance:g})\n"
            f"  worst frame {worst}, phi = {phi[worst]:.2f} deg:\n"
            f"    GROMACS difference : {gromacs_difference[worst]:.6f} kJ/mol\n"
            f"    analytic reference : {analytic_difference[worst]:.6f} kJ/mol\n"
            f"  {len(removed)} dihedral entries were zeroed about the central bond "
            f"({quad[1]}, {quad[2]}).\n"
            "  The topology edit did not do what was intended -- most likely the "
            "explicit zero parameters did not suppress the [ dihedraltypes ] lookup. "
            f"  Inspect {zero_itp} and the GROMACS logs under {workdir}. "
            "Fitting is aborted."
        )

    log.info(
        "torsion removal self-check passed: max deviation %.3g kJ/mol over %d frames",
        max_abs_error,
        len(phi),
    )
    return result
