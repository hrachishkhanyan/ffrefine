"""Post-fit validation (spec section 4.9).

Rebuild the topology with the fitted parameters, recompute MM single points on
the same geometries, and compare the total MM energy against QM before and
after.  Spec section 4.9: this is the number that matters.

Both curves compared by :func:`profile_rmse` are shifted to *their own*
minimum before the RMSE is taken -- QM's absolute energy scale and MM's
absolute energy scale have no relationship to one another (different
reference states, different levels of theory), so only relative energies are
meaningful.  Using one shared offset (e.g. always subtracting the QM
minimum) would leave a spurious residual baked into the "before" and "after"
comparisons and silently distort the one number spec 4.9 calls out as the
one that matters -- see the regression test
``test_profile_rmse_shared_offset_would_distort_the_result``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Binaries
from ..geometry import dihedral_deg_frames
from ..io.structure import Structure
from ..io.topology import Topology
from ..io.topwriter import make_system_top, write_refit_topology
from ..io.validate import validate_structure_topology
from .gromacs import single_point_energies
from .torsionset import validate_quad

log = logging.getLogger("ffrefine")

Quad = tuple[int, int, int, int]


@dataclass
class ValidationResult:
    """MM-against-QM agreement before and after refitting.

    ``mm_before_kj``, ``mm_after_kj`` and ``qm_kj`` are the raw (un-zeroed)
    profiles; :func:`ffrefine.plots.plot_validation` zeros each to its own
    minimum itself, so callers must not zero these before plotting them.

    ``mm_before_kj`` reflects whatever topology :func:`validate_refit` was
    actually called with -- see that function's "before" note for why this
    must be the original force field's topology, not a zeroed one, in real
    pipeline usage.
    """

    rmse_before: float
    rmse_after: float
    mm_before_kj: np.ndarray
    mm_after_kj: np.ndarray
    qm_kj: np.ndarray
    phi_deg: np.ndarray
    improved: bool


def profile_rmse(a_kj: np.ndarray, b_kj: np.ndarray) -> float:
    """RMSE between two energy profiles, each shifted to its own minimum.

    Absolute energies are not comparable between QM and MM (or between two
    different MM parameterisations), so both curves are referenced to their
    own minimum independently before differencing -- not to a shared offset
    (spec section 4.9).

    Args:
        a_kj: first energy profile, kJ/mol.
        b_kj: second energy profile, kJ/mol; same shape as ``a_kj``.

    Returns:
        The RMSE, in kJ/mol, between the two minimum-referenced curves.
    """
    a = np.asarray(a_kj, dtype=float)
    b = np.asarray(b_kj, dtype=float)
    return float(np.sqrt(np.mean(((a - a.min()) - (b - b.min())) ** 2)))


def _is_improved(rmse_before: float, rmse_after: float) -> bool:
    """Whether refitting improved MM/QM agreement.

    Strict inequality: an exact tie leaves agreement unchanged, so it does
    not count as an improvement. This is the one place that decision is made.
    """
    return rmse_after < rmse_before


def validate_refit(
    binaries: Binaries,
    topology: Topology,
    quad: Quad,
    structure: Structure,
    frames: np.ndarray,
    qm_kj: np.ndarray,
    k: np.ndarray,
    delta_deg: np.ndarray,
    mult: np.ndarray,
    workdir: str | Path,
    distribute: bool = False,
    nthreads: int = 1,
) -> ValidationResult:
    """Compare MM against QM before and after applying the fitted parameters.

    "Before" is ``topology`` exactly as given, rebuilt and re-run unchanged;
    "after" is the same topology with the fitted series written over the
    target torsion set via :func:`ffrefine.io.topwriter.write_refit_topology`.
    Both are rebuilt into ``workdir`` and re-run through the same
    single-point MM engine on the same ``frames`` used for the QM reference,
    so the two RMSE values are directly comparable.

    **Callers must pass the ORIGINAL force-field topology as** ``topology``
    **-- the one carrying the force field's own, pre-fit parameters for the
    target torsion -- never a topology with the target torsion set already
    zeroed out.** Spec section 4.9's "before refitting" means the original
    force field's torsion term, and ``rmse_before``/``rmse_after`` are the
    headline numbers the whole tool is judged by (spec 4.9: "this is the
    number that matters"). Passing a *zeroed* topology instead inflates
    ``rmse_before``: it then measures "how much this fit improves on having
    no torsion term at all" rather than "how much this fit improves on the
    force field's original guess", making the reported improvement look
    larger than it really is, with nothing downstream able to catch the
    substitution. ``write_refit_topology`` *replaces* the target torsion
    set's lines outright, so this function computes a correct RMSE for
    whatever topology it is actually given either way -- this requirement is
    about what the caller must supply, not a limitation of this function's
    own logic. A zeroed "before" is legitimate only as a deliberate,
    test-only construction used to manufacture a large, unambiguous
    improvement to exercise this code path (see
    ``tests/test_validate_fit.py::test_refit_reproduces_the_original_profile``
    and its docstring) -- never in real pipeline usage.

    Args:
        binaries: resolved binaries; ``gmx`` required.
        topology: the ORIGINAL topology to refit, as currently parsed --
            i.e. carrying the force field's own pre-fit parameters for the
            target torsion, not a zeroed stand-in. See the "before" note
            above.
        quad: 1-based target quadruplet.
        structure: naming for the rerun trajectory.
        frames: the same geometries the QM profile was computed on, shape
            (nframes, natoms, 3) in nm.
        qm_kj: QM energies in kJ/mol, one per frame.
        k: fitted force constants in kJ/mol.
        delta_deg: fitted phases in degrees.
        mult: retained multiplicities.
        workdir: stage directory; rebuilt topologies and GROMACS logs are
            written here, never into the input files.
        distribute: spread the fitted series over every torsion sharing the
            target's central bond (spec section 9.7), rather than writing it
            onto the target quadruplet alone. Passed straight through to
            :func:`write_refit_topology`.
        nthreads: mdrun thread count.

    Returns:
        RMSE before and after, and the profiles behind them.

    Raises:
        ValidationError: ``quad`` is malformed (delegated to
            :func:`ffrefine.mm.torsionset.validate_quad`), or ``structure``
            and ``topology`` describe different atom counts/elements
            (delegated to
            :func:`ffrefine.io.validate.validate_structure_topology`).
    """
    validate_quad(topology, quad)
    # grompp is invoked with -maxwarn (see single_point_energies), which
    # suppresses its atom-name-mismatch warning -- the only in-band guard
    # against a structure whose atom order disagrees with the topology.
    # Check explicitly before spending any GROMACS time on it.
    for warning in validate_structure_topology(structure, topology):
        log.warning("structure/topology validation: %s", warning)

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    before_itp = workdir / "before.itp"
    before_itp.write_text(Path(topology.source).read_text())
    before_top = workdir / "before.top"
    make_system_top(before_itp, before_top, topology.name)

    after_itp = workdir / "after.itp"
    write_refit_topology(topology, quad, k, delta_deg, mult, after_itp, distribute=distribute)
    after_top = workdir / "after.top"
    make_system_top(after_itp, after_top, topology.name)

    e_before = single_point_energies(
        binaries, before_top, structure, frames, workdir / "before", "before", nthreads
    )
    e_after = single_point_energies(
        binaries, after_top, structure, frames, workdir / "after", "after", nthreads
    )

    mm_before = e_before.potential()
    mm_after = e_after.potential()
    qm = np.asarray(qm_kj, dtype=float)
    rmse_before = profile_rmse(qm, mm_before)
    rmse_after = profile_rmse(qm, mm_after)

    # quad is 1-based (user-facing, topology-file convention); frames is
    # indexed 0-based, so convert here at the boundary.
    idx = (quad[0] - 1, quad[1] - 1, quad[2] - 1, quad[3] - 1)
    phi = dihedral_deg_frames(np.asarray(frames, dtype=float), idx)

    log.info(
        "validation: MM against QM RMSE %.3f kJ/mol before, %.3f kJ/mol after refitting",
        rmse_before,
        rmse_after,
    )
    improved = _is_improved(rmse_before, rmse_after)
    if not improved:
        log.warning(
            "refitting did not improve agreement (%.3f -> %.3f kJ/mol). Check the scan "
            "for conformational hops and the self-consistency diagnostics.",
            rmse_before,
            rmse_after,
        )

    return ValidationResult(
        rmse_before=rmse_before,
        rmse_after=rmse_after,
        mm_before_kj=mm_before,
        mm_after_kj=mm_after,
        qm_kj=qm,
        phi_deg=phi,
        improved=improved,
    )
