"""Spec section 7.1: the round-trip refit must recover the original parameters.

GROMACS supplies the MM energies with the original parameters, those stand in
for the QM reference, the target torsion is zeroed, and the refit has to come
back with what was removed.

Documented deviation from spec section 7.1's literal ``1e-6`` figure
-------------------------------------------------------------------
Spec line 214 reads "The recovered coefficients must match the originals to
1e-6".  That bound is **not reachable on this build**, through any file-based
energy extraction, and the two tolerances below are therefore set from a
measured noise floor instead.  This is a deliberate, documented deviation from
the spec figure (Task 13 fix round 4, rulings R25/R27), not an accidentally
loosened assertion.

Mechanism.  Every energy this package consumes is extracted from GROMACS as
*text*: ``gmx energy`` writes an ``.xvg`` whose data lines carry exactly six
decimal places -- verified directly on this build (``4.604160``,
``18.461096``, ``25.800283``) -- so each extracted value is quantized to
1e-6 kJ/mol irrespective of its magnitude.  Reading the ``.edr`` underneath
would not help: this is GROMACS 2025.4-conda_forge reporting
``Precision: mixed``, so energies are stored there as ``real`` = float32,
which is comparable or worse at these magnitudes (~3.1e-6 kJ/mol on a
25.8 kJ/mol ``Potential``).  The round trip then *differences* two
independently quantized values (full topology minus zeroed topology), giving
~2e-6 kJ/mol, and the 5-parameter least-squares cosine fit over the removed
set amplifies that by roughly 10x.

The floor is consequently **absolute** -- about 1e-6 kJ/mol per extracted
value -- and scales with neither term magnitude nor coordinate magnitude.
That is precisely why fix round 2's term-magnitude probe came back flat
(1.03-1.16x across a 7.5x range of term magnitude, with the two residual
curves correlated at 0.988) and why fix round 3's coordinate-magnitude probe
came back flat as well (<2%).

Measured floor, 36 frames, deterministic (bit-identical across three repeat
runs in fresh working directories):

    fit rmse                        1.031101e-05 kJ/mol
    max |predicted - reference|     1.797718e-05 kJ/mol

The tolerances below sit ~10x above those numbers and deliberately no higher:
a bound loose enough to pass on anything would not be testing anything.

How to tighten this.  A double-precision GROMACS (``gmx_d`` / ``gmx_mpi_d``,
neither present in this conda environment -- ``gmx`` is the only binary there)
writes more significant digits and stores double in the ``.edr``, and is
expected to reach the spec's ``1e-6``.  Before changing these constants,
re-run the measurement that established the floor's absoluteness in the first
place -- the *translation probe*: repeat the frame scan with the whole
molecule translated by 0, 1 and 5 nm, and measure the spread of the **zeroed**
topology's remaining ``Proper Dih.`` term (the methyl torsions about the two
untouched terminal bonds, geometrically invariant under rotation about the
central bond and therefore exactly constant, so any spread is pure numerical
noise).  It measured 4.0e-6 / 1.0e-6 / 4.0e-6 kJ/mol -- flat across a 20x
change in coordinate magnitude, and every value an exact integer multiple of
1e-6, which is what pointed at the ``.xvg`` text format to begin with.
"""

import numpy as np
import pytest
from ffrefine.config import Binaries
from ffrefine.fit.linear import fit_cosine_series
from ffrefine.io.structure import read_pdb
from ffrefine.io.topology import read_topology
from ffrefine.io.topwriter import make_system_top, write_zeroed_topology
from ffrefine.mm import gromacs
from ffrefine.mm.torsionset import collect_torsion_set


# Round-trip tolerances, set from the measured floor documented in the module
# docstring above rather than from spec 7.1's literal 1e-6.  Do not raise these
# without re-running the translation probe described there.
ROUND_TRIP_RMSE_TOL = 1e-4
"""kJ/mol.  Measured floor 1.031101e-05 -> 9.7x headroom (spec 7.1 says 1e-6)."""

ROUND_TRIP_MAX_DEV_TOL = 2e-4
"""kJ/mol.  Measured floor 1.797718e-05 -> 11.1x headroom (spec 7.1 says 1e-6)."""


@pytest.mark.needs_gmx
def test_round_trip_refit_recovers_original_parameters(
    tmp_path, butane_pdb, butane_itp, gmx_bin, rotated_frames
):
    """Spec section 7.1: refit must recover the original coefficients.

    The whole pipeline except xtb and ORCA is exercised here.
    """
    s = read_pdb(butane_pdb)
    t = read_topology(butane_itp)
    frames = rotated_frames(s, 36)
    b = Binaries(gmx=gmx_bin)

    full_itp = tmp_path / "full.itp"
    full_itp.write_text(butane_itp.read_text())
    full_top = tmp_path / "full.top"
    make_system_top(full_itp, full_top, "BUT")

    zero_itp = tmp_path / "zeroed.itp"
    removed = write_zeroed_topology(t, (1, 5, 8, 11), zero_itp)
    zero_top = tmp_path / "zeroed.top"
    make_system_top(zero_itp, zero_top, "BUT")

    e_full = gromacs.single_point_energies(b, full_top, s, frames, tmp_path / "f", "full")
    e_zero = gromacs.single_point_energies(b, zero_top, s, frames, tmp_path / "z", "zero")

    from ffrefine.geometry import dihedral_deg_frames

    phi = dihedral_deg_frames(frames, (0, 4, 7, 10))

    # Treat the full-topology MM energies as the "QM" reference (spec 7.1).
    qm = e_full.potential() - e_full.potential().min()
    mm0 = e_zero.potential() - e_zero.potential().min()
    target = qm - mm0

    r = fit_cosine_series(phi, target, [1, 2, 3, 4, 6])

    # The removed set is nine terms about the same central bond; their summed
    # analytic profile is what the fit must reproduce.
    from ffrefine.mm.analytic import torsion_set_energy

    reference = torsion_set_energy(removed, frames)

    # Spec section 7.1 is literally about the *coefficients* ("the recovered
    # coefficients must match the originals"), not only the reconstructed
    # energy curve -- pin those down directly (Task 13 fix round 1, ruling
    # R22). "The originals" here means the coefficients an exact, noiseless
    # fit of the analytic reference itself would recover, not a hand-summed
    # total of the nine members' force constants: a single-phi cosine
    # decomposition of nine dihedral terms at slightly different geometric
    # phases does not reduce to the naive sum of their amplitudes unless
    # those phases align exactly, which this fixture's real (non-ideal)
    # geometry does not guarantee. Confirmed empirically: fitting the exact
    # reference curve (zero noise, no GROMACS involved) reproduces itself to
    # rmse ~4e-15, so its own fitted k/delta_deg is the correct ground truth.
    truth = fit_cosine_series(phi, reference, [1, 2, 3, 4, 6])
    np.testing.assert_array_equal(r.mults, truth.mults)
    np.testing.assert_allclose(r.k, truth.k, atol=1e-4)
    # delta_deg is only meaningful where the amplitude is not numerically
    # zero: fit_cosine_series assigns 0 or 180 by the sign of a near-zero
    # coefficient, which is arbitrary noise for the mults (1, 4, 6 here) this
    # torsion set does not actually contain.
    significant = truth.k > 1e-3
    np.testing.assert_allclose(r.delta_deg[significant], truth.delta_deg[significant], atol=1e-2)

    reference = reference - reference.min()
    predicted = r.predicted - r.predicted.min()
    # Spec 7.1 asks for 1e-6 here; the .xvg six-decimal quantization floor over
    # a float32 .edr in this mixed-precision build makes that unreachable.  See
    # the module docstring: measured 1.797718e-05, tolerance carries 11.1x.
    assert np.max(np.abs(predicted - reference)) < ROUND_TRIP_MAX_DEV_TOL


@pytest.mark.needs_gmx
def test_round_trip_residuals_are_numerically_zero(
    tmp_path, butane_pdb, butane_itp, gmx_bin, rotated_frames
):
    s = read_pdb(butane_pdb)
    t = read_topology(butane_itp)
    frames = rotated_frames(s, 36)
    b = Binaries(gmx=gmx_bin)

    full_itp = tmp_path / "full.itp"
    full_itp.write_text(butane_itp.read_text())
    full_top = tmp_path / "full.top"
    make_system_top(full_itp, full_top, "BUT")
    zero_itp = tmp_path / "zeroed.itp"
    write_zeroed_topology(t, (1, 5, 8, 11), zero_itp)
    zero_top = tmp_path / "zeroed.top"
    make_system_top(zero_itp, zero_top, "BUT")

    e_full = gromacs.single_point_energies(b, full_top, s, frames, tmp_path / "f", "full")
    e_zero = gromacs.single_point_energies(b, zero_top, s, frames, tmp_path / "z", "zero")

    from ffrefine.geometry import dihedral_deg_frames

    phi = dihedral_deg_frames(frames, (0, 4, 7, 10))
    target = (e_full.potential() - e_full.potential().min()) - (
        e_zero.potential() - e_zero.potential().min()
    )
    r = fit_cosine_series(phi, target, [1, 2, 3, 4, 6])
    # Spec 7.1 asks for 1e-6 here; the .xvg six-decimal quantization floor over
    # a float32 .edr in this mixed-precision build makes that unreachable.  See
    # the module docstring: measured 1.031101e-05, tolerance carries 9.7x.
    assert r.rmse < ROUND_TRIP_RMSE_TOL
