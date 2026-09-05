import numpy as np
import pytest
from ffrefine.config import Binaries
from ffrefine.fit.linear import fit_cosine_series
from ffrefine.io.structure import read_pdb
from ffrefine.io.topology import read_topology
from ffrefine.io.topwriter import make_system_top, write_zeroed_topology
from ffrefine.mm import gromacs
from ffrefine.mm.torsionset import collect_torsion_set


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
    assert np.max(np.abs(predicted - reference)) < 1e-6


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
    assert r.rmse < 1e-6
