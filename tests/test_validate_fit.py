"""Tests for the spec 4.9 validation stage.

See the module docstring in ``ffrefine/mm/validate_fit.py`` for why both
curves in :func:`profile_rmse` are shifted to their own minimum rather than a
shared one.

The task-21 brief's own GROMACS round-trip test used the *same* unmodified
topology as both the QM stand-in and the "before" MM profile, which pins
``rmse_before`` to the ~1e-5 kJ/mol GROMACS text-precision floor (see
``tests/test_roundtrip.py``'s module docstring) that no imperfect refit could
ever beat -- ``rmse_after < rmse_before`` could never pass honestly under that
setup. ``test_refit_reproduces_the_original_profile`` below instead builds
"before" from the *zeroed* topology (the state before any torsion has been
fit for this bond), so the improvement asserted is real and measurable.
"""

from __future__ import annotations

import numpy as np
import pytest

from ffrefine.errors import ValidationError
from ffrefine.mm.validate_fit import ValidationResult, _is_improved, profile_rmse, validate_refit


def test_profile_rmse_is_zero_for_identical_shapes():
    phi = np.linspace(-180, 180, 37)
    a = np.cos(np.deg2rad(phi)) * 5.0
    assert profile_rmse(a, a) == pytest.approx(0.0, abs=1e-12)


def test_profile_rmse_ignores_constant_offset():
    """Only relative energies matter, so a rigid shift must not count."""
    phi = np.linspace(-180, 180, 37)
    a = np.cos(np.deg2rad(phi)) * 5.0
    assert profile_rmse(a, a + 1000.0) == pytest.approx(0.0, abs=1e-9)


def test_profile_rmse_detects_shape_difference():
    phi = np.linspace(-180, 180, 37)
    a = np.cos(np.deg2rad(phi)) * 5.0
    b = np.cos(2 * np.deg2rad(phi)) * 5.0
    assert profile_rmse(a, b) > 1.0


def test_profile_rmse_shared_offset_would_distort_the_result():
    """Pin the per-curve zeroing spec 4.9 requires.

    QM and MM live on unrelated absolute energy scales, so ``profile_rmse``
    must zero each curve to its *own* minimum. A shared-offset implementation
    -- e.g. zeroing both curves against only the first curve's minimum --
    would leave the gap between the two scales baked into the result even
    when the underlying shapes are identical.
    """
    phi = np.linspace(-180, 180, 19)
    shape = np.cos(np.deg2rad(phi)) * 2.0
    qm = shape + 1000.0  # arbitrary QM absolute reference
    mm = shape + 5.0  # arbitrary, unrelated MM absolute reference

    per_curve = profile_rmse(qm, mm)
    shared_offset = float(np.sqrt(np.mean(((qm - qm.min()) - (mm - qm.min())) ** 2)))

    assert per_curve == pytest.approx(0.0, abs=1e-9)
    assert shared_offset > 100.0
    assert abs(per_curve - shared_offset) > 100.0


def test_validation_result_reports_improvement():
    r = ValidationResult(
        rmse_before=4.0,
        rmse_after=0.5,
        mm_before_kj=np.zeros(3),
        mm_after_kj=np.zeros(3),
        qm_kj=np.zeros(3),
        phi_deg=np.zeros(3),
        improved=True,
    )
    assert r.improved


def test_is_improved_strict_inequality():
    assert _is_improved(4.0, 0.5) is True
    assert _is_improved(0.5, 4.0) is False


def test_is_improved_exact_tie_is_not_an_improvement():
    """Decision point: an exact tie means nothing changed for the better."""
    assert _is_improved(1.0, 1.0) is False


def test_validate_refit_rejects_bad_quad(tmp_path, butane_itp):
    """A malformed quad must be rejected before any GROMACS work happens.

    Task 18's ruling: a writer taking a raw ``quad`` validates it up front
    (see ``ffrefine/io/prm.py``); ``validate_refit`` does the same via
    :func:`ffrefine.mm.torsionset.validate_quad`, and does so before touching
    ``binaries`` -- this test passes no ``gmx`` binary at all and must still
    fail with ``ValidationError``, not an unrelated crash later on.
    """
    from ffrefine.config import Binaries
    from ffrefine.io.topology import read_topology

    t = read_topology(butane_itp)
    with pytest.raises(ValidationError):
        validate_refit(
            Binaries(),  # no gmx configured -- must never be reached
            t,
            (1, 5, 8, 99),  # atom 99 does not exist
            structure=None,
            frames=np.zeros((1, 14, 3)),
            qm_kj=np.zeros(1),
            k=np.array([1.0]),
            delta_deg=np.array([0.0]),
            mult=np.array([3]),
            workdir=tmp_path / "val",
        )


@pytest.mark.needs_gmx
def test_refit_reproduces_the_original_profile(
    tmp_path, butane_pdb, butane_itp, gmx_bin, rotated_frames
):
    """Round trip through the validation stage: refitting a zeroed torsion
    set must bring MM back into much closer agreement with QM.

    Central bond 5-8 carries nine members in the fixture: the target quad
    (1,5,8,11), with its own two-term series (n=3 k=0.58576, n=2 k=0.41840),
    plus eight others that all share a single n=3, k=0.66944, phase=0 term.
    "Before" is the topology with the whole set zeroed (the state before any
    fit was performed); refitting with ``distribute=True`` and
    ``k=9*0.66944`` spreads that value back over all nine members equally,
    exactly reconstructing the eight non-target members and giving the
    target quad a close (not exact -- the true target series also carries a
    distinct n=2 component this single-term fit cannot represent) single-term
    approximation of its own series.
    """
    from ffrefine.config import Binaries
    from ffrefine.io.structure import read_pdb
    from ffrefine.io.topology import read_topology
    from ffrefine.io.topwriter import make_system_top, write_zeroed_topology
    from ffrefine.mm import gromacs
    from ffrefine.plots import plot_validation

    s = read_pdb(butane_pdb)
    t_full = read_topology(butane_itp)
    frames = rotated_frames(s, 24)
    b = Binaries(gmx=gmx_bin)

    # QM stand-in: the true (full-parameter) topology's own MM energies.
    full_itp = tmp_path / "full.itp"
    full_itp.write_text(butane_itp.read_text())
    full_top = tmp_path / "full.top"
    make_system_top(full_itp, full_top, "BUT")
    e_full = gromacs.single_point_energies(b, full_top, s, frames, tmp_path / "q", "q")
    qm = e_full.potential()

    # "before": the target torsion set zeroed out entirely.
    zeroed_itp = tmp_path / "zeroed.itp"
    write_zeroed_topology(t_full, (1, 5, 8, 11), zeroed_itp)
    t_zeroed = read_topology(zeroed_itp)

    r = validate_refit(
        b,
        t_zeroed,
        (1, 5, 8, 11),
        s,
        frames,
        qm,
        np.array([9 * 0.66944]),
        np.array([0.0]),
        np.array([3]),
        tmp_path / "val",
        distribute=True,
    )

    assert r.rmse_after < r.rmse_before
    assert r.improved is True
    assert r.phi_deg.shape == qm.shape
    np.testing.assert_array_equal(r.qm_kj, qm)

    plot_path = tmp_path / "validation.png"
    plot_validation(
        r.phi_deg, r.qm_kj, r.mm_before_kj, r.mm_after_kj, r.rmse_before, r.rmse_after, plot_path
    )
    assert plot_path.exists()


@pytest.mark.needs_gmx
def test_a_worse_refit_logs_info_not_a_second_warning(
    tmp_path, butane_pdb, butane_itp, gmx_bin, rotated_frames, caplog
):
    """Final review M3: cli.cmd_validate's own ``_warn`` already tells the user a refit did
    not improve agreement, pointing at the 02_mm_scan diagnostics; validate_fit's own
    log.warning duplicated it as a second WARNING record once the logging-layer collector
    started catching every module's own warnings, not only cli._warn's. Downgraded to
    log.info here, so only one warning per not-improved bond reaches validation.json/report.md.
    """
    import logging

    from ffrefine.config import Binaries
    from ffrefine.io.structure import read_pdb
    from ffrefine.io.topology import read_topology
    from ffrefine.io.topwriter import make_system_top
    from ffrefine.mm import gromacs

    s = read_pdb(butane_pdb)
    t_full = read_topology(butane_itp)
    frames = rotated_frames(s, 24)
    b = Binaries(gmx=gmx_bin)

    full_itp = tmp_path / "full.itp"
    full_itp.write_text(butane_itp.read_text())
    full_top = tmp_path / "full.top"
    make_system_top(full_itp, full_top, "BUT")
    e_full = gromacs.single_point_energies(b, full_top, s, frames, tmp_path / "q", "q")
    qm = e_full.potential()

    caplog.set_level(logging.INFO, logger="ffrefine")
    # "before" is the true, already-correct topology (rmse_before ~ 0); zeroing
    # the central bond's torsion for "after" can only make agreement worse.
    r = validate_refit(
        b,
        t_full,
        (1, 5, 8, 11),
        s,
        frames,
        qm,
        np.array([0.0]),
        np.array([0.0]),
        np.array([3]),
        tmp_path / "val",
        distribute=True,
    )
    assert r.improved is False

    messages = [(rec.levelname, rec.getMessage()) for rec in caplog.records]
    assert any(level == "INFO" and "did not improve" in msg for level, msg in messages)
    assert not any(level == "WARNING" and "did not improve" in msg for level, msg in messages)


@pytest.mark.needs_gmx
def test_compare_topologies_of_one_topology_with_itself_agrees(
    tmp_path, butane_pdb, butane_itp, gmx_bin, rotated_frames
):
    from ffrefine.config import Binaries
    from ffrefine.io.structure import read_pdb
    from ffrefine.io.topology import read_topology
    from ffrefine.mm.validate_fit import compare_topologies

    s = read_pdb(butane_pdb)
    t = read_topology(butane_itp)
    frames = rotated_frames(s, 6)
    qm = np.linspace(0.0, 5.0, 6)
    r = compare_topologies(Binaries(gmx=gmx_bin), t, t, s, frames, qm, (1, 5, 8, 11), tmp_path)
    assert np.array_equal(r.mm_before_kj, r.mm_after_kj)
    assert r.rmse_before == r.rmse_after
    assert r.phi_deg.shape == (6,)
