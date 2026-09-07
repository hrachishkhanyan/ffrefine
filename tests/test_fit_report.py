import numpy as np
import pytest
from ffrefine.errors import ValidationError
from ffrefine.fit.linear import fit_cosine_series
from ffrefine.fit.report import fit_summary, profile_stats
from ffrefine.geometry import wrap_deg
from ffrefine.mm.analytic import series_energy

# R6 override: np.linspace(-180, 180, 361) contains both -180 and +180, which
# wrap_deg maps onto the same angle -- the duplicate registers as two minima
# or none, making the brief's asserted counts unreachable. One sample per
# degree over the periodic domain is the correct grid.
_GRID = np.linspace(-180, 180, 360, endpoint=False)


def test_threefold_profile_has_three_minima():
    phi = _GRID
    e = series_energy(np.array([2.0]), np.array([0.0]), np.array([3]), phi)
    s = profile_stats(phi, e)
    assert len(s.minima_deg) == 3


def test_minima_of_threefold_are_at_180_and_plus_minus_60():
    phi = _GRID
    e = series_energy(np.array([2.0]), np.array([0.0]), np.array([3]), phi)
    s = profile_stats(phi, e)
    found = sorted(abs(round(m)) for m in s.minima_deg)
    assert found == [60, 60, 180]


def test_barrier_height_of_threefold():
    """A single k=2 kJ/mol n=3 term has 4 kJ/mol barriers."""
    phi = _GRID
    e = series_energy(np.array([2.0]), np.array([0.0]), np.array([3]), phi)
    s = profile_stats(phi, e)
    assert s.max_barrier_kj == pytest.approx(4.0, abs=0.05)


def test_global_minimum_reported():
    """Finding (see task-17-report.md): the brief's own check --
    ``e[np.argmin(np.abs(phi - s.global_min_deg))]`` -- uses plain, non-periodic
    distance to relocate the reported angle in the raw grid.  Under the R6 grid
    the true minimum of this profile sits exactly at the wrap boundary
    (raw grid point phi=-180, energy 0.0), and profile_stats correctly reports
    it in the project's (-180, 180] canonical form as +180.0 (spec section
    9.6). The raw _GRID array itself never contains +180 (only -180), so a
    non-periodic nearest-match against the raw array picks phi=179 instead and
    the brief's literal assertion fails -- not because profile_stats is wrong,
    but because that lookup is not periodicity-aware. The brief's defective
    361-point grid (npoints=361, endpoint=True) masked this because +180 was a
    literal duplicate sample there. The fix below is a wrap-aware distance,
    not a loosened tolerance or a changed expected value.
    """
    phi = _GRID
    e = series_energy(np.array([2.0, 1.0]), np.array([0.0, 0.0]), np.array([3, 1]), phi)
    s = profile_stats(phi, e)
    nearest = np.argmin(np.abs(wrap_deg(phi - s.global_min_deg)))
    assert e[nearest] == pytest.approx(e.min(), abs=1e-6)


def test_minimum_at_the_wrap_boundary_is_found():
    """A minimum sitting at +/-180 must not be missed by an interior-only scan."""
    phi = _GRID
    e = series_energy(np.array([1.0]), np.array([0.0]), np.array([1]), phi)
    s = profile_stats(phi, e)
    assert len(s.minima_deg) == 1
    assert abs(s.minima_deg[0]) == pytest.approx(180.0, abs=1e-6)


def test_fit_summary_is_json_serialisable():
    import json

    phi = np.linspace(-180, 180, 73, endpoint=False)
    target = series_energy(np.array([2.0]), np.array([0.0]), np.array([3]), phi)
    r = fit_cosine_series(phi, target, [1, 2, 3])
    summary = fit_summary(r, phi, target)
    json.dumps(summary)  # must not raise
    assert summary["rmse_kj_per_mol"] == pytest.approx(0.0, abs=1e-9)
    assert summary["multiplicities"] == [1, 2, 3]


def test_fit_summary_reports_kcal_alongside_kj():
    phi = np.linspace(-180, 180, 73, endpoint=False)
    target = series_energy(np.array([4.184]), np.array([0.0]), np.array([3]), phi)
    r = fit_cosine_series(phi, target, [3])
    summary = fit_summary(r, phi, target)
    assert summary["terms"][0]["k_kcal_per_mol"] == pytest.approx(1.0, abs=1e-6)


def test_profile_stats_rejects_shape_mismatch():
    phi = np.linspace(-180, 180, 10, endpoint=False)
    e = np.zeros(9)
    with pytest.raises(ValidationError, match="shape"):
        profile_stats(phi, e)


def test_profile_stats_rejects_empty_input():
    with pytest.raises(ValidationError, match="empty"):
        profile_stats(np.array([]), np.array([]))


# --------------------------------------------------------------------------
# labelling (final review, item 7)
# --------------------------------------------------------------------------


def test_fit_summary_labels_the_correction_curve_not_the_torsion_profile():
    """Item 7: the stats are of the fit of dE = QM - MM0 -- the correction --
    not the torsion profile. The old key was a bare "profile", which a user
    holding a "syn barrier 14-20 kJ/mol" expectation reads as the torsion's."""
    phi = np.linspace(-180.0, 180.0, 24, endpoint=False)
    target = 5.0 * (1.0 + np.cos(np.radians(3.0 * phi)))
    result = fit_cosine_series(phi, target, [3])
    payload = fit_summary(result, phi, target)
    assert "profile" not in payload
    block = payload["correction_curve_profile"]
    assert "NOT the torsion" in block["what_this_describes"]
    assert "max_barrier_kj_per_mol" in block
    # No QM profile unless one is supplied.
    assert "qm_profile" not in payload


def test_fit_summary_reports_the_qm_profile_separately():
    phi = np.linspace(-180.0, 180.0, 24, endpoint=False)
    target = 5.0 * (1.0 + np.cos(np.radians(3.0 * phi)))
    qm = 14.0 * (1.0 + np.cos(np.radians(3.0 * phi)))
    result = fit_cosine_series(phi, target, [3])
    payload = fit_summary(result, phi, target, qm_kj=qm)
    qm_block = payload["qm_profile"]
    assert "ARE torsion minima and barriers" in qm_block["what_this_describes"]
    # The two blocks describe genuinely different curves.
    assert qm_block["max_barrier_kj_per_mol"] == pytest.approx(28.0, abs=0.5)
    assert payload["correction_curve_profile"]["max_barrier_kj_per_mol"] == pytest.approx(
        10.0, abs=0.5
    )
