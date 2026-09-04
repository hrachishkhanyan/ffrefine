import numpy as np
import pytest
from ffrefine.io.topology import Dihedral, read_topology
from ffrefine.mm import analytic


def test_single_term_matches_formula():
    phi = np.array([0.0, 60.0, 120.0, 180.0])
    out = analytic.dihedral_energy([0.0, 2.0, 3], phi)
    expected = 2.0 * (1.0 + np.cos(3 * np.deg2rad(phi) - 0.0))
    assert out == pytest.approx(expected)


def test_phase_180_shifts_curve():
    phi = np.array([0.0, 90.0])
    out = analytic.dihedral_energy([180.0, 1.5, 2], phi)
    expected = 1.5 * (1.0 + np.cos(2 * np.deg2rad(phi) - np.pi))
    assert out == pytest.approx(expected)


def test_energy_is_periodic_over_360():
    phi = np.linspace(-180.0, 180.0, 37)
    a = analytic.dihedral_energy([0.0, 1.0, 3], phi)
    b = analytic.dihedral_energy([0.0, 1.0, 3], phi + 360.0)
    assert a == pytest.approx(b)


def test_energy_is_non_negative():
    phi = np.linspace(-180.0, 180.0, 73)
    assert np.all(analytic.dihedral_energy([0.0, 1.0, 3], phi) >= -1e-12)


def test_threefold_minima_at_trans_and_gauche():
    phi = np.linspace(-180.0, 180.0, 361)
    e = analytic.dihedral_energy([0.0, 1.0, 3], phi)
    # Local minima: lower than both neighbours (endpoints treated as boundary).
    falls_from_left = np.r_[True, e[1:] < e[:-1]]
    rises_to_right = np.r_[e[:-1] < e[1:], True]
    minima = phi[falls_from_left & rises_to_right]
    assert np.any(np.isclose(np.abs(minima), 180.0, atol=2.0))


def test_torsion_set_energy_over_frames(butane_itp):
    from ffrefine.io.structure import read_pdb
    from ffrefine.mm.torsionset import collect_torsion_set

    t = read_topology(butane_itp)
    tset = collect_torsion_set(t, (1, 5, 8, 11))
    coords = np.zeros((2, 14, 3))
    coords[0] = np.arange(42).reshape(14, 3) * 0.1
    coords[1] = coords[0] + 0.01
    e = analytic.torsion_set_energy(tset, coords)
    assert e.shape == (2,)
    assert np.all(np.isfinite(e))


def test_torsion_set_energy_sums_members():
    d1 = Dihedral(1, 2, 3, 4, 9, [0.0, 1.0, 3])
    d2 = Dihedral(1, 2, 3, 4, 9, [180.0, 2.0, 2])
    coords = np.random.default_rng(1).normal(size=(3, 4, 3))
    total = analytic.torsion_set_energy([d1, d2], coords)
    apart = analytic.torsion_set_energy([d1], coords) + analytic.torsion_set_energy([d2], coords)
    assert total == pytest.approx(apart)


def test_torsion_set_energy_ignores_entries_without_parameters():
    """A line with no explicit parameters contributes nothing analytically."""
    d = Dihedral(1, 2, 3, 4, 9, [])
    coords = np.random.default_rng(2).normal(size=(2, 4, 3))
    assert analytic.torsion_set_energy([d], coords) == pytest.approx([0.0, 0.0])


def test_series_energy_matches_sum_of_terms():
    phi = np.linspace(-180.0, 180.0, 25)
    k = np.array([1.0, 0.5])
    delta = np.array([0.0, 180.0])
    mult = np.array([3, 2])
    out = analytic.series_energy(k, delta, mult, phi)
    expected = analytic.dihedral_energy([0.0, 1.0, 3], phi) + analytic.dihedral_energy(
        [180.0, 0.5, 2], phi
    )
    assert out == pytest.approx(expected)
