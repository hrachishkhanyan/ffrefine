import numpy as np
import pytest
from ffrefine import units


def test_hartree_constant_exact():
    assert units.HARTREE_TO_KJ == 2625.499639


def test_hartree_round_trip():
    x = 0.0371
    assert units.kj_to_hartree(units.hartree_to_kj(x)) == pytest.approx(x, abs=1e-15)


def test_kcal_round_trip():
    x = 12.5
    assert units.kcal_to_kj(units.kj_to_kcal(x)) == pytest.approx(x, abs=1e-12)


def test_kcal_constant_is_thermochemical():
    assert units.KCAL_TO_KJ == 4.184
    assert units.kcal_to_kj(1.0) == pytest.approx(4.184)


def test_degree_round_trip_array():
    d = np.array([-180.0, -90.0, 0.0, 179.9])
    assert np.allclose(units.rad_to_deg(units.deg_to_rad(d)), d)


def test_conversions_accept_arrays():
    a = np.array([1.0, 2.0])
    assert isinstance(units.hartree_to_kj(a), np.ndarray)
    assert units.hartree_to_kj(a)[1] == pytest.approx(2 * 2625.499639)
