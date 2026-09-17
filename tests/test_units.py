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


def test_angstrom_to_nm_roundtrip():
    from ffrefine.units import ANGSTROM_TO_NM, angstrom_to_nm, nm_to_angstrom

    assert ANGSTROM_TO_NM == 0.1
    assert angstrom_to_nm(10.0) == pytest.approx(1.0)
    assert nm_to_angstrom(1.0) == pytest.approx(10.0)
    assert nm_to_angstrom(angstrom_to_nm(3.7)) == pytest.approx(3.7)


def test_structure_still_exports_angstrom_to_nm():
    """Back-compat: tests/conftest.py and io/structure.py itself import it from there."""
    from ffrefine.io.structure import ANGSTROM_TO_NM as from_structure
    from ffrefine.units import ANGSTROM_TO_NM as from_units

    assert from_structure is from_units


def test_bohr_to_nm_uses_the_codata_value():
    from ffrefine.units import bohr_to_nm

    assert bohr_to_nm(1.0) == pytest.approx(0.0529177210903)


def test_bohr_round_trips_through_nm():
    from ffrefine.units import bohr_to_nm, nm_to_bohr

    assert nm_to_bohr(bohr_to_nm(2.5)) == pytest.approx(2.5, rel=1e-12)


def test_bohr_conversion_is_consistent_with_the_angstrom_one():
    """One bohr is 0.529177 Angstrom, and Angstrom already has a conversion.

    Two independent constants for the same physical relation is how a unit bug
    hides: each looks right on its own.
    """
    from ffrefine.units import angstrom_to_nm, bohr_to_nm

    assert bohr_to_nm(1.0) == pytest.approx(angstrom_to_nm(0.529177210903), rel=1e-12)


def test_bohr_conversions_work_on_arrays():
    from ffrefine.units import nm_to_bohr

    out = nm_to_bohr(np.array([0.0, 0.0529177210903]))
    assert out == pytest.approx([0.0, 1.0])
