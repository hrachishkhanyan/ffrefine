import numpy as np
import pytest

from ffrefine.charges.grid import MK_SHELL_SCALES, mk_grid, vdw_radius_nm
from ffrefine.errors import ValidationError


def test_an_element_with_no_radius_is_refused():
    """Defaulting a radius biases every charge and leaves no trace in the output."""
    with pytest.raises(ValidationError, match="Xe"):
        vdw_radius_nm("Xe")


def test_radii_are_in_nanometres():
    assert vdw_radius_nm("C") == pytest.approx(0.15)
    assert vdw_radius_nm("H") == pytest.approx(0.12)


def test_one_atom_puts_every_point_on_one_of_the_four_shells():
    coords = np.zeros((1, 3))
    points = mk_grid(coords, ["C"])

    radii = np.linalg.norm(points, axis=1)
    expected = sorted(s * 0.15 for s in MK_SHELL_SCALES)
    for r in radii:
        assert min(abs(r - e) for e in expected) < 1e-9


def test_the_grid_is_deterministic():
    """Random placement would make charges differ between identical runs."""
    coords = np.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]])

    first = mk_grid(coords, ["C", "O"])
    second = mk_grid(coords, ["C", "O"])

    assert first.shape == second.shape
    assert np.array_equal(first, second)


def test_point_count_grows_with_density():
    coords = np.zeros((1, 3))

    sparse = mk_grid(coords, ["C"], density_per_angstrom2=1.0)
    dense = mk_grid(coords, ["C"], density_per_angstrom2=4.0)

    assert len(dense) > 3.5 * len(sparse)


def test_point_count_grows_with_the_square_of_the_radius():
    """Points per unit area is the spec's unit, so count scales with area."""
    small = mk_grid(np.zeros((1, 3)), ["H"])  # 1.20 A
    large = mk_grid(np.zeros((1, 3)), ["S"])  # 1.75 A

    ratio = len(large) / len(small)
    assert ratio == pytest.approx((1.75 / 1.20) ** 2, rel=0.1)


def test_no_point_survives_inside_another_atoms_shell():
    """The overlap rule is what stops the inner shells burying points in atoms."""
    coords = np.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]])
    elements = ["C", "C"]

    points = mk_grid(coords, elements)

    inner = MK_SHELL_SCALES[0] * 0.15
    for atom in coords:
        assert np.linalg.norm(points - atom, axis=1).min() >= inner - 1e-9


def test_two_atoms_give_fewer_points_than_two_isolated_ones():
    """Because the overlap removal actually removed something."""
    lone = len(mk_grid(np.zeros((1, 3)), ["C"]))
    pair = len(mk_grid(np.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]]), ["C", "C"]))

    assert pair < 2 * lone


def test_a_mismatched_element_list_is_refused():
    with pytest.raises(ValidationError, match="3"):
        mk_grid(np.zeros((3, 3)), ["C", "H"])
