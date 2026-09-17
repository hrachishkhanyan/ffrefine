import numpy as np
import pytest

from ffrefine.charges.equivalence import average_within_groups, equivalent_hydrogen_groups
from ffrefine.io.topology import read_topology


def test_butane_gives_two_methyl_and_two_methylene_groups(butane_itp):
    groups = equivalent_hydrogen_groups(read_topology(butane_itp))

    assert [len(g) for g in groups] == [3, 2, 2, 3]
    assert groups[0] == [2, 3, 4]  # the H on C1


def test_a_hydrogen_with_a_different_atom_type_is_not_grouped(butane_itp):
    """Same parent is not enough: the atom type is part of the definition."""
    top = read_topology(butane_itp)
    top.atoms[1].atype = "HGA1"  # H11 no longer matches its siblings

    groups = equivalent_hydrogen_groups(top)

    assert [3, 4] in groups
    assert not any(2 in g for g in groups)


def test_a_lone_hydrogen_forms_no_group(butane_itp):
    """A group of one constrains nothing, so it must not be reported as one."""
    top = read_topology(butane_itp)
    top.atoms[2].atype = "HGA1"
    top.atoms[3].atype = "HGA2"

    assert not any(2 in g or 3 in g or 4 in g for g in equivalent_hydrogen_groups(top))


def test_the_element_comes_from_the_mass_not_the_name(butane_itp):
    """An atom named NA with nitrogen's mass is nitrogen; guess_element disagrees."""
    top = read_topology(butane_itp)
    top.atoms[0].name = "NA"
    top.atoms[0].mass = 14.007

    groups = equivalent_hydrogen_groups(top)

    assert groups[0] == [2, 3, 4]  # still a heavy parent carrying three H


def test_averaging_makes_a_group_equal_and_preserves_its_sum():
    """Preserving the sum is why averaging cannot disturb the molecular total."""
    charges = np.array([-0.3, 0.09, 0.11, 0.10, 0.0])
    before = charges.sum()

    out = average_within_groups(charges, [[2, 3, 4]])

    assert out[1] == pytest.approx(0.10)
    assert out[2] == pytest.approx(0.10)
    assert out[3] == pytest.approx(0.10)
    assert out.sum() == pytest.approx(before)
    assert out[0] == pytest.approx(-0.3)  # untouched


def test_averaging_leaves_everything_alone_when_there_are_no_groups():
    charges = np.array([0.1, -0.1])
    assert average_within_groups(charges, []) == pytest.approx(charges)
