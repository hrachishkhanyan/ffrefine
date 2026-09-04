import pytest
from ffrefine.errors import ValidationError
from ffrefine.io.topology import read_topology
from ffrefine.mm.torsionset import collect_torsion_set, find_target_entries, validate_quad


def test_collects_all_lines_sharing_central_bond(butane_itp):
    """Spec section 9.1: every torsion *line* about bond 5-8, not just the target.

    The butane fixture writes the target (1,5,8,11) as two funct-9 lines
    (multiplicity 2 and 3) plus eight other quadruplets with one line each,
    for 10 topology lines spanning 9 distinct quadruplets.
    """
    t = read_topology(butane_itp)
    tset = collect_torsion_set(t, (1, 5, 8, 11))
    assert len(tset) == 10
    for d in tset:
        assert d.central_bond == (5, 8)
    distinct_quads = {min((d.ai, d.aj, d.ak, d.al), (d.al, d.ak, d.aj, d.ai)) for d in tset}
    assert len(distinct_quads) == 9


def test_reversed_quad_gives_same_set(butane_itp):
    t = read_topology(butane_itp)
    a = collect_torsion_set(t, (1, 5, 8, 11))
    b = collect_torsion_set(t, (11, 8, 5, 1))
    assert {(d.ai, d.aj, d.ak, d.al) for d in a} == {(d.ai, d.aj, d.ak, d.al) for d in b}


def test_target_entries_are_the_two_funct9_lines(butane_itp):
    t = read_topology(butane_itp)
    entries = find_target_entries(t, (1, 5, 8, 11))
    assert len(entries) == 2
    assert {int(e.params[2]) for e in entries} == {2, 3}


def test_target_entries_match_reversed_direction(butane_itp):
    t = read_topology(butane_itp)
    assert len(find_target_entries(t, (11, 8, 5, 1))) == 2


def test_impropers_never_included(butane_itp):
    t = read_topology(butane_itp)
    tset = collect_torsion_set(t, (1, 5, 8, 11))
    assert all(d.funct not in (2, 4) for d in tset)


def test_methyl_torsions_excluded(butane_itp):
    """Torsions about bonds 1-5 and 8-11 must not appear in T for bond 5-8."""
    t = read_topology(butane_itp)
    tset = collect_torsion_set(t, (1, 5, 8, 11))
    assert all(d.central_bond != (1, 5) for d in tset)
    assert all(d.central_bond != (8, 11) for d in tset)


def test_validate_rejects_out_of_range_index(butane_itp):
    t = read_topology(butane_itp)
    with pytest.raises(ValidationError, match="out of range"):
        validate_quad(t, (1, 5, 8, 99))


def test_validate_rejects_non_bonded_central_pair(butane_itp):
    t = read_topology(butane_itp)
    with pytest.raises(ValidationError, match="not bonded"):
        validate_quad(t, (2, 1, 8, 11))


def test_validate_accepts_good_quad(butane_itp):
    t = read_topology(butane_itp)
    validate_quad(t, (1, 5, 8, 11))
