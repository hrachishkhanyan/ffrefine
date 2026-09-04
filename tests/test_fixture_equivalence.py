import numpy as np
import pytest
from ffrefine.io.structure import read_gro, read_pdb
from ffrefine.io.topology import read_topology
from ffrefine.io.validate import validate_structure_topology


def test_butane_topology_parses(butane_itp):
    t = read_topology(butane_itp)
    assert t.natoms == 14
    assert t.name == "BUT"
    assert t.defaults.fudge_qq == pytest.approx(1.0)
    assert len(t.impropers) == 0


def test_butane_charges_sum_to_zero(butane_itp):
    t = read_topology(butane_itp)
    assert sum(a.charge for a in t.atoms) == pytest.approx(0.0, abs=1e-9)


def test_pdb_and_gro_give_identical_coordinates(butane_pdb, butane_gro):
    """Spec section 7.3: the same molecule in both formats must agree."""
    a = read_pdb(butane_pdb)
    b = read_gro(butane_gro)
    assert a.names == b.names
    assert a.resnames == b.resnames
    assert np.allclose(a.coords, b.coords, atol=1e-3)


def test_both_formats_validate_against_topology(butane_pdb, butane_gro, butane_itp):
    t = read_topology(butane_itp)
    assert validate_structure_topology(read_pdb(butane_pdb), t) == []
    assert validate_structure_topology(read_gro(butane_gro), t) == []


def test_target_torsion_is_all_trans(butane_pdb):
    from ffrefine.geometry import dihedral_deg

    s = read_pdb(butane_pdb)
    # C1-C2-C3-C4 is 1-based atoms 1, 5, 8, 11 -> 0-based 0, 4, 7, 10
    phi = dihedral_deg(s.coords[0], s.coords[4], s.coords[7], s.coords[10])
    assert abs(phi) == pytest.approx(180.0, abs=1.0)
