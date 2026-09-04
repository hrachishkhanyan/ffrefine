import numpy as np
import pytest
from ffrefine.errors import ValidationError
from ffrefine.io.structure import Structure
from ffrefine.io.topology import Atom, Topology
from ffrefine.io.validate import validate_structure_topology


def _struct(names, elements):
    n = len(names)
    return Structure(
        names=list(names),
        resnames=["BUT"] * n,
        resids=[1] * n,
        elements=list(elements),
        coords=np.zeros((n, 3)),
    )


def _topo(names, masses):
    atoms = [
        Atom(nr=i + 1, atype="CG331", resnr=1, resname="BUT", name=nm, cgnr=1, charge=0.0, mass=m)
        for i, (nm, m) in enumerate(zip(names, masses))
    ]
    return Topology(
        name="BUT",
        nrexcl=3,
        atoms=atoms,
        bonds=[],
        pairs=[],
        angles=[],
        propers=[],
        impropers=[],
        defaults=None,
        raw_lines=[],
        source="x.itp",
    )


def test_matching_inputs_produce_no_warnings():
    s = _struct(["C1", "C2"], ["C", "C"])
    t = _topo(["C1", "C2"], [12.011, 12.011])
    assert validate_structure_topology(s, t) == []


def test_atom_count_mismatch_is_fatal():
    s = _struct(["C1", "C2"], ["C", "C"])
    t = _topo(["C1"], [12.011])
    with pytest.raises(ValidationError, match="atom count"):
        validate_structure_topology(s, t)


def test_element_ordering_mismatch_is_fatal():
    """A carbon in the structure against a hydrogen mass in the topology."""
    s = _struct(["C1", "C2"], ["C", "C"])
    t = _topo(["C1", "C2"], [12.011, 1.008])
    with pytest.raises(ValidationError, match="element"):
        validate_structure_topology(s, t)


def test_name_mismatch_warns_but_does_not_abort():
    s = _struct(["C1", "CX"], ["C", "C"])
    t = _topo(["C1", "C2"], [12.011, 12.011])
    warnings = validate_structure_topology(s, t, strict_names=False)
    assert any("name" in w for w in warnings)


def test_name_mismatch_is_fatal_when_strict():
    s = _struct(["C1", "CX"], ["C", "C"])
    t = _topo(["C1", "C2"], [12.011, 12.011])
    with pytest.raises(ValidationError, match="name"):
        validate_structure_topology(s, t, strict_names=True)
