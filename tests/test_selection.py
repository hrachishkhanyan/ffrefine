"""Selection from CGenFF penalties (design section 4), on the committed fixtures."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from ffrefine.errors import ValidationError
from ffrefine.io.cgenff import StrAtom, StrFile, StrParameter, read_str
from ffrefine.io.topology import read_topology
from ffrefine.selection import Thresholds, select, selection_to_json

FIXTURES = Path(__file__).parent.parent / "fixtures"
PFOA = FIXTURES / "pfoa_gromacs"
CAP = FIXTURES / "capecitabine_gromacs"


def _pfoa():
    return read_topology(PFOA / "pfoa_gmx.top"), read_str(PFOA / "pfoa.str")


def _cap():
    return read_topology(CAP / "capecitabine_gmx.top"), read_str(CAP / "capecitabine.str")


def _str_for(topology, penalties=None, parameters=()):
    """A StrFile describing ``topology`` exactly, with the given charge penalties."""
    penalties = penalties or {}
    atoms = [
        StrAtom(a.nr, a.name, a.atype, a.charge, float(penalties.get(a.nr, 0.0)))
        for a in topology.atoms
    ]
    return StrFile(atoms, list(parameters), None, None, None, None, Path("test.str"))


def test_pfoa_flags_23_atoms_and_two_bonds_in_penalty_order():
    sel = select(*_pfoa(), Thresholds())
    assert len(sel.flagged_atoms) == 23
    assert [a.nr for a in sel.atoms if not a.flagged] == [13, 14, 26]
    assert [(b.bond, b.penalty, b.quad) for b in sel.bonds] == [
        ((3, 4), 69.0, (1, 3, 4, 5)),
        ((1, 3), 67.0, (25, 1, 3, 4)),
    ]
    assert sel.quads == [(1, 3, 4, 5), (25, 1, 3, 4)]
    assert sel.ring_bonds == [] and sel.warnings == []


def test_capecitabine_flags_its_carbamate_at_exactly_the_threshold():
    sel = select(*_cap(), Thresholds())
    assert sel.flagged_atoms == []
    assert [(b.bond, b.penalty, b.quad) for b in sel.bonds] == [((7, 9), 50.0, (6, 7, 9, 10))]
    assert select(*_cap(), Thresholds(dihedral=51.0)).bonds == []
    assert sel.highest_dihedral_penalty == 50.0


def test_ring_bonds_are_never_selected():
    sel = select(*_cap(), Thresholds(dihedral=0.5))
    rings = {b.bond for b in sel.ring_bonds}
    assert rings == {(10, 16), (10, 11), (14, 15), (15, 16)}
    assert not rings & {b.bond for b in sel.bonds}


def test_a_different_atom_count_is_refused():
    t, _ = _cap()
    with pytest.raises(ValidationError, match="describes 26 atoms but the topology"):
        select(t, read_str(PFOA / "pfoa.str"), Thresholds())


def test_a_different_type_is_refused(tmp_path):
    bad = tmp_path / "pfoa.str"
    bad.write_text(
        (PFOA / "pfoa.str").read_text().replace("ATOM C1     CG2O2", "ATOM C1     CG331", 1)
    )
    with pytest.raises(ValidationError, match="atom 1 is CG2O2 in the topology, CG331 in the .str"):
        select(read_topology(PFOA / "pfoa_gmx.top"), read_str(bad), Thresholds())


def test_a_charge_difference_is_a_warning(tmp_path):
    edited = tmp_path / "pfoa.str"
    edited.write_text((PFOA / "pfoa.str").read_text().replace("CG2O2   0.789", "CG2O2   0.800", 1))
    sel = select(read_topology(PFOA / "pfoa_gmx.top"), read_str(edited), Thresholds())
    assert len(sel.warnings) == 1 and "differ" in sel.warnings[0] and "1 (+0.789" in sel.warnings[0]


def test_a_reordering_within_one_type_is_caught_by_the_charges(tmp_path):
    """Review focus 3: swapping two FGA2 atoms passes the type check."""
    lines = (PFOA / "pfoa.str").read_text().splitlines(keepends=True)
    i4 = next(i for i, line in enumerate(lines) if line.startswith("ATOM F4 "))
    i6 = next(i for i, line in enumerate(lines) if line.startswith("ATOM F6 "))
    lines[i4], lines[i6] = lines[i6], lines[i4]  # .str atoms 13 and 15, both FGA2
    swapped = tmp_path / "pfoa.str"
    swapped.write_text("".join(lines))
    sel = select(read_topology(PFOA / "pfoa_gmx.top"), read_str(swapped), Thresholds())
    assert sel.warnings and "13 (" in sel.warnings[0] and "15 (" in sel.warnings[0]


def test_flagged_hydrogens_extend_to_their_equivalence_group(butane_itp):
    t = read_topology(butane_itp)
    first_h = next(a.nr for a in t.atoms if a.atype == "HGA3")
    sel = select(t, _str_for(t, {first_h: 20.0}), Thresholds())
    methyl = [a for a in sel.atoms if a.flagged]
    assert len(methyl) == 3 and all(a.atype == "HGA3" for a in methyl)
    assert [a.by_equivalence for a in methyl] == [False, True, True]


def test_types_match_in_either_direction():
    t, s = _pfoa()
    reversed_types = StrParameter("dihedral", ("CG312", "CG312", "CG2O2", "OG311"), 67.0)
    sel = select(t, replace(s, parameters=[reversed_types]), Thresholds())
    assert [b.quad for b in sel.bonds] == [(25, 1, 3, 4)]


def test_an_unmatched_flagged_type_warns():
    t, s = _pfoa()
    alien = StrParameter("dihedral", ("NG2S1", "CG2O6", "NG2S1", "CG2R64"), 80.0)
    sel = select(t, replace(s, parameters=[alien]), Thresholds())
    assert sel.unmatched_types == [alien.types]
    assert any("another molecule" in w for w in sel.warnings)


def test_heavy_outer_atoms_win_a_penalty_tie(butane_itp):
    t = read_topology(butane_itp)
    types_of = {a.nr: a.atype for a in t.atoms}
    heavy = next(d for d in t.propers if (d.ai, d.aj, d.ak, d.al) == (1, 5, 8, 11))
    with_h = next(d for d in t.propers if (d.aj, d.ak) == (5, 8) and types_of[d.ai].startswith("H"))
    params = [
        StrParameter("dihedral", tuple(types_of[n] for n in (d.ai, d.aj, d.ak, d.al)), 60.0)
        for d in (with_h, heavy)
    ]
    sel = select(t, _str_for(t, parameters=params), Thresholds())
    assert [b.quad for b in sel.bonds] == [(1, 5, 8, 11)]


def test_selection_json_records_every_decision():
    t, s = _pfoa()
    data = selection_to_json(select(t, s, Thresholds()), s)
    json.dumps(data)  # serialisable
    assert data["thresholds"] == {"charge": 10.0, "dihedral": 50.0}
    assert data["cgenff_program_version"] == "4.0"
    assert data["resi_param_penalty"] == 69.0
    assert [b["quad_1_based"] for b in data["bonds"]] == [[1, 3, 4, 5], [25, 1, 3, 4]]
    atom1 = data["atoms"][0]
    assert atom1["top_name"] == "C1" and atom1["str_name"] == "C1" and atom1["flagged"] is True
    atom2 = data["atoms"][1]
    assert (atom2["top_name"], atom2["str_name"]) == ("O2", "O1")
    assert len(data["dihedral_types"]) == 13
    assert len(data["other_parameters"]) == 11
