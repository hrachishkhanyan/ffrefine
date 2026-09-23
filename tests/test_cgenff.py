"""The CGenFF .str parser, on the two committed CGenFF outputs (design section 4)."""

from __future__ import annotations

from pathlib import Path

import pytest
from ffrefine.errors import ParseError
from ffrefine.io.cgenff import StrAtom, read_str

FIXTURES = Path(__file__).parent.parent / "fixtures"
PFOA_STR = FIXTURES / "pfoa_gromacs" / "pfoa.str"
CAP_STR = FIXTURES / "capecitabine_gromacs" / "capecitabine.str"


def _penalty(strfile, kind, types):
    [match] = [p for p in strfile.of_kind(kind) if p.types == tuple(types)]
    return match.penalty


def test_pfoa_atoms_in_file_order():
    s = read_str(PFOA_STR)
    assert len(s.atoms) == 26
    assert s.atoms[0] == StrAtom(1, "C1", "CG2O2", 0.789, 25.023)
    assert s.atoms[1] == StrAtom(2, "O1", "OG2D1", -0.537, 17.721)
    assert s.atoms[-1] == StrAtom(26, "H", "HGP1", 0.429, 6.830)


def test_pfoa_parameters_and_header():
    s = read_str(PFOA_STR)
    counts = {kind: len(s.of_kind(kind)) for kind in ("bond", "angle", "dihedral", "improper")}
    assert counts == {"bond": 2, "angle": 8, "dihedral": 13, "improper": 1}
    assert (s.program_version, s.parameter_version) == ("4.0", "5.0")
    assert (s.param_penalty, s.charge_penalty) == (69.0, 41.068)
    assert s.source == PFOA_STR


def test_a_dihedral_over_several_multiplicity_lines_is_one_entry():
    s = read_str(PFOA_STR)
    assert _penalty(s, "dihedral", ["OG311", "CG2O2", "CG312", "CG312"]) == 67.0
    assert _penalty(s, "dihedral", ["CG2O2", "CG312", "CG312", "FGA2"]) == 67.2
    assert _penalty(s, "dihedral", ["OG2D1", "CG2O2", "CG312", "CG312"]) == 22.5
    assert _penalty(s, "angle", ["CG312", "CG2O2", "OG2D1"]) == 10.5
    assert _penalty(s, "bond", ["CG2O2", "CG312"]) == 10.0
    assert _penalty(s, "improper", ["CG2O2", "CG312", "OG2D1", "OG311"]) == 5.5


def test_capecitabine_skips_ring_lines_and_reads_urey_bradley_angles():
    s = read_str(CAP_STR)
    assert len(s.atoms) == 47
    assert s.atoms[9] == StrAtom(10, "C7", "CG2R64", 0.389, 5.422)
    assert _penalty(s, "angle", ["CG2R66", "CG2R64", "NG2S1"]) == 0.5
    assert _penalty(s, "dihedral", ["OG302", "CG2O6", "NG2S1", "CG2R64"]) == 50.0
    assert (s.param_penalty, s.charge_penalty) == (50.0, 5.422)


def test_crlf_line_endings_parse_the_same(tmp_path):
    """Review focus 4: a .str saved on Windows."""
    crlf = tmp_path / "pfoa.str"
    crlf.write_bytes(PFOA_STR.read_bytes().replace(b"\n", b"\r\n"))
    a, b = read_str(PFOA_STR), read_str(crlf)
    assert (a.atoms, a.parameters) == (b.atoms, b.parameters)


def test_a_missing_file_is_a_parse_error(tmp_path):
    """Review focus 5."""
    with pytest.raises(ParseError, match="could not be read"):
        read_str(tmp_path / "nope.str")


def test_a_file_with_no_atom_is_refused(tmp_path):
    bad = tmp_path / "empty.str"
    bad.write_text("* nothing here\nread rtf card append\nEND\n")
    with pytest.raises(ParseError, match="no ATOM line"):
        read_str(bad)


def test_a_malformed_atom_line_names_its_line(tmp_path):
    bad = tmp_path / "bad.str"
    bad.write_text(
        PFOA_STR.read_text().replace("ATOM C1     CG2O2   0.789", "ATOM C1     CG2O2", 1)
    )
    with pytest.raises(ParseError, match=r"bad\.str:20: malformed ATOM"):
        read_str(bad)


def test_a_non_numeric_penalty_is_refused(tmp_path):
    bad = tmp_path / "bad.str"
    bad.write_text(PFOA_STR.read_text().replace("penalty= 67", "penalty= sixty-seven", 1))
    with pytest.raises(ParseError, match="penalty 'sixty-seven' is not a number"):
        read_str(bad)
