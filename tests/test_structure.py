import numpy as np
import pytest
from ffrefine.errors import ParseError
from ffrefine.io import structure as st

PDB_TEXT = """\
TITLE     n-butane
ATOM      1  C1  BUT     1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  C2  BUT     1       1.520   0.000   0.000  1.00  0.00           C
ATOM      3  H11 BUT     1      -0.360   1.020   0.000  1.00  0.00           H
END
"""

GRO_TEXT = """\
n-butane
    3
    1BUT     C1    1   0.000   0.000   0.000
    1BUT     C2    2   0.152   0.000   0.000
    1BUT    H11    3  -0.036   0.102   0.000
   2.00000   2.00000   2.00000
"""


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_pdb_parses_names_and_converts_to_nm(tmp_path):
    s = st.read_pdb(_write(tmp_path, "m.pdb", PDB_TEXT))
    assert s.natoms == 3
    assert s.names == ["C1", "C2", "H11"]
    assert s.resnames == ["BUT", "BUT", "BUT"]
    # 1.520 Angstrom -> 0.152 nm
    assert s.coords[1, 0] == pytest.approx(0.152)


def test_pdb_elements(tmp_path):
    s = st.read_pdb(_write(tmp_path, "m.pdb", PDB_TEXT))
    assert s.elements == ["C", "C", "H"]


def test_gro_parses_in_nm(tmp_path):
    s = st.read_gro(_write(tmp_path, "m.gro", GRO_TEXT))
    assert s.natoms == 3
    assert s.names == ["C1", "C2", "H11"]
    assert s.coords[1, 0] == pytest.approx(0.152)
    assert s.box == pytest.approx([2.0, 2.0, 2.0])


def test_gro_negative_coordinate_columns(tmp_path):
    """GRO is fixed-width; a negative number must not eat the previous column."""
    s = st.read_gro(_write(tmp_path, "m.gro", GRO_TEXT))
    assert s.coords[2, 0] == pytest.approx(-0.036)
    assert s.coords[2, 1] == pytest.approx(0.102)


def test_pdb_and_gro_agree(tmp_path):
    a = st.read_pdb(_write(tmp_path, "m.pdb", PDB_TEXT))
    b = st.read_gro(_write(tmp_path, "m.gro", GRO_TEXT))
    assert a.names == b.names
    assert np.allclose(a.coords, b.coords, atol=1e-6)


def test_read_structure_dispatches(tmp_path):
    assert st.read_structure(_write(tmp_path, "m.pdb", PDB_TEXT)).natoms == 3
    assert st.read_structure(_write(tmp_path, "m.gro", GRO_TEXT)).natoms == 3


def test_read_structure_rejects_unknown_suffix(tmp_path):
    with pytest.raises(ParseError, match="unsupported"):
        st.read_structure(_write(tmp_path, "m.mol2", "junk"))


def test_gro_atom_count_mismatch_raises(tmp_path):
    bad = GRO_TEXT.replace("    3\n", "    9\n")
    with pytest.raises(ParseError, match="expected 9"):
        st.read_gro(_write(tmp_path, "bad.gro", bad))


def test_multimodel_pdb_round_trip(tmp_path):
    s = st.read_pdb(_write(tmp_path, "m.pdb", PDB_TEXT))
    frames = np.stack([s.coords, s.coords + 0.01, s.coords + 0.02])
    out = tmp_path / "traj.pdb"
    st.write_multimodel_pdb(out, s, frames)
    text = out.read_text()
    assert text.count("MODEL") == 3
    assert text.count("ENDMDL") == 3
    # names must survive so that mdrun -rerun matches the topology
    assert "C2" in text


def test_multimodel_pdb_writer_reader_agree(tmp_path):
    """The writer and read_pdb must agree on column positions.

    This is the regression test for the off-by-one column bug in the
    original brief: writing a *multi*-model PDB (three distinct frames)
    and reading it back with ``read_pdb`` must yield model 1's names,
    resnames, resids and coordinates exactly (coordinates to within 1e-3
    nm, the precision the PDB format carries in Angstrom) -- not model 2
    or 3, proving both the column layout and the ENDMDL stop-after-first-
    model behaviour are correct.
    """
    s = st.read_pdb(_write(tmp_path, "m.pdb", PDB_TEXT))
    frames = np.stack([s.coords, s.coords + 0.5, s.coords + 1.0])
    out = tmp_path / "traj.pdb"
    st.write_multimodel_pdb(out, s, frames)
    r = st.read_pdb(out)
    assert r.names == s.names
    assert r.resnames == s.resnames
    assert r.resids == s.resids
    assert np.allclose(r.coords, s.coords, atol=1e-3)


def test_guess_element_from_name():
    assert st.guess_element("C1") == "C"
    assert st.guess_element("H11") == "H"
    assert st.guess_element("CL2") == "Cl"
