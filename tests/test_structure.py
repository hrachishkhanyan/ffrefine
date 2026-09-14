import numpy as np
import pytest
from ffrefine.errors import ParseError, ValidationError
from ffrefine.io import structure as st
from ffrefine.io.structure import read_pdb, read_structure, read_xyz
from ffrefine.io.topology import read_topology
from ffrefine.io.validate import adopt_topology_naming, validate_structure_topology

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


# --------------------------------------------------------------------------
# XYZ (final review, item 5 -- deferred ledger item 99: this was shipping
# as an advertised interface with no tests at all, hiding a live defect)
# --------------------------------------------------------------------------


def test_read_xyz_parses_elements_and_converts_to_nm(tmp_path):
    path = tmp_path / "two.xyz"
    path.write_text(
        "2\nwater fragment\nO  0.000000  0.000000  0.000000\nH  0.960000  0.000000  0.000000\n"
    )
    s = read_xyz(path)
    assert s.natoms == 2
    assert s.elements == ["O", "H"]
    assert s.title == "water fragment"
    # 0.96 Angstrom -> 0.096 nm
    assert s.coords[1][0] == pytest.approx(0.096)
    assert s.coords.shape == (2, 3)


def test_read_xyz_flags_its_names_as_synthetic(tmp_path):
    path = tmp_path / "one.xyz"
    path.write_text("1\n\nCl 1.0 2.0 3.0\n")
    s = read_xyz(path)
    assert s.names_are_synthetic is True
    assert s.names == ["Cl1"]
    assert s.elements == ["Cl"]


def test_read_xyz_rejects_a_short_body(tmp_path):
    path = tmp_path / "bad.xyz"
    path.write_text("3\ntitle\nC 0 0 0\n")
    with pytest.raises(ParseError):
        read_xyz(path)


def test_read_xyz_rejects_a_malformed_line(tmp_path):
    path = tmp_path / "bad.xyz"
    path.write_text("1\ntitle\nC 0 0\n")
    with pytest.raises(ParseError):
        read_xyz(path)


def test_read_xyz_rejects_a_non_numeric_count(tmp_path):
    path = tmp_path / "bad.xyz"
    path.write_text("banana\ntitle\nC 0 0 0\n")
    with pytest.raises(ParseError):
        read_xyz(path)


def test_read_structure_dispatches_on_xyz(butane_xyz):
    s = read_structure(butane_xyz)
    assert s.natoms == 14
    assert s.names_are_synthetic is True


def test_read_structure_rejects_an_unknown_suffix(tmp_path):
    path = tmp_path / "mol.mol2"
    path.write_text("whatever\n")
    with pytest.raises(ParseError, match="unsupported structure format"):
        read_structure(path)


def test_xyz_agrees_with_the_pdb_it_came_from(butane_xyz, butane_pdb):
    x = read_xyz(butane_xyz)
    p = read_pdb(butane_pdb)
    assert x.elements == p.elements
    assert np.allclose(x.coords, p.coords, atol=1e-6)


def test_xyz_against_a_topology_is_not_fatal_under_strict_names(butane_xyz, butane_itp):
    """Item 5: 'C1'/'H2' can never match a topology, so the default strict-name
    check made every .xyz input fatal:

        error: atom name mismatch (13 atoms): atom 2: structure 'H2' vs
        topology 'H11'; ...
    """
    s = read_xyz(butane_xyz)
    t = read_topology(butane_itp)
    warnings = validate_structure_topology(s, t, strict_names=True)
    assert any("carries no atom names" in w for w in warnings)
    assert not any("atom name mismatch" in w for w in warnings)


def test_xyz_still_catches_a_wrong_element_order(tmp_path, butane_itp):
    """Skipping the name check must not weaken the check that matters."""
    path = tmp_path / "swapped.xyz"
    elements = ["H", "C"] + ["H"] * 12  # atom 1 is carbon in the topology
    lines = ["14", "swapped"] + [f"{e} 0.0 0.0 {i * 0.1}" for i, e in enumerate(elements)]
    path.write_text("\n".join(lines) + "\n")
    s = read_xyz(path)
    t = read_topology(butane_itp)
    with pytest.raises(ValidationError, match="element mismatch"):
        validate_structure_topology(s, t, strict_names=False)


def test_xyz_still_catches_a_wrong_atom_count(tmp_path, butane_itp):
    path = tmp_path / "short.xyz"
    path.write_text("2\nshort\nC 0 0 0\nC 0 0 1\n")
    s = read_xyz(path)
    t = read_topology(butane_itp)
    with pytest.raises(ValidationError, match="atom count mismatch"):
        validate_structure_topology(s, t)


def test_adopt_topology_naming_replaces_synthetic_names(butane_xyz, butane_itp):
    s = read_xyz(butane_xyz)
    t = read_topology(butane_itp)
    adopted = adopt_topology_naming(s, t)
    assert adopted.names == [a.name for a in t.atoms]
    assert adopted.resnames == [a.resname for a in t.atoms]
    assert adopted.names_are_synthetic is False
    # Coordinates untouched, and the original is not mutated.
    assert np.allclose(adopted.coords, s.coords)
    assert s.names[0] == "C1"


def test_adopt_topology_naming_leaves_real_names_alone(butane_pdb, butane_itp):
    s = read_pdb(butane_pdb)
    t = read_topology(butane_itp)
    assert adopt_topology_naming(s, t) is s


def test_multimodel_pdb_uses_lf_line_endings(tmp_path, butane_pdb):
    """Item 15: consistent with ffrefine.mm.gromacs and ffrefine.io.topwriter."""
    s = read_pdb(butane_pdb)
    out = tmp_path / "traj.pdb"
    st.write_multimodel_pdb(out, s, np.stack([s.coords, s.coords]))
    assert b"\r\n" not in out.read_bytes()


def test_write_xyz_roundtrips_through_read_xyz(tmp_path, butane_pdb):
    from ffrefine.io.structure import read_pdb, read_xyz, write_xyz

    original = read_pdb(butane_pdb)
    out = tmp_path / "mol.xyz"
    write_xyz(out, original, comment="butane test")

    back = read_xyz(out)
    assert back.natoms == original.natoms
    assert back.elements == original.elements
    assert back.coords == pytest.approx(original.coords, abs=1e-6)


def test_write_xyz_is_in_angstrom_and_lf(tmp_path, butane_pdb):
    from ffrefine.io.structure import read_pdb, write_xyz

    s = read_pdb(butane_pdb)
    out = tmp_path / "mol.xyz"
    write_xyz(out, s, comment="c")

    raw = out.read_bytes()
    assert b"\r\n" not in raw
    lines = out.read_text().splitlines()
    assert lines[0].strip() == str(s.natoms)
    assert lines[1] == "c"
    # First atom's x in Angstrom is ten times the nm value.
    assert float(lines[2].split()[1]) == pytest.approx(s.coords[0][0] * 10.0, abs=1e-6)


def test_write_xyz_accepts_explicit_coords(tmp_path, butane_pdb):
    """Scan frames are passed as a separate array, not via structure.coords."""
    import numpy as np

    from ffrefine.io.structure import read_pdb, read_xyz, write_xyz

    s = read_pdb(butane_pdb)
    shifted = s.coords + 0.5
    out = tmp_path / "mol.xyz"
    write_xyz(out, s, coords=shifted)

    assert read_xyz(out).coords == pytest.approx(shifted, abs=1e-6)
