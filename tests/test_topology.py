import pytest
from ffrefine.errors import ParseError
from ffrefine.io import topology as tp
from ffrefine.io.includes import MARKER_NOTE, MARKER_PREFIX
from ffrefine.io.topology import Defaults, parse_defaults_fields

TOP = """\
[ defaults ]
; nbfunc  comb-rule  gen-pairs  fudgeLJ  fudgeQQ
1         2          yes        1.0      1.0

#include "charmm36.itp"

[ moleculetype ]
; name  nrexcl
BUT     3

[ atoms ]
;  nr  type  resnr  residue  atom  cgnr  charge    mass
    1  CG331     1  BUT      C1       1  -0.270  12.0110
    2  CG321     1  BUT      C2       1  -0.180  12.0110
    3  CG321     1  BUT      C3       2  -0.180  12.0110
    4  CG331     1  BUT      C4       2  -0.270  12.0110

[ bonds ]
    1  2
    2  3
    3  4

[ pairs ]
    1  4

[ angles ]
    1  2  3
    2  3  4

[ dihedrals ]
;   ai  aj  ak  al  funct   phi0     kphi    mult
     1   2   3   4      9   0.000   0.5858      3
     1   2   3   4      9 180.000   0.4184      2

[ dihedrals ]
; impropers
     1   2   3   4      2   0.000  10.0000

#ifdef POSRES
[ position_restraints ]
     1   1  1000  1000  1000
#endif
"""

# Same as TOP but with an extra proper dihedral wrapped in its own #ifdef
# POSRES block, used to test real conditional-compilation behaviour (ruling
# R5): the brief's original test asserted a tautology (`"POSRES" not in
# line` is true for virtually any line), so instead we assert that a
# dihedral entry guarded by #ifdef POSRES is absent by default and present
# when POSRES is defined.
TOP_WITH_IFDEF_DIHEDRAL = TOP.replace(
    "#ifdef POSRES\n[ position_restraints ]",
    "#ifdef POSRES\n"
    "[ dihedrals ]\n"
    "     1   2   3   4      9  60.000   0.1000      1\n"
    "#endif\n"
    "\n"
    "#ifdef POSRES\n"
    "[ position_restraints ]",
)


def _top(tmp_path, text=TOP, name="mol.itp"):
    p = tmp_path / name
    p.write_text(text)
    return tp.read_topology(p)


def test_defaults_parsed(tmp_path):
    t = _top(tmp_path)
    assert t.defaults.fudge_qq == pytest.approx(1.0)
    assert t.defaults.fudge_lj == pytest.approx(1.0)
    assert t.defaults.comb_rule == 2
    assert t.defaults.gen_pairs is True


def test_moleculetype(tmp_path):
    t = _top(tmp_path)
    assert t.name == "BUT"
    assert t.nrexcl == 3


def test_atoms(tmp_path):
    t = _top(tmp_path)
    assert t.natoms == 4
    assert t.atoms[0].atype == "CG331"
    assert t.atoms[0].charge == pytest.approx(-0.270)
    assert t.atoms[0].mass == pytest.approx(12.0110)
    assert t.atoms[2].cgnr == 2


def test_atom_indices_are_one_based(tmp_path):
    t = _top(tmp_path)
    assert [a.nr for a in t.atoms] == [1, 2, 3, 4]


def test_bonds_pairs_angles(tmp_path):
    t = _top(tmp_path)
    assert t.bonds == [(1, 2), (2, 3), (3, 4)]
    assert t.pairs == [(1, 4)]
    assert t.angles == [(1, 2, 3), (2, 3, 4)]


def test_propers_multiline_funct9(tmp_path):
    t = _top(tmp_path)
    assert len(t.propers) == 2
    assert all(d.funct == 9 for d in t.propers)
    assert t.propers[0].params == pytest.approx([0.0, 0.5858, 3])
    assert t.propers[1].params == pytest.approx([180.0, 0.4184, 2])


def test_impropers_not_picked_up_as_propers(tmp_path):
    """Spec section 7.5 and 9.4: funct 2 is an improper and must stay untouched."""
    t = _top(tmp_path)
    assert len(t.impropers) == 1
    assert t.impropers[0].funct == 2
    assert all(d.funct != 2 for d in t.propers)


def test_dihedral_without_explicit_parameters(tmp_path):
    text = TOP.replace(
        "     1   2   3   4      9   0.000   0.5858      3\n"
        "     1   2   3   4      9 180.000   0.4184      2\n",
        "     1   2   3   4      9\n",
    )
    t = _top(tmp_path, text)
    assert len(t.propers) == 1
    assert t.propers[0].params == []


def test_ifdef_block_skipped_by_default(tmp_path):
    t = _top(tmp_path)
    assert not any("position_restraints" in line for line in t.raw_lines if line.startswith("["))


def test_ifdef_dihedral_included_only_when_defined(tmp_path):
    """Ruling R5: a dihedral guarded by #ifdef POSRES must be excluded from the
    parsed propers by default, and included when POSRES is passed in `defines`.
    """
    p = tmp_path / "m.itp"
    p.write_text(TOP_WITH_IFDEF_DIHEDRAL)

    t_default = tp.read_topology(p)
    assert len(t_default.propers) == 2

    t_defined = tp.read_topology(p, defines={"POSRES"})
    assert len(t_defined.propers) == 3
    assert t_defined.propers[-1].params == pytest.approx([60.0, 0.1000, 1])


def test_include_directive_does_not_crash(tmp_path):
    t = _top(tmp_path)
    assert t.natoms == 4


def test_comments_stripped_from_data(tmp_path):
    t = _top(tmp_path)
    assert t.atoms[0].name == "C1"


def test_bonded_neighbours(tmp_path):
    t = _top(tmp_path)
    assert sorted(t.bonded_neighbours(2)) == [1, 3]
    assert sorted(t.bonded_neighbours(1)) == [2]


def test_missing_atoms_section_raises(tmp_path):
    p = tmp_path / "bad.itp"
    p.write_text("[ moleculetype ]\nBUT 3\n")
    with pytest.raises(ParseError, match="atoms"):
        tp.read_topology(p)


def test_topology_include_base_defaults_to_its_directory(tmp_path):
    p = tmp_path / "butane.itp"
    p.write_text(TOP)
    t = tp.read_topology(p)
    assert t.include_base == p.resolve().parent


def test_topology_include_base_reads_the_marker(tmp_path):
    p = tmp_path / "copy.itp"
    base = tmp_path.resolve()
    p.write_text(f"{MARKER_NOTE}\n{MARKER_PREFIX}{base}\n" + TOP)
    t = tp.read_topology(p)
    assert t.include_marker == base
    assert t.include_base == base


def test_parse_defaults_fields():
    assert parse_defaults_fields(["1", "2", "yes", "0.5", "0.8333"]) == Defaults(
        nbfunc=1, comb_rule=2, gen_pairs=True, fudge_lj=0.5, fudge_qq=0.8333
    )
    assert parse_defaults_fields(["1", "1"]) == Defaults(1, 1, False, 1.0, 1.0)
