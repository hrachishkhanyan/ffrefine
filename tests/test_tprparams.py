"""Reading grompp's own torsion parameters back (milestone 6a design, section 6).

The parser tests read a real `gmx dump` of the trimmed PFOA fixture, captured
with GROMACS 2025.4 (fixtures/pfoa_gromacs/PROVENANCE.md), plus text-edited
copies of it for every refusal.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from ffrefine.errors import ValidationError
from ffrefine.io.topology import Defaults, Dihedral
from ffrefine.mm.tprparams import (
    SOURCE_DIHEDRALTYPES,
    SOURCE_TOPOLOGY,
    DumpedTerm,
    ResolvedTerm,
    cross_check,
    parse_defaults,
    parse_dump_torsions,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"
PFOA = FIXTURES / "pfoa_gromacs"
DUMP = PFOA / "gromacs" / "pfoa_dump_2025.4.txt"
SRC = Path("dump.txt")


@pytest.fixture(scope="module")
def dump_text() -> str:
    return DUMP.read_text()


def _edit(text: str, pattern: str, repl: str) -> str:
    """Apply one regex edit, and fail loudly if it matched nothing."""
    out, n = re.subn(pattern, repl, text, count=1)
    assert n == 1, f"fixture layout changed: {pattern!r} not found"
    return out


# --------------------------------------------------------------------------
# parse_dump_torsions
# --------------------------------------------------------------------------


def test_reads_every_proper_term_of_the_pfoa_dump(dump_text):
    terms = parse_dump_torsions(dump_text, "Other", 26, DUMP)
    assert len(terms) == 76
    about = [t for t in terms if sorted(t.quad[1:3]) == [1, 3]]
    assert len(about) == 20
    target = [t for t in about if t.quad == (25, 1, 3, 4)]
    assert [t.multiplicity for t in target] == [1, 2, 3, 6]
    assert [t.phi0_deg for t in target] == [0.0, 0.0, 0.0, 0.0]
    # pfo_ffbonded.itp: OG311 CG2O2 CG312 CG312, stored by grompp as float32.
    assert [t.k_kj_per_mol for t in target] == pytest.approx(
        [0.66944, 0.46024, 0.33472, 1.046], rel=1e-6
    )


def test_keeps_the_written_atom_order(dump_text):
    """The dump keeps each [ dihedrals ] line's order: 2 1 3 4, never 4 3 1 2."""
    quads = {t.quad for t in parse_dump_torsions(dump_text, "Other", 26, DUMP)}
    assert (2, 1, 3, 4) in quads and (4, 3, 1, 2) not in quads


def test_refuses_a_missing_moltype(dump_text):
    with pytest.raises(ValidationError, match="exactly one moltype named 'Nope'"):
        parse_dump_torsions(dump_text, "Nope", 26, DUMP)


def test_refuses_an_atom_count_mismatch(dump_text):
    with pytest.raises(ValidationError, match="has 26 atoms, the topology 25"):
        parse_dump_torsions(dump_text, "Other", 25, DUMP)


def test_refuses_a_free_energy_topology(dump_text):
    text = _edit(dump_text, r"cpB= 2\.88696003e\+00", "cpB= 3.00000000e+00")
    with pytest.raises(ValidationError, match="different A and B states"):
        parse_dump_torsions(text, "Other", 26, DUMP)


def test_refuses_a_count_mismatch(dump_text):
    text = _edit(dump_text, r"(Proper Dih\.:\n\s*nr: )380", r"\g<1>385")
    with pytest.raises(ValidationError, match="nr: 385"):
        parse_dump_torsions(text, "Other", 26, DUMP)


def test_refuses_a_missing_proper_block(dump_text):
    text = _edit(dump_text, r"Proper Dih\.:", "Proper Dih.X:")
    with pytest.raises(ValidationError, match="no 'Proper Dih.' block"):
        parse_dump_torsions(text, "Other", 26, DUMP)


def test_refuses_an_entry_naming_a_non_pdihs_functype(dump_text):
    text = _edit(dump_text, r"(Proper Dih\.:\n\s*nr: \d+\n\s*iatoms:\n\s*0 type=)\d+", r"\g<1>0")
    with pytest.raises(ValidationError, match="not PDIHS"):
        parse_dump_torsions(text, "Other", 26, DUMP)


def test_refuses_an_unrecognised_pdihs_line(dump_text):
    text = _edit(dump_text, r"(functype\[\d+\]=PDIHS, .*?mult=)\d+", r"\g<1>x")
    with pytest.raises(ValidationError, match="unrecognised PDIHS functype line"):
        parse_dump_torsions(text, "Other", 26, DUMP)


def test_refusals_blame_the_reader_not_the_topology(dump_text):
    with pytest.raises(ValidationError) as excinfo:
        parse_dump_torsions(dump_text, "Nope", 26, DUMP)
    assert "not a problem with your topology" in str(excinfo.value)
    assert str(DUMP) in str(excinfo.value)


# --------------------------------------------------------------------------
# parse_defaults
# --------------------------------------------------------------------------


def test_parse_defaults_reads_the_defaults_line():
    text = "; header\n[ defaults ]\n; nbfunc comb-rule\n1  2  yes  1.0  1.0\n[ atomtypes ]\n"
    assert parse_defaults(text, SRC) == Defaults(1, 2, True, 1.0, 1.0)


def test_parse_defaults_without_the_directive():
    assert parse_defaults("[ atomtypes ]\nCG321 6 12.011 0 A 0.3 0.2\n", SRC) is None


def test_parse_defaults_refuses_a_malformed_line():
    with pytest.raises(ValidationError, match="malformed"):
        parse_defaults("[ defaults ]\none two\n", SRC)


# --------------------------------------------------------------------------
# cross_check (design section 6.3)
# --------------------------------------------------------------------------

Q = (1, 2, 3, 4)
R = (5, 2, 3, 4)


def _line(quad, params=(), funct=9) -> Dihedral:
    return Dihedral(*quad, funct=funct, params=[float(p) for p in params])


def _term(quad, phi, k, n) -> DumpedTerm:
    return DumpedTerm(quad, phi, k, n)


def test_a_typed_line_takes_the_dumped_terms():
    out = cross_check([_line(Q)], [_term(Q, 0.0, 1.5, 1), _term(Q, 180.0, 0.5, 2)], SRC)
    assert [(r.dihedral.params, r.source) for r in out] == [
        ([0.0, 1.5, 1.0], SOURCE_DIHEDRALTYPES),
        ([180.0, 0.5, 2.0], SOURCE_DIHEDRALTYPES),
    ]
    assert all((r.dihedral.ai, r.dihedral.aj, r.dihedral.ak, r.dihedral.al) == Q for r in out)
    assert all(r.dihedral.funct == 9 for r in out)


def test_a_typed_line_with_no_dumped_terms_contributes_nothing():
    """Every CGenFF term k = 0: grompp builds nothing (capecitabine's 6 7 9 37)."""
    explicit = _line(R, (0.0, 2.0, 3))
    assert cross_check([_line(Q), explicit], [_term(R, 0.0, 2.0, 3)], SRC) == [
        ResolvedTerm(explicit, SOURCE_TOPOLOGY)
    ]


def test_explicit_lines_keep_their_own_values_within_float32_rounding():
    line = _line(Q, (0.0, 2.0, 3))
    out = cross_check([line], [_term(Q, 0.0, 2.0000001, 3)], SRC)
    assert out == [ResolvedTerm(line, SOURCE_TOPOLOGY)]


def test_an_explicit_line_that_disagrees_with_the_dump_is_refused():
    with pytest.raises(ValidationError, match="disagree"):
        cross_check([_line(Q, (0.0, 2.0, 3))], [_term(Q, 0.0, 2.5, 3)], SRC)


def test_an_explicit_zero_k_line_is_expected_absent_from_the_dump():
    line = _line(Q, (0.0, 0.0, 1))
    assert cross_check([line], [], SRC) == [ResolvedTerm(line, SOURCE_TOPOLOGY)]


def test_a_quad_both_typed_and_explicit_is_refused():
    with pytest.raises(ValidationError, match="ambiguous"):
        cross_check([_line(Q), _line(Q, (0.0, 1.0, 3))], [_term(Q, 0.0, 1.0, 3)], SRC)


def test_a_quad_typed_twice_is_refused():
    with pytest.raises(ValidationError, match="ambiguous"):
        cross_check([_line(Q), _line(Q)], [_term(Q, 0.0, 1.0, 3)] * 2, SRC)


def test_a_dumped_term_no_line_accounts_for_is_refused():
    with pytest.raises(ValidationError, match=r"no \[ dihedrals \] line"):
        cross_check(
            [_line(Q, (0.0, 1.0, 3))],
            [_term(Q, 0.0, 1.0, 3), _term((6, 3, 2, 7), 0.0, 1.0, 3)],
            SRC,
        )


def test_terms_about_other_bonds_are_ignored():
    out = cross_check([_line(Q)], [_term(Q, 0.0, 1.0, 3), _term((2, 3, 4, 8), 0.0, 9.0, 1)], SRC)
    assert len(out) == 1


def test_an_empty_torsion_set_resolves_to_nothing():
    assert cross_check([], [_term(Q, 0.0, 1.0, 3)], SRC) == []
