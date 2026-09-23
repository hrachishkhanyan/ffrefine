import re
from pathlib import Path

import numpy as np
import pytest
from ffrefine.io.includes import MARKER_NOTE, MARKER_PREFIX
from ffrefine.io.topology import read_topology
from ffrefine.io.topwriter import (
    CHARGES_PROVENANCE_PREFIX,
    charges_provenance,
    make_system_top,
    strip_system_sections,
    write_charged_topology,
    write_molecule_copy,
    write_refit_topology,
    write_zeroed_topology,
)
from ffrefine.mm.torsionset import collect_torsion_set


def test_zeroed_topology_returns_all_ten_removed_term_lines(tmp_path, butane_itp):
    """Spec section 4.5: the analytic self-check needs every removed term-line,
    not one per quadruplet -- dropping either half of the target's two
    funct-9 lines would silently omit its contribution from that sum.
    """
    t = read_topology(butane_itp)
    out = tmp_path / "zeroed.itp"
    removed = write_zeroed_topology(t, (1, 5, 8, 11), out)
    assert len(removed) == 10

    distinct_quads = {min((d.ai, d.aj, d.ak, d.al), (d.al, d.ak, d.aj, d.ai)) for d in removed}
    assert len(distinct_quads) == 9

    target = [d for d in removed if (d.ai, d.aj, d.ak, d.al) in ((1, 5, 8, 11), (11, 8, 5, 1))]
    assert {int(d.params[2]) for d in target} == {2, 3}
    assert {round(d.params[1], 5) for d in target} == {0.58576, 0.41840}

    z = read_topology(out)
    tset = collect_torsion_set(z, (1, 5, 8, 11))
    assert len(tset) == 9
    for d in tset:
        assert d.params[1] == pytest.approx(0.0)


def test_zeroed_lines_use_funct_9_with_explicit_zeros(tmp_path, butane_itp):
    t = read_topology(butane_itp)
    out = tmp_path / "zeroed.itp"
    write_zeroed_topology(t, (1, 5, 8, 11), out)
    z = read_topology(out)
    for d in collect_torsion_set(z, (1, 5, 8, 11)):
        assert d.funct == 9
        assert len(d.params) == 3
        assert d.params == pytest.approx([0.0, 0.0, 1.0])


def test_input_file_is_not_modified(tmp_path, butane_itp):
    before = butane_itp.read_text()
    t = read_topology(butane_itp)
    write_zeroed_topology(t, (1, 5, 8, 11), tmp_path / "z.itp")
    assert butane_itp.read_text() == before


def test_other_sections_survive_untouched(tmp_path, butane_itp):
    t = read_topology(butane_itp)
    out = tmp_path / "zeroed.itp"
    write_zeroed_topology(t, (1, 5, 8, 11), out)
    z = read_topology(out)
    assert z.natoms == 14
    assert len(z.bonds) == 13
    assert len(z.pairs) == 27
    assert len(z.angles) == 24
    assert "[ atomtypes ]" in out.read_text()


def test_pairs_are_never_removed(tmp_path, butane_itp):
    """Spec section 9.2: 1-4 interactions must survive in both topologies."""
    t = read_topology(butane_itp)
    out = tmp_path / "zeroed.itp"
    write_zeroed_topology(t, (1, 5, 8, 11), out)
    assert read_topology(out).pairs == t.pairs


def test_methyl_torsions_survive_zeroing(tmp_path, butane_itp):
    t = read_topology(butane_itp)
    out = tmp_path / "zeroed.itp"
    write_zeroed_topology(t, (1, 5, 8, 11), out)
    z = read_topology(out)
    methyl = [d for d in z.propers if d.central_bond == (1, 5)]
    assert len(methyl) == 9
    assert all(d.params[1] > 0 for d in methyl)


def test_impropers_survive_zeroing(tmp_path, butane_itp):
    """Impropers (funct 2/4) are never modified (spec section 9.4), even when
    they happen to share atom indices and a central bond with a zeroed proper.
    """
    text = butane_itp.read_text().replace(
        "[ dihedrals ]\n",
        "[ dihedrals ]\n"
        "    1   5   8  11      4    0.000    10.00000     2  ; synthetic improper\n",
    )
    src = tmp_path / "with_improper.itp"
    src.write_text(text)
    t = read_topology(src)
    assert len(t.impropers) == 1

    out = tmp_path / "zeroed.itp"
    write_zeroed_topology(t, (1, 5, 8, 11), out)
    z = read_topology(out)
    assert len(z.impropers) == 1
    assert z.impropers[0] == t.impropers[0]


def test_refit_topology_writes_series_on_target_only(tmp_path, butane_itp):
    t = read_topology(butane_itp)
    out = tmp_path / "refit.itp"
    write_refit_topology(
        t,
        (1, 5, 8, 11),
        np.array([1.25, 0.4]),
        np.array([0.0, 180.0]),
        np.array([3, 2]),
        out,
        distribute=False,
        includes="absolute",
    )
    r = read_topology(out)
    target = [d for d in r.propers if (d.ai, d.aj, d.ak, d.al) in ((1, 5, 8, 11), (11, 8, 5, 1))]
    assert len(target) == 2
    assert {int(d.params[2]) for d in target} == {2, 3}
    others = [
        d
        for d in collect_torsion_set(r, (1, 5, 8, 11))
        if (d.ai, d.aj, d.ak, d.al) not in ((1, 5, 8, 11), (11, 8, 5, 1))
    ]
    assert all(d.params[1] == pytest.approx(0.0) for d in others)


def test_refit_distribute_divides_by_set_size(tmp_path, butane_itp):
    """Spec section 9.7: with --distribute, k_n is divided by N_T."""
    t = read_topology(butane_itp)
    out = tmp_path / "refit.itp"
    write_refit_topology(
        t,
        (1, 5, 8, 11),
        np.array([9.0]),
        np.array([0.0]),
        np.array([3]),
        out,
        distribute=True,
        includes="absolute",
    )
    r = read_topology(out)
    tset = collect_torsion_set(r, (1, 5, 8, 11))
    assert len(tset) == 9
    for d in tset:
        assert d.params[1] == pytest.approx(1.0)  # 9.0 / 9 members


def test_make_system_top(tmp_path, butane_itp):
    itp = tmp_path / "mol.itp"
    itp.write_text(butane_itp.read_text())
    top = tmp_path / "topol.top"
    make_system_top(itp, top, "BUT")
    text = top.read_text()
    assert "mol.itp" in text
    assert "[ molecules ]" in text
    assert "BUT" in text


SYSTEM_TAIL = """

[ system ]
butane in vacuum

[ molecules ]
BUT     1
"""


def _butane_top(tmp_path, butane_itp):
    """A molecule .top: butane plus its own [ system ] / [ molecules ]."""
    src = tmp_path / "butane.top"
    src.write_text(butane_itp.read_text() + SYSTEM_TAIL, newline="\n")
    return src


def test_strip_system_sections_drops_both_and_keeps_the_rest(tmp_path, butane_itp):
    src = _butane_top(tmp_path, butane_itp)
    kept, dropped = strip_system_sections(src.read_text().splitlines())
    assert dropped == ["system", "molecules"]
    body = "\n".join(kept)
    assert "[ system ]" not in body
    assert "[ molecules ]" not in body
    assert "butane in vacuum" not in body
    # Everything the molecule needs survives untouched.
    assert "[ moleculetype ]" in body
    assert "[ atomtypes ]" in body
    assert "[ dihedrals ]" in body


def test_strip_system_sections_is_a_no_op_for_a_plain_itp(butane_itp):
    kept, dropped = strip_system_sections(butane_itp.read_text().splitlines())
    assert dropped == []
    assert kept == butane_itp.read_text().splitlines()


def test_molecule_top_input_yields_a_single_molecule_wrapper(tmp_path, butane_itp):
    """Item 4: spec section 2 promises -p accepts a molecule .top.

    Before the fix the source was copied verbatim, so the wrapper declared the
    molecule twice and grompp aborted with "number of coordinates in
    coordinate file (14) does not match topology (28)".
    """
    src = _butane_top(tmp_path, butane_itp)
    copy = tmp_path / "full.itp"
    dropped = write_molecule_copy(src, copy)
    assert dropped == ["system", "molecules"]

    top = tmp_path / "full.top"
    make_system_top(copy, top, "BUT")
    combined = copy.read_text() + top.read_text()
    assert combined.count("[ molecules ]") == 1
    assert combined.count("[ system ]") == 1


def test_zeroed_and_refit_outputs_carry_no_system_sections(tmp_path, butane_itp):
    """Both rewriters feed make_system_top, so neither may pass those through."""
    t = read_topology(_butane_top(tmp_path, butane_itp))
    zeroed = tmp_path / "zeroed.itp"
    write_zeroed_topology(t, (1, 5, 8, 11), zeroed)
    refit = tmp_path / "refit.itp"
    write_refit_topology(
        t,
        (1, 5, 8, 11),
        np.array([1.0]),
        np.array([0.0]),
        np.array([3]),
        refit,
        includes="absolute",
    )
    for path in (zeroed, refit):
        text = path.read_text()
        assert "[ system ]" not in text, path.name
        assert "[ molecules ]" not in text, path.name
        assert "[ moleculetype ]" in text, path.name


def test_make_system_top_emits_ff_includes_before_the_molecule(tmp_path, butane_itp):
    """Item 4: the `includes` parameter was wired to nothing."""
    itp = tmp_path / "mol.itp"
    itp.write_text(butane_itp.read_text(), newline="\n")
    top = tmp_path / "topol.top"
    make_system_top(itp, top, "BUT", ["charmm36.ff/forcefield.itp", "extra.itp"])
    text = top.read_text()
    assert '#include "charmm36.ff/forcefield.itp"' in text
    assert '#include "extra.itp"' in text
    # Order matters to grompp: force field first, molecule after, system last.
    assert (
        text.index("charmm36.ff/forcefield.itp")
        < text.index("extra.itp")
        < text.index("mol.itp")
        < text.index("[ system ]")
    )


def test_writers_emit_lf_line_endings(tmp_path, butane_itp):
    """Item 15: consistent with ffrefine.mm.gromacs, which passes newline at
    every one of its write sites."""
    t = read_topology(butane_itp)
    zeroed = tmp_path / "zeroed.itp"
    write_zeroed_topology(t, (1, 5, 8, 11), zeroed)
    refit = tmp_path / "refit.itp"
    write_refit_topology(
        t,
        (1, 5, 8, 11),
        np.array([1.0]),
        np.array([0.0]),
        np.array([3]),
        refit,
        includes="absolute",
    )
    top = tmp_path / "topol.top"
    make_system_top(zeroed, top, "BUT")
    for path in (zeroed, refit, top):
        assert b"\r\n" not in path.read_bytes(), path.name


def test_charged_topology_changes_only_the_charge_column(tmp_path, butane_itp):
    """Everything but the edited charges must survive byte-for-byte."""
    from ffrefine.charges.apply import ChargeAssignment
    from ffrefine.io.topwriter import write_charged_topology

    assignments = [ChargeAssignment(nr=1, name="C1", before=-0.270, after=-0.4058, source="fitted")]
    out = tmp_path / "charged.itp"

    changed = write_charged_topology(butane_itp, assignments, out)

    assert changed == [1]
    before = butane_itp.read_text().splitlines()
    after = out.read_text().splitlines()
    assert len(before) == len(after)
    differing = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert len(differing) == 1
    assert "-0.4058" in after[differing[0]]
    assert "CG331" in after[differing[0]]  # the rest of the line is intact


def test_charged_topology_rewrites_nothing_when_no_charge_moved(tmp_path, butane_itp):
    from ffrefine.charges.apply import ChargeAssignment
    from ffrefine.io.topwriter import write_charged_topology

    assignments = [
        ChargeAssignment(nr=1, name="C1", before=-0.270, after=-0.270, source="unchanged")
    ]
    out = tmp_path / "charged.itp"

    assert write_charged_topology(butane_itp, assignments, out) == []
    assert out.read_text() == butane_itp.read_text()


def test_charged_topology_reads_back_with_the_new_charges(tmp_path, butane_itp):
    """The point of the file: the parser must see what was written."""
    from ffrefine.charges.apply import ChargeAssignment
    from ffrefine.io.topwriter import write_charged_topology
    from ffrefine.io.topology import read_topology

    assignments = [
        ChargeAssignment(nr=1, name="C1", before=-0.270, after=-0.4058, source="fitted"),
        ChargeAssignment(nr=11, name="C4", before=-0.270, after=-0.4091, source="fitted"),
    ]
    out = tmp_path / "charged.itp"
    write_charged_topology(butane_itp, assignments, out)

    top = read_topology(out)
    assert top.atoms[0].charge == pytest.approx(-0.4058)
    assert top.atoms[10].charge == pytest.approx(-0.4091)
    assert top.atoms[1].charge == pytest.approx(0.090)  # untouched


def test_charged_topology_refuses_an_atom_it_cannot_find(tmp_path, butane_itp):
    """A silently skipped atom would give a topology with the wrong total."""
    from ffrefine.charges.apply import ChargeAssignment
    from ffrefine.io.topwriter import write_charged_topology
    from ffrefine.errors import ValidationError

    assignments = [ChargeAssignment(nr=99, name="ZZ", before=0.0, after=0.5, source="fitted")]

    with pytest.raises(ValidationError, match="99"):
        write_charged_topology(butane_itp, assignments, tmp_path / "charged.itp")


def test_charged_topology_charge_not_confused_with_earlier_substring(tmp_path):
    """Regression: the charge column was located by an offset computed from
    the summed lengths of the preceding fields, which ignores the whitespace
    between them and so lands short of the real column. ``str.index`` then
    matched the first occurrence of the (short) charge value at or after
    that point -- here the trailing "1" of the atom type "CG331" -- instead
    of the actual charge field, corrupting the atom type while leaving the
    charge untouched and still reporting the atom as changed.
    """
    from ffrefine.charges.apply import ChargeAssignment
    from ffrefine.io.topwriter import write_charged_topology

    src = tmp_path / "tricky.itp"
    src.write_text(
        "[ moleculetype ]\n"
        "BUT     3\n"
        "\n"
        "[ atoms ]\n"
        ";  nr  type   resnr  residue  atom  cgnr   charge     mass\n"
        "     1   CG331      1    BUT    C1      1     1     12.011   ; comment\n"
    )
    out = tmp_path / "charged.itp"
    assignments = [ChargeAssignment(nr=1, name="C1", before=1.0, after=0.5, source="fitted")]

    changed = write_charged_topology(src, assignments, out)

    assert changed == [1]
    text = out.read_text()
    assert "CG331" in text  # atom type intact, not corrupted
    assert "CG330.5000" not in text  # the bug: new charge spliced into the type
    assert "0.5000" in text  # the charge itself actually moved


PFOA = Path(__file__).parent.parent / "fixtures" / "pfoa_gromacs"
PFOA_FF = (PFOA / "charmm36.ff").resolve()


def _include_lines(path) -> list[str]:
    return [
        line for line in Path(path).read_text().splitlines() if re.match(r"^\s*#\s*include", line)
    ]


def _refit_args(t, quad):
    return (t, quad, np.array([1.0]), np.array([0.0]), np.array([3]))


def test_grompp_facing_copies_carry_absolute_includes(tmp_path):
    """Design section 5: a stage-directory copy must still find charmm36.ff/."""
    t = read_topology(PFOA / "pfoa_gmx.top")
    write_molecule_copy(t.source, tmp_path / "full.itp")
    write_zeroed_topology(t, (25, 1, 3, 4), tmp_path / "zeroed.itp")
    write_refit_topology(
        *_refit_args(t, (25, 1, 3, 4)), tmp_path / "after.itp", includes="absolute"
    )
    for name in ("full.itp", "zeroed.itp", "after.itp"):
        lines = _include_lines(tmp_path / name)
        assert f'#include "{PFOA_FF / "forcefield.itp"}"' in lines, name
        assert f'# include "{PFOA_FF / "pfo_ffbonded.itp"}"' in lines, name
        assert '#include "posre.itp"' in lines, name  # not beside the .top: left for grompp
        assert MARKER_PREFIX not in (tmp_path / name).read_text(), name


def test_kept_copies_carry_the_users_includes_and_the_marker(tmp_path):
    """Design section 5: the files the user keeps are drop-in replacements."""
    t = read_topology(PFOA / "pfoa_gmx.top")
    write_refit_topology(
        *_refit_args(t, (25, 1, 3, 4)), tmp_path / "molecule_refined.itp", includes="verbatim"
    )
    write_charged_topology(PFOA / "pfoa_gmx.top", [], tmp_path / "molecule_charged.itp")
    for name in ("molecule_refined.itp", "molecule_charged.itp"):
        path = tmp_path / name
        assert _include_lines(path) == _include_lines(PFOA / "pfoa_gmx.top"), name
        assert path.read_text().splitlines()[:2] == [
            MARKER_NOTE,
            f"{MARKER_PREFIX}{PFOA.resolve()}",
        ], name


def test_a_marked_copy_resolves_its_includes_against_the_marked_directory(tmp_path):
    """The charged topology moves to 04_charges/, yet later copies find the force field."""
    moved = tmp_path / "out" / "04_charges"
    moved.mkdir(parents=True)
    write_charged_topology(PFOA / "pfoa_gmx.top", [], moved / "molecule_charged.itp")
    t = read_topology(moved / "molecule_charged.itp")
    write_zeroed_topology(t, (25, 1, 3, 4), tmp_path / "zeroed.itp")
    write_molecule_copy(moved / "molecule_charged.itp", tmp_path / "full.itp")
    for name in ("zeroed.itp", "full.itp"):
        assert f'#include "{PFOA_FF / "forcefield.itp"}"' in _include_lines(tmp_path / name)


def test_rewritten_copies_keep_ifdef_branches_the_parser_did_not_take(tmp_path):
    """Refinement 8: CGenFF's POSRES_WATER block must survive in every rewritten copy."""
    t = read_topology(PFOA / "pfoa_gmx.top")
    write_zeroed_topology(t, (25, 1, 3, 4), tmp_path / "zeroed.itp")
    write_refit_topology(
        *_refit_args(t, (25, 1, 3, 4)), tmp_path / "molecule_refined.itp", includes="verbatim"
    )
    for name in ("zeroed.itp", "molecule_refined.itp"):
        lines = [line.strip() for line in (tmp_path / name).read_text().splitlines()]
        start = lines.index("#ifdef POSRES_WATER")
        assert lines[start : start + 5] == [
            "#ifdef POSRES_WATER",
            "; Position restraint for each water oxygen",
            "[ position_restraints ]",
            ";  i funct       fcx        fcy        fcz",
            "1    1       1000       1000       1000",
        ], name


def test_butane_outputs_are_identical_in_both_modes(tmp_path, butane_itp):
    """No #include, so no marker and no rewritten line: master's bytes exactly."""
    t = read_topology(butane_itp)
    write_refit_topology(*_refit_args(t, (1, 5, 8, 11)), tmp_path / "a.itp", includes="absolute")
    write_refit_topology(*_refit_args(t, (1, 5, 8, 11)), tmp_path / "v.itp", includes="verbatim")
    assert (tmp_path / "a.itp").read_bytes() == (tmp_path / "v.itp").read_bytes()
    assert MARKER_PREFIX not in (tmp_path / "a.itp").read_text()


def test_write_refit_topology_requires_an_include_mode(tmp_path, butane_itp):
    t = read_topology(butane_itp)
    with pytest.raises(TypeError, match="includes"):
        write_refit_topology(*_refit_args(t, (1, 5, 8, 11)), tmp_path / "x.itp")


def test_charged_topology_records_where_its_charges_came_from(tmp_path, butane_itp):
    charges_json = tmp_path / "04_charges" / "charges.json"
    out = tmp_path / "charged.itp"
    write_charged_topology(butane_itp, [], out, charges_json=charges_json)
    assert charges_provenance(out) == charges_json.resolve()
    assert out.read_text().count(CHARGES_PROVENANCE_PREFIX) == 1


def test_charged_topology_replaces_an_earlier_provenance_line(tmp_path, butane_itp):
    first = tmp_path / "a.itp"
    write_charged_topology(butane_itp, [], first, charges_json=tmp_path / "old" / "charges.json")
    second = tmp_path / "b.itp"
    write_charged_topology(first, [], second, charges_json=tmp_path / "new" / "charges.json")
    assert charges_provenance(second) == (tmp_path / "new" / "charges.json").resolve()
    assert second.read_text().count(CHARGES_PROVENANCE_PREFIX) == 1


def test_provenance_survives_the_refit_writer(tmp_path, butane_itp):
    """Every later topology in a chain inherits it, bond after bond."""
    charges_json = tmp_path / "charges.json"
    charged = tmp_path / "charged.itp"
    write_charged_topology(butane_itp, [], charged, charges_json=charges_json)
    t = read_topology(charged)
    refined = tmp_path / "refined.itp"
    args = (t, (1, 5, 8, 11), np.array([1.0]), np.array([0.0]), np.array([3]))
    write_refit_topology(*args, refined, includes="verbatim")
    assert charges_provenance(refined) == charges_json.resolve()


def test_a_topology_without_the_line_has_no_provenance(butane_itp):
    assert charges_provenance(butane_itp) is None


def test_charged_topology_is_a_molecule_include(tmp_path):
    """Milestone 5 finding M7: a charged .top kept [ system ] and [ molecules ]."""
    out = tmp_path / "molecule_charged.itp"
    write_charged_topology(PFOA / "pfoa_gmx.top", [], out)
    text = out.read_text()
    assert "[ system ]" not in text and "[ molecules ]" not in text
    assert "[ moleculetype ]" in text
