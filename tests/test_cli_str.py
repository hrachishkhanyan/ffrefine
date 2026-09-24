"""--str on the command line (milestone 6b design, sections 5 and 7)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ffrefine import cli
from ffrefine.cli import build_parser, main

FIXTURES = Path(__file__).parent.parent / "fixtures"
PFOA = FIXTURES / "pfoa_gromacs"
CAP = FIXTURES / "capecitabine_gromacs"


def _inputs(mol_dir: Path, stem: str) -> list[str]:
    return ["-s", str(mol_dir / f"{stem}_gmx.pdb"), "-p", str(mol_dir / f"{stem}_gmx.top")]


def _butane_str(path: Path, butane_itp: Path, penalties: dict[int, float]) -> Path:
    """A minimal CGenFF-style .str describing the butane fixture atom for atom."""
    from ffrefine.io.topology import read_topology

    t = read_topology(butane_itp)
    lines = ["* test stream", "read rtf card append", "RESI BUT 0.000", "GROUP"]
    for a in t.atoms:
        lines.append(f"ATOM {a.name} {a.atype} {a.charge:.3f} ! {penalties.get(a.nr, 0.0):.3f}")
    lines += ["END", "read param card flex append", "DIHEDRALS", "END", "RETURN"]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_penalty_threshold_parses_kind_and_value():
    args = build_parser().parse_args(
        ["scan", "-s", "a.pdb", "-p", "a.top", "--penalty-threshold", "dihedral=40"]
        + ["--penalty-threshold", "charge=12.5"]
    )
    assert cli._thresholds(args).dihedral == 40.0
    assert cli._thresholds(args).charge == 12.5
    defaults = build_parser().parse_args(["scan", "-s", "a.pdb", "-p", "a.top"])
    assert (cli._thresholds(defaults).charge, cli._thresholds(defaults).dihedral) == (10.0, 50.0)


@pytest.mark.parametrize("bad", ["dihedral", "angle=3", "charge=-1", "dihedral=x"])
def test_a_malformed_penalty_threshold_is_a_usage_error(bad, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["scan", "-s", "a.pdb", "-p", "a.top", "--penalty-threshold", bad])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--penalty-threshold" in err and "Traceback" not in err


def test_str_picks_the_one_flagged_bond_for_a_standalone_stage(tmp_path, capsys):
    argv = ["scan"] + _inputs(CAP, "capecitabine") + ["--str", str(CAP / "capecitabine.str")]
    assert main(argv + ["--outdir", str(tmp_path / "out"), "--dry-run"]) == 0
    assert "(6, 7, 9, 10)" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()  # a dry run writes nothing, selection.json included


def test_several_flagged_bonds_point_a_standalone_stage_to_run(tmp_path, capsys):
    argv = ["scan"] + _inputs(PFOA, "pfoa") + ["--str", str(PFOA / "pfoa.str")]
    assert main(argv + ["--outdir", str(tmp_path / "out"), "--dry-run"]) == 1
    err = capsys.readouterr().err
    assert "2 bonds" in err and "ffrefine run" in err and "1,3,4,5" in err


def test_no_flagged_bond_names_the_highest_penalty(tmp_path, capsys):
    argv = ["scan"] + _inputs(CAP, "capecitabine") + ["--str", str(CAP / "capecitabine.str")]
    argv += ["--penalty-threshold", "dihedral=51", "--outdir", str(tmp_path / "o"), "--dry-run"]
    assert main(argv) == 1
    err = capsys.readouterr().err
    assert "highest dihedral penalty" in err and "50" in err and "--penalty-threshold" in err


def test_a_missing_str_is_a_clean_error(tmp_path, capsys):
    """Review focus 5."""
    argv = ["scan"] + _inputs(CAP, "capecitabine") + ["--str", str(tmp_path / "nope.str")]
    assert main(argv + ["--outdir", str(tmp_path / "o"), "--dry-run"]) == 1
    err = capsys.readouterr().err
    assert "could not be read" in err and "Traceback" not in err


def test_two_dihedrals_on_a_standalone_stage_point_to_run(tmp_path, capsys):
    argv = ["scan"] + _inputs(PFOA, "pfoa") + ["--dihedral", "1,3,4,5", "--dihedral", "25,1,3,4"]
    assert main(argv + ["--outdir", str(tmp_path / "o"), "--dry-run"]) == 1
    assert "ffrefine run" in capsys.readouterr().err


def test_selection_is_computed_once_and_recorded(tmp_path):
    args = build_parser().parse_args(
        ["scan"]
        + _inputs(PFOA, "pfoa")
        + ["--str", str(PFOA / "pfoa.str"), "--outdir"]
        + [str(tmp_path / "out")]
    )
    cli._apply_config(args)
    from ffrefine.io.topology import read_topology

    t = read_topology(PFOA / "pfoa_gmx.top")
    first = cli._selection(args, t)
    assert cli._selection(args, t) is first
    data = json.loads((tmp_path / "out" / "00_input" / "selection.json").read_text())
    assert [b["quad_1_based"] for b in data["bonds"]] == [[1, 3, 4, 5], [25, 1, 3, 4]]
    assert (tmp_path / "out" / "00_input" / "pfoa.str").read_text() == (
        PFOA / "pfoa.str"
    ).read_text()


def test_charges_str_refines_the_flagged_atoms_and_their_group(
    tmp_path, butane_pdb, butane_itp, fake_esp
):
    """A flagged hydrogen brings its whole methyl group (design section 4)."""
    from ffrefine.io.topology import read_topology

    t = read_topology(butane_itp)
    first_h = next(a.nr for a in t.atoms if a.atype == "HGA3")
    strfile = _butane_str(tmp_path / "but.str", butane_itp, {first_h: 20.0})
    out = tmp_path / "out"
    argv = ["charges", "-s", str(butane_pdb), "-p", str(butane_itp), "--str", str(strfile)]
    assert main(argv + ["--charge-geom", "input", "--outdir", str(out)]) == 0
    charges = json.loads((out / "04_charges" / "charges.json").read_text())
    selected = charges["provenance"]["selected_atoms"]
    assert len(selected) == 3 and first_h in selected


def test_charges_str_with_nothing_flagged_runs_no_orca(
    tmp_path, butane_pdb, butane_itp, fake_esp, capsys
):
    strfile = _butane_str(tmp_path / "but.str", butane_itp, {})
    out = tmp_path / "out"
    argv = ["charges", "-s", str(butane_pdb), "-p", str(butane_itp), "--str", str(strfile)]
    assert main(argv + ["--charge-geom", "input", "--outdir", str(out)]) == 0
    assert "nothing to refine" in capsys.readouterr().out
    assert fake_esp.calls == 0
    assert not (out / "04_charges" / "charges.json").exists()


def test_replacing_most_of_the_molecule_warns(tmp_path, butane_pdb, butane_itp, fake_esp, capsys):
    from ffrefine.io.topology import read_topology

    t = read_topology(butane_itp)
    strfile = _butane_str(tmp_path / "but.str", butane_itp, {a.nr: 20.0 for a in t.atoms})
    out = tmp_path / "out"
    argv = ["charges", "-s", str(butane_pdb), "-p", str(butane_itp), "--str", str(strfile)]
    assert main(argv + ["--charge-geom", "input", "--outdir", str(out)]) == 0
    charges = json.loads((out / "04_charges" / "charges.json").read_text())
    assert any("close to replacing every charge" in w for w in charges["warnings"])
