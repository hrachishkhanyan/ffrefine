"""run: charges once, then every bond in sequence (milestone 6b design, section 5).

The stage commands are replaced by stand-ins that record what they were
given; the stand-in validate writes the refined topology the next bond reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ffrefine import cli
from ffrefine.cli import main
from ffrefine.errors import ValidationError

FIXTURES = Path(__file__).parent.parent / "fixtures"
PFOA = FIXTURES / "pfoa_gromacs"
CAP = FIXTURES / "capecitabine_gromacs"


def _pfoa(outdir: Path, *extra: str) -> list[str]:
    return [
        "run",
        "-s",
        str(PFOA / "pfoa_gmx.pdb"),
        "-p",
        str(PFOA / "pfoa_gmx.top"),
        "--str",
        str(PFOA / "pfoa.str"),
        "--outdir",
        str(outdir),
        *extra,
    ]


def _cap(outdir: Path, *extra: str) -> list[str]:
    return [
        "run",
        "-s",
        str(CAP / "capecitabine_gmx.pdb"),
        "-p",
        str(CAP / "capecitabine_gmx.top"),
        "--str",
        str(CAP / "capecitabine.str"),
        "--outdir",
        str(outdir),
        *extra,
    ]


@pytest.fixture
def stages(monkeypatch):
    """Stand-in stage commands; returns the list of (stage, outdir, topology, dihedral)."""
    calls: list[tuple] = []

    def record(name):
        def stage(a):
            calls.append((name, Path(a.outdir), a.topology, a.dihedral and list(a.dihedral)))
            if name == "validate":
                refined = Path(a.outdir) / "final" / "molecule_refined.itp"
                refined.parent.mkdir(parents=True, exist_ok=True)
                refined.write_text(f"; refined {a.dihedral[0]} from {a.topology}\n")
            if name == "charges":
                charged = Path(a.outdir) / "04_charges" / "molecule_charged.itp"
                charged.parent.mkdir(parents=True, exist_ok=True)
                charged.write_text(Path(a.topology).read_text())
            return 0

        return stage

    for name in ("charges", "scan", "mmscan", "fit", "validate"):
        monkeypatch.setattr(cli, f"cmd_{name}", record(name))
    combined: list[tuple] = []
    monkeypatch.setattr(
        cli,
        "_combined_check",
        lambda a, bonds, original, final: combined.append((bonds, original, final)),
    )
    calls_and_combined = (calls, combined)
    return calls_and_combined


def test_bonds_run_in_penalty_order_each_on_the_previous_ones_topology(tmp_path, stages):
    calls, combined = stages
    out = tmp_path / "out"
    assert main(_pfoa(out, "--no-charges")) == 0
    torsion_calls = [c for c in calls if c[0] != "charges"]
    assert [c[0] for c in torsion_calls] == ["scan", "mmscan", "fit", "validate"] * 2
    first, second = torsion_calls[:4], torsion_calls[4:]
    assert {c[1] for c in first} == {out / "torsion_1-3-4-5"}
    assert {c[1] for c in second} == {out / "torsion_25-1-3-4"}
    assert {c[3][0] for c in first} == {(1, 3, 4, 5)}
    assert {str(c[2]) for c in first} == {str(PFOA / "pfoa_gmx.top")}
    bond1_refined = out / "torsion_1-3-4-5" / "final" / "molecule_refined.itp"
    assert {str(c[2]) for c in second} == {str(bond1_refined)}
    assert not [c for c in calls if c[0] == "charges"]

    [(bonds, original, final)] = combined
    assert [q for q, _ in bonds] == [(1, 3, 4, 5), (25, 1, 3, 4)]
    assert Path(original) == PFOA / "pfoa_gmx.top"
    last = out / "torsion_25-1-3-4" / "final" / "molecule_refined.itp"
    assert Path(final) == last
    assert (out / "final" / "molecule_refined.itp").read_bytes() == last.read_bytes()

    status = json.loads((out / "final" / "run_status.json").read_text())
    assert status["charges"] == "skipped"
    assert [t["status"] for t in status["torsions"]] == ["complete", "complete"]
    assert [t["dir"] for t in status["torsions"]] == ["torsion_1-3-4-5", "torsion_25-1-3-4"]
    assert status["combined"] == "complete"


def test_flagged_charges_run_first_and_feed_the_first_bond(tmp_path, stages):
    calls, _ = stages
    out = tmp_path / "out"
    assert main(_pfoa(out)) == 0
    assert calls[0][0] == "charges"
    charged = out / "04_charges" / "molecule_charged.itp"
    first_scan = next(c for c in calls if c[0] == "scan")
    assert str(first_scan[2]) == str(charged)


def test_one_flagged_torsion_keeps_todays_layout(tmp_path, stages):
    calls, combined = stages
    out = tmp_path / "out"
    assert main(_cap(out)) == 0
    assert [c[0] for c in calls] == [
        "scan",
        "mmscan",
        "fit",
        "validate",
    ]  # nothing flagged to charge
    assert {c[1] for c in calls} == {out}
    assert {c[3][0] for c in calls} == {(6, 7, 9, 10)}
    assert combined == []
    status = json.loads((out / "final" / "run_status.json").read_text())
    assert status["torsions"][0]["dir"] == "." and status["combined"] == "not applicable"


def test_a_failing_bond_stops_the_run_and_records_where(tmp_path, stages, monkeypatch, capsys):
    calls, combined = stages
    out = tmp_path / "out"
    fit = cli.cmd_fit

    def failing_fit(a):
        if a.dihedral[0] == (25, 1, 3, 4):
            raise ValidationError("the fit fell over")
        return fit(a)

    monkeypatch.setattr(cli, "cmd_fit", failing_fit)
    assert main(_pfoa(out, "--no-charges")) == 1
    assert "the fit fell over" in capsys.readouterr().err
    status = json.loads((out / "final" / "run_status.json").read_text())
    assert status["torsions"][0]["status"] == "complete"
    assert status["torsions"][1] == {
        "quad_1_based": [25, 1, 3, 4],
        "dir": "torsion_25-1-3-4",
        "status": "failed",
        "failed_stage": "fit",
    }
    assert status["combined"] == "not run" and combined == []
    assert not (out / "final" / "molecule_refined.itp").exists()


def test_no_flagged_torsion_stops_cleanly(tmp_path, stages, capsys):
    calls, _ = stages
    out = tmp_path / "out"
    assert main(_cap(out, "--penalty-threshold", "dihedral=51")) == 0
    assert calls == []
    out_text = capsys.readouterr().out
    assert "no torsion" in out_text and "highest dihedral penalty is 50" in out_text


def test_no_charges_conflicts_with_asking_for_charges(tmp_path, stages, capsys):
    out = tmp_path / "out"
    assert main(_pfoa(out, "--no-charges", "--charge-mode", "all")) == 1
    assert "--no-charges" in capsys.readouterr().err


def test_a_multi_bond_dry_run_prints_the_plan_and_writes_nothing(tmp_path, stages, capsys):
    calls, _ = stages
    out = tmp_path / "out"
    assert main(_pfoa(out, "--dry-run")) == 0
    text = capsys.readouterr().out
    assert "torsion_1-3-4-5" in text and "torsion_25-1-3-4" in text and "23 flagged" in text
    assert calls == [] and not out.exists()


def test_repeated_dihedrals_run_in_the_order_given(tmp_path, stages):
    calls, _ = stages
    out = tmp_path / "out"
    argv = [
        "run",
        "-s",
        str(PFOA / "pfoa_gmx.pdb"),
        "-p",
        str(PFOA / "pfoa_gmx.top"),
        "--dihedral",
        "25,1,3,4",
        "--dihedral",
        "1,3,4,5",
        "--outdir",
        str(out),
    ]
    assert main(argv) == 0
    scans = [c for c in calls if c[0] == "scan"]
    assert [c[1].name for c in scans] == ["torsion_25-1-3-4", "torsion_1-3-4-5"]


def test_run_writes_the_report_even_when_a_bond_fails(tmp_path, stages, monkeypatch):
    calls, _ = stages

    def failing(a):
        raise ValidationError("scan fell over")

    monkeypatch.setattr(cli, "cmd_scan", failing)
    out = tmp_path / "out"
    assert main(_pfoa(out, "--no-charges")) == 1
    report = (out / "final" / "report.md").read_text(encoding="utf-8")
    assert "failed (scan)" in report and "not reached" in report


def test_run_with_nothing_flagged_still_writes_a_report(tmp_path, stages):
    out = tmp_path / "out"
    assert main(_cap(out, "--penalty-threshold", "dihedral=51")) == 0
    assert "No torsion was refit" in (out / "final" / "report.md").read_text(encoding="utf-8")


def _boom(outdir):
    raise RuntimeError("report.py has a bug")


def test_a_report_bug_does_not_change_a_successful_runs_outcome(tmp_path, stages, monkeypatch):
    """The report is a by-product; a bug writing it must not fail the run itself."""
    monkeypatch.setattr(cli, "write_report", _boom)
    out = tmp_path / "out"
    assert main(_cap(out)) == 0
    assert not (out / "final" / "report.md").exists()


def test_a_report_bug_does_not_mask_a_failing_stages_error(tmp_path, stages, monkeypatch, capsys):
    """A RuntimeError from report.py must not replace the stage's own FFRefineError."""
    calls, _ = stages

    def failing(a):
        raise ValidationError("scan fell over")

    monkeypatch.setattr(cli, "cmd_scan", failing)
    monkeypatch.setattr(cli, "write_report", _boom)
    out = tmp_path / "out"
    assert main(_pfoa(out, "--no-charges")) == 1
    assert "scan fell over" in capsys.readouterr().err


# --- final review I1: run owns its outdir; a stale selection.json must not survive


def test_run_without_str_removes_a_stale_selection_json(tmp_path, stages):
    calls, _ = stages
    out = tmp_path / "out"
    assert main(_pfoa(out, "--no-charges")) == 0
    selection_path = out / "00_input" / "selection.json"
    str_copy = out / "00_input" / "pfoa.str"
    assert selection_path.exists() and str_copy.exists()

    argv = [
        "run",
        "-s",
        str(PFOA / "pfoa_gmx.pdb"),
        "-p",
        str(PFOA / "pfoa_gmx.top"),
        "--dihedral",
        "1,3,4,5",
        "--outdir",
        str(out),
    ]
    assert main(argv) == 0
    assert not selection_path.exists()
    assert not str_copy.exists()
