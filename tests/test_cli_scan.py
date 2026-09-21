import json

import numpy as np
import pytest

from ffrefine.cli import build_parser, main


def test_scan_is_no_longer_reported_as_unimplemented(butane_pdb, butane_itp, tmp_path, capsys):
    rc = main(
        [
            "scan",
            "-s",
            str(butane_pdb),
            "-p",
            str(butane_itp),
            "--dihedral",
            "1,5,8,11",
            "--outdir",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "NOT IMPLEMENTED" not in out.upper()


def test_scan_dry_run_writes_nothing(butane_pdb, butane_itp, tmp_path):
    outdir = tmp_path / "out"
    main(
        [
            "scan",
            "-s",
            str(butane_pdb),
            "-p",
            str(butane_itp),
            "--dihedral",
            "1,5,8,11",
            "--outdir",
            str(outdir),
            "--dry-run",
        ]
    )
    assert not outdir.exists()


def test_help_no_longer_advertises_any_stage_as_unimplemented(capsys):
    """The subcommand list is where a user looks to see what the tool can do."""
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out

    assert "milestone 4" not in out  # scan is implemented now
    assert "milestone 5" not in out  # charges (chelpg) is implemented now too


def test_scan_rejects_a_step_that_does_not_divide_360(butane_pdb, butane_itp, tmp_path, capsys):
    rc = main(
        [
            "scan",
            "-s",
            str(butane_pdb),
            "-p",
            str(butane_itp),
            "--dihedral",
            "1,5,8,11",
            "--outdir",
            str(tmp_path / "o"),
            "--step",
            "7",
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert "Traceback" not in err


@pytest.mark.needs_xtb
def test_scan_writes_a_scan_json_the_fit_pipeline_accepts(
    butane_pdb, butane_itp, tmp_path, xtb_bin
):
    """End to end on real xtb, at 4 points to stay quick (24 would be minutes).

    Single-threaded deliberately.  The R44 guard in run_scan reports that the
    xtb relaxed-scan optimiser can stall under -P greater than 1 and hand back
    every frame identical to the input, and it does so intermittently -- this
    suite saw it pass at -P 4 in one test and trip the guard in the next.  A
    test that is fast most of the time and wrong the rest is worse than a slow
    one, so it takes the tool's own advice.
    """
    outdir = tmp_path / "out"
    rc = main(
        [
            "scan",
            "-s",
            str(butane_pdb),
            "-p",
            str(butane_itp),
            "--dihedral",
            "1,5,8,11",
            "--outdir",
            str(outdir),
            "--step",
            "90",
            "--no-bidirectional",
            "--nthreads",
            "1",
        ]
    )
    assert rc == 0

    payload = json.loads((outdir / "01_qm_scan" / "scan.json").read_text())
    frames = np.array(payload["frames_nm"])
    assert frames.shape == (4, 14, 3)
    assert len(payload["energy_kj"]) == 4
    assert len(payload["phi_deg"]) == 4
    assert payload["quad"] == [1, 5, 8, 11]
    # Coordinates are nm, so every butane atom sits within a couple of nm of the origin.
    assert np.abs(frames).max() < 5.0

    prov = payload["provenance"]
    assert prov["charge"] == 0
    assert prov["uhf"] == 0
    assert prov["step_deg"] == 90.0
    assert prov["bidirectional"] is False
    assert prov["xtb_version"]
    assert prov["sp_refine_keywords"] is None  # --sp-refine is opt-in
    assert (outdir / "01_qm_scan" / "scan.pdb").exists()


@pytest.mark.needs_xtb
def test_scan_output_feeds_mmscan_without_hand_editing(
    butane_pdb, butane_itp, tmp_path, xtb_bin, gmx_bin
):
    """The point of the milestone: no hand-written scan.json anywhere."""
    outdir = tmp_path / "out"
    assert (
        main(
            [
                "scan",
                "-s",
                str(butane_pdb),
                "-p",
                str(butane_itp),
                "--dihedral",
                "1,5,8,11",
                "--outdir",
                str(outdir),
                "--step",
                "90",
                "--no-bidirectional",
                "--nthreads",
                "1",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "mmscan",
                "-s",
                str(butane_pdb),
                "-p",
                str(butane_itp),
                "--dihedral",
                "1,5,8,11",
                "--outdir",
                str(outdir),
                "--nthreads",
                "2",
            ]
        )
        == 0
    )

    energies = json.loads((outdir / "02_mm_scan" / "energies.json").read_text())
    assert energies["quad"] == [1, 5, 8, 11]
    assert len(energies["phi_deg"]) == 4


def test_run_chains_the_scan_stage_now_that_it_exists(butane_pdb, butane_itp, tmp_path, capsys):
    """run told users to hand-supply scan.json; that was the gap milestone 4 closed."""
    rc = main(
        [
            "run",
            "-s",
            str(butane_pdb),
            "-p",
            str(butane_itp),
            "--dihedral",
            "1,5,8,11",
            "--outdir",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "milestone 4" not in out
    assert "must therefore be supplied" not in out
    assert "=== scan ===" in out
    # charges is implemented but not requested on this invocation, so it is
    # named as skipped rather than reported missing.
    assert "milestone 5" not in out
    assert "SKIPPING charges" in out


def test_run_accepts_the_scan_flags(butane_pdb, butane_itp, tmp_path):
    """cmd_run calls cmd_scan, which reads args.step and args.bidirectional.

    Without these on the run subparser that call raises AttributeError, which
    is a crash rather than the clean error the CLI promises everywhere else.
    """
    rc = main(
        [
            "run",
            "-s",
            str(butane_pdb),
            "-p",
            str(butane_itp),
            "--dihedral",
            "1,5,8,11",
            "--outdir",
            str(tmp_path / "out"),
            "--step",
            "90",
            "--no-bidirectional",
            "--dry-run",
        ]
    )
    assert rc == 0


def test_scan_accepts_the_sp_refine_flags():
    args = build_parser().parse_args(
        [
            "scan",
            "-s",
            "a.pdb",
            "-p",
            "a.itp",
            "--dihedral",
            "1,5,8,11",
            "--sp-refine",
            "--orca-sp-keywords",
            "! HF STO-3G",
        ]
    )
    assert args.sp_refine is True
    assert args.orca_sp_keywords == "! HF STO-3G"


def test_run_also_accepts_sp_refine():
    """run chains scan, so every flag cmd_scan reads has to reach it."""
    args = build_parser().parse_args(
        ["run", "-s", "a.pdb", "-p", "a.itp", "--dihedral", "1,5,8,11", "--sp-refine"]
    )
    assert args.sp_refine is True


def test_scan_defaults_to_no_sp_refinement():
    """The ORCA default is hybrid DFT on every frame; it must be opt-in."""
    args = build_parser().parse_args(
        ["scan", "-s", "a.pdb", "-p", "a.itp", "--dihedral", "1,5,8,11"]
    )
    assert args.sp_refine is False
    assert args.orca_sp_keywords is None
