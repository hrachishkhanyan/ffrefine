"""CLI surface and wiring (spec section 3, task 22).

Tests call :func:`ffrefine.cli.main` directly with an argv list rather than
spawning a subprocess: it is faster and a failure surfaces as a real
traceback instead of a captured exit code.

Deliberate deviation from the task brief's test bodies
-----------------------------------------------------
The brief's ``parse_dihedral`` tests expect ``SystemExit``.  Controller
ruling R29 requires every validation failure in this project to raise
:class:`ffrefine.errors.ValidationError`, "even where the brief's code says
otherwise", so ``parse_dihedral`` raises that instead and the parser adapts
it to argparse's own error channel.  The user-visible behaviour the brief's
tests were reaching for -- a bad ``--dihedral`` on the command line producing
a clean message and a nonzero exit rather than a traceback -- is asserted
directly in ``test_bad_dihedral_on_the_command_line_exits_cleanly``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from ffrefine.cli import build_parser, main, parse_dihedral
from ffrefine.errors import ValidationError
from ffrefine.io.structure import read_pdb
from ffrefine.mm.validate_fit import profile_rmse
from ffrefine.stages import hash_array

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _base_args(butane_pdb, butane_itp, outdir) -> list[str]:
    return [
        "-s",
        str(butane_pdb),
        "-p",
        str(butane_itp),
        "--dihedral",
        "1,5,8,11",
        "--outdir",
        str(outdir),
    ]


def _fake_frames(n: int, natoms: int = 14) -> np.ndarray:
    """Deterministic placeholder geometries of the right shape.

    ``fit`` never evaluates these -- it only hashes them to confirm the MM
    energies were computed on the same frames -- so their values are
    arbitrary, but they must be reproducible across the two payloads.
    """
    return np.arange(n * natoms * 3, dtype=float).reshape(n, natoms, 3) * 0.01


def _write_stage_inputs(outdir, phi_deg, qm_kj, mm0_kj=None, quad=(1, 5, 8, 11), frames=None):
    """Hand-build the 01_qm_scan and 02_mm_scan payloads ``fit`` consumes.

    Lets the whole fit path be exercised with no GROMACS and no xtb.  The two
    payloads are written *consistently* -- same quad, same geometries, and the
    matching ``frames_sha256`` -- because that is what ``mmscan`` produces;
    tests that want a mispairing construct it explicitly.
    """
    mm0 = np.zeros_like(np.asarray(qm_kj, dtype=float)) if mm0_kj is None else np.asarray(mm0_kj)
    frames = _fake_frames(len(list(phi_deg))) if frames is None else np.asarray(frames, float)
    qm_dir = outdir / "01_qm_scan"
    qm_dir.mkdir(parents=True, exist_ok=True)
    (qm_dir / "scan.json").write_text(
        json.dumps(
            {
                "phi_deg": list(phi_deg),
                "energy_kj": [float(v) for v in qm_kj],
                "frames_nm": frames.tolist(),
            }
        )
    )
    mm_dir = outdir / "02_mm_scan"
    mm_dir.mkdir(parents=True, exist_ok=True)
    (mm_dir / "energies.json").write_text(
        json.dumps(
            {
                "quad": list(quad),
                "frames_sha256": hash_array(frames),
                "phi_deg": [float(v) for v in phi_deg],
                "zeroed_potential_kj": [float(v) for v in mm0],
                "full_potential_kj": [float(v) for v in mm0],
            }
        )
    )


def _grid(n: int = 24) -> np.ndarray:
    return np.linspace(-180.0, 180.0, n, endpoint=False)


# --------------------------------------------------------------------------
# parse_dihedral -- the 1-based/0-based boundary (spec section 8)
# --------------------------------------------------------------------------


def test_parse_dihedral_one_based():
    assert parse_dihedral("1,5,8,11") == (1, 5, 8, 11)


def test_parse_dihedral_tolerates_spaces():
    assert parse_dihedral(" 1 , 5 , 8 , 11 ") == (1, 5, 8, 11)


def test_parse_dihedral_rejects_wrong_count():
    with pytest.raises(ValidationError):
        parse_dihedral("1,5,8")


def test_parse_dihedral_rejects_zero_index():
    """Indices are 1-based; 0 is a common off-by-one mistake."""
    with pytest.raises(ValidationError):
        parse_dihedral("0,5,8,11")


def test_parse_dihedral_rejects_non_integer():
    with pytest.raises(ValidationError):
        parse_dihedral("1,5,8,x")


def test_bad_dihedral_on_the_command_line_exits_cleanly(capsys):
    """Bad user input gets argparse's message and exit 2, never a traceback."""
    with pytest.raises(SystemExit) as exc:
        main(["mmscan", "-s", "a.pdb", "-p", "a.itp", "--dihedral", "0,5,8,11"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--dihedral" in err
    assert "1-based" in err


# --------------------------------------------------------------------------
# parser surface (spec section 3)
# --------------------------------------------------------------------------


def test_parser_has_all_six_subcommands():
    parser = build_parser()
    actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
    names = set()
    for a in actions:
        names |= set(a.choices)
    assert {"scan", "mmscan", "fit", "charges", "validate", "run"} <= names


def test_common_flags_present_on_mmscan():
    parser = build_parser()
    args = parser.parse_args(
        [
            "mmscan",
            "-s",
            "a.pdb",
            "-p",
            "a.itp",
            "--dihedral",
            "1,5,8,11",
            "--outdir",
            "out",
            "--nthreads",
            "4",
            "--resume",
            "--dry-run",
        ]
    )
    assert args.outdir == "out"
    assert args.nthreads == 4
    assert args.resume is True
    assert args.dry_run is True


def test_common_flags_present_on_every_subcommand():
    """Spec section 3: every subcommand takes the common flags."""
    parser = build_parser()
    for name in ("scan", "mmscan", "fit", "charges", "validate", "run"):
        argv = [name, "-s", "a.pdb", "-p", "a.itp", "--outdir", "o", "--nthreads", "2"]
        argv += ["--resume", "--dry-run", "--config", "c.json", "--gmx-bin", "g"]
        argv += ["--xtb-bin", "x", "--orca-bin", "o2"]
        args = parser.parse_args(argv)
        assert args.command == name
        assert args.outdir == "o"


def test_dihedral_flag_is_repeatable():
    parser = build_parser()
    args = parser.parse_args(
        ["mmscan", "-s", "a.pdb", "-p", "a.itp", "--dihedral", "1,5,8,11", "--dihedral", "2,5,8,11"]
    )
    assert len(args.dihedral) == 2


def test_multiplicities_flag_parses():
    parser = build_parser()
    args = parser.parse_args(
        [
            "fit",
            "-s",
            "a.pdb",
            "-p",
            "a.itp",
            "--dihedral",
            "1,5,8,11",
            "--multiplicities",
            "1,2,3",
        ]
    )
    assert args.multiplicities == [1, 2, 3]


def test_run_accepts_the_fit_flags():
    """``run`` chains fit, so it must accept the flags fit needs."""
    parser = build_parser()
    args = parser.parse_args(
        ["run", "-s", "a.pdb", "-p", "a.itp", "--dihedral", "1,5,8,11", "--free-phase"]
    )
    assert args.free_phase is True


# --------------------------------------------------------------------------
# out-of-scope subcommands (milestones 4 and 5)
# --------------------------------------------------------------------------


def test_scan_reports_not_implemented(capsys):
    code = main(["scan", "-s", "a.pdb", "-p", "a.itp", "--dihedral", "1,5,8,11"])
    assert code != 0
    assert "milestone 4" in capsys.readouterr().err.lower()


def test_charges_reports_not_implemented(capsys):
    code = main(["charges", "-s", "a.pdb", "-p", "a.itp"])
    assert code != 0
    assert "milestone 5" in capsys.readouterr().err.lower()


def test_unimplemented_subcommands_write_nothing(tmp_path, butane_pdb, butane_itp):
    """They must not be mistakable for working: no outdir, no files, exit != 0."""
    out = tmp_path / "out"
    # ``charges`` takes no --dihedral (spec section 5 selects atoms, not a torsion).
    for argv in (
        ["scan", "--dihedral", "1,5,8,11"],
        ["charges"],
    ):
        code = main(argv + ["-s", str(butane_pdb), "-p", str(butane_itp), "--outdir", str(out)])
        assert code != 0
        assert not out.exists()


# --------------------------------------------------------------------------
# --dry-run and input validation
# --------------------------------------------------------------------------


def test_dry_run_does_not_create_output(tmp_path, butane_pdb, butane_itp):
    out = tmp_path / "out"
    code = main(
        [
            "mmscan",
            "-s",
            str(butane_pdb),
            "-p",
            str(butane_itp),
            "--dihedral",
            "1,5,8,11",
            "--outdir",
            str(out),
            "--dry-run",
        ]
    )
    assert code == 0
    assert not (out / "02_mm_scan" / "full.top").exists()
    assert not out.exists()


def test_missing_dihedral_is_a_clean_error(tmp_path, butane_pdb, butane_itp, capsys):
    code = main(["mmscan", "-s", str(butane_pdb), "-p", str(butane_itp), "--outdir", str(tmp_path)])
    assert code == 1
    assert "--dihedral" in capsys.readouterr().err


def test_unbonded_central_pair_is_a_clean_error(tmp_path, butane_pdb, butane_itp, capsys):
    """A quad whose central pair is not a bond fails with a message, not a trace."""
    code = main(
        ["mmscan", "-s", str(butane_pdb), "-p", str(butane_itp)]
        + ["--dihedral", "1,2,3,4", "--outdir", str(tmp_path / "o"), "--dry-run"]
    )
    assert code == 1
    assert "not bonded" in capsys.readouterr().err


def test_multiple_dihedrals_is_a_clean_error(tmp_path, butane_pdb, butane_itp, capsys):
    """One torsion per run: spec section 6's layout holds one scan and one fit."""
    code = main(
        ["mmscan", "-s", str(butane_pdb), "-p", str(butane_itp)]
        + ["--dihedral", "1,5,8,11", "--dihedral", "2,5,8,11"]
        + ["--outdir", str(tmp_path / "o"), "--dry-run"]
    )
    assert code == 1
    assert "one --dihedral" in capsys.readouterr().err


# --------------------------------------------------------------------------
# missing/unreadable file paths (ruling R29): the commonest user error must
# never surface a raw OSError traceback, on any path the user can name --
# -s, -p or --config.
# --------------------------------------------------------------------------


def test_missing_structure_path_is_a_clean_error(tmp_path, butane_itp, capsys):
    missing = tmp_path / "does_not_exist.gro"
    code = main(
        ["mmscan", "-s", str(missing), "-p", str(butane_itp)]
        + ["--dihedral", "1,5,8,11", "--outdir", str(tmp_path / "o"), "--dry-run"]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert str(missing) in captured.err
    assert captured.err.count("\n") == 1  # exactly one clean line, no traceback
    assert "Traceback" not in captured.err
    assert "FileNotFoundError" not in captured.err


def test_missing_topology_path_is_a_clean_error(tmp_path, butane_pdb, capsys):
    missing = tmp_path / "does_not_exist.itp"
    code = main(
        ["mmscan", "-s", str(butane_pdb), "-p", str(missing)]
        + ["--dihedral", "1,5,8,11", "--outdir", str(tmp_path / "o"), "--dry-run"]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert str(missing) in captured.err
    assert captured.err.count("\n") == 1
    assert "Traceback" not in captured.err
    assert "FileNotFoundError" not in captured.err


def test_missing_config_path_is_a_clean_error(tmp_path, butane_pdb, butane_itp, capsys):
    missing = tmp_path / "does_not_exist.json"
    code = main(
        ["mmscan", "-s", str(butane_pdb), "-p", str(butane_itp)]
        + ["--dihedral", "1,5,8,11", "--config", str(missing)]
        + ["--outdir", str(tmp_path / "o"), "--dry-run"]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert str(missing) in captured.err
    assert captured.err.count("\n") == 1
    assert "Traceback" not in captured.err
    assert "FileNotFoundError" not in captured.err


def test_directory_given_as_topology_is_a_clean_error(tmp_path, butane_pdb, capsys):
    """A directory where a file is expected is the same class of error as missing."""
    a_directory = tmp_path / "a_directory.itp"
    a_directory.mkdir()
    code = main(
        ["mmscan", "-s", str(butane_pdb), "-p", str(a_directory)]
        + ["--dihedral", "1,5,8,11", "--outdir", str(tmp_path / "o"), "--dry-run"]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert str(a_directory) in captured.err
    assert captured.err.count("\n") == 1
    assert "Traceback" not in captured.err
    assert "IsADirectoryError" not in captured.err


# --------------------------------------------------------------------------
# --config (spec section 3: CLI overrides file values)
# --------------------------------------------------------------------------


def test_config_file_supplies_outdir(tmp_path, butane_pdb, butane_itp, capsys):
    cfg = tmp_path / "cfg.json"
    wanted = tmp_path / "from_config"
    cfg.write_text(json.dumps({"outdir": str(wanted), "nthreads": 3}))
    code = main(
        ["mmscan", "-s", str(butane_pdb), "-p", str(butane_itp)]
        + ["--dihedral", "1,5,8,11", "--config", str(cfg), "--dry-run"]
    )
    assert code == 0
    assert str(wanted) in capsys.readouterr().out


def test_cli_overrides_config(tmp_path, butane_pdb, butane_itp, capsys):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"outdir": str(tmp_path / "from_config")}))
    wanted = tmp_path / "from_cli"
    code = main(
        ["mmscan", "-s", str(butane_pdb), "-p", str(butane_itp)]
        + ["--dihedral", "1,5,8,11", "--config", str(cfg), "--outdir", str(wanted), "--dry-run"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert str(wanted) in out
    assert "from_config" not in out


def test_config_multiplicities_are_typed_not_iterated_as_a_string(tmp_path, butane_pdb, butane_itp):
    """A JSON value never went through argparse's type= callable."""
    out = tmp_path / "out"
    phi = _grid()
    qm = 5.0 * (1.0 + np.cos(np.radians(3.0 * phi)))
    _write_stage_inputs(out, phi, qm)
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"multiplicities": "2,3"}))
    argv = ["fit"] + _base_args(butane_pdb, butane_itp, out) + ["--config", str(cfg)]
    assert main(argv) == 0
    payload = json.loads((out / "03_fit" / "fit.json").read_text())
    assert payload["multiplicities"] == [3]


def test_unusable_config_value_is_a_clean_error(tmp_path, butane_pdb, butane_itp, capsys):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"nthreads": "many"}))
    code = main(
        ["mmscan", "-s", str(butane_pdb), "-p", str(butane_itp)]
        + ["--dihedral", "1,5,8,11", "--config", str(cfg), "--dry-run"]
    )
    assert code == 1
    assert "nthreads" in capsys.readouterr().err


def test_boltzmann_weighting_runs(tmp_path, butane_pdb, butane_itp):
    """--boltzmann with no argument takes the 300 K default."""
    out = tmp_path / "out"
    phi = _grid()
    qm = 5.0 * (1.0 + np.cos(np.radians(3.0 * phi)))
    _write_stage_inputs(out, phi, qm)
    argv = ["fit"] + _base_args(butane_pdb, butane_itp, out) + ["--boltzmann"]
    assert main(argv) == 0
    payload = json.loads((out / "03_fit" / "fit.json").read_text())
    # Boltzmann weights collapse onto the minima, but the profile is an exact
    # 3-fold cosine everywhere, so the recovered amplitude is unchanged.
    assert payload["multiplicities"] == [3]
    assert payload["terms"][0]["k_kj_per_mol"] == pytest.approx(5.0, abs=1e-6)


def test_emax_that_discards_everything_is_a_clean_error(tmp_path, butane_pdb, butane_itp, capsys):
    out = tmp_path / "out"
    phi = _grid()
    qm = 5.0 * (1.0 + np.cos(np.radians(3.0 * phi)))
    _write_stage_inputs(out, phi, qm)
    argv = ["fit"] + _base_args(butane_pdb, butane_itp, out) + ["--emax", "0.0"]
    assert main(argv) == 1
    err = capsys.readouterr().err
    assert "--emax" in err
    assert "underdetermined" in err


def test_too_few_points_for_the_model_is_rejected(tmp_path, butane_pdb, butane_itp, capsys):
    """An underdetermined lstsq returns a perfect-looking, meaningless fit.

    With 5 multiplicities the largest candidate has 6 parameters; 6 points
    would let it pass exactly through every one of them and report RMSE 0.
    """
    out = tmp_path / "out"
    phi = _grid(6)
    qm = 5.0 * (1.0 + np.cos(np.radians(3.0 * phi)))
    _write_stage_inputs(out, phi, qm)
    assert main(["fit"] + _base_args(butane_pdb, butane_itp, out)) == 1
    assert "underdetermined" in capsys.readouterr().err


def test_a_smaller_multiplicity_set_makes_few_points_fittable(tmp_path, butane_pdb, butane_itp):
    """The remedy the error names actually works."""
    out = tmp_path / "out"
    phi = _grid(6)
    qm = 5.0 * (1.0 + np.cos(np.radians(3.0 * phi)))
    _write_stage_inputs(out, phi, qm)
    argv = ["fit"] + _base_args(butane_pdb, butane_itp, out) + ["--multiplicities", "3"]
    assert main(argv) == 0
    payload = json.loads((out / "03_fit" / "fit.json").read_text())
    assert payload["terms"][0]["k_kj_per_mol"] == pytest.approx(5.0, abs=1e-6)


def test_unreadable_config_is_a_clean_error(tmp_path, butane_pdb, butane_itp, capsys):
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{not json")
    code = main(
        ["mmscan", "-s", str(butane_pdb), "-p", str(butane_itp)]
        + ["--dihedral", "1,5,8,11", "--config", str(cfg), "--dry-run"]
    )
    assert code == 1
    assert "config" in capsys.readouterr().err


# --------------------------------------------------------------------------
# fit -- exercised with hand-built stage payloads, no external binaries
# --------------------------------------------------------------------------


def test_fit_recovers_a_known_series(tmp_path, butane_pdb, butane_itp):
    out = tmp_path / "out"
    phi = _grid()
    qm = 5.0 * (1.0 + np.cos(np.radians(3.0 * phi)))
    _write_stage_inputs(out, phi, qm)
    assert main(["fit"] + _base_args(butane_pdb, butane_itp, out)) == 0

    payload = json.loads((out / "03_fit" / "fit.json").read_text())
    assert payload["multiplicities"] == [3]
    assert payload["terms"][0]["k_kj_per_mol"] == pytest.approx(5.0, abs=1e-6)
    assert payload["terms"][0]["phase_deg"] == pytest.approx(0.0, abs=1e-6)
    assert payload["rmse_kj_per_mol"] < 1e-9


def test_fit_writes_every_spec_4_8_output(tmp_path, butane_pdb, butane_itp):
    out = tmp_path / "out"
    phi = _grid()
    qm = 5.0 * (1.0 + np.cos(np.radians(3.0 * phi)))
    _write_stage_inputs(out, phi, qm)
    assert main(["fit"] + _base_args(butane_pdb, butane_itp, out)) == 0

    fit_dir = out / "03_fit"
    for name in ("fit.json", "dihedral.itp", "dihedral.prm"):
        assert (fit_dir / name).is_file(), name
    for name in ("qm_profile.png", "mm_zeroed.png", "fit.png", "residuals.png"):
        assert (fit_dir / "plots" / name).is_file(), name
    assert (out / "run.log").is_file()


def test_fit_warns_about_asymmetry_and_names_free_phase(tmp_path, butane_pdb, butane_itp):
    """Spec 4.7 / ruling R34: detect asymmetry, warn, and offer --free-phase."""
    out = tmp_path / "out"
    phi = _grid()
    qm = 5.0 * (1.0 + np.cos(np.radians(3.0 * phi - 40.0)))
    _write_stage_inputs(out, phi, qm)
    assert main(["fit"] + _base_args(butane_pdb, butane_itp, out)) == 0

    log_text = (out / "run.log").read_text()
    assert "asymmetric" in log_text.lower()
    assert "--free-phase" in log_text


def test_fit_does_not_warn_about_asymmetry_on_a_symmetric_profile(tmp_path, butane_pdb, butane_itp):
    out = tmp_path / "out"
    phi = _grid()
    qm = 5.0 * (1.0 + np.cos(np.radians(3.0 * phi)))
    _write_stage_inputs(out, phi, qm)
    assert main(["fit"] + _base_args(butane_pdb, butane_itp, out)) == 0

    log_text = (out / "run.log").read_text()
    assert "asymmetric" not in log_text.lower()
    assert "--free-phase" not in log_text


def test_fit_free_phase_recovers_the_shifted_phase(tmp_path, butane_pdb, butane_itp):
    out = tmp_path / "out"
    phi = _grid()
    qm = 5.0 * (1.0 + np.cos(np.radians(3.0 * phi - 40.0)))
    _write_stage_inputs(out, phi, qm)
    argv = ["fit"] + _base_args(butane_pdb, butane_itp, out) + ["--free-phase"]
    assert main(argv) == 0

    payload = json.loads((out / "03_fit" / "fit.json").read_text())
    term = next(t for t in payload["terms"] if t["multiplicity"] == 3)
    assert term["k_kj_per_mol"] == pytest.approx(5.0, abs=1e-6)
    assert term["phase_deg"] == pytest.approx(40.0, abs=1e-6)
    assert payload["free_phase"] is True


def test_fit_without_a_scan_is_a_clean_error(tmp_path, butane_pdb, butane_itp, capsys):
    out = tmp_path / "out"
    code = main(["fit"] + _base_args(butane_pdb, butane_itp, out))
    assert code == 1
    err = capsys.readouterr().err
    assert "scan.json" in err
    assert "milestone 4" in err.lower()


def test_fit_without_mm_energies_is_a_clean_error(tmp_path, butane_pdb, butane_itp, capsys):
    out = tmp_path / "out"
    qm_dir = out / "01_qm_scan"
    qm_dir.mkdir(parents=True)
    (qm_dir / "scan.json").write_text(json.dumps({"phi_deg": [0.0], "energy_kj": [0.0]}))
    code = main(["fit"] + _base_args(butane_pdb, butane_itp, out))
    assert code == 1
    assert "mmscan" in capsys.readouterr().err


def test_fit_rejects_mismatched_profile_lengths(tmp_path, butane_pdb, butane_itp, capsys):
    out = tmp_path / "out"
    phi = _grid()
    qm = 5.0 * (1.0 + np.cos(np.radians(3.0 * phi)))
    _write_stage_inputs(out, phi, qm)
    scan_path = out / "01_qm_scan" / "scan.json"
    payload = json.loads(scan_path.read_text())
    payload["energy_kj"] = payload["energy_kj"][:-1]
    scan_path.write_text(json.dumps(payload))
    code = main(["fit"] + _base_args(butane_pdb, butane_itp, out))
    assert code == 1
    assert "same number of points" in capsys.readouterr().err


def test_fit_resume_skips_a_completed_stage(tmp_path, butane_pdb, butane_itp, capsys):
    out = tmp_path / "out"
    phi = _grid()
    qm = 5.0 * (1.0 + np.cos(np.radians(3.0 * phi)))
    _write_stage_inputs(out, phi, qm)
    argv = ["fit"] + _base_args(butane_pdb, butane_itp, out)
    assert main(argv) == 0
    fit_json = out / "03_fit" / "fit.json"
    stamp = fit_json.stat().st_mtime_ns
    fit_json.write_text('{"sentinel": true}')
    capsys.readouterr()

    assert main(argv + ["--resume"]) == 0
    assert json.loads(fit_json.read_text()) == {"sentinel": True}
    assert "skip" in capsys.readouterr().out.lower()
    assert stamp is not None


def test_fit_resume_reruns_when_inputs_change(tmp_path, butane_pdb, butane_itp):
    out = tmp_path / "out"
    phi = _grid()
    qm = 5.0 * (1.0 + np.cos(np.radians(3.0 * phi)))
    _write_stage_inputs(out, phi, qm)
    argv = ["fit"] + _base_args(butane_pdb, butane_itp, out)
    assert main(argv) == 0

    _write_stage_inputs(out, phi, 2.0 * qm)
    assert main(argv + ["--resume"]) == 0
    payload = json.loads((out / "03_fit" / "fit.json").read_text())
    assert payload["terms"][0]["k_kj_per_mol"] == pytest.approx(10.0, abs=1e-6)


def test_fit_writes_the_00_input_stage(tmp_path, butane_pdb, butane_itp):
    """Spec section 6: normalized input copies plus a parsed model dump."""
    out = tmp_path / "out"
    phi = _grid()
    _write_stage_inputs(out, phi, 5.0 * (1.0 + np.cos(np.radians(3.0 * phi))))
    assert main(["fit"] + _base_args(butane_pdb, butane_itp, out)) == 0

    stage = out / "00_input"
    assert (stage / butane_pdb.name).is_file()
    assert (stage / butane_itp.name).is_file()
    model = json.loads((stage / "model.json").read_text())
    assert model["natoms"] == 14
    # Spec section 9.3: the 1-4 conventions are read, never assumed.
    assert model["fudgeQQ"] == pytest.approx(1.0)
    assert model["fudgeLJ"] == pytest.approx(1.0)
    assert model["torsion_set_size"] == 10
    # Inputs are copied, never modified in place (spec section 1).
    assert (stage / butane_itp.name).read_text() == butane_itp.read_text()


# --------------------------------------------------------------------------
# --no-strict-names must reach the checks that actually run (spec section 4.1)
# --------------------------------------------------------------------------


def _renamed_pdb(butane_pdb, tmp_path):
    """A copy of the fixture with one atom renamed but the same element.

    The PDB atom-name field is columns 13-16 (0-based 12:16), so the
    replacement keeps the column layout ``read_pdb`` depends on.
    """
    lines = []
    for line in butane_pdb.read_text().splitlines():
        if line.startswith("ATOM") and line[12:16] == " H11":
            line = line[:12] + " HX1" + line[16:]
        lines.append(line)
    path = tmp_path / "renamed.pdb"
    path.write_text(chr(10).join(lines) + chr(10))
    return path


def test_atom_name_mismatch_is_fatal_by_default(tmp_path, butane_pdb, butane_itp, capsys):
    renamed = _renamed_pdb(butane_pdb, tmp_path)
    code = main(
        ["mmscan", "-s", str(renamed), "-p", str(butane_itp)]
        + ["--dihedral", "1,5,8,11", "--outdir", str(tmp_path / "o")]
    )
    assert code == 1
    assert "atom name mismatch" in capsys.readouterr().err


@pytest.mark.needs_gmx
def test_no_strict_names_reaches_the_downstream_check(
    tmp_path, butane_pdb, butane_itp, gmx_bin, capsys
):
    """The flag must reach check_torsion_removal, not just _load_inputs.

    ``check_torsion_removal`` re-validates the structure against the topology
    itself (grompp runs with -maxwarn, so its own name check is suppressed).
    Before ``strict_names`` was threaded through, this run still aborted with
    a fatal name mismatch and the flag was decorative.
    """
    renamed = _renamed_pdb(butane_pdb, tmp_path)
    code = main(
        ["mmscan", "-s", str(renamed), "-p", str(butane_itp)]
        + ["--dihedral", "1,5,8,11", "--outdir", str(tmp_path / "o")]
        + ["--no-strict-names", "--gmx-bin", gmx_bin]
    )
    assert code == 0
    assert "atom name mismatch" in capsys.readouterr().err


# --------------------------------------------------------------------------
# validate and run
# --------------------------------------------------------------------------


def test_validate_without_a_fit_is_a_clean_error(tmp_path, butane_pdb, butane_itp, capsys):
    out = tmp_path / "out"
    phi = _grid()
    _write_stage_inputs(out, phi, 5.0 * (1.0 + np.cos(np.radians(3.0 * phi))))
    code = main(["validate"] + _base_args(butane_pdb, butane_itp, out))
    assert code == 1
    assert "fit.json" in capsys.readouterr().err


def test_validate_without_scan_geometries_is_a_clean_error(
    tmp_path, butane_pdb, butane_itp, capsys
):
    out = tmp_path / "out"
    phi = _grid()
    qm = 5.0 * (1.0 + np.cos(np.radians(3.0 * phi)))
    _write_stage_inputs(out, phi, qm)
    assert main(["fit"] + _base_args(butane_pdb, butane_itp, out)) == 0
    capsys.readouterr()
    # Drop the geometries after fitting: validate re-runs MM single points on
    # them (spec 4.9) and cannot proceed without them.
    scan_path = out / "01_qm_scan" / "scan.json"
    scan = json.loads(scan_path.read_text())
    del scan["frames_nm"]
    scan_path.write_text(json.dumps(scan))
    code = main(["validate"] + _base_args(butane_pdb, butane_itp, out))
    assert code == 1
    assert "frames_nm" in capsys.readouterr().err


def test_run_says_which_stages_it_skipped(tmp_path, butane_pdb, butane_itp, capsys):
    out = tmp_path / "out"
    code = main(["run"] + _base_args(butane_pdb, butane_itp, out) + ["--dry-run"])
    assert code == 0
    printed = capsys.readouterr().out.lower()
    assert "scan" in printed and "milestone 4" in printed
    assert "charges" in printed and "milestone 5" in printed


# --------------------------------------------------------------------------
# end-to-end through GROMACS (spec section 7.1 driven from the CLI)
# --------------------------------------------------------------------------


@pytest.mark.needs_gmx
def test_end_to_end_mmscan_fit_validate(tmp_path, butane_pdb, butane_itp, gmx_bin, rotated_frames):
    """mmscan -> fit -> validate on butane, with MM standing in for QM.

    Spec section 7.1's round trip, driven entirely through the CLI: the MM
    energies of the *original* topology are fed back in as the pseudo-QM
    reference, so a correct pipeline must recover the removed torsion and
    report a near-zero "before" RMSE in validation.

    That last assertion is the ruling R33 regression test.  ``validate``
    must hand :func:`ffrefine.mm.validate_fit.validate_refit` the ORIGINAL
    topology as its "before" state.  Here the pseudo-QM reference *is* the
    original topology's energy, so a correct wiring gives rmse_before ~ 0.
    Passing the zeroed topology instead would make rmse_before equal
    ``zeroed_rmse`` below -- the full amplitude of the removed torsion -- and
    inflate the reported improvement with nothing else able to catch it.
    """
    structure = read_pdb(butane_pdb)
    frames = rotated_frames(structure, 24)
    out = tmp_path / "out"
    qm_dir = out / "01_qm_scan"
    qm_dir.mkdir(parents=True)
    scan_path = qm_dir / "scan.json"
    scan_path.write_text(
        json.dumps(
            {
                "phi_deg": [0.0] * len(frames),
                "energy_kj": [0.0] * len(frames),
                "frames_nm": frames.tolist(),
            }
        )
    )
    base = _base_args(butane_pdb, butane_itp, out) + ["--gmx-bin", gmx_bin]

    assert main(["mmscan"] + base) == 0
    energies = json.loads((out / "02_mm_scan" / "energies.json").read_text())
    assert len(energies["phi_deg"]) == len(frames)
    assert energies["selfcheck_max_abs_error_kj"] < 1e-3

    # Spec 7.1: MM with the original parameters stands in for the QM scan.
    scan_path.write_text(
        json.dumps(
            {
                "phi_deg": energies["phi_deg"],
                "energy_kj": energies["full_potential_kj"],
                "frames_nm": frames.tolist(),
            }
        )
    )

    assert main(["fit"] + base) == 0
    fit_payload = json.loads((out / "03_fit" / "fit.json").read_text())
    # Butane's torsion set about the central C2-C3 bond is 3-fold.
    assert fit_payload["multiplicities"] == [3]
    # Not ~1e-5: spec 4.7's model selection deliberately buys parsimony with up
    # to 0.5 kJ/mol of RMSE, and the 2-parameter n=3 model is what it picks
    # here.  Measured 0.2631; the bound is the selection tolerance itself.
    assert fit_payload["rmse_kj_per_mol"] < 0.5

    assert main(["validate"] + base) == 0
    result = json.loads((out / "final" / "validation.json").read_text())

    qm = np.array(energies["full_potential_kj"], dtype=float)
    zeroed_rmse = profile_rmse(qm, np.array(energies["zeroed_potential_kj"], dtype=float))
    # Measured 7.586 kJ/mol -- the full amplitude of the removed torsion, and
    # exactly what rmse_before would report if the zeroed topology had been
    # passed as the "before" state.
    assert zeroed_rmse > 1.0, "the removed torsion must actually matter for this to test anything"
    # Ruling R33.  "Before" is the original topology, which generated the
    # pseudo-QM reference, so the two profiles are the same numbers: measured
    # exactly 0.0.  The bound carries headroom for GROMACS run-to-run noise
    # while staying ~1e4 below the zeroed alternative.
    assert result["rmse_before_kj_per_mol"] < 1e-3
    assert result["rmse_before_kj_per_mol"] < 0.01 * zeroed_rmse
    # Measured 0.4307 -- the same parsimony cost as the fit RMSE above,
    # ~18x below the un-refitted profile's error.
    assert result["rmse_after_kj_per_mol"] < 0.2 * zeroed_rmse
    assert (out / "final" / "molecule_refined.itp").is_file()
    assert (out / "final" / "plots" / "validation.png").is_file()


# --------------------------------------------------------------------------
# stage pairing (final review, item 3)
# --------------------------------------------------------------------------


def test_fit_refuses_energies_computed_for_another_torsion(
    tmp_path, butane_pdb, butane_itp, capsys
):
    """`fit --dihedral 2,1,5,8` after `mmscan --dihedral 1,5,8,11` used to fit
    torsion A's data and write the parameters onto torsion B, exit 0."""
    out = tmp_path / "out"
    phi = _grid()
    _write_stage_inputs(out, phi, 5.0 * (1.0 + np.cos(np.radians(3.0 * phi))))
    argv = [
        "fit",
        "-s",
        str(butane_pdb),
        "-p",
        str(butane_itp),
        "--dihedral",
        "2,1,5,8",
        "--outdir",
        str(out),
    ]
    assert main(argv) == 1
    err = capsys.readouterr().err
    assert "1,5,8,11" in err and "2,1,5,8" in err
    assert not (out / "03_fit" / "fit.json").exists()


def test_fit_refuses_when_the_scan_geometries_changed(tmp_path, butane_pdb, butane_itp, capsys):
    """Regenerating scan.json's geometries without re-running mmscan pairs the
    new QM energies with stale angles and a stale E_MM0."""
    out = tmp_path / "out"
    phi = _grid()
    qm = 5.0 * (1.0 + np.cos(np.radians(3.0 * phi)))
    _write_stage_inputs(out, phi, qm)
    scan_path = out / "01_qm_scan" / "scan.json"
    scan = json.loads(scan_path.read_text())
    moved = np.asarray(scan["frames_nm"], dtype=float)
    moved[0, 0, 0] += 0.05  # one atom, one frame
    scan["frames_nm"] = moved.tolist()
    scan_path.write_text(json.dumps(scan))

    assert main(["fit"] + _base_args(butane_pdb, butane_itp, out)) == 1
    err = capsys.readouterr().err
    assert "different geometries" in err
    assert "mmscan" in err


def test_fit_allows_replacing_only_the_qm_energies(tmp_path, butane_pdb, butane_itp):
    """The documented workflow, and spec 7.1's round trip: swap the QM column
    while keeping the geometries. Only frames_nm is hashed, so this is legal."""
    out = tmp_path / "out"
    phi = _grid()
    _write_stage_inputs(out, phi, np.zeros_like(phi))
    scan_path = out / "01_qm_scan" / "scan.json"
    scan = json.loads(scan_path.read_text())
    scan["energy_kj"] = list(5.0 * (1.0 + np.cos(np.radians(3.0 * phi))))
    scan_path.write_text(json.dumps(scan))
    assert main(["fit"] + _base_args(butane_pdb, butane_itp, out)) == 0


def test_fit_refuses_an_energies_file_that_predates_the_check(
    tmp_path, butane_pdb, butane_itp, capsys
):
    """A payload with no quad or no frames digest cannot be verified, so it is
    refused rather than trusted -- mmscan is cheap."""
    out = tmp_path / "out"
    phi = _grid()
    _write_stage_inputs(out, phi, 5.0 * (1.0 + np.cos(np.radians(3.0 * phi))))
    energies_path = out / "02_mm_scan" / "energies.json"
    payload = json.loads(energies_path.read_text())
    del payload["frames_sha256"]
    energies_path.write_text(json.dumps(payload))
    assert main(["fit"] + _base_args(butane_pdb, butane_itp, out)) == 1
    assert "frames_sha256" in capsys.readouterr().err


def test_fit_records_its_provenance(tmp_path, butane_pdb, butane_itp):
    out = tmp_path / "out"
    phi = _grid()
    _write_stage_inputs(out, phi, 5.0 * (1.0 + np.cos(np.radians(3.0 * phi))))
    assert main(["fit"] + _base_args(butane_pdb, butane_itp, out)) == 0
    payload = json.loads((out / "03_fit" / "fit.json").read_text())
    prov = payload["provenance"]
    assert prov["quad_1_based"] == [1, 5, 8, 11]
    assert prov["frames_sha256"] == hash_array(_fake_frames(len(phi)))


def test_validate_refuses_a_fit_made_for_another_torsion(tmp_path, butane_pdb, butane_itp, capsys):
    """cmd_validate applies fit.json's k/delta/mult to whatever --dihedral it
    is given; fit.json now says which torsion it was made for."""
    out = tmp_path / "out"
    phi = _grid()
    _write_stage_inputs(out, phi, 5.0 * (1.0 + np.cos(np.radians(3.0 * phi))))
    assert main(["fit"] + _base_args(butane_pdb, butane_itp, out)) == 0
    capsys.readouterr()
    argv = [
        "validate",
        "-s",
        str(butane_pdb),
        "-p",
        str(butane_itp),
        "--dihedral",
        "2,1,5,8",
        "--outdir",
        str(out),
    ]
    assert main(argv) == 1
    err = capsys.readouterr().err
    assert "1,5,8,11" in err and "2,1,5,8" in err
    assert not (out / "final" / "validation.json").exists()


def test_validate_refuses_a_fit_made_on_other_geometries(tmp_path, butane_pdb, butane_itp, capsys):
    out = tmp_path / "out"
    phi = _grid()
    _write_stage_inputs(out, phi, 5.0 * (1.0 + np.cos(np.radians(3.0 * phi))))
    assert main(["fit"] + _base_args(butane_pdb, butane_itp, out)) == 0
    capsys.readouterr()
    scan_path = out / "01_qm_scan" / "scan.json"
    scan = json.loads(scan_path.read_text())
    moved = np.asarray(scan["frames_nm"], dtype=float)
    moved[2, 3, 1] -= 0.07
    scan["frames_nm"] = moved.tolist()
    scan_path.write_text(json.dumps(scan))
    assert main(["validate"] + _base_args(butane_pdb, butane_itp, out)) == 1
    assert "different geometries" in capsys.readouterr().err


# --------------------------------------------------------------------------
# .xyz input (final review, item 5)
# --------------------------------------------------------------------------


def test_xyz_input_is_usable_by_default(tmp_path, butane_xyz, butane_itp, capsys):
    """Item 5: --structure a .xyz used to be fatal without --no-strict-names,
    because the reader's synthetic 'H2' can never match the topology's 'H11'."""
    argv = [
        "mmscan",
        "-s",
        str(butane_xyz),
        "-p",
        str(butane_itp),
        "--dihedral",
        "1,5,8,11",
        "--outdir",
        str(tmp_path / "o"),
        "--dry-run",
    ]
    assert main(argv) == 0
    captured = capsys.readouterr()
    assert "atom name mismatch" not in captured.err
    assert "carries no atom names" in captured.err


@pytest.mark.needs_gmx
def test_xyz_input_runs_mmscan_and_matches_the_pdb(
    tmp_path, butane_xyz, butane_pdb, butane_itp, gmx_bin
):
    """The .xyz path must reach GROMACS and give the same answer as the .pdb.

    Naming is adopted from the topology, so the generated coordinate file and
    rerun trajectory carry names grompp recognises.
    """
    from_xyz = tmp_path / "xyz_out"
    from_pdb = tmp_path / "pdb_out"
    for source, outdir in ((butane_xyz, from_xyz), (butane_pdb, from_pdb)):
        argv = [
            "mmscan",
            "-s",
            str(source),
            "-p",
            str(butane_itp),
            "--dihedral",
            "1,5,8,11",
            "--outdir",
            str(outdir),
            "--gmx-bin",
            gmx_bin,
        ]
        assert main(argv) == 0
    a = json.loads((from_xyz / "02_mm_scan" / "energies.json").read_text())
    b = json.loads((from_pdb / "02_mm_scan" / "energies.json").read_text())
    assert a["full_potential_kj"] == pytest.approx(b["full_potential_kj"], abs=1e-4)
    assert a["selfcheck_max_abs_error_kj"] < 1e-3
