import json

import numpy as np
import pytest

from ffrefine.cli import build_parser, main
from ffrefine.io.topology import read_topology


def _args(butane_pdb, butane_itp, outdir, *extra):
    return [
        "charges",
        "-s",
        str(butane_pdb),
        "-p",
        str(butane_itp),
        "--outdir",
        str(outdir),
        *extra,
    ]


def test_atom_list_parses_one_based_indices():
    from ffrefine.cli import _atom_list

    assert _atom_list("1,5,8") == (1, 5, 8)
    assert _atom_list(" 2 , 3 ") == (2, 3)


def test_atom_list_rejects_zero_and_negatives():
    from ffrefine.cli import _atom_list
    from ffrefine.errors import ValidationError

    with pytest.raises(ValidationError, match="1-based"):
        _atom_list("0,3")


def test_bad_charge_atoms_on_the_command_line_exits_cleanly(capsys):
    """Bad user input gets argparse's message and exit 2, never a traceback.

    ``--charge-atoms`` is wired as ``type=_atom_list_arg`` precisely so this
    never reaches ``main``'s ``try/except FFRefineError`` as a raw
    ``ValidationError``: argparse's own ``type=`` machinery runs during
    ``parse_args()``, before that except block exists, and only converts
    ``ValueError``/``TypeError``/``ArgumentTypeError`` into a clean usage
    error. ``test_atom_list_rejects_zero_and_negatives`` calls ``_atom_list``
    directly and so could never have caught a missing adapter here.
    """
    with pytest.raises(SystemExit) as exc:
        main(["charges", "-s", "a.pdb", "-p", "a.itp", "--charge-atoms", "0,3"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--charge-atoms" in err
    assert "1-based" in err
    assert "Traceback" not in err


def test_non_integer_charge_atoms_on_the_command_line_exits_cleanly(capsys):
    """A non-integer field is the same class of bad input, not a traceback either."""
    with pytest.raises(SystemExit) as exc:
        main(["charges", "-s", "a.pdb", "-p", "a.itp", "--charge-atoms", "x,3"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--charge-atoms" in err
    assert "not an integer" in err
    assert "Traceback" not in err


def test_charges_defaults_to_selected_and_chelpg():
    args = build_parser().parse_args(["charges", "-s", "a.pdb", "-p", "a.itp"])

    assert args.charge_scheme == "chelpg"
    assert args.charge_mode == "selected"
    assert args.charge_geom == "xtb"
    assert args.charge_atoms is None


def test_selected_mode_without_atoms_is_refused(butane_pdb, butane_itp, tmp_path, capsys):
    """Nothing is replaced silently, and the message names both ways forward."""
    rc = main(_args(butane_pdb, butane_itp, tmp_path / "out"))

    assert rc == 1
    err = capsys.readouterr().err
    assert "--charge-atoms" in err
    assert "--charge-mode all" in err
    assert "Traceback" not in err


def test_charges_dry_run_writes_nothing(butane_pdb, butane_itp, tmp_path):
    outdir = tmp_path / "out"
    rc = main(_args(butane_pdb, butane_itp, outdir, "--charge-mode", "all", "--dry-run"))

    assert rc == 0
    assert not outdir.exists()


def test_charges_needs_no_dihedral(butane_pdb, butane_itp, tmp_path, capsys):
    """Task 2's split exists for this: charges touches no torsion."""
    rc = main(_args(butane_pdb, butane_itp, tmp_path / "out", "--charge-mode", "all", "--dry-run"))

    assert rc == 0
    assert "--dihedral" not in capsys.readouterr().err


@pytest.mark.needs_orca
def test_chelpg_end_to_end_writes_a_usable_topology(
    butane_pdb, butane_itp, tmp_path, orca_bin, xtb_bin
):
    """Cheap keywords: the default level is hybrid DFT on a real molecule."""
    outdir = tmp_path / "out"
    rc = main(
        _args(
            butane_pdb,
            butane_itp,
            outdir,
            "--charge-mode",
            "all",
            "--charge-geom",
            "input",
            "--orca-esp-keywords",
            "! HF STO-3G",
            "--nthreads",
            "1",
        )
    )
    assert rc == 0

    payload = json.loads((outdir / "04_charges" / "charges.json").read_text())
    assert len(payload["atoms"]) == 14
    assert payload["total_after"] == pytest.approx(0.0, abs=1e-6)
    assert payload["rrms"] > 0.0
    assert payload["provenance"]["scheme"] == "chelpg"

    charged = read_topology(outdir / "04_charges" / "molecule_charged.itp")
    assert sum(a.charge for a in charged.atoms) == pytest.approx(0.0, abs=1e-6)
    # Equivalent hydrogens were averaged, so a methyl's three must be identical.
    assert charged.atoms[1].charge == charged.atoms[2].charge == charged.atoms[3].charge


@pytest.mark.needs_orca
def test_selected_mode_leaves_unselected_atoms_alone(butane_pdb, butane_itp, tmp_path, orca_bin):
    outdir = tmp_path / "out"
    assert (
        main(
            _args(
                butane_pdb,
                butane_itp,
                outdir,
                "--charge-atoms",
                "1",
                "--charge-geom",
                "input",
                "--orca-esp-keywords",
                "! HF STO-3G",
                "--nthreads",
                "1",
            )
        )
        == 0
    )

    charged = read_topology(outdir / "04_charges" / "molecule_charged.itp")
    original = read_topology(butane_itp)
    assert charged.atoms[0].charge != original.atoms[0].charge
    assert charged.atoms[7].charge == original.atoms[7].charge  # C3, far from the change
    assert sum(a.charge for a in charged.atoms) == pytest.approx(0.0, abs=1e-6)


def test_run_skips_charges_when_none_was_asked_for(butane_pdb, butane_itp, tmp_path, capsys):
    """Plain `ffrefine run` must keep working exactly as it did."""
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
    assert "=== charges ===" not in out
    assert "charges" in out.lower()  # it still says it skipped them


def test_run_chains_charges_first_when_asked(butane_pdb, butane_itp, tmp_path, capsys):
    """Charges must come before mmscan, not after the fit."""
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
            "--charge-mode",
            "all",
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert out.index("=== charges ===") < out.index("=== mmscan ===")
    assert out.index("=== charges ===") < out.index("=== scan ===")


def test_run_points_later_stages_at_the_charged_topology(
    butane_pdb, butane_itp, tmp_path, monkeypatch
):
    """The swap is the whole mechanism; assert it rather than trusting it."""
    import ffrefine.cli as cli

    outdir = tmp_path / "out"
    charged = outdir / "04_charges" / "molecule_charged.itp"
    charged.parent.mkdir(parents=True)
    charged.write_text(butane_itp.read_text())
    seen = []

    monkeypatch.setattr(cli, "cmd_charges", lambda a: 0)
    monkeypatch.setattr(cli, "cmd_scan", lambda a: seen.append(a.topology) or 0)
    monkeypatch.setattr(cli, "cmd_mmscan", lambda a: seen.append(a.topology) or 0)
    monkeypatch.setattr(cli, "cmd_fit", lambda a: seen.append(a.topology) or 0)
    monkeypatch.setattr(cli, "cmd_validate", lambda a: seen.append(a.topology) or 0)

    args = cli.build_parser().parse_args(
        [
            "run",
            "-s",
            str(butane_pdb),
            "-p",
            str(butane_itp),
            "--dihedral",
            "1,5,8,11",
            "--outdir",
            str(outdir),
            "--charge-mode",
            "all",
        ]
    )
    assert cli.cmd_run(args) == 0
    assert seen == [str(charged)] * 4


@pytest.mark.needs_orca
def test_resp_end_to_end_gives_equal_methyl_hydrogens(butane_pdb, butane_itp, tmp_path, orca_bin):
    """Stage 2 constrains the group exactly equal, not merely close."""
    outdir = tmp_path / "out"
    rc = main(
        _args(
            butane_pdb,
            butane_itp,
            outdir,
            "--charge-scheme",
            "resp",
            "--charge-mode",
            "all",
            "--charge-geom",
            "input",
            "--orca-esp-keywords",
            "! HF STO-3G",
            "--esp-density",
            "1.0",
            "--nthreads",
            "1",
        )
    )
    assert rc == 0

    payload = json.loads((outdir / "04_charges" / "charges.json").read_text())
    assert payload["provenance"]["scheme"] == "resp"
    assert payload["provenance"]["restraint"]["a_stage1"] == pytest.approx(0.0005)
    assert payload["provenance"]["grid"]["npoints"] > 500
    assert payload["total_after"] == pytest.approx(0.0, abs=1e-6)

    charged = read_topology(outdir / "04_charges" / "molecule_charged.itp")
    assert charged.atoms[1].charge == charged.atoms[2].charge == charged.atoms[3].charge


@pytest.mark.needs_orca
def test_resp_and_chelpg_broadly_agree_on_butane(butane_pdb, butane_itp, tmp_path, orca_bin):
    """Two schemes on one molecule should not disagree about its chemistry."""
    charges = {}
    for scheme in ("chelpg", "resp"):
        outdir = tmp_path / scheme
        assert (
            main(
                _args(
                    butane_pdb,
                    butane_itp,
                    outdir,
                    "--charge-scheme",
                    scheme,
                    "--charge-mode",
                    "all",
                    "--charge-geom",
                    "input",
                    "--orca-esp-keywords",
                    "! HF STO-3G",
                    "--nthreads",
                    "1",
                )
            )
            == 0
        )
        payload = json.loads((outdir / "04_charges" / "charges.json").read_text())
        charges[scheme] = np.array([a["charge_after"] for a in payload["atoms"]])

    # RESP restrains the buried carbons, so the two differ; they must still
    # agree on the sign of every terminal carbon and be within 0.2 e overall.
    assert np.abs(charges["chelpg"] - charges["resp"]).max() < 0.2
    assert np.sign(charges["chelpg"][0]) == np.sign(charges["resp"][0])
