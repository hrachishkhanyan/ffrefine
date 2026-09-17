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


def test_charge_scheme_resp_is_refused_for_now(butane_pdb, butane_itp, tmp_path, capsys):
    """RESP isn't implemented yet; nothing may silently mislabel a CHELPG fit as RESP."""
    rc = main(
        _args(
            butane_pdb,
            butane_itp,
            tmp_path / "out",
            "--charge-scheme",
            "resp",
            "--charge-mode",
            "all",
        )
    )

    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert "chelpg" in err.lower()
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
