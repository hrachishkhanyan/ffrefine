import json

import numpy as np
import pytest

from ffrefine.cli import build_parser, main
from ffrefine.io.topology import read_topology
from ffrefine.io.validate import element_of


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
    assert payload["rrms_written"] > 0.0
    assert payload["rrms_fitted"] > 0.0
    assert payload["rrms_before"] > 0.0
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

    # Selecting one atom writes a topology that is mostly the input's charges,
    # so its RRMS is not the fitted set's (ruling R20).
    payload = json.loads((outdir / "04_charges" / "charges.json").read_text())
    assert payload["rrms_written"] > payload["rrms_fitted"]


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
def test_resp_shrinks_heavy_atom_charges_relative_to_chelpg(
    butane_pdb, butane_itp, tmp_path, orca_bin
):
    """RESP's restraint only ever pulls a heavy atom's charge toward zero, so
    an *absolute*-agreement tolerance between the two schemes is the wrong
    thing to assert: on a badly-conditioned fit -- which is exactly what the
    cheap ``! HF STO-3G`` keywords below give, RRMS 0.6-0.8 for both schemes
    -- CHELPG's unrestrained fit hands buried carbons "large, arbitrary
    charges that happen to cancel" (``charges/resp.py``'s own words for why
    the restraint exists), and pulling those back toward zero can move a
    heavy atom by several tenths of an electron. That used to be asserted as
    a bug (max diff < 0.2 e); it is RESP working as designed. What must still
    hold is the direction (never away from zero) and the chemistry (no sign
    flip on a terminal carbon), so that is what this checks instead.

    The plain ``|RESP| <= |CHELPG| + 1e-3`` bound alone cannot tell a working
    restraint from a missing one: the MK grid differs slightly from ORCA's own
    CHELPG grid, and that grid difference *alone* already shrinks every carbon
    a little (measured directly, restraint off: C1 0.367 < 0.395, C5 0.078 <
    0.082, C8 0.069 < 0.090, C11 0.351 < 0.399 -- all satisfy the ``<=`` bound
    with the restraint disabled). A lower bound closes that gap. On the two
    terminal carbons, the restrained fit measures ~0.39 and ~0.35 of the
    CHELPG magnitude; an unrestrained fit on the same MK grid measures ~0.93
    and ~0.88. ``0.6`` sits with margin on both sides of that gap and would
    fail if the restraint were silently disabled -- verified directly by
    temporarily zeroing ``RESP_A_STAGE1``/``RESP_A_STAGE2`` and re-running
    this test (ffrefine task-14 report, round 2). This is calibrated for
    HF/STO-3G butane specifically -- the only level and molecule this test
    uses -- not a general property of RESP vs CHELPG.

    The band is two-sided (``0.25 <= ratio <= 0.6``). An upper bound alone
    passed a ``cmd_charges`` that dropped ``nm_to_bohr`` from the RESP
    wiring: every ``1/r`` then comes out about 18.9 times too large, the fit
    shrinks every charge to match, and the terminal carbons land at ~0.05 of
    CHELPG -- "shrunk", so the old one-sided check was satisfied (final
    review, ruling R23).
    """
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
        assert payload["total_after"] == pytest.approx(0.0, abs=1e-6)

    topology = read_topology(butane_itp)
    heavy = [i for i, atom in enumerate(topology.atoms) if element_of(topology, atom.nr) != "H"]

    # The restraint only ever pulls toward zero; 1e-3 is slack for the fit's
    # own numerical tolerance, not room for it to grow a charge.
    assert np.all(np.abs(charges["resp"][heavy]) <= np.abs(charges["chelpg"][heavy]) + 1e-3)
    # atoms 1 and 11 (0-based 0 and 10) are the terminal carbons.
    assert np.sign(charges["chelpg"][0]) == np.sign(charges["resp"][0])
    assert np.sign(charges["chelpg"][10]) == np.sign(charges["resp"][10])
    # The upper bound above cannot distinguish a working restraint from a
    # disabled one (the MK grid alone already shrinks every carbon a little).
    # This two-sided band can: with the restraint on, the terminal carbons
    # land at ~0.39/0.35 of the CHELPG magnitude; with it off, ~0.93/0.88;
    # with the bohr conversion dropped from the RESP wiring, ~0.05. Both
    # edges have margin (see the docstring for the measured ratios).
    for i in (0, 10):
        ratio = np.abs(charges["resp"][i]) / np.abs(charges["chelpg"][i])
        assert 0.25 <= ratio <= 0.6, f"atom {i + 1}: |RESP|/|CHELPG| = {ratio:.3f}"


# CODATA 2018, written out so this file's arithmetic does not lean on
# ffrefine.units, which is part of what the RRMS tests below check.
BOHR_IN_NM = 0.0529177210903


def _write_chelpg_vpot(path, atoms_bohr, grid_bohr, v):
    """Write a ``.vpot`` in the layout a CHELPG run leaves: ``V x y z``, bohr."""
    lines = [f"{len(atoms_bohr):6d} {len(grid_bohr):6d}"]
    lines += [f"   {x: .10E} {y: .10E} {z: .10E}" for x, y, z in atoms_bohr]
    lines += [f"   {p: .10E} {x: .10E} {y: .10E} {z: .10E}" for p, (x, y, z) in zip(v, grid_bohr)]
    path.write_text("\n".join(lines) + "\n")


def _shell_points_bohr(atoms_bohr, n=400, seed=0):
    """Points on shells well outside the molecule, so none sits on an atom."""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    centre = atoms_bohr.mean(axis=0)
    radius = np.linalg.norm(atoms_bohr - centre, axis=1).max() + 4.0
    return centre + directions * radius * rng.uniform(1.0, 1.5, size=(n, 1))


def _inverse_distances_bohr(atoms_bohr, grid_bohr):
    return 1.0 / np.linalg.norm(grid_bohr[:, None, :] - atoms_bohr[None, :, :], axis=-1)


def test_the_rrms_helper_scores_charges_against_a_potential_in_atomic_units(tmp_path, butane_pdb):
    """The helper ``cmd_charges`` uses for every RRMS it reports.

    The potential here is made from known charges, in bohr and Hartree/e, so
    the exact answers are known: the true charges score 0, half of them score
    exactly 0.5, none score 1. Dropping the nm -> bohr conversion inside the
    helper multiplies every ``1/r`` by about 18.9, and the true charges would
    then score about 17.9 (final review, ruling R23, mutant 2).
    """
    from ffrefine.cli import _rrms_on_orca_grid
    from ffrefine.io.structure import read_pdb

    s = read_pdb(butane_pdb)
    atoms_bohr = s.coords / BOHR_IN_NM
    grid_bohr = _shell_points_bohr(atoms_bohr)
    truth = np.array([-0.4, 0.1, 0.1, 0.1, 0.2, -0.05, -0.05] * 2)
    v = _inverse_distances_bohr(atoms_bohr, grid_bohr) @ truth
    _write_chelpg_vpot(tmp_path / "esp.vpot", atoms_bohr, grid_bohr, v)

    scores, npoints = _rrms_on_orca_grid(
        tmp_path / "esp.vpot",
        s.coords,
        {"truth": truth, "half": truth / 2.0, "none": np.zeros(14)},
    )

    assert npoints == len(grid_bohr)
    assert scores["truth"] == pytest.approx(0.0, abs=1e-8)
    assert scores["half"] == pytest.approx(0.5, rel=1e-8)
    assert scores["none"] == pytest.approx(1.0, rel=1e-12)


@pytest.mark.needs_orca
def test_the_rrms_helper_agrees_with_an_independent_fit_on_orcas_own_grid(
    tmp_path, butane_pdb, orca_bin
):
    """A pinned, independently known RRMS from a real ORCA run.

    ORCA's CHELPG charges are the total-charge-constrained least-squares fit
    to the potential on the grid it writes to ``esp.vpot`` (probe,
    2026-09-21: an independent fit reproduces them to 3e-8 e). So the RRMS
    of ORCA's charges is the minimum RRMS on that grid, which this test
    computes from the raw file -- its own atom coordinates, in bohr, with no
    ffrefine parser and no unit conversion -- and requires the helper to
    reproduce from the structure's nm coordinates.
    """
    from ffrefine.cli import _rrms_on_orca_grid
    from ffrefine.config import discover_binaries
    from ffrefine.io.structure import read_pdb
    from ffrefine.qm.orca import parse_chelpg_charges, run_esp_job

    s = read_pdb(butane_pdb)
    run_esp_job(discover_binaries({"orca": orca_bin}), s, s.coords, tmp_path, "! HF STO-3G", 0, 1)
    orca_charges = parse_chelpg_charges(tmp_path / "esp.chelpg.xyz")

    lines = (tmp_path / "esp.vpot").read_text().splitlines()
    natoms, npoints = (int(x) for x in lines[0].split()[:2])
    atoms_bohr = np.array([[float(x) for x in ln.split()[:3]] for ln in lines[1 : 1 + natoms]])
    rows = np.array(
        [[float(x) for x in ln.split()[:4]] for ln in lines[1 + natoms : 1 + natoms + npoints]]
    )
    v, grid_bohr = rows[:, 0], rows[:, 1:4]
    inv_r = _inverse_distances_bohr(atoms_bohr, grid_bohr)
    bordered = np.zeros((natoms + 1, natoms + 1))
    bordered[:natoms, :natoms] = inv_r.T @ inv_r
    bordered[:natoms, natoms] = bordered[natoms, :natoms] = 1.0
    least_squares = np.linalg.solve(bordered, np.append(inv_r.T @ v, 0.0))[:natoms]
    minimum_rrms = float(np.sqrt(((v - inv_r @ least_squares) ** 2).sum() / (v**2).sum()))

    # The premise: ORCA's charges are that fit, to the 8 decimals it writes.
    assert least_squares == pytest.approx(orca_charges, abs=1e-6)

    scores, count = _rrms_on_orca_grid(tmp_path / "esp.vpot", s.coords, {"orca": orca_charges})

    assert count == npoints
    assert scores["orca"] == pytest.approx(minimum_rrms, rel=1e-6)


# ---------------------------------------------------------------------------
# cmd_charges end to end without ORCA
# ---------------------------------------------------------------------------

BUTANE_ESP = np.array(
    [-0.40, 0.09, 0.09, 0.09, 0.20, -0.035, -0.035, 0.20, -0.035, -0.035, -0.40, 0.09, 0.09, 0.09]
)
"""Stand-in "ORCA" charges for butane: sum to zero, equal within each
equivalence group (so averaging leaves them alone), and unlike the fixture's
own charges, so a selected-mode result mixes two genuinely different sets."""


class FakeEsp:
    """What the fake ESP job wrote, and an independent RRMS against it."""

    def __init__(self) -> None:
        self.chelpg = BUTANE_ESP.copy()
        self.calls = 0
        self.atoms_bohr: np.ndarray | None = None
        self.grid_bohr: np.ndarray | None = None
        self.v: np.ndarray | None = None

    def rrms(self, charges) -> float:
        """RRMS from the arrays the fake wrote, in bohr, with no ffrefine code."""
        inv_r = _inverse_distances_bohr(self.atoms_bohr, self.grid_bohr)
        fit = inv_r @ np.asarray(charges, float)
        return float(np.sqrt(((self.v - fit) ** 2).sum() / (self.v**2).sum()))


@pytest.fixture
def fake_esp(monkeypatch):
    """Stand in for the ORCA ESP job, so cmd_charges runs end to end without ORCA.

    Writes an ``esp.chelpg.xyz`` carrying ``FakeEsp.chelpg`` and an
    ``esp.vpot`` whose potential those same charges produce exactly, in the
    layouts ORCA uses. Tests pass ``--charge-geom input`` so xtb is not
    needed either.
    """
    import ffrefine.cli as cli

    fake = FakeEsp()

    def fake_run_esp_job(binaries, structure, coords_nm, workdir, keywords, **kwargs):
        from pathlib import Path

        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        fake.calls += 1
        fake.atoms_bohr = np.asarray(coords_nm, float) / BOHR_IN_NM
        fake.grid_bohr = _shell_points_bohr(fake.atoms_bohr)
        fake.v = _inverse_distances_bohr(fake.atoms_bohr, fake.grid_bohr) @ fake.chelpg
        _write_chelpg_vpot(workdir / "esp.vpot", fake.atoms_bohr, fake.grid_bohr, fake.v)
        lines = [str(len(fake.chelpg)), "# ORCA CHELPG point charges xyz file"]
        for element, (x, y, z), q in zip(structure.elements, coords_nm * 10.0, fake.chelpg):
            lines.append(f"{element:<2s} {x:12.6f} {y:12.6f} {z:12.6f} {q:12.8f}")
        (workdir / "esp.chelpg.xyz").write_text("\n".join(lines) + "\n")
        return workdir / "esp"

    monkeypatch.setattr(cli, "run_esp_job", fake_run_esp_job)
    monkeypatch.setattr(cli, "_orca_version", lambda binaries: "fake ORCA")
    return fake


def _charges_json(outdir):
    return json.loads((outdir / "04_charges" / "charges.json").read_text())


def _summary_line(out: str) -> str:
    return next(line for line in out.splitlines() if line.startswith("chelpg charges:"))


def test_selected_mode_reports_the_rrms_of_the_charges_it_wrote(
    butane_pdb, butane_itp, tmp_path, fake_esp, capsys
):
    """Final review I1 (ruling R20): the headline RRMS must describe the written topology.

    Selecting C1 alone writes a topology whose charges mostly come from the
    input, not from the fit. The fully fitted set reproduces this potential
    exactly (RRMS 0); the written set reproduces it worse than the input
    topology did. The old code printed and recorded the fitted set's 0.0000
    next to that topology, and never warned.
    """
    outdir = tmp_path / "out"
    rc = main(
        _args(butane_pdb, butane_itp, outdir, "--charge-atoms", "1", "--charge-geom", "input")
    )
    assert rc == 0
    captured = capsys.readouterr()

    payload = _charges_json(outdir)
    written = [a["charge_after"] for a in payload["atoms"]]
    before = [a["charge_before"] for a in payload["atoms"]]

    assert payload["rrms_fitted"] == pytest.approx(0.0, abs=1e-6)
    assert payload["rrms_written"] == pytest.approx(fake_esp.rrms(written), rel=1e-6)
    assert payload["rrms_before"] == pytest.approx(fake_esp.rrms(before), rel=1e-6)
    # The case the review measured on real butane (0.65 / 1.70 / 1.33): the
    # written topology fits worse than the one it replaced.
    assert payload["rrms_written"] > payload["rrms_before"] > 0.5
    assert f"RRMS {payload['rrms_written']:.4f}" in _summary_line(captured.out)
    assert "poorly" in captured.err
    assert f"{payload['rrms_written']:.3f}" in captured.err


def test_all_mode_written_and_fitted_rrms_agree(butane_pdb, butane_itp, tmp_path, fake_esp, capsys):
    """With every atom replaced, the written set is the fitted set up to rounding."""
    outdir = tmp_path / "out"
    rc = main(
        _args(butane_pdb, butane_itp, outdir, "--charge-mode", "all", "--charge-geom", "input")
    )
    assert rc == 0
    captured = capsys.readouterr()

    payload = _charges_json(outdir)
    assert payload["rrms_written"] == pytest.approx(payload["rrms_fitted"], abs=1e-3)
    assert payload["rrms_before"] == pytest.approx(
        fake_esp.rrms([a["charge_before"] for a in payload["atoms"]]), rel=1e-6
    )
    assert "poorly" not in captured.err


def test_a_charge_that_disagrees_with_the_topology_is_refused_in_selected_mode(
    butane_pdb, butane_itp, tmp_path, fake_esp, capsys
):
    """--charge -1 on neutral butane with one atom selected (ruling R21).

    Before the fix this exited 0 with C1's three hydrogens taken from +0.090
    to about -0.2 each. Now it is refused, and no charged topology is written.
    """
    outdir = tmp_path / "out"
    rc = main(
        _args(
            butane_pdb,
            butane_itp,
            outdir,
            "--charge-atoms",
            "1",
            "--charge",
            "-1",
            "--charge-geom",
            "input",
        )
    )

    assert rc == 1
    err = capsys.readouterr().err
    assert "--charge-mode all" in err
    assert "Traceback" not in err
    assert not (outdir / "04_charges" / "molecule_charged.itp").exists()


def _six_decimal_itp(butane_itp, path):
    """butane.itp with 6-decimal charges (as antechamber writes), spacing kept."""
    import re

    charges = {
        "C1": "-0.270123",
        "C2": "-0.180246",
        "C3": "-0.180246",
        "C4": "-0.270123",
        "H11": "0.090041",
        "H12": "0.090041",
        "H13": "0.090041",
        "H21": "0.090123",
        "H22": "0.090123",
        "H31": "0.090123",
        "H32": "0.090123",
        "H41": "0.090041",
        "H42": "0.090041",
        "H43": "0.090041",
    }
    out, section = [], ""
    for line in butane_itp.read_text().splitlines():
        body = line.split(";", 1)[0].strip()
        if body.startswith("["):
            section = body.strip("[] ")
        elif section == "atoms" and body:
            tokens = list(re.finditer(r"\S+", line))
            q = tokens[6]
            line = line[: q.start()] + charges[tokens[4].group()] + line[q.end() :]
        out.append(line)
    path.write_text("\n".join(out) + "\n")
    return path


def test_a_six_decimal_topology_keeps_every_untouched_line(
    butane_pdb, butane_itp, tmp_path, fake_esp
):
    """Final review I3 (ruling R22), through cmd_charges and the writer.

    With C1 selected, only C1's charge group may change. Every other
    ``[ atoms ]`` line must be byte-identical to the input, every untouched
    atom labelled ``unchanged``, and the changed charges written at the
    input's own 6 decimals rather than 4.
    """
    itp = _six_decimal_itp(butane_itp, tmp_path / "butane6.itp")
    outdir = tmp_path / "out"
    rc = main(_args(butane_pdb, itp, outdir, "--charge-atoms", "1", "--charge-geom", "input"))
    assert rc == 0

    payload = _charges_json(outdir)
    assert payload["provenance"]["charge_decimals"] == 6
    assert [a["source"] for a in payload["atoms"][4:]] == ["unchanged"] * 10
    assert sum(a["charge_after"] for a in payload["atoms"]) == pytest.approx(0.0, abs=1e-9)

    written = (outdir / "04_charges" / "molecule_charged.itp").read_text().splitlines()
    original = itp.read_text().splitlines()
    differing = [w for w, o in zip(written, original) if w != o]
    assert len(written) == len(original)
    assert len(differing) == 4  # C1 and its three hydrogens, nothing else
    for line in differing:
        assert len(line.split()[6].split(".")[1]) == 6


def test_charges_json_keeps_every_warning_the_stage_printed(
    butane_pdb, butane_itp, tmp_path, fake_esp, capsys
):
    """Final review I6 (ruling R25): design section 7 puts warnings in charges.json.

    ``--charge-mode all`` on butane's CGenFF-style types triggers the CGenFF
    warning; milestone 6's report needs it, and the RRMS warning, from a
    durable source rather than scrollback.
    """
    outdir = tmp_path / "out"
    rc = main(
        _args(butane_pdb, butane_itp, outdir, "--charge-mode", "all", "--charge-geom", "input")
    )
    assert rc == 0
    err = capsys.readouterr().err

    warnings = _charges_json(outdir)["warnings"]
    cgenff = [w for w in warnings if "CGenFF" in w]
    assert len(cgenff) == 1
    assert cgenff[0] in err  # the same text the user saw


def test_charges_json_keeps_the_rrms_warning(butane_pdb, butane_itp, tmp_path, fake_esp, capsys):
    outdir = tmp_path / "out"
    rc = main(
        _args(butane_pdb, butane_itp, outdir, "--charge-atoms", "1", "--charge-geom", "input")
    )
    assert rc == 0
    err = capsys.readouterr().err

    warnings = _charges_json(outdir)["warnings"]
    poorly = [w for w in warnings if "poorly" in w]
    assert len(poorly) == 1
    assert poorly[0] in err
    assert not any("CGenFF" in w for w in warnings)  # selected mode: no CGenFF warning


def test_charges_json_records_each_atoms_element(butane_pdb, butane_itp, tmp_path, fake_esp):
    """Design section 4 lists a per-atom element."""
    outdir = tmp_path / "out"
    rc = main(
        _args(butane_pdb, butane_itp, outdir, "--charge-atoms", "1", "--charge-geom", "input")
    )
    assert rc == 0

    elements = [a["element"] for a in _charges_json(outdir)["atoms"]]
    assert elements == ["C", "H", "H", "H", "C", "H", "H", "C", "H", "H", "C", "H", "H", "H"]


def test_out_of_range_charge_atoms_is_refused_in_a_dry_run(
    butane_pdb, butane_itp, tmp_path, capsys
):
    """Final review M4: a dry run must not promise a run that would be refused."""
    outdir = tmp_path / "out"
    rc = main(_args(butane_pdb, butane_itp, outdir, "--charge-atoms", "1,15", "--dry-run"))

    assert rc == 1
    err = capsys.readouterr().err
    assert "--charge-atoms names atom(s) [15], but the topology has 14 atoms" in err
    assert not outdir.exists()


@pytest.mark.parametrize(
    "extra",
    [
        ("--charge-atoms", "15"),
        ("--charge-atoms", "1", "--charge", "-1"),
    ],
    ids=["atom-out-of-range", "whole-electron-offset"],
)
def test_a_bad_selection_is_refused_before_any_qm_job_runs(
    butane_pdb, butane_itp, tmp_path, fake_esp, monkeypatch, capsys, extra
):
    """Final review M4: refuse before xtb and ORCA, not after them.

    Both refusals need only the topology. They used to fire inside
    apply_charges, after the default --charge-geom xtb optimisation and the
    ORCA ESP job had already run.
    """
    import ffrefine.cli as cli

    xtb_calls = []
    monkeypatch.setattr(cli, "optimize_geometry", lambda *a, **k: xtb_calls.append(1) or a[2])

    rc = main(_args(butane_pdb, butane_itp, tmp_path / "out", *extra))

    assert rc == 1
    assert xtb_calls == []
    assert fake_esp.calls == 0
    assert "Traceback" not in capsys.readouterr().err


def test_charge_atoms_with_charge_mode_all_is_refused(butane_pdb, butane_itp, tmp_path, capsys):
    """--charge-mode all replaces every charge; a --charge-atoms list would be ignored."""
    rc = main(
        _args(
            butane_pdb,
            butane_itp,
            tmp_path / "out",
            "--charge-mode",
            "all",
            "--charge-atoms",
            "1",
            "--dry-run",
        )
    )

    assert rc == 1
    err = capsys.readouterr().err
    assert "--charge-atoms" in err
    assert "--charge-mode all" in err


def test_selecting_part_of_an_equivalence_group_is_warned_about_and_recorded(
    butane_pdb, butane_itp, tmp_path, fake_esp, capsys
):
    """Final review M3: one hydrogen of a methyl breaks spec 5.3's equal-charge rule.

    Atoms 2, 3 and 4 are C1's methyl hydrogens. Selecting only atom 2 replaces
    its charge and leaves 3 and 4 at theirs, so the group is no longer equal;
    that used to happen without a word.
    """
    outdir = tmp_path / "out"
    rc = main(
        _args(butane_pdb, butane_itp, outdir, "--charge-atoms", "2", "--charge-geom", "input")
    )
    assert rc == 0
    err = capsys.readouterr().err

    partial = [w for w in _charges_json(outdir)["warnings"] if "equivalent" in w]
    assert len(partial) == 1
    assert "[2, 3, 4]" in partial[0]
    assert "[3, 4] are not selected" in partial[0]
    assert partial[0] in err


def test_selecting_a_whole_equivalence_group_is_not_warned_about(
    butane_pdb, butane_itp, tmp_path, fake_esp
):
    outdir = tmp_path / "out"
    rc = main(
        _args(butane_pdb, butane_itp, outdir, "--charge-atoms", "1,2,3,4", "--charge-geom", "input")
    )
    assert rc == 0

    assert not [w for w in _charges_json(outdir)["warnings"] if "equivalent" in w]


def test_a_dry_run_warns_about_a_partial_equivalence_group(
    butane_pdb, butane_itp, tmp_path, capsys
):
    rc = main(_args(butane_pdb, butane_itp, tmp_path / "out", "--charge-atoms", "6", "--dry-run"))

    assert rc == 0
    err = capsys.readouterr().err
    assert "[6, 7]" in err
    assert "[7] are not selected" in err


# ---------------------------------------------------------------------------
# provenance of a chained run (final review I5, ruling R24)
# ---------------------------------------------------------------------------


def _run_args(butane_pdb, butane_itp, outdir):
    import ffrefine.cli as cli

    return cli.build_parser().parse_args(
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
            "--charge-atoms",
            "1",
        ]
    )


def _fake_charges_stage(butane_itp):
    """Stands in for cmd_charges: prepares 00_input as the real one does, then
    writes a charged topology visibly different from the input."""
    import ffrefine.cli as cli

    def fake(a):
        cli._prepare(a, need_quad=False)
        charged = cli.stage_dir(a.outdir, "charges") / "molecule_charged.itp"
        charged.write_text(
            butane_itp.read_text().replace("-0.270", "-0.300", 1).replace("0.090", "0.100", 1)
        )
        return 0

    return fake


def test_a_chained_run_keeps_the_users_topology_in_00_input(
    butane_pdb, butane_itp, tmp_path, monkeypatch
):
    """00_input/ is the user's input, never a stage output.

    After cmd_run repoints -p at 04_charges/molecule_charged.itp, each later
    stage's _prepare used to copy that file into 00_input/ and name it in
    model.json, so the run's record of its own input named its own output.
    """
    import ffrefine.cli as cli

    def later_stage(a):
        cli._prepare(a)
        return 0

    monkeypatch.setattr(cli, "cmd_charges", _fake_charges_stage(butane_itp))
    for name in ("cmd_scan", "cmd_mmscan", "cmd_fit", "cmd_validate"):
        monkeypatch.setattr(cli, name, later_stage)

    outdir = tmp_path / "out"
    assert cli.cmd_run(_run_args(butane_pdb, butane_itp, outdir)) == 0

    input_dir = outdir / "00_input"
    assert sorted(p.name for p in input_dir.iterdir()) == ["butane.itp", "butane.pdb", "model.json"]
    assert (input_dir / "butane.itp").read_text() == butane_itp.read_text()
    model = json.loads((input_dir / "model.json").read_text())
    assert model["topology"] == str(butane_itp.resolve())


def _fake_validation_inputs(outdir, butane_pdb):
    """scan.json and fit.json that cmd_validate's pairing checks accept."""
    from ffrefine.io.structure import read_pdb
    from ffrefine.stages import hash_array

    frames = np.stack([read_pdb(butane_pdb).coords] * 2)
    (outdir / "01_qm_scan").mkdir(parents=True, exist_ok=True)
    (outdir / "01_qm_scan" / "scan.json").write_text(
        json.dumps({"energy_kj": [0.0, 1.0], "frames_nm": frames.tolist()})
    )
    (outdir / "03_fit").mkdir(parents=True, exist_ok=True)
    (outdir / "03_fit" / "fit.json").write_text(
        json.dumps(
            {
                "terms": [{"multiplicity": 3, "k_kj_per_mol": 1.0, "phase_deg": 0.0}],
                "provenance": {"quad_1_based": [1, 5, 8, 11], "frames_sha256": hash_array(frames)},
            }
        )
    )


def _fake_validate_refit(seen):
    from types import SimpleNamespace

    def fake(binaries, topology, quad, structure, frames, qm, k, delta, mult, final, **kw):
        seen.append(topology)
        (final / "after.itp").write_text("; after\n")
        zeros = np.zeros(len(qm))
        return SimpleNamespace(
            rmse_before=1.0,
            rmse_after=0.5,
            improved=True,
            phi_deg=zeros,
            qm_kj=np.asarray(qm),
            mm_before_kj=zeros,
            mm_after_kj=zeros,
        )

    return fake


def test_validation_json_names_the_charge_refined_topology_it_measured_before_against(
    butane_pdb, butane_itp, tmp_path, monkeypatch
):
    """validation.json claimed "the original force field topology, unmodified"
    even when the before-RMSE was measured against 04_charges/molecule_charged.itp."""
    import ffrefine.cli as cli

    outdir = tmp_path / "out"
    _fake_validation_inputs(outdir, butane_pdb)
    seen = []
    monkeypatch.setattr(cli, "cmd_charges", _fake_charges_stage(butane_itp))
    for name in ("cmd_scan", "cmd_mmscan", "cmd_fit"):
        monkeypatch.setattr(cli, name, lambda a: 0)
    monkeypatch.setattr(cli, "validate_refit", _fake_validate_refit(seen))
    monkeypatch.setattr(cli, "plot_validation", lambda *a, **k: None)

    assert cli.cmd_run(_run_args(butane_pdb, butane_itp, outdir)) == 0

    charged = outdir / "04_charges" / "molecule_charged.itp"
    assert seen[0].atoms[0].charge == pytest.approx(-0.300)  # before really was the charged file
    validation = json.loads((outdir / "final" / "validation.json").read_text())
    assert validation["before_topology"] == str(charged.resolve())
    assert validation["charges_refined"] is True
    assert validation["input_topology"] == str(butane_itp.resolve())


def test_validation_json_names_the_users_topology_when_no_charges_ran(
    butane_pdb, butane_itp, tmp_path, monkeypatch
):
    import ffrefine.cli as cli

    outdir = tmp_path / "out"
    _fake_validation_inputs(outdir, butane_pdb)
    monkeypatch.setattr(cli, "validate_refit", _fake_validate_refit([]))
    monkeypatch.setattr(cli, "plot_validation", lambda *a, **k: None)

    args = cli.build_parser().parse_args(
        [
            "validate",
            "-s",
            str(butane_pdb),
            "-p",
            str(butane_itp),
            "--dihedral",
            "1,5,8,11",
            "--outdir",
            str(outdir),
        ]
    )
    cli._apply_config(args)
    assert cli.cmd_validate(args) == 0

    validation = json.loads((outdir / "final" / "validation.json").read_text())
    assert validation["before_topology"] == str(butane_itp.resolve())
    assert validation["charges_refined"] is False
    assert validation["input_topology"] == str(butane_itp.resolve())
