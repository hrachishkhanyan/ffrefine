"""The CLI on a real CGenFF topology: typed torsions and #include'd force fields.

Milestone 6a. The unit tests here stub the GROMACS-facing calls, so they need
no binary; the live tests are marked.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
from ffrefine import cli
from ffrefine.cli import main
from ffrefine.config import Binaries
from ffrefine.io.includes import MARKER_NOTE, MARKER_PREFIX
from ffrefine.io.structure import read_pdb
from ffrefine.io.topwriter import make_system_top, write_charged_topology, write_refit_topology
from ffrefine.mm.gromacs import single_point_energies
from ffrefine.mm.selfcheck import SelfCheckResult
from ffrefine.mm.torsionset import collect_torsion_set
from ffrefine.mm.tprparams import SOURCE_TOPOLOGY, ResolvedTerm, ResolvedTorsionSet
from ffrefine.mm.validate_fit import ValidationResult
from ffrefine.stages import hash_array

FIXTURES = Path(__file__).parent.parent / "fixtures"
PFOA = FIXTURES / "pfoa_gromacs"
PFOA_QUAD = (25, 1, 3, 4)
PFOA_FF = (PFOA / "charmm36.ff").resolve()


def _pfoa_args(outdir: Path, top: Path | None = None) -> list[str]:
    return [
        "-s",
        str(PFOA / "pfoa_gmx.pdb"),
        "-p",
        str(top or PFOA / "pfoa_gmx.top"),
        "--dihedral",
        ",".join(map(str, PFOA_QUAD)),
        "--outdir",
        str(outdir),
    ]


def _include_lines(path: Path) -> list[str]:
    return [
        line for line in Path(path).read_text().splitlines() if re.match(r"^\s*#\s*include", line)
    ]


def _marker(path: Path) -> list[str]:
    return Path(path).read_text().splitlines()[:2]


def _write_pfoa_stage_inputs(outdir: Path, n: int = 24) -> None:
    """Hand-built 01_qm_scan and 02_mm_scan payloads for PFOA (see test_cli.py).

    ``fit`` only hashes the frames, so placeholder coordinates of the right
    shape are enough, and they must be the same in both payloads.
    """
    phi = np.linspace(-180.0, 180.0, n, endpoint=False)
    qm = 5.0 * (1.0 + np.cos(np.radians(3.0 * phi)))
    frames = np.arange(n * 26 * 3, dtype=float).reshape(n, 26, 3) * 0.01
    (outdir / "01_qm_scan").mkdir(parents=True, exist_ok=True)
    (outdir / "01_qm_scan" / "scan.json").write_text(
        json.dumps(
            {"phi_deg": phi.tolist(), "energy_kj": qm.tolist(), "frames_nm": frames.tolist()}
        )
    )
    (outdir / "02_mm_scan").mkdir(parents=True, exist_ok=True)
    (outdir / "02_mm_scan" / "energies.json").write_text(
        json.dumps(
            {
                "quad": list(PFOA_QUAD),
                "frames_sha256": hash_array(frames),
                "phi_deg": phi.tolist(),
                "zeroed_potential_kj": [0.0] * n,
                "full_potential_kj": [0.0] * n,
            }
        )
    )


def _fake_validate_refit(
    binaries, topology, quad, structure, frames, qm, k, delta, mult, workdir, **kwargs
):
    """validate_refit without GROMACS: writes after.itp the way the real one does."""
    write_refit_topology(
        topology,
        quad,
        k,
        delta,
        mult,
        Path(workdir) / "after.itp",
        distribute=kwargs.get("distribute", False),
        includes="absolute",
    )
    n = len(qm)
    return ValidationResult(
        rmse_before=1.0,
        rmse_after=0.5,
        mm_before_kj=np.zeros(n),
        mm_after_kj=np.zeros(n),
        qm_kj=np.asarray(qm, dtype=float),
        phi_deg=np.linspace(-180.0, 180.0, n, endpoint=False),
        improved=True,
    )


def _stub_validate(monkeypatch, sha: str = "a" * 64) -> None:
    """Replace every GROMACS-facing call cmd_validate makes."""
    monkeypatch.setattr(cli, "validate_refit", _fake_validate_refit)
    monkeypatch.setattr(cli, "preprocessed_sha256", lambda *args, **kwargs: sha)


def test_fit_writes_dihedral_itp_with_the_users_includes(tmp_path):
    out = tmp_path / "out"
    _write_pfoa_stage_inputs(out)
    assert main(["fit"] + _pfoa_args(out)) == 0
    itp = out / "03_fit" / "dihedral.itp"
    assert _include_lines(itp) == _include_lines(PFOA / "pfoa_gmx.top")
    assert _marker(itp) == [MARKER_NOTE, f"{MARKER_PREFIX}{PFOA.resolve()}"]


def test_validate_writes_molecule_refined_as_a_drop_in(tmp_path, monkeypatch):
    """final/after.itp goes to grompp (absolute); molecule_refined.itp is kept (verbatim)."""
    out = tmp_path / "out"
    _write_pfoa_stage_inputs(out)
    assert main(["fit"] + _pfoa_args(out)) == 0
    _stub_validate(monkeypatch)
    assert main(["validate"] + _pfoa_args(out)) == 0
    final = out / "final"
    refined = final / "molecule_refined.itp"
    assert _include_lines(refined) == _include_lines(PFOA / "pfoa_gmx.top")
    assert _marker(refined) == [MARKER_NOTE, f"{MARKER_PREFIX}{PFOA.resolve()}"]
    assert f'#include "{PFOA_FF / "forcefield.itp"}"' in _include_lines(final / "after.itp")


NO_DEFAULTS_WARNING = "has no [ defaults ] directive"


def _without_defaults(butane_itp: Path, tmp_path: Path) -> Path:
    """butane.itp minus its [ defaults ] block, and with no #include either."""
    stripped, n = re.subn(r"\[ defaults \]\n;[^\n]*\n[^\n]*\n", "", butane_itp.read_text())
    assert n == 1, "butane fixture layout changed"
    out = tmp_path / "no_defaults.itp"
    out.write_text(stripped, newline="\n")
    return out


def test_model_json_defers_the_defaults_warning_for_an_included_force_field(tmp_path, capsys):
    """PFOA's [ defaults ] is in charmm36.ff/forcefield.itp: mmscan reads it, not 00_input."""
    out = tmp_path / "out"
    assert main(["fit"] + _pfoa_args(out)) == 1  # no scan yet; 00_input is written first
    assert NO_DEFAULTS_WARNING not in capsys.readouterr().err
    assert json.loads((out / "00_input" / "model.json").read_text())["fudgeQQ"] is None


def test_model_json_still_warns_with_neither_defaults_nor_includes(
    tmp_path, butane_pdb, butane_itp, capsys
):
    itp = _without_defaults(butane_itp, tmp_path)
    argv = ["fit", "-s", str(butane_pdb), "-p", str(itp), "--dihedral", "1,5,8,11"]
    assert main(argv + ["--outdir", str(tmp_path / "out")]) == 1
    assert NO_DEFAULTS_WARNING in capsys.readouterr().err


def test_model_json_defers_the_warning_for_ff_include(tmp_path, butane_pdb, butane_itp, capsys):
    itp = _without_defaults(butane_itp, tmp_path)
    ff = tmp_path / "ff.itp"
    ff.write_text("[ defaults ]\n1 2 yes 1.0 1.0\n")
    argv = ["fit", "-s", str(butane_pdb), "-p", str(itp), "--dihedral", "1,5,8,11"]
    argv += ["--ff-include", str(ff), "--outdir", str(tmp_path / "out")]
    assert main(argv) == 1
    assert NO_DEFAULTS_WARNING not in capsys.readouterr().err


def _fake_resolve(sha: str):
    def fake(binaries, topology, quad, structure, workdir, ff_includes=None):
        return ResolvedTorsionSet(
            terms=[ResolvedTerm(d, SOURCE_TOPOLOGY) for d in collect_torsion_set(topology, quad)],
            defaults=topology.defaults,
            pp_sha256=sha,
            full_top=Path(workdir) / "full.top",
            dump_path=Path(workdir) / "full_dump.stdout",
            gromacs_version="test",
        )

    return fake


def _fake_check(calls: list):
    def fake(*args, **kwargs):
        calls.append(kwargs.get("resolved"))
        one = np.zeros(1)
        return SelfCheckResult(
            max_abs_error=0.0,
            passed=True,
            gromacs_difference=one,
            analytic_difference=one,
            phi_deg=one,
            tolerance=1e-3,
            full_potential_kj=one,
            zeroed_potential_kj=one,
        )

    return fake


def test_mmscan_resume_key_covers_the_preprocessed_topology(
    tmp_path, butane_pdb, butane_itp, monkeypatch
):
    out = tmp_path / "out"
    argv = ["mmscan", "-s", str(butane_pdb), "-p", str(butane_itp), "--dihedral", "1,5,8,11"]
    argv += ["--outdir", str(out)]
    calls: list = []
    monkeypatch.setattr(cli, "check_torsion_removal", _fake_check(calls))
    monkeypatch.setattr(cli, "resolve_torsion_set", _fake_resolve("a" * 64))
    assert main(argv) == 0 and len(calls) == 1
    assert calls[0].pp_sha256 == "a" * 64  # the check gets mmscan's resolution, not its own
    assert main(argv + ["--resume"]) == 0 and len(calls) == 1  # same key: skipped
    monkeypatch.setattr(cli, "resolve_torsion_set", _fake_resolve("b" * 64))
    assert main(argv + ["--resume"]) == 0 and len(calls) == 2  # an included file changed

    payload = json.loads((out / "02_mm_scan" / "energies.json").read_text())
    assert payload["preprocessed_topology_sha256"] == "b" * 64
    assert len(payload["torsion_parameters"]) == 10
    assert payload["torsion_parameters"][0] == {
        "quad_1_based": [1, 5, 8, 11],
        "funct": 9,
        "phi0_deg": 0.0,
        "k_kj_per_mol": 0.58576,
        "multiplicity": 3,
        "source": "topology",
    }
    assert payload["defaults"] == {
        "fudgeQQ": 1.0,
        "fudgeLJ": 1.0,
        "gen_pairs": True,
        "source": "grompp -pp",
    }


@pytest.mark.needs_gmx
def test_mmscan_on_pfoa_records_grompps_parameters(tmp_path, gmx_bin, capsys):
    out = tmp_path / "out"
    assert main(["mmscan"] + _pfoa_args(out)) == 0
    assert NO_DEFAULTS_WARNING not in capsys.readouterr().err
    payload = json.loads((out / "02_mm_scan" / "energies.json").read_text())
    params = payload["torsion_parameters"]
    assert len(params) == 20
    assert {p["source"] for p in params} == {"dihedraltypes"}
    target = [p for p in params if p["quad_1_based"] == list(PFOA_QUAD)]
    assert [p["multiplicity"] for p in target] == [1, 2, 3, 6]
    assert payload["defaults"] == {
        "fudgeQQ": 1.0,
        "fudgeLJ": 1.0,
        "gen_pairs": True,
        "source": "grompp -pp",
    }
    assert payload["selfcheck_max_abs_error_kj"] < 1e-3


@pytest.mark.needs_gmx
def test_mmscan_on_a_marked_copy_finds_the_force_field(tmp_path, gmx_bin):
    """A charge-refined topology written elsewhere still finds charmm36.ff/ (R28's scenario)."""
    moved = tmp_path / "elsewhere"
    moved.mkdir()
    write_charged_topology(PFOA / "pfoa_gmx.top", [], moved / "molecule_charged.itp")
    out = tmp_path / "out"
    assert main(["mmscan"] + _pfoa_args(out, top=moved / "molecule_charged.itp")) == 0
    assert f'#include "{PFOA_FF / "forcefield.itp"}"' in _include_lines(
        out / "02_mm_scan" / "full.itp"
    )


@pytest.mark.needs_gmx
def test_mmscan_resume_reruns_when_an_included_file_changes(tmp_path, gmx_bin, capsys):
    work = tmp_path / "pfoa"
    shutil.copytree(PFOA, work)
    out = tmp_path / "out"
    argv = ["mmscan", "-s", str(work / "pfoa_gmx.pdb"), "-p", str(work / "pfoa_gmx.top")]
    argv += ["--dihedral", "25,1,3,4", "--outdir", str(out)]
    assert main(argv) == 0
    capsys.readouterr()
    assert main(argv + ["--resume"]) == 0
    assert "skipping" in capsys.readouterr().out

    # The first 0.669440 in pfo_ffbonded.itp is OG2D1 CG2O2 CG312 CG312, multiplicity 2:
    # quad 2 1 3 4, which is in the torsion set about C1-C3.
    ffbonded = work / "charmm36.ff" / "pfo_ffbonded.itp"
    ffbonded.write_text(ffbonded.read_text().replace("0.669440", "0.700000", 1))
    assert main(argv + ["--resume"]) == 0
    assert "skipping" not in capsys.readouterr().out
    payload = json.loads((out / "02_mm_scan" / "energies.json").read_text())
    ks = [
        p["k_kj_per_mol"]
        for p in payload["torsion_parameters"]
        if p["quad_1_based"] == [2, 1, 3, 4]
    ]
    assert ks[1] == pytest.approx(0.7, rel=1e-6)


def test_validate_resume_key_covers_the_preprocessed_topology(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    _write_pfoa_stage_inputs(out)
    assert main(["fit"] + _pfoa_args(out)) == 0
    _stub_validate(monkeypatch, "a" * 64)
    assert main(["validate"] + _pfoa_args(out)) == 0
    capsys.readouterr()
    assert main(["validate"] + _pfoa_args(out) + ["--resume"]) == 0
    assert "skipping" in capsys.readouterr().out
    _stub_validate(monkeypatch, "b" * 64)
    assert main(["validate"] + _pfoa_args(out) + ["--resume"]) == 0
    assert "skipping" not in capsys.readouterr().out


@pytest.mark.needs_gmx
@pytest.mark.needs_xtb
def test_run_on_pfoa_end_to_end_gives_a_drop_in_topology(tmp_path, gmx_bin, xtb_bin):
    """The milestone's goal: a real CGenFF .top, no hand edits, a usable result."""
    out = tmp_path / "out"
    assert main(["run"] + _pfoa_args(out)) == 0
    validation = json.loads((out / "final" / "validation.json").read_text())
    assert validation["quad_1_based"] == list(PFOA_QUAD)
    refined = out / "final" / "molecule_refined.itp"
    assert _include_lines(refined) == _include_lines(PFOA / "pfoa_gmx.top")

    # Drop-in: beside the original charmm36.ff/, the refined file runs as is.
    dropin = tmp_path / "dropin"
    dropin.mkdir()
    shutil.copytree(PFOA / "charmm36.ff", dropin / "charmm36.ff")
    shutil.copyfile(refined, dropin / "molecule_refined.itp")
    make_system_top(dropin / "molecule_refined.itp", dropin / "system.top", "Other")
    s = read_pdb(PFOA / "pfoa_gmx.pdb")
    energies = single_point_energies(
        Binaries(gmx=gmx_bin),
        dropin / "system.top",
        s,
        s.coords[None, :, :],
        dropin / "sp",
        "dropin",
    )
    assert np.isfinite(energies.potential()).all()


def test_energies_json_keeps_the_warnings_mmscan_raised(
    tmp_path, butane_pdb, butane_itp, monkeypatch
):
    """No scan.json: mmscan warns that it covers one geometry, and keeps the warning."""
    out = tmp_path / "out"
    monkeypatch.setattr(cli, "check_torsion_removal", _fake_check([]))
    monkeypatch.setattr(cli, "resolve_torsion_set", _fake_resolve("a" * 64))
    argv = ["mmscan", "-s", str(butane_pdb), "-p", str(butane_itp), "--dihedral", "1,5,8,11"]
    assert main(argv + ["--outdir", str(out)]) == 0
    energies = json.loads((out / "02_mm_scan" / "energies.json").read_text())
    assert any("no QM scan" in w for w in energies["warnings"])


def test_validation_json_keeps_the_did_not_improve_warning(tmp_path, monkeypatch):
    out = tmp_path / "out"
    _write_pfoa_stage_inputs(out)
    assert main(["fit"] + _pfoa_args(out)) == 0

    def worse(*args, **kwargs):
        import dataclasses

        result = _fake_validate_refit(*args, **kwargs)
        return dataclasses.replace(result, rmse_after=2.0, improved=False)

    _stub_validate(monkeypatch)
    monkeypatch.setattr(cli, "validate_refit", worse)
    assert main(["validate"] + _pfoa_args(out)) == 0
    validation = json.loads((out / "final" / "validation.json").read_text())
    assert any("did not improve" in w for w in validation["warnings"])
