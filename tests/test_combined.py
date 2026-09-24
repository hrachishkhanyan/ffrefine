"""The combined check: every bond's scan, original topology against final (design section 5)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ffrefine import cli
from ffrefine.cli import build_parser
from ffrefine.mm.validate_fit import ValidationResult

QUADS = [(1, 5, 8, 11), (2, 1, 5, 8)]


def _bond_tree(outdir: Path, quad, rmse_before: float, rmse_after: float) -> Path:
    """A finished bond subfolder: its scan and its own validation."""
    bond = outdir / cli._bond_dirname(quad)
    (bond / "01_qm_scan").mkdir(parents=True)
    frames = np.arange(4 * 14 * 3, dtype=float).reshape(4, 14, 3) * 0.01
    (bond / "01_qm_scan" / "scan.json").write_text(
        json.dumps(
            {"quad": list(quad), "energy_kj": [0.0, 1.0, 2.0, 1.0], "frames_nm": frames.tolist()}
        )
    )
    (bond / "final").mkdir()
    (bond / "final" / "validation.json").write_text(
        json.dumps({"rmse_before_kj_per_mol": rmse_before, "rmse_after_kj_per_mol": rmse_after})
    )
    return bond


def _args(outdir, butane_pdb, butane_itp, *extra):
    args = build_parser().parse_args(
        ["run", "-s", str(butane_pdb), "-p", str(butane_itp), "--outdir", str(outdir), *extra]
    )
    cli._apply_config(args)
    return args


def _stub(monkeypatch, calls, improved=(True, True)):
    outcomes = iter(improved)

    def fake_compare(binaries, before, after, structure, frames, qm, quad, workdir, **kwargs):
        calls.append(quad)
        good = next(outcomes)
        return ValidationResult(
            rmse_before=2.0,
            rmse_after=1.0 if good else 3.0,
            mm_before_kj=np.zeros(4),
            mm_after_kj=np.ones(4),
            qm_kj=np.asarray(qm, float),
            phi_deg=np.array([0.0, 90.0, 180.0, 270.0]),
            improved=good,
        )

    monkeypatch.setattr(cli, "compare_topologies", fake_compare)
    monkeypatch.setattr(cli, "preprocessed_sha256", lambda *a, **k: "d" * 64)
    monkeypatch.setattr(cli, "plot_validation", lambda *a, **k: None)


def test_combined_check_records_every_bond_in_run_order(
    tmp_path, butane_pdb, butane_itp, monkeypatch
):
    out = tmp_path / "out"
    bonds = [(q, _bond_tree(out, q, 3.0, 1.5)) for q in QUADS]
    calls: list = []
    _stub(monkeypatch, calls, improved=(True, False))
    cli._combined_check(_args(out, butane_pdb, butane_itp), bonds, butane_itp, butane_itp)
    assert calls == QUADS
    data = json.loads((out / "final" / "combined.json").read_text())
    assert [row["quad_1_based"] for row in data["bonds"]] == [list(q) for q in QUADS]
    first = data["bonds"][0]
    assert (first["rmse_original"], first["rmse_final"]) == (2.0, 1.0)
    assert (first["rmse_own_before"], first["rmse_own_after"]) == (3.0, 1.5)
    assert first["dir"] == "torsion_1-5-8-11"
    assert data["bonds"][1]["improved"] is False
    assert any("no better than the original" in w for w in data["warnings"])


def test_combined_check_resumes(tmp_path, butane_pdb, butane_itp, monkeypatch):
    """Review focus 1: a finished combined check is skipped under --resume,
    and recomputed when the final topology changes."""
    out = tmp_path / "out"
    bonds = [(q, _bond_tree(out, q, 3.0, 1.5)) for q in QUADS]
    final = tmp_path / "final.itp"
    final.write_text(butane_itp.read_text())
    calls: list = []
    _stub(monkeypatch, calls, improved=(True,) * 6)
    cli._combined_check(_args(out, butane_pdb, butane_itp), bonds, butane_itp, final)
    assert len(calls) == 2
    cli._combined_check(_args(out, butane_pdb, butane_itp, "--resume"), bonds, butane_itp, final)
    assert len(calls) == 2
    final.write_text(butane_itp.read_text() + "; edited\n")
    cli._combined_check(_args(out, butane_pdb, butane_itp, "--resume"), bonds, butane_itp, final)
    assert len(calls) == 4
