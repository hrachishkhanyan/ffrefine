"""Milestone 6b end to end, on the real CGenFF fixtures and the real programs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ffrefine.cli import main
from ffrefine.io.topwriter import charges_provenance

FIXTURES = Path(__file__).parent.parent / "fixtures"
PFOA = FIXTURES / "pfoa_gromacs"
CAP = FIXTURES / "capecitabine_gromacs"
PAYLOADS = (
    "01_qm_scan/scan.json",
    "02_mm_scan/energies.json",
    "03_fit/fit.json",
    "final/validation.json",
)


@pytest.mark.needs_gmx
@pytest.mark.needs_xtb
def test_capecitabine_run_with_str_refits_its_one_flagged_torsion(tmp_path, gmx_bin, xtb_bin):
    out = tmp_path / "out"
    argv = ["run", "-s", str(CAP / "capecitabine_gmx.pdb"), "-p", str(CAP / "capecitabine_gmx.top")]
    assert main(argv + ["--str", str(CAP / "capecitabine.str"), "--outdir", str(out)]) == 0
    validation = json.loads((out / "final" / "validation.json").read_text())
    assert validation["quad_1_based"] == [6, 7, 9, 10]
    assert not (out / "04_charges").exists()
    for rel in PAYLOADS:
        assert isinstance(json.loads((out / rel).read_text())["warnings"], list), rel
    report = (out / "final" / "report.md").read_text(encoding="utf-8")
    assert "| 6-7-9-10 |" in report and "(OG302 CG2O6 NG2S1 CG2R64)" in report
    assert "| 50 |" in report
    assert "valid only with the same nonbonded parameters" in report


@pytest.mark.needs_gmx
@pytest.mark.needs_xtb
def test_pfoa_run_with_str_refits_both_bonds_in_sequence(tmp_path, gmx_bin, xtb_bin):
    out = tmp_path / "out"
    argv = ["run", "-s", str(PFOA / "pfoa_gmx.pdb"), "-p", str(PFOA / "pfoa_gmx.top")]
    argv += ["--str", str(PFOA / "pfoa.str"), "--no-charges", "--outdir", str(out)]
    assert main(argv) == 0

    status = json.loads((out / "final" / "run_status.json").read_text())
    assert [t["dir"] for t in status["torsions"]] == ["torsion_1-3-4-5", "torsion_25-1-3-4"]
    assert {t["status"] for t in status["torsions"]} == {"complete"}

    bond1 = out / "torsion_1-3-4-5" / "final" / "molecule_refined.itp"
    second = json.loads((out / "torsion_25-1-3-4" / "final" / "validation.json").read_text())
    assert second["before_topology"] == str(bond1.resolve())

    final_lines = (out / "final" / "molecule_refined.itp").read_text().splitlines()

    def fitted(quad):
        head = [str(n) for n in quad]
        return [line for line in final_lines if line.split()[:4] == head and "ffrefine fit" in line]

    assert fitted((1, 3, 4, 5)) and fitted((25, 1, 3, 4))
    combined = json.loads((out / "final" / "combined.json").read_text())
    assert [row["quad_1_based"] for row in combined["bonds"]] == [[1, 3, 4, 5], [25, 1, 3, 4]]
    report = (out / "final" / "report.md").read_text(encoding="utf-8")
    assert "## Torsion 1-3-4-5 (`torsion_1-3-4-5/`)" in report
    assert "## Torsion 25-1-3-4 (`torsion_25-1-3-4/`)" in report
    assert "RMSE with every change" in report


@pytest.mark.needs_orca
def test_pfoa_charges_from_str(tmp_path, orca_bin):
    out = tmp_path / "out"
    argv = ["charges", "-s", str(PFOA / "pfoa_gmx.pdb"), "-p", str(PFOA / "pfoa_gmx.top")]
    argv += ["--str", str(PFOA / "pfoa.str"), "--charge-geom", "input", "--outdir", str(out)]
    assert main(argv) == 0
    charges = json.loads((out / "04_charges" / "charges.json").read_text())
    assert len(charges["provenance"]["selected_atoms"]) == 23
    assert any("close to replacing every charge" in w for w in charges["warnings"])
    assert charges["provenance"]["orca_version"].startswith("Program Version")
    charged = out / "04_charges" / "molecule_charged.itp"
    assert charges_provenance(charged) == (out / "04_charges" / "charges.json").resolve()
    assert (out / "00_input" / "selection.json").exists()
