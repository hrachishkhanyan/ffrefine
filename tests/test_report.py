"""report.md, built from payload trees on disk (milestone 6b design, section 6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ffrefine.errors import ValidationError
from ffrefine.report import write_report


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _selection(bonds):
    atoms = [
        {"nr": n, "top_name": name, "type": atype, "penalty": pen, "flagged": pen >= 10}
        for n, name, atype, pen in [
            (1, "C1", "CG2O2", 25.0),
            (3, "C3", "CG312", 30.5),
            (4, "C4", "CG312", 28.1),
            (5, "C5", "CG312", 30.8),
            (6, "O6", "OG302", 2.5),
            (7, "C7", "CG2O6", 2.6),
            (9, "N9", "NG2S1", 5.1),
            (10, "C10", "CG2R64", 5.4),
            (25, "O25", "OG311", 10.7),
        ]
    ]
    return {
        "str": "/data/mol.str",
        "cgenff_program_version": "4.0",
        "cgenff_parameter_version": "5.0",
        "resi_param_penalty": 69.0,
        "resi_charge_penalty": 41.068,
        "thresholds": {"charge": 10.0, "dihedral": 50.0},
        "atoms": atoms,
        "dihedral_types": [],
        "other_parameters": [
            {"kind": "angle", "types": ["CG312", "CG312", "CG312"], "penalty": 55.0}
        ],
        "bonds": [
            {"quad_1_based": list(q), "penalty": p, "bond_1_based": list(q[1:3])} for q, p in bonds
        ],
        "ring_bonds": [
            {"bond_1_based": [10, 16], "penalty": 60.0, "quad_1_based": [9, 10, 16, 15]}
        ],
        "unmatched_types": [],
        "warnings": ["a selection warning"],
    }


def _bond(
    root: Path,
    quad,
    rmse=(3.0, 1.5),
    warnings=None,
    defaults=True,
    stages=4,
    charges_refined=False,
    charges_json=None,
):
    warnings = warnings or {}
    _write(
        root / "01_qm_scan" / "scan.json",
        {
            "quad": list(quad),
            "provenance": {
                "gfn": 2,
                "xtb_version": "* xtb version 6.7.1 (edcfbbe)",
                "step_deg": 15.0,
                "npoints": 24,
                "bidirectional": True,
                "charge": 0,
                "uhf": 0,
                "hops": [],
                "hbond_changes": [],
            },
            "warnings": warnings.get("scan", []),
        },
    )
    if stages < 2:
        return
    energies = {
        "quad": list(quad),
        "torsion_parameters": [
            {
                "quad_1_based": list(quad),
                "multiplicity": 3,
                "k_kj_per_mol": 0.5858,
                "phi0_deg": 180.0,
                "source": "dihedraltypes",
                "funct": 9,
            }
        ],
        "selfcheck_max_abs_error_kj": 4.9e-05,
        "selfcheck_tolerance_kj": 0.001,
        "warnings": warnings.get("mmscan", []),
    }
    if defaults:
        energies["defaults"] = {
            "fudgeQQ": 1.0,
            "fudgeLJ": 1.0,
            "gen_pairs": True,
            "source": "grompp -pp",
        }
    _write(root / "02_mm_scan" / "energies.json", energies)
    if stages < 3:
        return
    _write(
        root / "03_fit" / "fit.json",
        {
            "provenance": {"quad_1_based": list(quad), "distribute": False},
            "terms": [
                {
                    "multiplicity": 3,
                    "k_kj_per_mol": 1.25,
                    "k_kcal_per_mol": 0.2988,
                    "phase_deg": 0.0,
                }
            ],
            "rmse_kj_per_mol": 0.4,
            "warnings": warnings.get("fit", []),
        },
    )
    if stages < 4:
        return
    _write(
        root / "final" / "validation.json",
        {
            "quad_1_based": list(quad),
            "rmse_before_kj_per_mol": rmse[0],
            "rmse_after_kj_per_mol": rmse[1],
            "improved": rmse[1] < rmse[0],
            "charges_refined": charges_refined,
            "charges_json": charges_json,
            "warnings": warnings.get("validate", []),
        },
    )


def _model(outdir: Path, **extra):
    _write(
        outdir / "00_input" / "model.json",
        {"molecule": "Other", "structure": "/data/mol.pdb", "topology": "/data/mol.top", **extra},
    )


def test_a_single_bond_report_has_every_required_part(tmp_path):
    _model(tmp_path)
    _write(tmp_path / "00_input" / "selection.json", _selection([((6, 7, 9, 10), 50.0)]))
    _bond(tmp_path, (6, 7, 9, 10))
    text = write_report(tmp_path).read_text(encoding="utf-8")
    for heading in (
        "# ffrefine report: Other",
        "## Summary",
        "## Torsion 6-7-9-10",
        "## Warnings",
        "## Validity",
        "## Files",
    ):
        assert heading in text
    assert "O6-C7-N9-C10 (OG302 CG2O6 NG2S1 CG2R64)" in text
    assert "| 50 |" in text
    assert "| 3.000 | 1.500 |" in text
    assert "GFN2-xTB (* xtb version 6.7.1 (edcfbbe))" in text
    assert "CGenFF program 4.0, parameters 5.0" in text
    assert "valid only with the same nonbonded parameters and the same 1-4 conventions" in text
    assert "fudgeQQ = 1, fudgeLJ = 1, gen-pairs = yes (as grompp read them)" in text
    assert "| 3 | 1.2500 | 0.2988 | 0.0 |" in text
    assert "| 6-7-9-10 | 3 | 0.5858 | 180.0 | dihedraltypes |" in text


def test_two_bonds_follow_the_run_status_order_with_the_combined_rmse(tmp_path):
    _model(tmp_path)
    _write(
        tmp_path / "00_input" / "selection.json",
        _selection([((1, 3, 4, 5), 69.0), ((25, 1, 3, 4), 67.0)]),
    )
    _bond(tmp_path / "torsion_1-3-4-5", (1, 3, 4, 5))
    _bond(tmp_path / "torsion_25-1-3-4", (25, 1, 3, 4), rmse=(2.0, 1.0))
    _write(
        tmp_path / "final" / "run_status.json",
        {
            "charges": "skipped",
            "torsions": [
                {
                    "quad_1_based": [1, 3, 4, 5],
                    "dir": "torsion_1-3-4-5",
                    "status": "complete",
                    "failed_stage": None,
                },
                {
                    "quad_1_based": [25, 1, 3, 4],
                    "dir": "torsion_25-1-3-4",
                    "status": "complete",
                    "failed_stage": None,
                },
            ],
            "combined": "complete",
        },
    )
    _write(
        tmp_path / "final" / "combined.json",
        {
            "bonds": [
                {"quad_1_based": [1, 3, 4, 5], "rmse_final": 1.75},
                {"quad_1_based": [25, 1, 3, 4], "rmse_final": 0.9},
            ],
            "warnings": [],
        },
    )
    text = write_report(tmp_path).read_text(encoding="utf-8")
    assert text.index("## Torsion 1-3-4-5") < text.index("## Torsion 25-1-3-4")
    assert "(`torsion_1-3-4-5/`)" in text
    assert "| 3.000 | 1.500 | 1.750 | complete |" in text
    assert "| 2.000 | 1.000 | 0.900 | complete |" in text


def test_warnings_are_grouped_by_torsion_and_collapsed(tmp_path):
    _model(tmp_path)
    same = "atom name mismatch at atom 3"
    _bond(
        tmp_path,
        (6, 7, 9, 10),
        warnings={"scan": [same], "mmscan": [same, "grompp: note"], "fit": [same]},
    )
    text = write_report(tmp_path).read_text(encoding="utf-8")
    section = text[text.index("## Warnings") : text.index("## Validity")]
    assert section.count(same) == 1
    assert f"(scan, mmscan, fit) {same}" in section
    assert "(mmscan) grompp: note" in section


def test_a_failed_bond_says_where_it_stopped(tmp_path):
    _model(tmp_path)
    _bond(tmp_path / "torsion_1-3-4-5", (1, 3, 4, 5))
    _bond(tmp_path / "torsion_25-1-3-4", (25, 1, 3, 4), stages=2)
    _write(
        tmp_path / "final" / "run_status.json",
        {
            "charges": "skipped",
            "torsions": [
                {
                    "quad_1_based": [1, 3, 4, 5],
                    "dir": "torsion_1-3-4-5",
                    "status": "complete",
                    "failed_stage": None,
                },
                {
                    "quad_1_based": [25, 1, 3, 4],
                    "dir": "torsion_25-1-3-4",
                    "status": "failed",
                    "failed_stage": "fit",
                },
            ],
            "combined": "not run",
        },
    )
    text = write_report(tmp_path).read_text(encoding="utf-8")
    assert "failed (fit)" in text
    assert "- Fit: not run" in text


def test_without_run_status_the_status_is_inferred(tmp_path):
    _model(tmp_path)
    _bond(tmp_path, (6, 7, 9, 10), stages=1)
    text = write_report(tmp_path).read_text(encoding="utf-8")
    assert "stopped after scan" in text
    assert "- MM energies: not run" in text


def test_a_malformed_payload_is_an_error(tmp_path):
    _model(tmp_path)
    _bond(tmp_path, (6, 7, 9, 10))
    (tmp_path / "03_fit" / "fit.json").write_text("{ not json")
    with pytest.raises(ValidationError, match=r"fit\.json is not valid JSON"):
        write_report(tmp_path)


def test_report_tolerates_payloads_from_an_older_build(tmp_path):
    """Review focus 2: no warnings, torsion_parameters, defaults or charges_json keys."""
    _model(tmp_path, fudgeQQ=None, fudgeLJ=None)
    _bond(tmp_path, (6, 7, 9, 10), defaults=False)
    for rel in (
        "01_qm_scan/scan.json",
        "02_mm_scan/energies.json",
        "03_fit/fit.json",
        "final/validation.json",
    ):
        path = tmp_path / rel
        data = json.loads(path.read_text())
        for key in ("warnings", "torsion_parameters", "charges_json", "defaults"):
            data.pop(key, None)
        path.write_text(json.dumps(data))
    text = write_report(tmp_path).read_text(encoding="utf-8")
    assert "not recorded" in text
    assert "fudgeQQ and fudgeLJ not recorded" in text


def test_the_charges_section_names_changed_atoms_penalties_and_provenance(tmp_path):
    _model(tmp_path)
    _write(tmp_path / "00_input" / "selection.json", _selection([((6, 7, 9, 10), 50.0)]))
    _write(
        tmp_path / "04_charges" / "charges.json",
        {
            "natoms": 26,
            "target_total": 0,
            "rrms_written": 0.21,
            "rrms_fitted": 0.18,
            "rrms_before": 0.35,
            "atoms": [
                {
                    "nr": 1,
                    "name": "C1",
                    "charge_before": 0.789,
                    "charge_after": 0.701,
                    "source": "fitted",
                },
                {
                    "nr": 2,
                    "name": "O2",
                    "charge_before": -0.537,
                    "charge_after": -0.537,
                    "source": "unchanged",
                },
            ],
            "provenance": {
                "scheme": "chelpg",
                "geometry": "xtb",
                "mode": "selected",
                "selected_atoms": list(range(1, 24)),
                "orca_esp_keywords": "! HF 6-31G*",
                "orca_version": "Program Version 6.1.1  -  RELEASE   -",
            },
            "warnings": [],
        },
    )
    _bond(
        tmp_path,
        (6, 7, 9, 10),
        charges_refined=True,
        charges_json="04_charges/charges.json",
    )
    text = write_report(tmp_path).read_text(encoding="utf-8")
    assert "## Charges" in text
    assert "| 1 | C1 | +0.7890 | +0.7010 | fitted | 25.000 |" in text
    assert "1 of 26 atoms changed" in text
    assert "Program Version 6.1.1" in text
    assert "**Caveat.**" in text  # 23 of 26 selected: most of the molecule
    assert "fitted against the refined charges recorded in `04_charges/charges.json`" in text


def test_penalised_but_not_refit_lists_rings_and_other_parameters(tmp_path):
    _model(tmp_path)
    _write(tmp_path / "00_input" / "selection.json", _selection([((6, 7, 9, 10), 50.0)]))
    _bond(tmp_path, (6, 7, 9, 10))
    text = write_report(tmp_path).read_text(encoding="utf-8")
    assert "## Penalised by CGenFF but not refit" in text
    assert "Ring bond 10-16 (penalty 60)" in text
    assert "angle CG312 CG312 CG312: penalty 55" in text


# --- final review I1: admit only the payloads this run produced -----------


def _status(torsions, charges="not run", combined="not applicable"):
    return {"charges": charges, "torsions": torsions, "combined": combined}


def _torsion_row(quad, status, failed_stage=None, dirname="."):
    return {
        "quad_1_based": list(quad),
        "dir": dirname,
        "status": status,
        "failed_stage": failed_stage,
    }


def test_charges_json_is_ignored_when_status_says_skipped(tmp_path):
    _model(tmp_path)
    _write(tmp_path / "00_input" / "selection.json", _selection([((6, 7, 9, 10), 50.0)]))
    _write(
        tmp_path / "04_charges" / "charges.json",
        {
            "natoms": 26,
            "target_total": 0,
            "rrms_written": 0.21,
            "rrms_fitted": 0.18,
            "rrms_before": 0.35,
            "atoms": [
                {
                    "nr": 1,
                    "name": "C1",
                    "charge_before": 0.789,
                    "charge_after": 0.701,
                    "source": "fitted",
                }
            ],
            "provenance": {"scheme": "chelpg", "geometry": "xtb", "mode": "selected"},
            "warnings": [],
        },
    )
    _bond(tmp_path, (6, 7, 9, 10))
    _write(
        tmp_path / "final" / "run_status.json",
        _status([_torsion_row((6, 7, 9, 10), "complete")], charges="skipped"),
    )
    text = write_report(tmp_path).read_text(encoding="utf-8")
    assert "Not refined (`--no-charges`), although CGenFF flags" in text
    assert "RRMS" not in text
    assert "+0.7890" not in text


def test_a_failed_bonds_stale_later_payloads_are_not_admitted(tmp_path):
    _model(tmp_path)
    _bond(tmp_path, (6, 7, 9, 10))  # writes all four stages
    _write(
        tmp_path / "final" / "run_status.json",
        _status([_torsion_row((6, 7, 9, 10), "failed", failed_stage="fit")]),
    )
    text = write_report(tmp_path).read_text(encoding="utf-8")
    assert "failed (fit)" in text
    assert "- Fit: not run" in text
    assert "- Validation: not run" in text
    assert "1.2500" not in text  # the stale fit.json's k must not leak
    assert "| 3.000 | 1.500 |" not in text  # the stale validation.json's RMSEs


def test_a_not_reached_bonds_stale_payloads_are_not_admitted(tmp_path):
    _model(tmp_path)
    _bond(tmp_path, (6, 7, 9, 10))  # every stage's payload physically present
    _write(
        tmp_path / "final" / "run_status.json",
        _status([_torsion_row((6, 7, 9, 10), "not reached")]),
    )
    text = write_report(tmp_path).read_text(encoding="utf-8")
    assert "- QM scan: not run" in text
    assert "- MM energies: not run" in text
    assert "1.2500" not in text
    assert "| 3.000 | 1.500 |" not in text


def test_combined_json_is_ignored_when_status_says_not_applicable(tmp_path):
    _model(tmp_path)
    _bond(tmp_path, (6, 7, 9, 10))
    _write(
        tmp_path / "final" / "combined.json",
        {"bonds": [{"quad_1_based": [6, 7, 9, 10], "rmse_final": 0.01}], "warnings": []},
    )
    _write(
        tmp_path / "final" / "run_status.json",
        _status([_torsion_row((6, 7, 9, 10), "complete")]),
    )
    text = write_report(tmp_path).read_text(encoding="utf-8")
    assert "0.010" not in text
    assert "| 3.000 | 1.500 | n/a | complete |" in text


def test_validity_names_the_charges_json_a_bond_actually_used(tmp_path):
    _model(tmp_path)
    _bond(tmp_path, (6, 7, 9, 10), charges_refined=True, charges_json="/elsewhere/charges.json")
    text = write_report(tmp_path).read_text(encoding="utf-8")
    assert "fitted against the refined charges recorded in `/elsewhere/charges.json`" in text


def test_validity_omits_the_charges_line_when_no_bond_used_refined_charges(tmp_path):
    _model(tmp_path)
    _write(tmp_path / "00_input" / "selection.json", _selection([((6, 7, 9, 10), 50.0)]))
    _write(
        tmp_path / "04_charges" / "charges.json",
        {
            "natoms": 26,
            "target_total": 0,
            "rrms_written": 0.1,
            "rrms_fitted": 0.1,
            "rrms_before": 0.1,
            "atoms": [],
            "provenance": {
                "scheme": "chelpg",
                "geometry": "xtb",
                "mode": "selected",
                "selected_atoms": [1],
            },
            "warnings": [],
        },
    )
    _bond(tmp_path, (6, 7, 9, 10))  # charges_refined defaults to False
    text = write_report(tmp_path).read_text(encoding="utf-8")
    assert "fitted against the refined charges" not in text


# --- minors: nothing-flagged charges, validity needs a fitted bond, ------
# --- a "complete" bond that did not improve ------------------------------


def test_charges_section_says_so_when_nothing_is_flagged(tmp_path):
    _model(tmp_path)
    _write(
        tmp_path / "00_input" / "selection.json",
        {
            "str": "/data/mol.str",
            "thresholds": {"charge": 10.0, "dihedral": 50.0},
            "atoms": [
                {"nr": 1, "top_name": "C1", "type": "CG2O2", "penalty": 2.0, "flagged": False},
            ],
            "bonds": [],
        },
    )
    text = write_report(tmp_path).read_text(encoding="utf-8")
    assert "## Charges" in text
    assert "No atom's CGenFF charge penalty reached 10; charges were not refined." in text


def test_validity_is_left_out_when_no_bond_was_fitted(tmp_path):
    _model(tmp_path)
    text = write_report(tmp_path).read_text(encoding="utf-8")
    assert "## Validity" not in text


def test_summary_marks_a_complete_bond_that_did_not_improve(tmp_path):
    _model(tmp_path)
    _bond(tmp_path, (6, 7, 9, 10), rmse=(109.0, 126.4))
    _write(
        tmp_path / "final" / "run_status.json",
        _status([_torsion_row((6, 7, 9, 10), "complete")]),
    )
    text = write_report(tmp_path).read_text(encoding="utf-8")
    assert "| complete, not improved |" in text
    assert "RMSE against QM 109.000 -> 126.400 kJ/mol (NOT improved)" in text
