"""``final/report.md``: what a run changed, readable on its own (design section 6).

Built only from the JSON payloads the stages leave on disk -- never from a
run's memory -- so a fresh run, a resumed one and a standalone ``validate``
all produce the same report. A payload that does not exist yet reads "not
run"; one that exists but does not parse is an error. Every key is read with
``.get``: a payload written by an older ffrefine degrades to "not recorded".

Pure: reads files, writes one file, starts no process.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .errors import FFRefineError, ValidationError
from .io.topology import read_topology

NOT_RECORDED = "not recorded"
NA = "n/a"

STAGES: tuple[tuple[str, str], ...] = (
    ("scan", "01_qm_scan/scan.json"),
    ("mmscan", "02_mm_scan/energies.json"),
    ("fit", "03_fit/fit.json"),
    ("validate", "final/validation.json"),
)

VALIDITY = (
    "The fitted torsion parameters are valid only with the same nonbonded parameters and the "
    "same 1-4 conventions as the topology they were fitted against: {conventions}. Change "
    "either and the torsion no longer reproduces the scan it was fitted to."
)

CHARGE_CAVEAT = (
    "Most of the molecule's charges were replaced. CGenFF charges are fitted to water "
    "interaction energies and dipoles, not to the gas-phase electrostatic potential, so "
    "replacing them with ESP charges is a change of derivation scheme and can degrade "
    "condensed-phase behaviour (spec section 5.3)."
)


@dataclass
class _Bond:
    label: str
    root: Path
    quad: list[int] | None
    payloads: dict[str, dict | None]
    status: str
    failed_stage: str | None


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValidationError(
            f"{path} is not valid JSON ({exc}); report.md cannot be written from it"
        ) from exc
    if not isinstance(data, dict):
        raise ValidationError(f"{path} does not hold a JSON object; report.md cannot use it")
    return data


def _num(value, digits: int = 3) -> str:
    return NA if value is None else f"{float(value):.{digits}f}"


def _dash(quad) -> str:
    return "-".join(str(n) for n in quad)


def _dir_order(path: Path):
    try:
        return (0, [int(n) for n in path.name.split("_", 1)[1].split("-")])
    except (IndexError, ValueError):
        return (1, [path.name])


def _quad_of(payloads: dict) -> list[int] | None:
    candidates = (
        (payloads["validate"] or {}).get("quad_1_based"),
        ((payloads["fit"] or {}).get("provenance") or {}).get("quad_1_based"),
        (payloads["mmscan"] or {}).get("quad"),
        (payloads["scan"] or {}).get("quad"),
    )
    return next((list(q) for q in candidates if q), None)


def _infer_status(payloads: dict) -> tuple[str, str | None]:
    if payloads["validate"] is not None:
        return "complete", None
    done = [stage for stage, _ in STAGES if payloads[stage] is not None]
    return ("not run", None) if not done else (f"stopped after {done[-1]}", None)


def _admitted_payloads(root: Path, state: str, failed_stage: str | None) -> dict[str, dict | None]:
    """Which stage payloads a status row permits ``_bonds`` to load (final review I1).

    A stage payload can survive on disk from an earlier run even though this
    run's own status says the bond stopped earlier or failed; the report must
    not show it. ``"complete"`` admits every payload. ``"failed"`` admits only
    the stages strictly before ``failed_stage`` (in :data:`STAGES` order, so a
    ``failed_stage`` this build does not recognise admits none). Anything else
    (``"not reached"``, ``"running"``, ...) admits none.
    """
    names = [stage for stage, _ in STAGES]
    if state == "complete":
        allowed = set(names)
    elif state == "failed" and failed_stage in names:
        allowed = set(names[: names.index(failed_stage)])
    else:
        allowed = set()
    return {stage: (_load(root / rel) if stage in allowed else None) for stage, rel in STAGES}


def _bonds(outdir: Path, status: dict | None) -> list[_Bond]:
    rows: list[tuple] = []
    if status is not None:
        # The run's own list, even when empty: a run that refit nothing must
        # not fall back to an earlier run's payloads on disk.
        for t in status.get("torsions") or []:
            label = t.get("dir") or "."
            root = outdir if label == "." else outdir / label
            rows.append(
                (label, root, t.get("quad_1_based"), t.get("status"), t.get("failed_stage"))
            )
    else:
        subdirs = sorted(outdir.glob("torsion_*"), key=_dir_order)
        if subdirs:
            rows = [(d.name, d, None, None, None) for d in subdirs]
        elif any((outdir / rel).exists() for _, rel in STAGES):
            rows = [(".", outdir, None, None, None)]
    bonds = []
    for label, root, quad, state, failed in rows:
        if state is None:
            payloads = {stage: _load(root / rel) for stage, rel in STAGES}
            state, failed = _infer_status(payloads)
        else:
            payloads = _admitted_payloads(root, state, failed)
        bonds.append(
            _Bond(label, root, list(quad) if quad else _quad_of(payloads), payloads, state, failed)
        )
    return bonds


def _atom_labels(model: dict | None, selection: dict | None) -> dict[int, tuple[str, str]]:
    if selection and selection.get("atoms"):
        return {a["nr"]: (a.get("top_name", ""), a.get("type", "")) for a in selection["atoms"]}
    path = (model or {}).get("topology")
    if not path:
        return {}
    try:
        topology = read_topology(path)
    except FFRefineError:
        return {}
    return {a.nr: (a.name, a.atype) for a in topology.atoms}


def _describe(quad, labels) -> str:
    if quad and all(n in labels for n in quad):
        names = "-".join(labels[n][0] for n in quad)
        types = " ".join(labels[n][1] for n in quad)
        return f"{names} ({types})"
    return NA


def _penalty(quad, selection: dict | None) -> str:
    if not quad or not selection:
        return NA
    for bond in selection.get("bonds", []):
        if list(bond.get("quad_1_based", [])) == list(quad):
            return f"{bond['penalty']:g}"
    return NA


def _status_text(bond: _Bond) -> str:
    return bond.status + (f" ({bond.failed_stage})" if bond.failed_stage else "")


def _summary_status_text(bond: _Bond) -> str:
    """Status text for the Summary table: a "complete" bond that made things worse says so.

    ``_bond_section``'s own verdict line already spells out "NOT improved";
    this is the one line a reader scanning only the Summary table sees
    (final review M5).
    """
    text = _status_text(bond)
    validation = bond.payloads["validate"]
    if bond.status == "complete" and validation is not None and validation.get("improved") is False:
        text += ", not improved"
    return text


def _rel(path: Path, start: Path) -> str:
    return Path(os.path.relpath(path, start)).as_posix()


def _header(model, selection) -> list[str]:
    lines = [
        f"# ffrefine report: {(model or {}).get('molecule', NA)}",
        "",
        f"Written {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by ffrefine "
        f"{__version__}.",
        "",
    ]
    if model:
        lines.append(f"- Structure: `{model.get('structure', NA)}`")
        lines.append(f"- Topology: `{model.get('topology', NA)}`")
    if selection:
        lines.append(
            f"- CGenFF stream file: `{selection.get('str', NA)}` (CGenFF program "
            f"{selection.get('cgenff_program_version') or NA}, parameters "
            f"{selection.get('cgenff_parameter_version') or NA}; highest parameter penalty "
            f"{_num(selection.get('resi_param_penalty'), 1)}, highest charge penalty "
            f"{_num(selection.get('resi_charge_penalty'), 3)})"
        )
        th = selection.get("thresholds") or {}
        lines.append(
            f"- Penalty thresholds: charge >= {th.get('charge', NA):g}, dihedral >= "
            f"{th.get('dihedral', NA):g}"
            if th
            else f"- Penalty thresholds: {NOT_RECORDED}"
        )
    lines.append("")
    return lines


def _summary(bonds, labels, selection, combined) -> list[str]:
    lines = ["## Summary", ""]
    if not bonds:
        reason = " No dihedral reached the penalty threshold." if selection else ""
        return lines + [f"No torsion was refit.{reason}", ""]
    final_rmse = {
        tuple(row.get("quad_1_based", [])): row.get("rmse_final")
        for row in (combined or {}).get("bonds", [])
    }
    lines += [
        "| Torsion | Atoms (types) | CGenFF penalty | RMSE before (kJ/mol) | RMSE after (kJ/mol) "
        "| RMSE with every change (kJ/mol) | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    for bond in bonds:
        validation = bond.payloads["validate"] or {}
        quad = bond.quad
        lines.append(
            f"| {_dash(quad) if quad else NA} | {_describe(quad, labels)} | "
            f"{_penalty(quad, selection)} | "
            f"{_num(validation.get('rmse_before_kj_per_mol'))} | "
            f"{_num(validation.get('rmse_after_kj_per_mol'))} | "
            f"{_num(final_rmse.get(tuple(quad)) if quad else None)} | "
            f"{_summary_status_text(bond)} |"
        )
    lines += [
        "",
        "RMSE is MM against QM over the torsion scan, each profile shifted to its own "
        'minimum. "Before" is the topology the torsion was fitted against; "after" is that '
        "topology with this torsion refit. The next column, filled for runs that refit "
        "several torsions, re-runs the scan with the final topology carrying every change.",
        "",
    ]
    return lines


def _charges_section(charges, selection, status) -> list[str]:
    state = (status or {}).get("charges")
    if not charges:
        if state == "failed":
            return [
                "## Charges",
                "",
                "The charge refit failed; see `04_charges/` and `run.log`.",
                "",
            ]
        flagged = [a for a in (selection or {}).get("atoms", []) if a.get("flagged")]
        if flagged and state == "skipped":
            return [
                "## Charges",
                "",
                f"Not refined (`--no-charges`), although CGenFF flags {len(flagged)} atom(s).",
                "",
            ]
        if selection is not None and not flagged:
            threshold = (selection.get("thresholds") or {}).get("charge", 10.0)
            return [
                "## Charges",
                "",
                f"No atom's CGenFF charge penalty reached {threshold:g}; charges were not "
                "refined.",
                "",
            ]
        return []
    prov = charges.get("provenance") or {}
    lines = [
        "## Charges",
        "",
        f"- Scheme: {prov.get('scheme', NA)}, geometry: {prov.get('geometry', NA)}, mode: "
        f"{prov.get('mode', NA)}",
        f"- ORCA level: `{prov.get('orca_esp_keywords') or NA}` "
        f"({prov.get('orca_version') or NOT_RECORDED})",
        f"- Total charge {charges.get('target_total', NA)}; ESP RRMS of the written charges "
        f"{_num(charges.get('rrms_written'), 4)}, of the fully fitted set "
        f"{_num(charges.get('rrms_fitted'), 4)}, of the input charges "
        f"{_num(charges.get('rrms_before'), 4)}",
        "",
    ]
    penalties = {a["nr"]: a.get("penalty") for a in (selection or {}).get("atoms", [])}
    changed = [a for a in charges.get("atoms", []) if a.get("source") != "unchanged"]
    lines += [
        f"{len(changed)} of {charges.get('natoms', NA)} atoms changed:",
        "",
        "| Atom | Name | Before | After | Why | CGenFF charge penalty |",
        "|---|---|---|---|---|---|",
    ]
    for a in changed:
        lines.append(
            f"| {a['nr']} | {a.get('name', '')} | {a['charge_before']:+.4f} | "
            f"{a['charge_after']:+.4f} | {a.get('source', '')} | {_num(penalties.get(a['nr']))} |"
        )
    natoms = charges.get("natoms") or 0
    selected = prov.get("selected_atoms") or []
    if prov.get("mode") == "all" or (natoms and len(selected) > natoms / 2):
        lines += ["", f"**Caveat.** {CHARGE_CAVEAT}"]
    lines.append("")
    return lines


def _bond_section(bond: _Bond, labels, report_dir: Path) -> list[str]:
    quad = bond.quad
    title = f"## Torsion {_dash(quad) if quad else bond.label}"
    if bond.label != ".":
        title += f" (`{bond.label}/`)"
    lines = [title, "", f"{_describe(quad, labels)}; status: {_status_text(bond)}.", ""]
    scan, energies, fit, validation = (bond.payloads[s] for s, _ in STAGES)

    if scan is None:
        lines.append("- QM scan: not run")
    else:
        prov = scan.get("provenance") or {}
        lines.append(
            f"- QM scan: GFN{prov.get('gfn', '?')}-xTB ({prov.get('xtb_version') or NOT_RECORDED}), "
            f"step {prov.get('step_deg', NA)} degrees, {prov.get('npoints', NA)} points, "
            f"{'bidirectional' if prov.get('bidirectional') else 'one direction'}, total charge "
            f"{prov.get('charge', NA)}, unpaired electrons {prov.get('uhf', NA)}"
        )
        if prov.get("sp_refine_keywords"):
            lines.append(
                f"- Energies refined with ORCA `{prov['sp_refine_keywords']}` "
                f"({prov.get('orca_version') or NOT_RECORDED})"
            )
        hops, hbonds = prov.get("hops") or [], prov.get("hbond_changes") or []
        lines.append(
            f"- Conformational hops: {len(hops)}; hydrogen-bond changes: {len(hbonds)}"
            + (" (see Warnings)" if hops or hbonds else "")
        )

    if energies is None:
        lines.append("- MM energies: not run")
    else:
        lines.append(
            f"- Self-check residual: {energies.get('selfcheck_max_abs_error_kj', NA)} kJ/mol "
            f"(tolerance {energies.get('selfcheck_tolerance_kj', NA)})"
        )
        params = energies.get("torsion_parameters")
        if params is None:
            lines.append(f"- Force-field terms removed about the central bond: {NOT_RECORDED}")
        else:
            lines += [
                "",
                "Force-field terms about the central bond, all zeroed before fitting:",
                "",
                "| Quad | n | k (kJ/mol) | phase (deg) | from |",
                "|---|---|---|---|---|",
            ]
            for p in params:
                lines.append(
                    f"| {_dash(p['quad_1_based'])} | {p['multiplicity']} | "
                    f"{p['k_kj_per_mol']:.4f} | {p['phi0_deg']:.1f} | {p.get('source', NA)} |"
                )
            lines.append("")

    if fit is None:
        lines.append("- Fit: not run")
    else:
        where = (
            "spread over every quadruplet about the bond (--distribute)"
            if (fit.get("provenance") or {}).get("distribute")
            else "on the target quadruplet, the others zeroed"
        )
        lines += [
            "",
            f"Fitted series, {where}; fit RMSE {_num(fit.get('rmse_kj_per_mol'))} kJ/mol:",
            "",
            "| n | k (kJ/mol) | k (kcal/mol) | phase (deg) |",
            "|---|---|---|---|",
        ]
        for t in fit.get("terms", []):
            lines.append(
                f"| {t['multiplicity']} | {t['k_kj_per_mol']:.4f} | "
                f"{t.get('k_kcal_per_mol', 0.0):.4f} | {t['phase_deg']:.1f} |"
            )
        lines.append("")

    if validation is None:
        lines.append("- Validation: not run")
    else:
        verdict = "improved" if validation.get("improved") else "NOT improved"
        lines.append(
            f"- Validation: RMSE against QM {_num(validation.get('rmse_before_kj_per_mol'))} -> "
            f"{_num(validation.get('rmse_after_kj_per_mol'))} kJ/mol ({verdict})"
        )

    plots = sorted((bond.root / "03_fit" / "plots").glob("*.png")) + sorted(
        (bond.root / "final" / "plots").glob("*.png")
    )
    if plots:
        lines.append("- Plots: " + ", ".join(f"[{p.stem}]({_rel(p, report_dir)})" for p in plots))
    lines.append("")
    return lines


def _not_refit(selection) -> list[str]:
    if not selection:
        return []
    threshold = (selection.get("thresholds") or {}).get("dihedral", 50.0)
    rings = selection.get("ring_bonds") or []
    unmatched = selection.get("unmatched_types") or []
    others = [p for p in selection.get("other_parameters") or [] if p["penalty"] >= threshold]
    if not (rings or unmatched or others):
        return []
    lines = ["## Penalised by CGenFF but not refit", ""]
    for r in rings:
        lines.append(
            f"- Ring bond {_dash(r['bond_1_based'])} (penalty {r['penalty']:g}): a relaxed scan "
            "cannot rotate a bond in a ring, so it was not refit."
        )
    for types in unmatched:
        lines.append(f"- Dihedral type {' '.join(types)}: matched no [ dihedrals ] line.")
    for p in others:
        lines.append(
            f"- {p['kind']} {' '.join(p['types'])}: penalty {p['penalty']:g}; ffrefine refits "
            "torsions only."
        )
    lines.append("")
    return lines


def _warnings_section(selection, charges, bonds, combined) -> list[str]:
    sources: list[tuple[str, list[tuple[str, dict]]]] = []
    if selection is not None:
        sources.append(("Selection", [("selection", selection)]))
    if charges is not None:
        sources.append(("Charges", [("charges", charges)]))
    for bond in bonds:
        name = f"Torsion {_dash(bond.quad)}" if bond.quad else f"Torsion {bond.label}"
        sources.append(
            (
                name,
                [
                    (stage, bond.payloads[stage])
                    for stage, _ in STAGES
                    if bond.payloads[stage] is not None
                ],
            )
        )
    if combined is not None:
        sources.append(("Combined check", [("combined", combined)]))

    lines = ["## Warnings", ""]
    written = False
    for title, payloads in sources:
        seen: dict[str, list[str]] = {}
        missing: list[str] = []
        for stage, payload in payloads:
            if "warnings" not in payload:
                missing.append(stage)
                continue
            for message in payload["warnings"]:
                seen.setdefault(message, []).append(stage)
        if not seen and not missing:
            continue
        lines += [f"### {title}", ""]
        for message, stages in seen.items():
            where = ", ".join(
                f"{stage} x{count}" if count > 1 else stage
                for stage, count in Counter(stages).items()
            )
            lines.append(f"- ({where}) {message}")
        if missing:
            lines.append(
                f"- Warnings {NOT_RECORDED} by: {', '.join(missing)} (written by an older ffrefine)."
            )
        lines.append("")
        written = True
    if not written:
        lines += ["No warnings were raised.", ""]
    return lines


def _validity(bonds, model) -> list[str]:
    if not any(b.payloads["fit"] for b in bonds):
        # Nothing fitted; the 1-4-convention caveat is about a fitted torsion
        # (final review M1), so there is nothing for it to be about.
        return []
    defaults = next(
        (
            b.payloads["mmscan"]["defaults"]
            for b in bonds
            if b.payloads["mmscan"] and b.payloads["mmscan"].get("defaults")
        ),
        None,
    )
    if defaults:
        conventions = (
            f"fudgeQQ = {defaults['fudgeQQ']:g}, fudgeLJ = {defaults['fudgeLJ']:g}, gen-pairs = "
            f"{'yes' if defaults.get('gen_pairs') else 'no'} (as grompp read them)"
        )
    elif model and model.get("fudgeQQ") is not None and model.get("fudgeLJ") is not None:
        conventions = (
            f"fudgeQQ = {model['fudgeQQ']:g}, fudgeLJ = {model['fudgeLJ']:g}, gen-pairs = "
            f"{'yes' if model.get('gen_pairs') else 'no'} (from the topology's own [ defaults ])"
        )
    else:
        conventions = (
            f"fudgeQQ and fudgeLJ {NOT_RECORDED}; read your force field's [ defaults ] before "
            "reusing these parameters"
        )
    lines = ["## Validity", "", VALIDITY.format(conventions=conventions)]
    # Driven by the admitted bonds' own validation.json, not by whether a
    # 04_charges/charges.json happens to sit on disk (final review I1): a
    # stale charges.json from an earlier, unrelated run must not make this
    # run claim its torsions depend on charges they were never fitted against.
    refined_charges_json = next(
        (
            b.payloads["validate"]["charges_json"]
            for b in bonds
            if b.payloads["validate"] and b.payloads["validate"].get("charges_refined")
        ),
        None,
    )
    if refined_charges_json:
        lines += [
            "",
            f"The torsions were fitted against the refined charges recorded in "
            f"`{refined_charges_json}`; ship them together.",
        ]
    lines.append("")
    return lines


def _run_completed(bonds, status: dict | None) -> bool:
    """Whether ``final/molecule_refined.itp`` is the finished topology, not a partial one.

    With ``run_status.json``: every torsion is ``"complete"`` and ``combined``
    is ``"complete"`` or ``"not applicable"``. Without it: the single bond's
    own validation payload exists (final review I1.3).
    """
    if status is not None:
        torsions = status.get("torsions") or []
        torsions_ok = bool(torsions) and all(t.get("status") == "complete" for t in torsions)
        return torsions_ok and status.get("combined") in ("complete", "not applicable")
    return len(bonds) == 1 and bonds[0].payloads.get("validate") is not None


def _files(outdir: Path, bonds, report_dir: Path, status: dict | None) -> list[str]:
    lines = ["## Files", ""]
    final_topology = outdir / "final" / "molecule_refined.itp"
    if _run_completed(bonds, status) and final_topology.exists():
        lines.append(
            f"- Refined topology: `{_rel(final_topology, report_dir)}` -- a molecule include file; "
            "`#include` it from your own `.top` beside the force field it names."
        )
    for bond in bonds:
        prm = bond.root / "03_fit" / "dihedral.prm"
        if prm.exists():
            lines.append(f"- CHARMM parameter block: `{_rel(prm, report_dir)}`")
    for name in ("combined.json", "run_status.json"):
        path = outdir / "final" / name
        if path.exists():
            lines.append(f"- `{_rel(path, report_dir)}`")
    for path in sorted((outdir / "final" / "plots").glob("combined_*.png")):
        lines.append(f"- [{path.stem}]({_rel(path, report_dir)})")
    lines.append(f"- Every command and message: `{_rel(outdir / 'run.log', report_dir)}`")
    lines.append("")
    return lines


def write_report(outdir: str | Path) -> Path:
    """Write ``<outdir>/final/report.md`` from the payloads under ``outdir``.

    ``outdir`` is either a run with one torsion (its stage directories at
    the top level) or a run with several (``torsion_*`` subfolders, ordered
    by ``final/run_status.json`` when present).

    Raises:
        ValidationError: a payload exists but is not a valid JSON object.
    """
    outdir = Path(outdir)
    model = _load(outdir / "00_input" / "model.json")
    selection = _load(outdir / "00_input" / "selection.json")
    charges = _load(outdir / "04_charges" / "charges.json")
    combined = _load(outdir / "final" / "combined.json")
    status = _load(outdir / "final" / "run_status.json")
    if status is not None:
        # A run_status.json makes its own claim about what actually completed;
        # a charges.json or combined.json left over from an earlier, different
        # run in the same outdir must not be admitted just because it exists
        # on disk (final review I1.1).
        if status.get("charges") != "complete":
            charges = None
        if status.get("combined") != "complete":
            combined = None
    bonds = _bonds(outdir, status)
    labels = _atom_labels(model, selection)
    report_dir = outdir / "final"

    lines: list[str] = []
    lines += _header(model, selection)
    lines += _summary(bonds, labels, selection, combined)
    lines += _charges_section(charges, selection, status)
    for bond in bonds:
        lines += _bond_section(bond, labels, report_dir)
    lines += _not_refit(selection)
    lines += _warnings_section(selection, charges, bonds, combined)
    lines += _validity(bonds, model)
    lines += _files(outdir, bonds, report_dir, status)

    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path
