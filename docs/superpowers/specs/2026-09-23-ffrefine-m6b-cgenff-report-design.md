# Design — milestone 6b: CGenFF penalties, automatic selection, multi-bond runs, `report.md`

Status: approved in brainstorming, 2026-09-23.
Implements spec build-order item 6 ("CGenFF `.str` penalty parsing, auto-selection, `report.md`
generation"), spec sections 2, 5.3 and 6. Follows milestone 6a (typed torsions, `#include`d
topologies), merged to master at 7663608.

---

## 1. Purpose and scope

Given a molecule's CGenFF output — the GROMACS `.top`/`.pdb` and the `.str` stream file —
`ffrefine run --str mol.str` decides from CGenFF's own penalties which charges and which
torsions need work, refits them, and writes one topology carrying every change plus a
`report.md` that stands on its own.

Success on the committed fixtures:

- `ffrefine run -s pfoa_gmx.pdb -p pfoa_gmx.top --str pfoa.str` refines PFOA's 23 flagged
  charges, refits bond 3-4 then bond 1-3, and writes `final/molecule_refined.itp` carrying both
  fitted series and a complete `final/report.md`.
- `ffrefine run -s capecitabine_gmx.pdb -p capecitabine_gmx.top --str capecitabine.str`
  refits the one torsion CGenFF penalises at 50 (`6,7,9,10`) and refines no charges.

### Decisions taken in brainstorming

| Question | Decision |
|---|---|
| Several flagged torsions | refit each, one subfolder per central bond; charges once, shared; one report (taken before milestone 6a) |
| How bonds interact | **sequential**: each bond is fitted against the topology already carrying the earlier bonds' refits; a final combined check re-runs every bond's scan with the final topology |
| Threshold comparison | **inclusive**: flagged at penalty >= threshold, for charges and dihedrals alike |
| Charges from `--str` | **automatic with opt-out**: `run` refines the flagged atoms unless `--no-charges`; warns when they are most of the molecule |
| Parked follow-ups | fix the ones this milestone builds on: R26, R27, R28, R29, M7/R11, and R9 promoted to a warning |
| Approach | **stages per bond**: `run` re-invokes the existing stage commands per bond; `report.md` is built only from the JSON payloads on disk |

### In scope

- `--str`, `--penalty-threshold`, `--no-charges`.
- `ffrefine/io/cgenff.py` (parser), `ffrefine/selection.py` (mapping and selection),
  `ffrefine/report.py` (report generator).
- Multi-bond `run`, the combined check, the combined final topology.
- A `warnings` list in every stage payload; xtb and ORCA versions recorded.
- `00_input/selection.json`, `final/combined.json`, `final/report.md`.
- The prerequisite fixes in section 8.
- README: the new flags, the multi-bond layout, a `--str` worked example, and the R11/R12
  corrections.

### Out of scope

- Refitting bond, angle or improper parameters. Their penalties are reported, never acted on;
  impropers are never modified (spec 9.4).
- A simultaneous fit across bonds.
- EGCG as a committed fixture.
- The cosmetic parked items M2 (rounding placement), M5 (RESP stage-2 element rule), M8 (vpot
  echo check).
- Narrowing `scan`'s `--resume` key to what xtb reads (section 5).

---

## 2. Environment findings

Established, not assumed.

### ORCA exits 0 with an unconverged SCF under `IgnoreConv`

ORCA 6.1.1, water at HF/6-31G* with `%scf IgnoreConv true MaxIter 1 end`: exit status 0,
`****ORCA TERMINATED NORMALLY****`, the lines `WARNING: Found SCFConvIgnored == true` and
`Warning (ORCA_LEANSCF): the SCF did not converge, but SCFConvIgnored is TRUE.`, and **no**
`SCF NOT CONVERGED` — the only marker the ESP job checks today — and **no**
`SCF CONVERGED AFTER N CYCLES`. A converged job (H2 at HF/STO-3G) prints
`*           SCF CONVERGED AFTER   1 CYCLES          *`. Both print
`Program Version 6.1.1  -  RELEASE   -`. So the robust check is positive: require the
`SCF CONVERGED AFTER` line (section 8, R29).

### The fixtures under the default thresholds

| | PFOA | capecitabine |
|---|---|---|
| atoms | 26 | 47 |
| charge penalty >= 10 | 23 atoms (all but `.top` 13, 14, 26: F4, F5 at 9.978, H at 6.830) | none (max 5.422) |
| dihedral types >= 50 | `CG2O2 CG312 CG312 CG312` 69; `CG2O2 CG312 CG312 FGA2` 67.2; `OG311 CG2O2 CG312 CG312` 67; `OG311 CG2O2 CG312 FGA2` 55 | `OG302 CG2O6 NG2S1 CG2R64` 50 |
| bonds | 3-4 (69), 1-3 (67) | 7-9 (50) |
| representative quads | `1,3,4,5`, then `25,1,3,4` | `6,7,9,10` |
| RESI line | `RESI ***** 0.000 ! param penalty= 69.000 ; charge penalty= 41.068` | `... param penalty= 50.000 ; charge penalty= 5.422` |

The CGenFF export renames atoms (`.str` `O1` is `.top` atom 2 `O2`); order, types and
charges agree index for index. Charges in both files carry three decimals.

---

## 3. Architecture

```
ffrefine/io/cgenff.py     NEW  pure text
    StrAtom(index, name, atype, charge, penalty)
    StrParameter(kind, types, penalty)        kind: bond | angle | dihedral | improper
    StrFile(atoms, parameters, program_version, parameter_version, source)
    read_str(path) -> StrFile

ffrefine/selection.py     NEW  pure; topology + StrFile + thresholds -> decisions
    Thresholds(charge=10.0, dihedral=50.0)
    check_mapping(topology, strfile) -> list[str]           warnings; raises on mismatch
    flagged_atoms(topology, strfile, thresholds) -> FlaggedAtoms
    flagged_bonds(topology, strfile, thresholds) -> FlaggedBonds
    Selection(...)            everything selection.json records

ffrefine/report.py        NEW  pure; reads payload files, writes markdown
    write_report(outdir) -> Path

ffrefine/cli.py           --str, --penalty-threshold, --no-charges; the per-bond loop in run;
                          the combined check; selection.json; warnings in every payload;
                          write_report at the end of run and validate
ffrefine/mm/validate_fit.py   compare two topologies on one scan's frames (combined check)
ffrefine/qm/xtb.py, qm/orca.py   record program versions; R29
ffrefine/io/topwriter.py  R28 provenance line; M7 strip
ffrefine/mm/tprparams.py  R9 warning
ffrefine/io/topology.py   R26: CHARGE_TOLERANCE and derive_charge move here
```

`selection.py` sits at the package top level, beside `stages.py`: it combines the topology
model with the `.str` model and belongs to neither `io/` nor `mm/`. `report.py` likewise.

---

## 4. Reading the `.str` and choosing

### Parsing (`io/cgenff.py`)

- The topology section runs from `read rtf card` to its `END`. `ATOM name type charge !
  penalty ...` gives one atom, numbered in file order from 1. `RESI`, `GROUP`, `BOND`, `IMPR`
  and `!RING` lines are not needed and are skipped; the `RESI` line's two penalties are kept
  for the report header.
- The parameter section runs from `read param card` to its `END`. Inside `BONDS`, `ANGLES`,
  `DIHEDRALS` and `IMPROPERS`, each line yields its leading 2, 3, 4 or 4 atom types and the
  number after `penalty=` in its comment. A line with no `penalty=` has penalty 0. A dihedral
  written over several multiplicity lines is one `StrParameter` with the highest penalty. Other
  sections are skipped.
- The header's `CHARMM General Force Field (CGenFF) program version X` and `For use with
  CGenFF topology and parameter files version Y` give the versions, when present.
- A file with no `ATOM` line, or an `ATOM` / parameter line that does not parse, is a
  `ParseError` naming the file and line.

### Mapping onto the topology

- The atom counts must be equal, and `.str` atom *i*'s type must equal `[ atoms ]` entry *i*'s
  type for every *i*; otherwise `ValidationError`, listing up to the first five mismatches and
  saying that the `.str` probably belongs to another molecule or another atom order. Names are
  never compared.
- A charge differing from the `.str`'s by more than 0.0015 is a warning naming the atoms.

### Flagged atoms

An atom is flagged when its charge penalty is >= the charge threshold. The flagged set is
extended to whole equivalent-hydrogen groups (`charges.equivalence`), each extension logged
and recorded. `--charge-atoms` replaces the flagged set entirely.

### Flagged bonds and their quadruplets

1. A dihedral `StrParameter` is flagged when its penalty is >= the dihedral threshold.
2. It matches every proper `[ dihedrals ]` line whose four atom types equal its types, forward
   or reversed.
3. The central bond of each matched line is a candidate, carrying the highest penalty among
   the flagged types it matched.
4. A candidate bond inside a ring (removing it leaves its two atoms connected) is dropped and
   recorded as "penalised, in a ring, not refit".
5. A flagged type matching no line is a warning ("the `.str` may belong to another molecule").
6. Each bond's quadruplet: among its matched lines, the one whose type has the highest
   penalty; then one whose two outer atoms are both heavy (not hydrogen); then the lowest
   `(i, l)` indices. Written in the orientation of its `[ dihedrals ]` line.
7. Bonds run by penalty, highest first; ties by the lower of the two central atom indices, then
   the higher.

Explicit `--dihedral` flags replace the automatic choice and run in the order given; `--str`
still drives the charges and the report's penalty tables.

### `00_input/selection.json`

Records: the `.str` path and versions; the thresholds; every atom (`.top` number and name,
`.str` name, type, charge, penalty, flagged, added-by-equivalence); every `StrParameter` with
its penalty, whether flagged, and the topology quads it matched; each selected bond with its
penalty, matched types and quadruplet; the ring exclusions; the unmatched flagged types; the
mapping warnings; and its own `warnings` list. A copy of the `.str` is written beside it.

---

## 5. How `run` drives the stages

1. **Selection.** With `--str`: parse, map, select, write `00_input/selection.json`.
2. **Charges.** Run when atoms are flagged, `--charge-atoms` is given, or `--charge-mode all`,
   unless `--no-charges`. Once, into `04_charges/`, as today. When the replaced atoms are more
   than half the molecule, a warning says that this is close to a wholesale replacement with
   gas-phase ESP charges, with the same caveat spec 5.3 attaches to `--charge-mode all`.
3. **Torsions.**
   - **One torsion:** today's flow, in `--outdir` itself.
   - **Two or more:** for each bond, a copy of the arguments with `--outdir` =
     `outdir/torsion_i-j-k-l/` (the quad), `--dihedral` = that quad only, and `-p` = the
     previous bond's `final/molecule_refined.itp` (the first bond: the charge-refined topology,
     or the user's). The same `scan -> mmscan -> fit -> validate` commands run unchanged. The
     include-base marker (milestone 6a) and the charge provenance line (section 8, R28) ride
     along the chain; each subfolder's `model.json` names the user's original input.
4. **Combined check** (two or more bonds): for each bond, its scan frames and QM energies from
   its own `01_qm_scan/scan.json` are re-run through GROMACS with the user's original topology
   and with the final one. `final/combined.json` records, per bond, RMSE against QM with each,
   beside that bond's own before/after from its `validation.json`; one plot per bond under
   `final/plots/`. A bond no better with the final topology than with the original draws the
   usual "did not improve" warning. Its `--resume` key covers every bond's `scan.json`, the
   final topology and the preprocessed digests of both topologies.
5. **Final topology** (two or more bonds): `final/molecule_refined.itp` is a byte copy of the
   last bond's refined topology.
6. **Report.** `final/report.md`, at the end of `run` and of `validate` (so each bond subfolder
   has a single-bond report too).

### Layout, two or more torsions

```
outdir/
  00_input/            input copies, the .str copy, model.json, selection.json
  04_charges/          when charges run
  torsion_1-3-4-5/     00_input/ 01_qm_scan/ 02_mm_scan/ 03_fit/ final/   (today's layout)
  torsion_25-1-3-4/    ...
  final/               molecule_refined.itp, combined.json, plots/, report.md
  run.log
```

### Resume, failure, standalone commands

- Stages resume per bond as today. A changed fit for bond 1 changes bond 2's input topology,
  so bond 2's stages rerun, its xtb scan included (the scan's key covers the whole topology
  file though xtb reads only the total charge; narrowing it is out of scope).
- `run` stops at the first failing stage and exits nonzero; `report.md` is still written,
  marking each bond complete, failed at a named stage, or not reached.
- Standalone `scan`/`mmscan`/`fit`/`validate` with `--str` and no `--dihedral` use the one
  flagged bond when there is exactly one, and refuse, pointing to `run`, when there are more.
  Two or more `--dihedral` flags are refused the same way. `charges --str` refines the flagged
  atoms.

---

## 6. Warnings that last, and `report.md`

### Warnings

Each stage payload gets a `warnings` list collected by the existing `_collect_warnings` hook:
`model.json`, `scan.json`, `energies.json`, `fit.json`, `validation.json`, `charges.json`
(already), `selection.json`, `combined.json`. A stage skipped by `--resume` keeps the
warnings of the run that produced it. `run.log` keeps everything, as before.

`scan.json` records the xtb version (from xtb's banner) and, with `--sp-refine`, the ORCA
version; `charges.json` records the ORCA version (`Program Version X`).

### `report.md`

Plain Markdown, tables with units, built only from payloads:

1. **Header:** molecule, date, ffrefine version, input files, CGenFF program and parameter
   versions and the `RESI` line's penalties.
2. **Summary:** one row per bond — quad with `.top` names and types, CGenFF penalty, RMSE
   against QM before and after its own refit and after everything (combined check), status.
3. **Charges** (when refined): scheme, geometry, ORCA level and version; each changed atom
   before -> after with its penalty; the three RRMS values; the all-mode / most-of-the-molecule
   caveat when it applies.
4. **Per bond:** xtb level and version, step, bidirectional, `--sp-refine` keywords; hops and
   hydrogen-bond changes; the CGenFF terms removed with their source (`energies.json`
   `torsion_parameters`); the fitted series (multiplicity, k, phase), `--distribute` or not; the
   self-check residual; links to its plots.
5. **Penalised but not refit:** ring bonds; flagged types that matched nothing; bond, angle and
   improper penalties >= the dihedral threshold.
6. **Warnings:** every payload's warnings, grouped by bond and stage, duplicates collapsed and
   counted.
7. **Validity:** "The fitted torsion parameters are valid only with the same nonbonded
   parameters and the same 1-4 conventions as the topology they were fitted against:
   fudgeQQ = X, fudgeLJ = Y, gen-pairs = Z (as grompp read them)." When charges were refined:
   the torsions were fitted against those charges (named by their `charges.json`) and must ship
   with them.
8. **Files:** the final topology, the `.prm` blocks, the plots.

A missing payload reads "not run"; a payload that exists but does not parse is a
`ValidationError` naming the file.

---

## 7. CLI surface

- `--str PATH` on `run`, `charges`, `scan`, `mmscan`, `fit`, `validate`.
- `--penalty-threshold KIND=VALUE`, repeatable, `KIND` in `charge`, `dihedral`, `VALUE` a
  non-negative number; defaults `charge=10`, `dihedral=50`. Anything else is an argparse usage
  error (exit 2).
- `--no-charges` on `run`. With `--charge-atoms` or `--charge-mode all` it is refused.
- No other new flag.

---

## 8. Prerequisite fixes (the first tasks)

- **R26 layering.** `CHARGE_TOLERANCE` and `derive_charge` move from `qm/xtb.py` to
  `io/topology.py`; `qm.xtb` re-exports them. An AST test fails if a module under
  `ffrefine/charges/` imports `subprocess`, `ffrefine.runner` or anything under `ffrefine.qm`.
- **R27 write order.** `cmd_charges` computes the three RRMS values (reading `esp.vpot`) before
  writing `molecule_charged.itp`; a failed read leaves no charged topology.
- **R28 charge provenance.** `write_charged_topology` writes one comment line,
  `; ffrefine charges: <absolute path of charges.json>`, replacing any earlier one. The refit
  writers copy source lines verbatim, so every later topology in a chain carries it.
  `validation.json`'s `charges_refined` becomes "the before-topology carries this line", and a
  new `charges_json` field records the path. True across outdirs and across bonds.
- **R29 SCF convergence.** The ESP job and `--sp-refine`'s single points require
  `SCF CONVERGED AFTER` in ORCA's output; its absence is an `ExternalToolError` explaining
  that the SCF did not converge (and that `IgnoreConv` hides this from ORCA's exit status). The
  existing `SCF NOT CONVERGED` check stays.
- **M7 / R11.** `write_charged_topology` strips `[ system ]` and `[ molecules ]` like the other
  writers, so every topology ffrefine writes is a molecule include file; the README's kept-file
  wording (R11) becomes true, and its `--resume` advice (R12) is corrected.
- **R9.** A typed line resolving to no terms is a warning — not an INFO line — when the source
  topology's `[ dihedrals ]` section contains an `#ifdef`, `#ifndef` or `#else`; the warning
  says the refit may be written onto a line grompp does not read.

---

## 9. Error handling

- Malformed `.str`: `ParseError` naming file and line.
- Mapping mismatch (count, type): `ValidationError` listing the first mismatches.
- Charge differences against the `.str`: warning.
- Malformed `--penalty-threshold`: argparse usage error.
- `--no-charges` with `--charge-atoms` / `--charge-mode all`: `ValidationError`.
- No dihedral flagged and no `--dihedral`: `run` refines flagged charges if any, then says that
  no torsion was flagged, naming the highest dihedral penalty found and
  `--penalty-threshold dihedral=`; exit 0 with a report. Nothing flagged at all: the same, and
  the report says so.
- Standalone torsion stages with several flagged bonds, or two or more `--dihedral`:
  `ValidationError` pointing to `run`.
- Report: missing payload "not run"; malformed payload `ValidationError`.
- Combined-check GROMACS failures propagate like any stage's.

---

## 10. Testing

### Unit, no binaries

- **Parser:** both committed `.str` files (PFOA 26 atoms, capecitabine 47, `!RING` skipped);
  highest penalty over multiplicity lines; Urey-Bradley angle lines; impropers; versions;
  malformed variants.
- **Selection:** PFOA -> bonds `(1,3,4,5)` then `(25,1,3,4)`, 23 atoms; capecitabine ->
  `(6,7,9,10)` at exactly 50, nothing at `dihedral=51`; ring exclusion (capecitabine at
  `dihedral=0.5`); type and count refusals; charge-difference warning; equivalent-hydrogen
  extension; `--dihedral` and `--charge-atoms` overrides.
- **Report:** synthetic payload trees — one bond, two bonds, a missing stage, a failed bond —
  checked for the validity statement with fudge values, QM level strings, penalty table,
  collapsed warnings, charge provenance.
- **Warnings** persisted in each payload.
- **`run` orchestration** with stage commands stubbed: order, per-bond outdir and topology
  chaining, charges once, `--no-charges`, stop at first failure with the report still written.
- **Each prerequisite fix:** R26 AST test; R27 orphan; R28 line survives the chain and
  `charges_refined` is true from another outdir; R29 converged / `IgnoreConv` / not-converged
  outputs; M7 strip; R9 warning only with an `#ifdef`.

### Live

- `needs_gmx` + `needs_xtb`: `run --str` on capecitabine (one bond, today's layout, no ORCA;
  `report.md` shows penalty 50 and the validity statement).
- `needs_gmx` + `needs_xtb`: `run --str --no-charges` on PFOA (two subfolders in order; bond 2's
  input topology is bond 1's refined one; the final topology carries both fitted series;
  `combined.json` has both bonds; `report.md` complete).
- `needs_orca`: `charges --str` on PFOA (23 atoms, the most-of-the-molecule warning, ORCA
  version recorded); an ESP job with `IgnoreConv MaxIter 1` on water is refused.
- Every existing test keeps passing, 0 skipped.

---

## 11. Open questions carried into planning

- Whether the combined check's plots overlay QM, the original topology and the final topology
  on one axis per bond (proposed) or reuse `plot_validation` as is.
- Where `run` records which bonds completed, for the report's status column when a bond fails
  before writing any payload: a small `final/run_status.json` written by `run` as it goes is
  the proposal.
