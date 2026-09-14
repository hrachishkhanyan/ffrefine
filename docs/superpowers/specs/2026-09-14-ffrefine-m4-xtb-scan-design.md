# Design — milestone 4: `ffrefine scan`, the xtb relaxed torsion scan driver

Date: 2026-09-14
Status: approved, ready for implementation planning
Binding authority: `ffrefine_spec.md` sections 4.3, 6, 8, 9, 10
Predecessor: `docs/superpowers/plans/2026-09-04-ffrefine-milestones-1-3.md` (milestones 1-3,
complete at commit `2985ca9`)

## 1. Purpose and scope

Milestones 1-3 built everything downstream of the QM scan: topology parsing, the GROMACS
energy engine, the spec 4.5 self-check, the fit, the outputs and the validation stage. The
one thing they could not supply is the QM torsion profile itself. Today the README's worked
example instructs the user to **hand-write** `01_qm_scan/scan.json`.

This milestone builds `ffrefine scan`, which produces that file from a relaxed xtb torsion
scan, closing the pipeline end to end.

### In scope

- xcontrol `$constrain`/`$scan` input generation, with the scan syntax verified against the
  installed xtb at runtime rather than assumed (spec 4.3).
- xtb invocation and `xtbscan.log` parsing.
- Dihedral angles measured from the returned coordinates, never from the requested grid
  (spec 9.5).
- Bidirectional scanning with hysteresis reporting (spec 4.3).
- Discontinuity detection: conformational hops and intramolecular hydrogen bonds (spec 4.3).
- `--freeze-dihedral a,b,c,d` (repeatable) for holding other rotatable bonds.
- `--sp-refine`: optional ORCA single points on the xtb geometries, replacing the xtb
  energies (spec 4.3), as the final task.
- `01_qm_scan/` stage outputs: `scan.json`, `scan.pdb`, xtb inputs and logs, manifest.

### Out of scope

- `--mm-relax` (spec 4.6). GROMACS-side, independent of this driver, and its own piece.
- The charges module (spec section 5, milestone 5).
- CGenFF `.str` parsing, auto-selection and `final/report.md` (milestone 6).

## 2. Environment findings

These were established by direct probe on 2026-09-14, not assumed. They are recorded
because two of them contradict a reasonable first guess.

- **xtb 6.7.1** and **GROMACS 2025.4** live in the WSL conda env `gromacs`.
- **ORCA 6.1.1** (`orca` and `orca_vpot`) lives at
  `/mnt/c/Users/hrach/Downloads/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg/` — a Linux
  build under the Windows Downloads folder — and is put on PATH by `~/.bashrc`.
- Because that PATH entry comes from `~/.bashrc`, a **login** shell (`bash -lc`) does not see
  ORCA at all. Use `bash -ic` with `conda activate gromacs`, which has all four binaries.
- ORCA is a **shared** build: it will not start unless its own directory is on
  `LD_LIBRARY_PATH`. This is exactly what the existing `Binaries.orca_libdir` field is for.
- ORCA's MPI targets OpenMPI 4.1.8; the system has 4.1.6. Serial execution is **confirmed
  working** (H2 HF/STO-3G returned `FINAL SINGLE POINT ENERGY -1.116759307204`). Parallel is
  untested, which matters for spec section 8's `--nthreads` -> `%pal nprocs` mapping.

### The `$scan` syntax probe

The spec says to verify the scan block against the installed version rather than assume it.
A live 4-point butane scan on 6.7.1 accepted the spec's literal syntax and exited 0 in 13 s:

```
$constrain
  force constant=1.0
  dihedral: 1,5,8,11,auto
$scan
  1: 0.0,360.0,4
$end
```

`xtbscan.log` is a plain multi-frame XYZ in **Angstrom**, each frame preceded by a comment
line of the form `energy: -13.665127531525 xtb: 6.7.1 (edcfbbe)`.

**The measured angles were not the requested grid.** Requesting `0.0,360.0,4` (i.e. 0/120/
240/360) produced measured dihedrals of 180.00, 120.00, -119.99 and -0.01 degrees. The
chemistry is correct — the 20.69 kJ/mol syn barrier sits inside the README's stated 14-20
kJ/mol expectation and the two gauche points agree to 0.001 kJ/mol — but the first point is
180 degrees, not 0. This is a live demonstration of why spec 9.5 requires angles to be
measured from coordinates. **No code may derive `phi_deg` from the requested grid.**

### The reverse-scan probe

A second probe settled how the reverse leg is expressed. Starting from the final forward
geometry with a **descending** range, `1: 360.0,0.0,4`, xtb accepted the block and exited 0
with the expected 4 frames. So the reverse leg is a descending `$scan` range on the forward
scan's last structure; no geometry reversal trick is needed.

The two legs' measured angles, from the same probe:

```
forward:  180.00   120.00  -119.99   -0.01     degrees
reverse:   -0.01  -119.99   119.99    0.01     degrees
```

This result is load-bearing for section 6.1 and is discussed there.

Timing: 4 points took 13 s, so the default 24-point grid is roughly 80 s and doubles under
bidirectional scanning. This governs the test strategy in section 8.

## 3. Architecture

A new `ffrefine/qm/` package, splitting tool-driving from analysis the way `ffrefine/mm/`
already splits `gromacs.py` (drives the tool) from `selfcheck.py` / `validate_fit.py` (the
policy that interprets it).

| Module | Responsibility | Subprocess? |
|---|---|---|
| `qm/xtb.py` | xcontrol input generation, xtb invocation, `xtbscan.log` parsing | yes |
| `qm/scan.py` | angle measurement, bidirectional merge, hysteresis, discontinuity detection | **no** |
| `qm/orca.py` | `--sp-refine` single points, ORCA output parsing | yes |

The analysis half never touches a subprocess. Every merge, hysteresis, hop and H-bond rule
is therefore testable on captured and synthetic data with no external tool present, which is
what makes the test strategy in section 8 possible.

`qm/orca.py` also gives milestone 5's charge module a home to inherit: binary discovery,
`LD_LIBRARY_PATH` handling and output parsing are built once here.

## 4. Data flow

```
structure (.pdb/.gro/.xyz) + topology (.itp/.top) + --dihedral i,j,k,l
  -> io/validate.py               existing: atom order, elements, names
  -> qm.xtb.write_scan_input      scan.inp   ($constrain / $scan xcontrol)
  -> io.structure.write_xyz       mol.xyz    (Angstrom, topology order)  [new shared helper]
  -> qm.xtb.run_scan              xtbscan.log       forward
  -> qm.xtb.run_scan              xtbscan_rev.log   reverse, from last forward frame
  -> qm.xtb.parse_scan_log        frames (Angstrom) + energies (Hartree)
  -> qm.scan.measure_angles       phi_deg, from coordinates only
  -> qm.scan.merge_bidirectional  lower energy per grid point + hysteresis report
  -> qm.scan.find_discontinuities hops + hydrogen bonds
  -> qm.orca.refine_energies      optional --sp-refine; replaces energy_kj
  -> 01_qm_scan/scan.json, scan.pdb, manifest
```

## 5. Contract with the existing pipeline

`scan.json` must keep the schema `cmd_mmscan`, `cmd_fit` and `cmd_validate` already consume:

| Key | Meaning |
|---|---|
| `energy_kj` | QM energy per frame, kJ/mol |
| `frames_nm` | `(nframes, natoms, 3)` coordinates in **nanometres**, topology atom order |
| `phi_deg` | measured dihedral per frame, degrees |

Two keys are added:

- `quad`: the 1-based target quadruplet, matching what `energies.json` and `fit.json` already
  record since the P1-3 fix, so `mmscan` can cross-check that it is scanning the torsion the
  scan was run for.
- `provenance`: xtb version string, the exact argv, charge, `uhf`, GFN level, step, the
  bidirectional flag, maximum observed hysteresis, the discontinuity findings, and the
  `--sp-refine` level when used.

Adding keys is safe: the staleness digest (`hash_array`) covers `frames_nm` only, by the
deliberate design recorded in `cmd_fit`'s docstring. The provenance block follows the P2-11
precedent that every setting which shaped a number gets recorded rather than being left on
stdout.

Once this ships, the README's "Supplying a QM scan" section changes from an instruction to
hand-write JSON into a description of what `ffrefine scan` produces.

## 6. The two decisions that carry real risk

### 6.1 Forward and reverse points pair by measured angle, never by index

Spec 4.3: "forward from the input geometry, then reverse starting from the final forward
structure. Take the lower energy at each grid point."

Two independent relaxed scans do not land on identical angles, and the section 2 probe shows
this is not a theoretical worry. The two legs' measured angles were:

```
forward:  180.00   120.00  -119.99   -0.01     degrees
reverse:   -0.01  -119.99   119.99    0.01     degrees
```

Pairing these by array index would pair **180 degrees with 0 degrees** — the anti minimum
against the syn barrier — and then "take the lower energy at each grid point" would write the
anti minimum's energy onto the syn point, **erasing the barrier the scan exists to measure**.
Three of the four index pairings are wrong. Reversed-index pairing does not rescue it either:
the reverse leg visits 0 degrees twice and never reaches 180 degrees at all, so the two legs
do not even cover the same set of angles.

So the merge pairs points by **wrap-aware nearest measured angle within an explicit
tolerance** (default half a step). Points with no partner inside the tolerance are reported,
never silently dropped and never silently paired.

This is non-negotiable because pairing by array position is the precise shape of defect P1-3,
where stage payloads bound by array length produced three different mispairings that each
exited 0 with a confident wrong answer. Here the same mistake would flatten a torsion barrier
while exiting 0.

Hysteresis: report `max |E_forward - E_reverse|` over paired points and warn prominently when
it exceeds `--hysteresis-tol` (default 4 kJ/mol), naming the worst angle.

### 6.2 Hop detection is a robust outlier test, not a fixed threshold

Adjacent frames differ by the scanned dihedral **by construction**, so the heavy-atom Kabsch
RMSD between them is never zero and an absolute cutoff carries no meaning. A genuine
conformational hop appears as a *spike* against the typical step.

So: compute the aligned heavy-atom RMSD between adjacent frames using the existing
`geometry.kabsch_rmsd`, then flag frames where the value exceeds `median + k * MAD` (k
default 5, exposed as a tunable). Report frame indices and their measured angles.

### 6.3 Hydrogen-bond detection

Why it belongs here: the scan is *relaxed*, so at each grid point everything except the
scanned dihedral is free to rearrange. In any molecule carrying both a donor (O-H, N-H) and
an acceptor (N, O, F), a hydroxyl can rotate mid-scan into an intramolecular hydrogen bond —
1,2-ethanediol scanning O-C-C-O is the canonical case, where the gauche region gains roughly
10-20 kJ/mol of stabilisation the anti region does not have. That well is **not** the
torsional potential. Fitting a cosine series through it bakes an electrostatic interaction
into the dihedral term, which the nonbonded terms then count a second time in any simulation
that uses the result.

Detection uses **topology bonds** (available in this stage) to identify donors — hydrogens
bonded to N/O/F — and acceptors (N/O/F), then a geometric criterion: H...acceptor under 2.5 A
with a donor-H...acceptor angle above 120 degrees, excluding 1-2 and 1-3 bonded pairs.
Report hydrogen bonds that appear or disappear between adjacent frames.

Per spec 4.3 the tool **reports and continues** — it does not reject the scan or correct the
profile. The user can then re-scan with `--freeze-dihedral`, choose a different starting
conformer, or knowingly accept it.

Butane, the only fixture, has no donors or acceptors and will correctly report none. Section
8 explains how this is nonetheless genuinely tested.

## 7. Error handling

Every boundary fails loudly, following the patterns milestones 1-3 established (ruling R29:
no traceback for bad input; all errors derive from `FFRefineError`).

| Condition | Behaviour |
|---|---|
| xtb not found | `Binaries.require("xtb")` — names the override flag and config key |
| xtb nonzero exit | `ExternalToolError` via `run_command(check=True)` |
| `$scan` block rejected | **explicit check**, not assumption: confirm the relaxed scan ran and assert the returned frame count matches the requested points; raise `ExternalToolError` naming the rejected block |
| frame count mismatch | `ExternalToolError` |
| topology charge sum not near an integer | `ValidationError` naming the sum and the tolerance |
| structure/topology mismatch | existing `io/validate.py` |
| `--sp-refine` without ORCA | `Binaries.require("orca")`, with `orca_libdir` defaulting to the ORCA binary's own directory |
| ORCA fails or prints no final energy | `ExternalToolError` naming the frame |

### Charge and multiplicity

Spec 4.3 runs xtb with `--chrg Q --uhf M` but never says where Q and M come from. Decided:

- **Q** is derived from the topology: the sum of `[ atoms ]` charges, rounded to the nearest
  integer, refusing with a `ValidationError` when the sum is not within a small tolerance of
  an integer. Butane's fixture sums to exactly 0. An explicit `--charge` overrides.
- **M** cannot be derived from a topology, so `--uhf` is a flag defaulting to 0 (closed
  shell). A parity guess from the electron count is explicitly rejected: it cannot
  distinguish a triplet from a singlet diradical, and would be a confident wrong answer.

Both are recorded in `scan.json`'s provenance block.

## 8. Testing

Three tiers, mirroring how `needs_gmx` works today.

**Captured (no external tool).** Commit the real 4-frame `xtbscan.log` from the section 2
probe as a fixture. Covers log parsing, energy extraction, Angstrom->nm conversion, and angle
measurement — including a direct regression test that the measured angles are 180/120/-120/0
and *not* the requested 0/120/240/360.

**Synthetic (no external tool).** Hand-built frames covering bidirectional merge, the
angle-pairing tolerance and its refusal path, hysteresis reporting, hop detection against a
planted spike, and hydrogen-bond detection. The H-bond detector reads coordinates, elements
and a bond list, so a hand-built frame pair where a hydroxyl rotates into and out of range is
a genuine test of the real function — no new topology fixture required.

**Live.** `needs_xtb`: a 4-6 point real scan end to end producing a valid `scan.json`, plus a
test that the `$scan` syntax is genuinely accepted by the installed xtb (the spec's runtime
verification requirement, as a test rather than as prose). `needs_orca`: `--sp-refine` with
`--orca-sp-keywords "! HF STO-3G"`; the real wB97X-D3/def2-TZVP default is 24 frames of heavy
DFT and belongs nowhere near a test suite.

Markers `needs_xtb` and `needs_orca` are declared in `pyproject.toml` **only**, with skipping
`xtb_bin` / `orca_bin` fixtures in `conftest.py` — mirroring `gmx_bin` and respecting ruling
R1, which forbids registering a marker in both `pyproject.toml` and `pytest_configure`.

Verification runs in the WSL env that has all four binaries:

```bash
wsl.exe -e bash -ic 'conda activate gromacs && cd /mnt/c/Users/hrach/Projects/ffrefine && \
  python -m pytest -q && python -m pytest -q -m needs_gmx && python -m black --check .'
```

Baseline to preserve: **327 passed, 0 skipped, 16 `needs_gmx` executing, `black --check`
clean.**

## 9. Two small in-scope cleanups

1. **Move `ANGSTROM_TO_NM` from `io/structure.py` into `units.py`.** `units.py`'s own
   docstring states "No other module in ffrefine may define a conversion constant", and
   `io/structure.py` currently contradicts it. M4 converts Angstrom to nm on every frame of
   every scan, so leaving it would compound a stated-contract violation. Re-export from
   `io/structure.py` if that keeps the diff small.
2. **Add `io.structure.write_xyz`.** `qm/xtb.py` must write `mol.xyz` for xtb, and
   `tests/conftest.py`'s `butane_xyz` fixture already hand-rolls the same format. One shared
   writer, used by both, rather than a third copy.

Both are small, both are in the path of this work, and neither is speculative refactoring.

## 10. Open questions carried into planning

- Whether `--nthreads` should map to xtb's `-P` only, or also to ORCA's `%pal nprocs` given
  the untested OpenMPI 4.1.6-vs-4.1.8 mismatch. Defaulting `--sp-refine` to serial ORCA until
  parallel is demonstrated is the safe call.
