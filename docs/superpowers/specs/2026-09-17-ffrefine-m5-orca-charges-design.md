# Design — milestone 5: `ffrefine charges`, ESP-derived partial charges via ORCA

Status: approved in brainstorming, 2026-09-17.
Implements spec section 5. Follows milestone 4 (`ffrefine scan`), merged to master.

---

## 1. Purpose and scope

Spec section 5 replaces or adjusts a topology's partial charges with charges fitted to the
gas-phase electrostatic potential computed by ORCA. Two schemes: CHELPG, where ORCA does the
fitting, and RESP, where ORCA supplies the potential and ffrefine does the fitting.

### In scope

- `ffrefine charges`, replacing the `cmd_not_implemented` stub.
- `--charge-geom {input,xtb}`: use the input geometry, or optimise first with
  `xtb --opt tight --gfn 2`.
- `--charge-scheme {chelpg,resp}`, CHELPG the default.
- Equivalent-hydrogen detection and enforcement (spec 5.3).
- `--charge-mode {selected,all}` with residual redistribution, rounding compensation and the
  CGenFF warning (spec 5.3).
- `04_charges/charges.json` and a charge-refined topology.
- Chaining into `ffrefine run` ahead of the torsion stages.

### Out of scope

- `--charge-geom orca`. The spec names no level of theory for it, and it is the slowest
  option. Deferred rather than invented.
- `--conformers` multi-conformer pooled fitting. It only means anything through our own grid
  fit, so it cannot work for CHELPG, where ORCA owns the fit and each conformer would be
  fitted separately. Deferred until after RESP exists.
- `--str` penalty parsing and penalty-driven atom selection. That is milestone 6. Until it
  lands, `selected` mode takes its atoms from `--charge-atoms` only.
- Molecule-wide topological symmetry. See section 6.1.

### CLI surface

New flags on `charges`, and on `run`, which chains it:

| Flag | Default | Notes |
|---|---|---|
| `--charge-scheme {chelpg,resp}` | `chelpg` | |
| `--charge-geom {input,xtb}` | `xtb` | |
| `--charge-mode {selected,all}` | `selected` | `selected` with no `--charge-atoms` is refused |
| `--charge-atoms i,j,k` | none | 1-based, repeatable |
| `--orca-esp-keywords` | `! HF 6-31G*` | level of theory only |
| `--esp-density` | `1.0` | MK grid points per square Angstrom; RESP only |

`--charge` and `--uhf` already exist from milestone 4 and are reused, with the ORCA
multiplicity taken as `uhf + 1`.

Two deliberate departures from the spec's wording, both so that one flag means one thing:

- Spec 5.2 gives the default keyword line as `! HF 6-31G* CHELPG`. Here
  `--orca-esp-keywords` carries the **level of theory only** and the scheme appends `CHELPG`
  itself. Otherwise overriding the level silently disables the scheme, and the failure would
  surface as a missing output file rather than as a wrong flag.
- Spec section 2 names `--multiplicity`. Milestone 4 already shipped `--uhf`, and carrying
  both would be two names for one quantity, so `--uhf` stands.

---

## 2. Environment findings

All probes were run on 2026-09-17 against the installed ORCA 6.1.1, in an **interactive**
WSL shell (`wsl -e bash -ic`), because ORCA's PATH entry comes from `~/.bashrc` and a login
shell does not see it.

### CHELPG is cheap

`! HF 6-31G* CHELPG` on the 14-atom butane fixture: **2.7 s wall, serial**. Cheap enough for
a `needs_orca` test on every run.

### ORCA writes the charges to a parseable file

The run leaves `<base>.chelpg.xyz`:

```
14
# ORCA CHELPG point charges xyz file
C       -1.926000     0.265000     0.000000  -0.40576138
H       -2.032000     0.897000    -0.884000   0.08817793
```

Element, Angstrom coordinates, charge. **Prefer this over scraping the stdout table**, which
carries six decimals to this file's eight, and whose surrounding banner is free to change
between ORCA versions.

Reference values for the butane fixture, for the regression test:
C1 -0.405761, C4 -0.409052, total -0.000000.

### `orca_vpot` invocation, established rather than assumed

Spec 5.2 says to verify this and not hardcode an argument order. Usage text:

```
orca_vpot GBWName PName XYZName POTName [BaseName of DensityContainer]
```

Probed three variants against the density from the CHELPG job:

| Variant | Result |
|---|---|
| `orca_vpot esp.gbw esp.scfp grid.xyz out.vpot` | **works** |
| `orca_vpot esp.gbw scfp grid.xyz out.vpot esp` | segfault, exit 139 |
| `orca_vpot esp.gbw esp.scfp grid.xyz out.vpot esp` | works |

So the density name is the **full `<base>.scfp`**, not the bare `scfp` the usage text
suggests, and the fifth argument is unnecessary. The values reproduce the potential ORCA
itself computed to about ten significant figures, which is what makes this trustworthy: it
is checked against ORCA's own number, not merely "it ran".

A wrong density name **segfaults** rather than exiting cleanly, so the driver must verify the
output file exists and holds the number of points that were sent.

### The two `.vpot` files have opposite column orders

This is the trap most likely to produce a plausible, wrong answer.

- `<base>.vpot`, written by the CHELPG run: header `natoms npoints`, then the atom
  coordinates, then grid lines of **`V x y z`**.
- `out.vpot`, written by `orca_vpot`: header `npoints`, then lines of **`x y z V`**.

Same extension, reversed order. Reading one with the other's parser yields coordinates that
look like potentials and vice versa, and the fit would still produce numbers.

Both files are in bohr, with the potential in Hartree/e.

---

## 3. Architecture

The repo already separates driving an external program from deciding something;
`qm/scan.py`'s docstring states the rule outright. This milestone keeps that line, because the
RESP fit is the part most likely to be wrong in a way no smoke test notices, and it must be
testable on synthetic data with no ORCA present.

New package `ffrefine/charges/`, mirroring the shape of `ffrefine/fit/`. **Nothing in it runs
a subprocess.**

| Module | Responsibility |
|---|---|
| `grid.py` | Merz-Singh-Kollman shell grid from coordinates and vdW radii |
| `resp.py` | Two-stage restrained fit; returns charges and the ESP fit RRMS |
| `equivalence.py` | Equivalent-hydrogen groups, from the topology's bonds |
| `apply.py` | `selected`/`all`, redistribution, rounding, exact total |

Extended, process-facing:

- `qm/orca.py` — the ESP job, CHELPG parsing, the `orca_vpot` driver.
- `qm/xtb.py` — `optimize_geometry` for `--charge-geom xtb`, verifying xtb *reports*
  convergence rather than trusting exit 0, as `run_scan` verifies its scan block.
- `io/topwriter.py` — `write_charged_topology`, line-preserving.
- `units.py` — bohr conversions. `orca_vpot` speaks bohr and Hartree/e, and spec 9.8 requires
  conversions to live in one place.
- `cli.py` — `cmd_charges`, the flags, and the `run` chaining.

---

## 4. Data flow

```
-s structure, -p topology
  |
  +-- geometry      --charge-geom input -> coordinates as given
  |                 --charge-geom xtb   -> xtb --opt tight --gfn 2 -> 04_charges/opt/
  |
  +-- one ORCA ESP job -> esp.gbw, esp.densities, esp.chelpg.xyz, esp.vpot
  |                       (the SCF density is stored inside esp.densities under
  |                        the name esp.scfp; there is no esp.scfp file)
  |
  +-- chelpg -> parse esp.chelpg.xyz                       -> raw charges
  |   resp   -> grid.py builds the MK grid (bohr)
  |             orca_vpot esp.gbw esp.scfp grid.xyz out.vpot
  |               (esp.scfp here names the density inside esp.densities)
  |             resp.py two-stage fit                      -> raw charges
  |
  +-- equivalence.py  equal charges within each equivalent-hydrogen group
  +-- apply.py        selected/all, redistribution, rounding, exact total
  |
  +-- 04_charges/charges.json
  +-- 04_charges/molecule_charged.itp
```

The ESP job runs **once** and serves both schemes. RESP needs its density regardless, and
CHELPG costs nothing on top of it.

Before it runs, every output a later step reads (`esp.chelpg.xyz`, `esp.vpot`, `esp.gbw`,
`esp.densities`, `esp.densitiesinfo`) is deleted, and afterwards the run is refused unless
ORCA's stdout says `ORCA TERMINATED NORMALLY` and does not say `SCF NOT CONVERGED`. ORCA
6.1.1 exits 0 when its SCF gives up, and `04_charges/` is reused by a re-run into the same
outdir, so an existence check alone would hand back the previous run's charges under the new
run's keyword line.

Because that job also leaves `esp.vpot` — ORCA's own grid carrying the QM potential — the ESP
fit **RRMS is reported for both schemes**, computed against the same reference grid. Spec 5.2
asks for RRMS only under RESP, but a user choosing between the two schemes wants a comparable
number, and here it is free.

`charges.json` records, per atom: `nr`, `name`, `element`, `charge_before`, `charge_after`,
and `source` (`fitted`, `redistributed`, `unchanged`); plus both totals, the RRMS, and
provenance: scheme, level of theory, geometry source, ORCA version, grid settings, the radii
used, the restraint constants, and the equivalence groups.

---

## 5. Contract with the existing pipeline

### Charges are fitted before the torsion, not after

Spec section 3 lists `run` as scan, mmscan, fit, charges, validate. **This design puts
charges first**, and the reason is physical rather than cosmetic: the fitted torsion absorbs
whatever the 1-4 electrostatics do not already account for, and those depend on the charges.
Fitting the torsion and then changing the charges leaves a topology whose torsion was fitted
against charges it no longer carries — exactly the convention-binding the README already
warns about. Charges first is also what ffTK and the Amber workflows do.

`run` therefore chains:

```
charges -> scan -> mmscan -> fit -> validate
```

The QM scan does not depend on MM charges, so only mmscan, fit and validate are affected.

### The refined topology reaches later stages by path, not by magic

`charges` writes `04_charges/molecule_charged.itp`. `run` then points `args.topology` at that
file for the remaining stages. Stages invoked on their own use `-p` exactly as given; to work
step by step, the user passes the charged file, and `charges` prints the exact command.

Two alternatives were rejected. Letting later stages auto-discover `04_charges/` makes `-p`
stop meaning what it says and lets a stale charges output silently change a standalone
mmscan. Threading a `--charges-from charges.json` through every stage requires all four to
learn to apply charge edits, for no gain over swapping a path.

`--resume` needs no new code: every stage already hashes `args.topology`, so pointing it at a
different file recomputes exactly the stages that depend on it.

### `run` without charge work still works

`run` chains `charges` only when charge work was requested (`--charge-atoms`, or
`--charge-mode all`). Otherwise it reports charges as skipped, the way it already reports
milestone 5, and today's `ffrefine run` on butane is unaffected.

---

## 6. The decisions that carry real risk

### 6.1 Equivalence is local, and global symmetry is refused rather than faked

Spec 5.3 says "same heavy-atom parent, same atom type, same connectivity depth", which can be
read more than one way. **This design fixes the reading: two hydrogens are equivalent exactly
when they are bonded to the same heavy atom and share an atom type.**

CHELPG applies it by averaging within each group, which preserves the group's sum and so
cannot disturb the molecular total. RESP applies it as equality constraints in stage 2.

Molecule-wide topological symmetry is **not** imposed. Butane's C1 and C4 are topologically
identical, and the CHELPG probe gave them -0.405761 and -0.409052. That 0.0033 gap is ESP
fitting noise. Removing it needs graph automorphism detection over the bond graph, which is a
larger piece of work than this milestone, and guessing at it is worse than reporting it. The
asymmetry is recorded in `charges.json`, not silently averaged away.

### 6.2 The MK grid must be deterministic

Four shells at 1.4, 1.6, 1.8 and 2.0 times the vdW radius; roughly one point per square
Angstrom (`--esp-density`, default 1.0); a point discarded when it lies within
`scale * r_j` of any other atom j.

Points are placed by a **golden-spiral construction, never a random draw**. Randomly placed
points would make the fitted charges differ between two runs on identical input, which
destroys reproducibility and leaves the grid tests unpinnable.

Radii are the Merz-Kollman values (H 1.20, C 1.50, N 1.50, O 1.40, F 1.35, P 1.80, S 1.75,
Cl 1.70 Angstrom), with Bondi values for Br (1.85) and I (1.98), which the MK set does not
define. **An element with no radius is refused, never defaulted.** A guessed radius moves the
fitting surface and biases every charge in the molecule, with nothing in the output to show
it happened. The table used is recorded in `charges.json`.

### 6.3 The RESP fit

Atomic units throughout: coordinates in bohr, potential in Hartree/e.

- Design matrix `A_jk = sum_i 1/(r_ij * r_ik)`, right-hand side `B_j = sum_i V_i / r_ij`.
- Total charge enforced by a Lagrange multiplier, as a bordered linear system.
- Hyperbolic restraint `a * sum_j (sqrt(q_j^2 + b^2) - b)` over non-hydrogen atoms, entering
  the diagonal as `a / sqrt(q_j^2 + b^2)` and re-linearised each iteration from the previous
  charges, until the largest charge movement falls below 1e-8 e. Non-convergence within the
  iteration cap is refused, not returned.
- Stage 1: `a = 0.0005`, `b = 0.1`, no equivalence constraints.
- Stage 2: `a = 0.001`, `b = 0.1`; only the equivalence groups and their parent carbons are
  free, every other charge frozen at its stage 1 value; equality constraints per group.
- `RRMS = sqrt(sum_i (V_i - V_fit_i)^2 / sum_i V_i^2)`.

### 6.4 `selected` mode redistributes per charge group, per spec 5.3

All atoms are fitted; the new charges are taken for the selected atoms only. Per CHARMM
conventions, the residual is absorbed separately by each charge group: for each group
containing a selected atom, that group's residual (the charge change the fit made within it)
is spread evenly over the unselected atoms in that group, falling back to unselected heavy
atoms bonded to a selected atom in that group, and refused when neither exists. This keeps
each charge group's net charge intact — that is what CHARMM charge groups are for. After all
per-group residuals are absorbed, any remaining global offset (e.g. from an explicit
`--charge` differing from the topology's sum) is spread over the pooled recipients. A nonzero
global offset with no recipients anywhere (every selected group has zero residual and none
has a recipient) is refused rather than silently misplaced — leaving the molecular total
wrong would be worse than stopping. Every reassignment is logged atom by atom.

The alternative, freezing the unselected atoms inside the fit as ffTK does, is better physics
and was considered and rejected for this milestone: spec 5.3 specifies redistribution, and
freezing is impossible for CHELPG, where ORCA owns the fit. Recorded in section 10.

### 6.5 Rounding leaves the total exact

Charges are written to four decimals. After rounding, any residual drift is absorbed by the
largest-magnitude atom among the redistribution recipients (when there are any); otherwise the
largest-magnitude atom overall. This ensures an atom reported as `source="fitted"` carries
exactly the fitted value: drift compensation is kept off fitted atoms when there is anywhere
else to put it. The final total is asserted to be within 1e-6 of the target integer. Four
decimals makes every written charge a multiple of 1e-4, so the compensated total is exact
rather than merely close. (The two compensation strategies yield identical totals to 1e-6,
differing only in which atom bears the rounding error.)

This compensation step is only allowed to absorb a rounding-sized drift — at most about
`natoms * 0.5 * 10**-decimals` — and this bound is checked in every mode, including `all`.
Anything larger is refused rather than folded in, because it means an offset was never placed
onto a recipient upstream; `all` mode has no legitimate large drift either, since the ESP job
runs at `charge=target_total` so the fitted charges already sum to it.

The target total comes from `derive_charge` (milestone 4), which reads the topology's own
`[ atoms ]` sum and refuses one that is not near an integer.

---

## 7. Error handling

Every avoidable ambiguity is a refusal with a named remedy.

| Condition | Behaviour |
|---|---|
| `selected` with no `--charge-atoms` | Refuse; name both `--charge-atoms` and `--charge-mode all` |
| `--charge-atoms` index out of range | Refuse, naming the index and the atom count |
| Element with no vdW radius | Refuse, naming the element and atom |
| `orca_vpot` produced no output, or the wrong point count | Refuse, naming the gbw and the density |
| RESP did not converge | Refuse, with the iteration count and last movement |
| xtb optimisation did not report convergence | Refuse, as `run_scan` does for its scan block |
| ORCA nonzero exit, or no CHELPG file | Refuse, with the tail of the output |
| ORCA stdout lacks `ORCA TERMINATED NORMALLY`, or says `SCF NOT CONVERGED` (ORCA exits 0 then) | Refuse, naming the condition and `esp.stdout` |
| `--charge-mode all` on a CGenFF topology | Warn, per spec 5.3, into the log and `charges.json` |
| RRMS above 0.5 | Warn. A poor fit is a judgement about the molecule, not a malfunction |

---

## 8. Testing

The reason `grid.py` and `resp.py` run no subprocesses is that their numerics get tested with
no ORCA installed.

**Without ORCA:**

- **RESP round-trip.** Generate an ESP from a known set of charges, fit it back with the
  restraint disabled, require the originals to 1e-6. This is the charge analogue of the
  project's existing round-trip refit, and it is what catches a wrong sign or a transposed
  design matrix, errors that still produce plausible numbers.
- **Restraint and constraints.** The restraint pulls heavy-atom charges toward zero; the
  equivalence constraint makes a group exactly equal; the total-charge constraint is honoured
  exactly; non-convergence is refused.
- **Grid.** Point count scales with r^2 and with density; no surviving point lies inside
  another atom's shell; identical input gives identical points.
- **Equivalence.** Butane yields the methyl and methylene groups; a hydrogen with a different
  atom type is not grouped with its siblings.
- **Apply.** Redistribution lands in the right charge group, falls back to bonded heavy atoms,
  and leaves the total exact to 1e-6 after rounding; `all` mode warns.
- **Writer.** Only the charge column of changed atoms differs; every other line stays
  byte-identical, as the existing topology writers are already tested.
- **Parsers.** Against fixtures committed from the probe runs in section 2, so CHELPG and both
  vpot layouts are covered with no ORCA present.
- **CLI.** `--dry-run` writes nothing; `selected` with no atoms is refused; `run` skips
  charges when none was requested, and swaps `-p` when it does not.

**With ORCA (`needs_orca`):** CHELPG and RESP end to end on butane; charges summing to zero;
methyl hydrogens equal to 1e-6 after equivalence; RRMS reported for both. The CHELPG
regression is pinned against the measured C1 -0.405761 and C4 -0.409052.

A skipped test is not a passing one: these must be run in an interactive WSL shell, and the
count of executing `needs_orca` tests checked, not just the pass count.

---

## 9. One in-scope cleanup

`_load_inputs` refuses to run without `--dihedral`, and `_write_input_stage` requires a quad.
`charges` needs no torsion, so as things stand it would demand a flag it never uses. Split
`_load_inputs` into loading the structure and topology, and resolving the target quad, and let
`00_input/model.json` omit the torsion set when there is none. This is a contained change to
existing code, not a parallel path around it.

---

## 10. Open questions carried into planning

1. **Frozen-atom fitting for RESP selected mode** (section 6.4). Better physics than
   redistribution and natural in the Lagrange framework, but a deviation from spec 5.3 and
   impossible for CHELPG. Revisit once the fitting machinery exists.
2. **Molecule-wide symmetry** (section 6.1). Butane's C1 and C4 differ by 0.0033 e. If that
   proves to matter in practice, graph automorphism detection is its own piece of work.
3. **Parallelism for the ESP job.** Milestone 4 confirmed ORCA runs at `nprocs 2` on this
   machine despite the OpenMPI 4.1.8 against 4.1.6 mismatch. The ESP job should pass
   `--nthreads` through the same way, but CHELPG on butane takes 2.7 s serial, so if parallel
   proves unreliable on a larger molecule, pinning it to serial costs little.
