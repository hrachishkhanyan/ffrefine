# ffrefine

Repairs high-penalty parameters from generalized force fields (CGenFF, GAFF) by
refitting proper dihedrals against a QM torsion scan, using GROMACS to remove
the existing torsion contribution first.

The idea is simple and the bookkeeping is not. To fit a torsion you need the
molecule's energy profile *without* that torsion's own contribution, so the
tool zeroes every dihedral entry sharing the target central bond, re-runs the
MM energies, and fits the difference against QM. Getting that subtraction wrong
produces a plausible-looking profile and silently wrong parameters, so a
mandatory self-consistency check (spec section 4.5) verifies the edit against
an independent analytic evaluation before any fitting happens, and aborts if it
disagrees by more than 1e-3 kJ/mol.

## Status: milestones 1-5 of 6

| Subcommand | State |
|---|---|
| `ffrefine scan` | works -- relaxed xtb torsion scan, bidirectional by default, reports conformational hops and hydrogen bonding, optional `--sp-refine` |
| `ffrefine mmscan` | works -- MM single points with and without the torsion, plus the mandatory self-check |
| `ffrefine fit` | works -- weighted linear cosine fit, BIC model selection, `.itp`/`.prm`/`fit.json`/plots |
| `ffrefine charges` | works -- ESP-derived partial charges via ORCA, CHELPG or two-stage restrained RESP, equivalent hydrogens constrained equal, `selected` mode by default |
| `ffrefine validate` | works -- rebuild with the fitted series, re-run MM, report RMSE before and after |
| `ffrefine run` | works -- chains all five stages, charges first when asked, and says which stage it skipped |

Also not implemented: CGenFF `.str` penalty parsing and the automatic torsion
and charge-atom selection it would drive (`--str`, milestone 6), the
`final/report.md` summary document (milestone 6), and `--mm-relax` (spec 4.6).

### What the scan and charges do not do

It **reports** conformational hops and intramolecular hydrogen bonds; it does
not correct or reject a profile containing them. Both make a smooth cosine
series the wrong model -- a hop is a discontinuity, and an internal hydrogen
bond writes an electrostatic interaction into the dihedral term that the
nonbonded terms then count a second time -- so a warning from either is a
reason to re-scan from another conformer or with `--freeze-dihedral`, not
something to fit through.

`--sp-refine` defaults to `! wB97X-D3 def2-TZVP TightSCF`, a hybrid DFT single
point on every frame of the grid. It is opt-in for that reason.

`ffrefine charges` has its own boundaries:

- **Equivalence is local.** Two hydrogens are treated as equivalent exactly
  when they share a heavy-atom parent and an atom type -- a methyl or
  methylene group. Molecule-wide symmetry is not detected, so butane's two
  chemically-identical terminal carbons come back a few thousandths of an
  electron apart. That gap is ESP-fitting noise, and it is reported in
  `charges.json`, not smoothed away by guessing at a symmetry the tool never
  checked.
- **Gas-phase ESP charges are not condensed-phase charges.** The default level
  of theory, HF/6-31G*, is chosen because its systematic overpolarisation
  partly compensates for the missing solvent -- the same reasoning Amber and
  CHARMM-adjacent charge workflows use -- but `--charge-mode all` on a CGenFF
  topology is still a change of derivation scheme, not a refinement of the
  existing one. `ffrefine` warns when it looks like you are doing that.
- **`--conformers` (pooled multi-conformer fitting) and `--charge-geom orca`
  are not implemented.** Geometry is either the input structure as given or a
  single xtb `--opt tight` minimum.

## Requirements

- Python 3.10+ with `numpy`, `scipy` and `matplotlib` -- no other runtime
  dependencies, and no chemistry libraries: every file format is parsed
  in-house.
- GROMACS 2023 or newer on `PATH` (or `--gmx-bin /path/to/gmx`).
- xtb 6.7 or newer on `PATH` (or `--xtb-bin /path/to/xtb`), for `ffrefine scan`
  and for `ffrefine charges`'s default `--charge-geom xtb`.
- ORCA, for `ffrefine scan --sp-refine` (optional) and for `ffrefine charges`
  (required: every charge scheme needs ORCA's electrostatic potential).
  `--charge-scheme resp` also needs `orca_vpot`, which ffrefine looks for next
  to the `orca` binary unless told otherwise. ORCA is usually on no `PATH` at
  all, so reach it with `--orca-bin /path/to/orca`. A shared ORCA build will
  not start unless its own directory is on `LD_LIBRARY_PATH`; ffrefine sets
  that from `orca_libdir`, which defaults to the directory holding the binary
  you named.
- No network access is used at any point.

## Install

    pip install -e .

## Inputs

- `-s/--structure`: `.pdb`, `.gro` or `.xyz`, one molecule, **atom order
  identical to the topology**.
- `-p/--topology`: the molecule's `.itp` or `.top` (CHARMM36/CGenFF or GAFF
  style). A molecule `.top` carrying its own `[ system ]` and `[ molecules ]`
  works: those two sections are dropped from the working copies ffrefine builds,
  because the wrapper it generates for `grompp` supplies its own. Read "What
  this build cannot read" below before pointing this at a real force field.
- `--dihedral i,j,k,l`: the target torsion, **1-based** atom indices matching
  the topology's `[ atoms ]` numbering. This is the only index convention the
  user ever sees; the conversion to 0-based array indices happens internally at
  the geometry boundary.

Input files are never modified. Everything is written under `--outdir`, and
normalized copies of the inputs land in `00_input/`.

## What this build cannot read

The topology parser reads **one file**, and it reads only what is written in
that file. Three common topologies therefore do not work, and all three fail
with a clear message rather than a wrong number. This is a stated scope
boundary, not a bug list.

**1. Target dihedrals whose parameters come from `[ dihedraltypes ]`.**
This is the normal shape of a CGenFF or GAFF molecule `.itp`: the
`[ dihedrals ]` lines carry `i j k l 9` and nothing else, and GROMACS looks the
force constants up by atom type. ffrefine never reads those type tables, so it
cannot build the independent analytic reference the mandatory spec 4.5
self-check compares GROMACS against -- it would be comparing a real torsion
energy against zero. `mmscan` detects this *before* running GROMACS and refuses,
naming the offending lines.

To proceed, write those `[ dihedrals ]` lines out with their explicit `phi0`,
`k` and multiplicity (the values your force field assigns them) in a **copy** of
the topology, and run against the copy. Explicit parameters do suppress the
`[ dihedraltypes ]` lookup -- that is verified against GROMACS 2025.4 by
`tests/test_selfcheck.py::test_dihedraltypes_entry_is_suppressed_by_explicit_params`,
not assumed.

**2. Topologies that need force-field `#include`s to be complete.** A molecule
`.itp` that has no `[ atomtypes ]` of its own, because it expects
`charmm36.ff/forcefield.itp` to have been included first, cannot be turned into
a runnable system by ffrefine alone. Pass the include with `--ff-include`, which
is emitted ahead of the molecule in the generated wrapper `.top`; the path is
handed to `grompp` verbatim, so relative paths resolve against the stage
directory.

**3. Anything that requires following an `#include`.** `#include` lines are
copied through verbatim and never followed. A `.top` whose `[ atoms ]` lives in
an `#include`d `.itp` parses to "no `[ atoms ]` section found". Point `-p` at the
file that actually contains the `[ moleculetype ]` you want to refit. Note the
same limitation applies to `--resume`: see the `--resume` bullet under Common
flags.

## Worked example: n-butane

`fixtures/butane/` holds a hand-authored n-butane with CGenFF-style atom types.
It is a test fixture, not a CGenFF distribution file; do not simulate with it.

The C1-C2-C3-C4 torsion is atoms **1, 5, 8, 11**. Its central bond 5-8 carries
nine distinct dihedral quadruplets (ten term-lines: the target itself is split
across two multiplicities). All of them are zeroed to build the MM reference --
zeroing only the selected quadruplet is spec section 9.1's first named pitfall.

### 1. Charges (optional, but first if you want them)

    ffrefine charges \
        -s fixtures/butane/butane.pdb \
        -p fixtures/butane/butane.itp \
        --charge-atoms 1,11 \
        --outdir butane_out

This optimises the geometry with xtb, runs one ORCA ESP job, fits CHELPG
charges to it, and writes `butane_out/04_charges/molecule_charged.itp` along
with `charges.json` recording every atom's charge before and after and why it
changed. `--charge-mode selected` is the default, so only atoms 1 and 11 (the
terminal carbons) are replaced; the rest of each carbon's charge group absorbs
the change so the group's own net charge stays put.

The run prints three RRMS values, each measuring how well a set of charges
reproduces ORCA's own electrostatic potential, all on the same grid: the
charges **actually written** to the topology (the headline number, and the one
the 0.5 warning looks at), the fully fitted set, and the input topology's own
charges. `charges.json` keeps all three, as `rrms_written`, `rrms_fitted` and
`rrms_before`.

On this fixture at the default HF/6-31G* the fully fitted set comes out around
0.74, above the 0.5 warning threshold -- and that is expected, not a bug.
Butane's potential is weak and dominated by its quadrupole, which point charges
are a poor model for; an independent least-squares fit on our own grid
reproduces ORCA's own CHELPG charges to within about 0.06 e, confirming the
fitting code is right and the poor RRMS is the molecule's, not ours.

In `selected` mode the written charges can score worse than both: with
`--charge-atoms 1,11` above, the written topology's RRMS is about 1.83, against
1.36 for the input topology it replaces. A topology mixing a few ESP-fitted
charges with force-field charges reproduces the potential neither way. That is
why the headline number and the warning describe the written charges: the
number that matters is the one describing the file you will use.

**Charges come before the torsion, not after.** The fitted torsion absorbs
whatever the 1-4 electrostatics do not already account for, so a torsion fitted
against one set of charges does not belong in a topology carrying another. Pass
the charge-refined topology to the later stages:

    ffrefine mmscan -s ... -p butane_out/04_charges/molecule_charged.itp ...

`ffrefine run` does that swap for you.

### 2. MM energies, with and without the torsion

    ffrefine mmscan \
        -s fixtures/butane/butane.pdb \
        -p fixtures/butane/butane.itp \
        --dihedral 1,5,8,11 \
        --outdir butane_out

This writes `full.top` and `zeroed.top`, runs `gmx mdrun -rerun` on both, and
runs the mandatory self-consistency check: `E_full - E_zeroed` must equal the
analytic sum of the removed torsion terms to better than 1e-3 kJ/mol at every
frame. If it does not, the run aborts rather than fitting nonsense. On this
fixture it agrees to about 4e-5 kJ/mol, which is GROMACS's own mixed-precision
noise floor.

With no `01_qm_scan/scan.json` present, only the input structure's single frame
is available, so this checks the topology edit but produces no profile. Supply
a scan to get one.

### 3. Running the QM scan

    ffrefine scan         -s fixtures/butane/butane.pdb         -p fixtures/butane/butane.itp         --dihedral 1,5,8,11         --outdir butane_out

This drives xtb over a full turn -- 24 points at the default 15 degree step --
scanning forward, then back from the forward leg's final geometry, and keeping
the lower energy at each grid point. The two legs are paired by *measured*
angle rather than by array index, because a relaxed scan does not land exactly
on the requested grid and index-pairing quietly compares different geometries.

It writes `butane_out/01_qm_scan/scan.json`, a multi-model `scan.pdb` of the
geometries, and warns about conformational hops, hydrogen-bond formation and
forward/reverse hysteresis. The molecular charge comes from the topology's
`[ atoms ]` sum unless `--charge` overrides it.

**Bringing your own QM instead.** `mmscan`, `fit` and `validate` need the file,
not the tool that produced it, so a scan from any package can be written to
`butane_out/01_qm_scan/scan.json` by hand:

```json
{
  "energy_kj": [0.0, 1.2, 4.5, "... one QM energy in kJ/mol per frame ..."],
  "frames_nm": ["... (nframes, natoms, 3) coordinates in nanometres ..."],
  "phi_deg": ["... optional, informational ..."]
}
```

`frames_nm` is in **nanometres**, in topology atom order. `energy_kj` is in
**kJ/mol**, one value per frame, in the same order. Absolute offsets do not
matter -- both profiles are referenced to their own minimum.

**Re-run `mmscan` once the file exists**, so the MM energies cover the scan
geometries. `mmscan` records the digest of the `frames_nm` it saw, and `fit`
and `validate` refuse to run against a `scan.json` whose geometries have
changed since -- pairing new QM energies with stale angles and a stale `E_MM0`
is silent, undetectable and wrong. Only the geometries are hashed, so replacing
the `energy_kj` column alone (swapping in a better QM level, say) needs no
re-run.

The same check covers the target torsion: `energies.json` records the `quad` it
was computed for and `fit.json` records the quad it was fitted for, so fitting
one torsion's energies and writing the parameters onto another is an error
rather than a plausible-looking answer.

The angles used for fitting are then measured from those geometries, never from
a requested grid (spec section 9.5).

### 4. Fit

    ffrefine fit \
        -s fixtures/butane/butane.pdb \
        -p fixtures/butane/butane.itp \
        --dihedral 1,5,8,11 \
        --multiplicities 1,2,3,4,6 \
        --outdir butane_out

Fits `dE(phi) = [E_QM - min E_QM] - [E_MM0 - min E_MM0]` to
`sum_n k_n [1 + cos(n phi - delta_n)]`. This is exactly linear least squares in
a cosine basis, so there is no optimiser and no starting guess. Every subset of
the allowed multiplicities is fitted, ranked by BIC, and the printed table shows
the ranking; the selected model is the *smallest* subset whose RMSE is within
0.5 kJ/mol of the best, which deliberately trades a little accuracy for
parsimony. Expect butane to select `n=3` alone.

Useful flags:

- `--emax 40` -- discard points this far above the QM minimum (default 40 kJ/mol).
- `--boltzmann [T]` -- weight by `exp(-E_QM/RT)` instead of uniformly (default 300 K).
- `--free-phase` -- fit `a_n cos + b_n sin` and recover arbitrary phases. Needed
  for chiral or otherwise phase-shifted profiles; `fit` warns and names this flag
  when it detects one.
- `--distribute` -- spread `k_n` over all nine equivalent quadruplets as `k_n/N_T`
  instead of writing the whole series onto the target quadruplet. See the
  limitations below; this is the difference between a molecule-specific fix and a
  reusable dihedral type.

Physically expected for butane: a 3-fold profile, trans as the global minimum,
gauche roughly 2.5-3.5 kJ/mol above trans, syn barrier 14-20 kJ/mol. **Check
that against the QM profile, not against the fit's own minima and barriers.**
`fit` reports two sets of numbers and labels them: the *fitted correction
curve* is the fit of `dE = QM - MM0`, so its "largest correction swing" is the
size of the repair, not a torsion barrier; the *QM reference profile* block is
the one to compare with a literature expectation. `fit.json` keeps the same
split under `correction_curve_profile` and `qm_profile`. The refined torsion
profile itself is MM plus the correction, and `validate` is what reports it.

### 5. Validate

    ffrefine validate \
        -s fixtures/butane/butane.pdb \
        -p fixtures/butane/butane.itp \
        --dihedral 1,5,8,11 \
        --outdir butane_out

Rebuilds the topology with the fitted series, re-runs the MM single points on
the same geometries, and reports MM-against-QM RMSE **before and after**
refitting. Spec section 4.9 calls this the number that matters. "Before" is the
original force field's own parameters -- not the zeroed topology -- so the
improvement reported is improvement over the force field, not over having no
torsion at all.

### Or all of it at once

    ffrefine run -s ... -p ... --dihedral 1,5,8,11 --outdir butane_out

By itself, `run` chains `scan -> mmscan -> fit -> validate` and states up front
that it is skipping `charges`, because neither `--charge-atoms` nor
`--charge-mode all` was given. Ask for charges and they run first, and the
remaining stages pick up the charge-refined topology automatically:

    ffrefine run -s ... -p ... --dihedral 1,5,8,11 --charge-atoms 1,11 --outdir butane_out

`run` then chains `charges -> scan -> mmscan -> fit -> validate`, swapping
`-p` to `04_charges/molecule_charged.itp` for every stage after `charges`.

## Output layout

    butane_out/
      00_input/     copies of the input structure and topology, plus model.json
                    (atom/bond/dihedral counts, fudgeQQ, fudgeLJ, the torsion set)
      01_qm_scan/   scan.json, scan.pdb (the geometries as a multi-model PDB),
                    forward/ and reverse/ xtb working directories and logs
      02_mm_scan/   mdp, tpr, edr, xvg, full.top, zeroed.top, GROMACS logs, and
                    energies.json (measured phi, full and zeroed potentials, the
                    self-check residual)
      03_fit/       fit.json (every number), dihedral.itp (the fitted [ dihedrals ]
                    lines, funct 9, kJ/mol and degrees), dihedral.prm (CHARMM
                    DIHEDRALS block in kcal/mol), plots/
      04_charges/   charges.json (every atom's charge before/after and why),
                    molecule_charged.itp, the ORCA ESP job's inputs and outputs
                    (esp.inp, esp.gbw, esp.chelpg.xyz, esp.vpot and its logs),
                    opt/ (xtb --opt tight working directory, --charge-geom xtb
                    only), and, under --charge-scheme resp, grid.xyz and
                    grid_out.vpot (the MK grid sent to and read back from
                    orca_vpot)
      final/        molecule_refined.itp, validation.json, plots/validation.png,
                    plus the before/after topologies and GROMACS logs
      run.log       every external command with its full command line

`final/report.md` is part of the layout but belongs to milestone 6 and is not
produced.

Plots: `03_fit/plots/` holds the QM profile, the MM-zeroed profile, the target
difference with the fitted series overlaid, and the residuals.
`final/plots/validation.png` holds total MM energy against QM, before and after.

## Common flags

Every subcommand accepts `--config file.json`, `--outdir`, `--nthreads`,
`--resume`, `--dry-run`, `--verbose`, `--gmx-bin`, `--xtb-bin`, `--orca-bin`,
`--ff-include` and `--no-strict-names`.

- `--config` values are overridden by anything given on the command line. A
  `"binaries": {"gmx": "..."}` object is honoured alongside the flat keys.
  `--ff-include` is command-line only.
- `--ff-include path.itp` adds a `#include` ahead of the molecule in the wrapper
  `.top` ffrefine generates for `grompp`. Repeatable; emitted in the order given.
  Use it when the molecule topology has no `[ atomtypes ]` of its own. A value
  naming a file ffrefine can see is made absolute, since the wrapper lives in the
  stage directory; anything else is passed through untouched so GMXLIB-relative
  names like `charmm36.ff/forcefield.itp` still resolve.
- `--resume` skips a stage whose recorded input hashes still match. A manifest
  that is missing, corrupt, or built from different inputs always recomputes.
  **The hash covers the top-level topology file only.** `#include` directives are
  not followed, so editing an `#include`d `.itp` -- or the force field named by
  `--ff-include` -- leaves the hash unchanged and the stage is skipped as current.
  Re-run without `--resume` after changing an included file.
- `--dry-run` validates the inputs, reports what would run, and creates nothing
  at all -- not even the output directory.
- `--no-strict-names` downgrades an atom-name mismatch between structure and
  topology to a warning. Atom count and element ordering stay fatal: `grompp`
  runs with `-maxwarn`, which suppresses its own atom-order guard, so this check
  is the only thing standing between a mis-ordered file and silently wrong
  energies.

### Scan flags

Accepted by `ffrefine scan`, and by `ffrefine run`, which chains it.

- `--step` scan step in degrees (default 15, so 24 points). It must divide 360
  into whole steps, or the run is refused rather than silently rounded.
- `--charge` molecular charge. Default: the topology's `[ atoms ]` charge sum,
  rounded to the nearest integer. A sum further than 0.05 from an integer is
  refused, because it usually means an incomplete topology and guessing beats
  nothing only until it does not.
- `--uhf` unpaired electrons (default 0). **Not derived**: electron-count
  parity cannot tell a triplet from a singlet diradical, so it stays explicit.
- `--gfn` xtb GFN parameterisation (default 2).
- `--bidirectional` / `--no-bidirectional` scan both ways and keep the lower
  energy per point, or walk one direction only. Bidirectional is the default.
- `--hysteresis-tol` warn when the two directions disagree by more than this
  many kJ/mol (default 4). A profile that depends on the direction it was
  walked has barrier heights you should not trust.
- `--freeze-dihedral i,j,k,l` hold another torsion fixed for the whole scan.
  Repeatable. The usual reason is to stop a second rotor following the first.
- `--sp-refine` recompute each frame's energy with an ORCA single point,
  keeping xtb's geometries. Improves the fit target without redoing the scan.
- `--orca-sp-keywords` the ORCA keyword line for `--sp-refine`
  (default `! wB97X-D3 def2-TZVP TightSCF`).

### Charge flags

Accepted by `ffrefine charges`, and by `ffrefine run`, which chains it.
`--charge` and `--uhf`, listed under Scan flags above, apply here too: charges
needs the same molecular charge and spin state the scan does.

- `--charge-scheme {chelpg,resp}` (default `chelpg`) -- ORCA fits the CHELPG
  charges itself; `resp` has ffrefine fit its own two-stage restrained least
  squares to the same ORCA-computed potential, on our own MK grid, so it can
  impose the equivalence constraints and report a comparable RRMS.
- `--charge-geom {input,xtb}` (default `xtb`) -- fit against the structure as
  given, or against an `xtb --opt tight --gfn 2` minimum computed first.
  Charges fitted to a strained input geometry describe a molecule nobody is
  going to simulate.
- `--charge-mode {selected,all}` (default `selected`) -- replace only the
  atoms named by `--charge-atoms`, or every charge in the topology.
  **`selected` refuses to run without `--charge-atoms`**: replacing every
  charge on a CGenFF topology changes the derivation scheme entirely, so
  defaulting silently to `all` would be the wrong kind of convenient.
  Automatic atom selection from CGenFF penalties arrives in milestone 6; until
  then, you name the atoms yourself.
- `--charge-atoms i,j,k` (default: none) -- 1-based atom numbers to replace
  under `--charge-mode selected`. Repeatable; every atom named across every
  occurrence is pooled into one set.
- `--orca-esp-keywords` (default `! HF 6-31G*`) -- the ORCA level of theory
  for the ESP job. **This carries the level of theory only**: the scheme
  appends `CHELPG` to the keyword line itself, so overriding the level can
  never silently turn the scheme off.
- `--esp-density` (default `1.0`) -- MK grid points per square Angstrom of
  shell surface, `--charge-scheme resp` only; CHELPG's grid is ORCA's own and
  ignores it.

ORCA is usually on no `PATH` at all, so `--orca-bin /path/to/orca` is the
normal way to reach it, same as for `--sp-refine`. A shared ORCA build will
not start unless its own directory is on `LD_LIBRARY_PATH`; ffrefine sets that
from `orca_libdir`, which defaults to the directory holding the binary you
named (see Requirements, above).

RESP's restraint follows the reference implementations (Bayly et al.'s
`resp.f`, and psi4-resp): the objective minimised is
`1/2 * sum(V - Phi q)^2 + a * sum(sqrt(q^2 + b^2) - b)` over the restrained
(non-hydrogen) atoms, with `a = 0.0005` in stage 1, `a = 0.001` in stage 2, and
`b = 0.1` throughout. Anyone checking numbers against another RESP program
should confirm it uses the same convention before comparing.

Expect RESP's charges to differ markedly from CHELPG's, and for a reason: the
restraint pulls heavy-atom charges toward zero, which is exactly what it is
for. On the butane fixture at HF/STO-3G, CHELPG's terminal carbons come out
around -0.40 e; RESP's come out around -0.15 e, and the internal CH2 carbons
move to almost nothing. That is the restraint working, not a discrepancy to
chase -- with the restraint disabled, the two schemes agree to within about
0.05 e.

Exit codes: `0` success, `1` a reported ffrefine error (printed as
`error: ...`, never a traceback), `2` a usage error or an unimplemented
subcommand.

One torsion per run. `--dihedral` is repeatable in the parser to match the spec's
CLI surface, but the stage layout holds a single scan and a single fit per output
directory, so passing more than one is rejected with a message rather than
silently fitting only the first. Use a separate `--outdir` per torsion.

## Physical assumptions and their limits

**The fitted torsion is convention-bound. This is the big one.** The refit
absorbs whatever the 1-4 interactions do not already account for, which is
exactly why `[ pairs ]` is never removed. The consequence is that the fitted
parameters are **only valid alongside the same nonbonded parameters and the same
1-4 conventions (`fudgeQQ`, `fudgeLJ`) as the topology they were fitted
against**. Those values are read from the topology's `[ defaults ]`, never
assumed, and recorded in `00_input/model.json`. CHARMM and Amber/GAFF differ
here. Moving a fitted dihedral between force fields is not meaningful, and
nothing in the output file will stop you from doing it.

**Milestone 6 does not exist yet.** There is no CGenFF `.str` parser, so there
is no automatic torsion or charge-atom selection: `--dihedral` and
`--charge-atoms` are the only way to point ffrefine at anything, and there is
no `final/report.md` summary document. `--mm-relax` (spec 4.6) is also
unimplemented.

**A charge-group's net charge is preserved, and the molecular total is exact.**
`--charge-mode selected` redistributes the fit's residual per charge group, not
across the whole molecule: each group containing a selected atom absorbs the
charge its own atoms gained or lost, spread over that group's unselected atoms
(falling back to unselected heavy atoms bonded to a selected one in that group
when the group has none). Only a small leftover global offset -- an explicit
`--charge` that differs from the topology's own sum by no more than 0.05 e,
the same tolerance used to derive the charge -- is pooled across groups. A
larger disagreement is refused before any QM runs: `--charge -1` on a neutral
topology would otherwise push a whole electron onto the two or three atoms
sharing the selected atoms' charge groups. Use `--charge-mode all`, which
refits every charge at the declared charge, or correct the topology.
Charges are written to four decimal places, and any leftover rounding residue
is absorbed by the largest-magnitude redistributed atom -- kept off a fitted
atom's own value whenever an unselected atom exists to take it instead. Under
`--charge-mode all`, where every atom is fitted and none is left to redistribute
onto, the residue lands on the largest-magnitude atom regardless. A drift
bigger than rounding can explain, or an offset with nowhere to go, is refused
rather than folded in quietly.

**Molecule-specific by default.** Without `--distribute` the fitted series is
written onto one quadruplet and the other torsions about that bond are set to
zero. That reproduces this molecule exactly but is not a transferable dihedral
type -- applied as a CHARMM type it would hit every quadruplet matching those
atom types. The emitted `.prm` block carries a loud warning to that effect
unless `--distribute` was used.

**The `.prm` block covers one dihedral type, not the whole torsion set.** Even
with `--distribute`, `03_fit/dihedral.prm` emits a single type signature -- the
target quadruplet's. The `N_T` quadruplets about the bond generally span
several types (on the butane fixture, 9 quadruplets span three), so the block
carries only the share belonging to that one type and does **not** reproduce
`03_fit/dihedral.itp`. The block states its own coverage and lists the
combinations it omits. Use the `.itp` for GROMACS; treat the `.prm` as a
starting point for a CHARMM parameterisation. Emitting the other signatures is
deliberately out of scope: a generic CGenFF type such as
`HGA2-CG321-CG321-HGA2` would, if redefined, perturb every molecule containing
that motif.

**Vacuum throughout.** The MM reference is gas phase, so the QM scan must be
too. No implicit solvent enters the torsion profile, and a torsion fitted in
vacuum is not automatically the right torsion in water.

**The fit is only as good as the scan.** Relaxed scans hop conformers; a
discontinuity in the QM profile will be fitted straight through. Look at
`03_fit/plots/fit.png` and the residuals before trusting anything.

**Asymmetric profiles need `--free-phase`.** The default cosine-only basis is
even in `phi` and restricts phases to 0 or 180 degrees, so it cannot represent a
chiral or phase-shifted torsion at all. `fit` measures the profile's mismatch
against its own mirror image and warns when that exceeds 0.5 kJ/mol, naming the
flag. Do not ignore that warning.

**Bonds, angles, LJ parameters and improper dihedrals are out of scope.**
Impropers (funct 2 and 4) are never modified, and `[ pairs ]` is never removed.

**Round-trip precision is bounded by GROMACS, not by the fitter.** Energies are
read back from `gmx energy` `.xvg` text at six decimal places over a
mixed-precision (float32) `.edr`, which puts an absolute floor of about
1e-5 kJ/mol on the recovered coefficients. See the module docstring of
`tests/test_roundtrip.py` for the measurement and how to tighten it.

## Tests

    pytest

Tests that need an external program are marked `needs_gmx`, `needs_xtb` or
`needs_orca`, and skip when that binary is not on `PATH`. A skipped test is not
a passing one: a plain green run on a machine with no `gmx`, `xtb` or `orca`
has exercised none of the code that drives them, so run where they are
available before trusting the result.

    pytest -m "needs_gmx or needs_xtb or needs_orca"   # and check nothing skipped

As of this milestone the suite holds 517 tests: 16 marked `needs_gmx`, 4
`needs_xtb` and 8 `needs_orca`. The rest need nothing external and always run.

The load-bearing one is spec section 7.1's round trip, which covers the whole
pipeline except xtb and ORCA: take the MM energies with the original
parameters, treat them as the QM reference, zero the torsion, refit, and require
the original profile back. `tests/test_cli.py::test_end_to_end_mmscan_fit_validate`
drives that same round trip through the CLI end to end and checks that
validation's "before" RMSE really is measured against the original force field.
