# Build `ffrefine`: a small-molecule force field parameter refinement tool

You are implementing a Python CLI package that repairs high-penalty parameters produced by generalized force fields (CGenFF, GAFF). Two modules only:

- **Torsions**: refit proper dihedral terms against a QM torsion scan (xtb), using GROMACS to remove the existing torsion contribution.
- **Charges**: replace or adjust partial charges using ESP-derived charges (ORCA).

Out of scope: bonds, angles, LJ parameters, improper dihedrals.

Name the package `ffrefine` (rename freely). Do not implement anything not specified here without asking.

---

## 1. Environment and constraints

- ORCA, `xtb`, and `gmx` are on PATH. GROMACS 2023 or newer. No internet at runtime.
- Python 3.10+. Hard dependencies limited to `numpy`, `scipy`, `matplotlib`. Parse all chemistry file formats in-house. No RDKit, no OpenBabel, no MDAnalysis, no AmberTools.
- Locate binaries with `shutil.which`, allow explicit overrides via CLI or config.
- Never modify input files in place.

## 2. Inputs

Required:
- Structure: `.pdb` or `.gro`, single molecule, atom order identical to the topology.
- Topology: `.itp` or `.top` for the molecule (CHARMM36/CGenFF or GAFF style).

Optional:
- `--str mol.str`: CGenFF stream file. Parse the penalty annotations so the tool can auto-select which charges and which dihedrals need work (`--penalty-threshold`, defaults: charge 10, dihedral 50).
- `--charge` (net charge, default 0), `--multiplicity` (default 1).

Torsion selection: `--dihedral i,j,k,l` using 1-based atom indices matching the topology `[ atoms ]` numbering. Repeatable for multiple torsions. If `--str` is given and no `--dihedral`, select automatically from penalties.

## 3. CLI

```
ffrefine scan      # QM relaxed torsion scan (xtb, optional ORCA single-point refinement)
ffrefine mmscan    # GROMACS single-point energies with and without the target torsions
ffrefine fit       # fit the dihedral series, write new parameters
ffrefine charges   # ESP/RESP charges via ORCA
ffrefine validate  # rebuild topology with new parameters, re-run MM, compare to QM
ffrefine run       # everything, in order
```

Every subcommand accepts `--config file.json` with CLI overriding file values, `--outdir`, `--nthreads`, `--resume`, `--dry-run`.

---

## 4. Torsion refinement pipeline

### 4.1 Topology model

Parse the topology into an explicit object model: atoms (index, type, resname, name, cgnr, charge, mass), bonds, pairs, angles, proper dihedrals (funct 1, 4, 9), improper dihedrals (funct 2, 4), plus `#include`, `#ifdef`, and `[ defaults ]` handling sufficient to find `fudgeQQ`, `fudgeLJ`, `gen-pairs`.

Validate that the structure and topology agree on atom count, names, and element ordering. Abort with a clear message on mismatch.

### 4.2 Target torsion set

Given quadruplet `(i,j,k,l)`, the central bond is `(j,k)`. Collect **every** proper dihedral entry in `[ dihedrals ]` whose central pair is `(j,k)` or `(k,j)`. Call this set `T`, size `N_T`. All of `T` is zeroed for the MM reference. Leave impropers alone.

By default the fitted profile is written onto the target quadruplet only, with the remaining members of `T` explicitly set to zero. This is exact for this molecule but produces a molecule-specific, non-transferable topology. Provide `--distribute` to instead spread the fitted `k_n` over all `N_T` members as `k_n / N_T`, which is what you need if the output is meant as a reusable dihedral *type*. State which mode was used in the report.

### 4.3 QM scan (xtb)

Write an xtb constraint/scan input:

```
$constrain
  force constant=1.0
  dihedral: i,j,k,l,auto
$scan
  1: 0.0,360.0,NPOINTS
$end
```

Run `xtb mol.xyz --opt tight --input scan.inp --chrg Q --uhf M --gfn 2 -P N`. Verify the exact `$scan` syntax against the installed xtb version's help output at runtime rather than assuming it; fail loudly if the scan block is rejected.

- Grid: `--step` degrees, default 15, spanning `0` to `360 - step` so endpoints are not duplicated.
- Parse energies from the comment lines of `xtbscan.log` (Hartree). Convert with 2625.499639 kJ/mol per Hartree.
- **Measure the actual dihedral value from each frame's coordinates.** Do not trust the requested grid. Use measured values throughout.
- Bidirectional scanning on by default (`--bidirectional/--no-bidirectional`): forward from the input geometry, then reverse starting from the final forward structure. Take the lower energy at each grid point. If forward and reverse differ by more than `--hysteresis-tol` (default 4 kJ/mol) anywhere, warn prominently.
- Discontinuity detection: compute heavy-atom RMSD between adjacent frames after alignment; flag frames where a conformational hop or an intramolecular hydrogen bond has formed. Report, do not silently continue.
- `--freeze-dihedral a,b,c,d` (repeatable) adds extra `$constrain` dihedral entries to hold other rotatable bonds fixed.
- Optional `--sp-refine`: run ORCA single points on the xtb geometries. Default level `! wB97X-D3 def2-TZVP TightSCF`, overridable with `--orca-sp-keywords`. Parse `FINAL SINGLE POINT ENERGY`. Use these energies in place of the xtb ones when present.

Gas phase throughout. No implicit solvent for the torsion profile, since the MM reference is vacuum.

### 4.4 MM energies (GROMACS)

Write the scan geometries as a single multi-MODEL PDB (or multi-frame GRO) using the atom and residue names from the input structure so `mdrun -rerun` matches the topology.

Build two topologies:
- `full.top`: unchanged.
- `zeroed.top`: identical, except every member of `T` in `[ dihedrals ]` is replaced by one explicit line `i j k l 9 0.000 0.000 1`. Explicit parameters on a `[ dihedrals ]` line suppress lookup in `[ dihedraltypes ]`; **verify this at runtime with the check in 4.5 rather than trusting it.**

Do not touch `[ pairs ]`. 1-4 interactions stay on in both energies.

Single-point mdp, vacuum, infinite cutoffs:

```
integrator              = md
nsteps                  = 0
pbc                     = no
cutoff-scheme           = Verlet
verlet-buffer-tolerance = -1
nstlist                 = 0
rlist                   = 0
coulombtype             = Cut-off
rcoulomb                = 0
vdwtype                 = Cut-off
rvdw                    = 0
epsilon_r               = 1
constraints             = none
continuation            = yes
nstcalcenergy           = 1
nstenergy               = 1
```

If the installed GROMACS rejects zero cutoffs, fall back to a large box (10 nm, `pbc = xyz`) with 3 nm cutoffs and plain cut-off electrostatics. Detect this automatically and log which path was used.

Run `gmx grompp` then `gmx mdrun -rerun`, then `gmx energy` extracting Potential plus the decomposed terms (Bond, Angle, Proper Dih., Improper Dih., LJ-14, Coulomb-14, LJ (SR), Coulomb (SR)) for diagnostics. Select `gmx energy` terms by name matching, not by fixed menu index; the numbering varies between versions. Assert exactly one energy frame per input frame.

### 4.5 Mandatory self-consistency check

Compute `E_full(phi) - E_zeroed(phi)` from GROMACS and compare against the analytic sum of the removed torsion terms evaluated at each measured `phi` using the original parameters. Agreement must be better than 1e-3 kJ/mol at every point. If it fails, the topology edit did not do what was intended: abort with a diagnostic, do not proceed to fitting.

### 4.6 Optional MM relaxation

`--mm-relax`: at each grid point, energy-minimize with `zeroed.top` plus a `[ dihedral_restraints ]` entry holding the torsion at the measured QM value (funct 1, dphi 0, high force constant, default 10000 kJ/mol/rad^2), and use the minimized energy and geometry. This removes bond and angle strain differences between QM and MM geometries. Default off; the simple single-point path is the default.

### 4.7 Fit

Target function, both curves shifted to their own minimum:

```
dE(phi) = [E_QM(phi) - min E_QM] - [E_MM0(phi) - min E_MM0]
```

Model:

```
V(phi) = sum_n k_n * [1 + cos(n*phi - delta_n)] + c
```

with `n` drawn from `{1,2,3,4,6}` (settable via `--multiplicities`), `k_n >= 0`, and `delta_n` restricted to 0 or 180 degrees.

Implementation: this is exactly linear least squares in a cosine basis. Fit `V = c' + sum_n a_n*cos(n*phi)`, then `k_n = |a_n|` and `delta_n = 0` if `a_n > 0` else `180`. The constant is absorbed and discarded.

- Asymmetry check: if the profile is not symmetric about `phi = 0` beyond a tolerance (chiral or otherwise phase-shifted torsions), the cosine-only basis cannot fit it. Detect this, warn, and offer `--free-phase` which fits `a_n*cos + b_n*sin` and converts to `k_n, delta_n` with `atan2`. GROMACS funct 9 accepts arbitrary phase.
- Weighting: default uniform, discarding points above `--emax` (default 40 kJ/mol relative to the QM minimum). `--boltzmann [T]` switches to `w = exp(-E_QM/RT)`, default 300 K.
- Model selection: evaluate all subsets of the allowed multiplicities, rank by BIC, and by default select the smallest subset whose RMSE is within 0.5 kJ/mol of the best. Print the full ranking table.
- Report: RMSE, maximum deviation, positions and depths of minima, barrier heights, QM vs fitted comparison.

### 4.8 Outputs

- New `.itp` with the fitted `[ dihedrals ]` lines (funct 9, one line per retained multiplicity, kJ/mol, degrees).
- CHARMM `.prm`-style `DIHEDRALS` block in kcal/mol using atom types, for users going back to CHARMM. Only emit this when `--distribute` is used, since a type-level block is meaningless otherwise; otherwise emit it with a loud comment warning.
- `fit.json` with every number.
- Plots: QM profile, MM-zeroed profile, target difference with the fitted series overlaid, residuals.

### 4.9 Validation (always runs at the end of `run`)

Rebuild the topology with the fitted parameters, redo the MM single points on the same geometries, and plot total MM energy against QM before and after refitting. Report both RMSE values. This is the number that matters.

---

## 5. Charge refinement

### 5.1 Geometry

`--charge-geom {input,xtb,orca}`, default `xtb`: optimize with `xtb --opt tight --gfn 2` before the ESP calculation. Optionally accept several conformers for a multi-conformer fit (`--conformers file.xyz`), fitting a single charge set to the pooled ESP data.

### 5.2 ESP calculation

Two schemes:

**`chelpg` (default, simplest):** run ORCA with `! HF 6-31G* CHELPG`. Parse the CHELPG charge table. HF/6-31G\* is deliberate: its systematic overpolarization approximates condensed-phase charges and is what both Amber and CHARMM-adjacent workflows use. Overridable via `--orca-esp-keywords`.

**`resp`:** generate a Merz-Singh-Kollman grid (4 shells at 1.4, 1.6, 1.8, 2.0 times the vdW radii, roughly 1 point per square Angstrom, points inside any inner shell removed), evaluate the ESP at those points with `orca_vpot` against the converged density, then run a standard two-stage restrained fit implemented in `numpy`:
- Stage 1: hyperbolic restraint `a = 0.0005`, `b = 0.1`, applied to non-hydrogen atoms, total-charge constraint via a Lagrange multiplier, solved iteratively to convergence.
- Stage 2: refit only the atoms in equivalent groups (methyl and methylene hydrogens and their carbon), `a = 0.001`, all other charges frozen at stage 1 values.

Verify `orca_vpot` invocation against the installed ORCA version; do not hardcode an argument order that may not match. Report the ESP fit RRMS.

### 5.3 Equivalence and bookkeeping

- Auto-detect topologically equivalent hydrogens (same heavy-atom parent, same atom type, same connectivity depth) and constrain them to equal charges.
- `--charge-mode {all,selected}`, default `selected`. In `selected` mode, only atoms listed via `--charge-atoms` or flagged above the CGenFF penalty threshold are replaced; the residual is redistributed so the total remains exactly the target integer, spread over the atoms in the same charge group, falling back to bonded heavy atoms. Log the redistribution atom by atom.
- After rounding to the output precision, compensate the rounding residual on the largest-magnitude atom so the total charge is exact to 1e-6.
- Print this warning in the log and the report whenever `--charge-mode all` is used with a CGenFF topology: CGenFF charges are fitted to water interaction energies and dipoles, not to the gas-phase ESP, so wholesale replacement with ESP charges is a change of derivation scheme and can degrade condensed-phase behavior. Recommend `selected`.

---

## 6. Output layout

```
outdir/
  00_input/     normalized copies of the structure and topology, parsed model dump
  01_qm_scan/   xtb inputs, xtbscan.log, optional ORCA SP, scan.json, scan.pdb
  02_mm_scan/   mdp, tpr, edr, xvg, full and zeroed topologies, energies.json
  03_fit/       fit.json, dihedral itp fragment, prm block, plots/
  04_charges/   ORCA inputs and outputs, grid, charges.json
  final/        molecule_refined.itp, report.md, plots/
  run.log
```

`report.md` should be readable standalone: what was changed, the fit quality before and after, every warning raised, the exact QM level used, and an explicit statement that the fitted torsion parameters are only valid with the same nonbonded parameters and the same 1-4 conventions (`fudgeQQ`, `fudgeLJ`) as the topology they were fitted against.

---

## 7. Tests

Write these as real tests, not as an afterthought.

1. **Round-trip refit (no QM needed, run in CI).** Take a topology, compute MM energies over a torsion grid with the original parameters, treat those as the "QM" reference, zero the torsion, refit. The recovered coefficients must match the originals to 1e-6. This exercises the entire pipeline except xtb and ORCA.
2. **Analytic torsion recovery** (section 4.5) as a unit test against a small hand-written topology.
3. **PDB vs GRO equivalence**: the same molecule in both formats must give identical results.
4. **Unit conversions**: Hartree to kJ/mol, kJ/mol to kcal/mol, degrees to radians, round trip.
5. **Parser tests**: `[ dihedrals ]` with and without explicit parameters, funct 9 multi-line entries, `#ifdef` blocks, impropers not being picked up as propers.
6. **Butane integration test** (skipped when binaries are absent): C1-C2-C3-C4 with CGenFF. Expect a 3-fold profile, trans global minimum, gauche roughly 2.5 to 3.5 kJ/mol above trans, syn barrier in the 14 to 20 kJ/mol range.

---

## 8. Engineering requirements

- Log every external command with its full command line; capture stdout and stderr to files under the relevant stage directory. Nonzero exit gives a clear error including the tail of the output.
- Resumable: each stage writes a manifest with input hashes; `--resume` skips completed stages whose inputs are unchanged.
- `--nthreads` maps to `xtb -P`, ORCA `%pal nprocs`, and `gmx mdrun -nt`.
- All user-facing atom indices are 1-based; document clearly where internal arrays are 0-based and convert at the boundary.
- Type hints throughout, docstrings on public functions, `black` formatting, `pyproject.toml`.
- README with a worked butane example plus a short section stating the physical assumptions and their limits.

---

## 9. Pitfalls, stated explicitly

1. Zero **all** torsions sharing the central bond, not only the selected quadruplet.
2. Never remove `[ pairs ]`. The 1-4 interactions must be present in both MM energies; the fitted torsion absorbs whatever the 1-4 terms do not capture, which is precisely why the parameters are convention-bound.
3. CHARMM and Amber/GAFF differ in 1-4 scaling. Read `fudgeQQ` and `fudgeLJ` from the topology, never assume, and record them in the report.
4. Improper dihedrals (funct 2 and 4) are never modified.
5. Use measured dihedral values from geometries, not the requested grid values.
6. Wrap angles consistently to (-180, 180] for fitting and plotting; the fit basis must be periodic.
7. With `--distribute`, divide the fitted `k_n` by the number of equivalent quadruplets.
8. GROMACS is kJ/mol, CHARMM `.prm` is kcal/mol, xtb is Hartree. Convert at one place only.
9. `gmx energy` term ordering is version dependent. Match by name.
10. Relaxed scans hop conformers. Detect and report; do not fit silently across a discontinuity.

---

## 10. Build order

1. Structure and topology parsers, topology object model, tests.
2. GROMACS energy engine, topology zeroing, section 4.5 self-check, round-trip refit test. At this milestone the whole fitting machinery is verified without any QM.
3. Fitting, model selection, outputs, plots, validation stage.
4. xtb scan driver, bidirectional scanning, discontinuity detection.
5. ORCA charges (CHELPG first, then RESP).
6. CGenFF `.str` penalty parsing, auto-selection, `report.md` generation.

Stop and ask before deviating from this spec, adding dependencies, or making a design decision it does not cover.
