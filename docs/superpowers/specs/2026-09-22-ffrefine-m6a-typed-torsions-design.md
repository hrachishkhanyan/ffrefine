# Design — milestone 6a: force-field-typed torsions and `#include`d topologies

Status: approved in brainstorming, 2026-09-22.
Prerequisite for milestone 6b (spec build-order item 6: CGenFF `.str` penalty parsing,
auto-selection, `report.md`). Follows milestone 5 (`ffrefine charges`), merged to master.

---

## 1. Purpose and scope

The two real CGenFF molecules supplied for milestone 6 -- PFOA and capecitabine, each a
`.str`, a `_gmx.top`, a `_gmx.pdb` and a `charmm36.ff/` -- cannot run through ffrefine today,
not even with a hand-picked `--dihedral`. Three things stop them, and a topology produced by
the CGenFF server's GROMACS export has all three:

1. **Typed torsions.** Every `[ dihedrals ]` line is a bare `i j k l 9`; the force constants
   come from `[ dihedraltypes ]` in `charmm36.ff/ffbonded.itp` and `<mol>_ffbonded.itp`.
   `mmscan` refuses this up front (`_reject_typed_terms`), because the spec 4.5 self-check
   has no analytic reference for parameters this build never read.
2. **Relative includes.** The `.top` includes `./charmm36.ff/forcefield.itp`. Every
   grompp-facing copy ffrefine writes lives in a stage directory, where that path does not
   resolve.
3. **`[ defaults ]` only in an include.** `forcefield.itp` holds it, so `model.json` records
   `fudgeQQ`/`fudgeLJ` as null and raises a warning that is false for this input.

Milestone 6a removes those three obstacles. Milestone 6b then adds `.str` parsing,
auto-selection and `report.md` on top of a pipeline that already runs on the real molecules.

### In scope

- Resolving the parameters of typed torsions from GROMACS's own processed topology.
- Include handling for every topology ffrefine writes: absolute in grompp-facing copies,
  verbatim plus an include-base marker in the files the user keeps.
- `[ defaults ]` resolved through `grompp -pp` and recorded by `mmscan`.
- `--resume` keys for `mmscan` and `validate` that cover every file grompp read.
- The PFOA and capecitabine fixtures, committed with a trimmed force field, and the script
  that trims it.
- README "What this build cannot read" items 1 and 2 rewritten; a PFOA worked example.

### Out of scope

- `.str` parsing, penalty-driven selection, multi-bond runs, `report.md` (milestone 6b).
- The parked milestone-5 follow-ups R26-R29 (milestone 6b, per the memory note: R28 and R29
  bear on report generation; R26/R27 on any work inside `charges/`, which 6a does not touch).
- A topology whose `[ atoms ]` lives in an `#include`d file (README item 3 stays).
- Free-energy (B-state) topologies, RB (funct 3) dihedrals.
- Mapping `.str` atom names to `.top` atoms. Noted here because it was found here: the
  CGenFF converter renames atoms (`.str` `O1` is `.top` `O2`, `.str` `C2` is `.top` `C3`);
  order, types and charges agree index for index. Milestone 6b maps by index, checked
  against type and charge.

No new CLI flag.

---

## 2. Environment findings

Established against GROMACS 2025.4 (conda-forge) on the PFOA fixture, not assumed.

### `grompp -pp` flattens the topology exactly as grompp reads it

`gmx grompp ... -pp flat.top` on `pfoa_gmx.top` wrote a 5.8 MB, 92 362-line file with every
include expanded and every `#ifdef` resolved: `[ defaults ]` at line 15, the CHARMM36
`[ dihedraltypes ]` blocks, `silcs.itp`'s and `pfo_ffbonded.itp`'s own blocks, then the
molecule. It carries no absolute paths and no timestamps, so its SHA-256 is a stable key for
"everything grompp read".

### `gmx dump` prints the parameters GROMACS will compute with

```
         functype[81]=PDIHS, phiA= 1.80000000e+02, cpA= 2.88696003e+00, phiB= 1.80000000e+02, cpB= 2.88696003e+00, mult=1
...
   moltype (0):
      name="Other"
         atom (26):
...
      Proper Dih.:
         nr: 380
         iatoms:
            0 type=81 (PDIHS)   1   0   2   3
            1 type=82 (PDIHS)   1   0   2   3
```

- Atom indices in `iatoms` are 0-based and keep the order written on the `[ dihedrals ]`
  line (`2 1 3 4` becomes `1 0 2 3`).
- A typed funct-9 line expands into one iatoms entry per multiplicity, consecutively.
- Values are the float32 numbers GROMACS computes with (`2.88696003` for `2.886960`), which
  is more faithful to what `mdrun` evaluates than the text values.
- Funct 1 and 9 both appear as `PDIHS`; funct 4 is `PIDIHS` and funct 2 `IDIHS`, neither of
  which is a proper torsion here.

### grompp drops zero-force-constant terms

`pfo_ffbonded.itp` gives `CG2O2 CG312 CG312 CG312` two lines, multiplicity 3 with
`k = 0.58576` and multiplicity 6 with `k = 0`. The tpr holds one term for quad
`1 3 4 5`, the multiplicity-3 one, and no `PDIHS` functype anywhere has `cpA = 0`. A term
with `k = 0` in both states is removed by grompp. The cross-check in section 6 is built
around this: a `k = 0` term is *expected* to be absent, and a typed line whose terms are all
zero legitimately resolves to no terms.

### CGenFF `.top` quirks the parser already survives

`# include "./charmm36.ff/pfo_ffbonded.itp"` (a space after `#`) is recorded as a directive
and left for grompp; `#ifdef POSRES` hides a `posre.itp` that is not shipped; `tip3p.itp` and
`ions.itp` are included after the molecule and add moleculetypes grompp parses but
`[ molecules ]` never lists. The parser needs no change for these; the include rewriter
(section 5) must recognise `#\s*include`.

---

## 3. Architecture

```
ffrefine/io/includes.py     NEW  pure text; no subprocess
    include_base(path) -> Path
    absolutise_includes(lines, base) -> list[str]
    mark_include_base(lines, base) -> list[str]

ffrefine/mm/tprparams.py    NEW  runs grompp -pp and gmx dump; parses; cross-checks
    resolve_torsion_set(binaries, topology, quad, full_top, workdir) -> ResolvedTorsionSet
        .terms       list[Dihedral], explicit params on every entry
        .sources     per term: "topology" | "dihedraltypes"
        .defaults    Defaults parsed from the flattened topology
        .pp_sha256   SHA-256 of the flattened topology
    preprocessed_sha256(binaries, top, structure, workdir) -> str

ffrefine/io/topology.py     Topology gains include_base (from the marker, else source dir)
ffrefine/io/topwriter.py    every writer takes an explicit include mode:
                              "absolute"  grompp-facing copies
                              "verbatim"  kept files: user's lines + marker
ffrefine/mm/selfcheck.py    _reject_typed_terms removed; analytic side uses resolved terms
ffrefine/cli.py             mmscan resolve step and payload; deferred defaults warning;
                            resume keys; molecule_refined.itp written, not copied
scripts/trim_forcefield.py  NEW  fixture force-field trimmer
```

`includes.py` sits in `io/` beside the parser and writers, and like them never starts a
process. `tprparams.py` sits in `mm/` beside `gromacs.py`, which already owns every grompp
and mdrun invocation; it reuses `run_command`, `write_mdp` and `_write_conf_gro`.

---

## 4. Data flow

### `mmscan`

1. `_prepare` writes `00_input/model.json` as today. When the molecule file has no
   `[ defaults ]` but does have an `#include` line, or `--ff-include` was given, it logs
   "`[ defaults ]` is resolved in 02_mm_scan" instead of warning. The existing warning stays
   for a file with neither.
2. **Resolve**, before the `--resume` check:
   - write `02_mm_scan/full.itp` (include mode `absolute`) and its wrapper `full.top`;
   - `gmx grompp -f resolve.mdp -c resolve_conf.gro -p full.top -pp full_pp.top
     -o resolve.tpr -maxwarn 5`, with the large-box fallback mdp: bonded parameters do not
     depend on the mdp, and that mdp is accepted by every supported version;
   - `gmx dump -s resolve.tpr`, stdout kept as `resolve_dump.txt`;
   - parse and cross-check (section 6). About one second on PFOA with CHARMM36.
3. The resume key is today's inputs and settings plus `pp_sha256`. Editing
   `pfo_ffbonded.itp`, `ffnonbonded.itp` or anything else grompp opens now makes the stage
   stale. The resolve step therefore runs even when `--resume` then skips the stage.
4. The self-check's full and zeroed reruns run as today. `full.itp` is the file written in
   step 2; `zeroed.itp` is written in mode `absolute`. The analytic reference is
   `torsion_set_energy(resolved.terms, frames)`.
5. `energies.json` gains:
   - `torsion_parameters`: one entry per term, `{"quad_1_based", "funct", "phi0_deg",
     "k_kj_per_mol", "multiplicity", "source"}`, where `source` is `"topology"` or
     `"dihedraltypes"`. Milestone 6b's report reads its "before" parameters here.
   - `defaults`: `{"fudgeQQ", "fudgeLJ", "gen_pairs", "source": "grompp -pp"}`.
   - `preprocessed_topology_sha256`.

### Chained `run` with charges

1. `04_charges/molecule_charged.itp` is written in mode `verbatim`: the user's include lines
   unchanged, plus the marker naming the input's directory.
2. `scan` never runs gmx and never looks at includes.
3. `mmscan` reads the base from the marker, not from `04_charges/`.
4. `fit` writes `03_fit/dihedral.itp` in mode `verbatim`; a marker already present in the
   source is replaced, never duplicated.

### `validate`

- `final/before.itp` and `final/after.itp` go to grompp: mode `absolute`.
- `final/molecule_refined.itp` is written by the same refit writer in mode `verbatim`,
  instead of `shutil.copyfile(after.itp, molecule_refined.itp)`. For a topology without
  relative includes (butane) the two modes produce identical bytes.
- The resume key adds `preprocessed_sha256` of `before.top`.

### Unchanged

- `00_input/` copies stay byte-for-byte copies of what the user passed.
- `--ff-include` keeps its current meaning and absolutisation (`_ff_includes`).

---

## 5. Include handling rules

### Which includes are rewritten

A line matching `^\s*#\s*include\s+"(?P<path>[^"]+)"` whose path is relative and names an
existing file under `base` is rewritten, in mode `absolute`, to
`#include "<base/path, resolved>"`. Everything else is copied verbatim:

- `#include <...>`: grompp searches its include path and GMXLIB;
- an absolute path;
- a relative path that does not exist under `base`: grompp's own search (the including
  file's directory, then GMXLIB) is left to do what it would have done for the original,
  which is the order cpp uses. Logged at info level, with the base named.

Only lines in the file ffrefine writes are rewritten. Includes nested inside included files
resolve against their own directories, as they always did.

### The include-base marker

Files the user keeps (mode `verbatim`) carry their include lines unchanged, so they work as a
drop-in replacement beside the original. Written anywhere else, those relative lines no longer
say where the force field is, so the file records it in two comment lines at the top:

```
; ffrefine: relative #include paths in this file are written for the directory below
; ffrefine include-base: /mnt/c/Users/hrach/Projects/ffrefine/fixtures/pfoa_gromacs
```

- Written only when the file has at least one relative include that resolved under the base.
  Butane's outputs therefore stay byte-identical to today's.
- `include_base(path)` returns the marked directory when the file carries a marker, otherwise
  the file's own resolved directory. A marker naming a missing directory is an error (section
  7), never a silent fallback: falling back could resolve a different `charmm36.ff/`.
- Two markers naming different directories are an error; a writer replaces an existing
  marker rather than adding a second.

This is what lets a later standalone `ffrefine mmscan -p out/04_charges/molecule_charged.itp`
find the force field without a new flag -- the scenario behind milestone 5's R28.

---

## 6. The decisions that carry real risk

### 6.1 Parameters come from the tpr, not from a reimplemented lookup

GROMACS's `[ dihedraltypes ]` matching (most non-wildcard matches, first in file order, both
orientations, consecutive-line blocks for funct 9, later definitions overriding earlier ones,
bonded types from `[ atomtypes ]`, two-type entries) is not reimplemented. grompp resolves it;
ffrefine reads the result. The self-check keeps its job, verifying that the zeroed copy
removed exactly those terms and nothing else. It no longer re-derives the parameters
independently, and does not need to: they are the ones `mdrun` uses.

The cost is a dependency on `gmx dump`'s text layout. The parser is strict, and fails loudly
rather than guessing (section 7).

### 6.2 Parsing

- `functype[N]=PDIHS, phiA= a, cpA= b, phiB= c, cpB= d, mult=m`: only `PDIHS` functypes are
  kept. `phiB`/`cpB` differing from `phiA`/`cpA` is a free-energy topology and is refused.
- The molecule block is the one `moltype (n):` whose `name="..."` equals the topology's
  `[ moleculetype ]` name; exactly one must match, and its `atom (N):` count must equal the
  topology's atom count.
- Inside it, the `Proper Dih.:` block's `nr:` must equal the number of
  `i type=T (PDIHS) a b c d` lines read, and every `T` must be a parsed `PDIHS` functype.

### 6.3 Cross-check against the torsion set

T is today's `collect_torsion_set`: every proper `[ dihedrals ]` line whose central bond
matches the target's. Its lines are grouped by quad as written (`ai aj ak al`). For each quad
`q`, with explicit lines `E_q`, typed lines `Y_q` and dumped terms `D_q` (iatoms equal to `q`,
converted to 1-based):

| case | rule | resolved terms |
|---|---|---|
| `Y_q` empty | the non-zero-`k` lines of `E_q` must equal `D_q` as a multiset | `E_q` as written, source `topology` |
| one typed line, `E_q` empty | none | `D_q`, source `dihedraltypes`; empty is legitimate (all terms `k = 0`) |
| both non-empty, or more than one typed line | refused as ambiguous | none |

Then, in the other direction, every dumped `PDIHS` term whose central bond matches T's must
belong to some `q` in T.

Equality uses the float32 numbers grompp stores: multiplicity exact,
`|Δphi0| ≤ 1e-4 deg`, `|Δk| ≤ 1e-6 · max(1, |k|)`.

Explicit lines keep their text values in the analytic reference, so butane's self-check
arithmetic is unchanged. Only typed lines take float32 values from the dump.

### 6.4 Trimmed fixtures resolve identically

A `[ dihedraltypes ]` entry naming an atom type that does not occur in the molecule matches
none of its quads, so removing it cannot change which entry grompp picks. The same holds for
every type-keyed section. The one-off equivalence check (section 8) confirms it on both
molecules rather than relying on the argument.

---

## 7. Error handling

Every case below is a `ValidationError` or `ExternalToolError` raised before any fit, naming
the file to look at.

- **Unrecognised dump layout** (no matching moltype, no `Proper Dih.` block, malformed
  functype or iatoms line, count mismatch): names `resolve_dump.txt` and the GROMACS version,
  and says this is a parse boundary, not a problem with the topology.
- **Free-energy topology** (B state differs): refused as unsupported.
- **Explicit line disagrees with the dump:** both values are printed. Either the parser
  misread the dump, or grompp did not read the line as written -- for example an `#ifdef` in
  the molecule file driven by a define set inside an included force-field file, which
  `read_topology` cannot see.
- **Dumped term about the central bond with no matching line:** the same diagnosis.
- **Ambiguous quad** (typed and explicit, or typed twice): refused, naming the lines.
- **Typed line grompp could not resolve:** grompp's own "No default Proper Dih. types" error,
  as today, through `ExternalToolError` with its log path.
- **Include not found:** grompp's error, with ffrefine naming the base it used and whether
  that base came from the file's directory or its marker.
- **Marker naming a missing directory, or two different markers:** refused; the remedy is to
  re-run from the original topology or move the force field back.
- **Self-check failure message:** "every term about this bond carries explicit parameters" is
  removed. The message lists each removed term's source and points at `resolve_dump.txt` as
  well as `zeroed.itp`.
- **Old outdirs:** their manifest hashes no longer match, so `--resume` recomputes. Nothing
  downstream in 6a requires the new `energies.json` fields.

---

## 8. Fixtures

```
fixtures/pfoa_gromacs/            pfoa.str, pfoa_gmx.top, pfoa_gmx.pdb   (verbatim)
                                  charmm36.ff/                           (trimmed)
                                  PROVENANCE.md
fixtures/capecitabine_gromacs/    capecitabine.str, capecitabine_gmx.top,
                                  capecitabine_gmx.pdb                   (verbatim)
                                  charmm36.ff/                           (trimmed)
                                  PROVENANCE.md
fixtures/_full_ff/<mol>/          the supplied full copies, gitignored, moved not deleted
scripts/trim_forcefield.py
```

`scripts/trim_forcefield.py <full ff dir> <molecule .top> <out ff dir>`:

- Collects the atom types of every `[ atoms ]` the `.top` pulls in, following its includes:
  the molecule, TIP3P and the ions. grompp checks those moleculetypes' types even though
  `[ molecules ]` never lists them.
- Copies every file grompp opens: `forcefield.itp` verbatim; `ffnonbonded.itp`,
  `ffbonded.itp`, `gb.itp`, `cmap.itp`, `old_c36_cmap.itp`, `nbfix.itp`, `silcs.itp`,
  `tip3p.itp`, `ions.itp` and `<mol>_ffbonded.itp`, trimmed.
- In `[ atomtypes ]`, `[ bondtypes ]`, `[ constrainttypes ]`, `[ angletypes ]`,
  `[ dihedraltypes ]`, `[ pairtypes ]`, `[ nonbond_params ]`, `[ cmaptypes ]` and
  `[ implicit_genborn_params ]`, keeps an entry only when every atom-type field is in that
  set or is `X`. `[ cmaptypes ]` entries run over `\`-continued lines and are kept or dropped
  whole. Comments, directives and other sections are copied through.
- Drops the pdb2gmx-only files (`merged.rtp`, `ic_info.txt`, `mol/`, `.hdb`, `.tdb`, `.r2b`,
  `.arn`, `.vsd`, `<mol>.rtp`, and the rest grompp never opens).
- Is deterministic, and idempotent: trimming a trimmed tree changes nothing.

`PROVENANCE.md` records the CGenFF and CHARMM36 versions from the file headers, the command
that trimmed the force field, and the result of the equivalence check: for each molecule, the
full-ff and trimmed-ff tprs dumped and diffed over the molecule's functypes, iatoms, pairs and
LJ parameters, which must be identical. That check is a one-off implementation step, not a
test, because a test needing `_full_ff/` would skip on any other machine.

---

## 9. Testing

### Unit, no binaries

- `includes.py`: each rewrite rule; `# include` with a space; `<...>` untouched; a missing
  target left verbatim; an absolute path untouched; marker written only when a relative
  include resolved; marker replaced, not duplicated; marker naming a missing directory and
  two conflicting markers refused.
- Dump parser, against a committed `gmx dump` of the PFOA fixture captured from GROMACS
  2025.4, plus text-edited variants for every refusal in section 7.
- Cross-check, on synthetic data: each row of the section 6.3 table, the reverse-direction
  check, `k = 0` explicit terms absent from the dump, and a typed line resolving to no terms.
- Writers: `absolute` copies carry absolute includes; `verbatim` copies carry the user's
  lines plus the marker; butane's outputs are byte-identical to master's in both modes;
  `molecule_refined.itp` is no longer `after.itp` when the topology has relative includes.
- `model.json`: the `[ defaults ]` warning is deferred when there are includes or
  `--ff-include`, and still raised when there are neither.
- Resume keys: a changed `pp_sha256` makes `mmscan` and `validate` stale, with the resolve
  step stubbed.
- `trim_forcefield.py`: idempotent; keeps `X` entries and continued cmap entries whole; drops
  an entry with one foreign type.

### Live, `needs_gmx`

- PFOA `mmscan` on the input frame, `--dihedral 25,1,3,4` (`OG311 CG2O2 CG312 CG312`,
  penalty 67): the self-check passes within 1e-3 kJ/mol; `energies.json` lists four
  `dihedraltypes` terms (multiplicities 1, 2, 3, 6) for the target quad, and defaults
  1.0/1.0.
- Capecitabine `mmscan`, `--dihedral 6,7,9,10` (`OG302 CG2O6 NG2S1 CG2R64`, penalty 50),
  the same way.
- The PFOA case again from a `write_charged_topology` copy in another directory: proves the
  marker without ORCA.
- Editing the trimmed `pfo_ffbonded.itp` in a copy of the fixture makes `mmscan --resume`
  recompute.
- The existing butane live tests keep passing, and now also exercise the explicit-line
  cross-check.

### Live, `needs_gmx` and `needs_xtb`

- `ffrefine run` on PFOA, `--dihedral 25,1,3,4`, without charges: it completes, and
  `final/molecule_refined.itp`, copied beside the original `.top` and `charmm36.ff/`,
  passes `grompp` -- the drop-in proof.

The whole suite runs in WSL with `bash -ic` inside the `gromacs` conda env, with zero skips.

---

## 10. Open questions carried into planning

- Whether the `gmx dump` capture committed for the parser tests should be the full PFOA dump
  (about 15 000 lines) or an excerpt holding only what the parser reads. The excerpt keeps the
  repo small; the full dump proves the parser skips everything else. Decide on the file size.
- Whether `resolve_torsion_set` should share the grompp call that `single_point_energies`
  makes for the full topology instead of running its own. A separate call is simpler and
  fails before any rerun is spent; sharing saves about a second.
