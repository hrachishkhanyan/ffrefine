# Capecitabine fixture

Capecitabine, as the CGenFF server returned it. Supplied by the project owner, 2026-09-22.

| File | Origin |
|---|---|
| `capecitabine.str` | CGenFF program 4.0, parameters 5.0; verbatim |
| `capecitabine_gmx.top`, `capecitabine_gmx.pdb` | the CGenFF server's GROMACS export (generated 2026-06-01); verbatim |
| `charmm36.ff/` | CHARMM36 November 2018 GROMACS port (`forcefield.doc`) plus the export's `cap_ffbonded.itp`, **trimmed** |

## The trim

    python scripts/trim_forcefield.py <full charmm36.ff> fixtures/capecitabine_gromacs/capecitabine_gmx.top fixtures/capecitabine_gromacs/charmm36.ff

Equivalence, checked 2026-09-22 with GROMACS 2025.4, with `scripts/compare_tpr_dumps.py`:
**identical, 1570 lines**. grompp numbers the atom types differently (`atnr` 21 with the full
force field, 19 trimmed): it merges types whose nonbonded rows are identical, judged against
every type present, and the full field's NBFIX pairs with types this molecule never uses keep
two of its types apart. No parameter any atom or interaction sees changes, which is what the
comparison checks.

## Notes for milestone 6b

- The only dihedral at the default penalty threshold is `OG302 CG2O6 NG2S1 CG2R64`, at
  exactly 50: `.top` quad `6 7 9 10`. Every charge penalty is below 10.
- `6 7 9 37` and `8 7 9 37` resolve to CGenFF terms that all have `k = 0`; grompp builds no
  term for them.
