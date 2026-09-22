# PFOA fixture

Perfluorooctanoic acid, neutral (COOH), as the CGenFF server returned it.
Supplied by the project owner, 2026-09-22.

| File | Origin |
|---|---|
| `pfoa.str` | CGenFF program 4.0, parameters 5.0; verbatim |
| `pfoa_gmx.top`, `pfoa_gmx.pdb` | the CGenFF server's GROMACS export (generated 2026-05-27); verbatim |
| `charmm36.ff/` | CHARMM36 November 2018 GROMACS port (`forcefield.doc`) plus the export's `pfo_ffbonded.itp`, **trimmed** |
| `gromacs/pfoa_dump_2025.4.txt` | `gmx dump -s` of a tpr built from this directory with GROMACS 2025.4 |

## The trim

    python scripts/trim_forcefield.py <full charmm36.ff> fixtures/pfoa_gromacs/pfoa_gmx.top fixtures/pfoa_gromacs/charmm36.ff

keeps every file grompp opens and, in each type-keyed section, only the entries whose atom
types all occur in the molecule, TIP3P or the ions (or are `X`). The pdb2gmx-only files are
not kept. `tests/test_trim_forcefield.py` checks that trimming this tree again changes nothing.

Equivalence, checked 2026-09-22 with GROMACS 2025.4: tprs built from the full and the trimmed
force field, compared with `scripts/compare_tpr_dumps.py` (per atom, per atom pair and per
interaction, never per type index): **identical, 590 lines**.

## Notes for milestone 6b

- The export renamed the atoms: `.str` `O1` is `.top` atom 2 (`O2`), `.str` `C2` is atom 3
  (`C3`), and so on. Order, types and charges agree index for index.
- Dihedral penalties above 50 sit on the `.str` bonds C1-C2 and C2-C3, i.e. `.top` atoms 1-3
  and 3-4. 23 of the 26 atoms carry a charge penalty of 10 or more.
