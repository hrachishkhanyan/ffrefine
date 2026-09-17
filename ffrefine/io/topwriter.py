"""Emit modified topologies without touching the input (spec section 4.4).

Rewriting is line-oriented: everything the parser does not model -- comments,
[ atomtypes ], #include directives -- is copied through verbatim, and only the
[ dihedrals ] lines belonging to the target set are replaced.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..errors import ValidationError
from ..mm.torsionset import EntryKey as _EntryKey
from ..mm.torsionset import collect_torsion_set, distinct_entry_keys
from .topology import Dihedral, Topology

if TYPE_CHECKING:
    # ffrefine/io/ is the lowest layer; ffrefine/charges/ builds on top of it
    # (charges/apply.py already imports io.topology and io.validate), so this
    # stays a type-checking-only import to avoid inverting that layering.
    from ..charges.apply import ChargeAssignment

log = logging.getLogger("ffrefine")

Quad = tuple[int, int, int, int]

ZEROED_TEMPLATE = "{ai:6d}{aj:6d}{ak:6d}{al:6d}{funct:7d}{phi0:12.4f}{k:12.5f}{mult:6d}"


_WRAPPER_SECTIONS = frozenset({"system", "molecules"})
"""Sections :func:`make_system_top` supplies itself, so a molecule copy must
not also carry them."""


def strip_system_sections(lines: list[str]) -> tuple[list[str], list[str]]:
    """Drop ``[ system ]`` and ``[ molecules ]`` blocks from a topology's lines.

    Spec section 2 promises ``-p`` accepts "``.itp`` or ``.top`` for the
    molecule", and a molecule ``.top`` normally ends with those two sections.
    Every grompp-facing copy this module writes is wrapped by
    :func:`make_system_top`, which appends its own pair; without this strip the
    wrapped system declares the molecule twice and grompp rejects it with
    ``number of coordinates in coordinate file (14) does not match topology
    (28)``.

    Applied to the written outputs as well as the grompp inputs, so a file
    ffrefine names ``.itp`` really is a molecule include file.  The user's own
    input is never modified.

    Args:
        lines: topology lines, comments included.

    Returns:
        ``(kept_lines, dropped_section_names)``.  ``dropped_section_names``
        is empty for a plain molecule ``.itp``, which is the common case.
    """
    kept: list[str] = []
    dropped: list[str] = []
    skipping = False
    for raw in lines:
        body = raw.split(";", 1)[0].strip()
        if body.startswith("[") and body.endswith("]"):
            name = body[1:-1].strip().lower()
            skipping = name in _WRAPPER_SECTIONS
            if skipping:
                dropped.append(name)
                continue
        if skipping:
            continue
        kept.append(raw)
    return kept, dropped


def write_molecule_copy(source: str | Path, out_path: str | Path) -> list[str]:
    """Copy a topology file, dropping the sections ``make_system_top`` supplies.

    Used for the "unmodified" grompp inputs (the section 4.5 full topology and
    the section 4.9 before topology), which are otherwise byte-for-byte copies
    of the user's file.  Reads the *source text*, not the parsed
    ``raw_lines``, so ``#ifdef`` branches this build did not take are still
    handed to grompp, which does its own preprocessing.

    Args:
        source: the user's topology file; never modified.
        out_path: destination.

    Returns:
        The names of any sections dropped.
    """
    lines, dropped = strip_system_sections(Path(source).read_text().splitlines())
    Path(out_path).write_text("\n".join(lines) + "\n", newline="\n")
    if dropped:
        log.info(
            "%s: dropped [ %s ] from the molecule copy %s; the generated wrapper .top "
            "supplies its own (spec section 2: -p accepts a molecule .top)",
            Path(source).name,
            " ] [ ".join(dropped),
            Path(out_path).name,
        )
    return dropped


def _entry_key(d: Dihedral) -> _EntryKey:
    """Identity of a dihedral line for matching during rewriting.

    A funct-9 torsion written as several lines (one per retained
    multiplicity) shares a single key here, since all of those lines describe
    the same physical quadruplet.
    """
    return (d.ai, d.aj, d.ak, d.al, d.funct)


def _format_line(
    ai: int,
    aj: int,
    ak: int,
    al: int,
    funct: int,
    phi0: float,
    k: float,
    mult: int,
    comment: str = "",
) -> str:
    line = ZEROED_TEMPLATE.format(
        ai=ai, aj=aj, ak=ak, al=al, funct=funct, phi0=phi0, k=k, mult=int(mult)
    )
    return f"{line}  ; {comment}" if comment else line


def _rewrite_dihedrals(
    topology: Topology, targets: set[_EntryKey], replacement: dict[_EntryKey, list[str]]
) -> list[str]:
    """Copy the source file, substituting the dihedral lines named in ``targets``.

    Args:
        topology: parsed source (supplies ``raw_lines``).
        targets: entry keys to replace or drop.
        replacement: entry key -> list of replacement lines (may be empty to drop).

    Returns:
        The output file as a list of lines.
    """
    out: list[str] = []
    section = ""
    emitted: set[_EntryKey] = set()
    for raw in topology.raw_lines:
        body = raw.split(";", 1)[0].strip()
        if body.startswith("[") and body.endswith("]"):
            section = body[1:-1].strip().lower()
            out.append(raw)
            continue
        if section != "dihedrals" or not body:
            out.append(raw)
            continue
        fields = body.split()
        try:
            key = (int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3]), int(fields[4]))
        except (IndexError, ValueError):
            out.append(raw)
            continue
        if key not in targets:
            out.append(raw)
            continue
        if key not in emitted:
            out.extend(replacement.get(key, []))
            emitted.add(key)
        # Subsequent lines of a multi-line funct-9 term are dropped; the
        # replacement already carries every retained multiplicity.
    return out


def write_zeroed_topology(topology: Topology, quad: Quad, out_path: str | Path) -> list[Dihedral]:
    """Write a copy of the topology with every member of T set to zero.

    Each distinct quadruplet in T becomes a single explicit output line
    ``i j k l 9 0.000 0.000 1``; a target originally written as several
    funct-9 lines (multiple multiplicities) collapses to that one zero line.
    Explicit parameters suppress the [ dihedraltypes ] lookup -- a claim the
    section 4.5 self-check verifies at runtime rather than trusting.

    [ pairs ] is never touched (spec section 9.2).

    Args:
        topology: parsed source topology.
        quad: 1-based target quadruplet.
        out_path: destination; the input is never modified.

    Returns:
        Every removed *term-line* of T, in topology order -- i.e. exactly
        ``collect_torsion_set(topology, quad)``, with duplicates for a
        multi-line funct-9 torsion **not** collapsed. This is one entry per
        removed term-line (10 for the butane fixture), not one per distinct
        quadruplet (9): the section 4.5 self-consistency check needs the
        analytic sum of every original term -- including both multiplicities
        of a split torsion -- to match GROMACS's ``E_full - E_zeroed``, and
        dropping either term-line of such a split would silently omit its
        contribution from that sum. The *output file* still gets only one
        collapsed null line per quadruplet; only this returned list is
        per-term-line.
    """
    tset = collect_torsion_set(topology, quad)
    targets = {_entry_key(d) for d in tset}
    replacement = {
        key: [
            _format_line(
                key[0], key[1], key[2], key[3], 9, 0.0, 0.0, 1, comment="zeroed by ffrefine"
            )
        ]
        for key in targets
    }
    lines, _ = strip_system_sections(_rewrite_dihedrals(topology, targets, replacement))
    Path(out_path).write_text("\n".join(lines) + "\n", newline="\n")
    return tset


def write_refit_topology(
    topology: Topology,
    quad: Quad,
    k: np.ndarray,
    delta_deg: np.ndarray,
    mult: np.ndarray,
    out_path: str | Path,
    distribute: bool = False,
) -> None:
    """Write the topology carrying the fitted torsion series (spec section 4.2).

    Args:
        topology: parsed source topology.
        quad: 1-based target quadruplet.
        k: fitted force constants in kJ/mol.
        delta_deg: fitted phases in degrees.
        mult: retained multiplicities.
        out_path: destination.
        distribute: spread k_n over all N_T members as k_n / N_T, giving a
            transferable dihedral type (spec section 9.7). Default writes the
            series onto the target quadruplet only and zeroes the rest, which
            is exact for this molecule but molecule-specific.
    """
    # N_T is the number of distinct *quadruplets*, not term-lines (9 vs 10 on
    # the butane fixture; rulings R14/R17).  It comes from the shared
    # torsionset helper so the CHARMM .prm block written by the same fit gets
    # the identical denominator.
    keys = distinct_entry_keys(topology, quad)
    targets = set(keys)
    forward, reverse = tuple(quad), tuple(reversed(quad))
    n_t = len(keys)

    replacement: dict[_EntryKey, list[str]] = {}
    for key in targets:
        ai, aj, ak, al, _ = key
        is_target = (ai, aj, ak, al) in (forward, reverse)
        if distribute:
            scale, note = 1.0 / n_t, f"ffrefine fit, distributed over {n_t} terms"
        elif is_target:
            scale, note = 1.0, "ffrefine fit"
        else:
            replacement[key] = [
                _format_line(
                    ai,
                    aj,
                    ak,
                    al,
                    9,
                    0.0,
                    0.0,
                    1,
                    comment="zeroed by ffrefine (fit carried by target quad)",
                )
            ]
            continue
        replacement[key] = [
            _format_line(ai, aj, ak, al, 9, float(d), float(kk) * scale, int(n), comment=note)
            for kk, d, n in zip(np.atleast_1d(k), np.atleast_1d(delta_deg), np.atleast_1d(mult))
        ]

    lines, _ = strip_system_sections(_rewrite_dihedrals(topology, targets, replacement))
    Path(out_path).write_text("\n".join(lines) + "\n", newline="\n")


def make_system_top(
    itp_path: str | Path,
    out_path: str | Path,
    molname: str,
    includes: list[str] | None = None,
) -> None:
    """Write a minimal .top wrapping an .itp so grompp can process it.

    The included molecule file must NOT carry ``[ system ]`` or
    ``[ molecules ]`` of its own -- this function supplies both, and a
    duplicate makes grompp build the molecule twice.  Every caller therefore
    routes its copy through :func:`write_molecule_copy`,
    :func:`write_zeroed_topology` or :func:`write_refit_topology`, all of
    which strip those sections.

    Args:
        itp_path: molecule .itp; referenced by name, so keep it beside the .top.
        out_path: destination .top.
        molname: the [ moleculetype ] name.
        includes: force field includes to emit before the molecule include,
            in order, each as written by the user's ``--ff-include``. Paths
            are emitted verbatim so a GMXLIB-relative name such as
            ``charmm36.ff/forcefield.itp`` still resolves; the CLI absolutises
            anything that names a file it can see, because this wrapper lives
            in the stage directory rather than the user's cwd.
    """
    lines = ["; generated by ffrefine -- do not edit", ""]
    for inc in includes or []:
        lines.append(f'#include "{inc}"')
    lines.append(f'#include "{Path(itp_path).name}"')
    lines += ["", "[ system ]", f"{molname} in vacuum", "", "[ molecules ]", f"{molname}    1", ""]
    Path(out_path).write_text("\n".join(lines), newline="\n")


def write_charged_topology(
    source: str | Path,
    assignments: list["ChargeAssignment"],
    out_path: str | Path,
    decimals: int = 4,
) -> list[int]:
    """Copy a topology, rewriting only the charge column of changed atoms.

    Reads the source *text* rather than the parsed ``raw_lines``, for the same
    reason :func:`write_molecule_copy` does: ``#ifdef`` branches this build did
    not take must still reach grompp, which runs its own preprocessor.

    Only the charge field is touched.  Atom type, residue, charge group and
    mass are left exactly as written, including their spacing, so a diff
    against the input shows the charges and nothing else.

    Args:
        source: the user's topology; never modified.
        assignments: from :func:`ffrefine.charges.apply.apply_charges`.  Entries
            whose charge did not move are skipped.
        out_path: destination.
        decimals: decimal places written into the charge field.  Defaults to
            :data:`ffrefine.charges.apply.CHARGE_DECIMALS`; pass that value
            (or a matching custom precision) explicitly if it ever changes,
            so the written charges stay consistent with what
            :func:`ffrefine.charges.apply.apply_charges` rounded to.

    Returns:
        The 1-based atom numbers whose lines were rewritten, in file order.

    Raises:
        ValidationError: an assignment names an atom that has no line in
            ``[ atoms ]``.  Skipping it silently would leave a topology whose
            charges no longer sum to the molecular charge.
    """
    wanted = {a.nr: a for a in assignments if a.after != a.before}
    lines = Path(source).read_text().splitlines()
    out: list[str] = []
    changed: list[int] = []
    section = ""
    for line in lines:
        body = line.split(";", 1)[0].strip()
        if body.startswith("[") and body.endswith("]"):
            section = body.strip("[] ").strip()
            out.append(line)
            continue
        if section != "atoms" or not body:
            out.append(line)
            continue
        fields = body.split()
        nr = int(fields[0]) if fields[0].isdigit() else None
        if nr is None or nr not in wanted:
            out.append(line)
            continue
        out.append(_replace_charge_field(line, wanted[nr].after, decimals))
        changed.append(nr)

    missing = sorted(set(wanted) - set(changed))
    if missing:
        raise ValidationError(
            f"{source}: no [ atoms ] line for atom(s) {missing}, so their new charges "
            "could not be written. The topology and the fitted charges disagree about "
            "how many atoms this molecule has."
        )
    Path(out_path).write_text("\n".join(out) + "\n", newline="\n")
    log.info("wrote %s with %d charge(s) changed", Path(out_path).name, len(changed))
    return changed


def _replace_charge_field(line: str, charge: float, decimals: int) -> str:
    """Rewrite the seventh whitespace field of an ``[ atoms ]`` line in place.

    Works on the character span of the existing field rather than reformatting
    the line, so the surrounding columns keep their original spacing and any
    trailing comment survives.
    """
    body, comment = (line.split(";", 1) + [""])[:2]
    fields = body.split()
    if len(fields) < 7:
        raise ValidationError(
            f"[ atoms ] line has {len(fields)} fields, expected at least 7: {line!r}"
        )
    old = fields[6]
    start = body.index(old, sum(len(f) for f in fields[:6]))
    new = f"{charge:.{decimals}f}".rjust(len(old))
    rebuilt = body[:start] + new + body[start + len(old) :]
    return rebuilt + (";" + comment if comment else "")
