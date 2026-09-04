"""Emit modified topologies without touching the input (spec section 4.4).

Rewriting is line-oriented: everything the parser does not model -- comments,
[ atomtypes ], #include directives -- is copied through verbatim, and only the
[ dihedrals ] lines belonging to the target set are replaced.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..mm.torsionset import collect_torsion_set
from .topology import Dihedral, Topology

Quad = tuple[int, int, int, int]

ZEROED_TEMPLATE = "{ai:6d}{aj:6d}{ak:6d}{al:6d}{funct:7d}{phi0:12.4f}{k:12.5f}{mult:6d}"

_EntryKey = tuple[int, int, int, int, int]


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

    Each removed quadruplet becomes a single explicit line
    ``i j k l 9 0.000 0.000 1``; a target originally written as several
    funct-9 lines (multiple multiplicities) collapses to one zero line.
    Explicit parameters suppress the [ dihedraltypes ] lookup -- a claim the
    section 4.5 self-check verifies at runtime rather than trusting.

    [ pairs ] is never touched (spec section 9.2).

    Args:
        topology: parsed source topology.
        quad: 1-based target quadruplet.
        out_path: destination; the input is never modified.

    Returns:
        One Dihedral entry per distinct quadruplet that was removed (i.e. one
        per zero line written), for the self-consistency check.
    """
    tset = collect_torsion_set(topology, quad)
    removed: dict[_EntryKey, Dihedral] = {}
    for d in tset:
        removed[_entry_key(d)] = d
    targets = set(removed.keys())
    replacement = {
        key: [
            _format_line(
                key[0], key[1], key[2], key[3], 9, 0.0, 0.0, 1, comment="zeroed by ffrefine"
            )
        ]
        for key in targets
    }
    lines = _rewrite_dihedrals(topology, targets, replacement)
    Path(out_path).write_text("\n".join(lines) + "\n")
    return list(removed.values())


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
    tset = collect_torsion_set(topology, quad)
    targets = {_entry_key(d) for d in tset}
    forward, reverse = tuple(quad), tuple(reversed(quad))
    n_t = len(targets)

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

    lines = _rewrite_dihedrals(topology, targets, replacement)
    Path(out_path).write_text("\n".join(lines) + "\n")


def make_system_top(
    itp_path: str | Path,
    out_path: str | Path,
    molname: str,
    includes: list[str] | None = None,
) -> None:
    """Write a minimal .top wrapping an .itp so grompp can process it.

    Args:
        itp_path: molecule .itp; referenced by name, so keep it beside the .top.
        out_path: destination .top.
        molname: the [ moleculetype ] name.
        includes: force field includes to emit before the molecule include.
    """
    lines = ["; generated by ffrefine -- do not edit", ""]
    for inc in includes or []:
        lines.append(f'#include "{inc}"')
    lines.append(f'#include "{Path(itp_path).name}"')
    lines += ["", "[ system ]", f"{molname} in vacuum", "", "[ molecules ]", f"{molname}    1", ""]
    Path(out_path).write_text("\n".join(lines))
