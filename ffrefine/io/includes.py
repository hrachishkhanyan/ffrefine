"""How the topologies ffrefine writes spell their ``#include`` lines.

Milestone 6a design, section 5. Two modes:

``absolute``
    For the copies grompp reads (``02_mm_scan/full.itp``, ``zeroed.itp``,
    ``final/before.itp``, ``after.itp``). They live in stage directories,
    where a relative include such as ``./charmm36.ff/forcefield.itp`` would
    no longer resolve, so every relative include that names an existing file
    under the include base is rewritten to its absolute path.

``verbatim``
    For the files the user keeps (``03_fit/dihedral.itp``,
    ``04_charges/molecule_charged.itp``, ``final/molecule_refined.itp``). The
    user's own include lines are kept, so each file is a drop-in replacement
    beside the original, and two comment lines at the top record the
    directory those lines were written for. That marker is how a later stage,
    or a later standalone command, finds the force field again.

The include base of a file is the directory its marker names, or the file's
own directory when it carries no marker. Only the lines of the file being
written are rewritten; includes nested inside included files resolve against
their own directories, as they always did.

Pure text: this module never starts a process.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

from ..errors import ParseError, ValidationError

log = logging.getLogger("ffrefine")

IncludeMode = Literal["absolute", "verbatim"]

MARKER_NOTE = "; ffrefine: relative #include paths in this file are written for the directory below"
MARKER_PREFIX = "; ffrefine include-base: "

_INCLUDE_RE = re.compile(r'^(?P<lead>\s*#\s*include\s*)"(?P<path>[^"]+)"(?P<tail>.*)$')
_ANY_INCLUDE_RE = re.compile(r"^\s*#\s*include\b")


def has_includes(lines: list[str]) -> bool:
    """Whether any line is an ``#include`` directive (quoted or angle-bracketed)."""
    return any(_ANY_INCLUDE_RE.match(line) for line in lines)


def find_marker(lines: list[str], source: str | Path) -> Path | None:
    """The include base a file's marker records, or ``None`` without one.

    Raises:
        ParseError: the file carries two markers naming different directories.
    """
    values = {
        line[len(MARKER_PREFIX) :].strip() for line in lines if line.startswith(MARKER_PREFIX)
    }
    if len(values) > 1:
        raise ParseError(
            f"{source}: two ffrefine include-base markers name different directories "
            f"({', '.join(sorted(values))}); keep the one the relative #include lines were "
            "written for and delete the other"
        )
    return Path(values.pop()) if values else None


def resolve_include_base(source: str | Path, marker: Path | None) -> Path:
    """The directory ``source``'s relative includes resolve against.

    Raises:
        ValidationError: the marker names a directory that does not exist.
            Falling back to the file's own directory could resolve a
            different force field, so this is never done silently.
    """
    if marker is None:
        return Path(source).resolve().parent
    if not marker.is_dir():
        raise ValidationError(
            f"{source} says its relative #include paths are written for {marker}, which "
            "does not exist. Re-run from the original topology, or move the force field "
            f"back to {marker}."
        )
    return marker


def include_base(path: str | Path) -> Path:
    """Read ``path`` and return the directory its relative includes resolve against."""
    lines = Path(path).read_text().splitlines()
    return resolve_include_base(path, find_marker(lines, path))


def strip_marker(lines: list[str]) -> list[str]:
    """``lines`` without ffrefine's include-base marker."""
    return [
        line
        for line in lines
        if line.rstrip() != MARKER_NOTE and not line.startswith(MARKER_PREFIX)
    ]


def _resolved_target(line: str, base: Path) -> Path | None:
    """The absolute file a relative quoted include names under ``base``, if it exists."""
    m = _INCLUDE_RE.match(line)
    if not m or Path(m.group("path")).is_absolute():
        return None
    candidate = Path(base) / m.group("path")
    return candidate.resolve() if candidate.is_file() else None


def absolutise_includes(lines: list[str], base: Path) -> list[str]:
    """Mode ``absolute``: rewrite every relative include that exists under ``base``.

    Angle-bracketed includes, absolute paths and relative paths with no file
    under ``base`` are copied verbatim; grompp then searches for them exactly
    as it would have for the original (the including file's directory, then
    GMXLIB). A marker is dropped: an absolute copy no longer needs one.
    """
    out: list[str] = []
    for line in strip_marker(lines):
        m = _INCLUDE_RE.match(line)
        target = _resolved_target(line, base)
        if target is not None:
            line = f'{m.group("lead")}"{target}"{m.group("tail")}'
        elif m and not Path(m.group("path")).is_absolute():
            log.info(
                "%s: no such file under %s; left for grompp's own include search",
                m.group("path"),
                base,
            )
        out.append(line)
    return out


def mark_include_base(lines: list[str], base: Path) -> list[str]:
    """Mode ``verbatim``: keep the user's include lines and record their base.

    The marker is written only when at least one relative include resolves
    under ``base``, so a topology without includes (the butane fixture) comes
    out unchanged. An existing marker is replaced, never duplicated.
    """
    body = strip_marker(lines)
    if not any(_resolved_target(line, base) is not None for line in body):
        return body
    return [MARKER_NOTE, f"{MARKER_PREFIX}{Path(base).resolve()}", *body]


def apply_include_mode(lines: list[str], base: Path, mode: IncludeMode) -> list[str]:
    """Dispatch to :func:`absolutise_includes` or :func:`mark_include_base`."""
    if mode == "absolute":
        return absolutise_includes(lines, base)
    if mode == "verbatim":
        return mark_include_base(lines, base)
    raise ValueError(f"unknown include mode {mode!r}; expected 'absolute' or 'verbatim'")
