"""Read a CGenFF stream file (``.str``): atom charges, parameters, penalties.

Milestone 6b design, section 4. CGenFF annotates every atom it assigned a
charge to with a *charge penalty* (the first number after ``!`` on its
``ATOM`` line) and every parameter it had to guess by analogy with
``penalty= X`` in the line's comment. Higher means less trustworthy; the
file's own header gives the bands (below 10 fair, 10 to 50 validate, above
50 poor).

Pure text: this module never starts a process.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..errors import ParseError

PARAMETER_KINDS: dict[str, tuple[str, int]] = {
    "BONDS": ("bond", 2),
    "ANGLES": ("angle", 3),
    "DIHEDRALS": ("dihedral", 4),
    "IMPROPERS": ("improper", 4),
}
"""Parameter section header -> (kind, number of leading atom-type fields)."""

_OTHER_PARAMETER_SECTIONS = frozenset({"NONBONDED", "NBFIX", "CMAP", "HBOND"})
_PENALTY_RE = re.compile(r"penalty=\s*(\S+)")
_RESI_RE = re.compile(r"param penalty=\s*([-+0-9.eE]+)\s*;\s*charge penalty=\s*([-+0-9.eE]+)")
_PROGRAM_RE = re.compile(r"CGenFF\)\s+program version\s+(\S+)")
_PARAMETER_VERSION_RE = re.compile(r"topology and parameter files version\s+(\S+)")


@dataclass(frozen=True)
class StrAtom:
    """One ``ATOM`` line. ``index`` is 1-based, in file order."""

    index: int
    name: str
    atype: str
    charge: float
    penalty: float


@dataclass(frozen=True)
class StrParameter:
    """One bonded parameter CGenFF assigned by analogy.

    ``kind`` is ``bond``, ``angle``, ``dihedral`` or ``improper``. A dihedral
    written over several multiplicity lines is one entry carrying the highest
    of their penalties.
    """

    kind: str
    types: tuple[str, ...]
    penalty: float


@dataclass
class StrFile:
    """A parsed ``.str`` file."""

    atoms: list[StrAtom]
    parameters: list[StrParameter]
    program_version: str | None
    parameter_version: str | None
    param_penalty: float | None
    charge_penalty: float | None
    source: Path

    def of_kind(self, kind: str) -> list[StrParameter]:
        """The parameters of one kind, in file order."""
        return [p for p in self.parameters if p.kind == kind]


def _number(text: str, path: Path, lineno: int, what: str) -> float:
    try:
        return float(text)
    except ValueError as exc:
        raise ParseError(f"{path}:{lineno}: {what} {text!r} is not a number") from exc


def read_str(path: str | Path) -> StrFile:
    """Parse a CGenFF ``.str`` file.

    The topology block runs from ``read rtf card`` to its ``END``; each
    ``ATOM name type charge ! penalty ...`` line is one atom. The parameter
    block runs from ``read param card`` to its ``END``; inside ``BONDS``,
    ``ANGLES``, ``DIHEDRALS`` and ``IMPROPERS`` each line gives its leading
    atom types and the number after ``penalty=`` (0 when absent).

    Raises:
        ParseError: the file cannot be read, holds no ``ATOM`` line, or has an
            ``ATOM`` or parameter line that does not parse.
    """
    path = Path(path)
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ParseError(f"{path}: could not be read: {exc}") from exc

    atoms: list[StrAtom] = []
    parameters: dict[tuple[str, tuple[str, ...]], float] = {}
    program_version = parameter_version = None
    param_penalty = charge_penalty = None
    block = ""
    section = ""
    for lineno, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if program_version is None and (m := _PROGRAM_RE.search(stripped)):
            program_version = m.group(1)
        if parameter_version is None and (m := _PARAMETER_VERSION_RE.search(stripped)):
            parameter_version = m.group(1)
        lowered = stripped.lower()
        if lowered.startswith("read rtf card"):
            block, section = "rtf", ""
            continue
        if lowered.startswith("read param card"):
            block, section = "param", ""
            continue
        if stripped.upper() == "END":
            block, section = "", ""
            continue
        if not stripped or stripped.startswith(("!", "*")):
            continue
        body = stripped.split("!", 1)[0].split()
        if block == "rtf":
            keyword = body[0].upper() if body else ""
            if keyword == "RESI" and (m := _RESI_RE.search(stripped)):
                param_penalty, charge_penalty = float(m.group(1)), float(m.group(2))
            elif keyword == "ATOM":
                comment = stripped.split("!", 1)[1].split() if "!" in stripped else []
                if len(body) < 4 or not comment:
                    raise ParseError(f"{path}:{lineno}: malformed ATOM line {raw!r}")
                atoms.append(
                    StrAtom(
                        index=len(atoms) + 1,
                        name=body[1],
                        atype=body[2],
                        charge=_number(body[3], path, lineno, "charge"),
                        penalty=_number(comment[0], path, lineno, "charge penalty"),
                    )
                )
        elif block == "param":
            header = stripped.split()[0].upper()
            if header in PARAMETER_KINDS or header in _OTHER_PARAMETER_SECTIONS:
                section = header
                continue
            if section not in PARAMETER_KINDS:
                continue
            kind, ntypes = PARAMETER_KINDS[section]
            if len(body) <= ntypes:
                raise ParseError(f"{path}:{lineno}: malformed {section} line {raw!r}")
            m = _PENALTY_RE.search(stripped)
            penalty = _number(m.group(1), path, lineno, "penalty") if m else 0.0
            key = (kind, tuple(body[:ntypes]))
            parameters[key] = max(penalty, parameters.get(key, penalty))

    if not atoms:
        raise ParseError(f"{path}: no ATOM line found; is this a CGenFF .str file?")
    return StrFile(
        atoms=atoms,
        parameters=[StrParameter(kind, types, pen) for (kind, types), pen in parameters.items()],
        program_version=program_version,
        parameter_version=parameter_version,
        param_penalty=param_penalty,
        charge_penalty=charge_penalty,
        source=path,
    )
