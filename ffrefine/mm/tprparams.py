"""The torsion parameters grompp actually built (milestone 6a design, section 6).

A CGenFF or GAFF molecule topology writes its torsions as bare ``i j k l 9``
lines; the force constants come from ``[ dihedraltypes ]`` in the force-field
files it includes. This build does not reimplement grompp's matching rules
(most non-wildcard matches, first in file order, both orientations,
consecutive-line blocks for funct 9, later definitions overriding earlier
ones, bonded types, two-type entries). It asks grompp: ``grompp -pp`` writes
the fully preprocessed topology, ``gmx dump`` prints the parameters stored in
the ``.tpr``, and :func:`cross_check` pairs each line of the torsion set T
with the terms grompp built for it.

Two facts measured on GROMACS 2025.4 shape the cross-check:

* A typed funct-9 line expands into one term per multiplicity, and the dump
  keeps the atom order the line was written in.
* grompp builds **no** term for a ``k = 0`` term, typed or explicit. A ``k = 0``
  term is therefore expected to be absent, and a typed line whose terms all
  have ``k = 0`` legitimately resolves to nothing.

Indices are 1-based throughout, like the topology file; the dump's 0-based
atom indices are converted as they are read.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from ..config import Binaries
from ..errors import ExternalToolError, ValidationError
from ..io.structure import Structure
from ..io.topology import Defaults, Dihedral, Topology, parse_defaults_fields
from ..io.topwriter import make_system_top, write_molecule_copy
from .gromacs import dump_tpr, preprocess_topology
from .torsionset import collect_torsion_set

log = logging.getLogger("ffrefine")

Quad = tuple[int, int, int, int]

SOURCE_TOPOLOGY = "topology"
"""The term's parameters are written on its own ``[ dihedrals ]`` line."""

SOURCE_DIHEDRALTYPES = "dihedraltypes"
"""The term came from ``[ dihedraltypes ]``, read back from grompp's ``.tpr``."""

PHI_TOLERANCE_DEG = 1e-4
K_RELATIVE_TOLERANCE = 1e-6
"""Explicit lines against the dump: grompp stores float32, printed to 9 digits."""

_FUNCTYPE_RE = re.compile(r"^\s*functype\[(\d+)\]=(\w+),\s*(.*?)\s*$")
_PDIHS_RE = re.compile(
    r"^phiA=\s*(\S+),\s*cpA=\s*(\S+),\s*phiB=\s*(\S+),\s*cpB=\s*(\S+),\s*mult=\s*(-?\d+)$"
)
_MOLTYPE_RE = re.compile(r"^\s*moltype \((\d+)\):\s*$")
_NAME_RE = re.compile(r'^\s*name="(.*)"\s*$')
_ATOM_COUNT_RE = re.compile(r"^\s*atom \((\d+)\):\s*$")
_PROPER_HEADING = "Proper Dih.:"
_NR_RE = re.compile(r"^\s*nr:\s*(\d+)\s*$")
_IATOM_RE = re.compile(r"^\s*(\d+)\s+type=(\d+)\s+\((\w+)\)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$")


@dataclass(frozen=True)
class DumpedTerm:
    """One ``Proper Dih.`` entry of the dump.  ``quad`` is 1-based, as written."""

    quad: Quad
    phi0_deg: float
    k_kj_per_mol: float
    multiplicity: int


@dataclass(frozen=True)
class ResolvedTerm:
    """One term of T with explicit parameters, and where they came from.

    ``dihedral.params`` is always ``[phi0_deg, k_kj_per_mol, multiplicity]``.
    ``source`` is :data:`SOURCE_TOPOLOGY` or :data:`SOURCE_DIHEDRALTYPES`.
    """

    dihedral: Dihedral
    source: str


@dataclass
class ResolvedTorsionSet:
    """T with every term's parameters, as grompp built them.

    Attributes:
        terms: one entry per term, in topology order.
        defaults: ``[ defaults ]`` from grompp's preprocessed topology, or
            ``None`` if it has none.
        pp_sha256: SHA-256 of that preprocessed topology: a digest of
            everything grompp read, used as part of the ``--resume`` key.
        full_top: the wrapper ``.top`` grompp was given for the unmodified
            topology; the self-check's "full" energies use the same file.
        dump_path: where the ``gmx dump`` text is on disk.
        gromacs_version: from the dump's banner, for messages.
    """

    terms: list[ResolvedTerm]
    defaults: Defaults | None
    pp_sha256: str
    full_top: Path
    dump_path: Path
    gromacs_version: str

    @property
    def dihedrals(self) -> list[Dihedral]:
        """The terms as dihedral entries, for :func:`ffrefine.mm.analytic.torsion_set_energy`."""
        return [t.dihedral for t in self.terms]


def _dump_error(source: Path, what: str) -> ValidationError:
    return ValidationError(
        f"could not read the torsion parameters from {source}: {what}. This is a "
        "limit of ffrefine's reader of 'gmx dump' output, not a problem with your "
        "topology."
    )


def _pdihs_functypes(lines: list[str], source: Path) -> dict[int, tuple[float, float, int]]:
    """``functype[N]=PDIHS`` entries: ``N -> (phi0_deg, k, multiplicity)``."""
    out: dict[int, tuple[float, float, int]] = {}
    for line in lines:
        m = _FUNCTYPE_RE.match(line)
        if not m or m.group(2) != "PDIHS":
            continue
        p = _PDIHS_RE.match(m.group(3))
        if not p:
            raise _dump_error(source, f"unrecognised PDIHS functype line {line.strip()!r}")
        phi_a, k_a, phi_b, k_b = (float(p.group(i)) for i in range(1, 5))
        if phi_a != phi_b or k_a != k_b:
            raise ValidationError(
                f"{source}: functype[{m.group(1)}] has different A and B states "
                f"(phi {phi_a} / {phi_b}, k {k_a} / {k_b}). Free-energy (B-state) "
                "topologies are not supported."
            )
        out[int(m.group(1))] = (phi_a, k_a, int(p.group(5)))
    return out


def parse_dump_torsions(text: str, molname: str, natoms: int, source: Path) -> list[DumpedTerm]:
    """Every proper-dihedral term of one moltype in a ``gmx dump -s`` output.

    Args:
        text: the dump.
        molname: the topology's ``[ moleculetype ]`` name.
        natoms: the topology's atom count; the moltype must agree.
        source: the dump's path, for messages.

    Returns:
        One entry per ``Proper Dih.`` iatoms line, in order, 1-based.

    Raises:
        ValidationError: the layout is not what this reader knows, the
            moltype is missing or ambiguous or has the wrong atom count, or a
            term has different A and B states.
    """
    lines = text.splitlines()
    functypes = _pdihs_functypes(lines, source)

    starts = [i for i, line in enumerate(lines) if _MOLTYPE_RE.match(line)]
    blocks: list[tuple[str | None, list[str]]] = []
    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        name = next(
            (m.group(1) for m in map(_NAME_RE.match, lines[start + 1 : start + 3]) if m), None
        )
        blocks.append((name, lines[start + 1 : end]))
    matching = [block for name, block in blocks if name == molname]
    if len(matching) != 1:
        names = [name for name, _ in blocks]
        raise _dump_error(
            source, f"expected exactly one moltype named {molname!r}, found moltypes {names}"
        )
    block = matching[0]

    count = next((int(m.group(1)) for m in map(_ATOM_COUNT_RE.match, block) if m), None)
    if count != natoms:
        raise _dump_error(source, f"moltype {molname!r} has {count} atoms, the topology {natoms}")

    try:
        head = next(i for i, line in enumerate(block) if line.strip() == _PROPER_HEADING)
    except StopIteration:
        raise _dump_error(source, f"moltype {molname!r} has no 'Proper Dih.' block") from None
    nr = _NR_RE.match(block[head + 1]) if head + 1 < len(block) else None
    if not nr:
        raise _dump_error(source, "the 'Proper Dih.' block has no 'nr:' line")
    expected = int(nr.group(1))
    terms: list[DumpedTerm] = []
    if expected:
        if head + 2 >= len(block) or block[head + 2].strip() != "iatoms:":
            raise _dump_error(source, "the 'Proper Dih.' block has no 'iatoms:' line")
        for line in block[head + 3 :]:
            m = _IATOM_RE.match(line)
            if not m:
                break
            if m.group(3) != "PDIHS":
                raise _dump_error(source, f"'Proper Dih.' entry of type {m.group(3)}: {line!r}")
            index = int(m.group(2))
            if index not in functypes:
                raise _dump_error(source, f"an entry names functype[{index}], which is not PDIHS")
            phi0, k, mult = functypes[index]
            quad = (
                int(m.group(4)) + 1,
                int(m.group(5)) + 1,
                int(m.group(6)) + 1,
                int(m.group(7)) + 1,
            )
            terms.append(DumpedTerm(quad, phi0, k, mult))
    # nr counts integers: five per entry (the functype index and four atoms).
    if 5 * len(terms) != expected:
        raise _dump_error(
            source, f"'Proper Dih.' says nr: {expected} but {len(terms)} entries were read"
        )
    return terms


def parse_defaults(text: str, source: Path) -> Defaults | None:
    """``[ defaults ]`` from grompp's preprocessed topology, or ``None``.

    Raises:
        ValidationError: the directive's line is malformed.
    """
    section = ""
    for lineno, raw in enumerate(text.splitlines(), start=1):
        body = raw.split(";", 1)[0].strip()
        if not body or body.startswith("#"):
            continue
        if body.startswith("[") and body.endswith("]"):
            section = body[1:-1].strip().lower()
            continue
        if section == "defaults":
            try:
                return parse_defaults_fields(body.split())
            except (IndexError, ValueError) as exc:
                raise ValidationError(
                    f"{source}:{lineno}: malformed [ defaults ] line {raw!r}"
                ) from exc
    return None


def _central(quad: Quad) -> tuple[int, int]:
    return (min(quad[1], quad[2]), max(quad[1], quad[2]))


def _fmt(terms: list[tuple[float, float, int]]) -> str:
    return ", ".join(f"(phi0 {p:g}, k {k:g}, n {n})" for p, k, n in terms) or "none"


def _same_terms(expected: list[tuple[float, float, int]], got: list[DumpedTerm]) -> bool:
    """Multiset equality within the float32 tolerances."""
    if len(expected) != len(got):
        return False
    exp = sorted(expected, key=lambda t: (t[2], t[0], t[1]))
    obs = sorted(
        ((g.phi0_deg, g.k_kj_per_mol, g.multiplicity) for g in got),
        key=lambda t: (t[2], t[0], t[1]),
    )
    for (p1, k1, n1), (p2, k2, n2) in zip(exp, obs):
        if n1 != n2 or abs(p1 - p2) > PHI_TOLERANCE_DEG:
            return False
        if abs(k1 - k2) > K_RELATIVE_TOLERANCE * max(1.0, abs(k1)):
            return False
    return True


_IFDEF_HINT = (
    "Either ffrefine misread the dump, or grompp did not read the [ dihedrals ] lines as "
    "ffrefine did -- for example an #ifdef in the molecule file driven by a define that an "
    "included force-field file sets, which ffrefine cannot see."
)


def cross_check(
    tset: list[Dihedral], dumped: list[DumpedTerm], dump_path: Path
) -> list[ResolvedTerm]:
    """Pair every member of T with the terms grompp built for it (design section 6.3).

    Args:
        tset: :func:`ffrefine.mm.torsionset.collect_torsion_set`, one entry
            per term-line.
        dumped: :func:`parse_dump_torsions` for the whole molecule.
        dump_path: for messages.

    Returns:
        T with explicit parameters on every term, in topology order. An
        explicit line keeps its own values (the analytic reference for an
        explicit-only set is unchanged); a typed line takes grompp's.

    Raises:
        ValidationError: a quad written both typed and explicit, or typed
            twice; an explicit line whose non-zero terms differ from grompp's;
            a term grompp built about the central bond that no line accounts
            for.
    """
    if not tset:
        return []
    bond = tset[0].central_bond
    groups: dict[Quad, list[Dihedral]] = {}
    for d in tset:
        groups.setdefault((d.ai, d.aj, d.ak, d.al), []).append(d)
    by_quad: dict[Quad, list[DumpedTerm]] = {}
    for t in dumped:
        if _central(t.quad) == bond:
            by_quad.setdefault(t.quad, []).append(t)

    resolved: list[ResolvedTerm] = []
    for quad, lines in groups.items():
        where = " ".join(map(str, quad))
        typed = [d for d in lines if not d.params]
        explicit = [d for d in lines if d.params]
        got = by_quad.pop(quad, [])
        if (typed and explicit) or len(typed) > 1:
            raise ValidationError(
                f"the torsion {where} is written {len(lines)} times in [ dihedrals ] "
                f"({len(typed)} without parameters, {len(explicit)} with), and is refused as "
                "ambiguous: grompp builds terms for every line, and there is no unambiguous "
                "way to say which of them belong to which line. Merge them into one line, or "
                "give every line explicit parameters."
            )
        if typed:
            for t in got:
                resolved.append(
                    ResolvedTerm(
                        Dihedral(
                            *quad,
                            funct=typed[0].funct,
                            params=[t.phi0_deg, t.k_kj_per_mol, float(t.multiplicity)],
                            comment="from [ dihedraltypes ], read back with gmx dump",
                        ),
                        SOURCE_DIHEDRALTYPES,
                    )
                )
            if not got:
                log.info(
                    "torsion %s: every [ dihedraltypes ] term grompp matched has k = 0, so "
                    "it contributes nothing",
                    where,
                )
            continue
        for d in explicit:
            if len(d.params) < 3:
                raise ValidationError(
                    f"the torsion {where} has {len(d.params)} parameter(s) on its line; a "
                    "funct 1 or 9 dihedral needs phi0, k and a multiplicity"
                )
        expected = [
            (float(d.params[0]), float(d.params[1]), int(round(float(d.params[2]))))
            for d in explicit
            if float(d.params[1]) != 0.0
        ]
        if not _same_terms(expected, got):
            observed = [(g.phi0_deg, g.k_kj_per_mol, g.multiplicity) for g in got]
            raise ValidationError(
                f"the torsion {where} carries explicit parameters, but the terms grompp "
                "built for it disagree with them.\n"
                f"  in the topology (k != 0): {_fmt(expected)}\n"
                f"  in {dump_path}: {_fmt(observed)}\n  {_IFDEF_HINT}"
            )
        resolved += [ResolvedTerm(d, SOURCE_TOPOLOGY) for d in explicit]

    if by_quad:
        extra = [
            f"{' '.join(map(str, q))} (phi0 {t.phi0_deg:g}, k {t.k_kj_per_mol:g}, n {t.multiplicity})"
            for q, ts in by_quad.items()
            for t in ts
        ]
        raise ValidationError(
            f"grompp built {len(extra)} term(s) about the central bond {bond} that no "
            f"[ dihedrals ] line ffrefine read accounts for: {'; '.join(extra)} "
            f"(from {dump_path}).\n  {_IFDEF_HINT}"
        )
    return resolved


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _preprocess(
    binaries: Binaries,
    topology: Topology,
    top: Path,
    structure: Structure,
    workdir: str | Path,
    stem: str,
) -> tuple[Path, Path]:
    """:func:`preprocess_topology`, naming the include base when an include is missing.

    grompp's "Topology include file ... not found" says what it could not
    open, not where ffrefine pointed it; design section 7 asks for both.
    """
    try:
        return preprocess_topology(binaries, top, structure, workdir, stem)
    except ExternalToolError as exc:
        if "include file" not in str(exc):
            raise
        how = (
            f"named by the ffrefine include-base marker in {topology.source}"
            if topology.include_marker is not None
            else f"the directory of {topology.source}"
        )
        raise ExternalToolError(
            f"{exc}\n  ffrefine resolved the relative #include lines of "
            f"{Path(topology.source).name} against {topology.include_base} ({how}). An "
            "include with no file there is left for grompp's own search, which is what "
            "failed above."
        ) from exc


def write_full_system(
    topology: Topology, workdir: str | Path, stem: str, ff_includes: list[str] | None = None
) -> Path:
    """Write ``<stem>.itp`` (the unmodified topology, includes absolute) and ``<stem>.top``.

    Returns:
        The wrapper ``.top`` to hand to grompp.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    itp = workdir / f"{stem}.itp"
    write_molecule_copy(topology.source, itp)
    top = workdir / f"{stem}.top"
    make_system_top(itp, top, topology.name, ff_includes)
    return top


def resolve_torsion_set(
    binaries: Binaries,
    topology: Topology,
    quad: Quad,
    structure: Structure,
    workdir: str | Path,
    ff_includes: list[str] | None = None,
) -> ResolvedTorsionSet:
    """Give every term of T explicit parameters, as grompp built them.

    Writes ``full.itp``/``full.top`` (the unmodified topology), runs
    ``grompp -pp`` and ``gmx dump`` on it, and cross-checks the result
    against T (design section 6.3). About a second on PFOA with CHARMM36.

    Args:
        binaries: ``gmx`` required.
        topology: the unmodified topology.
        quad: 1-based target quadruplet; only its central bond matters.
        structure: for the conf file.
        workdir: stage directory.
        ff_includes: ``--ff-include`` values for the wrapper.

    Raises:
        ExternalToolError: grompp or gmx dump failed.
        ValidationError: the dump could not be read, or the cross-check
            refused (see :func:`cross_check`).
    """
    full_top = write_full_system(topology, workdir, "full", ff_includes)
    tpr, flat = _preprocess(binaries, topology, full_top, structure, workdir, "full")
    dump = dump_tpr(binaries, tpr, workdir, "full_dump")
    try:
        dumped = parse_dump_torsions(dump.text, topology.name, topology.natoms, dump.path)
    except ValidationError as exc:
        raise ValidationError(f"{exc} (GROMACS {dump.gromacs_version})") from exc
    terms = cross_check(collect_torsion_set(topology, quad), dumped, dump.path)
    n_typed = sum(t.source == SOURCE_DIHEDRALTYPES for t in terms)
    log.info(
        "torsion set about bond (%d, %d): %d term(s), %d of them from [ dihedraltypes ] "
        "as grompp built them (%s)",
        quad[1],
        quad[2],
        len(terms),
        n_typed,
        dump.path.name,
    )
    return ResolvedTorsionSet(
        terms=terms,
        defaults=parse_defaults(flat.read_text(), flat),
        pp_sha256=_sha256(flat),
        full_top=full_top,
        dump_path=dump.path,
        gromacs_version=dump.gromacs_version,
    )


def preprocessed_sha256(
    binaries: Binaries,
    topology: Topology,
    structure: Structure,
    workdir: str | Path,
    stem: str,
    ff_includes: list[str] | None = None,
) -> str:
    """SHA-256 of grompp's preprocessed view of ``topology``.

    Covers every file grompp opens -- force-field includes too -- which the
    ``--resume`` key of a stage that runs grompp must (design section 4).
    Writes ``<stem>.itp``, ``<stem>.top`` and ``<stem>_pp.top`` into ``workdir``.
    """
    top = write_full_system(topology, workdir, stem, ff_includes)
    _, flat = _preprocess(binaries, topology, top, structure, workdir, stem)
    return _sha256(flat)
