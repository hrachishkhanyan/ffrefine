"""GROMACS single-point energy engine (spec section 4.4).

Energies are obtained with ``gmx mdrun -rerun`` over a multi-frame GROMOS-96
(.g96) trajectory and extracted with ``gmx energy``.  Terms are always
selected by name: the menu numbering differs between GROMACS versions (spec
section 9.9).  G96, not PDB, because PDB's fixed-width Angstrom fields only
keep 3 decimal places (~1e-4 nm); the section 4.5 self-consistency check
needs finer coordinate precision than that to be meaningful (Task 13 fix
round 1, ruling R22) -- see :func:`_write_g96_trajectory`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Binaries
from ..errors import ExternalToolError, ValidationError
from ..io.structure import ANGSTROM_TO_NM, Structure
from ..runner import run_command

log = logging.getLogger("ffrefine")

SP_MDP = """\
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
"""

FALLBACK_CUTOFF_NM = 3.0
"""Cut-off radius in nm on the fallback path -- rlist, rcoulomb and rvdw alike.

Named rather than repeated so :func:`_check_molecule_fits_cutoff` measures the
molecule against the number the mdp actually uses, and the two cannot drift.
"""

# nstlist=10 here (vs. 0 in SP_MDP): the Verlet cutoff scheme on GROMACS
# 2025.4 rejects nstlist=0 outright, independent of pbc/cutoffs -- confirmed
# by Task 6's supplementary grompp check, where nstlist=10 succeeded. nsteps=0
# makes the neighbour-list rebuild interval otherwise irrelevant here.
FALLBACK_MDP = f"""\
integrator              = md
nsteps                  = 0
pbc                     = xyz
cutoff-scheme           = Verlet
verlet-buffer-tolerance = -1
nstlist                 = 10
rlist                   = {FALLBACK_CUTOFF_NM}
coulombtype             = Cut-off
rcoulomb                = {FALLBACK_CUTOFF_NM}
vdwtype                 = Cut-off
rvdw                    = {FALLBACK_CUTOFF_NM}
epsilon_r               = 1
constraints             = none
continuation            = yes
nstcalcenergy           = 1
nstenergy               = 1
"""

DEFAULT_TERMS = (
    "Potential",
    "Bond",
    "Angle",
    "Proper Dih.",
    "Improper Dih.",
    "LJ-14",
    "Coulomb-14",
    "LJ (SR)",
    "Coulomb (SR)",
)

FALLBACK_BOX_NM = 10.0
"""Box edge used when the installed GROMACS rejects zero cutoffs."""


@dataclass
class EnergyResult:
    """Decomposed single-point energies, one row per input frame."""

    terms: dict[str, np.ndarray]
    nframes: int
    used_fallback: bool

    def potential(self) -> np.ndarray:
        """Total potential energy in kJ/mol."""
        if "Potential" not in self.terms:
            raise ExternalToolError(
                f"gmx energy did not return a Potential term; got {sorted(self.terms)}"
            )
        return self.terms["Potential"]


def write_mdp(path: str | Path, fallback: bool = False) -> None:
    """Write the single-point mdp, optionally the large-box fallback variant."""
    Path(path).write_text(FALLBACK_MDP if fallback else SP_MDP, newline="\n")


def parse_energy_xvg(path: str | Path) -> tuple[list[str], np.ndarray]:
    """Parse a ``gmx energy`` .xvg into legend names and a data array.

    Args:
        path: the .xvg file.

    Returns:
        ``(names, data)`` where ``names`` has one entry per data column after
        the time column and ``data`` has shape (nframes, nterms).
    """
    names: list[str] = []
    rows: list[list[float]] = []
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if (
            stripped.startswith("@")
            and " legend " in stripped
            and stripped.split()[1].startswith("s")
        ):
            names.append(stripped.split('"')[1])
        elif stripped.startswith(("#", "@")) or not stripped:
            continue
        else:
            rows.append([float(v) for v in stripped.split()])
    if not rows:
        raise ExternalToolError(f"{path}: no energy data rows found")
    data = np.array(rows, dtype=float)[:, 1:]  # drop the time column
    if data.shape[1] != len(names):
        raise ExternalToolError(f"{path}: {data.shape[1]} data columns but {len(names)} legends")
    return names, data


def _box_line(fallback: bool) -> np.ndarray | None:
    return np.array([FALLBACK_BOX_NM] * 3) if fallback else None


def max_interatomic_distance(frames: np.ndarray) -> float:
    """Largest distance between any two atoms, over every frame, in nm.

    Args:
        frames: shape (nframes, natoms, 3) in nm.

    Returns:
        The maximum pairwise distance, or 0.0 for fewer than two atoms.
    """
    arr = np.asarray(frames, dtype=float)
    if arr.ndim != 3 or arr.shape[1] < 2 or arr.shape[0] == 0:
        return 0.0
    worst = 0.0
    for frame in arr:
        # The bounding-box diagonal is an exact upper bound on any pair
        # distance in this frame, so a frame that cannot beat the running
        # maximum is skipped without the O(natoms^2) pass.
        if float(np.linalg.norm(frame.max(axis=0) - frame.min(axis=0))) <= worst:
            continue
        delta = frame[:, None, :] - frame[None, :, :]
        worst = max(worst, float(np.sqrt((delta * delta).sum(axis=-1)).max()))
    return worst


def _check_molecule_fits_cutoff(frames: np.ndarray, cutoff_nm: float = FALLBACK_CUTOFF_NM) -> None:
    """Refuse a molecule larger than the cutoff the fallback mdp applies.

    The spec's literal zero-cutoff vacuum mdp cannot succeed on GROMACS 2020
    or newer (it implies ``cutoff-scheme = group``, which was removed), so the
    "fallback" is in practice the only path that ever runs.  A molecule whose
    atom pairs exceed ``rvdw`` in some conformers silently loses those
    interactions.

    Nothing else can catch this.  The section 4.5 self-check compares
    ``E_full - E_zeroed`` against the analytic torsion sum, and the truncation
    is *identical* in both topologies -- so it cancels exactly in that
    difference while still distorting ``E_MM0``, and therefore the fit target
    and every parameter derived from it.

    Args:
        frames: shape (nframes, natoms, 3) in nm.
        cutoff_nm: the mdp's rvdw/rcoulomb.

    Raises:
        ValidationError: some frame has a pair of atoms further apart than
            ``cutoff_nm``.
    """
    span = max_interatomic_distance(frames)
    if span <= cutoff_nm:
        log.debug(
            "largest interatomic distance %.3f nm, within the %.3g nm cutoff", span, cutoff_nm
        )
        return
    raise ValidationError(
        f"the molecule does not fit inside the MM cutoff: its largest interatomic distance "
        f"over these {np.asarray(frames).shape[0]} frame(s) is {span:.3f} nm, but this "
        f"GROMACS build rejected the spec's zero-cutoff vacuum mdp and the fallback path "
        f"applies a {cutoff_nm:g} nm cutoff to both rvdw and rcoulomb (spec section 4.4).\n"
        "  Every atom pair beyond that contributes nothing, so the nonbonded energy is "
        "truncated. The spec 4.5 self-consistency check cannot detect it: the truncation is "
        "identical in the full and zeroed topologies and cancels exactly in their "
        "difference, while still distorting E_MM0 and the fit target built from it.\n"
        "  ffrefine refines small molecules; it has no correct path for one this large."
    )


def single_point_energies(
    binaries: Binaries,
    topology_path: str | Path,
    structure: Structure,
    frames: np.ndarray,
    workdir: str | Path,
    log_stem: str,
    nthreads: int = 1,
    terms: tuple[str, ...] = DEFAULT_TERMS,
) -> EnergyResult:
    """Compute single-point energies for every frame with ``mdrun -rerun``.

    Tries the vacuum zero-cutoff mdp first and falls back to a large box with
    3 nm cutoffs if the installed GROMACS rejects it (spec section 4.4).

    The rerun trajectory is a GROMOS-96 (.g96) file (see
    :func:`_write_g96_trajectory`), not PDB: PDB's fixed ``%8.3f`` Angstrom
    fields quantize coordinates to ~1e-4 nm, which is coarse enough to matter
    to the section 4.5 self-consistency check (ruling R22, Task 13 fix
    round 1). G96 stores nanometre coordinates at ``%15.9f``, eliminating
    that error.

    Args:
        binaries: resolved binary paths; ``gmx`` is required.
        topology_path: a .top that grompp can process.
        structure: supplies atom and residue names for the rerun trajectory.
        frames: shape (nframes, natoms, 3) in nm.
        workdir: stage directory for inputs and logs.
        log_stem: prefix for log files.
        nthreads: passed to ``mdrun -nt``.
        terms: energy terms to request, matched by name.

    Returns:
        The decomposed energies.

    Raises:
        ExternalToolError: on any GROMACS failure or a frame-count mismatch.
        ValidationError: the molecule is larger than the fallback path's
            cutoff, so its nonbonded energy would be silently truncated (see
            :func:`_check_molecule_fits_cutoff`).
    """
    gmx = binaries.require("gmx")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frames, dtype=float)
    nframes = frames.shape[0]

    # Box is embedded per-frame in the g96 file itself (see
    # _write_g96_trajectory), unlike the PDB approach this replaced, which
    # needed a CRYST1 record injected after the fact once the fallback path
    # was known (see the now-unused _inject_pdb_box, kept for reference).
    traj = workdir / f"{log_stem}_traj.g96"
    _write_g96_trajectory(traj, structure, frames, FALLBACK_BOX_NM)

    used_fallback = False
    succeeded_stem = None
    for attempt, fallback in enumerate((False, True)):
        # Each attempt gets its own mdp/conf/log filenames: if the vacuum
        # attempt fails and the fallback succeeds, the vacuum attempt's own
        # grompp logs must stay on disk (spec section 8 -- every external
        # command's output is captured), not be overwritten by the retry.
        attempt_stem = f"{log_stem}_{'fallback' if fallback else 'vacuum'}"
        mdp = workdir / f"{attempt_stem}.mdp"
        write_mdp(mdp, fallback=fallback)

        conf = workdir / f"{attempt_stem}_conf.gro"
        _write_conf_gro(conf, structure, frames[0], box=_box_line(fallback))

        grompp_stem = f"{attempt_stem}_grompp"
        try:
            run_command(
                [
                    gmx,
                    "grompp",
                    "-f",
                    mdp.name,
                    "-c",
                    conf.name,
                    "-p",
                    str(Path(topology_path).resolve()),
                    "-o",
                    f"{attempt_stem}.tpr",
                    "-maxwarn",
                    "5",
                ],
                workdir,
                grompp_stem,
            )
        except ExternalToolError:
            # Read the full grompp logs rather than the exception's truncated
            # tail: the Verlet-scheme rejection of a zero-cutoff vacuum mdp
            # emits several ERROR blocks (cutoff radius, pbc, nstlist), and
            # the one mentioning cutoffs is often pushed out of the tail.
            # Guarded with exists(): a bad gmx path raises ExternalToolError
            # from run_command's own OSError branch before any log file is
            # written, and that must not become an unrelated crash here.
            text = _grompp_log_text(workdir, grompp_stem).lower()
            cutoff_problem = any(
                t in text for t in ("rlist", "cut-off", "cutoff", "rvdw", "rcoulomb")
            )
            if attempt == 0 and cutoff_problem:
                log.warning(
                    "grompp rejected the zero-cutoff vacuum mdp; retrying with the "
                    "%.0f nm box and 3 nm cutoff fallback (spec section 4.4)",
                    FALLBACK_BOX_NM,
                )
                continue
            raise
        # -maxwarn lets grompp proceed past warnings instead of aborting;
        # echo them to our own log so a suppressed warning (e.g. the net
        # charge or atom-name-mismatch notice) is never silently invisible.
        _log_grompp_warnings(_grompp_log_text(workdir, grompp_stem))
        used_fallback = fallback
        succeeded_stem = attempt_stem
        break

    log.info(
        "GROMACS single point using %s path", "large-box fallback" if used_fallback else "vacuum"
    )
    if used_fallback:
        # Only the fallback truncates anything: the vacuum mdp's rvdw = 0 means
        # no cutoff at all. Checked before mdrun so no energies are produced
        # that a caller could mistake for correct ones.
        _check_molecule_fits_cutoff(frames)

    run_command(
        [
            gmx,
            "mdrun",
            "-s",
            f"{succeeded_stem}.tpr",
            "-rerun",
            traj.name,
            "-e",
            f"{log_stem}.edr",
            "-g",
            f"{log_stem}.log",
            "-o",
            f"{log_stem}.trr",
            "-nt",
            str(nthreads),
        ],
        workdir,
        f"{log_stem}_mdrun",
    )

    # gmx energy's interactive matcher requires the hyphenated form of a
    # multi-word term ("Proper-Dih.", not "Proper Dih.") even though the
    # parsed .xvg legend keeps the original name with a space -- confirmed
    # empirically against GROMACS 2025.4.  Blank line ends the selection.
    selection = "\n".join(t.replace(" ", "-") for t in terms) + "\n\n"
    run_command(
        [gmx, "energy", "-f", f"{log_stem}.edr", "-o", f"{log_stem}.xvg"],
        workdir,
        f"{log_stem}_energy",
        stdin=selection,
    )

    names, data = parse_energy_xvg(workdir / f"{log_stem}.xvg")
    if data.shape[0] != nframes:
        raise ExternalToolError(
            f"expected exactly one energy frame per input frame: got "
            f"{data.shape[0]} energy frames for {nframes} input frames"
        )
    missing = [t for t in terms if t not in names]
    if missing:
        log.warning("gmx energy did not provide these terms: %s", ", ".join(missing))

    return EnergyResult(
        terms={name: data[:, i] for i, name in enumerate(names)},
        nframes=nframes,
        used_fallback=used_fallback,
    )


def _grompp_log_text(workdir: Path, stem: str) -> str:
    """Read the stdout+stderr ``run_command`` already captured for ``stem``.

    Guarded with ``exists()``: when ``gmx`` itself cannot be executed (e.g. a
    bad ``--gmx-bin`` override), ``run_command`` raises before writing any log
    file at all, and that must not turn into an unrelated ``FileNotFoundError``
    here.

    Args:
        workdir: stage directory passed to ``run_command``.
        stem: the ``log_stem`` used for that ``run_command`` call.

    Returns:
        The concatenated stdout and stderr text, or ``""`` if neither exists.
    """
    stdout_path = workdir / f"{stem}.stdout"
    stderr_path = workdir / f"{stem}.stderr"
    stdout = stdout_path.read_text() if stdout_path.exists() else ""
    stderr = stderr_path.read_text() if stderr_path.exists() else ""
    return stdout + stderr


def _log_grompp_warnings(text: str) -> None:
    """Echo grompp's own WARNING blocks to the ffrefine log.

    ``-maxwarn`` lets grompp proceed past warnings (e.g. a net charge or an
    atom-name mismatch) instead of aborting; a warning grompp printed and we
    chose to suppress must still surface somewhere other than a raw log file
    on disk.  Log only -- never raise, this runs after a successful grompp.

    Args:
        text: grompp's combined stdout+stderr, as returned by
            :func:`_grompp_log_text`.
    """
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("WARNING"):
            block = [lines[i]]
            i += 1
            while i < len(lines) and lines[i].strip():
                block.append(lines[i])
                i += 1
            log.warning("grompp: %s", " ".join(s.strip() for s in block))
        else:
            i += 1


def _inject_pdb_box(path: Path, box_nm: float) -> None:
    """Insert a single cubic ``CRYST1`` record ahead of the first MODEL.

    ``gmx mdrun -rerun`` reads the periodic box once from the trajectory file
    and applies it to every frame; without a CRYST1 record the fallback mdp's
    ``pbc = xyz`` makes mdrun abort with "does not contain a box, while pbc
    is used" (confirmed against GROMACS 2025.4).

    Unused since Task 13 fix round 1 (ruling R22): ``single_point_energies``
    now writes a .g96 trajectory (:func:`_write_g96_trajectory`), which
    carries a ``BOX`` block in every frame directly, making this after-the-fact
    injection unnecessary. Kept, rather than deleted, per that ruling.

    Args:
        path: multi-model PDB written by
            :func:`ffrefine.io.structure.write_multimodel_pdb`.
        box_nm: cubic box edge length in nanometres.
    """
    edge_angstrom = box_nm / ANGSTROM_TO_NM
    cryst = (
        f"CRYST1{edge_angstrom:9.3f}{edge_angstrom:9.3f}{edge_angstrom:9.3f}"
        f"{90.0:7.2f}{90.0:7.2f}{90.0:7.2f} P 1           1"
    )
    lines = path.read_text().splitlines()
    lines.insert(1, cryst)  # after TITLE, before the first MODEL
    path.write_text("\n".join(lines) + "\n", newline="\n")


def _write_g96_trajectory(
    path: str | Path, structure: Structure, frames: np.ndarray, box_nm: float
) -> None:
    """Write a multi-frame GROMOS-96 (.g96) trajectory for ``mdrun -rerun``.

    G96 stores coordinates directly in nanometres at ``%15.9f`` -- about
    1e-9 nm per atom, versus the ~1e-4 nm a PDB ATOM record's fixed
    ``%8.3f`` Angstrom fields keep. That difference matters: the section 4.5
    self-consistency check compares GROMACS's output against an independent
    analytic reference to 1e-3 kJ/mol, and PDB's coordinate quantization
    alone was large enough to fail that gate regardless of whether the
    topology edit was correct (Task 13 fix round 1, ruling R22).

    Confirmed empirically against GROMACS 2025.4: a bare concatenation of
    ``TITLE``/``POSITION``/``BOX`` blocks per frame (no ``TIMESTEP`` block,
    and only one leading ``TITLE``) is accepted by ``mdrun -rerun``, which
    reads every frame in order. The per-atom line layout
    (``%5d %-5s %-5s%7d%15.9f%15.9f%15.9f``) matches byte-for-byte what
    ``gmx editconf`` itself produces when converting a .gro to .g96.

    This is private to the energy engine and independent of
    :func:`ffrefine.io.structure.write_multimodel_pdb`, which remains the
    package's public, user-facing multi-model structure writer.

    Args:
        path: output file.
        structure: supplies atom names, residue names and residue ids.
        frames: shape (nframes, natoms, 3) in nm.
        box_nm: cubic box edge length written into every frame's BOX block.

    Raises:
        ValueError: if ``frames`` is not shaped (nframes, natoms, 3).
    """
    frames = np.asarray(frames, dtype=float)
    if frames.ndim != 3 or frames.shape[1] != structure.natoms:
        raise ValueError(f"frames shape {frames.shape} incompatible with {structure.natoms} atoms")
    lines = ["TITLE", structure.title or "ffrefine scan", "END"]
    for frame in frames:
        lines.append("POSITION")
        for i in range(structure.natoms):
            x, y, z = frame[i]
            lines.append(
                f"{structure.resids[i]:5d} {structure.resnames[i]:<5s} "
                f"{structure.names[i]:<5s}{i + 1:7d}{x:15.9f}{y:15.9f}{z:15.9f}"
            )
        lines.append("END")
        lines.append("BOX")
        lines.append(f"{box_nm:15.9f}{box_nm:15.9f}{box_nm:15.9f}")
        lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n", newline="\n")


def _write_conf_gro(
    path: Path, structure: Structure, coords: np.ndarray, box: np.ndarray | None
) -> None:
    """Write a single-frame GRO for grompp."""
    edge = box if box is not None else np.array([3.0, 3.0, 3.0])
    lines = ["ffrefine single point", f"{structure.natoms:5d}"]
    for i in range(structure.natoms):
        x, y, z = coords[i]
        lines.append(
            f"{structure.resids[i]:5d}{structure.resnames[i]:<5s}"
            f"{structure.names[i]:>5s}{i + 1:5d}{x:8.3f}{y:8.3f}{z:8.3f}"
        )
    lines.append(f"{edge[0]:10.5f}{edge[1]:10.5f}{edge[2]:10.5f}")
    path.write_text("\n".join(lines) + "\n", newline="\n")
