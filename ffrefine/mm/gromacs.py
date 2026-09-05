"""GROMACS single-point energy engine (spec section 4.4).

Energies are obtained with ``gmx mdrun -rerun`` over a multi-model PDB and
extracted with ``gmx energy``.  Terms are always selected by name: the menu
numbering differs between GROMACS versions (spec section 9.9).
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Binaries
from ..errors import ExternalToolError
from ..io.structure import ANGSTROM_TO_NM, Structure, write_multimodel_pdb
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

# nstlist=10 here (vs. 0 in SP_MDP): the Verlet cutoff scheme on GROMACS
# 2025.4 rejects nstlist=0 outright, independent of pbc/cutoffs -- confirmed
# by Task 6's supplementary grompp check, where nstlist=10 succeeded. nsteps=0
# makes the neighbour-list rebuild interval otherwise irrelevant here.
FALLBACK_MDP = """\
integrator              = md
nsteps                  = 0
pbc                     = xyz
cutoff-scheme           = Verlet
verlet-buffer-tolerance = -1
nstlist                 = 10
rlist                   = 3.0
coulombtype             = Cut-off
rcoulomb                = 3.0
vdwtype                 = Cut-off
rvdw                    = 3.0
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
    """
    gmx = binaries.require("gmx")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frames, dtype=float)
    nframes = frames.shape[0]

    traj = workdir / f"{log_stem}_traj.pdb"
    write_multimodel_pdb(traj, structure, frames)

    used_fallback = False
    for attempt, fallback in enumerate((False, True)):
        mdp = workdir / f"{log_stem}.mdp"
        write_mdp(mdp, fallback=fallback)

        conf = workdir / f"{log_stem}_conf.gro"
        _write_conf_gro(conf, structure, frames[0], box=_box_line(fallback))

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
                    f"{log_stem}.tpr",
                    "-maxwarn",
                    "5",
                ],
                workdir,
                f"{log_stem}_grompp",
            )
        except ExternalToolError:
            # Read the full grompp logs rather than the exception's truncated
            # tail: the Verlet-scheme rejection of a zero-cutoff vacuum mdp
            # emits several ERROR blocks (cutoff radius, pbc, nstlist), and
            # the one mentioning cutoffs is often pushed out of the tail.
            stdout = (workdir / f"{log_stem}_grompp.stdout").read_text()
            stderr = (workdir / f"{log_stem}_grompp.stderr").read_text()
            text = (stdout + stderr).lower()
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
        used_fallback = fallback
        break

    log.info(
        "GROMACS single point using %s path", "large-box fallback" if used_fallback else "vacuum"
    )

    if used_fallback:
        # mdrun -rerun reads the box for every frame from the trajectory file
        # itself, once; a multi-model PDB from write_multimodel_pdb carries no
        # CRYST1 record, so the pbc=xyz fallback needs one injected here.
        _inject_pdb_box(traj, FALLBACK_BOX_NM)

    run_command(
        [
            gmx,
            "mdrun",
            "-s",
            f"{log_stem}.tpr",
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
    _feed_stdin_energy(gmx, workdir, log_stem, selection)

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


def _feed_stdin_energy(gmx: str, workdir: Path, log_stem: str, selection: str) -> None:
    """Run ``gmx energy`` with the term names piped to stdin."""
    argv = [gmx, "energy", "-f", f"{log_stem}.edr", "-o", f"{log_stem}.xvg"]
    printable = " ".join(argv)
    log.info("running in %s: %s <<< %r", workdir, printable, selection)
    (workdir / f"{log_stem}_energy.cmd").write_text(
        printable + f"\n<<< {selection!r}\n", newline="\n"
    )
    proc = subprocess.run(argv, cwd=str(workdir), input=selection, capture_output=True, text=True)
    (workdir / f"{log_stem}_energy.stdout").write_text(proc.stdout or "", newline="\n")
    (workdir / f"{log_stem}_energy.stderr").write_text(proc.stderr or "", newline="\n")
    if proc.returncode != 0:
        tail = "\n".join(((proc.stdout or "") + (proc.stderr or "")).splitlines()[-30:])
        raise ExternalToolError(
            f"gmx energy failed with exit code {proc.returncode}:\n"
            f"  {printable}\n"
            f"  workdir: {workdir}\n"
            f"--- last 30 lines of output ---\n{tail}"
        )


def _inject_pdb_box(path: Path, box_nm: float) -> None:
    """Insert a single cubic ``CRYST1`` record ahead of the first MODEL.

    ``gmx mdrun -rerun`` reads the periodic box once from the trajectory file
    and applies it to every frame; without a CRYST1 record the fallback mdp's
    ``pbc = xyz`` makes mdrun abort with "does not contain a box, while pbc
    is used" (confirmed against GROMACS 2025.4).

    Args:
        path: multi-model PDB written by :func:`write_multimodel_pdb`.
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
