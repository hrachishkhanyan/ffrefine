"""xtb relaxed torsion scan driver (spec section 4.3).

xtb speaks Angstrom and Hartree; ffrefine speaks nm and kJ/mol.  Both
conversions go through :mod:`ffrefine.units`, which is the only place they
are allowed to live (spec section 9.8).

The ``$scan`` syntax this module emits was verified against xtb 6.7.1 by live
probe rather than assumed -- see the design document's section 2.  Spec 4.3
requires exactly that, and :func:`run_scan` re-verifies at runtime that the
block was actually honoured.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..config import Binaries
from ..errors import ExternalToolError, ParseError, ValidationError
from ..io.structure import Structure, write_xyz
from ..io.topology import Topology
from ..runner import run_command
from ..units import angstrom_to_nm, hartree_to_kj

Quad = tuple[int, int, int, int]


def scan_grid(step_deg: float) -> tuple[float, float, int]:
    """Return ``(start_deg, end_deg, npoints)`` for a full turn at ``step_deg``.

    Spec 4.3: the grid spans 0 to ``360 - step`` so the endpoints are not
    duplicated.  0 and 360 degrees are the same angle; sampling both would put
    two points on one geometry, which biases any averaging over the profile
    and produces a duplicated abscissa in every plot.

    Raises:
        ValidationError: ``step_deg`` is not positive, or does not divide 360
            into a whole number of steps (which would leave the grid unable to
            close the turn).
    """
    if step_deg <= 0:
        raise ValidationError(f"--step must be positive, got {step_deg}")
    npoints = round(360.0 / step_deg)
    if abs(npoints * step_deg - 360.0) > 1e-9:
        raise ValidationError(
            f"--step {step_deg} does not divide 360 into whole steps "
            f"({360.0 / step_deg:.4f} of them). Use a step such as 5, 10, 15, 20 or 30."
        )
    return 0.0, 360.0 - step_deg, npoints


def _constraint_line(quad: Sequence[int]) -> str:
    if len(set(quad)) != 4:
        raise ValidationError(f"dihedral {tuple(quad)} repeats an atom index")
    return "  dihedral: " + ",".join(str(i) for i in quad) + ",auto"


def write_scan_input(
    path: str | Path,
    quad: Quad,
    start_deg: float,
    end_deg: float,
    npoints: int,
    force_constant: float = 1.0,
    freeze: Iterable[Quad] = (),
) -> str:
    """Write the xcontrol file driving a relaxed scan of ``quad``.

    ``auto`` makes xtb take each constrained dihedral's starting value from the
    input geometry, so the frozen dihedrals hold wherever the user's structure
    put them.  The scanned dihedral is listed first and is the only one given a
    ``$scan`` entry; ``freeze`` entries are constrained but never scanned,
    which is the whole point of ``--freeze-dihedral`` (spec 4.3).

    Args:
        path: destination xcontrol file.
        quad: 1-based target dihedral.
        start_deg, end_deg: scan range; ``end_deg`` may be below ``start_deg``
            for a reverse leg, which xtb accepts (verified on 6.7.1).
        npoints: number of scan points.
        force_constant: xtb constraint force constant.
        freeze: additional 1-based dihedrals to hold fixed.

    Returns:
        The text written.

    Raises:
        ValidationError: a quadruplet repeats an atom index.
    """
    lines = ["$constrain", f"  force constant={force_constant}", _constraint_line(quad)]
    for other in freeze:
        lines.append(_constraint_line(other))
    lines.append("$scan")
    lines.append(f"  1: {start_deg},{end_deg},{npoints}")
    lines.append("$end")
    text = "\n".join(lines) + "\n"
    Path(path).write_text(text, newline="\n")
    return text


def parse_scan_log(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read an ``xtbscan.log`` into coordinates and energies.

    The file is a plain multi-frame XYZ: an atom-count line, a comment line of
    the form ``energy: -13.665127531525 xtb: 6.7.1 (edcfbbe)``, then one line
    per atom with the element symbol and Angstrom coordinates.  Format
    confirmed against xtb 6.7.1 output, committed under ``fixtures/butane/xtb``.

    Every failure mode here is a *loud* one on purpose: a killed xtb leaves a
    truncated final frame, and silently returning the frames that did parse
    would hand the fit a profile with a missing grid point and no indication
    anything went wrong.

    Args:
        path: the log file.

    Returns:
        ``(frames_nm, energy_kj)`` with shapes ``(nframes, natoms, 3)`` in
        nanometres and ``(nframes,)`` in kJ/mol.  Energies are absolute, not
        referenced to a minimum.

    Raises:
        ParseError: the file is empty, a frame is truncated, a comment line
            carries no ``energy:``, or the atom count varies between frames.
    """
    path = Path(path)
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ParseError(f"could not read {path}: {exc}") from exc

    frames: list[list[list[float]]] = []
    energies: list[float] = []
    natoms: int | None = None
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        try:
            count = int(lines[i].split()[0])
        except (ValueError, IndexError) as exc:
            raise ParseError(f"{path}: expected an atom count at line {i + 1}") from exc
        if natoms is None:
            natoms = count
        elif count != natoms:
            raise ParseError(
                f"{path}: atom count changes from {natoms} to {count} at line {i + 1}; "
                "every frame must describe the same molecule"
            )
        if i + 1 >= len(lines):
            raise ParseError(f"{path}: truncated -- no comment line after line {i + 1}")
        comment = lines[i + 1]
        if "energy:" not in comment:
            raise ParseError(
                f"{path}: line {i + 2} carries no 'energy:' value: {comment.strip()!r}"
            )
        try:
            energies.append(float(comment.split("energy:")[1].split()[0]))
        except (ValueError, IndexError) as exc:
            raise ParseError(f"{path}: unreadable energy on line {i + 2}") from exc

        body = lines[i + 2 : i + 2 + count]
        if len(body) != count:
            raise ParseError(
                f"{path}: truncated -- frame {len(frames) + 1} declares {count} atoms "
                f"but only {len(body)} follow. xtb probably did not finish."
            )
        frame = []
        for offset, line in enumerate(body):
            parts = line.split()
            if len(parts) < 4:
                raise ParseError(f"{path}: incomplete atom line {i + 3 + offset}: {line!r}")
            try:
                frame.append([float(v) for v in parts[1:4]])
            except ValueError as exc:
                raise ParseError(
                    f"{path}: unreadable coordinates on line {i + 3 + offset}"
                ) from exc
        frames.append(frame)
        i += 2 + count

    if not frames:
        raise ParseError(f"{path}: no frames found; xtb wrote no scan output")
    return angstrom_to_nm(np.array(frames, dtype=float)), hartree_to_kj(
        np.array(energies, dtype=float)
    )


SCAN_LOG_NAME = "xtbscan.log"
"""Filename xtb writes its relaxed-scan trajectory to, inside the working directory."""


def run_scan(
    binaries: Binaries,
    structure: Structure,
    quad: Quad,
    workdir: str | Path,
    *,
    start_deg: float,
    end_deg: float,
    npoints: int,
    charge: int,
    uhf: int,
    gfn: int = 2,
    nthreads: int = 1,
    freeze: Iterable[Quad] = (),
    force_constant: float = 1.0,
    coords_nm: np.ndarray | None = None,
    log_stem: str = "xtb_scan",
) -> tuple[np.ndarray, np.ndarray]:
    """Run one relaxed-scan leg and return its geometries and energies.

    Each leg gets its own ``workdir`` because xtb always writes
    ``xtbscan.log`` into the current directory: running a forward and a
    reverse leg in one directory would silently overwrite the first.

    Spec 4.3 requires the ``$scan`` syntax be verified against the installed
    xtb at runtime rather than assumed.  ``xtb --help`` documents no xcontrol
    block at all (checked on 6.7.1), so verification here is behavioural: the
    run must report a relaxed scan, must leave a log, and that log must hold
    exactly the requested number of frames.  A rejected block can otherwise
    leave xtb exiting 0 having quietly optimised the input geometry once.

    Args:
        binaries: resolved external programs; ``xtb`` must be set.
        structure: supplies elements and the default geometry.
        quad: 1-based target dihedral.
        workdir: directory to run in; created if absent.
        start_deg, end_deg, npoints: the scan range (see :func:`scan_grid`).
        charge: total molecular charge, passed as ``--chrg``.
        uhf: number of unpaired electrons, passed as ``--uhf``.
        gfn: GFN parameterisation, passed as ``--gfn``.
        nthreads: passed as ``-P`` (spec section 8).
        freeze: extra dihedrals held fixed.
        force_constant: constraint force constant.
        coords_nm: starting geometry in nm; defaults to ``structure.coords``.
            The reverse leg passes the forward leg's final frame.
        log_stem: base name for the runner's .cmd/.stdout/.stderr files.

    Returns:
        ``(frames_nm, energy_kj)`` as from :func:`parse_scan_log`.

    Raises:
        FFRefineError: xtb is not available.
        ExternalToolError: xtb failed, wrote no scan log, or returned a
            different number of frames than were requested.
    """
    exe = binaries.require("xtb")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    mol = workdir / "mol.xyz"
    write_xyz(mol, structure, coords_nm, comment=f"ffrefine scan input, torsion {quad}")
    inp = workdir / "scan.inp"
    write_scan_input(inp, quad, start_deg, end_deg, npoints, force_constant, freeze)

    proc = run_command(
        [
            exe,
            mol.name,
            "--opt",
            "tight",
            "--input",
            inp.name,
            "--chrg",
            str(charge),
            "--uhf",
            str(uhf),
            "--gfn",
            str(gfn),
            "-P",
            str(nthreads),
        ],
        workdir,
        log_stem,
    )

    log_path = workdir / SCAN_LOG_NAME
    if not log_path.exists():
        raise ExternalToolError(
            f"xtb exited 0 but wrote no {SCAN_LOG_NAME} in {workdir}, so no relaxed scan "
            f"ran. The $scan block in {inp.name} was most likely rejected by this xtb "
            f"build. Check {log_stem}.stdout for what it made of the input."
        )
    if "RELAXED SCAN" not in (proc.stdout or ""):
        raise ExternalToolError(
            f"xtb wrote {SCAN_LOG_NAME} but never reported 'RELAXED SCAN'; the $scan block "
            f"in {inp.name} was not honoured. See {log_stem}.stdout."
        )

    frames_nm, energy_kj = parse_scan_log(log_path)
    if len(energy_kj) != npoints:
        raise ExternalToolError(
            f"requested {npoints} scan points but {log_path} holds {len(energy_kj)}. "
            "The scan did not complete, so the profile would have a hole in it."
        )
    if frames_nm.shape[1] != structure.natoms:
        raise ExternalToolError(
            f"{log_path} has {frames_nm.shape[1]} atoms per frame, but the structure has "
            f"{structure.natoms}"
        )
    # R44: xtb can report RELAXED SCAN, exit 0, and write a log with exactly the
    # requested frame count while the constrained optimiser stalled and never
    # moved the geometry past its pre-scan value -- every check above passes on
    # that output. A real relaxed scan moves the atoms at every grid point by
    # construction, so frames identical to the first are direct proof the scan
    # did not run and must be refused rather than returned as data.
    if len(frames_nm) > 1 and np.all(frames_nm[1:] == frames_nm[0]):
        raise ExternalToolError(
            f"all {len(frames_nm)} frames in {log_path} are identical to the input "
            "geometry; the scan reported success but never moved the structure. This "
            "has been seen when the xtb relaxed-scan optimiser stalls under -P values "
            "greater than 1. Re-run with --nthreads 1."
        )
    return frames_nm, energy_kj


CHARGE_TOLERANCE = 0.05
"""How far the topology's charge sum may sit from an integer before it is refused."""


def derive_charge(
    topology: Topology, override: int | None = None, tol: float = CHARGE_TOLERANCE
) -> int:
    """Determine the total molecular charge for xtb's ``--chrg``.

    Spec 4.3 requires the flag but does not say where the value comes from.
    The topology knows: ``[ atoms ]`` charges sum to the molecular charge by
    construction.  Deriving it beats asking, because a user who forgets
    ``--charge`` on an ionic molecule gets a silently wrong electronic
    structure and therefore a silently wrong torsion profile -- there is no
    error, just a bad answer.

    Force field charges print to three or four decimals, so the sum is rarely
    exact; it is rounded.  A sum far from any integer means the topology is
    incomplete or corrupt, and guessing a charge for it would be worse than
    stopping.

    Multiplicity is deliberately NOT derived.  Electron-count parity cannot
    distinguish a triplet from a singlet diradical, so ``--uhf`` stays explicit
    and defaults to 0.

    Args:
        topology: the parsed molecule.
        override: an explicit ``--charge`` from the user; wins outright, and is
            not re-checked against the sum -- it is the escape hatch the
            refusal below advertises, so it has to work on the topology that
            triggered the refusal.
        tol: largest allowed departure from an integer.

    Returns:
        The integer charge.

    Raises:
        ValidationError: the charge sum is not within ``tol`` of an integer.
    """
    if override is not None:
        return int(override)
    total = sum(atom.charge for atom in topology.atoms)
    nearest = round(total)
    if abs(total - nearest) > tol:
        raise ValidationError(
            f"the topology's [ atoms ] charges sum to {total:.4f}, which is "
            f"{abs(total - nearest):.4f} from the nearest integer ({nearest}) and outside "
            f"the {tol} tolerance. That usually means the topology is incomplete. Pass "
            "--charge to state the molecular charge explicitly."
        )
    return int(nearest)
