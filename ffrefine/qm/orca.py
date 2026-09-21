"""ORCA single-point driver for --sp-refine (spec section 4.3).

ORCA reports energies in Hartree, converted here through
:mod:`ffrefine.units`.  Geometries go out in Angstrom.

The installed ORCA is a *shared* build: it will not start unless its own
directory is on ``LD_LIBRARY_PATH``, which is what
:attr:`ffrefine.config.Binaries.orca_libdir` is for.  It is also on no PATH by
default, so ``--orca-bin`` is the normal way to reach it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..config import Binaries
from ..errors import ExternalToolError
from ..io.structure import Structure, write_xyz  # noqa: F401  (re-exported for symmetry)
from ..runner import run_command
from ..units import ANGSTROM_TO_NM, bohr_to_nm, hartree_to_kj, nm_to_bohr

DEFAULT_SP_KEYWORDS = "! wB97X-D3 def2-TZVP TightSCF"
"""Spec 4.3's default single-point level; overridable with --orca-sp-keywords."""

ESP_DEFAULT_KEYWORDS = "! HF 6-31G*"
"""Spec 5.2's ESP level of theory, without the scheme keyword.

HF/6-31G* is deliberate rather than dated: its systematic overpolarisation
approximates condensed-phase charges, which is why both Amber and the
CHARMM-adjacent workflows fit charges at this level.

The scheme appends ``CHELPG`` itself (see :func:`esp_keyword_line`), so that
overriding the level with --orca-esp-keywords cannot silently switch the
scheme off.
"""


def orca_env(binaries: Binaries) -> dict[str, str]:
    """Environment additions a shared ORCA build needs to start.

    Defaults ``orca_libdir`` to the ORCA binary's own directory, which is where
    a stock unpacked ORCA keeps its shared objects.
    """
    libdir = binaries.orca_libdir or str(Path(binaries.require("orca")).parent)
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    return {"LD_LIBRARY_PATH": f"{libdir}:{existing}" if existing else libdir}


def write_sp_input(
    path: str | Path,
    structure: Structure,
    coords_nm: np.ndarray,
    keywords: str,
    charge: int,
    mult: int,
    nprocs: int = 1,
) -> str:
    """Write an ORCA single-point input for one geometry.

    Args:
        path: file to write.
        structure: supplies the element symbols, in topology order.
        coords_nm: shape (natoms, 3) in nanometres; written in Angstrom.
        keywords: the ``!`` keyword line, e.g. ``"! HF STO-3G"``.
        charge: molecular charge.
        mult: spin multiplicity, ``uhf + 1``.
        nprocs: parallel workers; no ``%pal`` block is written for 1.

    Returns:
        The text written, so callers can assert on it without re-reading.
    """
    lines = [keywords]
    if nprocs > 1:
        lines.append(f"%pal nprocs {nprocs} end")
    lines.append("")
    lines.append(f"*xyz {charge} {mult}")
    for element, (x, y, z) in zip(
        structure.elements, np.asarray(coords_nm, float) / ANGSTROM_TO_NM
    ):
        lines.append(f"{element:<2s} {x:>14.8f} {y:>14.8f} {z:>14.8f}")
    lines.append("*")
    text = "\n".join(lines) + "\n"
    Path(path).write_text(text, newline="\n")
    return text


def parse_final_energy(text: str) -> float:
    """Extract ``FINAL SINGLE POINT ENERGY`` in Hartree from ORCA output.

    Takes the last occurrence: an optimisation prints one per cycle and only
    the converged one is meaningful.

    Raises:
        ExternalToolError: no such line, which means the job did not converge
            or died before reporting.
    """
    value = None
    for line in text.splitlines():
        if "FINAL SINGLE POINT ENERGY" in line:
            parts = line.split()
            try:
                value = float(parts[-1])
            except ValueError:
                continue
    if value is None:
        raise ExternalToolError(
            "ORCA produced no 'FINAL SINGLE POINT ENERGY' line, so the calculation did not "
            "reach a converged energy. Check the .stdout log in the stage directory."
        )
    return value


def refine_energies(
    binaries: Binaries,
    structure: Structure,
    frames_nm: np.ndarray,
    workdir: str | Path,
    keywords: str = DEFAULT_SP_KEYWORDS,
    charge: int = 0,
    mult: int = 1,
    nprocs: int = 1,
) -> np.ndarray:
    """Run one ORCA single point per frame, returning energies in kJ/mol.

    Spec 4.3: "Use these energies in place of the xtb ones when present."  The
    geometries are xtb's; only the energies change, so the fit target improves
    without the scan being redone.

    Raises:
        ExternalToolError: any frame's job fails or reports no final energy.
    """
    exe = binaries.require("orca")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    env = orca_env(binaries)

    out = []
    for index, frame in enumerate(np.asarray(frames_nm, float)):
        stem = f"sp_{index:03d}"
        inp = workdir / f"{stem}.inp"
        write_sp_input(inp, structure, frame, keywords, charge, mult, nprocs)
        proc = run_command([exe, inp.name], workdir, stem, extra_env=env)
        try:
            out.append(parse_final_energy(proc.stdout or ""))
        except ExternalToolError as exc:
            raise ExternalToolError(f"frame {index}: {exc}") from exc
    return hartree_to_kj(np.array(out, dtype=float))


ESP_ARTEFACTS: tuple[str, ...] = (
    ".chelpg.xyz",
    ".vpot",
    ".gbw",
    ".densities",
    ".densitiesinfo",
)
"""Suffixes of the ESP job's outputs that later steps read, deleted before each run.

``.chelpg.xyz`` and ``.vpot`` feed the CHELPG charges and the RRMS; ``.gbw``,
``.densities`` and ``.densitiesinfo`` are what ``orca_vpot`` reads under RESP.
There is no ``.scfp`` file to delete: that is the density's name inside
``.densities``.
"""

ORCA_NORMAL_TERMINATION_MARKER = "ORCA TERMINATED NORMALLY"
"""Printed at the very end of a run that finished; absent after any error."""

ORCA_SCF_FAILED_MARKER = "SCF NOT CONVERGED"
"""Printed when the SCF gives up -- and ORCA 6.1.1 still exits 0."""


def esp_keyword_line(keywords: str, chelpg: bool) -> str:
    """The ORCA keyword line for an ESP job, adding CHELPG when the scheme needs it."""
    if chelpg and "chelpg" not in keywords.lower():
        return f"{keywords} CHELPG"
    return keywords


def run_esp_job(
    binaries: Binaries,
    structure: Structure,
    coords_nm: np.ndarray,
    workdir: str | Path,
    keywords: str = ESP_DEFAULT_KEYWORDS,
    charge: int = 0,
    mult: int = 1,
    nprocs: int = 1,
    chelpg: bool = True,
) -> Path:
    """Run the single ESP job both charge schemes are built on.

    One job serves both: RESP needs the converged density for ``orca_vpot``,
    and CHELPG costs nothing on top of it.  The run also leaves ``<base>.vpot``
    -- ORCA's own grid carrying the QM potential -- which is what lets the
    CHELPG path report a fit RRMS comparable with RESP's.

    **Exit 0 proves nothing here.**  ORCA 6.1.1 exits 0 when its SCF does
    not converge, prints ``SCF NOT CONVERGED`` and ``ORCA finished by error
    termination``, and writes no ``.chelpg.xyz`` (probed 2026-09-21 with
    ``%scf MaxIter 3 end``).  ``04_charges/`` is reused when a stage is
    re-run into the same outdir, so a check that the output files merely
    exist would find the previous run's files and return its charges under
    the new run's keyword line.  Two things close that:

    * every artefact a later step reads (:data:`ESP_ARTEFACTS`) is deleted
      before ORCA starts, so nothing from an earlier run can survive a
      failed one; and
    * the run is refused unless ORCA's stdout says ``ORCA TERMINATED
      NORMALLY`` and does not say ``SCF NOT CONVERGED``.

    Returns:
        The job's base path.  ``<base>.gbw``, ``<base>.densities``,
        ``<base>.chelpg.xyz`` and ``<base>.vpot`` hang off it.  The SCF
        density lives *inside* ``<base>.densities`` under the name
        ``<base>.scfp``; that name is not a file on disk, and it is exactly
        what :func:`run_vpot` passes as ``orca_vpot``'s density argument.

    Raises:
        ExternalToolError: ORCA exited nonzero, its SCF did not converge, it
            did not terminate normally, or it produced no CHELPG file when
            asked.
    """
    exe = binaries.require("orca")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    base = workdir / "esp"
    for suffix in ESP_ARTEFACTS:
        Path(str(base) + suffix).unlink(missing_ok=True)
    write_sp_input(
        base.with_suffix(".inp"),
        structure,
        coords_nm,
        esp_keyword_line(keywords, chelpg),
        charge,
        mult,
        nprocs,
    )
    proc = run_command(
        [exe, base.with_suffix(".inp").name], workdir, "esp", extra_env=orca_env(binaries)
    )
    stdout = proc.stdout or ""
    log_path = workdir / f"{base.name}.stdout"
    if ORCA_SCF_FAILED_MARKER in stdout:
        raise ExternalToolError(
            f"ORCA's SCF did not converge in the ESP job: its output says "
            f"'{ORCA_SCF_FAILED_MARKER}'. ORCA exits 0 when this happens, so the exit code "
            f"proves nothing, and there is no density to take charges or a potential from. "
            f"See {log_path}. A different geometry (--charge-geom), a corrected --charge or "
            "--uhf, or another level of theory may converge."
        )
    if ORCA_NORMAL_TERMINATION_MARKER not in stdout:
        raise ExternalToolError(
            f"ORCA's ESP job exited without printing '{ORCA_NORMAL_TERMINATION_MARKER}', so "
            f"it stopped before finishing even though its exit code was 0. See {log_path} "
            "for the error ORCA reported."
        )
    if chelpg and not Path(str(base) + ".chelpg.xyz").exists():
        raise ExternalToolError(
            f"ORCA terminated normally with a converged SCF but wrote no "
            f"{base.name}.chelpg.xyz, so the CHELPG step never produced charges. See "
            f"{log_path} for the CHELPG section of the output."
        )
    return base


def parse_chelpg_charges(path: str | Path) -> np.ndarray:
    """Read ORCA's ``<base>.chelpg.xyz`` into an array of charges in e.

    Preferred over the stdout table: this file carries eight decimals to the
    table's six, and no surrounding banner that can change between versions.

    Raises:
        ExternalToolError: the file has no charge column, which means the run
            did not produce CHELPG charges.
    """
    lines = Path(path).read_text().splitlines()
    if len(lines) < 3:
        raise ExternalToolError(f"{path}: too short to be a CHELPG point-charge file")
    count = int(lines[0].split()[0])
    charges = []
    for line in lines[2 : 2 + count]:
        fields = line.split()
        if len(fields) < 5:
            raise ExternalToolError(
                f"{path}: line {line!r} has no charge column. ORCA writes element, x, y, z "
                "and charge; a four-column file is a plain xyz, not a CHELPG result."
            )
        charges.append(float(fields[4]))
    if len(charges) != count:
        raise ExternalToolError(f"{path}: header says {count} atoms, found {len(charges)}")
    return np.array(charges, dtype=float)


def read_chelpg_vpot(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the ``.vpot`` written by a CHELPG run.

    **Column order matters.** This file is ``V x y z``; the file ``orca_vpot``
    writes is ``x y z V``.  Same extension, reversed order, and reading either
    with the other's parser yields coordinates that look like potentials while
    the fit carries on producing numbers.

    Returns:
        ``(grid_nm, v_hartree_per_e)`` with shapes ``(npoints, 3)`` and
        ``(npoints,)``.  Coordinates are converted from bohr here, at the
        process boundary (spec section 9.8).

    Raises:
        ExternalToolError: the file holds fewer points than its header claims.
    """
    lines = Path(path).read_text().splitlines()
    natoms, npoints = (int(v) for v in lines[0].split()[:2])
    rows = [ln.split() for ln in lines[1 + natoms : 1 + natoms + npoints] if ln.strip()]
    if len(rows) != npoints:
        raise ExternalToolError(
            f"{path}: header claims {npoints} grid points, found {len(rows)}. The file is "
            "truncated, which usually means ORCA was killed part-way through."
        )
    data = np.array([[float(x) for x in row[:4]] for row in rows], dtype=float)
    return bohr_to_nm(data[:, 1:4]), data[:, 0]


def write_vpot_grid(path: str | Path, grid_nm: np.ndarray) -> None:
    """Write grid points for ``orca_vpot``: a count line, then bohr coordinates."""
    grid = nm_to_bohr(np.asarray(grid_nm, float))
    lines = [str(len(grid))]
    lines += [f"{x:>22.14f} {y:>22.14f} {z:>22.14f}" for x, y, z in grid]
    Path(path).write_text("\n".join(lines) + "\n", newline="\n")


def read_vpot_output(path: str | Path, expected_points: int) -> np.ndarray:
    """Read the potential from the file ``orca_vpot`` writes.

    **Columns are ``x y z V``** here -- the reverse of the ``.vpot`` a CHELPG
    run writes, which is ``V x y z``.  Two functions rather than one flag,
    because the wrong choice produces a plausible array either way.

    Args:
        path: the output file.
        expected_points: how many points were sent.  Checked, because a wrong
            density name makes ``orca_vpot`` segfault, and a truncated or
            missing file is the only evidence of that.

    Returns:
        Potentials in Hartree/e, shape ``(expected_points,)``.

    Raises:
        ExternalToolError: the file is missing, short, or malformed.
    """
    path = Path(path)
    if not path.exists():
        raise ExternalToolError(
            f"orca_vpot wrote no {path}. It exits by segfault when it cannot read the "
            "density, so an absent output usually means the density name was wrong."
        )
    rows = [ln.split() for ln in path.read_text().splitlines()[1:] if ln.strip()]
    if len(rows) != expected_points:
        raise ExternalToolError(
            f"{path}: {len(rows)} potential values for {expected_points} grid points. "
            "orca_vpot did not finish; check vpot.stdout in the stage directory."
        )
    return np.array([float(row[3]) for row in rows], dtype=float)


def run_vpot(
    binaries: Binaries,
    base: Path,
    grid_nm: np.ndarray,
    workdir: str | Path,
) -> np.ndarray:
    """Evaluate the QM potential at ``grid_nm`` against a finished ESP job.

    The invocation was established by probe against ORCA 6.1.1 rather than
    taken from the usage text, as spec 5.2 requires:

        orca_vpot <base>.gbw <base>.scfp grid.xyz out.vpot

    The density argument is the **full** ``<base>.scfp``.  Passing the bare
    ``scfp`` that the usage text suggests segfaults (exit 139), and the optional
    fifth argument is unnecessary.  ``<base>.scfp`` is not a file: it is the
    name under which ORCA 6.1.1 stores the SCF density inside
    ``<base>.densities``, and ``orca_vpot`` looks it up there.

    ``binaries.orca_vpot`` defaults to ``orca_vpot`` in ``orca``'s own
    directory when not set explicitly, the same fallback :func:`orca_env`
    already applies to ``orca_libdir`` -- a stock ORCA unpack keeps the two
    binaries side by side.

    ``out_path`` is removed before the call, not merely overwritten: a
    segfaulting ``orca_vpot`` leaves no output at all, and without this a
    stale file from an earlier call into the same ``workdir`` -- e.g. a
    re-run of the RESP stage with different ESP keywords but the same grid
    -- could carry the same point count and pass every check in
    :func:`read_vpot_output` while holding the previous run's potentials.

    Returns:
        Potentials in Hartree/e, one per grid point.

    Raises:
        ExternalToolError: orca_vpot failed, or produced the wrong point count.
    """
    exe = binaries.orca_vpot or str(Path(binaries.require("orca")).parent / "orca_vpot")
    workdir = Path(workdir)
    grid_path = workdir / "grid.xyz"
    out_path = workdir / "grid_out.vpot"
    write_vpot_grid(grid_path, grid_nm)
    out_path.unlink(missing_ok=True)
    run_command(
        [exe, f"{base.name}.gbw", f"{base.name}.scfp", grid_path.name, out_path.name],
        workdir,
        "vpot",
        extra_env=orca_env(binaries),
        check=False,
    )
    return read_vpot_output(out_path, expected_points=len(grid_nm))
