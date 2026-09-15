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
from ..units import ANGSTROM_TO_NM, hartree_to_kj

DEFAULT_SP_KEYWORDS = "! wB97X-D3 def2-TZVP TightSCF"
"""Spec 4.3's default single-point level; overridable with --orca-sp-keywords."""


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
