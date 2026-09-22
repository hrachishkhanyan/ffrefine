"""Command line interface (spec section 3).

Subcommands: ``scan``, ``mmscan``, ``fit``, ``charges``, ``validate``,
``run``.  All are implemented.  ``ffrefine charges`` runs both
``--charge-scheme chelpg`` and ``--charge-scheme resp`` end to end.

Atom-index convention (spec section 8)
--------------------------------------
Every atom index on the command line is **1-based**, matching the topology's
``[ atoms ]`` numbering.  :func:`parse_dihedral` is where that convention
enters the program and it does no conversion; the 1-based quadruplet is passed
through to the topology-level helpers, which are 1-based throughout, and the
single conversion to 0-based array indices happens at the geometry boundary
inside :func:`ffrefine.mm.selfcheck.check_torsion_removal` and
:func:`ffrefine.mm.validate_fit.validate_refit`.  This module never indexes a
coordinate array itself.

How errors reach the user (ruling R29)
--------------------------------------
There are exactly two channels, and neither produces a traceback for bad
input:

* argparse usage errors -- an unknown flag, a missing required option, a
  malformed ``--dihedral`` -- print argparse's own message and raise
  ``SystemExit(2)``.  :func:`parse_dihedral` raises
  :class:`~ffrefine.errors.ValidationError` like every other validator in this
  project; ``_dihedral_arg`` adapts it into ``argparse.ArgumentTypeError`` at
  the parser boundary so argparse can report it in its usual style.
* Any :class:`~ffrefine.errors.FFRefineError` raised by the pipeline (which
  includes ``ValidationError``, ``ParseError`` and ``ExternalToolError``) is
  caught in :func:`main`, reported as ``error: ...`` on stderr, and turned
  into exit code 1.

Exit codes: 0 success, 1 a reported ffrefine failure, 2 a usage error.

Stage files this module reads and writes
----------------------------------------
The directory names come from :data:`ffrefine.stages.STAGE_DIRS` (spec
section 6).  Two JSON payloads carry data between stages:

``01_qm_scan/scan.json`` -- produced by ``ffrefine scan`` once milestone 4
exists, and accepted from the user in the meantime::

    {"phi_deg": [...],            # measured dihedral, degrees (informational)
     "energy_kj": [...],          # QM energies, kJ/mol, one per frame
     "frames_nm": [[[x, y, z], ...], ...]}   # (nframes, natoms, 3), nm

``02_mm_scan/energies.json`` -- written by ``mmscan``::

    {"quad": [i, j, k, l], "phi_deg": [...],
     "full_potential_kj": [...], "zeroed_potential_kj": [...],
     "frames_sha256": "...", "selfcheck_max_abs_error_kj": ..., "nframes": ...}

``phi_deg`` in ``energies.json`` is *measured from the geometries GROMACS
actually saw*, not the requested grid (spec section 9.5), and it is the angle
array the fit uses.

How stage payloads are paired
-----------------------------
Matching array lengths are NOT evidence that two payloads belong together.
``energies.json`` records the ``quad`` it was computed for and a
``frames_sha256`` digest of the geometries it saw; ``fit.json`` records both
again under ``provenance``.  :func:`_check_stage_pairing` and
:func:`_check_fit_applies_here` compare them and refuse a mismatch, so
fitting one torsion's energies and writing the answer onto another -- or
fitting against geometries ``mmscan`` never saw -- is an error rather than a
confident wrong number.  The digest covers ``frames_nm`` only, so replacing
the QM ``energy_kj`` column (the documented workflow, and spec section 7.1's
round trip) stays legal.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import logging
import os
import re
import shutil
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import __version__
from .charges.apply import apply_charges, charge_decimals, check_selection
from .charges.equivalence import average_within_groups, equivalent_hydrogen_groups
from .charges.esp import inverse_distances, potential_from_charges, rrms
from .charges.grid import MK_SHELL_SCALES, VDW_RADII_ANGSTROM, mk_grid
from .charges.resp import RESP_A_STAGE1, RESP_A_STAGE2, RESP_B, resp_fit
from .config import Binaries, discover_binaries, load_config
from .errors import FFRefineError, ValidationError
from .fit.linear import DEFAULT_ASYMMETRY_TOLERANCE, asymmetry_metric, is_asymmetric
from .fit.report import fit_summary, profile_stats
from .fit.selection import (
    DEFAULT_MULTIPLICITIES,
    format_ranking_table,
    rank_candidates,
    select_model,
)
from .fit.weights import DEFAULT_EMAX, DEFAULT_TEMPERATURE, boltzmann_weights, uniform_weights
from .io.includes import has_includes
from .io.prm import write_prm_block
from .io.structure import Structure, read_structure, write_multimodel_pdb
from .io.topology import Defaults, Topology, read_topology
from .io.topwriter import write_charged_topology, write_refit_topology
from .io.validate import adopt_topology_naming, element_of, validate_structure_topology
from .mm.selfcheck import check_torsion_removal
from .mm.torsionset import collect_torsion_set, validate_quad
from .mm.tprparams import ResolvedTerm, preprocessed_sha256, resolve_torsion_set
from .mm.validate_fit import validate_refit
from .plots import plot_fit, plot_mm_zeroed, plot_qm_profile, plot_residuals, plot_validation
from .qm.scan import (
    ScanLeg,
    find_hops,
    hbond_changes,
    heavy_atom_indices,
    measure_angles,
    merge_bidirectional,
)
from .qm.orca import (
    DEFAULT_SP_KEYWORDS,
    ESP_DEFAULT_KEYWORDS,
    orca_env,
    parse_chelpg_charges,
    read_chelpg_vpot,
    refine_energies,
    run_esp_job,
    run_vpot,
)
from .qm.xtb import derive_charge, optimize_geometry, run_scan, scan_grid
from .stages import (
    STAGE_DIRS,
    hash_array,
    hash_inputs,
    log_to_run_log,
    setup_logging,
    stage_dir,
    stage_is_current,
    write_manifest,
)
from .units import nm_to_bohr

log = logging.getLogger("ffrefine")

Quad = tuple[int, int, int, int]

_CONFIG_KEYS: tuple[str, ...] = (
    "outdir",
    "nthreads",
    "gmx_bin",
    "xtb_bin",
    "orca_bin",
    "multiplicities",
    "emax",
    "boltzmann",
    "step",
)
"""Options a ``--config`` file may set.  A CLI value always wins over a file
value, which is why every one of these has an argparse default of ``None``:
without that, argparse's own default would be indistinguishable from a value
the user typed and would silently beat the config file (spec section 3)."""

_DEFAULTS: dict[str, object] = {
    "outdir": "ffrefine_out",
    "nthreads": 1,
    "emax": DEFAULT_EMAX,
    "multiplicities": list(DEFAULT_MULTIPLICITIES),
    "step": 15.0,
}
"""Applied after the config file, for keys neither the CLI nor the file set."""


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


def parse_dihedral(text: str) -> Quad:
    """Parse an ``i,j,k,l`` dihedral specification of 1-based atom indices.

    Args:
        text: four comma-separated integers; surrounding spaces are ignored.

    Returns:
        The quadruplet, still 1-based -- no conversion happens here.

    Raises:
        ValidationError: wrong number of fields, a non-integer field, or an
            index below 1.  Whether the indices exist in a particular
            topology is a separate check
            (:func:`ffrefine.mm.torsionset.validate_quad`), made once the
            topology has been read.
    """
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise ValidationError(
            f"--dihedral needs exactly 4 comma-separated indices, got {text!r} ({len(parts)})"
        )
    try:
        idx = tuple(int(p) for p in parts)
    except ValueError:
        raise ValidationError(f"--dihedral indices must be integers, got {text!r}") from None
    if any(i < 1 for i in idx):
        raise ValidationError(
            f"--dihedral indices are 1-based (they match the topology's [ atoms ] "
            f"numbering); {text!r} contains a value below 1"
        )
    return idx  # type: ignore[return-value]


def _dihedral_arg(text: str) -> Quad:
    """argparse ``type=`` adapter for :func:`parse_dihedral`.

    argparse only recognises ``ValueError``, ``TypeError`` and
    ``ArgumentTypeError`` as "this value is bad"; anything else escapes as an
    unhandled traceback.  Translating here keeps ``parse_dihedral`` raising
    the project's own ``ValidationError`` (ruling R29) while still giving the
    user argparse's clean ``invalid value`` message and exit code 2.
    """
    try:
        return parse_dihedral(text)
    except ValidationError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _parse_mults(text: str) -> list[int]:
    """Parse a ``1,2,3`` multiplicity list."""
    try:
        values = [int(p.strip()) for p in text.split(",") if p.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--multiplicities must be comma-separated integers, got {text!r}"
        ) from None
    if not values:
        raise argparse.ArgumentTypeError("--multiplicities must list at least one multiplicity")
    if any(v < 1 for v in values):
        raise argparse.ArgumentTypeError(f"multiplicities must be >= 1, got {text!r}")
    return values


def _add_common(parser: argparse.ArgumentParser) -> None:
    """Add the flags spec section 3 requires on every subcommand."""
    parser.add_argument("-s", "--structure", required=True, help="input .pdb, .gro or .xyz")
    parser.add_argument("-p", "--topology", required=True, help="input .itp or .top")
    parser.add_argument("--config", help="JSON config file; CLI flags override it")
    parser.add_argument("--outdir", help="output directory (default: ffrefine_out)")
    parser.add_argument("--nthreads", type=int, help="threads for external tools (default: 1)")
    parser.add_argument(
        "--resume", action="store_true", help="skip stages whose inputs are unchanged"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would run, change nothing"
    )
    parser.add_argument("--verbose", action="store_true", help="debug-level logging")
    parser.add_argument("--gmx-bin", help="explicit path to gmx")
    parser.add_argument("--xtb-bin", help="explicit path to xtb")
    parser.add_argument("--orca-bin", help="explicit path to orca")
    parser.add_argument(
        "--no-strict-names",
        action="store_true",
        help="downgrade atom-name mismatches to warnings",
    )
    parser.add_argument(
        "--ff-include",
        action="append",
        default=None,
        metavar="PATH",
        help=(
            "force field .itp to #include ahead of the molecule in the generated "
            "wrapper .top; repeatable, in order. Needed when the molecule topology "
            "has no [ atomtypes ] of its own."
        ),
    )


def _add_torsion(parser: argparse.ArgumentParser) -> None:
    """Add the target-torsion flags."""
    parser.add_argument(
        "--dihedral",
        action="append",
        type=_dihedral_arg,
        default=None,
        help="target torsion as i,j,k,l (1-based); repeatable",
    )
    parser.add_argument(
        "--distribute",
        action="store_true",
        help="spread the fitted k_n over all equivalent quadruplets (spec section 9.7)",
    )


def _add_qm_flags(parser: argparse.ArgumentParser) -> None:
    """Add the flags describing the QM electronic state.

    Shared by ``scan`` (xtb relaxed scans) and ``charges`` (ORCA ESP jobs and
    the xtb geometry optimisation ``--charge-geom xtb`` runs before them), and
    by ``run``, which chains both.
    """
    parser.add_argument("--charge", type=int, help="molecular charge (default: from the topology)")
    parser.add_argument("--uhf", type=int, default=0, help="unpaired electrons (default: 0)")


def _add_scan_flags(parser: argparse.ArgumentParser) -> None:
    """Add the QM scan flags (also needed by ``run``, which chains the scan)."""
    parser.add_argument("--step", type=float, help="scan step in degrees (default: 15)")
    parser.add_argument("--gfn", type=int, default=2, help="xtb GFN parameterisation (default: 2)")
    parser.add_argument(
        "--bidirectional",
        dest="bidirectional",
        action="store_true",
        default=True,
        help="scan forward then back, taking the lower energy (default)",
    )
    parser.add_argument(
        "--no-bidirectional",
        dest="bidirectional",
        action="store_false",
        help="scan in one direction only",
    )
    parser.add_argument(
        "--hysteresis-tol",
        type=float,
        default=4.0,
        help="warn when the two directions differ by more than this, kJ/mol (default: 4)",
    )
    parser.add_argument(
        "--freeze-dihedral",
        action="append",
        type=_dihedral_arg,
        default=None,
        metavar="i,j,k,l",
        help="hold another dihedral fixed; repeatable",
    )
    parser.add_argument(
        "--sp-refine",
        action="store_true",
        help="refine the xtb energies with ORCA single points on the same geometries",
    )
    parser.add_argument(
        "--orca-sp-keywords",
        default=None,
        help=f"ORCA keyword line for --sp-refine (default: {DEFAULT_SP_KEYWORDS!r})",
    )


def _atom_list(text: str) -> tuple[int, ...]:
    """Parse ``1,5,8`` into 1-based atom numbers.

    Raises:
        ValidationError: a field is not a positive integer.  Ruling R29 makes
            every validation failure a ValidationError; :func:`_atom_list_arg`
            is what adapts it to argparse's own error channel when this is
            used as a ``type=`` callback, the same way :func:`_dihedral_arg`
            adapts :func:`parse_dihedral`.
    """
    out = []
    for field in text.split(","):
        field = field.strip()
        try:
            value = int(field)
        except ValueError:
            raise ValidationError(f"--charge-atoms: {field!r} is not an integer") from None
        if value < 1:
            raise ValidationError(
                f"--charge-atoms: {value} is not a valid atom number; indices are 1-based, "
                "matching the topology's [ atoms ] numbering"
            )
        out.append(value)
    if not out:
        raise ValidationError("--charge-atoms: no atoms given")
    return tuple(out)


def _atom_list_arg(text: str) -> tuple[int, ...]:
    """argparse ``type=`` adapter for :func:`_atom_list`.

    Mirrors :func:`_dihedral_arg`: argparse only recognises ``ValueError``,
    ``TypeError`` and ``ArgumentTypeError`` as "this value is bad" from a
    ``type=`` callback; anything else -- including ``_atom_list``'s own
    ``ValidationError`` -- propagates as an unhandled traceback instead of
    argparse's clean usage error, because ``parse_args()`` runs before
    ``main``'s ``try/except FFRefineError``. Translating here keeps
    ``_atom_list`` raising the project's own ``ValidationError`` (ruling R29)
    while still giving the user argparse's clean ``invalid value`` message
    and exit code 2.
    """
    try:
        return _atom_list(text)
    except ValidationError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _add_charge_flags(parser: argparse.ArgumentParser) -> None:
    """Add the charge-refinement flags (also needed by ``run``, which chains them)."""
    parser.add_argument(
        "--charge-scheme",
        choices=("chelpg", "resp"),
        default="chelpg",
        help="ESP charge scheme (default: chelpg)",
    )
    parser.add_argument(
        "--charge-geom",
        choices=("input", "xtb"),
        default="xtb",
        help="geometry for the ESP job (default: xtb, an xtb --opt tight optimisation)",
    )
    parser.add_argument(
        "--charge-mode",
        choices=("selected", "all"),
        default="selected",
        help="replace only the named atoms, or every charge (default: selected)",
    )
    parser.add_argument(
        "--charge-atoms",
        action="append",
        type=_atom_list_arg,
        default=None,
        metavar="i,j,k",
        help="1-based atoms whose charges to replace; repeatable",
    )
    parser.add_argument(
        "--orca-esp-keywords",
        default=ESP_DEFAULT_KEYWORDS,
        help=f"ORCA level of theory for the ESP job (default: {ESP_DEFAULT_KEYWORDS!r})",
    )
    parser.add_argument(
        "--esp-density",
        type=float,
        default=1.0,
        help="MK grid points per square Angstrom, --charge-scheme resp only (default: 1.0)",
    )


def _add_fit_flags(parser: argparse.ArgumentParser) -> None:
    """Add the fit-stage flags (also needed by ``run``, which chains the fit)."""
    parser.add_argument(
        "--multiplicities", type=_parse_mults, help="e.g. 1,2,3,4,6 (default: 1,2,3,4,6)"
    )
    parser.add_argument(
        "--emax",
        type=float,
        help=f"discard points this far above the QM minimum, kJ/mol (default: {DEFAULT_EMAX:g})",
    )
    parser.add_argument(
        "--boltzmann",
        nargs="?",
        type=float,
        const=DEFAULT_TEMPERATURE,
        default=None,
        help=f"Boltzmann weighting at the given temperature (default {DEFAULT_TEMPERATURE:g} K)",
    )
    parser.add_argument(
        "--free-phase",
        action="store_true",
        help="fit arbitrary phases rather than 0/180 only (spec section 4.7)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser (spec section 3)."""
    parser = argparse.ArgumentParser(
        prog="ffrefine",
        description="Refine small-molecule force field torsions and charges.",
    )
    parser.add_argument("--version", action="version", version=f"ffrefine {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="QM relaxed torsion scan (xtb)")
    _add_common(scan)
    _add_torsion(scan)
    _add_qm_flags(scan)
    _add_scan_flags(scan)

    mmscan = sub.add_parser("mmscan", help="GROMACS single points with and without the torsion")
    _add_common(mmscan)
    _add_torsion(mmscan)

    fit = sub.add_parser("fit", help="fit the dihedral series and write new parameters")
    _add_common(fit)
    _add_torsion(fit)
    _add_fit_flags(fit)

    charges = sub.add_parser("charges", help="ESP/RESP charges via ORCA")
    _add_common(charges)
    _add_qm_flags(charges)
    _add_charge_flags(charges)

    validate = sub.add_parser("validate", help="rebuild with fitted parameters and compare to QM")
    _add_common(validate)
    _add_torsion(validate)

    run = sub.add_parser("run", help="run every implemented stage in order")
    _add_common(run)
    _add_torsion(run)
    _add_qm_flags(run)
    _add_scan_flags(run)
    _add_fit_flags(run)
    _add_charge_flags(run)

    return parser


# ---------------------------------------------------------------------------
# configuration, logging and shared input handling
# ---------------------------------------------------------------------------


def _coerce_config_value(key: str, value: object) -> object:
    """Give a config-file value the type the corresponding CLI flag produces.

    A value that arrived on the command line has already been through
    argparse's ``type=`` callable; one that arrived from JSON has not, so
    ``{"multiplicities": "1,2,3"}`` would otherwise reach the fitter as a
    string and be iterated character by character.  Anything that cannot be
    converted is rejected here, by name, rather than failing obscurely later.

    Args:
        key: the option name, one of :data:`_CONFIG_KEYS`.
        value: the raw value from the config file.

    Returns:
        The value in the type the rest of the CLI expects.

    Raises:
        ValidationError: the value cannot be converted.
    """
    if value is None:
        return None
    try:
        if key == "multiplicities":
            if isinstance(value, str):
                return [int(p.strip()) for p in value.split(",") if p.strip()]
            values = [int(v) for v in value]  # type: ignore[union-attr]
            if not values:
                raise ValueError("empty multiplicity list")
            return values
        if key == "nthreads":
            return int(value)  # type: ignore[arg-type]
        if key in ("emax", "boltzmann", "step"):
            return float(value)  # type: ignore[arg-type]
        return str(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"config value for {key!r} is unusable ({value!r}): {exc}") from exc


def _apply_config(args: argparse.Namespace) -> None:
    """Merge ``--config`` into ``args`` and fill in defaults, in place.

    Precedence, highest first: an explicit CLI flag, the config file, then
    :data:`_DEFAULTS`.  Only :data:`_CONFIG_KEYS` may come from the file; a
    ``"binaries"`` object is also honoured, because
    :meth:`ffrefine.config.Binaries.require`'s error message tells users that
    form works.  Unrecognised keys are warned about rather than rejected, so
    a config shared with a later milestone still loads.

    Raises:
        FFRefineError: the config file is missing or is not valid JSON
            (delegated to :func:`ffrefine.config.load_config`).
    """
    cli = {key: getattr(args, key, None) for key in _CONFIG_KEYS}
    merged = load_config(getattr(args, "config", None), cli)

    binaries = merged.pop("binaries", None) or {}
    if not isinstance(binaries, dict):
        raise ValidationError('config "binaries" must be an object mapping name to path')
    for name in ("gmx", "xtb", "orca"):
        attr = f"{name}_bin"
        if getattr(args, attr, None) is None and binaries.get(name):
            setattr(args, attr, binaries[name])

    for key, value in merged.items():
        if key in _CONFIG_KEYS:
            setattr(args, key, _coerce_config_value(key, value))
        elif key not in {"command", "structure", "topology"}:
            _warn(f"ignoring unrecognised config key {key!r}")

    for key, fallback in _DEFAULTS.items():
        if hasattr(args, key) and getattr(args, key) is None:
            setattr(args, key, fallback)


def _reset_logging() -> None:
    """Detach any handlers from the ``ffrefine`` logger.

    ``--dry-run`` must not create the output directory, so it never calls
    :func:`ffrefine.stages.setup_logging`.  Handlers left behind by an earlier
    command in the same process would otherwise send a dry run's warnings into
    the *previous* run's log file, so they are cleared first and
    :func:`_warn` falls back to stderr.
    """
    for handler in list(log.handlers):
        log.removeHandler(handler)
        handler.close()


_WARNING_SINKS: list[list[str]] = []
"""Lists currently collecting every :func:`_warn` message; see :func:`_collect_warnings`."""


@contextmanager
def _collect_warnings() -> Iterator[list[str]]:
    """Collect the text of every :func:`_warn` issued inside the block.

    For stages that must keep their warnings in their own output file, not
    only in ``run.log`` and on stderr -- ``charges.json``'s ``warnings`` list
    (design section 7, ruling R25), which milestone 6b's report reads.
    Collecting at :func:`_warn` rather than at the call sites means a warning
    added later is kept without anyone remembering to keep it.
    """
    sink: list[str] = []
    _WARNING_SINKS.append(sink)
    try:
        yield sink
    finally:
        _WARNING_SINKS.remove(sink)


def _warn(message: str) -> None:
    """Emit a user-facing warning exactly once.

    Once logging is configured, ``setup_logging``'s own console handler puts
    the message on stderr *and* into ``run.log``; printing as well would
    double it.  Before that (dry runs, argument handling) there is no logger
    to carry it, so it goes straight to stderr.  Any collector opened with
    :func:`_collect_warnings` also receives the text.
    """
    for sink in _WARNING_SINKS:
        sink.append(message)
    if log.handlers:
        log.warning(message)
    else:
        print(f"warning: {message}", file=sys.stderr)


def _ff_includes(args: argparse.Namespace) -> list[str]:
    """Resolve ``--ff-include`` paths for the generated wrapper ``.top``.

    The wrapper lives in the stage directory, not the user's cwd, so a plain
    relative path the user typed would not resolve there.  A value naming a
    file we can actually see is therefore absolutised; anything else is passed
    through verbatim, because the common case
    (``charmm36.ff/forcefield.itp``) is resolved by grompp against GMXLIB and
    must not be rewritten.
    """
    out: list[str] = []
    for value in getattr(args, "ff_include", None) or []:
        candidate = Path(value)
        out.append(str(candidate.resolve()) if candidate.exists() else str(value))
    return out


def _binaries(args: argparse.Namespace) -> Binaries:
    """Resolve external binaries, honouring the CLI/config overrides."""
    return discover_binaries(
        {
            "gmx": getattr(args, "gmx_bin", None),
            "xtb": getattr(args, "xtb_bin", None),
            "orca": getattr(args, "orca_bin", None),
        }
    )


def _load_structure_and_topology(args: argparse.Namespace) -> tuple[Structure, Topology]:
    """Read and cross-check the structure and topology, with no torsion involved.

    Split out of :func:`_load_inputs` so ``charges``, which refines charges and
    touches no dihedral, does not have to supply a ``--dihedral`` it never uses.

    Raises:
        ParseError: a structure or topology file could not be parsed.
    """
    structure = read_structure(args.structure)
    topology = read_topology(args.topology)
    for warning in validate_structure_topology(
        structure, topology, strict_names=not args.no_strict_names
    ):
        _warn(warning)
    # XYZ carries no atom names, so the reader's placeholders would reach
    # grompp and never match the topology. Order is already checked above, so
    # the topology's own naming is the right one to use from here on.
    return adopt_topology_naming(structure, topology), topology


def _resolve_quad(args: argparse.Namespace, topology: Topology) -> Quad:
    """Resolve the single target torsion from ``--dihedral``.

    Raises:
        ValidationError: no ``--dihedral`` was given, more than one was given,
            or the quadruplet is unusable with this topology.
    """
    quads = getattr(args, "dihedral", None) or []
    if not quads:
        raise ValidationError(
            "no torsion selected. Pass --dihedral i,j,k,l using 1-based atom "
            "indices. Automatic selection from a CGenFF .str file arrives in "
            "milestone 6b and is not available in this build."
        )
    if len(quads) > 1:
        raise ValidationError(
            f"{len(quads)} torsions requested, but this build fits one torsion per run: "
            "spec section 6's output layout holds a single QM scan and a single fit per "
            "outdir. Re-run with one --dihedral and its own --outdir for each torsion."
        )
    quad = quads[0]
    validate_quad(topology, quad)
    return quad


def _load_inputs(args: argparse.Namespace) -> tuple[Structure, Topology, Quad]:
    """Read, cross-check and return the structure, topology and target quad."""
    structure, topology = _load_structure_and_topology(args)
    return structure, topology, _resolve_quad(args, topology)


def _write_input_stage(
    args: argparse.Namespace, structure: Structure, topology: Topology, quad: Quad | None
) -> None:
    """Populate ``00_input/`` (spec section 6).

    Copies -- never moves or rewrites -- the two input files and dumps the
    parsed model, including the 1-4 conventions that make the eventual fit
    convention-bound (spec section 9.3).  ``quad`` is ``None`` for stages that
    refine something other than a torsion; the torsion-set keys are then
    omitted rather than recording a target that was never chosen.

    The topology copied and recorded is always the one the user passed with
    ``-p`` (:func:`_input_topology`), never a stage output: once ``run`` has
    repointed ``args.topology`` at ``04_charges/molecule_charged.itp``, every
    later stage used to copy that file here and name it in ``model.json``
    (final review I5, ruling R24).
    """
    target = stage_dir(args.outdir, "input")
    input_topology = Path(_input_topology(args))
    for source in (Path(args.structure), input_topology):
        destination = target / source.name
        if destination.resolve() != source.resolve():
            shutil.copyfile(source, destination)

    defaults = topology.defaults
    model = {
        "structure": str(Path(args.structure).resolve()),
        "topology": str(input_topology.resolve()),
        "molecule": topology.name,
        "natoms": topology.natoms,
        "nbonds": len(topology.bonds),
        "npairs": len(topology.pairs),
        "nangles": len(topology.angles),
        "npropers": len(topology.propers),
        "nimpropers": len(topology.impropers),
        "fudgeQQ": None if defaults is None else defaults.fudge_qq,
        "fudgeLJ": None if defaults is None else defaults.fudge_lj,
        "gen_pairs": None if defaults is None else defaults.gen_pairs,
    }
    if quad is not None:
        torsion_set = collect_torsion_set(topology, quad)
        model["target_quad_1_based"] = list(quad)
        model["central_bond_1_based"] = [quad[1], quad[2]]
        model["torsion_set_size"] = len(torsion_set)
        model["torsion_set_1_based"] = [[d.ai, d.aj, d.ak, d.al] for d in torsion_set]
    if defaults is None:
        if has_includes(topology.raw_lines) or _ff_includes(args):
            # grompp reads the includes; mmscan asks it (milestone 6a design, section 4).
            log.info(
                "%s: [ defaults ] is not in this file itself; mmscan reads it from grompp's "
                "preprocessed topology and records it in 02_mm_scan/energies.json",
                input_topology,
            )
        else:
            _warn(
                f"{input_topology} has no [ defaults ] directive, so fudgeQQ/fudgeLJ could not "
                "be read. The fitted torsion is only valid with the 1-4 conventions it was "
                "fitted against (spec section 9.3); record them yourself."
            )
    (target / "model.json").write_text(json.dumps(model, indent=2) + "\n")


def _input_topology(args: argparse.Namespace) -> str:
    """The topology the user passed with ``-p``.

    Usually ``args.topology`` itself.  ``cmd_run`` repoints ``args.topology``
    at the charge-refined file after the charges stage, and records the
    user's own path as ``args.input_topology`` before doing so.
    """
    return getattr(args, "input_topology", None) or args.topology


def _prepare(
    args: argparse.Namespace, *, need_quad: bool = True
) -> tuple[Structure, Topology, Quad | None]:
    """Configure logging, read the inputs and write ``00_input/``.

    ``need_quad=False`` is for stages that refine something other than a
    torsion; they get ``None`` and ``00_input/model.json`` omits the torsion
    set rather than recording one that was never chosen.
    """
    setup_logging(args.outdir, args.verbose)
    structure, topology = _load_structure_and_topology(args)
    quad = _resolve_quad(args, topology) if need_quad else None
    _write_input_stage(args, structure, topology, quad)
    return structure, topology, quad


def _read_json(path: Path, what: str, remedy: str) -> dict:
    """Read a stage JSON payload, with a message naming the missing stage."""
    if not path.exists():
        raise ValidationError(f"no {what} found at {path}. {remedy}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"could not read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValidationError(f"{path}: expected a JSON object, got {type(payload).__name__}")
    return payload


def _require_array(payload: dict, key: str, path: Path) -> np.ndarray:
    """Pull a required numeric array out of a stage payload."""
    if key not in payload:
        raise ValidationError(f"{path} is missing the {key!r} array")
    try:
        return np.asarray(payload[key], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{path}: {key!r} is not a numeric array: {exc}") from exc


def _scan_path(args: argparse.Namespace) -> Path:
    return Path(args.outdir) / STAGE_DIRS["qm_scan"] / "scan.json"


def _energies_path(args: argparse.Namespace) -> Path:
    return Path(args.outdir) / STAGE_DIRS["mm_scan"] / "energies.json"


def _charges_path(args: argparse.Namespace) -> Path:
    return Path(args.outdir) / STAGE_DIRS["charges"] / "charges.json"


def _orca_version(binaries: Binaries) -> str:
    """Best-effort ORCA version for the provenance block; never fatal."""
    try:
        proc = subprocess.run(
            [binaries.require("orca"), "--version"],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, **orca_env(binaries)},
        )
    except (OSError, subprocess.SubprocessError, FFRefineError):
        return "unknown"
    for line in (proc.stdout or "").splitlines():
        if "version" in line.lower():
            return line.strip()
    return "unknown"


def _partial_group_warnings(groups: list[list[int]], selected: list[int] | None) -> list[str]:
    """Warnings for equivalence groups that ``--charge-atoms`` selects only part of.

    Spec 5.3 gives equivalent hydrogens one charge.  ``selected`` mode
    replaces only the named atoms, so naming one hydrogen of a methyl leaves
    its siblings at their topology charges and the group unequal -- which
    used to happen without a word (final review, M3).

    Args:
        groups: 1-based groups from
            :func:`~ffrefine.charges.equivalence.equivalent_hydrogen_groups`.
        selected: 1-based selected atoms, or ``None`` for ``--charge-mode all``
            (which replaces every atom and so splits no group).

    Returns:
        One message per split group, naming the group and its unselected members.
    """
    if selected is None:
        return []
    chosen = set(selected)
    out = []
    for group in groups:
        unselected = [nr for nr in group if nr not in chosen]
        if unselected and len(unselected) < len(group):
            out.append(
                f"--charge-atoms selects only part of the equivalent-hydrogen group {group} "
                f"(bonded to the same atom, same atom type): {unselected} are not selected and "
                "keep their topology charges, so spec 5.3's equal-charge rule no longer holds "
                f"for this group. Select all of {group} to keep it equal."
            )
    return out


def _looks_like_cgenff(topology: Topology) -> bool:
    """True when the atom types look like CGenFF (CG331, HGA3, OG311).

    Decides only whether spec 5.3's warning applies.  A false positive prints a
    warning that is merely irrelevant; a false negative withholds one that
    matters, so the test is deliberately loose.
    """
    return any(re.match(r"^[A-Z]{1,2}G\d", atom.atype or "") for atom in topology.atoms)


_STAGE_OUTPUTS: dict[str, tuple[str, ...]] = {
    "qm_scan": ("scan.json",),
    "mm_scan": ("energies.json",),
    "fit": ("fit.json", "dihedral.itp", "dihedral.prm"),
    "charges": ("charges.json", "molecule_charged.itp"),
    "final": ("validation.json", "molecule_refined.itp", "after.itp"),
}
"""Files each stage must have left behind for ``--resume`` to skip it.

A manifest records that a stage *ran*, not that its results survived: deleting
``03_fit/fit.json`` and re-running with ``--resume`` printed "fit: skipping,
... is current" and produced no file. Named here, one list per stage, rather
than inline at each call site so a new output cannot be forgotten in one place
and remembered in another."""


def _stage_outputs(args: argparse.Namespace, stage: str) -> list[str | Path]:
    """Paths ``stage`` must have produced, for the ``--resume`` liveness check.

    Constructed with plain path joins, never :func:`stage_dir`, so checking
    does not create the directory it is checking.
    """
    base = Path(args.outdir) / STAGE_DIRS[stage]
    return [base / name for name in _STAGE_OUTPUTS[stage]]


_SCAN_REMEDY = (
    "Run 'ffrefine scan' first to produce it, or supply the file yourself: a JSON object "
    "with 'energy_kj' (QM energies in kJ/mol, one per frame) and 'frames_nm' (an "
    "(nframes, natoms, 3) array of coordinates in nanometres, in topology atom order)."
)


def _check_fit_applies_here(
    quad: Quad, fit_payload: dict, fit_path: Path, frames: np.ndarray, scan_path: Path
) -> None:
    """Refuse to validate a fit that was not made for this torsion or scan.

    ``cmd_validate`` reads ``k``/``delta``/``mult`` out of ``fit.json`` and
    applies them to whatever ``--dihedral`` the invocation carries.  Before
    ``fit.json`` recorded a quad, a mistyped ``--dihedral`` wrote one
    torsion's fitted series onto a different torsion and reported an RMSE for
    it -- exit 0, no warning.

    Args:
        quad: the 1-based quadruplet this invocation is validating.
        fit_payload: parsed ``03_fit/fit.json``.
        fit_path: its path, for messages.
        frames: the geometries from ``scan.json``, shape (nframes, natoms, 3).
        scan_path: the scan's path, for messages.

    Raises:
        ValidationError: the fit belongs to a different torsion or a
            different set of geometries, or predates the provenance block.
    """
    provenance = fit_payload.get("provenance") or {}
    recorded_quad = provenance.get("quad_1_based")
    if recorded_quad is None:
        raise ValidationError(
            f"{fit_path} records no target torsion, so there is no way to tell whether its "
            "parameters belong to the torsion being validated. It predates this check; "
            "re-run 'ffrefine fit'."
        )
    if list(recorded_quad) != list(quad):
        shown = ",".join(str(i) for i in recorded_quad)
        wanted = ",".join(str(i) for i in quad)
        raise ValidationError(
            f"{fit_path} holds the fit for torsion {shown}, but this validation is for "
            f"torsion {wanted}. Applying one torsion's fitted series to another and "
            f"reporting its RMSE is meaningless. Re-run 'ffrefine fit --dihedral {wanted}', "
            f"or validate torsion {shown}."
        )
    recorded_digest = provenance.get("frames_sha256")
    if recorded_digest is None:
        raise ValidationError(
            f"{fit_path} records no 'frames_sha256', so it cannot be tied to this scan's "
            "geometries. It predates this check; re-run 'ffrefine fit'."
        )
    if recorded_digest != hash_array(frames):
        raise ValidationError(
            f"{fit_path} was fitted against different geometries than {scan_path} now "
            "holds, so validating it on these frames compares a fit to a profile it never "
            "saw. Re-run 'ffrefine mmscan' and 'ffrefine fit' before validating."
        )


def _check_stage_pairing(
    quad: Quad, scan: dict, scan_path: Path, energies: dict, energies_path: Path
) -> str:
    """Refuse an ``mmscan`` payload that does not belong to this fit.

    The two stage payloads used to be paired by array length alone, so three
    different mispairings all exited 0 with a confident wrong answer:

    * ``fit --dihedral 2,1,5,8`` after ``mmscan --dihedral 1,5,8,11`` in the
      same outdir fitted torsion A's angles and MM-zeroed energies and wrote
      the parameters onto torsion B.  ``energies.json`` recorded the quad it
      was computed for; nothing read it.
    * Regenerating ``scan.json``'s geometries without re-running ``mmscan``
      paired the new QM energies with stale ``phi_deg`` and a stale
      ``E_MM0``.

    Both are checked here, before any fitting.  The geometry check hashes
    ``frames_nm`` rather than the whole ``scan.json`` file, because that is
    precisely what ``mmscan``'s outputs depend on: replacing the QM
    ``energy_kj`` column while keeping the geometries is the documented
    workflow (and spec section 7.1's round trip) and stays legal, while
    touching a coordinate does not.

    Args:
        quad: the 1-based quadruplet this invocation is fitting.
        scan: parsed ``01_qm_scan/scan.json``.
        scan_path: its path, for messages.
        energies: parsed ``02_mm_scan/energies.json``.
        energies_path: its path, for messages.

    Returns:
        The hex digest of ``scan``'s ``frames_nm``, for recording in
        ``fit.json`` so ``validate`` can make the same check.

    Raises:
        ValidationError: the payloads disagree, or either lacks the field
            needed to tell.  A payload written by an older ffrefine is
            refused rather than trusted -- ``mmscan`` is cheap and a wrong
            pairing is not detectable any other way.
    """
    recorded = energies.get("quad")
    if recorded is None:
        raise ValidationError(
            f"{energies_path} records no 'quad', so there is no way to tell which torsion "
            "its energies belong to. It predates this check; re-run 'ffrefine mmscan'."
        )
    if list(recorded) != list(quad):
        shown = ",".join(str(i) for i in recorded)
        wanted = ",".join(str(i) for i in quad)
        raise ValidationError(
            f"{energies_path} holds the MM energies for torsion {shown}, but this fit is "
            f"for torsion {wanted}. Fitting one torsion's energies and writing the result "
            f"onto another produces plausible-looking, wrong parameters. Re-run "
            f"'ffrefine mmscan --dihedral {wanted}' in this outdir, or fit torsion "
            f"{shown}, or use a separate --outdir per torsion."
        )

    if "frames_nm" not in scan:
        raise ValidationError(
            f"{scan_path} has no 'frames_nm' array, so the fit cannot confirm that "
            f"{energies_path} was computed on these geometries. " + _SCAN_REMEDY
        )
    digest = hash_array(_require_array(scan, "frames_nm", scan_path))
    recorded_digest = energies.get("frames_sha256")
    if recorded_digest is None:
        raise ValidationError(
            f"{energies_path} records no 'frames_sha256', so the fit cannot confirm it was "
            "computed on this scan's geometries. It predates this check; re-run "
            "'ffrefine mmscan'."
        )
    if recorded_digest != digest:
        raise ValidationError(
            f"{energies_path} was computed on different geometries than {scan_path} now "
            "holds. Its 'phi_deg' and 'zeroed_potential_kj' therefore do not correspond to "
            "this scan's frames, and the fit target would pair the new QM energies with "
            "stale angles and a stale E_MM0. Re-run 'ffrefine mmscan' before fitting. "
            "(Replacing only the QM 'energy_kj' column is fine and does not trigger this; "
            "only the geometries are hashed.)"
        )
    return digest


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------


def _xtb_version(binaries: Binaries) -> str:
    """Best-effort xtb version string for the provenance block; never fatal.

    Provenance that failed the run it is describing would be worse than no
    provenance at all, so every failure here degrades to ``"unknown"``.
    """
    try:
        proc = subprocess.run(
            [binaries.require("xtb"), "--version"], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError, FFRefineError):
        return "unknown"
    for line in (proc.stdout or "").splitlines():
        if "version" in line.lower():
            return line.strip()
    return "unknown"


def cmd_scan(args: argparse.Namespace) -> int:
    """Run the xtb relaxed torsion scan and write ``01_qm_scan/scan.json``.

    This is the stage that closes the pipeline: before it existed, the README
    told users to hand-write ``scan.json``.

    Bidirectional scanning runs a second leg backwards from the first leg's
    final geometry and takes the lower energy at each grid point (spec 4.3).
    The two legs are paired by *measured angle*, never by array index -- see
    :func:`ffrefine.qm.scan.merge_bidirectional` for why that distinction
    decides whether a torsion barrier survives.
    """
    step = args.step if args.step is not None else 15.0
    if args.dry_run:
        structure, topology, quad = _load_inputs(args)
        start, end, npoints = scan_grid(step)
        legs = 2 if args.bidirectional else 1
        print(
            f"would scan torsion {quad} of {topology.name} ({structure.natoms} atoms) "
            f"over {npoints} points from {start} to {end} degrees, {legs} leg(s), "
            f"into {args.outdir}"
        )
        return 0

    structure, topology, quad = _prepare(args)
    start, end, npoints = scan_grid(step)
    charge = derive_charge(topology, args.charge)
    freeze = tuple(args.freeze_dihedral or ())

    settings = {
        "quad": list(quad),
        "step_deg": step,
        "charge": charge,
        "uhf": args.uhf,
        "gfn": args.gfn,
        "bidirectional": bool(args.bidirectional),
        "freeze": [list(f) for f in freeze],
        "sp_refine": bool(args.sp_refine),
        "sp_refine_keywords": args.orca_sp_keywords,
    }
    digest = hash_inputs([args.topology, args.structure], settings)
    if args.resume and stage_is_current(
        args.outdir, "qm_scan", digest, _stage_outputs(args, "qm_scan")
    ):
        print(f"scan: skipping, {_scan_path(args)} is current (--resume)")
        return 0

    binaries = _binaries(args)
    workdir = stage_dir(args.outdir, "qm_scan")

    forward_frames, forward_energy = run_scan(
        binaries,
        structure,
        quad,
        workdir / "forward",
        start_deg=start,
        end_deg=end,
        npoints=npoints,
        charge=charge,
        uhf=args.uhf,
        gfn=args.gfn,
        nthreads=args.nthreads,
        freeze=freeze,
        log_stem="xtb_forward",
    )
    forward = ScanLeg(
        forward_frames, forward_energy, measure_angles(forward_frames, quad), "forward"
    )

    merged = None
    if args.bidirectional:
        reverse_frames, reverse_energy = run_scan(
            binaries,
            structure,
            quad,
            workdir / "reverse",
            start_deg=end,
            end_deg=start,
            npoints=npoints,
            charge=charge,
            uhf=args.uhf,
            gfn=args.gfn,
            nthreads=args.nthreads,
            freeze=freeze,
            coords_nm=forward_frames[-1],
            log_stem="xtb_reverse",
        )
        reverse = ScanLeg(
            reverse_frames, reverse_energy, measure_angles(reverse_frames, quad), "reverse"
        )
        merged = merge_bidirectional(forward, reverse, tol_deg=step / 2.0)
        frames_nm, energy_kj, phi_deg = merged.frames_nm, merged.energy_kj, merged.phi_deg
        if merged.max_hysteresis_kj > args.hysteresis_tol:
            _warn(
                f"forward and reverse scans differ by up to "
                f"{merged.max_hysteresis_kj:.2f} kJ/mol (worst at "
                f"{merged.worst_angle_deg:.1f} degrees), above the "
                f"{args.hysteresis_tol:g} kJ/mol tolerance. The scan is not reversible "
                "here, so the profile depends on the direction it was walked; treat the "
                "barrier heights with suspicion."
            )
        if merged.unpaired_forward_deg or merged.unpaired_reverse_deg:
            _warn(
                f"{len(merged.unpaired_forward_deg)} forward and "
                f"{len(merged.unpaired_reverse_deg)} reverse points had no partner within "
                f"{step / 2.0:g} degrees, so those grid points come from one direction only."
            )
    else:
        frames_nm, energy_kj, phi_deg = forward.frames_nm, forward.energy_kj, forward.phi_deg

    sp_keywords = None
    if args.sp_refine:
        sp_keywords = args.orca_sp_keywords or DEFAULT_SP_KEYWORDS
        energy_kj = refine_energies(
            binaries,
            structure,
            frames_nm,
            workdir / "orca_sp",
            sp_keywords,
            charge=charge,
            mult=args.uhf + 1,
            nprocs=args.nthreads,
        )
        print(f"refined {len(energy_kj)} energies with ORCA ({sp_keywords})")

    hops = find_hops(frames_nm, heavy_atom_indices(topology), phi_deg)
    for hop in hops:
        _warn(
            f"possible conformational hop at {hop.phi_deg:.1f} degrees (frame {hop.index}): "
            f"heavy-atom RMSD {hop.rmsd_nm:.4f} nm against a {hop.threshold_nm:.4f} nm "
            "outlier threshold. A smooth cosine series fitted through a hop is meaningless."
        )
    hbonds = hbond_changes(frames_nm, topology, phi_deg)
    for change in hbonds:
        _warn(
            f"hydrogen bonding changed at {change.phi_deg:.1f} degrees (frame "
            f"{change.index}): formed {sorted(change.formed)}, broke {sorted(change.broken)}. "
            "That stabilisation is not the torsional potential; fitting through it writes "
            "an electrostatic interaction into the dihedral term."
        )

    write_multimodel_pdb(workdir / "scan.pdb", structure, frames_nm)
    payload = {
        "quad": list(quad),
        "nframes": int(frames_nm.shape[0]),
        "phi_deg": [float(v) for v in phi_deg],
        "energy_kj": [float(v) for v in energy_kj],
        "frames_nm": [[[float(c) for c in atom] for atom in frame] for frame in frames_nm],
        "frames_sha256": hash_array(frames_nm),
        "provenance": {
            "xtb_version": _xtb_version(binaries),
            "charge": charge,
            "uhf": int(args.uhf),
            "gfn": int(args.gfn),
            "step_deg": float(step),
            "npoints": int(npoints),
            "bidirectional": bool(args.bidirectional),
            "hysteresis_tol_kj": float(args.hysteresis_tol),
            "max_hysteresis_kj": float(merged.max_hysteresis_kj) if merged else None,
            "worst_hysteresis_angle_deg": float(merged.worst_angle_deg) if merged else None,
            "source_per_point": merged.source if merged else ["forward"] * len(energy_kj),
            "freeze": [list(f) for f in freeze],
            "sp_refine_keywords": sp_keywords,
            "hops": [{"frame": h.index, "phi_deg": h.phi_deg, "rmsd_nm": h.rmsd_nm} for h in hops],
            "hbond_changes": [
                {
                    "frame": c.index,
                    "phi_deg": c.phi_deg,
                    "formed": [list(b) for b in sorted(c.formed)],
                    "broken": [list(b) for b in sorted(c.broken)],
                }
                for c in hbonds
            ],
        },
    }
    _scan_path(args).write_text(json.dumps(payload, indent=2) + "\n", newline="\n")
    write_manifest(args.outdir, "qm_scan", digest, {"quad": list(quad), "npoints": npoints})

    relative = energy_kj - energy_kj.min()
    print(
        f"torsion {quad}: {npoints}-point relaxed scan complete, span "
        f"{relative.max():.2f} kJ/mol, written to {_scan_path(args)}"
    )
    return 0


@dataclass
class RespCharges:
    """What :func:`_resp_charges_on_mk_grid` produced.

    Attributes:
        charges: fitted charges in e, shape (natoms,), 0-based topology order.
        grid: the MK grid's provenance for ``charges.json``.
        restraint: the restraint constants actually used, and the iteration
            counts of both stages.
    """

    charges: np.ndarray
    grid: dict[str, object]
    restraint: dict[str, object]


def _resp_charges_on_mk_grid(
    binaries: Binaries,
    base: Path,
    topology: Topology,
    coords_nm: np.ndarray,
    workdir: Path,
    target_total: int,
    esp_density: float,
    *,
    a_stage1: float = RESP_A_STAGE1,
    a_stage2: float = RESP_A_STAGE2,
) -> RespCharges:
    """Fit RESP charges on our own MK grid against a finished ESP job.

    The whole RESP wiring, in one place: build the MK grid, evaluate the QM
    potential on it with ``orca_vpot``, convert both coordinate sets to bohr
    for :func:`~ffrefine.charges.esp.inverse_distances`, turn the 1-based
    equivalence groups into 0-based ones and derive the stage-2 free set, then
    run :func:`~ffrefine.charges.resp.resp_fit`.

    It is a function rather than inline code in :func:`cmd_charges` so the
    test that proves the wiring right -- with the restraint off, it must
    reproduce ORCA's own CHELPG charges -- calls the code that ships instead
    of a copy of it.  A copy is how a dropped ``nm_to_bohr`` once shipped
    RESP charges about eight times too small past every test (ruling R23).

    Args:
        binaries: resolved binaries; ``orca_vpot`` is needed.
        base: the ESP job's base path, from :func:`run_esp_job`.
        topology: supplies elements, bonds and the equivalence groups.
        coords_nm: the geometry the ESP job ran on, shape (natoms, 3), nm.
        workdir: the stage directory ``orca_vpot`` runs in.
        target_total: the molecular charge, enforced exactly.
        esp_density: MK grid points per square Angstrom.
        a_stage1, a_stage2: restraint strengths.  Only a test that needs the
            unrestrained fit should pass anything but the defaults.

    Returns:
        The charges, with the grid and restraint provenance.

    Raises:
        ExternalToolError: ``orca_vpot`` failed.
        ValidationError: the grid or the fit was refused.
    """
    elements = [element_of(topology, atom.nr) for atom in topology.atoms]
    grid_nm = mk_grid(coords_nm, elements, density_per_angstrom2=esp_density)
    v_grid = run_vpot(binaries, base, grid_nm, workdir)
    inv_r = inverse_distances(nm_to_bohr(coords_nm), nm_to_bohr(grid_nm))
    groups = equivalent_hydrogen_groups(topology)
    zero_based = [[nr - 1 for nr in group] for group in groups]
    parents = {
        n - 1
        for group in groups
        for n in topology.bonded_neighbours(group[0])
        if element_of(topology, n) != "H"
    }
    stage2_free = sorted(parents.union(i for g in zero_based for i in g))
    result = resp_fit(
        inv_r,
        v_grid,
        restrain=np.array([e != "H" for e in elements]),
        groups=zero_based,
        stage2_free=stage2_free,
        total_charge=float(target_total),
        a_stage1=a_stage1,
        a_stage2=a_stage2,
    )
    return RespCharges(
        charges=result.charges,
        grid={
            "npoints": int(len(grid_nm)),
            "density_per_angstrom2": float(esp_density),
            "shell_scales": list(MK_SHELL_SCALES),
            "radii_angstrom": {e: VDW_RADII_ANGSTROM[e] for e in sorted(set(elements))},
        },
        restraint={
            "a_stage1": float(a_stage1),
            "a_stage2": float(a_stage2),
            "b": RESP_B,
            "iterations": list(result.iterations),
        },
    )


def _rrms_on_orca_grid(
    vpot_path: Path, coords_nm: np.ndarray, charge_sets: dict[str, np.ndarray]
) -> tuple[dict[str, float], int]:
    """RRMS of each charge set against the QM potential on ORCA's CHELPG grid.

    Every set is scored on the same grid -- the ``<base>.vpot`` the ESP job
    wrote -- so the numbers are comparable with each other and across the two
    charge schemes.  The grid comes back from :func:`read_chelpg_vpot` in nm
    and the atoms are in nm; both go to bohr here, because the potential is
    in Hartree/e and ``1/r`` must be in 1/bohr to match it.

    Args:
        vpot_path: the ESP job's ``<base>.vpot``.
        coords_nm: the geometry the ESP job ran on, shape (natoms, 3), nm.
        charge_sets: name -> charges in e, 0-based topology order.

    Returns:
        ``(rrms_by_name, npoints)``.
    """
    grid_nm, v_ref = read_chelpg_vpot(vpot_path)
    inv_r = inverse_distances(nm_to_bohr(coords_nm), nm_to_bohr(grid_nm))
    scores = {
        name: rrms(v_ref, potential_from_charges(inv_r, np.asarray(q, float)))
        for name, q in charge_sets.items()
    }
    return scores, int(len(v_ref))


def cmd_charges(args: argparse.Namespace) -> int:
    """Fit partial charges to an ORCA electrostatic potential (spec section 5).

    Charges are refined *before* the torsion stages, never after: the fitted
    torsion absorbs whatever the 1-4 electrostatics do not account for, and
    those depend on these charges.  Fitting the torsion first and then changing
    the charges leaves a topology whose torsion was fitted against charges it
    no longer carries.

    Every warning issued while the stage runs is also written to
    ``charges.json`` under ``"warnings"``.
    """
    with _collect_warnings() as emitted:
        return _charges_stage(args, emitted)


def _charges_stage(args: argparse.Namespace, emitted: list[str]) -> int:
    """The body of :func:`cmd_charges`; ``emitted`` collects its warnings."""
    selected_atoms = None
    if args.charge_mode == "selected":
        if not args.charge_atoms:
            raise ValidationError(
                "--charge-mode selected replaces only the atoms you name, and none were "
                "named. Pass --charge-atoms i,j,k (1-based), or --charge-mode all to "
                "replace every charge -- which on a CGenFF topology changes the derivation "
                "scheme, so read the warning it prints. Choosing atoms automatically from "
                "a CGenFF .str file arrives in milestone 6b."
            )
        selected_atoms = sorted({nr for group in args.charge_atoms for nr in group})
    elif args.charge_atoms:
        raise ValidationError(
            "--charge-atoms names the atoms to replace under --charge-mode selected, but "
            "--charge-mode all replaces every charge, so the list would be ignored. Drop "
            "--charge-atoms to replace every charge, or drop --charge-mode all to replace only "
            "the atoms you named."
        )

    if args.dry_run:
        structure, topology = _load_structure_and_topology(args)
        # The same refusals the real run makes, so a dry run never promises a
        # run that would be refused.
        check_selection(topology, selected_atoms, derive_charge(topology, args.charge))
        for warning in _partial_group_warnings(
            equivalent_hydrogen_groups(topology), selected_atoms
        ):
            _warn(warning)
        what = "every charge" if selected_atoms is None else f"atoms {selected_atoms}"
        print(
            f"would fit {args.charge_scheme} charges for {what} of {topology.name} "
            f"({structure.natoms} atoms), geometry from {args.charge_geom}, "
            f"into {args.outdir}"
        )
        return 0

    structure, topology, _ = _prepare(args, need_quad=False)
    target_total = derive_charge(topology, args.charge)
    # Needs only the topology, so it runs before any xtb or ORCA job rather
    # than after them inside apply_charges (which checks again regardless).
    check_selection(topology, selected_atoms, target_total)
    groups = equivalent_hydrogen_groups(topology)
    for warning in _partial_group_warnings(groups, selected_atoms):
        _warn(warning)
    settings = {
        "scheme": args.charge_scheme,
        "geom": args.charge_geom,
        "mode": args.charge_mode,
        "atoms": selected_atoms,
        "keywords": args.orca_esp_keywords,
        "esp_density": float(args.esp_density),
        "charge": target_total,
        "uhf": int(args.uhf),
    }
    digest = hash_inputs([args.topology, args.structure], settings)
    if args.resume and stage_is_current(
        args.outdir, "charges", digest, _stage_outputs(args, "charges")
    ):
        print(f"charges: skipping, {_charges_path(args)} is current (--resume)")
        return 0

    binaries = _binaries(args)
    workdir = stage_dir(args.outdir, "charges")

    coords_nm = structure.coords
    if args.charge_geom == "xtb":
        coords_nm = optimize_geometry(
            binaries,
            structure,
            coords_nm,
            workdir / "opt",
            charge=target_total,
            uhf=args.uhf,
            nthreads=args.nthreads,
        )

    base = run_esp_job(
        binaries,
        structure,
        coords_nm,
        workdir,
        args.orca_esp_keywords,
        charge=target_total,
        mult=args.uhf + 1,
        nprocs=args.nthreads,
        chelpg=True,
    )
    grid_provenance: dict[str, object] | None = None
    restraint_provenance: dict[str, object] | None = None

    if args.charge_scheme == "chelpg":
        raw = parse_chelpg_charges(Path(str(base) + ".chelpg.xyz"))
        raw = average_within_groups(raw, groups)
    else:
        fitted = _resp_charges_on_mk_grid(
            binaries, base, topology, coords_nm, workdir, target_total, args.esp_density
        )
        raw = fitted.charges
        grid_provenance = fitted.grid
        restraint_provenance = fitted.restraint

    if selected_atoms is None and _looks_like_cgenff(topology):
        _warn(
            "--charge-mode all replaces every charge on what looks like a CGenFF topology. "
            "CGenFF charges are fitted to water interaction energies and dipoles, not to "
            "the gas-phase ESP, so replacing them wholesale is a change of derivation "
            "scheme and can degrade condensed-phase behaviour. --charge-mode selected with "
            "--charge-atoms is the safer choice."
        )

    # One precision for the arithmetic and the file: never coarser than the
    # input's own, so untouched atoms keep their exact values (ruling R22).
    decimals = charge_decimals(topology)
    assignments = apply_charges(topology, raw, selected_atoms, target_total, decimals=decimals)
    for a in assignments:
        if a.source == "redistributed":
            log.info(
                f"charges: atom %d (%s) %+.{decimals}f -> %+.{decimals}f, absorbing the "
                "redistribution residual",
                a.nr,
                a.name,
                a.before,
                a.after,
            )
    charged_itp = workdir / "molecule_charged.itp"
    write_charged_topology(args.topology, assignments, charged_itp, decimals=decimals)

    # Ruling R20: in selected mode the written topology carries only a few
    # fitted charges, so the fitted set's RRMS describes charges that were
    # never written. All three sets are scored on the same grid; the one the
    # user is shown and warned about is the one in the topology.
    scores, npoints = _rrms_on_orca_grid(
        Path(str(base) + ".vpot"),
        coords_nm,
        {
            "fitted": raw,
            "written": np.array([a.after for a in assignments]),
            "before": np.array([a.before for a in assignments]),
        },
    )
    if scores["written"] > 0.5:
        _warn(
            f"the charges written to {charged_itp.name} reproduce the QM electrostatic "
            f"potential poorly: RRMS {scores['written']:.3f} (the fully fitted set scores "
            f"{scores['fitted']:.3f}, the input topology's charges {scores['before']:.3f}). "
            "If the fitted set scores poorly too, point charges may simply be a poor model "
            "for this molecule; if only the written set does, the charges kept from the "
            "topology do not fit this potential, which --charge-mode selected cannot "
            "change. Treat the result with suspicion rather than assuming a bug."
        )

    payload = {
        "molecule": topology.name,
        "natoms": topology.natoms,
        "total_before": float(sum(a.before for a in assignments)),
        "total_after": float(sum(a.after for a in assignments)),
        "target_total": int(target_total),
        "rrms_written": float(scores["written"]),
        "rrms_fitted": float(scores["fitted"]),
        "rrms_before": float(scores["before"]),
        "warnings": list(emitted),
        "atoms": [
            {
                "nr": a.nr,
                "name": a.name,
                "element": element_of(topology, a.nr),
                "charge_before": a.before,
                "charge_after": a.after,
                "source": a.source,
            }
            for a in assignments
        ],
        "provenance": {
            "scheme": args.charge_scheme,
            "geometry": args.charge_geom,
            "mode": args.charge_mode,
            "selected_atoms": selected_atoms,
            "orca_esp_keywords": args.orca_esp_keywords,
            "orca_version": _orca_version(binaries),
            "charge": int(target_total),
            "uhf": int(args.uhf),
            "equivalence_groups": groups,
            "charge_decimals": int(decimals),
            "npoints": npoints,
            "grid": grid_provenance,
            "restraint": restraint_provenance,
        },
    }
    _charges_path(args).write_text(json.dumps(payload, indent=2) + "\n", newline="\n")
    write_manifest(args.outdir, "charges", digest, {"scheme": args.charge_scheme})

    changed = sum(1 for a in assignments if a.source != "unchanged")
    print(
        f"{args.charge_scheme} charges: {changed} of {topology.natoms} atoms changed, "
        f"ESP RRMS {scores['written']:.4f} for the charges written (fully fitted set "
        f"{scores['fitted']:.4f}, input topology {scores['before']:.4f}), written to "
        f"{charged_itp}"
    )
    print(f"to use them, pass -p {charged_itp} to the remaining stages")
    return 0


def _torsion_record(term: ResolvedTerm) -> dict:
    """One ``energies.json`` ``torsion_parameters`` entry (milestone 6a design, section 4)."""
    d = term.dihedral
    return {
        "quad_1_based": [d.ai, d.aj, d.ak, d.al],
        "funct": d.funct,
        "phi0_deg": float(d.params[0]),
        "k_kj_per_mol": float(d.params[1]),
        "multiplicity": int(round(float(d.params[2]))),
        "source": term.source,
    }


def _defaults_record(defaults: Defaults | None) -> dict | None:
    """``[ defaults ]`` as grompp read it, for ``energies.json``."""
    if defaults is None:
        return None
    return {
        "fudgeQQ": defaults.fudge_qq,
        "fudgeLJ": defaults.fudge_lj,
        "gen_pairs": defaults.gen_pairs,
        "source": "grompp -pp",
    }


def cmd_mmscan(args: argparse.Namespace) -> int:
    """Compute MM single points with and without the target torsion.

    Runs the mandatory spec 4.5 self-consistency check on the same two energy
    profiles it reports, and writes ``02_mm_scan/energies.json``.

    Geometries come from ``01_qm_scan/scan.json`` when it exists.  Without it
    -- the normal situation until milestone 4 lands -- only the input
    structure's single frame is available; the self-check still runs on that
    one frame and the user is told a profile needs a QM scan.
    """
    if args.dry_run:
        structure, topology, quad = _load_inputs(args)
        print(
            f"would run mmscan for torsion {quad} of {topology.name} "
            f"({structure.natoms} atoms) into {args.outdir}"
        )
        return 0

    structure, topology, quad = _prepare(args)
    scan_path = _scan_path(args)
    inputs: list[str | Path] = [args.topology, args.structure]

    if scan_path.exists():
        scan = _read_json(scan_path, "QM scan", _SCAN_REMEDY)
        if "frames_nm" not in scan:
            raise ValidationError(
                f"{scan_path} has no 'frames_nm' array, so there are no geometries to "
                "compute MM energies on. " + _SCAN_REMEDY
            )
        frames = _require_array(scan, "frames_nm", scan_path)
        if frames.ndim != 3 or frames.shape[1:] != (structure.natoms, 3):
            raise ValidationError(
                f"{scan_path}: 'frames_nm' has shape {frames.shape}, expected "
                f"(nframes, {structure.natoms}, 3) to match the structure"
            )
        inputs.append(scan_path)
    else:
        frames = structure.coords[None, :, :]
        _warn(
            f"no QM scan at {scan_path}, so the MM energies cover only the single input "
            "geometry. The spec 4.5 self-consistency check still runs, but fitting needs "
            "a torsion profile: " + _SCAN_REMEDY
        )

    ff_includes = _ff_includes(args)
    binaries = _binaries(args)
    workdir = stage_dir(args.outdir, "mm_scan")
    # Resolved before the --resume check: the key must cover every file grompp
    # reads, and only grompp knows which files those are (milestone 6a design,
    # section 4). About a second, even when the stage is then skipped.
    resolved = resolve_torsion_set(binaries, topology, quad, structure, workdir, ff_includes)
    settings = {
        "quad": list(quad),
        "nframes": int(frames.shape[0]),
        "ff_includes": ff_includes,
        "preprocessed_topology_sha256": resolved.pp_sha256,
    }
    digest = hash_inputs(inputs, settings)
    if args.resume and stage_is_current(
        args.outdir, "mm_scan", digest, _stage_outputs(args, "mm_scan")
    ):
        print(f"mmscan: skipping, {_energies_path(args)} is current (--resume)")
        return 0

    result = check_torsion_removal(
        binaries,
        topology,
        quad,
        structure,
        frames,
        workdir,
        args.nthreads,
        strict_names=not args.no_strict_names,
        ff_includes=ff_includes,
        resolved=resolved,
    )
    print(
        f"torsion {quad}: spec 4.5 self-consistency check passed over "
        f"{len(result.phi_deg)} frame(s), max deviation {result.max_abs_error:.3g} kJ/mol "
        f"(tolerance {result.tolerance:g})"
    )
    if resolved.defaults is None:
        _warn(
            f"grompp's preprocessed topology {workdir / 'full_pp.top'} has no [ defaults ] "
            "directive, so fudgeQQ/fudgeLJ could not be recorded. The fitted torsion is only "
            "valid with the 1-4 conventions it was fitted against (spec section 9.3); record "
            "them yourself."
        )

    payload = {
        "quad": list(quad),
        "nframes": int(frames.shape[0]),
        # Binds these energies to the geometries they were computed on, so
        # 'fit' and 'validate' can refuse a stale pairing instead of trusting
        # matching array lengths (see _check_stage_pairing).
        "frames_sha256": hash_array(frames),
        "frames_source": str(scan_path) if scan_path.exists() else str(args.structure),
        "phi_deg": [float(v) for v in result.phi_deg],
        "full_potential_kj": [float(v) for v in result.full_potential_kj],
        "zeroed_potential_kj": [float(v) for v in result.zeroed_potential_kj],
        "torsion_contribution_kj": [float(v) for v in result.gromacs_difference],
        "selfcheck_max_abs_error_kj": float(result.max_abs_error),
        "selfcheck_tolerance_kj": float(result.tolerance),
        # Milestone 6a: the parameters the self-check used, as grompp built them,
        # with where each came from; milestone 6b's report reads its "before"
        # values here. And [ defaults ] as grompp read it, include or not.
        "torsion_parameters": [_torsion_record(t) for t in resolved.terms],
        "defaults": _defaults_record(resolved.defaults),
        "preprocessed_topology_sha256": resolved.pp_sha256,
    }
    (workdir / "energies.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_manifest(args.outdir, "mm_scan", digest, {"quad": list(quad)})
    print(f"mmscan: wrote {workdir / 'energies.json'}")
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    """Fit the dihedral series from the QM scan and the MM-zeroed energies.

    The target curve is spec 4.7's ``dE = (E_QM - min E_QM) - (E_MM0 - min
    E_MM0)``.  The angles are the ones ``mmscan`` measured from the
    geometries GROMACS was given (spec section 9.5), not a requested grid.
    """
    if args.dry_run:
        _, topology, quad = _load_inputs(args)
        print(f"would fit torsion {quad} of {topology.name} into {args.outdir}")
        return 0

    _, topology, quad = _prepare(args)

    scan_path = _scan_path(args)
    scan = _read_json(scan_path, "QM scan (scan.json)", _SCAN_REMEDY)
    qm = _require_array(scan, "energy_kj", scan_path)

    energies_path = _energies_path(args)
    energies = _read_json(
        energies_path, "MM energies (energies.json)", "Run 'ffrefine mmscan' first."
    )
    mm0 = _require_array(energies, "zeroed_potential_kj", energies_path)
    phi = _require_array(energies, "phi_deg", energies_path)

    # Before anything numeric: are these two payloads even about the same
    # torsion and the same geometries? Matching array lengths are not evidence.
    frames_digest = _check_stage_pairing(quad, scan, scan_path, energies, energies_path)

    if not (qm.shape == mm0.shape == phi.shape):
        raise ValidationError(
            "the QM scan and the MM energies must cover the same number of points, in the "
            f"same frame order: {scan_path} has {qm.size} QM energies, {energies_path} has "
            f"{mm0.size} MM energies and {phi.size} angles."
        )
    if qm.size < 2:
        raise ValidationError(
            f"a torsion fit needs a profile, but only {qm.size} point(s) are available. "
            + _SCAN_REMEDY
        )

    settings = {
        "quad": list(quad),
        "multiplicities": list(args.multiplicities),
        "emax": args.emax,
        "boltzmann": args.boltzmann,
        "free_phase": bool(args.free_phase),
        "distribute": bool(args.distribute),
    }
    digest = hash_inputs([args.topology, scan_path, energies_path], settings)
    if args.resume and stage_is_current(args.outdir, "fit", digest, _stage_outputs(args, "fit")):
        print(f"fit: skipping, {Path(args.outdir) / STAGE_DIRS['fit']} is current (--resume)")
        return 0

    target = (qm - qm.min()) - (mm0 - mm0.min())

    # Spec 4.7 / ruling R34: the cosine-only basis is even in phi, so a
    # phase-shifted or chiral profile cannot be represented by it at all.
    # Detect that, say so, and name the remedy -- silently returning a
    # cosine-only fit of an asymmetric profile is exactly the failure this
    # check exists to prevent.
    metric = asymmetry_metric(phi, target)
    if is_asymmetric(phi, target):
        if args.free_phase:
            log.info(
                "target profile is asymmetric about phi = 0 (RMS mirror mismatch "
                "%.3f kJ/mol); --free-phase is set, so arbitrary phases are fitted.",
                metric,
            )
        else:
            _warn(
                f"target profile is asymmetric about phi = 0: RMS mirror mismatch "
                f"{metric:.3f} kJ/mol, above the {DEFAULT_ASYMMETRY_TOLERANCE:g} kJ/mol "
                f"tolerance. The cosine-only basis restricts phases to 0/180 degrees and "
                f"cannot represent a phase-shifted profile, so this fit will be biased. "
                f"Re-run with --free-phase to fit arbitrary phases (spec section 4.7)."
            )

    if args.boltzmann is not None:
        weights = boltzmann_weights(qm, args.boltzmann, args.emax)
    else:
        weights = uniform_weights(qm, args.emax)
    discarded = int(np.count_nonzero(weights == 0.0))
    if discarded:
        print(f"discarded {discarded} of {qm.size} points above {args.emax:g} kJ/mol (--emax)")
    kept = int(np.count_nonzero(weights))
    # An underdetermined least-squares problem does not fail: np.linalg.lstsq
    # returns a minimum-norm solution that passes exactly through every
    # surviving point, so the fit reports RMSE 0 and looks perfect while
    # meaning nothing. Refuse it rather than emit that.
    max_params = 1 + len(args.multiplicities) * (2 if args.free_phase else 1)
    if kept <= max_params:
        raise ValidationError(
            f"only {kept} of {qm.size} point(s) survive the --emax {args.emax:g} kJ/mol cut, "
            f"but the largest candidate model has {max_params} parameters. The fit would be "
            f"underdetermined and would report a meaningless RMSE of zero. Raise --emax, or "
            f"pass fewer --multiplicities."
        )

    candidates = rank_candidates(
        phi, target, args.multiplicities, weights=weights, free_phase=args.free_phase
    )
    table = format_ranking_table(candidates)
    print(table)
    # Spec section 8: the durable record is run.log. The table was stdout-only,
    # so a resumed or redirected run kept no evidence of what was ranked or
    # why. file_only keeps stderr from showing it a second time.
    log_to_run_log("BIC ranking of candidate multiplicity subsets:\n" + table)
    chosen = select_model(candidates)
    result = chosen.result
    log_to_run_log(
        f"fit settings: quad {list(quad)}, candidates {list(args.multiplicities)}, "
        f"emax {args.emax:g} kJ/mol, "
        f"{'uniform' if args.boltzmann is None else f'boltzmann {args.boltzmann:g} K'} "
        f"weighting, free_phase {bool(args.free_phase)}, distribute "
        f"{bool(args.distribute)}; {kept} of {qm.size} point(s) used, {discarded} discarded; "
        f"selected {list(chosen.mults)}, design-matrix rank {result.rank} of "
        f"{result.n_params} parameter(s)"
    )
    print(
        f"\nselected multiplicities {list(chosen.mults)}: RMSE {result.rmse:.4f} kJ/mol, "
        f"max deviation {result.max_dev:.4f} kJ/mol"
    )
    # These describe the fitted CORRECTION curve (the fit of dE = QM - MM0),
    # not the molecule's torsion profile. Printing them unlabelled invited
    # reading "largest barrier" as the syn barrier, which it is not: the
    # refined torsion profile is MM plus this correction, and 'validate'
    # reports that one.
    stats = profile_stats(phi, result.predicted)
    print("\nfitted correction curve (dE = QM - MM0), NOT the torsion profile:")
    for angle, depth in zip(stats.minima_deg, stats.minima_kj):
        print(f"  correction minimum at {angle:8.2f} deg, {depth:8.3f} kJ/mol")
    print(
        f"  correction global minimum {stats.global_min_deg:.2f} deg, "
        f"largest correction swing {stats.max_barrier_kj:.3f} kJ/mol"
    )
    qm_stats = profile_stats(phi, qm)
    print("QM reference profile (these ARE torsion barriers):")
    print(
        f"  global minimum {qm_stats.global_min_deg:.2f} deg, "
        f"largest barrier {qm_stats.max_barrier_kj:.3f} kJ/mol"
    )

    outdir = stage_dir(args.outdir, "fit")
    # Spec 4.8 asks fit.json to carry every number. That includes the settings
    # that shaped the fit, not just its results: without them a fit.json cannot
    # be reproduced or judged, and the BIC ranking existed only on stdout.
    provenance = {
        "quad_1_based": list(quad),
        "frames_sha256": frames_digest,
        "scan_json": str(scan_path),
        "energies_json": str(energies_path),
        "ffrefine_version": __version__,
        "candidate_multiplicities": [int(m) for m in args.multiplicities],
        "emax_kj_per_mol": float(args.emax),
        "weighting": (
            "uniform" if args.boltzmann is None else f"boltzmann at {float(args.boltzmann):g} K"
        ),
        "boltzmann_temperature_k": None if args.boltzmann is None else float(args.boltzmann),
        "n_points_total": int(qm.size),
        "n_points_discarded_by_emax": discarded,
        "n_points_used": kept,
        "free_phase": bool(args.free_phase),
        "distribute": bool(args.distribute),
        "asymmetry_kj_per_mol": float(metric),
        "asymmetry_tolerance_kj_per_mol": float(DEFAULT_ASYMMETRY_TOLERANCE),
        "bic_ranking": [
            {
                "multiplicities": [int(m) for m in c.mults],
                "n_parameters": int(c.n_params),
                "rmse_kj_per_mol": float(c.rmse),
                "bic": float(c.bic),
            }
            for c in candidates
        ],
        "selected_multiplicities": [int(m) for m in chosen.mults],
    }
    (outdir / "fit.json").write_text(
        json.dumps(fit_summary(result, phi, target, provenance, qm_kj=qm), indent=2) + "\n"
    )
    write_refit_topology(
        topology,
        quad,
        result.k,
        result.delta_deg,
        result.mults,
        outdir / "dihedral.itp",
        distribute=args.distribute,
        includes="verbatim",
    )
    write_prm_block(
        topology,
        quad,
        result.k,
        result.delta_deg,
        result.mults,
        outdir / "dihedral.prm",
        args.distribute,
    )
    if not args.distribute:
        _warn(
            "fitted without --distribute: the series is written onto the target "
            "quadruplet alone and every other torsion about that bond is zeroed. That is "
            "exact for this molecule but is not a transferable dihedral type, and the "
            "CHARMM .prm block carries a matching warning."
        )

    plots = outdir / "plots"
    plot_qm_profile(phi, qm, plots / "qm_profile.png")
    plot_mm_zeroed(phi, mm0, plots / "mm_zeroed.png")
    plot_fit(phi, target, result, plots / "fit.png")
    plot_residuals(phi, result, plots / "residuals.png")

    write_manifest(
        args.outdir,
        "fit",
        digest,
        {
            "quad": list(quad),
            "multiplicities": [int(m) for m in result.mults],
            "rmse_kj_per_mol": float(result.rmse),
            "distribute": bool(args.distribute),
            "free_phase": bool(args.free_phase),
        },
    )
    print(f"fit: wrote {outdir / 'fit.json'}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Rebuild with the fitted parameters and compare MM to QM (spec 4.9).

    Ruling R33: the "before" state handed to
    :func:`ffrefine.mm.validate_fit.validate_refit` is ``topology`` -- the
    topology read from ``--topology``, carrying the force field's own pre-fit
    torsion parameters.  Nothing in this function zeroes it first, and the
    zeroed copy ``mmscan`` writes into ``02_mm_scan/`` is never read here.
    Using that zeroed copy would report improvement over *no torsion at all*
    as the headline number.

    In a ``run`` that refined charges first, ``--topology`` has been
    repointed at ``04_charges/molecule_charged.itp``, so "before" is that
    charge-refined topology: the force field's torsion terms with the new
    charges.  ``validation.json`` records the file actually used
    (``before_topology``), whether it is this outdir's charges-stage output
    (``charges_refined``) and the user's own ``-p`` (``input_topology``)
    (ruling R24).
    """
    if args.dry_run:
        _, topology, quad = _load_inputs(args)
        print(f"would validate torsion {quad} of {topology.name} in {args.outdir}")
        return 0

    structure, topology, quad = _prepare(args)

    fit_path = Path(args.outdir) / STAGE_DIRS["fit"] / "fit.json"
    fit_payload = _read_json(fit_path, "fit result (fit.json)", "Run 'ffrefine fit' first.")
    terms = fit_payload.get("terms")
    if not terms:
        raise ValidationError(f"{fit_path} carries no fitted 'terms'; re-run 'ffrefine fit'")
    mult = np.array([int(t["multiplicity"]) for t in terms], dtype=int)
    k = np.array([float(t["k_kj_per_mol"]) for t in terms], dtype=float)
    delta = np.array([float(t["phase_deg"]) for t in terms], dtype=float)

    scan_path = _scan_path(args)
    scan = _read_json(scan_path, "QM scan (scan.json)", _SCAN_REMEDY)
    qm = _require_array(scan, "energy_kj", scan_path)
    if "frames_nm" not in scan:
        raise ValidationError(
            f"{scan_path} has no 'frames_nm' array. Validation re-runs the MM single "
            "points on the very geometries the QM profile was computed on (spec section "
            "4.9), so the coordinates are required. " + _SCAN_REMEDY
        )
    frames = _require_array(scan, "frames_nm", scan_path)
    if frames.ndim != 3 or frames.shape[1:] != (structure.natoms, 3):
        raise ValidationError(
            f"{scan_path}: 'frames_nm' has shape {frames.shape}, expected "
            f"(nframes, {structure.natoms}, 3) to match the structure"
        )
    if qm.shape[0] != frames.shape[0]:
        raise ValidationError(
            f"{scan_path}: {qm.size} QM energies but {frames.shape[0]} geometries"
        )

    # The fit records nothing about the torsion it was fitted for unless we put
    # it there, so a --dihedral typo used to apply one torsion's parameters to
    # another and report a confident RMSE for it.
    _check_fit_applies_here(quad, fit_payload, fit_path, frames, scan_path)

    ff_includes = _ff_includes(args)
    binaries = _binaries(args)
    final = stage_dir(args.outdir, "final")
    # The "before" energies depend on every force-field file grompp reads, so
    # the key covers grompp's preprocessed view of the topology (milestone 6a
    # design, section 4). About a second, even when the stage is then skipped.
    pp_sha256 = preprocessed_sha256(binaries, topology, structure, final, "before", ff_includes)
    settings = {
        "quad": list(quad),
        "distribute": bool(args.distribute),
        "ff_includes": ff_includes,
        "preprocessed_topology_sha256": pp_sha256,
    }
    digest = hash_inputs([args.topology, scan_path, fit_path], settings)
    if args.resume and stage_is_current(
        args.outdir, "final", digest, _stage_outputs(args, "final")
    ):
        print(f"validate: skipping, {Path(args.outdir) / STAGE_DIRS['final']} is current")
        return 0

    result = validate_refit(
        binaries,
        topology,  # R33: --topology as given (charge-refined in a chained run), never zeroed.
        quad,
        structure,
        frames,
        qm,
        k,
        delta,
        mult,
        final,
        distribute=args.distribute,
        nthreads=args.nthreads,
        strict_names=not args.no_strict_names,
        ff_includes=ff_includes,
    )

    print(
        f"validation (spec 4.9): MM against QM RMSE {result.rmse_before:.4f} kJ/mol before "
        f"refitting, {result.rmse_after:.4f} kJ/mol after"
    )
    if not result.improved:
        _warn(
            f"refitting did not improve MM/QM agreement ({result.rmse_before:.4f} -> "
            f"{result.rmse_after:.4f} kJ/mol). Check the scan for conformational hops and "
            "the 02_mm_scan diagnostics before using these parameters."
        )

    # Written, not copied from after.itp: after.itp is a grompp input and so
    # carries absolute #include paths, while the file the user keeps carries
    # their own (milestone 6a design, section 4). Same bytes for butane.
    write_refit_topology(
        topology,
        quad,
        k,
        delta,
        mult,
        final / "molecule_refined.itp",
        distribute=args.distribute,
        includes="verbatim",
    )
    before_topology = Path(args.topology).resolve()
    charged_output = Path(args.outdir) / STAGE_DIRS["charges"] / "molecule_charged.itp"
    charges_refined = before_topology == charged_output.resolve()
    plot_validation(
        result.phi_deg,
        result.qm_kj,
        result.mm_before_kj,
        result.mm_after_kj,
        result.rmse_before,
        result.rmse_after,
        final / "plots" / "validation.png",
    )
    (final / "validation.json").write_text(
        json.dumps(
            {
                "quad_1_based": list(quad),
                "rmse_before_kj_per_mol": float(result.rmse_before),
                "rmse_after_kj_per_mol": float(result.rmse_after),
                "improved": bool(result.improved),
                "distribute": bool(args.distribute),
                "phi_deg": [float(v) for v in result.phi_deg],
                "qm_kj_per_mol": [float(v) for v in result.qm_kj],
                "mm_before_kj_per_mol": [float(v) for v in result.mm_before_kj],
                "mm_after_kj_per_mol": [float(v) for v in result.mm_after_kj],
                "before_topology": str(before_topology),
                "charges_refined": bool(charges_refined),
                "input_topology": str(Path(_input_topology(args)).resolve()),
            },
            indent=2,
        )
        + "\n"
    )
    write_manifest(
        args.outdir,
        "final",
        digest,
        {"rmse_before": float(result.rmse_before), "rmse_after": float(result.rmse_after)},
    )
    print(f"validate: wrote {final / 'molecule_refined.itp'}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run every implemented stage in order, naming the ones it cannot run.

    Charges lead, when they were asked for: the fitted torsion absorbs whatever
    the 1-4 electrostatics do not account for, so refining charges afterwards
    would leave a torsion fitted against charges the topology no longer has.
    """
    wants_charges = bool(args.charge_atoms) or args.charge_mode == "all"
    if wants_charges:
        print(
            "ffrefine run: chaining charges -> scan -> mmscan -> fit -> validate. The "
            "remaining stages use the charge-refined topology, because a torsion fitted "
            "against different charges than it ships with is not transferable."
        )
    else:
        print(
            "ffrefine run: chaining scan -> mmscan -> fit -> validate and SKIPPING "
            "charges, which needs --charge-atoms or --charge-mode all. The topology keeps "
            "the charges it came with."
        )

    stages: list[tuple[str, object]] = []
    if wants_charges:
        stages.append(("charges", cmd_charges))
    stages += [
        ("scan", cmd_scan),
        ("mmscan", cmd_mmscan),
        ("fit", cmd_fit),
        ("validate", cmd_validate),
    ]

    for name, handler in stages:
        print(f"\n=== {name} ===")
        code = handler(args)
        if code != 0:
            print(f"ffrefine run: stopping, stage {name} failed", file=sys.stderr)
            return code
        if name == "charges" and not args.dry_run:
            charged = Path(args.outdir) / STAGE_DIRS["charges"] / "molecule_charged.itp"
            if not charged.exists():
                raise ValidationError(
                    f"the charges stage reported success but wrote no {charged}, so the "
                    "remaining stages have no refined topology to use."
                )
            # 00_input/ and validation.json must still be able to name the
            # user's own input once -p points at a stage output (ruling R24).
            args.input_topology = _input_topology(args)
            args.topology = str(charged)
            print(f"run: remaining stages will use {charged}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns a process exit code.

    Args:
        argv: argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        0 on success, 1 for a reported ffrefine failure.  argparse usage
        errors (exit 2, and ``--help`` / ``--version``, exit 0) raise
        ``SystemExit`` before this function can return, which is argparse's
        own convention and still produces a clean message rather than a
        traceback.
    """
    args = build_parser().parse_args(argv)
    handlers = {
        "scan": cmd_scan,
        "charges": cmd_charges,
        "mmscan": cmd_mmscan,
        "fit": cmd_fit,
        "validate": cmd_validate,
        "run": cmd_run,
    }
    try:
        if args.dry_run:
            _reset_logging()
        _apply_config(args)
        return handlers[args.command](args)
    except FFRefineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised via the console script
    sys.exit(main())
