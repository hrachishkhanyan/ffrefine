"""Command line interface (spec section 3).

Subcommands: ``scan``, ``mmscan``, ``fit``, ``charges``, ``validate``,
``run``.  ``scan`` (xtb, milestone 4) and ``charges`` (ORCA, milestone 5) are
registered so the CLI surface matches spec section 3, but they refuse to run
and exit nonzero rather than pretending to work.

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

Exit codes: 0 success, 1 a reported ffrefine failure, 2 a usage error or a
subcommand that is not implemented in this build.

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
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

from . import __version__
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
from .io.prm import write_prm_block
from .io.structure import Structure, read_structure
from .io.topology import Topology, read_topology
from .io.topwriter import write_refit_topology
from .io.validate import adopt_topology_naming, validate_structure_topology
from .mm.selfcheck import check_torsion_removal
from .mm.torsionset import collect_torsion_set, validate_quad
from .mm.validate_fit import validate_refit
from .plots import plot_fit, plot_mm_zeroed, plot_qm_profile, plot_residuals, plot_validation
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

    scan = sub.add_parser("scan", help="QM relaxed torsion scan (milestone 4, NOT IMPLEMENTED)")
    _add_common(scan)
    _add_torsion(scan)
    scan.add_argument("--step", type=float, help="scan step in degrees (default: 15)")

    mmscan = sub.add_parser("mmscan", help="GROMACS single points with and without the torsion")
    _add_common(mmscan)
    _add_torsion(mmscan)

    fit = sub.add_parser("fit", help="fit the dihedral series and write new parameters")
    _add_common(fit)
    _add_torsion(fit)
    _add_fit_flags(fit)

    charges = sub.add_parser("charges", help="ESP/RESP charges via ORCA (milestone 5, NOT IMPL.)")
    _add_common(charges)

    validate = sub.add_parser("validate", help="rebuild with fitted parameters and compare to QM")
    _add_common(validate)
    _add_torsion(validate)

    run = sub.add_parser("run", help="run every implemented stage in order")
    _add_common(run)
    _add_torsion(run)
    _add_fit_flags(run)

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


def _warn(message: str) -> None:
    """Emit a user-facing warning exactly once.

    Once logging is configured, ``setup_logging``'s own console handler puts
    the message on stderr *and* into ``run.log``; printing as well would
    double it.  Before that (dry runs, argument handling) there is no logger
    to carry it, so it goes straight to stderr.
    """
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


def _load_inputs(args: argparse.Namespace) -> tuple[Structure, Topology, Quad]:
    """Read, cross-check and return the structure, topology and target quad.

    Raises:
        ValidationError: no ``--dihedral`` was given, more than one was given,
            or the quadruplet is unusable with this topology.
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
    structure = adopt_topology_naming(structure, topology)

    quads = getattr(args, "dihedral", None) or []
    if not quads:
        raise ValidationError(
            "no torsion selected. Pass --dihedral i,j,k,l using 1-based atom "
            "indices. Automatic selection from a CGenFF .str file arrives in "
            "milestone 6 and is not available in this build."
        )
    if len(quads) > 1:
        raise ValidationError(
            f"{len(quads)} torsions requested, but this build fits one torsion per run: "
            "spec section 6's output layout holds a single QM scan and a single fit per "
            "outdir. Re-run with one --dihedral and its own --outdir for each torsion."
        )
    quad = quads[0]
    validate_quad(topology, quad)
    return structure, topology, quad


def _write_input_stage(
    args: argparse.Namespace, structure: Structure, topology: Topology, quad: Quad
) -> None:
    """Populate ``00_input/`` (spec section 6).

    Copies -- never moves or rewrites -- the two input files and dumps the
    parsed model, including the 1-4 conventions that make the eventual fit
    convention-bound (spec section 9.3).
    """
    target = stage_dir(args.outdir, "input")
    for source in (Path(args.structure), Path(args.topology)):
        destination = target / source.name
        if destination.resolve() != source.resolve():
            shutil.copyfile(source, destination)

    defaults = topology.defaults
    torsion_set = collect_torsion_set(topology, quad)
    model = {
        "structure": str(Path(args.structure).resolve()),
        "topology": str(Path(args.topology).resolve()),
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
        "target_quad_1_based": list(quad),
        "central_bond_1_based": [quad[1], quad[2]],
        "torsion_set_size": len(torsion_set),
        "torsion_set_1_based": [[d.ai, d.aj, d.ak, d.al] for d in torsion_set],
    }
    if defaults is None:
        _warn(
            f"{args.topology} has no [ defaults ] directive, so fudgeQQ/fudgeLJ could not "
            "be read. The fitted torsion is only valid with the 1-4 conventions it was "
            "fitted against (spec section 9.3); record them yourself."
        )
    (target / "model.json").write_text(json.dumps(model, indent=2) + "\n")


def _prepare(args: argparse.Namespace) -> tuple[Structure, Topology, Quad]:
    """Configure logging, read the inputs and write ``00_input/``."""
    setup_logging(args.outdir, args.verbose)
    structure, topology, quad = _load_inputs(args)
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


_STAGE_OUTPUTS: dict[str, tuple[str, ...]] = {
    "mm_scan": ("energies.json",),
    "fit": ("fit.json", "dihedral.itp", "dihedral.prm"),
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
    "Run 'ffrefine scan' first -- but that stage is milestone 4 and is not implemented "
    "in this build, so supply the file yourself: a JSON object with 'energy_kj' (QM "
    "energies in kJ/mol, one per frame) and 'frames_nm' (an (nframes, natoms, 3) array "
    "of coordinates in nanometres, in topology atom order)."
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


def cmd_not_implemented(name: str, milestone: int) -> int:
    """Report a subcommand that belongs to a later milestone.

    Writes nothing and touches no directory, so a caller cannot mistake the
    call for a stage that ran.
    """
    print(
        f"ffrefine {name}: NOT IMPLEMENTED -- this subcommand arrives in milestone "
        f"{milestone} and nothing was written. Implemented in this build: mmscan, fit, "
        f"validate, run.",
        file=sys.stderr,
    )
    return 2


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
    settings = {
        "quad": list(quad),
        "nframes": int(frames.shape[0]),
        "ff_includes": ff_includes,
    }
    digest = hash_inputs(inputs, settings)
    if args.resume and stage_is_current(
        args.outdir, "mm_scan", digest, _stage_outputs(args, "mm_scan")
    ):
        print(f"mmscan: skipping, {_energies_path(args)} is current (--resume)")
        return 0

    binaries = _binaries(args)
    workdir = stage_dir(args.outdir, "mm_scan")
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
    )
    print(
        f"torsion {quad}: spec 4.5 self-consistency check passed over "
        f"{len(result.phi_deg)} frame(s), max deviation {result.max_abs_error:.3g} kJ/mol "
        f"(tolerance {result.tolerance:g})"
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
    ORIGINAL force-field topology read from ``--topology``, carrying the force
    field's own pre-fit parameters.  Nothing in this function zeroes it first,
    and the zeroed copy ``mmscan`` writes into ``02_mm_scan/`` is never read
    here.  Using that zeroed copy would report improvement over *no torsion at
    all* as the headline number.
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
    settings = {
        "quad": list(quad),
        "distribute": bool(args.distribute),
        "ff_includes": ff_includes,
    }
    digest = hash_inputs([args.topology, scan_path, fit_path], settings)
    if args.resume and stage_is_current(
        args.outdir, "final", digest, _stage_outputs(args, "final")
    ):
        print(f"validate: skipping, {Path(args.outdir) / STAGE_DIRS['final']} is current")
        return 0

    binaries = _binaries(args)
    final = stage_dir(args.outdir, "final")
    result = validate_refit(
        binaries,
        topology,  # R33: the ORIGINAL topology, never a zeroed one.
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

    shutil.copyfile(final / "after.itp", final / "molecule_refined.itp")
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
                "before_topology": "the original force field topology, unmodified",
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
    """Run every implemented stage in order, naming the ones it cannot run."""
    print(
        "ffrefine run: this build implements milestones 1-3, so it chains "
        "mmscan -> fit -> validate and SKIPS two stages:\n"
        "  scan    (QM relaxed torsion scan, xtb) -- milestone 4, not implemented\n"
        "  charges (ESP/RESP charges, ORCA)       -- milestone 5, not implemented\n"
        "The QM scan must therefore be supplied as 01_qm_scan/scan.json; charges are "
        "left exactly as the input topology has them."
    )
    for name, handler in (("mmscan", cmd_mmscan), ("fit", cmd_fit), ("validate", cmd_validate)):
        print(f"\n=== {name} ===")
        code = handler(args)
        if code != 0:
            print(f"ffrefine run: stopping, stage {name} failed", file=sys.stderr)
            return code
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns a process exit code.

    Args:
        argv: argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        0 on success, 1 for a reported ffrefine failure, 2 for a subcommand
        this build does not implement.  argparse usage errors (and ``--help``
        / ``--version``) raise ``SystemExit`` before this function can return,
        which is argparse's own convention and still produces a clean message
        rather than a traceback.
    """
    args = build_parser().parse_args(argv)
    handlers = {
        "scan": lambda a: cmd_not_implemented("scan", 4),
        "charges": lambda a: cmd_not_implemented("charges", 5),
        "mmscan": cmd_mmscan,
        "fit": cmd_fit,
        "validate": cmd_validate,
        "run": cmd_run,
    }
    try:
        # Unimplemented stages are dispatched before any config handling or
        # file reading, so they cannot half-run and cannot create an outdir.
        if args.command in ("scan", "charges"):
            return handlers[args.command](args)
        if args.dry_run:
            _reset_logging()
        _apply_config(args)
        return handlers[args.command](args)
    except FFRefineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised via the console script
    sys.exit(main())
