"""Output layout, input hashing and resumable stages (spec sections 6 and 8).

The pipeline writes its outputs into a fixed set of numbered subdirectories
under a single ``outdir`` (spec section 6). Each stage that completes writes
a small JSON manifest recording a hash of everything that fed into it; a
later ``--resume`` run can then compare a freshly computed hash against the
recorded one to decide whether the stage can be skipped (spec section 8).

Getting :func:`stage_is_current` wrong is asymmetric: treating a stale stage
as current silently reuses results computed from different inputs and
produces a plausible-looking but wrong final answer. Treating a current
stage as stale merely wastes time recomputing it. This module is therefore
deliberately conservative: any manifest that is missing, unreadable, or of
the wrong shape is treated as "stage not current" rather than raising or
guessing.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .errors import ValidationError

STAGE_DIRS: dict[str, str] = {
    "input": "00_input",
    "qm_scan": "01_qm_scan",
    "mm_scan": "02_mm_scan",
    "fit": "03_fit",
    "charges": "04_charges",
    "final": "final",
}
"""The spec section 6 output layout: logical stage name -> directory name."""

MANIFEST_NAME = "manifest.json"
"""Filename of the per-stage resume manifest, written inside each stage dir."""


def _validate_stage(stage: str) -> None:
    """Raise :class:`ValidationError` if ``stage`` is not a known stage name."""
    if stage not in STAGE_DIRS:
        raise ValidationError(f"unknown stage {stage!r}; expected one of {sorted(STAGE_DIRS)}")


def stage_dir(outdir: str | Path, stage: str) -> Path:
    """Return the directory for a pipeline stage, creating it if necessary.

    Args:
        outdir: the run's top-level output directory.
        stage: logical stage name, one of the keys of :data:`STAGE_DIRS`.

    Returns:
        The stage's directory, guaranteed to exist.

    Raises:
        ValidationError: if ``stage`` is not a recognised stage name.
    """
    _validate_stage(stage)
    path = Path(outdir) / STAGE_DIRS[stage]
    path.mkdir(parents=True, exist_ok=True)
    return path


def hash_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file's contents, hex encoded.

    Args:
        path: file to hash. Read in binary mode; never modified.

    Returns:
        Lowercase hex digest.

    Raises:
        ValidationError: ``path`` does not exist, is a directory, or is not
            readable.
    """
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValidationError(f"{path}: could not be read: {exc}") from exc
    return digest.hexdigest()


def hash_array(values) -> str:
    """Return the SHA-256 digest of a numeric array's float64 contents.

    Used to bind one stage's output to the geometries another stage consumed
    (``02_mm_scan/energies.json`` and ``03_fit/fit.json`` both record the
    digest of ``01_qm_scan/scan.json``'s ``frames_nm``).  Hashing the array
    rather than the whole scan file is deliberate: the MM energies and
    measured angles depend on the *geometries* only, so replacing the QM
    ``energy_kj`` column -- which is the documented workflow, and spec section
    7.1's round trip -- must not invalidate them, while changing a coordinate
    must.

    Args:
        values: anything ``numpy.asarray(..., dtype=float)`` accepts.

    Returns:
        Lowercase hex digest, stable across runs for identical values.
    """
    import numpy as np

    arr = np.ascontiguousarray(np.asarray(values, dtype=float))
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def hash_inputs(paths: list[str | Path], extra: dict | None = None) -> str:
    """Return a combined SHA-256 hash of input files and stage settings.

    Used to decide whether a stage's recorded manifest still matches its
    current inputs (see :func:`stage_is_current`): the same files with the
    same settings must always hash identically, and changing either the file
    contents, the settings, or which files are included must change the
    hash.

    ``paths`` are hashed by content, in the order given. Order is
    significant: callers pass a stage's inputs in a fixed, meaningful
    sequence (e.g. topology before structure), and this function does not
    second-guess that ordering.

    ``extra`` captures whatever non-file settings also affect the stage's
    output (grid step, energy cutoff, multiplicities, ...). It is serialised
    to JSON with keys sorted, so the *order* keys happen to appear in the
    dict never changes the hash -- only the key/value contents do, which is
    the only thing that is semantically meaningful for a plain dict.

    Args:
        paths: input files, hashed by content in the given order.
        extra: JSON-serialisable settings that also affect the stage;
            defaults to an empty mapping.

    Returns:
        Lowercase hex digest.
    """
    digest = hashlib.sha256()
    for path in paths:
        digest.update(hash_file(path).encode("utf-8"))
    digest.update(json.dumps(extra or {}, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


def write_manifest(
    outdir: str | Path, stage: str, input_hash: str, metadata: dict | None = None
) -> None:
    """Record that a stage completed, and the hash of the inputs it used.

    Creates the stage directory if it does not already exist.

    Args:
        outdir: the run's top-level output directory.
        stage: logical stage name, one of the keys of :data:`STAGE_DIRS`.
        input_hash: the hash (from :func:`hash_inputs`) of this stage's
            inputs at the time it completed.
        metadata: arbitrary JSON-serialisable extra information about the
            run (e.g. fitted multiplicities); defaults to an empty mapping.

    Raises:
        ValidationError: if ``stage`` is not a recognised stage name.
    """
    path = stage_dir(outdir, stage) / MANIFEST_NAME
    payload = {
        "stage": stage,
        "input_hash": input_hash,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def stage_is_current(
    outdir: str | Path,
    stage: str,
    input_hash: str,
    outputs: list[str | Path] | None = None,
) -> bool:
    """Decide whether a stage can be skipped under ``--resume``.

    Returns ``True`` only when the stage's manifest exists, parses as a JSON
    object, its ``input_hash`` field is a string, and that string matches
    ``input_hash`` exactly. Every other case -- the outdir or stage directory
    not existing, no manifest having been written yet, a manifest that fails
    to parse (e.g. truncated by a killed run), a manifest whose ``input_hash``
    is missing or explicitly ``null``, or a manifest whose hash differs --
    returns ``False``. This function never raises on a bad manifest; a
    corrupt or unreadable manifest is treated the same as a missing one, so a
    damaged run always gets recomputed rather than silently trusted.

    The requirement that the recorded ``input_hash`` be a string (not merely
    equal to the caller's value) is deliberate: without it, a manifest with
    no recorded hash compared against a caller-supplied ``None`` would
    degenerate to ``None == None`` and report a stale stage as current, which
    is exactly the dangerous direction of error this predicate exists to
    avoid. A caller is expected to always pass the ``str`` produced by
    :func:`hash_inputs`; this function does not separately reject a
    non-``str`` ``input_hash`` argument, since the string-typed check on the
    *recorded* side already makes that case safe (it can only ever compare
    unequal, never spuriously equal).

    Args:
        outdir: the run's top-level output directory.
        stage: logical stage name, one of the keys of :data:`STAGE_DIRS`.
        input_hash: the hash (from :func:`hash_inputs`) of the stage's
            current inputs, to compare against what was recorded.
        outputs: files the stage is expected to have left behind. A manifest
            proves the stage *ran*, not that its results survived: deleting
            ``03_fit/fit.json`` and re-running with ``--resume`` used to print
            "fit: skipping, ... is current" and leave no file. Any of these
            missing makes the stage stale. Paths are checked, never created.

    Returns:
        Whether the previously completed stage's recorded inputs match and
        its outputs are still on disk.

    Raises:
        ValidationError: if ``stage`` is not a recognised stage name (this
            indicates a programming error, not a stale-cache condition, so
            it is not swallowed like a corrupt manifest is).
    """
    _validate_stage(stage)
    manifest_path = Path(outdir) / STAGE_DIRS[stage] / MANIFEST_NAME
    if not manifest_path.exists():
        return False
    try:
        data = json.loads(manifest_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    recorded = data.get("input_hash")
    if not isinstance(recorded, str):
        # Missing key, or explicit null: treat exactly like a corrupt manifest
        # rather than comparing. Without this guard, a manifest with no
        # recorded hash compared against a caller-supplied ``None`` would
        # degenerate to ``None == None`` and report a stale stage as current.
        return False
    if recorded != input_hash:
        return False
    for output in outputs or ():
        if not Path(output).exists():
            logging.getLogger("ffrefine").info(
                "stage %s has a current manifest but %s is missing; recomputing", stage, output
            )
            return False
    return True


class WarningCollector(logging.Handler):
    """Collects the text of every WARNING-or-worse record while attached.

    ``cli._collect_warnings`` attaches one around each stage, so the stage's
    JSON payload can carry every warning it raised -- including the ones a
    module logs itself (GROMACS's echoed warnings, the self-check's structure
    warnings), not only the CLI's own. :func:`setup_logging` leaves these
    attached; they belong to whoever opened them.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def setup_logging(outdir: str | Path, verbose: bool = False) -> None:
    """Configure the ``ffrefine`` logger to log to the console and run.log.

    :mod:`ffrefine.runner` logs every external command (its full command
    line, and where its captured stdout/stderr live) through the
    ``"ffrefine"`` logger; this function is what routes those messages to
    the console and to ``<outdir>/run.log``. Call it once at CLI startup.

    Idempotent: calling this more than once (e.g. because a resumed run
    re-invokes CLI setup) replaces any handlers this function previously
    attached rather than adding a second copy, so log lines are never
    duplicated. Propagation to the root logger is disabled so this function
    fully owns where ``ffrefine`` messages go, regardless of what other
    logging configuration (pytest's, or a future caller's) exists on the
    root logger.

    Args:
        outdir: the run's top-level output directory; ``run.log`` is written
            directly inside it (spec section 6). Created if it does not
            already exist.
        verbose: if set, emit DEBUG-level messages; otherwise INFO.

    Handlers that are :class:`WarningCollector` instances are left in place.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger("ffrefine")
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    log.propagate = False

    for handler in list(log.handlers):
        if isinstance(handler, WarningCollector):
            continue
        log.removeHandler(handler)
        handler.close()

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.addFilter(_FileOnlyFilter())
    log.addHandler(console)

    file_handler = logging.FileHandler(outdir / "run.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    log.addHandler(file_handler)


class _FileOnlyFilter(logging.Filter):
    """Drop records marked ``file_only`` so they reach run.log but not stderr.

    Used for material a stage already prints to stdout in a nicer form (the
    BIC ranking table): it belongs in the durable log, but echoing it on
    stderr as well would show the user the same table twice.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(record, "file_only", False)


def log_to_run_log(message: str) -> None:
    """Record ``message`` in ``run.log`` without echoing it to the console.

    A no-op before :func:`setup_logging` has run (a ``--dry-run`` writes
    nothing at all), since the logger then has no handlers.

    Args:
        message: text to record; may span several lines.
    """
    logging.getLogger("ffrefine").info("%s", message, extra={"file_only": True})
