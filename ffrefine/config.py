"""Binary discovery and configuration merging (spec sections 1 and 3)."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .errors import FFRefineError


@dataclass
class Binaries:
    """Resolved paths to the external programs ffrefine drives.

    ``orca_libdir`` is the directory holding the ORCA shared libraries; ORCA
    shared builds will not start unless it is on LD_LIBRARY_PATH.
    """

    gmx: str | None = None
    xtb: str | None = None
    orca: str | None = None
    orca_vpot: str | None = None
    orca_libdir: str | None = None

    def require(self, name: str) -> str:
        """Return a binary path or raise a clear error naming the override flag."""
        value = getattr(self, name, None)
        if not value:
            raise FFRefineError(
                f"{name} was not found on PATH. Install it, or pass "
                f"--{name.replace('_', '-')}-bin /path/to/{name}, or set "
                f'"binaries": {{"{name}": "..."}} in the config file.'
            )
        return value


def discover_binaries(overrides: dict[str, str] | None = None) -> Binaries:
    """Locate external binaries via shutil.which, honouring explicit overrides.

    Args:
        overrides: mapping of binary name to an explicit path; wins over PATH.

    Returns:
        Resolved binaries.  Missing entries are left as None; call
        :meth:`Binaries.require` at the point of use so the error names the
        program actually needed.
    """
    overrides = dict(overrides or {})
    resolved = Binaries()
    for name in ("gmx", "xtb", "orca", "orca_vpot"):
        explicit = overrides.get(name)
        setattr(resolved, name, explicit or shutil.which(name))
    if resolved.gmx is None and not overrides.get("gmx"):
        resolved.gmx = shutil.which("gmx_mpi")
    if resolved.orca:
        resolved.orca_libdir = str(Path(resolved.orca).resolve().parent)
        if not resolved.orca_vpot:
            candidate = Path(resolved.orca).resolve().parent / "orca_vpot"
            if candidate.exists():
                resolved.orca_vpot = str(candidate)
    return resolved


def orca_env(binaries: Binaries) -> dict[str, str]:
    """Environment additions needed to run an ORCA shared build."""
    if not binaries.orca_libdir:
        return {}
    import os

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    joined = f"{binaries.orca_libdir}:{existing}" if existing else binaries.orca_libdir
    return {"LD_LIBRARY_PATH": joined}


def load_config(path: str | Path | None, cli_overrides: dict) -> dict:
    """Merge a JSON config file with CLI values; CLI wins (spec section 3).

    CLI values that are None mean "not supplied on the command line" and never
    override the file.

    Args:
        path: the ``--config`` file, or None.
        cli_overrides: values from the command line; None means "not given".

    Returns:
        The merged settings.

    Raises:
        FFRefineError: the file is missing, is not valid JSON, or is valid
            JSON that is not an object. Ruling R29: bad input never produces a
            traceback, and a top-level JSON array used to reach ``dict.update``
            and escape ``main``'s handler as a bare ``TypeError``.
    """
    merged: dict = {}
    if path is not None:
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise FFRefineError(f"could not read config file {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise FFRefineError(
                f"config file {path} must contain a JSON object mapping option names to "
                f'values, e.g. {{"outdir": "run1", "emax": 40}}; found a '
                f"{type(data).__name__}."
            )
        merged.update(data)
    for key, value in cli_overrides.items():
        if value is not None:
            merged[key] = value
    return merged
