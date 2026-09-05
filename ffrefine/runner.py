"""Logged execution of external binaries (spec section 8).

Every external command is written to ``<log_stem>.cmd`` with its full argument
vector, and its streams to ``<log_stem>.stdout`` / ``<log_stem>.stderr`` inside
the stage directory.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
from pathlib import Path

from .errors import ExternalToolError

log = logging.getLogger("ffrefine")

TAIL_LINES = 30
"""How many lines of captured output to quote in an error message."""


def run_command(
    argv: list[str],
    workdir: str | Path,
    log_stem: str,
    extra_env: dict[str, str] | None = None,
    check: bool = True,
    stdin: str | None = None,
) -> subprocess.CompletedProcess:
    """Run an external command, logging it and capturing its output.

    Args:
        argv: full argument vector; argv[0] is the executable.
        workdir: directory to run in and to write log files into.
        log_stem: base name for the .cmd/.stdout/.stderr log files.
        extra_env: environment variables layered on top of os.environ.
        check: raise on a nonzero exit status.
        stdin: text piped to the process's stdin (e.g. an interactive tool's
            menu selection); logged alongside argv so the invocation stays
            reproducible from the .cmd file alone.

    Returns:
        The completed process, with ``stdout`` and ``stderr`` populated.

    Raises:
        ExternalToolError: on nonzero exit when ``check`` is set, or when the
            executable cannot be run at all.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    printable = " ".join(shlex.quote(a) for a in argv)
    if stdin is None:
        log.info("running in %s: %s", workdir, printable)
        (workdir / f"{log_stem}.cmd").write_text(printable + "\n")
    else:
        log.info("running in %s: %s <<< %r", workdir, printable, stdin)
        (workdir / f"{log_stem}.cmd").write_text(f"{printable}\n<<< {stdin!r}\n")

    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)

    try:
        proc = subprocess.run(
            argv, cwd=str(workdir), env=env, capture_output=True, text=True, input=stdin
        )
    except OSError as exc:
        raise ExternalToolError(f"could not execute {argv[0]!r}: {exc}") from exc

    (workdir / f"{log_stem}.stdout").write_text(proc.stdout or "")
    (workdir / f"{log_stem}.stderr").write_text(proc.stderr or "")

    if check and proc.returncode != 0:
        combined = ((proc.stdout or "") + (proc.stderr or "")).splitlines()
        tail = "\n".join(combined[-TAIL_LINES:])
        raise ExternalToolError(
            f"command failed with exit code {proc.returncode}:\n"
            f"  {printable}\n"
            f"  workdir: {workdir}\n"
            f"  logs: {log_stem}.stdout, {log_stem}.stderr\n"
            f"--- last {TAIL_LINES} lines of output ---\n{tail}"
        )
    return proc
