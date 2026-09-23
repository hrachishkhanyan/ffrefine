"""Every warning a stage raises reaches its payload (milestone 6b design, section 6).

Refinement 1: warnings are collected at the logging layer, so a module's own
``log.warning`` (GROMACS's echoed warnings, the self-check's) is kept as well
as the CLI's ``_warn``.
"""

from __future__ import annotations

import logging

from ffrefine import cli
from ffrefine.stages import WarningCollector, setup_logging

log = logging.getLogger("ffrefine")


def test_a_collector_keeps_module_and_cli_warnings_in_order():
    with cli._collect_warnings() as got:
        log.warning("from a module")
        cli._warn("from the cli")
        log.info("not a warning")
    assert got == ["from a module", "from the cli"]


def test_setup_logging_leaves_a_collector_attached(tmp_path):
    with cli._collect_warnings() as got:
        setup_logging(tmp_path)
        cli._warn("after setup_logging")
    assert got == ["after setup_logging"]
    assert "after setup_logging" in (tmp_path / "run.log").read_text()
    assert not any(isinstance(h, WarningCollector) for h in log.handlers)


def test_current_warnings_reads_the_innermost_collector():
    assert cli._current_warnings() == []
    with cli._collect_warnings():
        cli._warn("outer")
        with cli._collect_warnings() as inner:
            cli._warn("inner")
            assert cli._current_warnings() == ["inner"]
        assert inner == ["inner"]


def test_a_warning_with_no_log_handler_is_printed_and_collected(capsys):
    cli._reset_logging()
    with cli._collect_warnings() as got:
        cli._warn("before logging exists")
    assert got == ["before logging exists"]
    assert "warning: before logging exists" in capsys.readouterr().err
