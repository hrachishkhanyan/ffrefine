import sys

import pytest
from ffrefine import config, runner
from ffrefine.errors import ExternalToolError


def test_run_command_captures_stdout(tmp_path):
    r = runner.run_command([sys.executable, "-c", "print('hello')"], tmp_path, "probe")
    assert r.returncode == 0
    assert "hello" in (tmp_path / "probe.stdout").read_text()


def test_run_command_logs_full_command_line(tmp_path):
    runner.run_command([sys.executable, "-c", "print(1)"], tmp_path, "probe")
    logged = (tmp_path / "probe.cmd").read_text()
    assert sys.executable in logged
    assert "print(1)" in logged


def test_run_command_raises_with_output_tail(tmp_path):
    code = "import sys; sys.stderr.write('BOOM detail line\\n'); sys.exit(3)"
    with pytest.raises(ExternalToolError, match="BOOM detail line"):
        runner.run_command([sys.executable, "-c", code], tmp_path, "fail")


def test_run_command_check_false_returns_nonzero(tmp_path):
    r = runner.run_command(
        [sys.executable, "-c", "import sys; sys.exit(3)"], tmp_path, "soft", check=False
    )
    assert r.returncode == 3


def test_run_command_applies_extra_env(tmp_path):
    code = "import os; print(os.environ.get('FFREFINE_PROBE', 'unset'))"
    r = runner.run_command(
        [sys.executable, "-c", code], tmp_path, "env", extra_env={"FFREFINE_PROBE": "set-value"}
    )
    assert "set-value" in r.stdout


def test_discover_binaries_honours_override(tmp_path):
    fake = tmp_path / "mygmx"
    fake.write_text("")
    b = config.discover_binaries({"gmx": str(fake)})
    assert b.gmx == str(fake)


def test_discover_binaries_derives_orca_libdir(tmp_path):
    orca = tmp_path / "orcadir" / "orca"
    orca.parent.mkdir()
    orca.write_text("")
    b = config.discover_binaries({"orca": str(orca)})
    assert b.orca_libdir == str(orca.parent)


def test_load_config_cli_overrides_file(tmp_path):
    cfg_file = tmp_path / "c.json"
    cfg_file.write_text('{"step": 15, "emax": 40.0}')
    merged = config.load_config(cfg_file, {"step": 30})
    assert merged["step"] == 30
    assert merged["emax"] == 40.0


def test_load_config_ignores_none_cli_values(tmp_path):
    cfg_file = tmp_path / "c.json"
    cfg_file.write_text('{"step": 15}')
    merged = config.load_config(cfg_file, {"step": None})
    assert merged["step"] == 15
