import json
import logging

import pytest
from ffrefine import stages
from ffrefine.errors import ValidationError

# --- STAGE_DIRS / stage_dir --------------------------------------------------


def test_stage_dirs_match_spec_layout():
    assert stages.STAGE_DIRS["input"] == "00_input"
    assert stages.STAGE_DIRS["qm_scan"] == "01_qm_scan"
    assert stages.STAGE_DIRS["mm_scan"] == "02_mm_scan"
    assert stages.STAGE_DIRS["fit"] == "03_fit"
    assert stages.STAGE_DIRS["charges"] == "04_charges"
    assert stages.STAGE_DIRS["final"] == "final"


def test_stage_dir_is_created(tmp_path):
    d = stages.stage_dir(tmp_path, "fit")
    assert d.exists()
    assert d.name == "03_fit"


def test_stage_dir_rejects_unknown_stage(tmp_path):
    with pytest.raises(ValidationError):
        stages.stage_dir(tmp_path, "bogus")


# --- hash_file ----------------------------------------------------------------


def test_hash_file_is_stable(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("content")
    assert stages.hash_file(p) == stages.hash_file(p)


def test_hash_file_changes_with_content(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("one")
    first = stages.hash_file(p)
    p.write_text("two")
    assert stages.hash_file(p) != first


# --- hash_inputs ----------------------------------------------------------------


def test_hash_inputs_is_stable_across_calls(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("x")
    assert stages.hash_inputs([p], {"step": 15}) == stages.hash_inputs([p], {"step": 15})


def test_hash_inputs_changes_with_file_content(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("one")
    first = stages.hash_inputs([p], {"step": 15})
    p.write_text("two")
    assert stages.hash_inputs([p], {"step": 15}) != first


def test_hash_inputs_depends_on_extra(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("x")
    a = stages.hash_inputs([p], {"step": 15})
    b = stages.hash_inputs([p], {"step": 30})
    assert a != b


def test_hash_inputs_is_order_independent_for_extra(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("x")
    a = stages.hash_inputs([p], {"step": 15, "emax": 40})
    b = stages.hash_inputs([p], {"emax": 40, "step": 15})
    assert a == b


def test_hash_inputs_changes_with_different_path_set(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("aaa")
    b = tmp_path / "b.txt"
    b.write_text("bbb")
    only_a = stages.hash_inputs([a], {})
    both = stages.hash_inputs([a, b], {})
    only_b = stages.hash_inputs([b], {})
    assert only_a != both
    assert only_a != only_b


def test_hash_inputs_defaults_extra_to_empty(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("x")
    # Calling with no extra at all must not raise, and must be stable.
    assert stages.hash_inputs([p]) == stages.hash_inputs([p], {})


# --- stage_is_current -----------------------------------------------------------


def test_stage_not_current_before_any_run(tmp_path):
    # No outdir, no stage dir, no manifest at all.
    assert not stages.stage_is_current(tmp_path, "fit", "abc123")


def test_stage_not_current_when_outdir_does_not_exist(tmp_path):
    missing = tmp_path / "does" / "not" / "exist"
    assert not stages.stage_is_current(missing, "fit", "abc123")


def test_stage_not_current_when_stage_dir_exists_but_no_manifest(tmp_path):
    stages.stage_dir(tmp_path, "fit")
    assert not stages.stage_is_current(tmp_path, "fit", "abc123")


def test_stage_current_after_matching_manifest(tmp_path):
    stages.stage_dir(tmp_path, "fit")
    stages.write_manifest(tmp_path, "fit", "abc123", {"note": "done"})
    assert stages.stage_is_current(tmp_path, "fit", "abc123")


def test_stage_not_current_when_inputs_changed(tmp_path):
    stages.stage_dir(tmp_path, "fit")
    stages.write_manifest(tmp_path, "fit", "abc123", {})
    assert not stages.stage_is_current(tmp_path, "fit", "different")


def test_stage_not_current_when_manifest_is_corrupt(tmp_path):
    d = stages.stage_dir(tmp_path, "fit")
    # Simulate a run killed mid-write: a truncated, non-JSON manifest.
    (d / "manifest.json").write_text('{"stage": "fit", "input_hash": "abc123"')
    assert not stages.stage_is_current(tmp_path, "fit", "abc123")


def test_stage_not_current_when_manifest_is_empty(tmp_path):
    d = stages.stage_dir(tmp_path, "fit")
    (d / "manifest.json").write_text("")
    assert not stages.stage_is_current(tmp_path, "fit", "abc123")


def test_stage_not_current_when_manifest_is_not_an_object(tmp_path):
    d = stages.stage_dir(tmp_path, "fit")
    # Syntactically valid JSON, but not the expected shape.
    (d / "manifest.json").write_text("42")
    assert not stages.stage_is_current(tmp_path, "fit", "abc123")


def test_stage_is_current_rejects_unknown_stage(tmp_path):
    with pytest.raises(ValidationError):
        stages.stage_is_current(tmp_path, "bogus", "abc123")


# --- write_manifest / manifest format --------------------------------------------


def test_manifest_is_readable_json(tmp_path):
    stages.stage_dir(tmp_path, "fit")
    stages.write_manifest(tmp_path, "fit", "h", {"multiplicities": [1, 3]})
    data = json.loads((tmp_path / "03_fit" / "manifest.json").read_text())
    assert data["input_hash"] == "h"
    assert data["metadata"]["multiplicities"] == [1, 3]


def test_write_manifest_creates_stage_dir(tmp_path):
    # write_manifest must not require the caller to have called stage_dir first.
    stages.write_manifest(tmp_path, "fit", "h", {})
    assert (tmp_path / "03_fit" / "manifest.json").exists()


def test_write_manifest_rejects_unknown_stage(tmp_path):
    with pytest.raises(ValidationError):
        stages.write_manifest(tmp_path, "bogus", "h", {})


# --- setup_logging -----------------------------------------------------------


def test_setup_logging_creates_run_log(tmp_path):
    stages.setup_logging(tmp_path)
    logging.getLogger("ffrefine").info("hello from the test")
    assert "hello from the test" in (tmp_path / "run.log").read_text()


def test_setup_logging_is_idempotent(tmp_path):
    stages.setup_logging(tmp_path)
    stages.setup_logging(tmp_path)
    logging.getLogger("ffrefine").info("only once please")
    contents = (tmp_path / "run.log").read_text()
    assert contents.count("only once please") == 1


def test_setup_logging_verbose_enables_debug(tmp_path):
    stages.setup_logging(tmp_path, verbose=True)
    logging.getLogger("ffrefine").debug("debug detail")
    assert "debug detail" in (tmp_path / "run.log").read_text()
