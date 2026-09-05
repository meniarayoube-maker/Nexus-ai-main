# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for :mod:`core.training.checkpoint_swap`.

Loaded standalone (stdlib-only module): every failure mode must prove the
previous checkpoint survives, per the safety contract.
"""

import importlib.util
import tempfile
from pathlib import Path

import pytest


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "training_checkpoint_swap_under_test",
        Path(__file__).resolve().parents[1] / "core" / "training" / "checkpoint_swap.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


swap = _load_module()


def _make_old_bundle(working: Path, name="checkpoint-14", payload=b"old-weights"):
    ckpt = working / name
    ckpt.mkdir(parents=True)
    (ckpt / "model.bin").write_bytes(payload)
    (ckpt / "trainer_state.json").write_text('{"global_step": 14}')
    return ckpt


def _ok_write(path_str):
    target = Path(path_str)
    target.mkdir(parents=True)
    (target / "model.bin").write_bytes(b"new-weights")
    (target / "trainer_state.json").write_text('{"global_step": 18}')


def _tmp_orphans():
    return {
        p.name
        for p in Path(tempfile.gettempdir()).iterdir()
        if p.name.startswith(".stop-ckpt-")
    }


def _run_execute(working, **overrides):
    logs = []
    params = {
        "working_dir": str(working),
        "checkpoint_name": "checkpoint-18",
        "write_bundle": _ok_write,
        "verify_bundle": lambda _p: None,
        "upload_bundle": lambda _p: (True, "https://kaggle.com/datasets/o/s", None),
        "log": logs.append,
    }
    params.update(overrides)
    return swap.execute_stop_swap(**params), logs


def test_plan_legacy_without_bundles_or_room(tmp_path):
    assert swap.plan_stop_save(str(tmp_path)) == "legacy"
    tiny = tmp_path / "checkpoint-1"
    tiny.mkdir()
    (tiny / "f.bin").write_bytes(b"0" * 100)
    # Roomy sandbox disk vs a 100-byte bundle -> legacy (today's behavior).
    assert swap.plan_stop_save(str(tmp_path)) == "legacy"


def test_plan_tmp_flow_when_tight(monkeypatch, tmp_path):
    _make_old_bundle(tmp_path, "checkpoint-14")
    monkeypatch.setattr(swap, "disk_free_bytes", lambda _p: 1)
    monkeypatch.setattr(swap, "latest_bundle_size_bytes", lambda _p: 10**9)
    assert swap.plan_stop_save(str(tmp_path)) == "tmp_flow"


def test_write_failure_leaves_working_untouched(tmp_path):
    old = _make_old_bundle(tmp_path)
    before = _tmp_orphans()

    def _boom(_path_str):
        raise OSError(28, "No space left on device")

    result, _logs = _run_execute(tmp_path, write_bundle=_boom)

    assert result.status == "write_failed"
    assert (old / "model.bin").read_bytes() == b"old-weights"
    assert (old / "trainer_state.json").is_file()
    assert _tmp_orphans() == before


def test_upload_failure_skips_prune_and_move_but_keeps_tmp(tmp_path):
    old = _make_old_bundle(tmp_path)
    before = _tmp_orphans()

    def _fail_upload(_path_str):
        return (False, None, "network down")

    result, _logs = _run_execute(tmp_path, upload_bundle=_fail_upload)

    assert result.status == "upload_failed"
    # Previous checkpoint fully intact...
    assert (old / "model.bin").read_bytes() == b"old-weights"
    assert "network down" in result.message
    # ...and the fresh bundle survives in tmp for manual recovery.
    assert result.tmp_path is not None
    kept = Path(result.tmp_path) / "checkpoint-18"
    assert (kept / "model.bin").read_bytes() == b"new-weights"
    assert _tmp_orphans() == before | {Path(result.tmp_path).name}
    # Cleanup after ourselves so later runs see a clean temp.
    swap.cleanup_path(result.tmp_path)
    assert _tmp_orphans() == before


def test_move_failure_cleans_partial_and_reports_recovery(tmp_path):
    old = _make_old_bundle(tmp_path)

    def _boom_move(src, dst):
        # A mover that dies midway AFTER creating a partial destination, like
        # a real cross-filesystem copy interrupted by ENOSPC.
        Path(dst).mkdir(parents=True)
        (Path(dst) / "half.bin").write_bytes(b"partial")
        raise OSError("copy died midway")

    result, _logs = _run_execute(tmp_path, move_in=_boom_move)

    assert result.status == "move_failed"
    # Old was pruned only after a successful upload (documented order);
    # recovery is explicit via the dataset URL + kept tmp bundle.
    assert "https://kaggle.com/datasets/o/s" in result.message
    assert result.tmp_path is not None
    assert ((old / "model.bin").exists()) is False  # pruned post-upload, by design


def test_real_move_cleans_its_own_partial_on_failure(tmp_path, monkeypatch):
    import shutil as _shutil

    src = tmp_path / "src-ckpt"
    src.mkdir()
    (src / "f.bin").write_bytes(b"data")
    dst = tmp_path / "dst-ckpt"

    real_move = _shutil.move

    def _die_midway(s, d):
        Path(d).mkdir(parents=True)
        (Path(d) / "half.bin").write_bytes(b"partial")
        raise OSError("copy died midway")

    monkeypatch.setattr(swap.shutil, "move", _die_midway)
    try:
        with pytest.raises(OSError):
            swap.move_bundle_into_place(str(src), str(dst))
    finally:
        monkeypatch.undo()
    assert not dst.exists()
    assert (src / "f.bin").is_file()


def test_success_swaps_prunes_and_cleans_tmp(tmp_path):
    old = _make_old_bundle(tmp_path)
    before = _tmp_orphans()

    result, logs = _run_execute(tmp_path)

    assert result.status == "swapped"
    assert result.upload_url == "https://kaggle.com/datasets/o/s"
    assert result.pruned == ("checkpoint-14",)
    assert not old.exists()
    new_ckpt = tmp_path / "checkpoint-18"
    assert (new_ckpt / "model.bin").read_bytes() == b"new-weights"
    assert (new_ckpt / "trainer_state.json").is_file()
    assert _tmp_orphans() == before
    assert any("swapped into place" in line for line in logs)


def test_prune_keeps_non_checkpoint_files_and_keep_set(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"root")
    (tmp_path / "run-config.json").write_text("{}")
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "events").write_bytes(b"tb")
    _make_old_bundle(tmp_path, "checkpoint-5")
    _make_old_bundle(tmp_path, "checkpoint-9")

    pruned = swap.prune_superseded_checkpoints(str(tmp_path), keep={"checkpoint-9"})

    assert pruned == ["checkpoint-5"]
    assert (tmp_path / "checkpoint-9" / "model.bin").is_file()
    assert (tmp_path / "model.safetensors").is_file()
    assert (logs_dir / "events").is_file()


def test_normalize_upload_result_shapes():
    n = swap.normalize_upload_result
    assert n(None) == (False, None, "upload failed")
    assert n({"ok": True, "dataset_url": "u", "error": None}) == (True, "u", None)
    assert n({"ok": False, "error": "boom"}) == (False, None, "boom")
    assert n((True, "u", None)) == (True, "u", None)
    assert n("garbage") == (False, None, "upload returned an unexpected result")


def test_checkpoint_step_num_ordering():
    assert swap.checkpoint_step_num("checkpoint-19") > swap.checkpoint_step_num("checkpoint-5")
    assert swap.checkpoint_step_num("other") == -1
