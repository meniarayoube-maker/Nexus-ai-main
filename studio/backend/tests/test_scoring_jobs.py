# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Background per-example scoring jobs (Pilot, user-triggered only).

Proves the job lifecycle without torch/GPU: acceptance guards, duplicate and
VRAM-busy refusals (as values, never silent), error surfacing, atomic file
writes, row-cap enforcement, and terminal-job pruning. The heavy scorers run
with injected fakes; the real GPU path is covered by the documented manual
run, never by unit tests.
"""

from __future__ import annotations

import json
import os
import time

from core.training.scoring_jobs import (
    atomic_write_jsonl,
    check_row_cap,
    get_scoring_job,
    start_scoring_job,
)


def _unique_run(prefix = "run"):
    _unique_run.counter += 1
    return f"{prefix}-{_unique_run.counter}-{time.time_ns()}"


_unique_run.counter = 0


def _write_run_config(output_dir, **overrides):
    config = {
        "model_name": "m",
        "dataset_source": "upload",
        "local_datasets": ["data.jsonl"],
        "format_type": "auto",
        "max_seq_length": 2048,
    }
    config.update(overrides)
    with open(os.path.join(output_dir, "run-config.json"), "w",
              encoding = "utf-8") as handle:
        json.dump(config, handle)


def _wait_for_status(run_id, terminal=("done", "error"), timeout_s = 10.0):
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        last = get_scoring_job(run_id)
        if last is not None and last["status"] in terminal:
            return last
        time.sleep(0.05)
    raise AssertionError(f"job {run_id} never reached {terminal}: {last}")


def test_acceptance_lifecycle_with_fake_scorer(tmp_path):
    run_id = _unique_run()
    output_dir = str(tmp_path)
    _write_run_config(output_dir)
    calls = []

    def fake_scorer(output_path, run_config, progress_cb):
        assert output_path == output_dir
        assert run_config["model_name"] == "m"
        progress_cb(1, 2)
        progress_cb(2, 2)
        calls.append(True)
        return os.path.join(output_path, "per_example_loss.jsonl")

    outcome = start_scoring_job(
        run_id, output_dir,
        training_active_check = lambda: False,
        run_scoring_fn = fake_scorer,
    )
    assert outcome["accepted"] is True
    final = _wait_for_status(run_id)
    assert final["status"] == "done"
    assert calls == [True]
    assert "per_example_loss.jsonl" in final["message"]


def test_duplicate_job_refused_while_running(tmp_path):
    import threading as _threading

    run_id = _unique_run()
    output_dir = str(tmp_path)
    _write_run_config(output_dir)
    release = _threading.Event()

    def slow_scorer(output_path, run_config, progress_cb):
        assert release.wait(timeout = 10.0)
        return os.path.join(output_path, "per_example_loss.jsonl")

    first = start_scoring_job(
        run_id, output_dir,
        training_active_check = lambda: False,
        run_scoring_fn = slow_scorer,
    )
    assert first["accepted"] is True
    try:
        second = start_scoring_job(
            run_id, output_dir,
            training_active_check = lambda: False,
            run_scoring_fn = slow_scorer,
        )
        assert second["accepted"] is False
        assert second["status"] == "running"
    finally:
        release.set()
    final = _wait_for_status(run_id)
    assert final["status"] == "done"


def test_training_active_refuses_without_spawning(tmp_path):
    run_id = _unique_run()
    output_dir = str(tmp_path)
    _write_run_config(output_dir)
    spawned = []

    def fake_scorer(output_path, run_config, progress_cb):
        spawned.append(True)
        return "x"

    outcome = start_scoring_job(
        run_id, output_dir,
        training_active_check = lambda: True,
        run_scoring_fn = fake_scorer,
    )
    assert outcome["accepted"] is False
    assert "GPU" in outcome["message"] or "busy" in outcome["message"]
    assert spawned == []
    assert get_scoring_job(run_id) is None


def test_missing_output_dir_and_config_refused(tmp_path):
    outcome = start_scoring_job(
        "nope-missing", str(tmp_path / "absent"),
        training_active_check = lambda: False,
    )
    assert outcome["accepted"] is False

    run_id = _unique_run()
    # output dir exists but no run-config.json -> clear synchronous refusal.
    outcome = start_scoring_job(
        run_id, str(tmp_path),
        training_active_check = lambda: False,
    )
    assert outcome["accepted"] is False
    assert "run-config" in outcome["message"]


def test_scorer_error_surfaces_as_status(tmp_path):
    run_id = _unique_run()
    output_dir = str(tmp_path)
    _write_run_config(output_dir)

    def boom(output_path, run_config, progress_cb):
        raise RuntimeError("CUDA out of memory (simulated)")

    outcome = start_scoring_job(
        run_id, output_dir,
        training_active_check = lambda: False,
        run_scoring_fn = boom,
    )
    assert outcome["accepted"] is True
    final = _wait_for_status(run_id)
    assert final["status"] == "error"
    assert "CUDA out of memory" in final["message"]


def test_row_cap_and_atomic_write(tmp_path):
    import pytest as _pytest

    with _pytest.raises(ValueError, match = "over the scoring cap"):
        check_row_cap(20001)
    check_row_cap(20000)

    records = [{"row_id": 0, "loss": 1.5}, {"row_id": 1, "loss": None}]
    final_path = os.path.join(str(tmp_path), "per_example_loss.jsonl")
    assert atomic_write_jsonl(records, final_path) == final_path
    assert not os.path.exists(final_path + ".tmp")
    with open(final_path, encoding = "utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0])["loss"] == 1.5


def test_terminal_jobs_pruned_after_ttl(tmp_path):
    import core.training.scoring_jobs as _jobs

    run_id = _unique_run()
    _jobs._JOBS[run_id] = {
        "status": "done", "done": 1, "total": 1, "message": "old",
        "started_at": 0.0, "finished_at": 0.0,
    }
    assert get_scoring_job(run_id) is None
    assert run_id not in _jobs._JOBS


def test_scoring_source_upload_local_and_missing(tmp_path):
    from core.training.scoring_jobs import _resolve_scoring_source

    dataset_file = tmp_path / "data.jsonl"
    dataset_file.write_text('{"a": 1}\n', encoding = "utf-8")
    kind, spec = _resolve_scoring_source({
        "dataset_source": "upload", "local_datasets": [str(dataset_file)],
    })
    assert kind == "upload"
    assert spec["files"] == [str(dataset_file)]

    import pytest as _pytest

    with _pytest.raises(ValueError, match = "looked in"):
        _resolve_scoring_source({
            "dataset_source": "upload",
            "local_datasets": ["no-such-file.jsonl"],
        })


def test_scoring_source_huggingface_and_rejections():
    import pytest as _pytest

    from core.training.scoring_jobs import _resolve_scoring_source

    kind, spec = _resolve_scoring_source({
        "dataset_source": "huggingface", "hf_dataset": "org/ds",
        "subset": "cfg", "train_split": "validation", "hf_token": "tok",
    })
    assert kind == "huggingface"
    assert spec == {"hf_dataset": "org/ds", "subset": "cfg",
                    "train_split": "validation", "hf_token": "tok"}
    # Fallback when the source flag is missing but an id exists.
    kind, spec = _resolve_scoring_source({"hf_dataset": "org/ds"})
    assert kind == "huggingface" and spec["train_split"] == "train"
    with _pytest.raises(ValueError, match = "no.*hf_dataset id"):
        _resolve_scoring_source({"dataset_source": "huggingface"})
    with _pytest.raises(ValueError, match = "supports local/upload"):
        _resolve_scoring_source({"dataset_source": "s3"})
    with _pytest.raises(ValueError, match = "no usable dataset"):
        _resolve_scoring_source({})


def test_nested_run_config_shape_resolves_dataset(tmp_path):
    # Regression: real run-config.json is nested ({saved_at, config}); reading
    # the top level yielded nothing usable and every run was refused.
    from core.training.scoring_jobs import _read_run_config, _resolve_scoring_source

    dataset_file = tmp_path / "data.jsonl"
    dataset_file.write_text('{"a": 1}\n', encoding = "utf-8")
    inner = {
        "dataset_source": "upload",
        "local_datasets": [str(dataset_file)],
        "model_name": "m",
    }
    (tmp_path / "run-config.json").write_text(
        json.dumps({"saved_at": "t", "config": inner}), encoding = "utf-8"
    )
    loaded = _read_run_config(str(tmp_path))
    assert loaded["model_name"] == "m"
    kind, spec = _resolve_scoring_source(loaded)
    assert kind == "upload"
    assert spec["files"] == [str(dataset_file)]


def test_nested_hf_run_config_resolves_dataset():
    from core.training.scoring_jobs import _resolve_scoring_source

    kind, spec = _resolve_scoring_source({
        "dataset_source": "huggingface",
        "hf_dataset": "org/ds",
        "subset": None,
        "train_split": "train",
    })
    assert kind == "huggingface"
    assert spec["hf_dataset"] == "org/ds"


def test_garbage_run_config_names_its_keys(tmp_path):
    from core.training.scoring_jobs import _read_run_config

    import pytest as _pytest

    (tmp_path / "run-config.json").write_text(
        '{"foo": 1}', encoding = "utf-8"
    )
    with _pytest.raises(ValueError, match = "foo"):
        _read_run_config(str(tmp_path))


def test_jsonl_reader_names_file_line_and_preview(tmp_path):
    from core.training.scoring_jobs import read_jsonl_rows

    import pytest as _pytest

    good = tmp_path / "good.jsonl"
    good.write_text('{"a": 1}\n\n{"b": 2}\n', encoding = "utf-8")
    assert read_jsonl_rows(str(good)) == [{"a": 1}, {"b": 2}]

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"a": 1}\nNOT-JSON\n', encoding = "utf-8")
    with _pytest.raises(ValueError, match = "bad.jsonl.*line 2"):
        read_jsonl_rows(str(bad))

    not_obj = tmp_path / "line.jsonl"
    not_obj.write_text('[1, 2]\n', encoding = "utf-8")
    with _pytest.raises(ValueError, match = "must be a JSON object"):
        read_jsonl_rows(str(not_obj))


def test_json_array_files_match_training_loader(tmp_path):
    # The training pipeline (datasets json loader) accepts top-level arrays;
    # scoring must accept exactly the same files it trained on.
    from core.training.scoring_jobs import read_jsonl_rows

    import pytest as _pytest

    pretty = tmp_path / "pretty.jsonl"
    pretty.write_text(
        '[\n  {"a": 1,\n   "messages": [{"content": "hi"}]},\n  {"a": 2}\n]\n',
        encoding = "utf-8",
    )
    rows = read_jsonl_rows(str(pretty))
    assert [r["a"] for r in rows] == [1, 2]
    assert rows[0]["messages"] == [{"content": "hi"}]

    single_line = tmp_path / "flat.jsonl"
    single_line.write_text('[{"a": 1}, {"a": 2}]', encoding = "utf-8")
    assert [r["a"] for r in read_jsonl_rows(str(single_line))] == [1, 2]

    # A single top-level object is valid JSONL content (one row).
    obj_file = tmp_path / "obj.jsonl"
    obj_file.write_text('{"a": 1}', encoding = "utf-8")
    assert read_jsonl_rows(str(obj_file)) == [{"a": 1}]

    bad_item = tmp_path / "mixed.jsonl"
    bad_item.write_text('[{"a": 1}, 42]', encoding = "utf-8")
    with _pytest.raises(ValueError, match = "item 1 must be a JSON object"):
        read_jsonl_rows(str(bad_item))

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n  \n", encoding = "utf-8")
    assert read_jsonl_rows(str(empty)) == []


def test_read_per_example_records_tolerates_absent_files(tmp_path):
    from core.training.offline_scoring import read_per_example_records

    assert read_per_example_records(None) == (False, [], 0)
    assert read_per_example_records(str(tmp_path)) == (False, [], 0)
