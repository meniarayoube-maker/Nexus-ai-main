# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Batch composition tracking (Pilot): the id column must never reach the model.

Covers the sidecar lifecycle (micro-batch -> optimizer step -> append across
resume), the eval/train gate, the packing/streaming/online refusals, and the
loss join. Heavy training deps are NOT needed: the module under test is
stdlib-only and collaborators are fakes.
"""

from __future__ import annotations

import os

import pytest

from core.training.composition_log import (
    ROW_ID_COLUMN,
    CompositionRecorder,
    RowIdRecordingCollator,
    attribute_steps,
    ensure_row_ids,
    join_with_losses,
    present_step_attribution,
    prune_columns_for_tracking,
    read_composition_records,
    read_run_composition,
    stamp_row_ids,
    validate_tracking_prerequisites,
)


class _FakeBaseCollator:
    """Stands in for DataCollatorForLanguageModeling: echoes received keys."""

    def __init__(self):
        self.seen_keys = []

    def __call__(self, features):
        self.seen_keys.append([sorted(f.keys()) if isinstance(f, dict) else None for f in features])
        return {"batched": len(features)}


def _recorder(tmp_path):
    return CompositionRecorder(str(tmp_path / "composition.jsonl"))


def test_row_id_never_reaches_model(tmp_path):
    base = _FakeBaseCollator()
    recorder = _recorder(tmp_path)
    collator = RowIdRecordingCollator(base, recorder)
    try:
        out = collator(
            [
                {"input_ids": [1, 2], ROW_ID_COLUMN: 7},
                {"input_ids": [3], ROW_ID_COLUMN: 9},
            ]
        )
        record = recorder.finalize_optimizer_step(3)
    finally:
        recorder.close()
    assert out == {"batched": 2}
    # The model-facing batch contains no trace of the tracking column...
    for keys in base.seen_keys[0]:
        assert ROW_ID_COLUMN not in keys
        assert keys == ["input_ids"]
    # ...while the recorder captured both ids in order.
    assert record["row_ids"] == [7, 9]
    assert record["step"] == 3


def test_batches_without_id_column_pass_through_untouched(tmp_path):
    base = _FakeBaseCollator()
    recorder = _recorder(tmp_path)
    collator = RowIdRecordingCollator(base, recorder)
    try:
        out = collator([{"input_ids": [1, 2]}, {"input_ids": [3]}])
        record = recorder.finalize_optimizer_step(1)
    finally:
        recorder.close()
    assert out == {"batched": 2}
    # Eval-style batches (never stamped) record nothing and keep every key.
    assert record["micro_batches"] == []
    assert record["row_ids"] == []
    assert base.seen_keys[0] == [["input_ids"], ["input_ids"]]


def test_grad_accum_micro_batches_aggregate_to_one_optimizer_step(tmp_path):
    recorder = _recorder(tmp_path)
    try:
        collator = RowIdRecordingCollator(lambda feats: feats, recorder)
        collator([{"input_ids": [1], ROW_ID_COLUMN: 0}, {"input_ids": [2], ROW_ID_COLUMN: 1}])
        collator([{"input_ids": [3], ROW_ID_COLUMN: 2}, {"input_ids": [4], ROW_ID_COLUMN: 3}])
        collator([{"input_ids": [5], ROW_ID_COLUMN: 4}])
        record = recorder.finalize_optimizer_step(7)
    finally:
        recorder.close()
    assert record["step"] == 7
    assert record["micro_batches"] == [[0, 1], [2, 3], [4]]
    assert record["row_ids"] == [0, 1, 2, 3, 4]
    assert record["num_micro_batches"] == 3
    assert record["num_rows"] == 5


def test_recording_needs_no_capture_gate_batches_collate_before_step_begin(tmp_path):
    # Regression for the empty-sidecar run: in the HF loop the batch for step
    # N is collated BEFORE on_step_begin(N) fires, so any begin/end gate is
    # always one micro-batch late and records nothing. Recording must not
    # depend on callback timing at all.
    recorder = _recorder(tmp_path)
    try:
        collator = RowIdRecordingCollator(lambda feats: feats, recorder)
        # No callbacks fire before the first collator call — ids still land.
        collator([{"input_ids": [1], ROW_ID_COLUMN: 0}])
        record = recorder.finalize_optimizer_step(1)
    finally:
        recorder.close()
    assert record["row_ids"] == [0]


def test_pretrain_pulls_are_cleared_before_step_one(tmp_path):
    # Regression for the polluted step-1 record: _preflight_first_batch pulls
    # real batches through the wrapped collator before train() starts (then
    # train() re-iterates from row 0, so the same rows repeat). Clearing at
    # train begin keeps step 1 exact.
    recorder = _recorder(tmp_path)
    try:
        collator = RowIdRecordingCollator(lambda feats: feats, recorder)
        collator([{"input_ids": [1], ROW_ID_COLUMN: 3}])
        collator([{"input_ids": [2], ROW_ID_COLUMN: 0}])
        collator([{"input_ids": [3], ROW_ID_COLUMN: 1}])
        dropped = recorder.reset()  # on_train_begin
        assert dropped == 3
        collator([{"input_ids": [1], ROW_ID_COLUMN: 3}])
        record = recorder.finalize_optimizer_step(1)
    finally:
        recorder.close()
    assert record["micro_batches"] == [[3]]
    assert record["row_ids"] == [3]


def test_micro_seqs_number_every_collation_monotonically(tmp_path):
    # Diagnostic contract: seqs expose the true collation count/order, so an
    # audit can separate "extra collation" (gap/duplicates in seqs) from a
    # "late step flush" (contiguous seqs, shifted grouping).
    recorder = _recorder(tmp_path)
    try:
        collator = RowIdRecordingCollator(lambda feats: feats, recorder)
        collator([{"input_ids": [1], ROW_ID_COLUMN: 0}])
        collator([{"input_ids": [2], ROW_ID_COLUMN: 1}])
        first = recorder.finalize_optimizer_step(1)
        collator([{"input_ids": [3], ROW_ID_COLUMN: 2}])
        second = recorder.finalize_optimizer_step(2)
    finally:
        recorder.close()
    assert first["micro_seqs"] == [0, 1]
    assert second["micro_seqs"] == [2]


def test_reset_keeps_arrival_counter_monotonic(tmp_path):
    recorder = _recorder(tmp_path)
    try:
        collator = RowIdRecordingCollator(lambda feats: feats, recorder)
        collator([{"input_ids": [1], ROW_ID_COLUMN: 0}])
        assert recorder.reset() == 1
        collator([{"input_ids": [2], ROW_ID_COLUMN: 1}])
        record = recorder.finalize_optimizer_step(1)
    finally:
        recorder.close()
    # The dropped pre-train micro keeps its seq; nothing is renumbered.
    assert record["micro_seqs"] == [1]
    assert record["row_ids"] == [1]


def test_preflight_fingerprint_only_under_probe_frame(tmp_path):
    recorder = _recorder(tmp_path)
    try:
        collator = RowIdRecordingCollator(lambda feats: feats, recorder)

        def _preflight_first_batch(rows):
            return collator(
                [{"input_ids": [1], ROW_ID_COLUMN: r} for r in rows]
            )

        _preflight_first_batch([0])
        collator([{"input_ids": [2], ROW_ID_COLUMN: 1}])
        record = recorder.finalize_optimizer_step(1)
    finally:
        recorder.close()
    assert record["micro_via_preflight"] == [True, False]
    assert len(record["micro_t"]) == 2
    assert all(isinstance(t, float) for t in record["micro_t"])


def test_timeline_events_land_in_separate_file(tmp_path):
    from core.training.composition_log import CompositionRecorder as _Recorder

    recorder = _Recorder(str(tmp_path / "batch_composition.jsonl"))
    try:
        recorder.mark_event("train_begin", 0)
        recorder.mark_event("step_end", 1)
        collator = RowIdRecordingCollator(lambda feats: feats, recorder)
        collator([{"input_ids": [1], ROW_ID_COLUMN: 0}])
        record = recorder.finalize_optimizer_step(1)
    finally:
        recorder.close()
    # Step records stay clean (no event keys leak into them).
    assert set(record) >= {
        "step", "micro_batches", "micro_seqs", "row_ids",
        "num_micro_batches", "num_rows",
    }
    assert "type" not in record and "event" not in record
    timeline = (
        tmp_path / "batch_composition_timeline.jsonl"
    ).read_text(encoding = "utf-8").strip().splitlines()
    import json as _json

    events = [_json.loads(line) for line in timeline]
    assert [(e["event"], e["step"]) for e in events] == [
        ("train_begin", 0),
        ("step_end", 1),
    ]
    assert all(isinstance(e["t"], float) for e in events)


def test_evaluate_boundary_resets_stale_buffer(tmp_path):
    # Safety net for evaluate-on-train-data: micros buffered outside an
    # optimizer step must never leak into the next step's record.
    recorder = _recorder(tmp_path)
    try:
        collator = RowIdRecordingCollator(lambda feats: feats, recorder)
        collator([{"input_ids": [1], ROW_ID_COLUMN: 0}])
        first = recorder.finalize_optimizer_step(1)
        collator([{"input_ids": [9], ROW_ID_COLUMN: 99}])
        dropped = recorder.reset()
        collator([{"input_ids": [2], ROW_ID_COLUMN: 1}])
        second = recorder.finalize_optimizer_step(2)
    finally:
        recorder.close()
    assert first["row_ids"] == [0]
    assert dropped == 1
    assert second["row_ids"] == [1]
    assert 99 not in first["row_ids"] + second["row_ids"]


def test_resume_appends_instead_of_overwriting(tmp_path):
    path = str(tmp_path / "composition.jsonl")
    first = CompositionRecorder(path)
    try:
        first.record_micro_batch([0, 1])
        first.finalize_optimizer_step(1)
    finally:
        first.close()
    # A resumed run reopens the same file and continues.
    second = CompositionRecorder(path)
    try:
        second.record_micro_batch([2])
        second.finalize_optimizer_step(2)
    finally:
        second.close()
    records = read_composition_records(path)
    assert [r["step"] for r in records] == [1, 2]
    assert records[1]["row_ids"] == [2]


def test_refuses_packing_streaming_and_online():
    with pytest.raises(ValueError, match = "packing=False"):
        validate_tracking_prerequisites(
            packing = True, is_streaming = False, online_tokenization_enabled = False
        )
    with pytest.raises(ValueError, match = "non-streaming"):
        validate_tracking_prerequisites(
            packing = False, is_streaming = True, online_tokenization_enabled = False
        )
    with pytest.raises(ValueError, match = "eager tokenization"):
        validate_tracking_prerequisites(
            packing = False, is_streaming = False, online_tokenization_enabled = True
        )
    # The pilot configuration passes.
    validate_tracking_prerequisites(
        packing = False, is_streaming = False, online_tokenization_enabled = False,
        dataset_length = 8,
    )
    with pytest.raises(ValueError, match = "non-empty"):
        validate_tracking_prerequisites(
            packing = False, is_streaming = False, online_tokenization_enabled = False,
            dataset_length = 0,
        )


def test_join_with_losses_keeps_steps_without_loss():
    records = [
        {"step": 1, "row_ids": [0, 1], "micro_batches": [[0, 1]], "num_rows": 2},
        {"step": 2, "row_ids": [2, 3], "micro_batches": [[2, 3]], "num_rows": 2},
    ]
    joined = join_with_losses(records, [(2, 0.5), (1, 0.9)])
    assert [(j["step"], j["loss"]) for j in joined] == [(1, 0.9), (2, 0.5)]
    assert joined[0]["row_ids"] == [0, 1]
    # logging_steps > 1 leaves gaps: no invented losses.
    sparse = join_with_losses(records, [(2, 0.5)])
    assert sparse[0]["loss"] is None
    assert sparse[1]["loss"] == 0.5


class _MaskedDatasetStub:
    """Standing in for trainer.train_dataset after Unsloth response masking."""

    def __init__(self, rows, keep_id):
        self._rows = list(rows)
        self.column_names = (
            ["input_ids", ROW_ID_COLUMN] if keep_id else ["input_ids"]
        )

    def __len__(self):
        return len(self._rows)

    def add_column(self, name, values):
        assert name == ROW_ID_COLUMN
        assert len(values) == len(self._rows)
        return _MaskedDatasetStub(
            [dict(r, **{name: v}) for r, v in zip(self._rows, values)],
            keep_id = True,
        )

    def row_ids(self):
        return [r.get(ROW_ID_COLUMN) for r in self._rows]


def test_masking_kept_column_passes_through_untouched():
    dataset = _MaskedDatasetStub([{"input_ids": [1]}, {"input_ids": [2]}], keep_id = True)
    out, action = ensure_row_ids(dataset, expected_rows = 2)
    assert out is dataset
    assert action == "present"


def test_masking_dropped_column_restamps_when_no_rows_filtered():
    # Mirrors the pilot log: masking drops every non-model column while all
    # 8 rows survive (Post-filter dataset size == stamp-time size).
    dataset = _MaskedDatasetStub([{"input_ids": [1]}] * 8, keep_id = False)
    out, action = ensure_row_ids(dataset, expected_rows = 8)
    assert action == "restamped"
    assert out.row_ids() == list(range(8))


def test_masking_dropped_rows_is_refused_not_guessed():
    dataset = _MaskedDatasetStub([{"input_ids": [1]}] * 6, keep_id = False)
    with pytest.raises(ValueError, match = "rows were filtered"):
        ensure_row_ids(dataset, expected_rows = 8)
    with pytest.raises(ValueError, match = "rows were filtered"):
        ensure_row_ids(dataset, expected_rows = None)


def _forensic_run_records():
    # Verbatim shape of the pilot sidecar (grad_accum=1, 8 rows, 4 epochs):
    # first record of each epoch holds 2 micros, last holds none.
    raw = [
        (1, [[3, 0], [1, 7]], [2, 3]),
        (2, [[2, 5]], [4]),
        (3, [[6, 4]], [5]),
        (4, [], []),
        (5, [[0, 4], [6, 1]], [6, 7]),
        (6, [[3, 2]], [8]),
        (7, [[7, 5]], [9]),
        (8, [], []),
        (9, [[0, 1], [6, 7]], [10, 11]),
        (10, [[4, 5]], [12]),
        (11, [[3, 2]], [13]),
        (12, [], []),
        (13, [[6, 4], [0, 2]], [14, 15]),
        (14, [[7, 3]], [16]),
        (15, [[1, 5]], [17]),
        (16, [], []),
    ]
    records = []
    for step, micros, seqs in raw:
        flat = [r for micro in micros for r in micro]
        records.append(
            {
                "step": step,
                "micro_batches": micros,
                "micro_seqs": seqs,
                "row_ids": flat,
                "num_micro_batches": len(micros),
                "num_rows": len(flat),
            }
        )
    return records


def test_attribute_steps_matches_forensic_run_exactly():
    attributed = {
        entry["step"]: entry for entry in attribute_steps(_forensic_run_records())
    }
    expected_rows = {
        1: [3, 0], 2: [1, 7], 3: [2, 5], 4: [6, 4],
        5: [0, 4], 6: [6, 1], 7: [3, 2], 8: [7, 5],
        9: [0, 1], 10: [6, 7], 11: [4, 5], 12: [3, 2],
        13: [6, 4], 14: [0, 2], 15: [7, 3], 16: [1, 5],
    }
    for step in range(1, 17):
        assert attributed[step]["attributed_row_ids"] == expected_rows[step], step
    # Every epoch attributes each of the 8 rows exactly once: no loss, no dup.
    for first in (1, 5, 9, 13):
        epoch_rows = []
        for step in range(first, first + 4):
            epoch_rows.extend(attributed[step]["attributed_row_ids"])
        assert sorted(epoch_rows) == list(range(8)), first
    # Branch audit: epoch openers use their own first micro, the rest the
    # previous record's tail.
    for step in (1, 5, 9, 13):
        assert attributed[step]["rule"] == "epoch_first", step
    for step in (2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16):
        assert attributed[step]["rule"] == "prev_tail", step
    # Seqs stay attached for the audit trail.
    assert attributed[1]["attributed_seqs"] == [2]
    assert attributed[2]["attributed_seqs"] == [3]


def test_attribute_steps_tolerates_legacy_records_without_seqs():
    records = [
        {"step": 1, "micro_batches": [[0, 1]], "row_ids": [0, 1]},
        {"step": 2, "micro_batches": [], "row_ids": []},
    ]
    attributed = {entry["step"]: entry for entry in attribute_steps(records)}
    assert attributed[1]["attributed_row_ids"] == [0, 1]
    # Unknown seqs stay explicit Nones rather than invented numbers.
    assert attributed[1]["attributed_seqs"] == [None]
    assert attributed[1]["rule"] == "epoch_first"
    assert attributed[2]["attributed_row_ids"] == [0, 1]
    assert attributed[2]["rule"] == "prev_tail"


def test_read_run_composition_serving_helper(tmp_path):
    # Missing dir / missing file / empty dir => not exists, never raises.
    assert read_run_composition(None) == (False, [], 0)
    assert read_run_composition("") == (False, [], 0)
    assert read_run_composition(str(tmp_path)) == (False, [], 0)
    # With a sidecar: records back, capped, total preserved.
    sidecar = tmp_path / "batch_composition.jsonl"
    sidecar.write_text(
        '{"step": 1, "row_ids": [0]}\n{"step": 2, "row_ids": [1]}\nnot-json\n',
        encoding = "utf-8",
    )
    exists, records, total = read_run_composition(str(tmp_path))
    assert exists is True
    assert total == 2
    assert [r["step"] for r in records] == [1, 2]
    exists, records, total = read_run_composition(str(tmp_path), limit = 1)
    assert (exists, len(records), total) == (True, 1, 2)


def test_stamp_and_prune_helpers():
    keep, ids = stamp_row_ids(["messages", "text", "source"], 3)
    assert keep == ["text", ROW_ID_COLUMN]
    assert ids == [0, 1, 2]
    with pytest.raises(ValueError, match = "'text' column"):
        stamp_row_ids(["messages"], 3)

    class _StubDataset:
        def __init__(self):
            self.column_names = ["text", "source", ROW_ID_COLUMN]
            self.removed = None

        def remove_columns(self, names):
            self.removed = list(names)
            return self

    stub = _StubDataset()
    out, pruned = prune_columns_for_tracking(stub, keep)
    assert out is stub
    assert pruned == ["source"]
    assert stub.removed == ["source"]


def _drive_step(recorder, collator, row_id_pairs, step, metrics=None):
    """One optimizer step as the loop performs it: collate, log, finalize."""
    for first, second in row_id_pairs:
        collator(
            [
                {"input_ids": [1], ROW_ID_COLUMN: first},
                {"input_ids": [2], ROW_ID_COLUMN: second},
            ]
        )
    if metrics is not None:
        recorder.note_metrics(step, **metrics)
    return recorder.finalize_optimizer_step(step)


def _records_by_step(path):
    import json as _json

    with open(path, encoding = "utf-8") as handle:
        return {
            int(_json.loads(line)["step"]): _json.loads(line)
            for line in handle
            if line.strip()
        }


MODULE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "core", "training", "composition_log.py"
)


def test_step_loss_mapping_exact_value_from_logging_event(tmp_path):
    # Test 1 — the value in the file IS the logging event's value, ==
    recorder = _recorder(tmp_path)
    collator = RowIdRecordingCollator(lambda feats: feats, recorder)
    try:
        _drive_step(recorder, collator, [(3, 0)], 6, {"loss": 1.7207})
        _drive_step(recorder, collator, [(1, 7)], 7, {"loss": 0.4})
        # Reverse order too (logging event after finalize, the real HF order):
        # metrics still land on their own step at flush time.
        collator(
            [
                {"input_ids": [1], ROW_ID_COLUMN: 4},
                {"input_ids": [2], ROW_ID_COLUMN: 5},
            ]
        )
        recorder.finalize_optimizer_step(8)
        recorder.note_metrics(8, loss = 0.35)
    finally:
        recorder.close()
    saved = _records_by_step(recorder.path)
    assert saved[6]["loss"] == 1.7207
    assert saved[6]["row_ids"] == [3, 0]
    assert saved[8]["loss"] == 0.35
    assert saved[8]["row_ids"] == [4, 5]


def test_step_grad_norm_exact_value(tmp_path):
    recorder = _recorder(tmp_path)
    collator = RowIdRecordingCollator(lambda feats: feats, recorder)
    try:
        _drive_step(recorder, collator, [(0, 1)], 6, {"grad_norm": 63.9088})
        _drive_step(recorder, collator, [(2, 3)], 7, {})
    finally:
        recorder.close()
    saved = _records_by_step(recorder.path)
    assert saved[6]["grad_norm"] == 63.9088
    assert saved[7]["grad_norm"] is None


def test_step_learning_rate_exact_value(tmp_path):
    recorder = _recorder(tmp_path)
    collator = RowIdRecordingCollator(lambda feats: feats, recorder)
    try:
        _drive_step(recorder, collator, [(0, 1)], 6, {"learning_rate": 1.333e-5})
        _drive_step(recorder, collator, [(2, 3)], 7, {})
    finally:
        recorder.close()
    saved = _records_by_step(recorder.path)
    assert saved[6]["learning_rate"] == 1.333e-5
    assert saved[7]["learning_rate"] is None


def test_step_smoothed_loss_present_or_null_never_invented(tmp_path):
    recorder = _recorder(tmp_path)
    collator = RowIdRecordingCollator(lambda feats: feats, recorder)
    try:
        _drive_step(recorder, collator, [(0, 1)], 6, {"smoothed_loss": 2.316})
        _drive_step(recorder, collator, [(2, 3)], 7, {})
    finally:
        recorder.close()
    saved = _records_by_step(recorder.path)
    assert saved[6]["smoothed_loss"] == 2.316
    # The backend logging event carries no smoothed value (the Charts smooth
    # client-side), so absence stays an honest null.
    assert saved[7]["smoothed_loss"] is None


def test_metrics_path_never_recomputes_or_touches_model():
    # Test 5 — static proof: the metrics plumbing cannot forward, backward,
    # optimize, schedule, or import any training framework, at any level.
    import ast as _ast

    source = open(MODULE_PATH, encoding = "utf-8").read()
    tree = _ast.parse(source)
    import_roots = set()
    for node in tree.body:
        if isinstance(node, _ast.Import):
            import_roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, _ast.ImportFrom) and node.module and node.level == 0:
            import_roots.add(node.module.split(".")[0])
    assert import_roots <= {
        "__future__", "json", "math", "os", "time", "traceback", "typing",
    }, import_roots
    called_attrs = set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Attribute):
            called_attrs.add(node.func.attr)
    forbidden = {
        "forward", "backward", "train", "compute_loss", "zero_grad",
        "step", "optimizer_step", "training_step",
    }
    hits = called_attrs & forbidden
    assert hits == set(), f"training-loop calls reachable: {hits}"


def test_empty_step_keeps_metrics_without_fake_attribution(tmp_path):
    # Test 6 — Metric exists != examples exist: step 4's metrics live on in
    # lossHistory, never in the sidecar (no line written); attribution over
    # the remaining lines still opens epoch 2 at step 5 via the numbering gap.
    from core.training.composition_log import attribute_steps

    recorder = _recorder(tmp_path)
    collator = RowIdRecordingCollator(lambda feats: feats, recorder)
    try:
        _drive_step(recorder, collator, [], 4, {"loss": 0.5, "grad_norm": 9.0})
        _drive_step(recorder, collator, [(0, 1)], 5, {"loss": 0.4})
    finally:
        recorder.close()
    saved = _records_by_step(recorder.path)
    assert 4 not in saved
    assert saved[5]["loss"] == 0.4
    assert saved[5]["row_ids"] == [0, 1]
    attributed = {
        entry["step"]: entry
        for entry in attribute_steps(list(saved.values()))
    }
    assert attributed[5]["rule"] == "epoch_first"
    assert attributed[5]["attributed_row_ids"] == [0, 1]


def test_metrics_do_not_perturb_attribution_forensic_flow(tmp_path):
    # Test 7 — the full 16-step pilot shape driven WITH metrics stashed must
    # attribute exactly like the metrics-free forensic proof.
    from core.training.composition_log import attribute_steps

    recorder = _recorder(tmp_path)
    collator = RowIdRecordingCollator(lambda feats: feats, recorder)
    forensic = {
        entry["step"]: entry for entry in _forensic_run_records()
    }
    try:
        for step in range(1, 17):
            micros = forensic[step]["micro_batches"]
            for first, *rest in micros:
                pair = (first, rest[0]) if rest else (first, first)
                collator(
                    [
                        {"input_ids": [1], ROW_ID_COLUMN: pair[0]},
                        {"input_ids": [2], ROW_ID_COLUMN: pair[1]},
                    ]
                )
            recorder.note_metrics(step, loss = 2.0 - step * 0.1)
            recorder.finalize_optimizer_step(step)
    finally:
        recorder.close()
    saved = _records_by_step(recorder.path)
    # Trailing-epoch empties are never persisted; their steps keep metrics
    # only in lossHistory, never in this file.
    assert sorted(saved) == [s for s in range(1, 17) if s not in (4, 8, 12, 16)]
    assert [saved[s]["loss"] for s in sorted(saved)] == [
        pytest.approx(2.0 - s * 0.1) for s in sorted(saved)
    ]
    expected_rows = {
        1: [3, 0], 2: [1, 7], 3: [2, 5], 4: [6, 4],
        5: [0, 4], 6: [6, 1], 7: [3, 2], 8: [7, 5],
        9: [0, 1], 10: [6, 7], 11: [4, 5], 12: [3, 2],
        13: [6, 4], 14: [0, 2], 15: [7, 3], 16: [1, 5],
    }
    attributed = {
        entry["step"]: entry
        for entry in attribute_steps(list(saved.values()))
    }
    # Gap-tolerant attribution: trailing-epoch steps (4/8/12/16) have no
    # record at all, yet every surviving step attributes exactly its own
    # consumed rows — identical rows to the with-empties proof.
    assert sorted(attributed) == sorted(saved)
    for step in sorted(saved):
        assert attributed[step]["attributed_row_ids"] == expected_rows[step], step
    for step in (1, 5, 9, 13):
        assert attributed[step]["rule"] == "epoch_first", step
    for step in (4, 8, 12, 16):
        assert step not in attributed


def test_resume_metrics_keyed_by_true_step_no_renumber(tmp_path):
    # Test 8 — resume appends; steps and their metrics never restart at zero.
    path = str(tmp_path / "composition.jsonl")
    first = CompositionRecorder(path)
    try:
        _drive_step(first, RowIdRecordingCollator(lambda f: f, first),
                    [(0, 1)], 1, {"loss": 1.0})
        _drive_step(first, RowIdRecordingCollator(lambda f: f, first),
                    [(2, 3)], 2, {"loss": 0.9})
    finally:
        first.close()
    second = CompositionRecorder(path)
    try:
        _drive_step(second, RowIdRecordingCollator(lambda f: f, second),
                    [(4, 5)], 3, {"loss": 0.8})
    finally:
        second.close()
    saved = _records_by_step(path)
    assert sorted(saved) == [1, 2, 3]
    assert [saved[s]["loss"] for s in (1, 2, 3)] == [1.0, 0.9, 0.8]
    assert saved[3]["row_ids"] == [4, 5]


def test_old_sidecar_without_metrics_stays_readable(tmp_path):
    # Test 9 — pre-metrics files: no metric keys, still joinable/attributable.
    from core.training.composition_log import attribute_steps

    path = tmp_path / "composition.jsonl"
    path.write_text(
        '{"step": 1, "micro_batches": [[0, 1]], "row_ids": [0, 1]}\n'
        '{"step": 2, "micro_batches": [], "row_ids": []}\n',
        encoding = "utf-8",
    )
    saved = _records_by_step(str(path))
    assert saved[1].get("loss") is None
    joined = join_with_losses(list(saved.values()), [(1, 0.5)])
    assert joined[0]["loss"] == 0.5
    attributed = {
        entry["step"]: entry for entry in attribute_steps(list(saved.values()))
    }
    assert attributed[1]["attributed_row_ids"] == [0, 1]
    # An empty record still borrows the previous record's tail: the step ran
    # (its loss exists in lossHistory), only its own micros list is empty.
    assert attributed[2]["attributed_row_ids"] == [0, 1]
    assert attributed[2]["rule"] == "prev_tail"


def test_finalized_record_carries_full_api_contract(tmp_path):
    # Regression for the stripped-download incident: the on-disk record must
    # contain every key the API/UI contract serves (composition + metrics +
    # diagnostics), so a download can never silently drop fields again.
    recorder = _recorder(tmp_path)
    collator = RowIdRecordingCollator(lambda feats: feats, recorder)
    try:
        _drive_step(
            recorder, collator, [(3, 0)], 6,
            {"loss": 1.5, "grad_norm": 2.5, "learning_rate": 0.001},
        )
        _drive_step(recorder, collator, [(1, 7)], 7, {})
    finally:
        recorder.close()
    saved = _records_by_step(recorder.path)
    expected_keys = {
        "step", "micro_batches", "micro_seqs", "micro_t",
        "micro_via_preflight", "row_ids", "num_micro_batches", "num_rows",
        "loss", "smoothed_loss", "grad_norm", "learning_rate",
    }
    assert expected_keys <= set(saved[6]), sorted(set(saved[6]))
    assert saved[6]["loss"] == 1.5
    assert saved[6]["grad_norm"] == 2.5
    assert saved[6]["learning_rate"] == 0.001
    assert saved[6]["smoothed_loss"] is None
    assert saved[6]["micro_t"] != []
    assert saved[6]["micro_via_preflight"] == [False]


def test_enriched_record_matches_logging_event_exactly(tmp_path):
    # Test 10 — all four metrics at once, == not approx, file round-trip.
    recorder = _recorder(tmp_path)
    collator = RowIdRecordingCollator(lambda feats: feats, recorder)
    try:
        _drive_step(
            recorder, collator, [(3, 0)], 6,
            {"loss": 1.7207, "smoothed_loss": 2.316,
             "grad_norm": 63.9088, "learning_rate": 1.333e-5},
        )
        _drive_step(recorder, collator, [(1, 7)], 7, {})
    finally:
        recorder.close()
    saved = _records_by_step(recorder.path)
    record = saved[6]
    assert record["loss"] == 1.7207
    assert record["smoothed_loss"] == 2.316
    assert record["grad_norm"] == 63.9088
    assert record["learning_rate"] == 1.333e-5
    assert record["row_ids"] == [3, 0]
    assert saved[7]["loss"] is None
    assert saved[7]["smoothed_loss"] is None
    assert saved[7]["grad_norm"] is None
    assert saved[7]["learning_rate"] is None


def test_nonempty_record_written_on_both_flush_paths(tmp_path):
    # A real batch lands in the file whether flushed mid-run (next finalize)
    # or at close (train end) — with its own metrics, never another's.
    import json as _json

    recorder = _recorder(tmp_path)
    collator = RowIdRecordingCollator(lambda feats: feats, recorder)
    try:
        _drive_step(recorder, collator, [(0, 1)], 1, {"loss": 1.0})
        _drive_step(recorder, collator, [(2, 3)], 2, {"loss": 0.9})
        mid_run = [
            _json.loads(line)
            for line in open(recorder.path, encoding = "utf-8")
            if line.strip()
        ]
        assert [r["step"] for r in mid_run] == [1]
        assert mid_run[0]["loss"] == 1.0
    finally:
        recorder.close()
    saved = _records_by_step(recorder.path)
    assert sorted(saved) == [1, 2]
    assert saved[2]["loss"] == 0.9
    assert saved[2]["row_ids"] == [2, 3]


def test_metrics_never_migrate_across_steps(tmp_path):
    recorder = _recorder(tmp_path)
    collator = RowIdRecordingCollator(lambda feats: feats, recorder)
    try:
        _drive_step(
            recorder, collator, [(0, 1)], 6,
            {"loss": 1.7, "grad_norm": 60.0, "learning_rate": 0.01},
        )
        _drive_step(
            recorder, collator, [(2, 3)], 7,
            {"loss": 0.4, "grad_norm": 20.0, "learning_rate": 0.005},
        )
    finally:
        recorder.close()
    saved = _records_by_step(recorder.path)
    assert (saved[6]["loss"], saved[6]["grad_norm"],
            saved[6]["learning_rate"]) == (1.7, 60.0, 0.01)
    assert (saved[7]["loss"], saved[7]["grad_norm"],
            saved[7]["learning_rate"]) == (0.4, 20.0, 0.005)


def test_resume_with_metrics_appends_keyed_records(tmp_path):
    path = str(tmp_path / "composition.jsonl")
    first = CompositionRecorder(path)
    try:
        _drive_step(first, RowIdRecordingCollator(lambda f: f, first),
                    [(0, 1)], 1, {"loss": 1.0, "grad_norm": 5.0})
    finally:
        first.close()
    second = CompositionRecorder(path)
    try:
        _drive_step(second, RowIdRecordingCollator(lambda f: f, second),
                    [(2, 3)], 2, {"loss": 0.9, "grad_norm": 4.0})
    finally:
        second.close()
    saved = _records_by_step(path)
    assert sorted(saved) == [1, 2]
    assert (saved[1]["loss"], saved[2]["loss"]) == (1.0, 0.9)
    assert (saved[1]["grad_norm"], saved[2]["grad_norm"]) == (5.0, 4.0)


def _pilot_gap_records():
    # Step 5/6 region of the real pilot sidecar (grad_accum=1, batch=2).
    return [
        {"step": 4, "micro_batches": [], "micro_seqs": [],
         "row_ids": []},
        {"step": 5, "micro_batches": [[0, 4], [6, 1]], "micro_seqs": [6, 7],
         "row_ids": [0, 4, 6, 1]},
        {"step": 6, "micro_batches": [[3, 2]], "micro_seqs": [8],
         "row_ids": [3, 2]},
    ]


def test_presentation_resolves_step_five_and_six_truthfully():
    view = {
        row["trainer_step"]: row
        for row in present_step_attribution(
            _pilot_gap_records(), [(5, 2.9812), (6, 1.7207)]
        )
    }
    # The approved rule, surfaced per step: step 5 owns the epoch-opening
    # micro only; step 6 owns the previous record's tail. No renumbering.
    assert view[5]["attributed_row_ids"] == [0, 4]
    assert view[5]["rule"] == "epoch_first"
    assert view[5]["loss"] == 2.9812
    assert view[6]["attributed_row_ids"] == [6, 1]
    assert view[6]["rule"] == "prev_tail"
    assert view[6]["loss"] == 1.7207


def test_presentation_keys_loss_by_true_step_never_position():
    records = _pilot_gap_records()
    view = {
        row["trainer_step"]: row
        for row in present_step_attribution(
            records, [(6, 1.5), (5, 2.5)]
        )
    }
    # Out-of-order / gapped loss input still lands on true steps...
    assert view[5]["loss"] == 2.5
    assert view[6]["loss"] == 1.5
    # ...steps without a loss point stay honestly empty...
    view2 = {
        row["trainer_step"]: row
        for row in present_step_attribution(records, [(5, 2.5)])
    }
    assert view2[6]["loss"] is None
    # ...and trainer_step values pass through verbatim (no relabeling).
    assert [row["trainer_step"] for row in present_step_attribution(records)] == [4, 5, 6]


def test_presentation_never_mutates_raw_records():
    import copy as _copy

    records = _pilot_gap_records()
    snapshot = _copy.deepcopy(records)
    present_step_attribution(records, [(4, 0.1), (5, 0.2), (6, 0.3)])
    assert records == snapshot


def test_presentation_empty_record_carries_no_rows_but_keeps_loss():
    view = {
        row["trainer_step"]: row
        for row in present_step_attribution(_pilot_gap_records(), [(4, 9.9)])
    }
    assert view[4]["attributed_row_ids"] == []
    assert view[4]["rule"] == "none"
    assert view[4]["loss"] == 9.9


class _FakeOptimizer:
    """Stands in for a torch optimizer: records step calls verbatim."""

    def __init__(self):
        self.calls = []
        self.result = object()

    def step(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


def test_wrap_optimizer_step_counts_without_changing_behavior():
    from core.training.composition_log import wrap_optimizer_step

    optimizer = _FakeOptimizer()
    marks = []
    assert wrap_optimizer_step(optimizer, lambda: marks.append(1)) is True
    assert optimizer.step(closure_arg := object()) is optimizer.result
    assert optimizer.calls == [((closure_arg,), {})]
    assert marks == [1]
    optimizer.step()
    assert marks == [1, 1]


def test_wrap_optimizer_step_never_breaks_training():
    from core.training.composition_log import wrap_optimizer_step

    assert wrap_optimizer_step(object(), lambda: None) is False
    assert wrap_optimizer_step(None, lambda: None) is False

    class _ReadOnlyStep:
        @property
        def step(self):
            raise RuntimeError("nope")

    assert wrap_optimizer_step(_ReadOnlyStep(), lambda: None) is False

    optimizer = _FakeOptimizer()

    def _boom():
        raise RuntimeError("timeline disk full (simulated)")

    assert wrap_optimizer_step(optimizer, _boom) is True
    assert optimizer.step() is optimizer.result


def _real_pilot_micros():
    # Verbatim collation order of the grad_accum diagnostic run (seqs 2..17):
    # four full reshuffled epochs, batch size 2.
    rows = [
        [3, 0], [1, 7], [2, 5], [6, 4],
        [0, 4], [6, 1], [3, 2], [7, 5],
        [0, 1], [6, 7], [4, 5], [3, 2],
        [6, 4], [0, 2], [7, 3], [1, 5],
    ]
    return [(seq, list(pair)) for seq, pair in zip(range(2, 18), rows)]


def test_grouping_k1_identity_no_empty_groups():
    from core.training.composition_log import resolve_optimizer_steps

    entries = [(0, [0]), (1, [1]), (2, [2])]
    groups, report = resolve_optimizer_steps(entries, 1)
    assert [g["optimizer_step"] for g in groups] == [1, 2, 3]
    assert all(g["num_micros"] >= 1 and g["num_rows"] >= 1 for g in groups)
    assert report["complete"] is True


def test_grouping_k4_matches_pilot_epochs_exactly():
    # grad_accum=4 shape over the REAL captured micros: 4 groups, each a
    # full reshuffled epoch (proves timing-independent grouping: only order
    # is read, never micro_t/prefetch gaps).
    from core.training.composition_log import resolve_optimizer_steps

    groups, report = resolve_optimizer_steps(_real_pilot_micros(), 4)
    assert report == {
        "micros_per_step": 4, "num_groups": 4, "total_micros": 16,
        "total_rows": 32, "contiguous": True, "complete": True,
    }
    for group in groups:
        assert group["num_micros"] == 4
        assert sorted(group["row_ids"]) == list(range(8))
    assert groups[0]["micro_seqs"] == [2, 3, 4, 5]
    assert groups[0]["row_ids"] == [3, 0, 1, 7, 2, 5, 6, 4]


def test_grouping_k8_target_shape_batch2():
    # The task success criterion: 32 micros / batch 2 chunked 8 per true
    # optimizer step -> 4 groups of 8 micros / 16 rows, no empties, no loss,
    # no duplication.
    from core.training.composition_log import resolve_optimizer_steps

    entries = [(seq, [2 * seq, 2 * seq + 1]) for seq in range(32)]
    groups, report = resolve_optimizer_steps(entries, 8)
    assert report["num_groups"] == 4
    assert report["complete"] is True
    assert groups[0]["row_ids"] == list(range(16))
    assert groups[1]["row_ids"] == list(range(16, 32))
    assert groups[2]["row_ids"] == list(range(32, 48))
    assert groups[3]["row_ids"] == list(range(48, 64))
    assert all(g["num_micros"] == 8 and g["num_rows"] == 16 for g in groups)


def test_grouping_batch4_shapes():
    from core.training.composition_log import resolve_optimizer_steps

    entries = [(seq, [4 * seq + i for i in range(4)]) for seq in range(8)]
    groups, _ = resolve_optimizer_steps(entries, 2)
    assert [g["num_micros"] for g in groups] == [2, 2, 2, 2]
    assert [g["num_rows"] for g in groups] == [8, 8, 8, 8]
    groups8, report8 = resolve_optimizer_steps(entries, 8)
    assert len(groups8) == 1
    assert groups8[0]["num_micros"] == 8
    assert groups8[0]["num_rows"] == 32
    assert report8["complete"] is True


def test_grouping_rejects_gaps_duplicates_and_ragged():
    from core.training.composition_log import resolve_optimizer_steps

    entries = [(0, [0]), (1, [1]), (2, [2])]
    with pytest.raises(ValueError, match = "contiguous"):
        resolve_optimizer_steps([(0, [0]), (2, [2])], 1)
    with pytest.raises(ValueError, match = "contiguous"):
        resolve_optimizer_steps([(0, [0]), (0, [1])], 1)
    with pytest.raises(ValueError, match = "positive int"):
        resolve_optimizer_steps(entries, 0)
    with pytest.raises(ValueError, match = "positive int"):
        resolve_optimizer_steps(entries, -3)
    with pytest.raises(ValueError, match = "whole steps"):
        resolve_optimizer_steps(entries, 2)


def test_grouping_carries_no_metrics_and_keeps_schema():
    # Grouping is composition-only: it neither invents metrics (loss lives
    # with on_log keyed steps, joined separately) nor drops composition keys.
    from core.training.composition_log import resolve_optimizer_steps

    groups, _ = resolve_optimizer_steps([(0, [0, 1]), (1, [2, 3])], 1)
    assert set(groups[0]) == {
        "optimizer_step", "micro_seqs", "row_ids", "num_micros", "num_rows",
    }
    assert "loss" not in groups[0] and "grad_norm" not in groups[0]


def test_gap_input_attributes_identically_to_sentinel_input(tmp_path):
    # The strongest "rule unchanged" proof: the same 16-step pilot flow,
    # once WITH trailing-epoch empties and once WITHOUT, must attribute
    # every surviving step to the exact same rows.
    from core.training.composition_log import attribute_steps

    with_empties = []
    without_empties = []
    for step in range(1, 17):
        micros = _forensic_run_records()[step - 1]["micro_batches"]
        flat = [r for micro in micros for r in micro]
        record = {
            "step": step, "micro_batches": micros,
            "micro_seqs": [], "row_ids": flat,
        }
        with_empties.append(record)
        if flat:
            without_empties.append(record)
    attributed_full = {
        entry["step"]: entry for entry in attribute_steps(with_empties)
    }
    attributed_gap = {
        entry["step"]: entry for entry in attribute_steps(without_empties)
    }
    for step in range(1, 17):
        if step in (4, 8, 12, 16):
            assert step not in attributed_gap
            continue
        assert (
            attributed_gap[step]["attributed_row_ids"]
            == attributed_full[step]["attributed_row_ids"]
        ), step
