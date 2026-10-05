# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Batch composition tracking (Pilot): the id column must never reach the model.

Covers the sidecar lifecycle (micro-batch -> optimizer step -> append across
resume), the eval/train gate, the packing/streaming/online refusals, and the
loss join. Heavy training deps are NOT needed: the module under test is
stdlib-only and collaborators are fakes.
"""

from __future__ import annotations

import pytest

from core.training.composition_log import (
    ROW_ID_COLUMN,
    CompositionRecorder,
    RowIdRecordingCollator,
    join_with_losses,
    prune_columns_for_tracking,
    read_composition_records,
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
    recorder.set_capturing(True)
    collator = RowIdRecordingCollator(base, recorder)
    try:
        out = collator(
            [
                {"input_ids": [1, 2], ROW_ID_COLUMN: 7},
                {"input_ids": [3], ROW_ID_COLUMN: 9},
            ]
        )
    finally:
        recorder.close()
    assert out == {"batched": 2}
    # The model-facing batch contains no trace of the tracking column...
    for keys in base.seen_keys[0]:
        assert ROW_ID_COLUMN not in keys
        assert keys == ["input_ids"]
    # ...while the recorder captured both ids in order.
    records = read_composition_records(recorder.path)
    assert records == []
    recorder2 = CompositionRecorder(recorder.path)
    try:
        recorder2.set_capturing(True)
        collator2 = RowIdRecordingCollator(base, recorder2)
        collator2([{"input_ids": [1], ROW_ID_COLUMN: 7}])
        record = recorder2.finalize_optimizer_step(3)
    finally:
        recorder2.close()
    assert record["row_ids"] == [7]
    assert record["step"] == 3


def test_batches_without_id_column_pass_through_untouched(tmp_path):
    base = _FakeBaseCollator()
    recorder = _recorder(tmp_path)
    recorder.set_capturing(True)
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
        recorder.set_capturing(True)
        collator = RowIdRecordingCollator(lambda feats: {"n": len(feats)}, recorder)
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


def test_capture_gate_excludes_eval_phase(tmp_path):
    recorder = _recorder(tmp_path)
    try:
        recorder.set_capturing(True)
        collator = RowIdRecordingCollator(lambda feats: feats, recorder)
        collator([{"input_ids": [1], ROW_ID_COLUMN: 0}])
        first = recorder.finalize_optimizer_step(1)
        # Eval runs between optimizer steps with capture off: silently ignored
        # and never leaks into the next step's record.
        recorder.set_capturing(False)
        collator([{"input_ids": [9], ROW_ID_COLUMN: 99}])
        recorder.set_capturing(True)
        collator([{"input_ids": [2], ROW_ID_COLUMN: 1}])
        second = recorder.finalize_optimizer_step(2)
    finally:
        recorder.close()
    assert first["row_ids"] == [0]
    assert second["row_ids"] == [1]
    assert 99 not in first["row_ids"] + second["row_ids"]


def test_resume_appends_instead_of_overwriting(tmp_path):
    path = str(tmp_path / "composition.jsonl")
    first = CompositionRecorder(path)
    try:
        first.set_capturing(True)
        first.record_micro_batch([0, 1])
        first.finalize_optimizer_step(1)
    finally:
        first.close()
    # A resumed run reopens the same file and continues.
    second = CompositionRecorder(path)
    try:
        second.set_capturing(True)
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
