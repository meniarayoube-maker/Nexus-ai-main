# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-example loss attribution as offline analysis (Pilot).

Proves the scoring unit is faithful (same template/truncation/masking),
honest (null, never 0.0, for rows with no trainable tokens; joinable by
row_id/example_id), and structurally incapable of training anything
(stdlib-only imports, no training-loop calls anywhere in the module, no
coupling to batch-composition tracking internals).
"""

from __future__ import annotations

import ast
import json
import math
import os

import pytest

import core.training.offline_scoring as scoring
from core.training.offline_scoring import (
    OfflineScoreConfig,
    find_marker_spans,
    masked_nll,
    mask_response_spans,
    score_rows,
    tokenize_and_mask,
    write_records_jsonl,
)

MODULE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "core", "training", "offline_scoring.py"
)


class FakeTokenizer:
    """Word-level tokenizer over an explicit vocab (deterministic ids)."""

    def __init__(self, vocab):
        self.vocab = dict(vocab)

    def __call__(self, text, add_special_tokens = True):
        ids = [self.vocab[w] for w in str(text).split() if w in self.vocab]
        return {"input_ids": ids}

    def encode(self, text, add_special_tokens = False):
        return self(text, add_special_tokens = add_special_tokens)["input_ids"]


def fake_format(rows, model_name = None, tokenizer = None, format_type = "auto",
                custom_format_mapping = None):
    """Same contract as format_and_template_dataset, chat-style join."""
    seen = {"model_name": model_name, "format_type": format_type}
    fake_format.seen.append(seen)
    dataset = []
    for row in rows:
        messages = row.get("messages") or []
        text = "<T> " + " ".join(str(m.get("content", "")) for m in messages) + " </T>"
        dataset.append({**row, "text": text})
    return {
        "dataset": dataset,
        "detected_format": "fake_chat",
        "final_format": "fake_chat",
        "success": True,
        "warnings": [],
        "errors": [],
    }


fake_format.seen = []

CONFIG = OfflineScoreConfig(checkpoint = "ckpt-final", max_seq_length = 2048)


def _forward_capturing(captured, logits):
    def forward(tokenizer, input_ids):
        captured.append(list(input_ids))
        return logits

    return forward


def test_metadata_never_lost_or_invented():
    rows = [
        {"messages": [{"content": "hi"}], "example_id": "ex-1",
         "source": "s", "level": "L0", "batch_id": "b", "extra": "dropped-ok"},
        {"messages": [{"content": "yo"}], "example_id": "ex-2"},
    ]
    captured = []
    records = score_rows(
        rows, config = CONFIG, model_name = "m",
        tokenizer = FakeTokenizer({"<T>": 1, "hi": 2, "yo": 3, "</T>": 4}),
        forward_fn = _forward_capturing(captured, [[[0.0, 0.0, 0.0, 0.0, 0.0]] * 3]),
        format_fn = fake_format,
    )
    assert [r["example_id"] for r in records] == ["ex-1", "ex-2"]
    assert records[0]["source"] == "s"
    assert records[0]["level"] == "L0"
    assert records[0]["batch_id"] == "b"
    assert records[1]["source"] is None
    assert "extra" not in records[0]


def test_same_chat_template_reaches_scoring_verbatim():
    tokenizer = FakeTokenizer({"<T>": 1, "hi": 2, "</T>": 3})
    captured = []
    score_rows(
        [{"messages": [{"content": "hi"}]}],
        config = CONFIG, model_name = "model-x", tokenizer = tokenizer,
        forward_fn = _forward_capturing(captured, [[[0.0] * 4]] * 3),
        format_fn = fake_format,
    )
    assert fake_format.seen[-1] == {"model_name": "model-x", "format_type": "auto"}
    # The exact template-wrapped text is what gets tokenized for scoring.
    assert captured and captured[0] == [1, 2, 3]


def test_same_truncation_limit_applies_before_scoring():
    words = {f"w{i}": 10 + i for i in range(10)}
    tokenizer = FakeTokenizer(words)
    captured = []
    config = OfflineScoreConfig(checkpoint = "c", max_seq_length = 4)
    records = score_rows(
        [{"messages": [{"content": " ".join(f"w{i}" for i in range(10))}]}],
        config = config, model_name = "m", tokenizer = tokenizer,
        forward_fn = _forward_capturing(captured, [[[0.0] * 20]] * 10),
        format_fn = lambda rows, **kw: {
            "dataset": [{**r, "text": " ".join(f"w{i}" for i in range(10))} for r in rows],
            "success": True, "warnings": [], "errors": [],
        },
    )
    assert captured and len(captured[0]) == 4
    assert records[0]["num_total_tokens"] == 4


def test_masking_unmasks_only_response_spans():
    assert mask_response_spans([9, 1, 2, 3, 9, 4], response_ids = [1], instruction_ids = [9]) == [-100, 1, 2, 3, -100, -100]
    assert mask_response_spans([5, 6], response_ids = [1]) == [-100, -100]
    assert mask_response_spans([5, 6], response_ids = None) == [-100, -100]
    assert find_marker_spans([1, 2, 1, 2, 3], [1, 2]) == [(0, 2), (2, 4)]


def test_loss_counts_only_response_tokens():
    # Masked position holds deliberately insane logits; it must not move loss.
    logits = [[[0.0, 0.0], [100.0, -100.0]]]
    loss, count = masked_nll(logits, [0, -100])
    assert count == 1
    assert loss == pytest.approx(math.log(2.0))
    # Hand-checkable two-token mean.
    logits = [[[2.0, 0.0], [0.0, 0.0]]]
    loss, count = masked_nll(logits, [0, 1])
    expected = ((math.log(math.exp(2.0) + 1.0) - 2.0) + math.log(2.0)) / 2.0
    assert loss == pytest.approx(expected)
    assert count == 2


def test_row_without_loss_tokens_records_null_never_zero(tmp_path):
    tokenizer = FakeTokenizer({"a": 1, "b": 2})
    records = score_rows(
        [{"messages": [{"content": "a b"}], "example_id": "empty"}],
        config = CONFIG, model_name = "m", tokenizer = tokenizer,
        forward_fn = lambda tok, ids: [[[0.0, 0.0]] * len(ids)],
        format_fn = fake_format,
    )
    assert len(records) == 1
    assert records[0]["loss"] is None
    assert records[0]["num_loss_tokens"] == 0
    assert records[0]["status"] == "no_loss_tokens"
    assert records[0]["reason"]
    out = write_records_jsonl(records, str(tmp_path / "per_example_loss.jsonl"))
    line = open(out, encoding = "utf-8").readline()
    assert json.loads(line)["loss"] is None


def test_records_joinable_by_row_and_example_id():
    rows = [
        {"messages": [{"content": "hi"}], "example_id": f"ex-{i}"}
        for i in range(3)
    ]
    records = score_rows(
        rows, config = CONFIG, model_name = "m",
        tokenizer = FakeTokenizer({"<T>": 1, "hi": 2, "</T>": 3}),
        forward_fn = lambda tok, ids: [[[0.0] * 4]] * len(ids),
        format_fn = fake_format,
    )
    assert [r["row_id"] for r in records] == [0, 1, 2]
    assert [r["example_id"] for r in records] == ["ex-0", "ex-1", "ex-2"]
    assert all(r["loss_kind"] == "offline_evaluation_loss" for r in records)
    assert all(r["checkpoint"] == "ckpt-final" for r in records)


def test_module_contains_no_training_loop_calls():
    source = open(MODULE_PATH, encoding = "utf-8").read()
    forbidden = [
        "loss.backward", ".backward(", "optimizer", "Trainer(",
        "SFTTrainer", ".train(", "zero_grad", "gradient_accumulation",
        "torch.compile",
    ]
    hits = [token for token in forbidden if token in source]
    assert hits == [], f"training constructs leaked in: {hits}"


def test_module_imports_stdlib_only_at_top_level():
    source = open(MODULE_PATH, encoding = "utf-8").read()
    tree = ast.parse(source)
    roots = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(part.split(".")[0] for part in
                         (a.name for a in node.names))
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                roots.add(node.module.split(".")[0])
    allowed = {
        "__future__", "hashlib", "json", "math", "os",
        "dataclasses", "typing",
    }
    assert roots <= allowed, f"non-stdlib top-level imports: {roots - allowed}"


def test_no_coupling_to_tracking_or_trainer_internals():
    source = open(MODULE_PATH, encoding = "utf-8").read()
    for token in ("composition_log", "_train_worker", "core.training.trainer",
                  "core.training.worker"):
        assert token not in source, token


def test_tokenize_and_mask_end_to_end_ids():
    tokenizer = FakeTokenizer({"u": 1, "hi": 2, "a": 3, "yo": 4})
    ids, labels = tokenize_and_mask(tokenizer, "u hi a yo", [3], [1], 2048)
    assert ids == [1, 2, 3, 4]
    assert labels == [-100, -100, 3, 4]
