# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Render-preview (Pilot v1): metadata audit, token accounting, request validation.

The preview must prove that metadata columns living outside ``messages``
(``example_id`` / ``source`` / ``level`` / ``batch_id``) do not leak into the
exact text training consumes. Heavy IO (dataset download, tokenizer load, real
formatting) is stubbed; the audit/count/validation logic itself is exercised.
"""

from __future__ import annotations

import pytest

from core.training import render_preview as preview
from models.training import RenderPreviewRequest


class _FakeEncoding(dict):
    pass


class _FakeTokenizer:
    """Whitespace tokenizer: ids are word ordinals, decode joins them back."""

    def __call__(self, text, add_special_tokens = True):
        words = text.split()
        return _FakeEncoding({"input_ids": list(range(1000, 1000 + len(words)))})

    def decode(self, ids, skip_special_tokens = False):
        return " ".join(f"w{i - 1000}" for i in ids)


def _fake_format_dataset(dataset, **kwargs):
    rows = [dict(dataset[i]) for i in range(len(dataset))]
    texts = []
    for row in rows:
        messages = row.get("messages") or []
        texts.append(
            " ".join(
                str(m.get("content", "")) for m in messages if isinstance(m, dict)
            )
        )
    out = dataset.to_dict()
    out["text"] = texts
    from datasets import Dataset

    return {
        "dataset": Dataset.from_dict(out),
        "detected_format": "chatml_conversations",
        "final_format": "chatml_conversations",
        "success": True,
        "warnings": [],
        "errors": [],
    }


def _run(monkeypatch, raw_rows, cap_at = 2048):
    from datasets import Dataset

    monkeypatch.setattr(
        preview, "load_preview_tokenizer", lambda **kwargs: (_FakeTokenizer(), "fake")
    )
    monkeypatch.setattr(
        preview,
        "load_preview_sample_rows",
        lambda *args, **kwargs: (
            [dict(r) for r in raw_rows],
            list(raw_rows[0].keys()),
            len(raw_rows),
        ),
    )
    monkeypatch.setattr(preview, "format_and_template_dataset", _fake_format_dataset)
    return preview.render_preview_samples(
        dataset_source = "upload",
        model_name = "fake-model",
        num_samples = len(raw_rows),
        local_datasets = ["fake.jsonl"],
        max_seq_length = cap_at,
    )


def test_metadata_outside_messages_does_not_leak(monkeypatch):
    rows = [
        {
            "example_id": "ex-001",
            "source": "pilot-seed",
            "level": "easy",
            "batch_id": "b7",
            "messages": [
                {"role": "user", "content": "hello world"},
                {"role": "assistant", "content": "hi there"},
            ],
        }
    ]
    result = _run(monkeypatch, rows)
    assert result["success"] is True
    sample = result["samples"][0]
    assert sample["rendered_text"] == "hello world hi there"
    audit = {a["column"]: a["leaked_into_text"] for a in sample["metadata_audit"]}
    assert audit == {
        "example_id": False,
        "source": False,
        "level": False,
        "batch_id": False,
    }
    # Tokenizer input is fully reported: 4 words -> 4 ids, capped count equal.
    assert sample["token_count_full"] == 4
    assert sample["token_count_capped"] == 4
    assert sample["input_ids_head"] == [1000, 1001, 1002, 1003]
    assert sample["input_ids_tail"] == []


def test_metadata_value_inside_message_content_is_flagged(monkeypatch):
    rows = [
        {
            "example_id": "ex-002",
            "messages": [{"role": "user", "content": "my id is ex-002 ok"}],
        }
    ]
    result = _run(monkeypatch, rows)
    audit = {a["column"]: a["leaked_into_text"] for a in result["samples"][0]["metadata_audit"]}
    assert audit == {"example_id": True}


def test_token_cap_is_reported_not_hidden(monkeypatch):
    rows = [
        {"messages": [{"role": "user", "content": " ".join(f"w{i}" for i in range(100))}]}
    ]
    result = _run(monkeypatch, rows, cap_at = 10)
    sample = result["samples"][0]
    assert sample["token_count_full"] == 100
    assert sample["token_count_capped"] == 10
    assert sample["cap_applied_at"] == 10
    assert len(sample["input_ids_head"]) == 32
    assert len(sample["input_ids_tail"]) == 32


def test_jsonable_never_raises_on_odd_cells():
    assert preview._jsonable({"a": object()})["a"].startswith("<")
    assert isinstance(preview._jsonable(b"\x00\x01"), str)
    long_str = "x" * 5000
    assert preview._jsonable(long_str).endswith("chars]")


def test_unsupported_source_is_rejected():
    with pytest.raises(ValueError, match = "v1 supports"):
        preview.load_preview_sample_rows("s3", 3)


def test_request_requires_source_fields():
    with pytest.raises(Exception, match = "hf_dataset is required"):
        RenderPreviewRequest(dataset_source = "huggingface", model_name = "m")
    with pytest.raises(Exception, match = "local_datasets is required"):
        RenderPreviewRequest(dataset_source = "upload", model_name = "m")
    ok = RenderPreviewRequest(
        dataset_source = "upload", model_name = "m", local_datasets = ["a.jsonl"]
    )
    assert ok.num_samples == 3
