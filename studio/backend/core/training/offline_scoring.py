# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-example loss attribution as pure post-training analysis (Pilot).

Answers "how hard is each example for the model AT THIS CHECKPOINT?" — not
"which example caused a historical training-step loss". Read the module
docstring contract carefully before using the numbers:

* OFFLINE: loads a saved checkpoint independently (``model.eval()``,
  ``torch.no_grad()``). Never touches the Trainer, parameter updates, the
  scheduler, gradients, or any training loop. Importing this module pulls
  stdlib only; torch/transformers/datasets are imported lazily inside the
  functions that truly need them, so analysis tooling stays importable
  anywhere (and this is asserted by the test-suite).
* SAME TEXT: example texts are rebuilt with the exact training pipeline
  (:func:`utils.datasets.format_and_template_dataset`), same tokenizer, same
  chat template, same ``max_seq_length`` truncation (head kept).
* SAME MASKING (documented approximation): labels are unmasked only inside
  response spans — from each ``response_part`` marker to the next
  ``instruction_part`` marker (or end of sequence when the instruction marker
  is unknown). Rows with zero trainable tokens record ``loss=None``
  (``no_loss_tokens``), mirroring training dropping them — never ``0.0``.
* JOINABLE: every record carries ``row_id`` + the source row's metadata
  (``example_id``/``source``/``level``/``batch_id``), so it joins directly
  with ``batch_composition.jsonl`` (which example was in which step) and with
  the preview service (which text was scored — see ``text_sha256``).

Scientific caveats (do not compare naively with training loss):
* Training loss is computed DURING optimization (evolving weights, LR
  schedule, data order, grad-accum averaging, train-mode layers). These
  scores use FIXED final (or chosen-checkpoint) weights in eval mode.
* Only checkpoints that were actually saved can be scored. Runs with
  ``save_strategy=no`` keep the final model only: no per-step history exists.
* A high offline loss means "hard for this checkpoint", NOT "caused a past
  loss spike". Causality needs the batch-composition join + step timeline.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

LOSS_KIND = "offline_evaluation_loss"

# Metadata columns carried verbatim from the source row into every record.
# Missing keys become None (never dropped, never invented).
METADATA_COLUMNS = ("example_id", "source", "level", "batch_id")


@dataclass
class OfflineScoreConfig:
    """Everything the scorer needs; all of it readable from run artifacts."""

    checkpoint: str
    max_seq_length: int = 2048
    text_field: str = "text"
    instruction_part: Optional[str] = None
    response_part: Optional[str] = None
    device: str = "cuda"
    trust_remote_code: bool = False
    # Mirror training: when the run trained on full sequences
    # (train_on_completions off), score full sequences too instead of
    # masking everything into nulls.
    apply_masking: bool = True


@dataclass
class ExampleScore:
    """One scored example. ``loss`` is None exactly when there is no number."""

    row_id: int
    example_id: Any = None
    source: Any = None
    level: Any = None
    batch_id: Any = None
    loss: Optional[float] = None
    num_loss_tokens: int = 0
    checkpoint: str = ""
    status: str = "scored"
    reason: str = ""
    loss_kind: str = LOSS_KIND
    num_total_tokens: int = 0
    text_sha256: str = ""
    # Where the masking markers came from: "explicit" | "auto" | "table" |
    # "none" (+ "-unmasked" when apply_masking is off). If every record of a
    # run says "none", masking never engaged — check this field first.
    masking_source: str = ""

    def to_record(self) -> Dict[str, Any]:
        return {
            "row_id": int(self.row_id),
            "example_id": self.example_id,
            "source": self.source,
            "level": self.level,
            "batch_id": self.batch_id,
            "loss": None if self.loss is None else float(self.loss),
            "num_loss_tokens": int(self.num_loss_tokens),
            "checkpoint": str(self.checkpoint),
            "status": str(self.status),
            "reason": str(self.reason),
            "loss_kind": str(self.loss_kind),
            "num_total_tokens": int(self.num_total_tokens),
            "text_sha256": str(self.text_sha256),
            "masking_source": str(self.masking_source),
        }


def find_marker_spans(
    input_ids: Sequence[int],
    marker_ids: Sequence[int],
) -> List[Tuple[int, int]]:
    """All [start, end) windows where ``marker_ids`` occurs in ``input_ids``."""
    needle = [int(v) for v in marker_ids]
    haystack = [int(v) for v in input_ids]
    if not needle or len(needle) > len(haystack):
        return []
    width = len(needle)
    return [
        (i, i + width)
        for i in range(len(haystack) - width + 1)
        if haystack[i : i + width] == needle
    ]


def mask_response_spans(
    input_ids: Sequence[int],
    response_ids: Optional[Sequence[int]] = None,
    instruction_ids: Optional[Sequence[int]] = None,
) -> List[int]:
    """Labels with ``-100`` everywhere outside response spans.

    A response span opens at each ``response_part`` marker occurrence and runs
    to the next ``instruction_part`` occurrence (or end of sequence when the
    instruction marker is unknown). With no usable response marker the whole
    sequence stays masked — the caller then records ``loss=None``, exactly
    like training rows that contribute zero tokens.
    """
    ids = [int(v) for v in input_ids]
    labels = [-100] * len(ids)
    if not response_ids:
        return labels
    for start, _ in find_marker_spans(ids, response_ids):
        end = len(ids)
        if instruction_ids:
            for stop, _ in find_marker_spans(ids, instruction_ids):
                if stop > start:
                    end = stop
                    break
        for position in range(start, end):
            labels[position] = ids[position]
    return labels


def masked_nll(
    logits: Sequence[Sequence[Sequence[float]]],
    labels: Sequence[int],
) -> Tuple[Optional[float], int]:
    """Mean negative log-likelihood over unmasked positions (pure Python).

    ``logits`` is ``[batch][positions][vocab]`` (use batch size 1). Returns
    ``(None, 0)`` when no position is trainable — never ``0.0``.
    """
    if not logits or not logits[0]:
        return None, 0
    row = list(logits[0])
    count = 0
    total = 0.0
    for position, label in enumerate(labels):
        if position >= len(row):
            break
        target = int(label)
        if target < 0:
            continue
        scores = [float(v) for v in row[position]]
        if target >= len(scores) or not scores:
            continue
        peak = max(scores)
        log_denominator = peak + math.log(
            sum(math.exp(score - peak) for score in scores)
        )
        total += log_denominator - float(scores[target])
        count += 1
    if count == 0:
        return None, 0
    return total / count, count


def shift_labels_for_causal_lm(
    input_ids: Sequence[int], mask: Sequence[int]
) -> List[int]:
    """Align labels the way Hugging Face causal-LM loss does.

    ``logits[i]`` predicts token ``i+1``, so position ``i`` trains on
    ``input_ids[i + 1]`` when that next token is unmasked — never on the
    token sitting in its own context. Position 0 is therefore never scored
    and the returned list is one shorter than the input. Without this shift
    the numbers are not comparable to any training loss; uniform-logit unit
    tests cannot catch its absence, which is why it is tested explicitly.
    """
    ids = [int(v) for v in input_ids]
    flags = [1 if m else 0 for m in mask]
    return [
        ids[index + 1] if flags[index + 1] else -100
        for index in range(len(ids) - 1)
    ]


def tokenize_and_mask(
    tokenizer: Any,
    text: str,
    response_ids: Optional[Sequence[int]],
    instruction_ids: Optional[Sequence[int]],
    max_seq_length: int,
) -> Tuple[List[int], List[int]]:
    """Tokenize (no truncation here), keep the head, mask, then HF-shift."""
    encoding = tokenizer(text, add_special_tokens = True)
    input_ids = [int(v) for v in encoding["input_ids"]]
    cap = max(1, int(max_seq_length or 2048))
    input_ids = input_ids[:cap]
    mask = [
        0 if label < 0 else 1
        for label in mask_response_spans(input_ids, response_ids, instruction_ids)
    ]
    return input_ids, shift_labels_for_causal_lm(input_ids, mask)


def prepare_scoring_texts(
    rows: Sequence[Dict[str, Any]],
    *,
    model_name: str,
    tokenizer: Any,
    format_type: str = "auto",
    custom_format_mapping: Optional[Dict[str, Any]] = None,
    text_field: str = "text",
    format_fn: Optional[Callable[..., Any]] = None,
) -> List[Dict[str, Any]]:
    """Rebuild the exact training texts with the training pipeline.

    Runs :func:`utils.datasets.format_and_template_dataset` over the rows
    (lazily imported so this module stays torch-free), then pairs each
    formatted ``text`` back with its ORIGINAL row by position (``map``/``select``
    preserve order) plus that row's metadata. Raises ``ValueError`` when
    formatting itself fails — same fail-loud rule as training.
    """
    rows = [dict(r) for r in rows]
    if format_fn is None:
        # Real path: lazy imports keep this module torch-free at import time.
        try:
            from datasets import Dataset as _Dataset
        except Exception as exc:
            raise ValueError(f"scoring needs the 'datasets' package: {exc}") from exc
        from utils.datasets import format_and_template_dataset as format_fn

        formatted_source = _Dataset.from_list(rows)
    else:
        # Test seam: fakes receive the raw row list and return the same
        # result-dict shape ({"dataset": [...], "success": ...}).
        formatted_source = rows
    result = format_fn(
        formatted_source,
        model_name = model_name,
        tokenizer = tokenizer,
        format_type = format_type or "auto",
        custom_format_mapping = custom_format_mapping,
    )
    if not result.get("success", False):
        errors = result.get("errors", []) or ["dataset formatting failed"]
        raise ValueError("; ".join(str(e) for e in errors))
    if result.get("is_image") or result.get("is_audio"):
        raise ValueError("offline scoring v1 is text-only")
    formatted = result["dataset"]
    if len(formatted) != len(rows):
        raise ValueError(
            f"format pipeline changed row count ({len(rows)} -> {len(formatted)}); "
            "positional metadata join would be wrong"
        )
    prepared: List[Dict[str, Any]] = []
    for index, raw in enumerate(rows):
        text = formatted[index].get(text_field, "") or ""
        prepared.append(
            {
                "row_id": int(index),
                "text": str(text),
                "text_sha256": hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16],
                **{column: raw.get(column) for column in METADATA_COLUMNS},
            }
        )
    return prepared


def resolve_markers(
    tokenizer: Any,
    model_name: str,
    instruction_part: Optional[str] = None,
    response_part: Optional[str] = None,
) -> Tuple[Optional[List[int]], Optional[List[int]], str]:
    """Token id sequences for the masking markers (explicit or detected).

    Mirrors training's precedence in
    ``utils.datasets.completion_masking.apply_completion_masking`` exactly:
    explicit strings win (exact run reproducibility), then unsloth_zoo
    chat-template auto-detection (imported from the SAME
    ``unsloth_zoo.dataset_utils`` path training uses), then the manual
    template table. Returns ``(instruction_ids, response_ids, source)``
    where source is ``"explicit"`` | ``"auto"`` | ``"table"`` | ``"none"``,
    so a run of all-null rows is diagnosable from its own records.
    """
    explicit_instruction = (instruction_part or "").strip() or None
    explicit_response = (response_part or "").strip() or None
    if explicit_instruction is not None or explicit_response is not None:
        return (
            _encode_marker(tokenizer, explicit_instruction),
            _encode_marker(tokenizer, explicit_response),
            "explicit",
        )
    try:
        from unsloth_zoo.dataset_utils import (
            get_chat_template_parts as detect_fn,
        )
    except Exception:
        detect_fn = None  # type: ignore[assignment]
    if detect_fn is not None:
        try:
            parts = detect_fn(tokenizer)
            instruction_ids = _encode_marker(
                tokenizer, parts[0] if len(parts) > 0 else None
            )
            response_ids = _encode_marker(
                tokenizer, parts[1] if len(parts) > 1 else None
            )
            if instruction_ids or response_ids:
                return instruction_ids, response_ids, "auto"
        except Exception:
            pass
    try:
        from utils.datasets.completion_masking import lookup_manual_markers
    except Exception:
        lookup_manual_markers = None  # type: ignore[assignment]
    if lookup_manual_markers is not None:
        try:
            _, instruction_text, response_text = lookup_manual_markers(model_name)
        except Exception:
            instruction_text, response_text = None, None
        if instruction_text or response_text:
            return (
                _encode_marker(tokenizer, instruction_text),
                _encode_marker(tokenizer, response_text),
                "table",
            )
    return None, None, "none"


def _encode_marker(tokenizer: Any, text: Optional[str]) -> Optional[List[int]]:
    if not (text or "").strip():
        return None
    try:
        ids = tokenizer.encode(text, add_special_tokens = False)
    except Exception:
        return None
    ids = [int(v) for v in ids or []]
    return ids or None


def forward_to_logits(model: Any, input_ids: Sequence[int]) -> List[Any]:
    """One forward pass, eval-mode, no gradients. Thin torch adapter.

    Only called by live scoring runs (never by unit tests): imports torch
    lazily, forces ``model.eval()`` + ``torch.no_grad()``, and returns plain
    nested lists so every downstream step stays framework-free.
    """
    import torch  # lazy: keeps this module importable without torch

    model.eval()
    tensor = torch.tensor([list(int(v) for v in input_ids)])
    device = getattr(next(model.parameters(), torch.tensor(0.0)), "device", None)
    if device is not None and getattr(device, "type", "") != "cpu":
        try:
            tensor = tensor.to(device)
        except Exception:
            pass
    with torch.no_grad():
        outputs = model(input_ids = tensor)
    logits = outputs.logits.detach().cpu().tolist()
    return logits


def load_model_for_scoring(
    checkpoint: str,
    device: str = "cuda",
    trust_remote_code: bool = False,
) -> Tuple[Any, Any]:
    """Load ``(model, tokenizer)`` for scoring from a saved output dir.

    Full-weight checkpoints only in v1: adapter-only directories (LoRA
    ``adapter_model.*`` without merged weights) raise an informative error
    instead of scoring the wrong weights. Both objects come from the SAME
    directory training saved, so template and vocabulary match by construction.
    """
    import torch  # lazy
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not (checkpoint or "").strip():
        raise ValueError("offline scoring needs a checkpoint directory")
    try:
        entries = set(os.listdir(checkpoint))
    except Exception as exc:
        raise ValueError(f"cannot read checkpoint directory '{checkpoint}': {exc}") from exc
    adapter_files = {
        "adapter_model.safetensors",
        "adapter_model.bin",
        "adapter_config.json",
    } & entries
    has_full_weights = any(
        name == "model.safetensors"
        or (name.startswith("model-") and name.endswith(".safetensors"))
        or name in ("pytorch_model.bin", "model.safetensors.index.json")
        for name in entries
    )
    if adapter_files and not has_full_weights:
        raise ValueError(
            f"checkpoint '{checkpoint}' holds LoRA adapter files only "
            f"({sorted(adapter_files)}); merge or point at full weights first"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, trust_remote_code = trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype = "auto", trust_remote_code = trust_remote_code
    )
    try:
        model.to(device)
    except Exception:
        pass
    model.eval()
    return model, tokenizer


def score_prepared_text(
    prepared: Dict[str, Any],
    *,
    tokenizer: Any,
    forward_fn: Callable[[Any, List[int]], Any],
    response_ids: Optional[Sequence[int]],
    instruction_ids: Optional[Sequence[int]],
    max_seq_length: int,
    checkpoint: str,
    masking_source: str = "",
    apply_masking: bool = True,
) -> ExampleScore:
    """Score one prepared example: tokenize, mask, forward, NLL over spans."""
    score = ExampleScore(
        row_id = int(prepared.get("row_id", 0)),
        example_id = prepared.get("example_id"),
        source = prepared.get("source"),
        level = prepared.get("level"),
        batch_id = prepared.get("batch_id"),
        checkpoint = str(checkpoint or ""),
        text_sha256 = str(prepared.get("text_sha256", "")),
        masking_source = str(masking_source or ""),
    )
    try:
        text = str(prepared.get("text", "") or "")
        input_ids, labels = tokenize_and_mask(
            tokenizer, text, response_ids, instruction_ids, max_seq_length
        )
        if not apply_masking:
            # Training ran on full sequences: every token trains — still
            # through the causal shift (position i trains on input_ids[i+1]).
            labels = shift_labels_for_causal_lm(
                input_ids, [1] * len(input_ids)
            )
            score.masking_source = (
                f"{masking_source}-unmasked" if masking_source else "unmasked"
            )
        score.num_total_tokens = len(input_ids)
        logits = forward_fn(tokenizer, input_ids)
        loss, count = masked_nll(logits, labels)
        if loss is None:
            score.status = "no_loss_tokens"
            score.reason = (
                "no trainable response tokens after masking/truncation "
                "(mirrors training dropping the row)"
            )
            return score
        score.loss = float(loss)
        score.num_loss_tokens = int(count)
        score.status = "scored"
        score.reason = ""
        return score
    except Exception as exc:
        score.status = "error"
        score.reason = str(exc) or type(exc).__name__
        score.loss = None
        score.num_loss_tokens = 0
        return score


def score_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    config: OfflineScoreConfig,
    model_name: str,
    tokenizer: Any,
    forward_fn: Callable[[Any, List[int]], Any],
    format_type: str = "auto",
    custom_format_mapping: Optional[Dict[str, Any]] = None,
    format_fn: Optional[Callable[..., Any]] = None,
) -> List[Dict[str, Any]]:
    """Score prepared rows end-to-end and return joinable record dicts.

    Markers resolve ONCE (same masking for every row, like training), then
    each row scores independently; per-row failures degrade to ``error``
    records instead of aborting the batch. ``config.apply_masking`` mirrors
    the run's ``train_on_completions``: full-sequence runs score every token.
    """
    response_ids: Optional[Sequence[int]]
    instruction_ids: Optional[Sequence[int]]
    masking_source = "none"
    response_ids, instruction_ids = (None, None)
    try:
        instruction_ids, response_ids, masking_source = resolve_markers(
            tokenizer,
            model_name,
            config.instruction_part,
            config.response_part,
        )
    except Exception:
        instruction_ids, response_ids, masking_source = None, None, "none"
    prepared = prepare_scoring_texts(
        rows,
        model_name = model_name,
        tokenizer = tokenizer,
        format_type = format_type,
        custom_format_mapping = custom_format_mapping,
        text_field = config.text_field,
        format_fn = format_fn,
    )
    records: List[Dict[str, Any]] = []
    for item in prepared:
        records.append(
            score_prepared_text(
                item,
                tokenizer = tokenizer,
                forward_fn = forward_fn,
                response_ids = response_ids,
                instruction_ids = instruction_ids,
                max_seq_length = config.max_seq_length,
                checkpoint = config.checkpoint,
                masking_source = masking_source,
                apply_masking = bool(config.apply_masking),
            ).to_record()
        )
    return records


def write_records_jsonl(records: Iterable[Dict[str, Any]], path: str) -> str:
    """Write ``per_example_loss.jsonl`` (one record per line). Returns path."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok = True)
    count = 0
    with open(path, "w", encoding = "utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii = False) + "\n")
            count += 1
    if count == 0:
        raise ValueError(f"refusing to write empty score file to '{path}'")
    return path


PER_EXAMPLE_FILENAME = "per_example_loss.jsonl"


def read_per_example_records(
    output_dir: Optional[str], limit: int = 20000
) -> Tuple[bool, List[Dict[str, Any]], int]:
    """Read a run's per-example scores for API serving. Never raises.

    Returns ``(exists, records, total)`` with ``records`` capped at ``limit``
    so a pathological file cannot blow up a response. Torn lines are skipped;
    only dict lines survive. ``output_dir`` comes from the run record itself,
    never from user input.
    """
    if not (output_dir or "").strip():
        return False, [], 0
    path = os.path.join(str(output_dir).strip(), PER_EXAMPLE_FILENAME)
    try:
        kept: List[Dict[str, Any]] = []
        total = 0
        with open(path, "r", encoding = "utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                total += 1
                if isinstance(parsed, dict) and len(kept) < max(0, int(limit)):
                    kept.append(parsed)
    except (OSError, ValueError):
        return False, [], 0
    if total == 0:
        return False, [], 0
    return True, kept, total
