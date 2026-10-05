# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Batch composition tracking (Pilot: text path, ``packing=False`` only).

Answers "which examples were in the batch behind this loss?" without changing
what the model learns:

* a ``__row_id__`` integer column identifies each row of the final training
  split (stable under filtering/shuffling because it is an id, not a position);
* :class:`RowIdRecordingCollator` records the ids of every micro-batch, then
  **pops the column before delegating** — it provably never reaches the model;
* :class:`CompositionRecorder` aggregates micro-batches into optimizer-step
  records (``step -> row_ids``) in a JSONL sidecar opened in **append** mode,
  so resume continues the same file instead of overwriting it;
* :func:`join_with_losses` merges those records with the existing loss history
  (``lossHistory`` is already keyed by ``global_step`` on both sides).

Stdlib only: no torch / datasets / transformers imports, so this module and
its tests run anywhere.

Out of scope by design (fail fast with ``ValueError``):
packing (chunks merge several examples), streaming datasets (no stable rows),
and online/worker-side tokenization (the id column may not survive it).
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ROW_ID_COLUMN = "__row_id__"

# Granularity contract: with gradient_accumulation_steps=k, one optimizer step
# consumes exactly k micro-batches and the logged loss is their mean. Records
# keep both levels so the join never pretends a micro-batch loss exists.


def validate_tracking_prerequisites(
    *,
    packing: bool,
    is_streaming: bool,
    online_tokenization_enabled: bool,
    dataset_length: Optional[int] = None,
) -> None:
    """Fail fast when composition tracking cannot be exact. Never warns-and-continues."""
    if packing:
        raise ValueError(
            "Batch composition tracking requires packing=False: packed chunks "
            "merge several examples into one sequence, so a chunk cannot be "
            "attributed to a single example_id."
        )
    if is_streaming:
        raise ValueError(
            "Batch composition tracking requires a non-streaming dataset: "
            "streams have no stable row indices to record."
        )
    if online_tokenization_enabled:
        raise ValueError(
            "Batch composition tracking requires eager tokenization: with "
            "online (worker-side) tokenization the __row_id__ column may not "
            "survive to the data loader."
        )
    if dataset_length is not None and dataset_length <= 0:
        raise ValueError("Batch composition tracking needs a non-empty training dataset")


class CompositionRecorder:
    """Append-only JSONL writer: micro-batches in, optimizer-step records out.

    The caller (collator wrapper + TrainerCallback) drives the lifecycle:

    * ``set_capturing(True)`` at train-step begin, ``False`` at step end/eval;
    * :meth:`record_micro_batch` once per collator call while capturing;
    * :meth:`finalize_optimizer_step` with the Trainer's ``global_step``.

    Records are keyed by ``global_step`` — the same number ``lossHistory``
    uses — so resume never duplicates: already-written steps are simply
    history in an append-only file.
    """

    def __init__(self, sidecar_path: str) -> None:
        parent = os.path.dirname(os.path.abspath(sidecar_path))
        os.makedirs(parent, exist_ok = True)
        self._path = sidecar_path
        self._handle = open(sidecar_path, "a", encoding = "utf-8")
        self._capturing = False
        self._pending: List[List[int]] = []

    @property
    def path(self) -> str:
        return self._path

    @property
    def capturing(self) -> bool:
        return self._capturing

    def set_capturing(self, enabled: bool) -> None:
        if not enabled:
            self._pending = []
        self._capturing = bool(enabled)

    def record_micro_batch(self, row_ids: Sequence[int]) -> None:
        if not self._capturing:
            return
        ids = [int(r) for r in row_ids]
        if ids:
            self._pending.append(ids)

    def finalize_optimizer_step(self, global_step: int) -> Dict[str, Any]:
        """Flush buffered micro-batches as one step record. Always writes."""
        flat: List[int] = [r for micro in self._pending for r in micro]
        record = {
            "step": int(global_step),
            "micro_batches": [list(m) for m in self._pending],
            "row_ids": flat,
            "num_micro_batches": len(self._pending),
            "num_rows": len(flat),
        }
        self._handle.write(json.dumps(record) + "\n")
        self._handle.flush()
        self._pending = []
        return record

    def close(self) -> None:
        try:
            self._handle.flush()
        finally:
            self._handle.close()


def read_composition_records(sidecar_path: str) -> List[Dict[str, Any]]:
    """Read back every step record (tolerates a torn last line)."""
    records: List[Dict[str, Any]] = []
    if not os.path.exists(sidecar_path):
        return records
    with open(sidecar_path, "r", encoding = "utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def join_with_losses(
    records: Iterable[Dict[str, Any]],
    losses: Iterable[Tuple[int, float]],
) -> List[Dict[str, Any]]:
    """Merge composition records with loss points on optimizer step.

    ``losses`` mirrors ``lossHistory``: ``(global_step, loss)`` pairs. Steps
    without a loss point (e.g. ``logging_steps > 1``) keep ``loss = None``
    instead of inventing one.
    """
    loss_by_step = {int(step): float(value) for step, value in losses}
    joined: List[Dict[str, Any]] = []
    for record in records:
        step = int(record.get("step", -1))
        joined.append(
            {
                "step": step,
                "loss": loss_by_step.get(step),
                "row_ids": list(record.get("row_ids", [])),
                "micro_batches": [list(m) for m in record.get("micro_batches", [])],
                "num_rows": int(record.get("num_rows", 0)),
            }
        )
    joined.sort(key = lambda row: row["step"])
    return joined


class RowIdRecordingCollator:
    """Wrap any data collator: record ``__row_id__``, then remove it.

    The base collator receives exactly the features it would have received had
    tracking been off (minus the id column), so model inputs — and therefore
    gradients — are byte-identical with tracking on or off. Batches without
    the column (e.g. an eval split that was never stamped) pass through
    untouched and record nothing.
    """

    def __init__(
        self,
        base_collate_fn: Callable[[List[Any]], Any],
        recorder: Optional[CompositionRecorder] = None,
    ) -> None:
        if not callable(base_collate_fn):
            raise ValueError("RowIdRecordingCollator needs a callable base collator")
        self._base = base_collate_fn
        self._recorder = recorder

    @property
    def recorder(self) -> Optional[CompositionRecorder]:
        return self._recorder

    def __call__(self, features: List[Any]) -> Any:
        row_ids: List[int] = []
        cleaned: List[Any] = []
        for feature in features:
            if isinstance(feature, dict) and ROW_ID_COLUMN in feature:
                try:
                    row_ids.append(int(feature[ROW_ID_COLUMN]))
                except (TypeError, ValueError):
                    pass
                rest = dict(feature)
                del rest[ROW_ID_COLUMN]
                cleaned.append(rest)
            else:
                cleaned.append(feature)
        if self._recorder is not None and row_ids:
            self._recorder.record_micro_batch(row_ids)
        return self._base(cleaned)


def stamp_row_ids(
    column_names: Sequence[str],
    num_rows: int,
    text_field: str = "text",
) -> Tuple[List[str], List[int]]:
    """Decide the post-stamp schema: keep the text field, add the id column.

    Returns ``(keep_columns, row_ids)``. Callers prune every other column so
    that, with ``remove_unused_columns=False``, the base collator still sees
    exactly what it saw before tracking existed.
    """
    names = [str(c) for c in column_names]
    if text_field not in names:
        raise ValueError(
            f"Batch composition tracking needs a '{text_field}' column "
            f"(found: {names})"
        )
    return [text_field, ROW_ID_COLUMN], list(range(int(num_rows)))


def prune_columns_for_tracking(
    dataset: Any,
    keep: Sequence[str],
) -> Tuple[Any, List[str]]:
    """Drop every column except ``keep``. Returns ``(dataset, pruned_names)``.

    Operates on the in-memory Arrow schema only (no data rewrite) and only
    ever runs when tracking is explicitly enabled; the source files and the
    user's column mapping are untouched.
    """
    existing = list(getattr(dataset, "column_names", None) or [])
    drop = [c for c in existing if c not in set(keep)]
    if drop and hasattr(dataset, "remove_columns"):
        dataset = dataset.remove_columns(drop)
    return dataset, drop
