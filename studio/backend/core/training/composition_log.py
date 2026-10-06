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
import time
import traceback
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ROW_ID_COLUMN = "__row_id__"

# Granularity contract: with gradient_accumulation_steps=k, one optimizer step
# consumes exactly k micro-batches and the logged loss is their mean. Records
# keep both levels so the join never pretends a micro-batch loss exists.
#
# Ordering contract (read before re-adding any capture gate): in the HF
# training loop the next batch is collated BEFORE on_step_begin fires for its
# step (the for-loop pulls ``next(dataloader)`` first, then runs the step
# body). A gate toggled in on_step_begin/on_step_end is therefore ALWAYS one
# micro-batch late: every batch is collated while capture still reflects the
# previous step, and the sidecar fills with perfect empty records. Recording
# must be unconditional; train/eval separation comes from the key itself
# (eval splits are never stamped), plus a defensive reset() on evaluate.


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

    * :meth:`record_micro_batch` on every collator call carrying ids;
    * :meth:`finalize_optimizer_step` with the Trainer's ``global_step`` once
      per optimizer step (whatever is buffered belongs to that step);
    * :meth:`reset` on evaluate/predict as a safety net.

    Recording is deliberately UNGATED (see the ordering contract above):
    batches without the id column (eval splits are never stamped) record
    nothing on their own. Records are keyed by ``global_step`` — the same
    number ``lossHistory`` uses — so resume never duplicates: already-written
    steps are simply history in an append-only file.
    """

    def __init__(self, sidecar_path: str, expected_rows: Optional[int] = None) -> None:
        parent = os.path.dirname(os.path.abspath(sidecar_path))
        os.makedirs(parent, exist_ok = True)
        self._path = sidecar_path
        self._handle = open(sidecar_path, "a", encoding = "utf-8")
        base, ext = os.path.splitext(sidecar_path)
        if os.path.basename(sidecar_path) == SIDECAR_FILENAME:
            timeline_path = os.path.join(parent, TIMELINE_FILENAME)
        else:
            timeline_path = base + "_timeline" + (ext or ".jsonl")
        self._timeline_path = timeline_path
        self._timeline_handle = open(timeline_path, "a", encoding = "utf-8")
        self._pending: List[Tuple[int, List[int], float, bool]] = []
        # Monotonic arrival counter, NEVER reset (not by finalize, not by
        # reset()): it numbers every collator call in arrival order, so a
        # later audit can tell an extra collation apart from a late finalize.
        # If exactly the expected micro-batches were collated, seqs are
        # contiguous 0..N-1 across the whole file; any gap or duplicate
        # proves out-of-band pulls.
        self._micro_seq = 0
        # Row count at stamp time. Used after transforms that may drop columns
        # (Unsloth response masking) to decide whether a positional re-stamp is
        # still exact (same length) or must be refused (rows were filtered).
        self.expected_rows = expected_rows

    @property
    def path(self) -> str:
        return self._path

    def record_micro_batch(self, row_ids: Sequence[int]) -> None:
        ids = [int(r) for r in row_ids]
        if ids:
            # Observation only: wall time + preflight fingerprint per micro.
            # Never branches training logic; the collator output is unchanged.
            self._pending.append(
                (self._micro_seq, ids, time.time(), _collated_via_preflight())
            )
            self._micro_seq += 1

    def mark_event(self, name: str, global_step: Optional[int] = None) -> None:
        """Append one lifecycle event (callback timing) to the timeline file.

        Observation only. Kept in a SEPARATE file so step records stay clean
        and existing readers/joins keep working untouched.
        """
        try:
            self._timeline_handle.write(
                json.dumps(
                    {
                        "type": "event",
                        "event": str(name),
                        "step": None if global_step is None else int(global_step),
                        "t": time.time(),
                    }
                )
                + "\n"
            )
            self._timeline_handle.flush()
        except Exception:
            pass

    def reset(self) -> int:
        """Drop buffered micro-batches (evaluate/predict safety net).

        Returns the number of dropped micro-batches so callers can log it.
        """
        dropped = len(self._pending)
        self._pending = []
        return dropped

    def finalize_optimizer_step(self, global_step: int) -> Dict[str, Any]:
        """Flush buffered micro-batches as one step record. Always writes."""
        flat: List[int] = [r for _, micro, _, _ in self._pending for r in micro]
        record = {
            "step": int(global_step),
            "micro_batches": [list(micro) for _, micro, _, _ in self._pending],
            # Arrival order of each micro-batch; see _micro_seq contract above.
            "micro_seqs": [seq for seq, _, _, _ in self._pending],
            # Diagnostic only (forensic run): collation wall time + preflight
            # fingerprint per micro, parallel to micro_batches.
            "micro_t": [stamp for _, _, stamp, _ in self._pending],
            "micro_via_preflight": [via for _, _, _, via in self._pending],
            "row_ids": list(flat),
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
        try:
            self._timeline_handle.flush()
        finally:
            self._timeline_handle.close()


SIDECAR_FILENAME = "batch_composition.jsonl"
TIMELINE_FILENAME = "batch_composition_timeline.jsonl"

# Frame name that proves a collator call came from the pre-train probe
# (trainer._preflight_first_batch). Checked by function NAME so the check
# stays stdlib-only and import-free.
_PREFLIGHT_FRAME_NAME = "_preflight_first_batch"


def _collated_via_preflight() -> bool:
    """True when the current collator call runs under the pre-train probe."""
    try:
        for frame in traceback.extract_stack():
            if frame.name == _PREFLIGHT_FRAME_NAME:
                return True
    except Exception:
        pass
    return False


def read_run_composition(
    output_dir: Optional[str], limit: int = 50000
) -> Tuple[bool, List[Dict[str, Any]], int]:
    """Read a run's sidecar for API serving. Never raises for missing files.

    Returns ``(exists, records, total)`` with ``records`` capped at ``limit``
    so a pathological file cannot blow up a response. ``output_dir`` comes
    from the run record itself, never from user input.
    """
    if not (output_dir or "").strip():
        return False, [], 0
    sidecar = os.path.join(str(output_dir).strip(), SIDECAR_FILENAME)
    records = read_composition_records(sidecar)
    if not records:
        return False, [], 0
    return True, records[: max(0, int(limit))], len(records)


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


def attribute_steps(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attribute each optimizer step to the micro-batch(es) that fed its loss.

    Proven rule (forensic runs, grad_accum=1): the training loop collates one
    extra micro-batch just before the first step of each epoch, so step
    records group as ``[2, 1, 1, 0, ...]`` micros instead of ``[1, 1, 1, 1]``.
    Therefore, loss(step N) was computed from:

    * the FIRST micro of its own record when the step opens an epoch
      (detected data-driven: first record overall, or previous record empty —
      an empty record only ever closes an epoch, since every training step
      consumes at least one micro-batch); otherwise
    * the LAST micro of the PREVIOUS step's record.

    Returns one entry per input record: ``step``, ``attributed_row_ids``,
    ``attributed_seqs`` and the ``rule`` branch taken (``epoch_first`` /
    ``prev_tail`` / ``none``) so every attribution stays auditable. Pure
    read-time transform: no training behavior depends on it.
    """
    ordered = sorted(
        (dict(r) for r in records), key = lambda r: int(r.get("step", -1))
    )
    attributed: List[Dict[str, Any]] = []
    for index, record in enumerate(ordered):
        step = int(record.get("step", -1))
        own_micros = [list(m) for m in record.get("micro_batches", []) or []]
        own_seqs = list(record.get("micro_seqs", []) or [])
        previous = ordered[index - 1] if index > 0 else None
        prev_empty = previous is None or not (previous.get("row_ids") or [])
        if prev_empty:
            if own_micros:
                seq = (
                    own_seqs[0]
                    if len(own_seqs) == len(own_micros)
                    else None
                )
                chosen = [(seq, own_micros[0])]
            else:
                chosen = []
            rule = "epoch_first" if own_micros else "none"
        else:
            prev_micros = [list(m) for m in previous.get("micro_batches", []) or []]
            prev_seqs = list(previous.get("micro_seqs", []) or [])
            if prev_micros and prev_seqs and len(prev_micros) == len(prev_seqs):
                chosen = [(prev_seqs[-1], prev_micros[-1])]
            elif prev_micros:
                chosen = [(None, prev_micros[-1])]
            else:
                chosen = []
            rule = "prev_tail" if chosen else "none"
        chosen_seqs = [seq for seq, _ in chosen]
        chosen_rows: List[int] = [r for _, micro in chosen for r in micro]
        attributed.append(
            {
                "step": step,
                "attributed_row_ids": chosen_rows,
                "attributed_seqs": chosen_seqs,
                "rule": rule,
            }
        )
    return attributed


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
    untouched and record nothing: key presence, not a capture gate, is what
    separates train from eval batches (see the ordering contract above).
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


def ensure_row_ids(
    dataset: Any,
    expected_rows: Optional[int],
) -> Tuple[Any, str]:
    """Re-check the id column after transforms that may drop columns.

    Unsloth's ``train_on_responses_only`` masking drops every non-model
    column from ``trainer.train_dataset`` — including a ``__row_id__`` stamped
    earlier. Without this check the collator would silently record empty rows.

    Returns ``(dataset, action)`` with ``action`` one of ``"present"`` (column
    survived, dataset untouched) or ``"restamped"`` (column was dropped but no
    row was filtered, so a positional re-stamp is exact: filtering preserves
    order, and equal length means equal positions).

    Raises ``ValueError`` when attribution would be wrong: unsized dataset,
    row count changed (rows were filtered out), or no way to re-add the column.
    Fail fast, never record unattributable steps.
    """
    columns = list(getattr(dataset, "column_names", None) or [])
    if ROW_ID_COLUMN in columns:
        return dataset, "present"
    try:
        current_len = len(dataset)
    except Exception as exc:
        raise ValueError(
            "Batch composition tracking lost __row_id__ after response "
            f"masking and the dataset has no length to re-stamp: {exc}"
        ) from exc
    if expected_rows is None or int(current_len) != int(expected_rows):
        raise ValueError(
            "Batch composition tracking lost __row_id__ after response "
            f"masking and rows were filtered ({expected_rows} -> "
            f"{current_len}); refusing to record unattributable steps."
        )
    if not hasattr(dataset, "add_column"):
        raise ValueError(
            "Batch composition tracking lost __row_id__ after response "
            "masking and the dataset cannot be re-stamped."
        )
    return dataset.add_column(ROW_ID_COLUMN, list(range(int(current_len)))), "restamped"


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
