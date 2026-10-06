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
import math
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

    # Training metric keys ever attached to step records. Fixed schema: every
    # new record carries all four (None when the logging event had no value),
    # so readers never branch on key presence. Old files simply lack them.
    METRIC_KEYS = ("loss", "smoothed_loss", "grad_norm", "learning_rate")

    def __init__(self, sidecar_path: str, expected_rows: Optional[int] = None) -> None:
        parent = os.path.dirname(os.path.abspath(sidecar_path))
        os.makedirs(parent, exist_ok = True)
        self._path = sidecar_path
        self._handle = open(sidecar_path, "a", encoding = "utf-8")
        # Metrics captured from training logging events, keyed by TRUE
        # global_step (never by array position). See note_metrics.
        self._pending_metrics: Dict[int, Dict[str, Any]] = {}
        # One-step-delayed flush (see finalize_optimizer_step): the record
        # built now is written when the NEXT step finalizes (or at close).
        self._held_record: Optional[Dict[str, Any]] = None
        base, ext = os.path.splitext(sidecar_path)
        if os.path.basename(sidecar_path) == SIDECAR_FILENAME:
            timeline_path = os.path.join(parent, TIMELINE_FILENAME)
        else:
            timeline_path = base + "_timeline" + (ext or ".jsonl")
        self._timeline_path = timeline_path
        self._timeline_handle = open(timeline_path, "a", encoding = "utf-8")
        self._pending: List[Tuple[int, List[int], float, bool]] = []
        # Forensic counters only (never training state): completed optimizer
        # marks observed, for the expected-vs-actual diagnostic line.
        self._optimizer_marks = 0
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

    @property
    def optimizer_mark_count(self) -> int:
        return int(self._optimizer_marks)

    @property
    def total_micros(self) -> int:
        return int(self._micro_seq)

    def mark_event(self, name: str, global_step: Optional[int] = None) -> None:
        """Append one lifecycle event (callback timing) to the timeline file.

        Observation only. Kept in a SEPARATE file so step records stay clean
        and existing readers/joins keep working untouched.
        """
        if name == "optimizer_step":
            try:
                self._optimizer_marks += 1
            except Exception:
                pass
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

    def note_metrics(
        self,
        global_step: int,
        *,
        loss: Any = None,
        smoothed_loss: Any = None,
        grad_norm: Any = None,
        learning_rate: Any = None,
    ) -> None:
        """Stash one step's training metrics, keyed by TRUE global_step.

        Values must come straight from the training logging event (the same
        source the Charts render) — this function never computes, rounds, or
        invents anything; see _sanitize_metric_value. Never array position:
        ``logging_steps > 1`` leaves honest gaps instead of shifted numbers.
        """
        self._pending_metrics[int(global_step)] = {
            "loss": _sanitize_metric_value(loss),
            "smoothed_loss": _sanitize_metric_value(smoothed_loss),
            "grad_norm": _sanitize_metric_value(grad_norm),
            "learning_rate": _sanitize_metric_value(learning_rate),
        }

    def _attach_metrics(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich a built record with stashed metrics for its own step only.

        Later information never leaks across steps: entries keyed by other
        steps stay parked for their own flush. Missing entries keep None.
        """
        known = self._pending_metrics.pop(int(record.get("step", -1)), None) or {}
        for key in self.METRIC_KEYS:
            value = known.get(key)
            if value is not None:
                record[key] = value
        return record

    def _write_record(self, record: Dict[str, Any]) -> None:
        # The sidecar is a BATCH-composition log, not a per-global-step log:
        # records with no actual batch (empty micro_batches/row_ids, e.g. the
        # trailing record of each epoch) are never persisted. Their step
        # metrics live on untouched in lossHistory/progress; nothing is moved
        # to another record and no fake attribution is ever synthesized.
        if (
            not record.get("micro_batches")
            and not record.get("row_ids")
            and not record.get("num_rows")
        ):
            return
        self._handle.write(json.dumps(record) + "\n")
        self._handle.flush()

    def finalize_optimizer_step(self, global_step: int) -> Dict[str, Any]:
        """Build this step's record; write the PREVIOUS step's record first.

        Why the one-step delay: the training logging event for step N fires
        AFTER step N's step-end callback, so metrics for N only exist once
        step N+1 finalizes (or the run closes). Enrichment therefore happens
        at flush time from step-keyed metrics — correct no matter which of
        the two callbacks fires first. The returned record is the freshly
        built one (metric keys present, values attached when already known).

        Crash window: a hard kill loses at most the single trailing record;
        normal completion (including graceful stop) flushes everything via
        close(). Resume appends new steps; numbering never restarts here.
        """
        if self._held_record is not None:
            held, self._held_record = self._held_record, None
            self._write_record(self._attach_metrics(held))
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
            "loss": None,
            "smoothed_loss": None,
            "grad_norm": None,
            "learning_rate": None,
        }
        self._pending = []
        self._attach_metrics(record)
        self._held_record = record
        return record

    def close(self) -> None:
        if self._held_record is not None:
            held, self._held_record = self._held_record, None
            try:
                self._write_record(self._attach_metrics(held))
            except Exception:
                pass
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


def expected_optimizer_step_count(
    *,
    num_rows: Any = None,
    batch_size: Any = None,
    num_epochs: Any = None,
    grad_accum: Any = None,
    max_steps: Any = None,
) -> Tuple[Optional[int], str]:
    """Expected optimizer steps from training arithmetic (pure, no I/O).

    ``ceil(rows / batch)`` micros per epoch times epochs, divided by the
    grad-accum horizon. Returns ``(steps | None, note)``: None with a reason
    whenever the inputs are missing/invalid, indivisible, or truncated by
    ``max_steps`` is itself the cap. Single process, ``drop_last=False``
    (the pilot shape); anything else surfaces as an explicit note, never a
    silent number.
    """
    try:
        rows = int(num_rows)  # type: ignore[arg-type]
        per_batch = int(batch_size)  # type: ignore[arg-type]
        epochs = int(num_epochs)  # type: ignore[arg-type]
        accum = int(grad_accum)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None, "missing or non-numeric inputs"
    if rows <= 0 or per_batch <= 0 or epochs <= 0 or accum <= 0:
        return None, "non-positive inputs"
    micros = -(-rows // per_batch) * epochs
    full, remainder = divmod(micros, accum)
    note = f"{micros} micros ÷ {accum}"
    if remainder:
        return None, note + " is not whole (trailing partial group)"
    try:
        cap = int(max_steps) if max_steps else 0  # type: ignore[arg-type]
    except (TypeError, ValueError):
        cap = 0
    if cap > 0 and full > cap:
        return cap, note + f" capped by max_steps={cap}"
    return full, note


def diagnostic_summary_line(
    *,
    grad_accum: Any = None,
    num_rows: Any = None,
    batch_size: Any = None,
    num_epochs: Any = None,
    expected_steps: Optional[int] = None,
    expected_note: str = "",
    actual_marks: Any = None,
    actual_micros: Any = None,
) -> str:
    """One grep-able diagnostic line: config expectation vs observed reality.

    Pure string building (no I/O): the caller logs it. ``None`` renders as
    ``?`` so missing data is visible, never blank.
    """
    def _show(value: Any) -> str:
        return "?" if value is None else str(value)

    line = (
        "Batch composition diagnostic: "
        f"grad_accum={_show(grad_accum)}, "
        f"examples={_show(num_rows)}, "
        f"batch={_show(batch_size)}, "
        f"epochs={_show(num_epochs)}, "
        f"expected_optimizer_steps={_show(expected_steps)}"
    )
    if expected_note:
        line += f" ({expected_note})"
    line += (
        f", actual_optimizer_step_marks={_show(actual_marks)}, "
        f"actual_micro_batches={_show(actual_micros)}"
    )
    return line


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
      (detected data-driven: first record overall, previous record empty, or
      a step-numbering gap where an empty trailing record was suppressed at
      write time — an empty record only ever closes an epoch, since every
      training step consumes at least one micro-batch); otherwise
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
        try:
            prev_step = (
                int(previous.get("step", step - 1))
                if previous is not None
                else None
            )
        except (TypeError, ValueError):
            prev_step = step - 1
        gap = (
            previous is not None
            and prev_step != step
            and prev_step != step - 1
        )
        if prev_empty or gap:
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


def present_step_attribution(
    records: Iterable[Dict[str, Any]],
    losses: Iterable[Tuple[int, float]] = (),
) -> List[Dict[str, Any]]:
    """Presentation layer: one truthful row per recorded step.

    Combines the approved forensic rule (:func:`attribute_steps`) with step
    losses keyed by TRUE step (same keying as ``lossHistory`` — never array
    position) into::

        {trainer_step, loss, attributed_row_ids, attributed_seqs, rule}

    * ``trainer_step`` is never renumbered: it stays aligned with the chart
      axis, ``lossHistory`` keys, ``checkpoint-N`` names and resume.
    * The raw records (and the sidecar file) are never modified; steps whose
      record holds no batch attribute no rows (``rule`` says why).
    * Read-only transform for display/diagnosis. No training behavior reads
      or depends on it.
    """
    attributed = attribute_steps(records)
    loss_by_step: Dict[int, float] = {}
    for step, value in losses:
        try:
            loss_by_step[int(step)] = float(value)
        except (TypeError, ValueError):
            continue
    view: List[Dict[str, Any]] = []
    for entry in attributed:
        step = int(entry.get("step", -1))
        view.append(
            {
                "trainer_step": step,
                "loss": loss_by_step.get(step),
                "attributed_row_ids": list(entry.get("attributed_row_ids", [])),
                "attributed_seqs": list(entry.get("attributed_seqs", [])),
                "rule": str(entry.get("rule", "none")),
            }
        )
    return view


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


def _sanitize_metric_value(value: Any) -> Optional[float]:
    """Keep a metric exactly as reported, or None when it has no number.

    No rounding, no recomputation, no invention: plain int/float values pass
    through bit-identical (``==`` holds against the logging event). Booleans,
    strings, NaN/inf (unrepresentable in strict JSON) and anything else become
    ``None`` — the honest "unavailable", never a fabricated zero.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


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


def wrap_optimizer_step(optimizer: Any, on_optimizer_step: Callable[[], None]) -> bool:
    """Count REAL optimizer steps without changing training behavior.

    Replaces ``optimizer.step`` with a pass-through wrapper that forwards
    every positional/keyword argument, returns the original result, and only
    then invokes ``on_optimizer_step()`` (used to mark the timeline). Any
    failure — missing ``.step``, read-only attribute, or an exception inside
    the callback — leaves the optimizer exactly as found and returns False.
    Training can never break from diagnostics: the mark path is doubly
    guarded (here and inside ``mark_event``).
    """
    try:
        step_method = getattr(optimizer, "step", None)
    except Exception:
        return False
    if not callable(step_method):
        return False

    def _counting_step(*args: Any, **kwargs: Any) -> Any:
        result = step_method(*args, **kwargs)
        try:
            on_optimizer_step()
        except Exception:
            pass
        return result

    try:
        optimizer.step = _counting_step  # type: ignore[method-assign]
    except Exception:
        return False
    return True


def resolve_optimizer_steps(
    micro_entries: Sequence[Tuple[int, List[int]]],
    micros_per_step: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Group collation-ordered micros into true optimizer steps (read-time).

    ``micro_entries`` is ``[(seq, row_ids)]`` in arrival order (e.g. expanded
    from sidecar records); ``micros_per_step`` K is the micros each optimizer
    step consumed. Grouping is purely positional — timestamps and prefetch
    timing never enter it — so prefetch can neither help nor corrupt it.

    Validation is strict and loud (fail-loud, never silent misattribution):
    K must be a positive int; seqs must be contiguous with no gaps or
    duplicates; every group holds exactly K micros (no empties by
    construction); the flat rows across groups must equal the input multiset
    (no loss, no duplication). Returns ``(groups, report)`` where each group
    is ``{optimizer_step (1-based ordinal), micro_seqs, row_ids, num_micros,
    num_rows}``.
    """
    try:
        width = int(micros_per_step)
    except (TypeError, ValueError):
        raise ValueError(
            f"micros_per_step must be a positive int (got {micros_per_step!r})"
        ) from None
    if width <= 0:
        raise ValueError(
            f"micros_per_step must be a positive int (got {micros_per_step!r})"
        )
    entries = [(int(seq), [int(r) for r in rows]) for seq, rows in micro_entries]
    seqs_only = [seq for seq, _ in entries]
    for first, second in zip(seqs_only, seqs_only[1:]):
        if second != first + 1:
            raise ValueError(
                f"micro seqs not contiguous ({first} -> {second}); cannot "
                "attribute without inventing order"
            )
    if entries and len(entries) % width != 0:
        raise ValueError(
            f"{len(entries)} micros do not split into whole steps of "
            f"{width}; refusing partial attribution"
        )
    groups: List[Dict[str, Any]] = []
    for ordinal in range(len(entries) // width if entries else 0):
        chunk = entries[ordinal * width : (ordinal + 1) * width]
        chunk_rows: List[int] = [r for _, rows in chunk for r in rows]
        groups.append(
            {
                "optimizer_step": ordinal + 1,
                "micro_seqs": [seq for seq, _ in chunk],
                "row_ids": chunk_rows,
                "num_micros": len(chunk),
                "num_rows": len(chunk_rows),
            }
        )
    flat_in = sorted(r for _, rows in entries for r in rows)
    flat_out = sorted(r for group in groups for r in group["row_ids"])
    report = {
        "micros_per_step": width,
        "num_groups": len(groups),
        "total_micros": len(entries),
        "total_rows": len(flat_in),
        "contiguous": True,
        "complete": flat_in == flat_out,
    }
    if flat_in != flat_out:
        raise ValueError("grouping lost or duplicated rows; refusing result")
    return groups, report


def present_optimizer_steps(
    groups: Sequence[Dict[str, Any]],
    steps: Sequence[int],
    losses_by_step: Optional[Dict[int, Any]] = None,
) -> List[Dict[str, Any]]:
    """Attach training metrics to TRUE-step groups by EXPLICIT step numbers.

    ``groups[i]`` belongs to ``global_step steps[i]`` — the caller asserts
    this correspondence (e.g. from optimizer-step marks or from the
    lossHistory cadence proof), it is never inferred here. Lengths must match
    and steps must strictly increase, otherwise ValueError: silently pairing
    a group with the wrong step's loss would be fabricated attribution.
    ``losses_by_step`` mirrors ``lossHistory`` (``{global_step: loss}``);
    missing entries stay an honest ``None``. Read-only presentation.
    """
    group_list = list(groups)
    step_list = [int(s) for s in steps]
    if len(group_list) != len(step_list):
        raise ValueError(
            f"{len(group_list)} groups cannot pair with {len(step_list)} steps"
        )
    for first, second in zip(step_list, step_list[1:]):
        if second <= first:
            raise ValueError(
                f"global steps must strictly increase (got {step_list})"
            )
    losses = losses_by_step or {}
    view: List[Dict[str, Any]] = []
    for group, step in zip(group_list, step_list):
        view.append(
            {
                "optimizer_step": int(group.get("optimizer_step", 0)),
                "global_step": int(step),
                "loss": losses.get(step),
                "micro_seqs": list(group.get("micro_seqs", [])),
                "row_ids": list(group.get("row_ids", [])),
                "num_micros": int(group.get("num_micros", 0)),
                "num_rows": int(group.get("num_rows", 0)),
            }
        )
    return view
