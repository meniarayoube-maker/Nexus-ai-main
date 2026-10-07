# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Background per-example scoring jobs (Pilot: user-triggered only).

Nothing here starts on its own: the UI button POSTs, this module validates
synchronously (run exists, output dir readable, no training hogging the GPU,
no duplicate job) and only then spawns ONE daemon thread per run. The thread
loads the ALREADY-SAVED checkpoint + tokenizer, scores every dataset row
with :mod:`core.training.offline_scoring` (eval mode, no gradients, no
training loop anywhere near), and writes ``per_example_loss.jsonl``
atomically (temp file + rename) so readers never see a half-written file.

Safety rails (all tested):
* VRAM guard: refuses while training is active (injectable check).
* Duplicate guard: one live job per run; reruns allowed after terminal state.
* Row cap: refuses absurd datasets with a clear message instead of OOMing.
* Crash safety: exceptions become ``error`` status + message, never a hang;
  terminal jobs older than an hour are pruned on access.
* No persistence: job state is in-memory only. The sidecar FILE is the source
  of truth across restarts (``exists`` in the read endpoint).

Top-level imports are stdlib-only; torch/transformers/datasets enter lazily
inside the worker (and are fully injectable for tests).
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

PER_EXAMPLE_FILENAME = "per_example_loss.jsonl"

# Pilot-scale guardrail: refuse (with a message) instead of OOMing the GPU.
MAX_SCORING_ROWS = 20000

# Terminal job records older than this are dropped on access (memory hygiene;
# the sidecar file remains the durable source of truth).
_JOB_TTL_SECONDS = 3600.0

_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


def _now() -> float:
    return time.time()


def _prune_terminal_locked() -> None:
    cutoff = _now() - _JOB_TTL_SECONDS
    stale = [
        run_id
        for run_id, job in _JOBS.items()
        if job.get("status") in ("done", "error")
        and float(job.get("finished_at", 0.0) or 0.0) < cutoff
    ]
    for run_id in stale:
        del _JOBS[run_id]


def _snapshot_locked(job: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": str(job.get("status", "queued")),
        "done": int(job.get("done", 0) or 0),
        "total": int(job.get("total", 0) or 0),
        "message": str(job.get("message", "") or ""),
    }


def get_scoring_job(run_id: str) -> Optional[Dict[str, Any]]:
    """Current job snapshot for a run (None when never started/pruned)."""
    with _JOBS_LOCK:
        _prune_terminal_locked()
        job = _JOBS.get(str(run_id))
        return _snapshot_locked(job) if job is not None else None


def _default_training_active() -> bool:
    from core.training.training import get_training_backend

    try:
        return bool(get_training_backend().is_training_active())
    except Exception:
        # Unknown counts as busy: never risk VRAM contention on a guess.
        return True


def _read_run_config(output_dir: str) -> Dict[str, Any]:
    path = os.path.join(output_dir, "run-config.json")
    with open(path, "r", encoding = "utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("run-config.json is not an object")
    return data


def _resolve_scoring_source(run_config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Decide HOW to load a run's dataset rows (pure: no I/O beyond exists checks).

    Returns ``("upload", {"files": [...]})`` for local files or
    ``("huggingface", {"hf_dataset": ..., "subset": ..., "train_split": ...,
    "hf_token": ...})``. Anything else (s3, unknown, or nothing usable)
    raises a message that names exactly what was found, so the UI can show
    why scoring refused instead of failing silently.
    """
    source = str(run_config.get("dataset_source") or "").strip().lower()
    names = [
        str(item or "").strip()
        for item in (run_config.get("local_datasets") or [])
        if str(item or "").strip()
    ]
    hf_dataset = str(run_config.get("hf_dataset") or "").strip()
    if source in ("upload", "local", "") and names:
        return "upload", {"files": _resolve_local_files(names)}
    if source == "huggingface" or (not names and hf_dataset):
        if not hf_dataset:
            raise ValueError(
                "run uses a Hugging Face dataset but run-config has no "
                "hf_dataset id"
            )
        return "huggingface", {
            "hf_dataset": hf_dataset,
            "subset": (str(run_config.get("subset") or "").strip() or None),
            "train_split": (
                str(run_config.get("train_split") or "").strip() or "train"
            ),
            "hf_token": run_config.get("hf_token") or None,
        }
    if source not in ("upload", "local", "", "huggingface"):
        raise ValueError(
            f"scoring supports local/upload files and Hugging Face datasets "
            f"(run uses dataset_source={source!r})"
        )
    raise ValueError(
        "run-config has no usable dataset reference "
        "(no local_datasets entries and no hf_dataset id)"
    )


def _resolve_local_files(names: List[str]) -> List[str]:
    """First existing file per name (absolute, cwd-relative, uploads roots)."""
    roots = [
        "",
        "/root/.unsloth/studio/assets/datasets/uploads",
        "/content/Nexus-ai-main/studio/backend/assets/datasets/uploads",
    ]
    resolved: List[str] = []
    tried: List[str] = []
    for name in names:
        candidates = [name]
        if not os.path.isabs(name):
            candidates.extend(
                os.path.join(root, os.path.basename(name))
                for root in roots
                if root
            )
        found = next(
            (candidate for candidate in candidates
             if candidate and os.path.isfile(candidate)),
            "",
        )
        tried.extend(candidates)
        if not found:
            raise ValueError(
                "dataset file not found for scoring; looked in: "
                + "; ".join(tried[:8])
            )
        resolved.append(found)
    return resolved


def _resolve_dataset_file(run_config: Dict[str, Any]) -> str:
    """Backward-compatible single-file resolution for upload runs."""
    kind, spec = _resolve_scoring_source(run_config)
    if kind != "upload":
        raise ValueError(
            "manual scoring supports local/upload dataset files "
            f"(run uses dataset_source={run_config.get('dataset_source')!r})"
        )
    files = spec.get("files") or []
    if not files:
        raise ValueError("run-config has no local_datasets entries")
    return files[0]


def check_row_cap(count: int) -> None:
    """Refuse absurd datasets before touching the GPU (pure, testable)."""
    if int(count) > MAX_SCORING_ROWS:
        raise ValueError(
            f"dataset has {count} rows, over the scoring cap "
            f"({MAX_SCORING_ROWS}); slice it first"
        )


def atomic_write_jsonl(records: List[Dict[str, Any]], final_path: str) -> str:
    """Write records atomically: temp file + rename, so readers never see a
    half-written sidecar. Returns the final path."""
    tmp_path = str(final_path) + ".tmp"
    with open(tmp_path, "w", encoding = "utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii = False) + "\n")
    os.replace(tmp_path, final_path)
    return str(final_path)


def _default_load_rows(dataset_file: str) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    suffix = os.path.splitext(dataset_file)[1].lower()
    loader: str
    kwargs: Dict[str, Any]
    if suffix in (".json", ".jsonl"):
        loader, kwargs = "json", {"data_files": dataset_file}
    elif suffix == ".csv":
        loader, kwargs = "csv", {"data_files": dataset_file}
    elif suffix == ".parquet":
        loader, kwargs = "parquet", {"data_files": dataset_file}
    else:
        raise ValueError(f"unsupported dataset format for scoring: {dataset_file}")
    dataset = load_dataset(loader, split = "train", **kwargs)
    rows = [dict(row) for row in dataset]
    check_row_cap(len(rows))
    return rows


def _load_hf_rows(
    hf_dataset: str,
    subset: Optional[str],
    train_split: str,
    hf_token: Optional[str],
) -> List[Dict[str, Any]]:
    """Stream a Hugging Face dataset up to the scoring cap (lazy import)."""
    from itertools import islice

    from datasets import load_dataset

    load_kwargs: Dict[str, Any] = {
        "path": hf_dataset.strip(),
        "split": (train_split or "train"),
        "streaming": True,
        "token": hf_token,
    }
    if (subset or "").strip():
        load_kwargs["name"] = subset.strip()
    try:
        streamed = load_dataset(**load_kwargs)
    except Exception as exc:
        raise ValueError(
            f"Could not load dataset '{hf_dataset}' for scoring: {exc}"
        ) from exc
    rows = [dict(row) for row in islice(streamed, MAX_SCORING_ROWS + 1)]
    if len(rows) > MAX_SCORING_ROWS:
        raise ValueError(
            f"dataset '{hf_dataset}' exceeds the scoring cap "
            f"({MAX_SCORING_ROWS} rows); slice it first"
        )
    if not rows:
        raise ValueError(
            f"dataset '{hf_dataset}' returned no rows for split "
            f"'{train_split}'"
        )
    return rows


def _default_run_scoring(
    output_dir: str,
    run_config: Dict[str, Any],
    progress_cb: Callable[[int, int], None],
) -> str:
    """The real GPU pipeline. Runs ONLY inside the job thread."""
    from core.training.offline_scoring import (
        OfflineScoreConfig,
        forward_to_logits,
        load_model_for_scoring,
        score_rows,
        write_records_jsonl,
    )

    dataset_file = _resolve_dataset_file(run_config)
    model_name = str(run_config.get("model_name") or "").strip()
    if not model_name:
        raise ValueError("run-config has no model_name for template detection")
    try:
        max_len = int(run_config.get("max_seq_length") or 2048)
    except (TypeError, ValueError):
        max_len = 2048

    model, tokenizer = load_model_for_scoring(
        output_dir,
        trust_remote_code = bool(run_config.get("trust_remote_code", False)),
    )
    try:
        kind, spec = _resolve_scoring_source(run_config)
        rows: List[Dict[str, Any]] = []
        if kind == "huggingface":
            rows = _load_hf_rows(
                spec["hf_dataset"], spec.get("subset"),
                spec.get("train_split") or "train", spec.get("hf_token"),
            )
        else:
            for dataset_file in spec.get("files", []):
                if os.path.splitext(dataset_file)[1].lower() in (
                    ".json", ".jsonl",
                ):
                    with open(dataset_file, "r", encoding = "utf-8") as handle:
                        rows.extend(
                            json.loads(line) for line in handle if line.strip()
                        )
                else:
                    rows.extend(_default_load_rows(dataset_file))
        check_row_cap(len(rows))
        if not rows:
            raise ValueError("dataset contains no rows to score")
        config = OfflineScoreConfig(
            checkpoint = output_dir, max_seq_length = max_len
        )
        collected: List[Dict[str, Any]] = []
        chunk = 4
        for start in range(0, len(rows), chunk):
            part = score_rows(
                rows[start : start + chunk],
                config = config,
                model_name = model_name,
                tokenizer = tokenizer,
                forward_fn = lambda tok, ids: forward_to_logits(model, ids),
                format_type = run_config.get("format_type") or "auto",
                custom_format_mapping = run_config.get("custom_format_mapping"),
            )
            # row_id values are positional per chunk; rebase to global rows.
            for offset, record in enumerate(part):
                record["row_id"] = start + offset
            collected.extend(part)
            progress_cb(len(collected), len(rows))
        final_path = os.path.join(output_dir, PER_EXAMPLE_FILENAME)
        atomic_write_jsonl(collected, final_path)
        return final_path
    finally:
        try:
            del model
        except Exception:
            pass
        try:
            del tokenizer
        except Exception:
            pass
        try:
            import gc as _gc

            _gc.collect()
        except Exception:
            pass
        try:
            import torch as _torch

            if getattr(_torch, "cuda", None) is not None and _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass


def start_scoring_job(
    run_id: str,
    output_dir: str,
    *,
    training_active_check: Optional[Callable[[], bool]] = None,
    run_scoring_fn: Optional[
        Callable[[str, Dict[str, Any], Callable[[int, int], None]], str]
    ] = None,
) -> Dict[str, Any]:
    """Validate synchronously, then spawn one daemon thread per run.

    Returns ``{"accepted": bool, "status": ..., "message": ...}`` — refusals
    are values, never exceptions, so the UI can display the reason directly.
    """
    run_id = str(run_id or "").strip()
    output_dir = str(output_dir or "").strip()
    if not run_id:
        return {"accepted": False, "status": "error",
                "message": "run_id is required"}
    if not output_dir or not os.path.isdir(output_dir):
        return {"accepted": False, "status": "error",
                "message": f"training output not found: {output_dir or '(empty)'}"}
    check = training_active_check or _default_training_active
    try:
        if check():
            return {"accepted": False, "status": "refused",
                    "message": "Training is running — GPU is busy. "
                               "Generate scores after it finishes."}
    except Exception as exc:
        return {"accepted": False, "status": "error",
                "message": f"could not verify GPU availability: {exc}"}
    try:
        run_config = _read_run_config(output_dir)
    except Exception as exc:
        return {"accepted": False, "status": "error",
                "message": f"cannot read run-config.json: {exc}"}

    with _JOBS_LOCK:
        _prune_terminal_locked()
        live = _JOBS.get(run_id)
        if live is not None and live.get("status") in ("queued", "running"):
            return {"accepted": False, "status": live.get("status"),
                    "message": "a scoring job is already running for this run"}
        job: Dict[str, Any] = {
            "status": "queued", "done": 0, "total": 0,
            "message": "queued", "started_at": _now(), "finished_at": 0.0,
        }
        _JOBS[run_id] = job

    scorer = run_scoring_fn or _default_run_scoring

    def _progress(done: int, total: int) -> None:
        with _JOBS_LOCK:
            current = _JOBS.get(run_id)
            if current is job:
                current["done"] = int(done)
                current["total"] = int(total)
                current["message"] = f"scoring {done}/{total}"

    def _thread_main() -> None:
        with _JOBS_LOCK:
            if _JOBS.get(run_id) is job:
                job["status"] = "running"
                job["message"] = "scoring started"
        try:
            final_path = scorer(output_dir, run_config, _progress)
            with _JOBS_LOCK:
                if _JOBS.get(run_id) is job:
                    job["status"] = "done"
                    job["message"] = f"scores written to {final_path}"
                    job["finished_at"] = _now()
        except Exception as exc:
            with _JOBS_LOCK:
                if _JOBS.get(run_id) is job:
                    job["status"] = "error"
                    job["message"] = str(exc) or type(exc).__name__
                    job["finished_at"] = _now()

    worker = threading.Thread(
        target = _thread_main,
        name = f"per-example-scoring-{run_id}",
        daemon = True,
    )
    worker.start()
    with _JOBS_LOCK:
        return {"accepted": True, **_snapshot_locked(_JOBS.get(run_id, job))}
