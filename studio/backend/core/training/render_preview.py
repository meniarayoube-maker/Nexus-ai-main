# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Render-preview for training datasets (Pilot, text-only v1).

Runs the *same* formatting pipeline training uses
(:func:`utils.datasets.format_and_template_dataset`) over a tiny sample slice —
no training, no model weights — and reports, per sample row:

1. the original dataset fields,
2. the final text after the chat template (pre-tokenization),
3. what actually enters the tokenizer (input_ids + counts + decode round-trip),
4. a metadata audit proving columns like ``example_id`` / ``source`` / ``level`` /
   ``batch_id`` did not leak into the training text when they live outside
   ``messages``.

Loss tracking is deliberately out of scope here; this module only answers
"what exactly entered training".
"""

from __future__ import annotations

import os
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loggers import get_logger
from utils.datasets import format_and_template_dataset
from utils.paths import dataset_files_in_dir, resolve_dataset_path

logger = get_logger(__name__)

# Display caps: the full values are always used for the leak check and the
# token counts; only the *displayed* strings are truncated.
_MAX_TEXT_CHARS = 6000
_MAX_DECODE_CHARS = 2000
_MAX_RAW_STR_CHARS = 4000
_IDS_EDGE = 32

# Column names treated as row metadata (never part of messages). A value from
# one of these columns appearing verbatim in the rendered text is reported.
METADATA_COLUMN_HINTS = frozenset(
    {"example_id", "source", "level", "batch_id", "id", "index", "row_id"}
)

_TOKENIZER_CACHE: Dict[str, Any] = {}


def load_preview_tokenizer(
    model_name: str,
    model_local_path: Optional[str] = None,
    hf_token: Optional[str] = None,
    trust_remote_code: bool = False,
) -> Tuple[Any, str]:
    """Load only the tokenizer (never model weights), cached per resolved repo.

    Mirrors the kwargs training uses (``token`` / ``trust_remote_code``).
    Raises ``ValueError`` with a user-facing message when the tokenizer cannot
    be read (model not downloaded, gated repo without token, ...).
    """
    from transformers import AutoTokenizer

    candidates: List[Tuple[str, bool, str]] = []
    if model_local_path and Path(model_local_path).exists():
        candidates.append((model_local_path, True, f"local:{model_local_path}"))
    candidates.append((model_name, False, f"hub:{model_name}"))

    errors: List[str] = []
    for repo, local_only, label in candidates:
        cached = _TOKENIZER_CACHE.get(repo)
        if cached is not None:
            return cached, label + " (cached)"
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                repo,
                trust_remote_code = trust_remote_code,
                token = hf_token,
                local_files_only = local_only,
            )
        except Exception as exc:
            errors.append(f"{label}: {exc}")
            continue
        _TOKENIZER_CACHE[repo] = tokenizer
        return tokenizer, label
    detail = "; ".join(errors) if errors else "unknown error"
    raise ValueError(
        "Could not load a tokenizer for "
        f"'{model_name}'. Download/cached model files are required "
        f"(gated models need an HF token). Details: {detail}"
    )


def _resolve_preview_local_files(file_paths: List[str]) -> List[str]:
    """Same resolution training uses: abs path, cwd-relative, or Studio dataset root."""
    all_files: List[str] = []
    for dataset_file in file_paths:
        if os.path.isabs(dataset_file):
            file_path = dataset_file
        elif os.path.exists(dataset_file):
            file_path = os.path.abspath(dataset_file)
        else:
            file_path = str(resolve_dataset_path(dataset_file))
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
        if file_path_obj.is_dir():
            all_files.extend(str(p) for p in dataset_files_in_dir(file_path_obj))
        else:
            all_files.append(str(file_path_obj))
    if not all_files:
        raise ValueError("No dataset files found for preview")
    return all_files


def _loader_for_files(files: List[str]) -> str:
    """Same loader selection training uses (json / csv / parquet)."""
    first_ext = Path(files[0]).suffix.lower()
    if first_ext in (".json", ".jsonl"):
        return "json"
    elif first_ext == ".csv":
        return "csv"
    elif first_ext == ".parquet":
        return "parquet"
    raise ValueError(f"Unsupported dataset format for preview: {files[0]}")


def load_preview_sample_rows(
    dataset_source: str,
    num_samples: int,
    hf_dataset: Optional[str] = None,
    subset: Optional[str] = None,
    train_split: str = "train",
    local_datasets: Optional[List[str]] = None,
    hf_token: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[str], Optional[int]]:
    """First ``num_samples`` raw rows + input columns + total row count (if known).

    Hugging Face loads stream (no full download); local/upload files load fully
    (pilot-scale) and are sliced. Only ``huggingface`` and ``upload`` sources
    are supported in v1.
    """
    from datasets import Dataset, load_dataset

    if dataset_source == "huggingface":
        if not (hf_dataset or "").strip():
            raise ValueError("hf_dataset is required for dataset_source='huggingface'")
        load_kwargs: Dict[str, Any] = {
            "path": hf_dataset.strip(),
            "split": train_split or "train",
            "streaming": True,
            "token": hf_token,
        }
        if (subset or "").strip():
            load_kwargs["name"] = subset.strip()
        try:
            streamed = load_dataset(**load_kwargs)
        except Exception as exc:
            raise ValueError(f"Could not load dataset '{hf_dataset}': {exc}") from exc
        rows = list(islice(streamed, num_samples))
        if not rows:
            raise ValueError(f"Dataset '{hf_dataset}' returned no rows for split '{train_split}'")
        columns = list(rows[0].keys())
        return rows, columns, None

    if dataset_source == "upload":
        files = [f for f in (local_datasets or []) if str(f or "").strip()]
        if not files:
            raise ValueError("local_datasets is required for dataset_source='upload'")
        all_files = _resolve_preview_local_files(files)
        loader = _loader_for_files(all_files)
        try:
            dataset = load_dataset(loader, data_files = all_files, split = "train")
        except Exception as exc:
            raise ValueError(f"Could not read local dataset files: {exc}") from exc
        total = len(dataset)
        if total == 0:
            raise ValueError("Local dataset files contain no rows")
        take = min(num_samples, total)
        rows = [dict(dataset[i]) for i in range(take)]
        columns = list(dataset.column_names or [])
        return rows, columns, total

    raise ValueError(
        f"Preview v1 supports dataset_source 'huggingface' and 'upload' (got '{dataset_source}')"
    )


def _jsonable(value: Any) -> Any:
    """Coerce a raw cell for JSON display (truncated, never raising)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str) and len(value) > _MAX_RAW_STR_CHARS:
            return value[:_MAX_RAW_STR_CHARS] + f"… [truncated {len(value) - _MAX_RAW_STR_CHARS} chars]"
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value[:50]]
    return str(value)


def _truncate_display(text: str, limit: int) -> Tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit] + f"… [truncated {len(text) - limit} chars]", True


def _audit_metadata(raw_row: Dict[str, Any], rendered_text: str) -> List[Dict[str, Any]]:
    """For each metadata-hint column, report whether its value leaked into the text."""
    audit: List[Dict[str, Any]] = []
    for column, value in raw_row.items():
        if column not in METADATA_COLUMN_HINTS:
            continue
        if value is None:
            continue
        if isinstance(value, bool):
            continue
        text_value = str(value) if not isinstance(value, str) else value
        if len(text_value.strip()) < 2:
            continue
        audit.append(
            {
                "column": column,
                "value_preview": text_value[:200],
                "leaked_into_text": text_value in rendered_text,
            }
        )
    return audit


def render_preview_samples(
    dataset_source: str,
    model_name: str,
    num_samples: int = 3,
    hf_dataset: Optional[str] = None,
    subset: Optional[str] = None,
    train_split: str = "train",
    local_datasets: Optional[List[str]] = None,
    format_type: str = "auto",
    custom_format_mapping: Optional[Dict[str, Any]] = None,
    model_local_path: Optional[str] = None,
    hf_token: Optional[str] = None,
    trust_remote_code: bool = False,
    max_seq_length: int = 2048,
) -> Dict[str, Any]:
    """Full render-preview pipeline. Raises ``ValueError``/``FileNotFoundError``."""
    from datasets import Dataset

    num_samples = max(1, min(int(num_samples or 3), 5))
    cap_at = max(1, int(max_seq_length or 2048))

    tokenizer, tokenizer_source = load_preview_tokenizer(
        model_name,
        model_local_path = model_local_path,
        hf_token = hf_token,
        trust_remote_code = trust_remote_code,
    )

    raw_rows, columns_in, total_rows = load_preview_sample_rows(
        dataset_source,
        num_samples,
        hf_dataset = hf_dataset,
        subset = subset,
        train_split = train_split or "train",
        local_datasets = local_datasets,
        hf_token = hf_token,
    )

    # Same pipeline training runs (positional order preserved by map/select).
    result = format_and_template_dataset(
        Dataset.from_list(raw_rows),
        model_name = model_name,
        tokenizer = tokenizer,
        format_type = format_type or "auto",
        custom_format_mapping = custom_format_mapping,
    )
    if result.get("is_image") or result.get("is_audio"):
        raise ValueError("Render preview v1 is text-only; image/audio datasets are not supported yet")
    if not result.get("success", False):
        errors = result.get("errors", []) or ["Dataset formatting failed"]
        return {
            "success": False,
            "samples": [],
            "columns_in": columns_in,
            "columns_out": [],
            "detected_format": result.get("detected_format", "unknown"),
            "final_format": result.get("final_format", "unknown"),
            "warnings": result.get("warnings", []),
            "errors": [str(e) for e in errors],
            "tokenizer_source": tokenizer_source,
            "total_rows": total_rows,
        }

    formatted = result["dataset"]
    columns_out = list(getattr(formatted, "column_names", None) or [])
    samples: List[Dict[str, Any]] = []
    for i in range(min(len(formatted), len(raw_rows))):
        raw_row = raw_rows[i]
        rendered = formatted[i].get("text", "") or ""
        encoding = tokenizer(rendered, add_special_tokens = True)
        input_ids = list(encoding["input_ids"])
        full_count = len(input_ids)
        capped_count = min(full_count, cap_at)
        if full_count > _IDS_EDGE * 2:
            shown_head = input_ids[:_IDS_EDGE]
            shown_tail = input_ids[-_IDS_EDGE:]
        else:
            shown_head = input_ids
            shown_tail = []
        decoded_full = tokenizer.decode(input_ids, skip_special_tokens = False)
        rendered_display, rendered_truncated = _truncate_display(rendered, _MAX_TEXT_CHARS)
        decoded_display, _ = _truncate_display(decoded_full, _MAX_DECODE_CHARS)
        samples.append(
            {
                "index": i,
                "raw_fields": {str(k): _jsonable(v) for k, v in raw_row.items()},
                "columns_in_row": list(raw_row.keys()),
                "rendered_text": rendered_display,
                "rendered_char_count": len(rendered),
                "rendered_truncated": rendered_truncated,
                "token_count_full": full_count,
                "token_count_capped": capped_count,
                "cap_applied_at": cap_at,
                "input_ids_head": shown_head,
                "input_ids_tail": shown_tail,
                "decoded_text": decoded_display,
                "roundtrip_exact": decoded_full == rendered,
                "metadata_audit": _audit_metadata(raw_row, rendered),
            }
        )

    return {
        "success": True,
        "samples": samples,
        "columns_in": columns_in,
        "columns_out": columns_out,
        "detected_format": result.get("detected_format", "unknown"),
        "final_format": result.get("final_format", "unknown"),
        "warnings": [str(w) for w in result.get("warnings", [])],
        "errors": [],
        "tokenizer_source": tokenizer_source,
        "total_rows": total_rows,
    }
