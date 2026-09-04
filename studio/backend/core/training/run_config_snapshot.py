# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Full run-configuration snapshots for session restore.

A checkpoint directory only carries weights + trainer state -- the *session*
(the model id, training type, dataset, format, hyperparameters the run trained
with) lives in the database, which does not survive a new Colab/Kaggle
session.  To make a restored run resumable with zero manual fields, every
training worker writes ``run-config.json`` (this module) into its output dir
*before* training starts, so the file travels inside every upload
(Drive/HF/Kaggle) automatically.  The restore route then rebuilds the history
row from that file instead of asking the user.

Stdlib-only on purpose: both the heavy worker and the unit tests import this
without pulling torch/transformers.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

RUN_CONFIG_FILENAME = "run-config.json"

# Config keys are never persisted when their name hints at a secret. Username
# and approval fingerprints are intentionally kept (needed for restore).
_SENSITIVE_SUBSTRINGS = ("token", "key", "secret", "password", "passwd")

# Keys that describe ONE run invocation, never reusable for a restored row.
_RUN_SPECIFIC_KEYS = frozenset(
    {
        "output_dir",
        "resume_from_checkpoint",
        "start_request_id",
        "run_id",
        "job_id",
    }
)

# Keys bound to the ORIGINAL session's filesystem/HF cache.  Their absolute
# snapshot paths die with the old session, and the provenance gate resolves
# them strictly -- keeping them would permanently refuse resume in any new
# session ("exact snapshot no longer available").  Dropping them forces fresh
# resolution (re-download + re-attestation) while the checkpoint weights
# themselves stay exact from disk.  Repo ids / fingerprints are kept.
_SESSION_BOUND_KEYS = frozenset(
    {
        "resource_provenance",
        "model_snapshot_path",
        "dataset_snapshot_path",
        "model_local_path",
        "dataset_local_path",
        "model_known_cached",
        "dataset_known_cached",
    }
)


def sanitize_run_config(config: Mapping[str, Any]) -> dict:
    """Return a JSON-safe copy of ``config`` without secrets.

    Never raises: unserializable values degrade to ``str()`` so a snapshot can
    never break (or leak through) training.
    """
    clean: dict = {}
    for raw_key, value in dict(config).items():
        key = str(raw_key)
        lowered = key.lower()
        if any(marker in lowered for marker in _SENSITIVE_SUBSTRINGS):
            continue
        clean[key] = value
    try:
        return json.loads(json.dumps(clean, ensure_ascii = False, default = str))
    except (TypeError, ValueError):
        return {key: str(value) for key, value in clean.items()}


def write_run_config_snapshot(
    output_dir: "str | os.PathLike[str]",
    config: Mapping[str, Any],
) -> Optional[str]:
    """Persist ``run-config.json`` into an already-created ``output_dir``.

    Returns the file path, or None when anything goes wrong.  Never raises:
    a snapshot must never break or delay training.
    """
    try:
        target = Path(output_dir).expanduser() / RUN_CONFIG_FILENAME
        payload = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "config": sanitize_run_config(config),
        }
        target.write_text(
            json.dumps(payload, indent = 2, ensure_ascii = False), encoding = "utf-8"
        )
        return str(target)
    except Exception:
        return None


def load_run_config_snapshot(output_dir: "str | os.PathLike[str]") -> Optional[dict]:
    """Read back a snapshot written by :func:`write_run_config_snapshot`.

    Returns the stored config dict, or None when the file is absent, corrupt,
    or not a mapping (e.g. hand-made datasets predate this feature).
    """
    try:
        raw = json.loads(
            (Path(output_dir).expanduser() / RUN_CONFIG_FILENAME).read_text(
                encoding = "utf-8-sig"
            )
        )
    except (OSError, ValueError, AttributeError):
        return None
    config = raw.get("config") if isinstance(raw, dict) else None
    return dict(config) if isinstance(config, dict) else None


def build_restored_config(
    *,
    file_config: Optional[dict],
    inferred_model: str,
    inferred_training_type: str,
    manual_hf_dataset: Optional[str],
    slug: str,
    storage_target: str,
) -> dict:
    """Build the history-row config for a restored run.

    The snapshot file (the truth about the original session) wins wherever it
    speaks; inference (adapter/base scan) and manual fields only fill gaps.
    Run-specific keys (stale output paths, request ids) are scrubbed so a
    resumed start can never write into -- or resume from -- the dead session's
    paths.
    """
    base = dict(file_config) if isinstance(file_config, dict) else {}
    for stale in _RUN_SPECIFIC_KEYS | _SESSION_BOUND_KEYS:
        base.pop(stale, None)
    if not base.get("model_name"):
        base["model_name"] = inferred_model
    if not base.get("training_type"):
        base["training_type"] = inferred_training_type
    if not base.get("format_type"):
        base["format_type"] = "alpaca"
    if not base.get("hf_dataset"):
        if manual_hf_dataset:
            base["hf_dataset"] = manual_hf_dataset
        else:
            base.pop("hf_dataset", None)
    base["restored_from_kaggle"] = slug
    base["storage_target"] = storage_target
    return base
