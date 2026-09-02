# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Post-training upload for the ``huggingface`` storage target.

When a training run completes (natural end or stop-and-save) with
``storage_target == "huggingface"`` and an ``hf_repo_id`` provided, the finished
output directory (already written locally under the outputs root as the staging
directory) is pushed to Hugging Face Hub as a model repository.

The push is deliberately best-effort and non-fatal: training results are already
saved locally, so an upload failure must not roll back or destroy a finished run.
Callers catch/log any exception here and continue.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from loggers import get_logger

logger = get_logger(__name__)


def push_output_to_huggingface(
    output_dir: "str | os.PathLike[str]",
    repo_id: str,
    *,
    hf_token: Optional[str] = None,
    private: bool = False,
    commit_message: Optional[str] = None,
) -> Optional[str]:
    """Upload ``output_dir`` to the Hugging Face repo ``repo_id``.

    Returns the repo_url on success or ``None`` when skipped/errored. Raises on a
    genuine upload failure so the caller can log it, but the caller must not
    treat a raised error as fatal (artifacts already live on disk).
    """
    if not repo_id or not str(repo_id).strip():
        logger.warning("HF push skipped: no hf_repo_id provided for huggingface storage target")
        return None

    root = Path(output_dir).expanduser()
    if not root.is_dir():
        logger.warning("HF push skipped: output_dir not a directory: %s", root)
        return None

    # Resolve the token the same way the rest of the app does: explicit arg wins,
    # then the standard HF env var, then the cached HF token on disk.
    token = hf_token or os.environ.get("HF_TOKEN") or ""

    try:
        from huggingface_hub import HfApi
    except Exception as exc:  # noqa: BLE001
        logger.warning("HF push unavailable (huggingface_hub not installed): %s", exc)
        return None

    try:
        api = HfApi(token=token or None)
        repo_id = api.create_repo(
            repo_id, private=private, exist_ok=True, token=token or None
        ).repo_id
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(root),
            commit_message=commit_message or f"Upload training output from {root.name}",
            token=token or None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("HF push failed for %s -> %s: %s", root, repo_id, exc)
        raise
    logger.info("HF push complete: %s -> %s", root, repo_id)
    return f"https://huggingface.co/{repo_id}"
