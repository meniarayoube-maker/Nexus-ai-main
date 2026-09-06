# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Restore helper for the ``huggingface`` storage target.

Mirrors :mod:`utils.paths.kaggle_push`'s download contract
(``(ok, path, error)``): download a Hub repo (a previously uploaded training
output) into a local directory so the run-history restore flow can register
it.  Uploads stream per file (no local archive), so downloads likewise need
no staging: ``snapshot_download`` writes straight into ``dest_dir``.

huggingface_hub is imported lazily so unit tests can stub it and hosts
without the package fail precisely instead of at import time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

from loggers import get_logger

logger = get_logger(__name__)

PullResult = Tuple[bool, Optional[str], Optional[str]]
"""``(ok, path, error)``."""


def _validate_repo_id(repo_id: object) -> Optional[str]:
    """Normalize ``owner/name`` or return None when malformed.

    Accepts full Hub URLs (query/fragment stripped) since users paste them
    verbatim.  An ``owner/name`` shape is required: bare names resolve
    ambiguously on the Hub.
    """
    text = str(repo_id or "").strip()
    lower = text.lower()
    if "://" in text or lower.startswith("huggingface.co/"):
        path_part = text.split("://", 1)[-1].split("/", 1)[-1]
        segments = [seg for seg in path_part.split("/") if seg]
        # huggingface.co/<owner>/<name>[/*] -- drop any trailing tabs.
        if len(segments) >= 3 and segments[0].lower() == "huggingface.co":
            text = "/".join(segments[1:3])
        elif len(segments) >= 2:
            text = "/".join(segments[-2:])
    text = text.split("?", 1)[0].split("#", 1)[0].strip().strip("/")
    parts = [p for p in text.split("/") if p]
    if len(parts) != 2 or any(p in (".", "..") for p in parts):
        return None
    return f"{parts[0]}/{parts[1]}"


def _describe_error(exc: BaseException) -> str:
    """Map download failures to precise, user-facing reasons."""
    text = f"{type(exc).__name__}: {exc}".strip()
    lowered = text.lower()
    if any(marker in lowered for marker in ("401", "403", "unauthorized", "forbidden", "gated")):
        return (
            "Hugging Face authentication failed or the repo is gated. Check the "
            "HF token (write/read access as needed) and that access was granted."
        )
    if "404" in lowered or "not found" in lowered or "repositorynotfound" in lowered.replace(" ", ""):
        return (
            "Hugging Face repo or revision not found. Verify the repo id and "
            "that the revision (branch/tag/commit) exists."
        )
    if "revision" in lowered:
        return f"Hugging Face revision error: {text}"
    return f"Hugging Face download failed: {text}"


def _remove_new_entries(dest: Path, pre_existing: set) -> None:
    """Remove entries that appeared in ``dest`` during a failed download.

    Prevents partial trees from tripping the "already restored" guard on the
    next attempt.  Best-effort only; never raises.
    """
    import shutil as _shutil

    try:
        current = {p.name for p in dest.iterdir()}
    except OSError:
        return
    for name in sorted(current - set(pre_existing)):
        try:
            target = dest / name
            if target.is_dir() and not target.is_symlink():
                _shutil.rmtree(target, ignore_errors = True)
            else:
                target.unlink(missing_ok = True)
        except OSError:
            continue


def download_output_from_huggingface(
    repo_id: str,
    dest_dir: "str | os.PathLike[str]",
    *,
    revision: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> PullResult:
    """Download a Hub repo into ``dest_dir`` for run-history restore.

    Returns ``(ok, path, error)`` with the same non-fatal contract as the
    Kaggle downloader: callers surface ``error`` but already-existing state is
    untouched (partials from this call are removed).
    """
    slug = _validate_repo_id(repo_id)
    if slug is None:
        return (False, None, "Invalid Hugging Face repo: expected 'owner/name'.")
    dest = Path(dest_dir).expanduser()
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return (False, None, f"Could not create restore directory {dest}: {exc}")
    try:
        pre_existing = {p.name for p in dest.iterdir()}
    except OSError:
        pre_existing = set()

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # noqa: BLE001
        return (
            False,
            None,
            "Hugging Face download unavailable (huggingface_hub not installed). "
            f"Install huggingface_hub and retry. ({exc})",
        )

    try:
        snapshot_download(
            repo_id = slug,
            revision = (revision.strip() if revision and revision.strip() else None),
            local_dir = str(dest),
            token = (hf_token.strip() if hf_token and hf_token.strip() else None),
        )
        logger.info("Hugging Face dataset downloaded: %s -> %s", slug, dest)
        return (True, str(dest), None)
    except Exception as exc:  # noqa: BLE001
        _remove_new_entries(dest, pre_existing)
        reason = _describe_error(exc)
        logger.warning("Hugging Face download failed for %s -> %s: %s", slug, dest, reason)
        return (False, None, reason)
