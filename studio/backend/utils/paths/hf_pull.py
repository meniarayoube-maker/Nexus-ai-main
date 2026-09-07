# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Restore helper for the ``huggingface`` storage target.

Mirrors :mod:`utils.paths.kaggle_push`'s download contract
(``(ok, path, error)``): download a Hub repo (a previously uploaded training
output) into a local directory so the run-history restore flow can register
it.  Uploads stream per file (no local archive), so downloads likewise need
no staging: ``snapshot_download`` writes straight into ``dest_dir``.

Restores are sparse by default: the repo accumulates every save point, so a
full download can exceed the working disk.  The file list is fetched first
(``repo_info``) and only the newest ``checkpoint-N`` bundle plus root-level
files are requested via exact ``allow_patterns``.  When the listing fails for
any reason the code falls back to the historical full download -- never worse
than today.

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


def _format_bytes(num: object) -> str:
    """Tiny human-readable byte formatter (local so this module never imports
    the heavy kaggle client just for formatting)."""
    try:
        value = float(num or 0)
    except (TypeError, ValueError):
        return "unknown size"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def _checkpoint_dir_step(name: str) -> Optional[int]:
    """Numeric step of a top-level ``checkpoint-<N>`` dir name, else None.

    Local twin of the kaggle uploader's sorter (kept here so this module never
    imports the heavy kaggle client): lexicographic order lies
    (``checkpoint-19`` < ``checkpoint-5``), so the newest bundle must be
    picked numerically.
    """
    prefix, sep, num = str(name or "").strip().partition("-")
    if prefix != "checkpoint" or not sep or not num.isdigit():
        return None
    return int(num)


def _select_sparse_patterns(
    siblings: object,
) -> "Tuple[list, Optional[int], int, int]":
    """Pick the sparse-restore file set from ``repo_info`` siblings.

    Returns ``(patterns, newest_step, omitted_count, omitted_bytes)`` where
    ``patterns`` holds exact repo-relative paths: everything under the
    numerically-newest top-level ``checkpoint-N`` dir, plus root-level files
    that have NO same-named twin inside that bundle.  The stop-save writes the
    full training state twice (final files at the root AND the checkpoint
    bundle), so fetching both doubles the footprint for zero resume value:
    the checkpoint copy is authoritative for the exact step.  Older checkpoint
    bundles and any other subdirectories are omitted (a training output is
    flat apart from its checkpoint dirs).  With no checkpoint dir at all
    (adapter-only output) every root file is kept.
    """
    names = [str(getattr(sibling, "rfilename", None) or "") for sibling in siblings or []]
    names = [name for name in names if name]
    newest_name: Optional[str] = None
    newest_step: Optional[int] = None
    for name in names:
        parts = name.split("/")
        if len(parts) == 2 and parts[0]:
            step = _checkpoint_dir_step(parts[0])
            if step is not None and (newest_step is None or step > newest_step):
                newest_step = step
                newest_name = parts[0]
    newest_prefix = f"{newest_name}/" if newest_name else None
    # Basenames already carried inside the newest bundle: their root-level
    # twins are byte-duplicates with no resume value.
    bundled: set = set()
    if newest_prefix:
        for name in names:
            if name.startswith(newest_prefix):
                rest = name[len(newest_prefix):]
                if rest and "/" not in rest:
                    bundled.add(rest)
    patterns: list = []
    omitted = 0
    omitted_bytes = 0
    by_name = {}
    for sibling in siblings or []:
        rfilename = str(getattr(sibling, "rfilename", None) or "")
        if rfilename:
            by_name[rfilename] = sibling
    for name in names:
        if "/" not in name:
            if name in bundled:
                omitted += 1
                try:
                    omitted_bytes += int(getattr(by_name[name], "size", None) or 0)
                except (TypeError, ValueError):
                    continue
                continue
            patterns.append(name)
            continue
        if newest_prefix and name.startswith(newest_prefix):
            patterns.append(name)
            continue
        omitted += 1
        try:
            omitted_bytes += int(getattr(by_name[name], "size", None) or 0)
        except (TypeError, ValueError):
            continue
    return patterns, newest_step, omitted, omitted_bytes


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

    The download is sparse: only the newest ``checkpoint-N`` bundle plus
    root-level files are fetched (see :func:`_select_sparse_patterns`), so a
    repo holding many save points still restores inside a small working disk.
    When the remote file listing fails, the call falls back to a full
    download.
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

    revision = revision.strip() if revision and revision.strip() else None
    token = hf_token.strip() if hf_token and hf_token.strip() else None

    # Sparse shortlist first: exact paths keep the restore to ~one save point
    # even when the repo accumulated many.  Any listing failure falls back to
    # the historical full download (and never changes the call shape the
    # existing callers/tests rely on).
    allow_patterns: Optional[list] = None
    try:
        from huggingface_hub import repo_info as _hub_repo_info

        siblings = getattr(
            _hub_repo_info(repo_id = slug, revision = revision, token = token),
            "siblings",
            None,
        )
        if siblings:
            selected, newest_step, omitted, omitted_bytes = _select_sparse_patterns(siblings)
            if selected:
                allow_patterns = selected
                newest = f"checkpoint-{newest_step}" if newest_step is not None else "none"
                logger.info(
                    "Hugging Face sparse restore: %s -> %s: %d files (newest %s), "
                    "omitting %d superseded files (%s)",
                    slug,
                    dest,
                    len(selected),
                    newest,
                    omitted,
                    _format_bytes(omitted_bytes),
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Hugging Face file listing failed for %s; falling back to full download: %s",
            slug,
            exc,
        )
        allow_patterns = None

    try:
        if allow_patterns is None:
            snapshot_download(
                repo_id = slug,
                revision = revision,
                local_dir = str(dest),
                token = token,
            )
        else:
            snapshot_download(
                repo_id = slug,
                revision = revision,
                local_dir = str(dest),
                token = token,
                allow_patterns = allow_patterns,
            )
        logger.info("Hugging Face dataset downloaded: %s -> %s", slug, dest)
        return (True, str(dest), None)
    except Exception as exc:  # noqa: BLE001
        _remove_new_entries(dest, pre_existing)
        reason = _describe_error(exc)
        logger.warning("Hugging Face download failed for %s -> %s: %s", slug, dest, reason)
        return (False, None, reason)
