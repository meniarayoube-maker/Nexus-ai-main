# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Post-training upload for the ``huggingface`` storage target.

When a training run completes (natural end or stop-and-save) with
``storage_target == "huggingface"`` and an ``hf_repo_id`` provided, the finished
output directory (already written locally under the outputs root as the staging
directory) is pushed to Hugging Face Hub as a model repository.

The push is deliberately non-fatal: training results are already saved locally,
so an upload failure must not roll back or destroy a finished run. However, the
helper does **not** swallow the outcome silently. It returns a structured
``(ok, repo_url, error)`` triple so callers can surface a precise reason to the
UI (invalid token, repo/connection failure, etc.) instead of only logging.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

from loggers import get_logger

logger = get_logger(__name__)

PushResult = Tuple[bool, Optional[str], Optional[str]]
"""``(ok, repo_url, error)``: ``ok`` True only on a fully successful upload.
``repo_url`` is set on success. ``error`` is a precise human-readable reason on
failure (wired through to the visible UI status/warning message)."""


def _resolve_token(hf_token: Optional[str]) -> Optional[str]:
    """Resolve the HF token following app-wide priority.

    Explicit/manual ``hf_token`` (from the UI request) wins first. Otherwise we
    fall back to whatever ``huggingface_hub.get_token()`` resolves in this
    environment — the standard ``HF_TOKEN``/alias env vars first, then the
    credentials cached on disk (e.g. from ``huggingface-cli login``), which makes
    Colab/Kaggle secret-token runs work without re-prompting.
    """
    if hf_token and str(hf_token).strip():
        return str(hf_token).strip()
    try:
        from huggingface_hub import get_token
    except Exception:  # noqa: BLE001 - huggingface_hub unavailable; anonymous push
        return None
    try:
        # get_token() reads HF_TOKEN (and alias env keys) first, then the cached
        # token from ~/.cache/huggingface/token. Returns "" when none is present.
        token = get_token()
    except Exception as exc:  # noqa: BLE001
        logger.debug("HF token resolution fallback unavailable: %s", exc)
        return None
    if token and str(token).strip():
        return str(token).strip()
    return None


def _ensure_readme(root: Path) -> bool:
    """Add a minimal README if the upload folder has no markdown/model card.

    A folder full of raw weights with no README renders as an empty/broken repo
    on the Hub, which users often mistake for a failed upload. Writing a tiny
    card makes the uploaded repo presentable. Returns True if we wrote one.
    """
    if any(root.glob("*.md")):
        return False
    try:
        target = root / "README.md"
        target.write_text(
            "# Training Output\n\n"
            "Checkpoint and adapter training output uploaded from fine-tuning.\n",
            encoding="utf-8",
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not write README for %s: %s", root, exc)
        return False


def _describe_error(exc: Exception) -> str:
    """Map a huggingface_hub error to a precise, user-facing reason."""
    text = str(exc) or exc.__class__.__name__
    try:
        from huggingface_hub.errors import HfHubHTTPError

        if isinstance(exc, HfHubHTTPError):
            status = getattr(exc, "response", None)
            code = getattr(status, "status_code", None) if status is not None else None
            if code in (401, 403):
                return (
                    "Hugging Face authentication failed — the token is missing, "
                    "invalid, or lacks write permission. Check the HF token in the "
                    "UI or the HF_TOKEN environment variable."
                )
            if code == 404:
                return (
                    "Hugging Face repo not found or you lack access — verify the "
                    "repo id and that the token can access it."
                )
            if code in (429, 503):
                return "Hugging Face rate limited or temporarily unavailable — try again shortly."
            if code is not None:
                return f"Hugging Face API error (HTTP {code}): {text}"
    except Exception:  # noqa: BLE001
        pass

    lowered = text.lower()
    if "authentication" in lowered or "401" in lowered or "403" in lowered:
        return (
            "Hugging Face authentication failed — the token is missing, invalid, "
            "or lacks write permission."
        )
    if "rate limit" in lowered or "quota" in lowered:
        return "Hugging Face rate limited or quota exceeded — try again shortly."
    # Connection-ish failures reachable on Colab/Kaggle.
    if "connect" in lowered or "resolve" in lowered or "timeout" in lowered or "ssl" in lowered:
        return (
            "Hugging Face network error (could not connect). Check internet access "
            "and retry."
        )
    return f"Hugging Face upload failed: {text}"


def push_output_to_huggingface(
    output_dir: "str | os.PathLike[str]",
    repo_id: str,
    *,
    hf_token: Optional[str] = None,
    private: bool = False,
    commit_message: Optional[str] = None,
) -> PushResult:
    """Upload ``output_dir`` to the Hugging Face repo ``repo_id``.

    Returns ``(ok, repo_url, error)``:
      - ``(True, url, None)``       on a fully successful upload.
      - ``(False, None, reason)``   on a genuine failure, with a precise message.
      - ``(False, None, None)``     when skipped (no repo id / no output dir).

    Callers should surface ``error`` to the UI but never treat a failure as fatal
    (artifacts already live on disk).
    """
    if not repo_id or not str(repo_id).strip():
        logger.warning("HF push skipped: no hf_repo_id provided for huggingface storage target")
        return (False, None, None)
    repo_id = str(repo_id).strip()

    root = Path(output_dir).expanduser()
    if not root.is_dir():
        logger.warning("HF push skipped: output_dir not a directory: %s", root)
        return (False, None, None)

    token = _resolve_token(hf_token)

    try:
        from huggingface_hub import HfApi
    except Exception as exc:  # noqa: BLE001
        logger.warning("HF push unavailable (huggingface_hub not installed): %s", exc)
        reason = (
            "Hugging Face upload unavailable (huggingface_hub not installed). "
            "Install huggingface_hub and retry."
        )
        return (False, None, reason)

    # Presentable repo: fold-level README missing would render an empty Hub repo.
    _ensure_readme(root)

    try:
        api = HfApi(token=token)
        created = api.create_repo(repo_id, private=private, exist_ok=True, token=token)
        resolved_repo_id = created.repo_id if created is not None else repo_id
        api.upload_folder(
            repo_id=resolved_repo_id,
            folder_path=str(root),
            commit_message=commit_message or f"Upload training output from {root.name}",
            token=token,
        )
    except Exception as exc:  # noqa: BLE001
        reason = _describe_error(exc)
        logger.warning("HF push failed for %s -> %s: %s", root, repo_id, reason)
        return (False, None, reason)

    repo_url = f"https://huggingface.co/{resolved_repo_id}"
    logger.info("HF push complete: %s -> %s", root, repo_url)
    return (True, repo_url, None)
