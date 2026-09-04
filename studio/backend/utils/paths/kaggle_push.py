# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Post-training upload for the ``kaggle`` storage target.

When a training run completes (natural end or stop-and-save) with
``storage_target == "kaggle"``, the finished output directory (already written
locally at ``/kaggle/working/unsloth-outputs/<run>``) is published as a Kaggle
Dataset -- private by default, switchable to public via ``is_private``.

The push is non-fatal: training artifacts already live on disk, so an upload
failure must never change the run's terminal status.  However, the result is
explicitly surfaced -- callers emit a ``status`` / ``warning`` event so the UI
shows a precise reason (missing package, missing credentials, auth failure,
network, rate-limit) instead of silently swallowing the error.

Credentials
-----------
The official ``kaggle`` package (``kaggle-api``) resolves credentials in this
order, which matches Kaggle's own documentation:

1. Explicit ``username`` / ``key`` keyword arguments (training UI fields).
2. ``KAGGLE_USERNAME`` + ``KAGGLE_KEY`` environment variables (already present
   on Kaggle notebook hosts).
3. ``~/.kaggle/kaggle.json`` credential file (``huggingface-cli login``-style
   persistent auth).

On a Kaggle notebook, credentials are injected automatically, so this module
"just works" with zero configuration.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Tuple

from loggers import get_logger

logger = get_logger(__name__)

PushResult = Tuple[bool, Optional[str], Optional[str]]
"""``(ok, dataset_url, error)``."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_kaggle_credentials(
    username: Optional[str],
    key: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve Kaggle credentials: explicit args > env vars > kaggle.json."""
    user = (str(username).strip() if username else None) or os.environ.get("KAGGLE_USERNAME") or None
    kkey = (str(key).strip() if key else None) or os.environ.get("KAGGLE_KEY") or None
    # If env vars provided only one half, we need both; fall back to kaggle.json.
    if (user and not kkey) or (kkey and not user):
        user, kkey = None, None
    if user and kkey:
        return user, kkey
    # kaggle.json fallback: KaggleApi.authenticate() handles this natively.
    return None, None


def _slug_from_output_dir(output_dir: Path) -> str:
    """Derive a default dataset slug from the output directory name."""
    import re
    name = re.sub(r"[^a-z0-9]+", "-", output_dir.name.lower().strip())[:80].strip("-")
    return name or "unsloth-output"


def _write_metadata(
    dataset_dir: Path,
    slug: str,
    is_private: bool,
    description: str,
    *,
    owner: Optional[str] = None,
) -> None:
    """Write ``dataset-metadata.json`` into the upload folder.

    The real ``kaggle-api`` client reads the title/slug/visibility from this
    file (``dataset_create_new``/``dataset_create_version`` take only a folder
    argument) and requires the ``id`` (``owner/slug``) and ``licenses`` fields,
    so both are included whenever the owner is known.
    """
    title = slug.replace("-", " ").replace("_", " ").title()
    metadata = {
        "title": title,
        "id": f"{owner}/{slug}" if owner else slug,
        "licenses": [{"name": "cc-by-sa-4.0"}],
        "description": description,
        "isPrivate": is_private,
    }
    target = dataset_dir / "dataset-metadata.json"
    try:
        target.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not write dataset-metadata.json for %s: %s", dataset_dir, exc)


def _call_kaggle_api(api, method_name: str, primary: object, primary_keys: tuple, extra: dict) -> None:
    """Call a ``KaggleApi`` dataset method across ``kaggle-api`` versions.

    ``primary`` is the main argument (upload folder / dataset slug) and
    ``primary_keys`` are its candidate parameter names (``folder`` in current
    releases, ``folder_path``-style names in older/alternate builds).  The
    supported parameters are probed with ``inspect`` and only those are
    passed.  When the signature cannot be inspected, the documented
    primary-first convention is used positionally.
    """
    import inspect

    method = getattr(api, method_name)
    candidates: dict[str, object] = dict(extra)
    for key in primary_keys:
        candidates.setdefault(key, primary)
    try:
        params = inspect.signature(method).parameters
    except (TypeError, ValueError):
        params = {}
    if params:
        names = set(params.keys())
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        kwargs = {k: v for k, v in candidates.items() if has_var_kw or k in names}
        present = [k for k in primary_keys if k in kwargs]
        if not present:
            # No recognized primary parameter: pass the primary positionally.
            return method(primary, **kwargs)
        # Drop alias primary keys so only the accepted one is sent.
        drop = set(primary_keys) - {present[0]}
        kwargs = {k: v for k, v in kwargs.items() if k not in drop}
        return method(**kwargs)
    return method(primary, **{k: v for k, v in extra.items()})


def _authenticate_kaggle(
    username: Optional[str],
    key: Optional[str],
) -> "tuple[object | None, Optional[str], Optional[str]]":
    """Import the client and authenticate. Returns ``(api, owner, error)``.

    ``owner`` is the dataset owner slug used for metadata/URLs: explicit or
    injected credentials first, then the client's own username attribute.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore[import-untyped]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Kaggle push unavailable (kaggle package not installed): %s", exc)
        return (
            None,
            None,
            "Kaggle upload unavailable (kaggle package not installed).  "
            "Install it with ``pip install kaggle`` and retry.",
        )
    resolved_user, resolved_key = _resolve_kaggle_credentials(username, key)
    owner: Optional[str] = None
    try:
        api = KaggleApi()
        # KaggleApi.authenticate() reads KAGGLE_USERNAME/KAGGLE_KEY env vars
        # and/or ~/.kaggle/kaggle.json.  If explicit creds were resolved above
        # we inject them as env vars so authenticate() picks them up.
        env_backup = {}
        if resolved_user:
            env_backup["KAGGLE_USERNAME"] = os.environ.get("KAGGLE_USERNAME")
            os.environ["KAGGLE_USERNAME"] = resolved_user
        if resolved_key:
            env_backup["KAGGLE_KEY"] = os.environ.get("KAGGLE_KEY")
            os.environ["KAGGLE_KEY"] = resolved_key
        try:
            api.authenticate()
            owner = resolved_user or os.environ.get("KAGGLE_USERNAME") or getattr(api, "username", None) or None
            if owner is not None:
                owner = str(owner).strip() or None
        finally:
            # Restore env to original state (never leave stale creds in the process).
            for env_key, original in env_backup.items():
                if original is None:
                    os.environ.pop(env_key, None)
                else:
                    os.environ[env_key] = original
    except Exception as exc:  # noqa: BLE001
        reason = _describe_error(exc)
        logger.warning("Kaggle auth failed: %s", reason)
        return (None, None, reason)
    return (api, owner, None)


def _validate_dataset_slug(dataset: object) -> Optional[str]:
    """Normalize ``owner/slug`` or return None when malformed."""
    text = str(dataset or "").strip().strip("/")
    parts = [p for p in text.split("/") if p]
    if len(parts) != 2 or any(p in (".", "..") for p in parts):
        return None
    return f"{parts[0]}/{parts[1]}"


def download_output_from_kaggle(
    dataset: str,
    dest_dir: "str | os.PathLike[str]",
    *,
    username: Optional[str] = None,
    key: Optional[str] = None,
) -> "tuple[bool, Optional[str], Optional[str]]":
    """Download a Kaggle dataset (``owner/slug``) into ``dest_dir``.

    Returns ``(ok, path, error)`` with the same non-fatal contract as the
    upload: callers surface ``error`` but already-existing state is untouched.
    The archive is unzipped in place so checkpoint layouts land directly under
    ``dest_dir``.
    """
    slug = _validate_dataset_slug(dataset)
    if slug is None:
        return (False, None, "Invalid Kaggle dataset: expected 'owner/slug'.")
    dest = Path(dest_dir).expanduser()
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return (False, None, f"Could not create restore directory {dest}: {exc}")

    api, _owner, auth_error = _authenticate_kaggle(username, key)
    if auth_error is not None or api is None:
        return (False, None, auth_error or "Kaggle authentication failed.")

    # Method names differ across kaggle-api releases: whole-dataset download
    # is ``dataset_download_files`` in current builds (incl. 2.0.x); older or
    # alternate builds expose ``dataset_download``.  Probe in order instead of
    # assuming one name so a missing method is a precise error, not a crash.
    method_name = next(
        (name for name in ("dataset_download_files", "dataset_download") if hasattr(api, name)),
        None,
    )
    if method_name is None:
        reason = (
            "Kaggle download failed: this kaggle package version has no "
            "dataset download API.  Upgrade with ``pip install -U kaggle`` "
            "and retry."
        )
        logger.warning("Kaggle download failed for %s: %s", slug, reason)
        return (False, None, reason)

    try:
        _call_kaggle_api(
            api,
            method_name,
            slug,
            ("dataset", "dataset_slug", "dataset_name", "dataset_id"),
            {
                "path": str(dest),
                "download_dir": str(dest),
                "dest": str(dest),
                "unzip": True,
                "quiet": True,
            },
        )
        logger.info("Kaggle dataset downloaded: %s -> %s", slug, dest)
        return (True, str(dest), None)
    except Exception as exc:  # noqa: BLE001
        reason = _describe_error(exc, operation = "download")
        logger.warning("Kaggle download failed for %s -> %s: %s", slug, dest, reason)
        return (False, None, reason)


def _describe_error(exc: Exception, *, operation: str = "upload") -> str:
    """Map a kaggle-package / Kaggle-API error to a precise, user-facing reason.

    ``operation`` is ``"upload"`` or ``"download"`` so the fallback message
    names the step that actually failed.
    """
    text = str(exc) or exc.__class__.__name__
    lowered = text.lower()

    if "401" in lowered or "403" in lowered or "forbidden" in lowered:
        return (
            "Kaggle authentication failed -- the username or API key is missing, "
            "invalid, or lacks dataset-write permission.  Verify KAGGLE_USERNAME "
            "and KAGGLE_KEY are set correctly."
        )
    if "404" in lowered or "not found" in lowered:
        return (
            "Kaggle dataset not found or you lack access.  Verify the dataset slug "
            "and that your account owns it."
        )
    if "already exists" in lowered or "409" in lowered:
        return (
            "Kaggle dataset already exists with a conflicting slug.  The run was "
            "saved locally; manual cleanup or a unique slug is needed."
        )
    if "rate limit" in lowered or "429" in lowered or "quota" in lowered:
        return "Kaggle API rate-limited or quota exceeded -- try again shortly."
    if "connect" in lowered or "resolve" in lowered or "timeout" in lowered or "ssl" in lowered:
        return (
            "Kaggle network error (could not connect).  Check internet access "
            "and retry."
        )
    if "credential" in lowered or "kaggle.json" in lowered:
        return (
            "Kaggle credentials not found.  On a Kaggle notebook this should be "
            "automatic; otherwise set KAGGLE_USERNAME + KAGGLE_KEY or write "
            "~/.kaggle/kaggle.json."
        )
    if "has no attribute" in lowered:
        return (
            f"Kaggle {operation} failed: this kaggle package version has no "
            f"matching API ({text}).  Upgrade with ``pip install -U kaggle`` "
            "and retry."
        )
    return f"Kaggle {operation} failed: {text}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def push_output_to_kaggle(
    output_dir: "str | os.PathLike[str]",
    *,
    slug: Optional[str] = None,
    is_private: bool = True,
    username: Optional[str] = None,
    key: Optional[str] = None,
    description: Optional[str] = None,
) -> PushResult:
    """Upload ``output_dir`` to the Kaggle platform as a Dataset.

    Returns ``(ok, dataset_url, error)``:
      - ``(True, url, None)``       on a fully successful upload.
      - ``(False, None, reason)``   on a genuine failure, with a precise message.
      - ``(False, None, None)``     when skipped (no output dir).

    On Kaggle notebooks ``KAGGLE_USERNAME`` and ``KAGGLE_KEY`` are injected
    automatically by the platform, so the caller need not pass credentials.

    Callers should surface ``error`` to the UI but never treat a failure as fatal
    (artifacts already live on disk).
    """
    root = Path(output_dir).expanduser()
    if not root.is_dir():
        logger.warning("Kaggle push skipped: output_dir not a directory: %s", root)
        return (False, None, None)

    # --- 1-3. Import, resolve credentials, authenticate ---
    api, owner, auth_error = _authenticate_kaggle(username, key)
    if auth_error is not None or api is None:
        return (False, None, auth_error or "Kaggle authentication failed.")

    # --- 4. Build dataset slug & metadata ---
    resolved_slug = (str(slug).strip() if slug else None) or _slug_from_output_dir(root)
    description = description or f"Unsloth training output: {root.name}"
    _write_metadata(root, resolved_slug, is_private, description, owner=owner)
    dataset_url = (
        f"https://www.kaggle.com/datasets/{owner}/{resolved_slug}" if owner else None
    )

    # --- 5. Create or update the dataset ---
    # The real kaggle-api client takes only a folder (+ options); title, slug
    # and visibility come from dataset-metadata.json written above.  When the
    # dataset already exists the API raises; we catch that and fall back to
    # ``dataset_create_version`` so re-runs don't break.
    try:
        _call_kaggle_api(
            api,
            "dataset_create_new",
            str(root),
            ("folder", "folder_path", "dir", "path", "dataset_dir"),
            {"public": not is_private, "is_private": is_private, "private": is_private, "quiet": True},
        )
        logger.info("Kaggle dataset created: %s -> %s", root, dataset_url)
        return (True, dataset_url, None)
    except Exception as create_exc:  # noqa: BLE001
        create_text = str(create_exc).lower()
        if "already exists" not in create_text and "409" not in create_text:
            return (False, None, _describe_error(create_exc))

    # Dataset already exists: push a new version.
    try:
        _call_kaggle_api(
            api,
            "dataset_create_version",
            str(root),
            ("folder", "folder_path", "dir", "path", "dataset_dir"),
            {"version_notes": f"Auto-updated by Unsloth training run: {root.name}", "notes": f"Auto-updated by Unsloth training run: {root.name}", "quiet": True},
        )
        logger.info("Kaggle dataset version updated: %s -> %s", root, dataset_url)
        return (True, dataset_url, None)
    except Exception as exc:  # noqa: BLE001
        reason = _describe_error(exc)
        logger.warning("Kaggle push failed for %s -> %s: %s", root, resolved_slug, reason)
        return (False, None, reason)
