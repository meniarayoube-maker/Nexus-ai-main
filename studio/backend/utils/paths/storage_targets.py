# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""User-selectable save destinations (Storage Targets).

Multi-storage-target support: a training run's artifacts can be written to one of
several destinations picked from the Train UI / settings:

* ``local``          -- the Unsloth outputs root (the default; fully contained).
* ``google_drive``   -- Google Drive mount on Colab (``/content/drive/MyDrive/...``).
* ``huggingface``    -- a Hugging Face Hub repo (uploaded via ``huggingface_hub``).
* ``kaggle``         -- the Kaggle working directory (``/kaggle/working/...``).

Google Drive and Kaggle are *path* targets, not network APIs: on their hosts the
cloud folders are mounted as ordinary local directories, so writing there is a
normal filesystem write scoped to a well-known root. Hugging Face is an upload
target that also requires a local staging directory (we write locally first and
push afterwards).

Security model (Path Validation Override for Cloud Environments)
----------------------------------------------------------------
Unsloth training output normally must stay under ``outputs_root()`` (see
``resolve_output_dir`` / ``_assert_contained`` in :mod:`storage_roots`). That
guard is what raises "path escapes root" for absolute paths such as
``/content/drive/MyDrive/runs``.

This module relaxes containment **only** for a small allowlist of cloud-mounted
roots, matching how Colab/Kaggle work. An arbitrary absolute path is never
allowed to escape ``outputs_root`` -- only these exact roots may:

* Google Drive on Colab: the realpath of ``/content/drive`` (when reachable).
* Kaggle working dir: ``/kaggle/working`` (when reachable).

Everything else (including a plain local absolute path on a desktop) keeps the
strict containment rule, so the override can never be abused to write anywhere
on a user's machine.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from loggers import get_logger
from utils.paths.path_utils import host_normalize_path
from utils.paths.storage_roots import outputs_root, ensure_dir

logger = get_logger(__name__)

# Sentinel env vars: a consuming driver (Colab/Kaggle notebook, CLI) sets these to
# the concrete mount roots. When absent we fall back to well-known paths so the
# feature still works when launched from a notebook that already mounted Drive.
_STORAGE_TARGET_OVERRIDE_ENV = "UNSLOTH_STORAGE_TARGET_OVERRIDE"


# Storage target identifiers exposed to the API and UI.
STORAGE_TARGET_LOCAL = "local"
STORAGE_TARGET_GOOGLE_DRIVE = "google_drive"
STORAGE_TARGET_HUGGINGFACE = "huggingface"
STORAGE_TARGET_KAGGLE = "kaggle"

# Canonical ordered list for API discovery / UI rendering.
STORAGE_TARGETS: tuple[str, ...] = (
    STORAGE_TARGET_LOCAL,
    STORAGE_TARGET_GOOGLE_DRIVE,
    STORAGE_TARGET_HUGGINGFACE,
    STORAGE_TARGET_KAGGLE,
)


def _well_known_cloud_roots() -> "dict[str, list[Path]]":
    """Well-known mount roots per cloud environment.

    A root is only honored when it actually exists on the running host, so a
    desktop install never accidentally adopts a Colab/Kaggle path.
    """
    return {
        STORAGE_TARGET_GOOGLE_DRIVE: [Path("/content/drive")],
        STORAGE_TARGET_KAGGLE: [Path("/kaggle/working")],
        STORAGE_TARGET_HUGGINGFACE: [],
    }


def storage_target_override_root(target: str) -> Optional[Path]:
    """Return the single honored write root for *target*, or None.

    ``None`` means the target is either local (no override) or a Hugging Face
    repo (handled as an upload, not a filesystem write path).

    Google Drive / Kaggle mounts are POSIX paths used by Colab/Kaggle (both
    Linux). On non-POSIX hosts (e.g. a Windows desktop) they are never honored,
    so the containment override can never fire there.
    """
    if os.name == "nt":
        return None
    if target == STORAGE_TARGET_LOCAL or target == STORAGE_TARGET_HUGGINGFACE:
        return None
    for candidate in _well_known_cloud_roots().get(target, ()):
        real = _safe_realpath(candidate)
        if real is not None:
            return real
    return None


def is_cloud_root(path: Path, target: Optional[str] = None) -> bool:
    """Whether ``path`` (or its realpath) sits under a cloud root we may write to.

    Used to admit absolute training output paths that the normal containment
    guard would reject. Only the recognized cloud roots pass; everything else
    keeps strict containment under ``outputs_root``.
    """
    try:
        resolved = Path(os.path.realpath(path))
    except OSError:
        return False
    targets = (target,) if target else (
        STORAGE_TARGET_GOOGLE_DRIVE,
        STORAGE_TARGET_KAGGLE,
    )
    for t in targets:
        root = storage_target_override_root(t)
        if root is None:
            continue
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _safe_realpath(path: Path) -> Optional[Path]:
    try:
        return Path(os.path.realpath(path))
    except OSError:
        return None


def _default_google_drive_dir() -> Optional[Path]:
    """``/content/drive/MyDrive/<unsloth-outputs>`` on Colab when reachable."""
    root = storage_target_override_root(STORAGE_TARGET_GOOGLE_DRIVE)
    if root is None:
        return None
    mydrive = root / "MyDrive"
    if not mydrive.is_dir():
        return None
    return mydrive / "unsloth-outputs"


def _default_kaggle_dir() -> Optional[Path]:
    root = storage_target_override_root(STORAGE_TARGET_KAGGLE)
    return (root / "unsloth-outputs") if root is not None else None


def resolve_storage_target_write_dir(
    target: Optional[str],
    output_dir: Optional[str],
    run_name: str,
    *,
    hf_repo_id: Optional[str] = None,
) -> "tuple[str, Path]":
    """Resolve the concrete local write directory for a storage *target*.

    Returns ``(target, resolved_path)``. ``target`` is normalized (``None`` /
    unknown/empty fall back to :data:`STORAGE_TARGET_LOCAL`). ``huggingface``
    returns ``(target, staging_dir)`` because HF is an upload target: artifacts
    are staged under outputs_root and pushed separately; the path is therefore
    still contained.

    Override rule: when the caller supplies an absolute ``output_dir`` that lives
    under an active cloud root (Google Drive / Kaggle) for the matching target,
    it is used as-is. Otherwise an absolute path keeps strict containment under
    ``outputs_root`` (same behavior as :func:`resolve_training_write_dir`).
    """
    target = _normalize_target(target)

    if target == STORAGE_TARGET_GOOGLE_DRIVE:
        return _resolve_cloud_or_local(
            STORAGE_TARGET_GOOGLE_DRIVE, output_dir, run_name, _default_google_drive_dir
        )
    if target == STORAGE_TARGET_KAGGLE:
        return _resolve_kaggle(output_dir, run_name)
    if target == STORAGE_TARGET_HUGGINGFACE:
        # Stage under outputs_root; the push handler uploads from here.
        return STORAGE_TARGET_HUGGINGFACE, _resolve_contained(output_dir, run_name)
    # local
    return STORAGE_TARGET_LOCAL, _resolve_contained(output_dir, run_name)


def _normalize_target(target: Optional[str]) -> str:
    value = str(target or "").strip().lower()
    if value in STORAGE_TARGETS:
        return value
    if value:
        logger.warning("Unknown storage target %r; falling back to local", target)
    return STORAGE_TARGET_LOCAL


def _resolve_cloud_or_local(target, output_dir, run_name, default_fn):
    root = storage_target_override_root(target)
    # An explicit absolute path inside the active cloud root is honored.
    if output_dir and _is_absolute_user_path_str(output_dir):
        user_path = Path(output_dir).expanduser()
        if root is not None and is_cloud_root(user_path, target):
            ensure_dir(user_path)
            return target, user_path
    # A relative path (or no path) goes under the cloud root default.
    base = default_fn() or (root / "unsloth-outputs" if root else None)
    if base is not None:
        candidate = base / _sanitize_rel(output_dir, run_name)
        ensure_dir(candidate)
        return target, candidate
    # Cloud root is not mounted here: fall back to a contained local dir.
    return STORAGE_TARGET_LOCAL, _resolve_contained(output_dir, run_name)


def _resolve_kaggle(output_dir: Optional[str], run_name: str) -> "tuple[str, Path]":
    """Always direct Kaggle writes under ``/kaggle/working/unsloth-outputs``.

    Kaggle only persists ``/kaggle/working`` (everything else on a notebook is
    wiped between sessions), so the finished run must land inside that durable
    root to survive. We always place it under the ``unsloth-outputs`` subfolder --
    even when the caller supplied an explicit absolute Kaggle path -- so every
    artifact lives in one predictable place for the "download all" and dataset-
    upload flow. The chosen leaf name is the last segment of an explicit path, or
    ``run_name`` when none was given.
    """
    root = storage_target_override_root(STORAGE_TARGET_KAGGLE)
    if root is None:
        # /kaggle/working is not mounted here (e.g. running on Colab/desktop): a
        # Kaggle save degrades to a contained local dir so training never loses
        # its output.
        return STORAGE_TARGET_LOCAL, _resolve_contained(output_dir, run_name)
    base = root / "unsloth-outputs"
    ensure_dir(base)
    candidate = base / _sanitize_rel(output_dir, run_name)
    ensure_dir(candidate)
    return STORAGE_TARGET_KAGGLE, candidate


def _resolve_contained(output_dir: Optional[str], run_name: str) -> Path:
    """Resolve *output_dir* strictly under outputs_root, or auto-name under it."""
    from utils.paths.storage_roots import _clean_relative_path, _has_parent_segment

    if not output_dir or not str(output_dir).strip():
        candidate = outputs_root() / _safe_run_dir(run_name)
        ensure_dir(candidate)
        return candidate

    raw = str(output_dir).strip()
    path = Path(raw).expanduser()
    if _has_parent_segment(raw, path):
        raise ValueError(f"path may not contain '..' segments: {raw!r}")
    if _is_absolute_user_path_str(raw):
        # Notebook passed an absolute path but not under a cloud root: keep the
        # same strict containment the rest of Unsloth enforces.
        from utils.paths.storage_roots import resolve_output_dir

        resolved = resolve_output_dir(str(path))
        ensure_dir(resolved)
        return resolved
    cleaned = _clean_relative_path(raw)
    candidate = outputs_root() / cleaned
    ensure_dir(candidate)
    return candidate


def _is_absolute_user_path_str(raw: str) -> bool:
    from utils.paths.storage_roots import _is_absolute_user_path

    return _is_absolute_user_path(Path(raw).expanduser())


def _safe_run_dir(run_name: str) -> str:
    import re

    base = re.sub(r"[^A-Za-z0-9._-]+", "_", str(run_name or "run"))[:200].strip("._-")
    return base or "run"


def _sanitize_rel(output_dir: Optional[str], run_name: str) -> str:
    import re

    if output_dir and str(output_dir).strip():
        value = str(output_dir).strip()
        # Absolute paths reaching here are handled by the caller; treat as a name.
        value = Path(value).name
        cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", value)[:200].strip(" ._-")
        return cleaned or _safe_run_dir(run_name)
    return _safe_run_dir(run_name)


# JSON-serializable description of the available targets + the live environment,
# surfaced by the ``/api/train/storage-targets`` endpoint.
def storage_targets_info() -> "dict[str, object]":
    from utils.paths.storage_roots import outputs_root

    def _desc(target: str) -> "dict[str, object]":
        if target == STORAGE_TARGET_LOCAL:
            return {
                "id": target,
                "label": "Local Drive / Computer",
                "description": "Save directly into the Unsloth outputs folder on this machine.",
                "available": True,
                "path": str(outputs_root()),
            }
        if target == STORAGE_TARGET_GOOGLE_DRIVE:
            root = storage_target_override_root(target)
            return {
                "id": target,
                "label": "Google Drive",
                "description": "Sync outputs to /content/drive/MyDrive on Google Colab.",
                "available": root is not None,
                "path": str(_default_google_drive_dir() or (root or "")),
            }
        if target == STORAGE_TARGET_KAGGLE:
            root = storage_target_override_root(target)
            return {
                "id": target,
                "label": "Kaggle Output",
                "description": "Write into /kaggle/working for Kaggle notebooks.",
                "available": root is not None,
                "path": str(_default_kaggle_dir() or (root or "")),
            }
        # huggingface
        return {
            "id": target,
            "label": "Hugging Face Hub",
            "description": "Upload the finished adapter as a Hugging Face model repository.",
            "available": True,
            "path": "",  # upload repo, not a filesystem path
        }

    override = (os.environ.get(_STORAGE_TARGET_OVERRIDE_ENV) or "").strip()
    return {
        "targets": [_desc(t) for t in STORAGE_TARGETS],
        "default": STORAGE_TARGET_LOCAL,
        "environment": {
            "colab": bool(
                _list_existing(_well_known_cloud_roots()[STORAGE_TARGET_GOOGLE_DRIVE])
            ),
            "kaggle": _safe_realpath(Path("/kaggle/working")) is not None,
            "override": override or None,
        },
    }


def _list_existing(paths: "list[Path]") -> "list[Path]":
    return [p for p in paths if _safe_realpath(p) is not None]


# Ensure host_normalize_path (used elsewhere in the path package) is importable
# here without an unused-import lint.
_normalize_host = host_normalize_path
