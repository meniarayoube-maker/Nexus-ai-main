# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Stop-and-save through /tmp staging for tight disks.

Contract (the whole point of this module): when the working disk cannot hold
old + new checkpoint side by side, the new bundle is written to the roomy
system temp, verified, and uploaded FIRST -- and only then are the superseded
local checkpoints pruned and the bundle moved into place.  At every failure
point the previous checkpoint stays intact locally, except after a successful
upload (where Kaggle dataset versions are the rollback).

Design rules, all enforced below and covered by tests:
- Decision errors cost performance, never correctness: "tmp_flow" on a roomy
  disk just adds a copy + an extra dataset version; "legacy" on a tight disk
  fails exactly like today (loud ENOSPC error, old intact).
- Prune NEVER runs before a successful upload of the new bundle.
- Move failures clean their partial destination; the tmp bundle is kept for
  manual recovery and its path is reported.
- No torch/transformers/kaggle imports at module level: stdlib only, so unit
  tests load this without the GPU stack (heavy integrations stay lazy).

Statuses: "swapped" | "write_failed" | "verify_failed" | "upload_failed" |
"prune_failed" | "move_failed" | "reverify_failed".
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

CHECKPOINT_PREFIX = "checkpoint-"
TMP_SWAP_PREFIX = ".stop-ckpt-"
# Legacy in-place need ~= new bundle + final root rewrite + margin.  Only used
# to DECIDE the path; a wrong guess costs a slower (but correct) tmp-flow.
LEGACY_NEED_MARGIN = 2.2


class StopSwapResult:
    """Outcome of :func:`execute_stop_swap` (never raises; plain class)."""

    def __init__(
        self,
        status,
        message,
        tmp_path = None,
        new_path = None,
        pruned = (),
        upload_url = None,
    ):
        self.status = status
        self.message = message
        self.tmp_path = tmp_path
        self.new_path = new_path
        self.pruned = tuple(pruned)
        self.upload_url = upload_url

    def __repr__(self):  # pragma: no cover - debugging aid.
        return f"StopSwapResult(status={self.status!r}, message={self.message!r})"


def checkpoint_step_num(name: object) -> int:
    """Numeric step of a ``checkpoint-N`` dir name, else -1 (sorts last)."""
    text = name if isinstance(name, str) else getattr(name, "name", str(name))
    if not text.startswith(CHECKPOINT_PREFIX):
        return -1
    try:
        return int(text.split("-", 1)[1])
    except (ValueError, IndexError):
        return -1


def disk_free_bytes(path: object) -> Optional[int]:
    """Free bytes on ``path``'s filesystem, None when unmeasurable."""
    try:
        return shutil.disk_usage(str(path)).free
    except Exception:
        return None


def dir_size_bytes(path: object) -> Optional[int]:
    """Recursive byte size, None when unreadable."""
    total = 0
    try:
        for dirpath, _dirnames, filenames in os.walk(path):
            for name in filenames:
                try:
                    total += (Path(dirpath) / name).stat().st_size
                except OSError:
                    continue
    except OSError:
        return None
    return total


def latest_bundle_size_bytes(working_dir: object) -> Optional[int]:
    """Size of the newest ``checkpoint-*`` dir, None when absent/unreadable."""
    try:
        candidates = [
            p
            for p in Path(str(working_dir)).iterdir()
            if p.is_dir() and p.name.startswith(CHECKPOINT_PREFIX)
        ]
    except OSError:
        return None
    if not candidates:
        return None
    newest = max(candidates, key = lambda p: checkpoint_step_num(p.name))
    return dir_size_bytes(newest)


def plan_stop_save(working_dir: object) -> str:
    """``"tmp_flow"`` when a legacy in-place save likely won't fit.

    Compares free space against ``latest_bundle * LEGACY_MARGIN`` (new bundle
    + final root rewrite + margin).  Unmeasurable anything -> ``"legacy"``
    (today's behavior, unchanged).
    """
    try:
        latest = latest_bundle_size_bytes(working_dir)
        if latest is None:
            return "legacy"
        free = disk_free_bytes(working_dir)
        if free is None:
            return "legacy"
        return "tmp_flow" if free < int(latest * LEGACY_NEED_MARGIN) else "legacy"
    except Exception:
        return "legacy"


def normalize_upload_result(out: object) -> "tuple[bool, Optional[str], Optional[str]]":
    """Normalize an upload callback result to ``(ok, url, error)``."""
    try:
        if out is None:
            return (False, None, "upload failed")
        if isinstance(out, dict):
            ok = bool(out.get("ok"))
            url = out.get("dataset_url") or out.get("repo_url")
            error = out.get("error")
            return (ok, str(url) if url else None, str(error) if error else (None if ok else "upload failed"))
        if isinstance(out, (tuple, list)) and len(out) == 3:
            ok, url, error = out
            return (bool(ok), (str(url) if url else None), (str(error) if error else None))
    except Exception:
        pass
    return (False, None, "upload returned an unexpected result")


def safe_upload_bundle(upload_cb: Callable[[str], object], bundle_path: str) -> "tuple[bool, Optional[str], Optional[str]]":
    """Call an upload callback without ever raising."""
    try:
        return normalize_upload_result(upload_cb(bundle_path))
    except Exception as exc:  # noqa: BLE001 - total callback, report it.
        return (False, None, f"upload dispatch crashed: {exc}")


def prune_superseded_checkpoints(
    working_dir: object, keep: object = frozenset()
) -> list:
    """Delete ``checkpoint-*`` dirs in ``working_dir`` except ``keep`` names.

    Only checkpoint-prefixed DIRECTORIES are ever touched; every other file
    (weights, tokenizer, configs, snapshots) is left alone.  Returns pruned
    dir names, sorted.  Raises on failure (caller decides; nothing is hidden).
    """
    keep_names = set(keep) if keep else set()
    pruned = []
    root = Path(str(working_dir))
    for entry in sorted(root.iterdir(), key = lambda p: p.name):
        if not entry.is_dir() or not entry.name.startswith(CHECKPOINT_PREFIX):
            continue
        if entry.name in keep_names:
            continue
        shutil.rmtree(str(entry))
        pruned.append(entry.name)
    return pruned


def move_bundle_into_place(src_ckpt_dir: object, dst_ckpt_dir: object) -> str:
    """Move a verified bundle into the working dir; cleans partials on failure."""
    src = str(src_ckpt_dir)
    dst = str(dst_ckpt_dir)
    try:
        shutil.move(src, dst)
        return dst
    except Exception:
        try:
            target = Path(dst)
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(str(target), ignore_errors = True)
        except Exception:
            pass
        raise


def cleanup_path(path: object) -> None:
    """Best-effort recursive remove; never raises."""
    try:
        target = Path(str(path))
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(str(target), ignore_errors = True)
        elif target.exists() or target.is_symlink():
            target.unlink(missing_ok = True)
    except Exception:
        pass


def orphaned_tmp_bundles() -> list:
    """Leftover ``.stop-ckpt-*`` staging dirs in system temp (informational)."""
    found = []
    try:
        tmp = Path(tempfile.gettempdir())
        for entry in tmp.iterdir():
            if entry.is_dir() and entry.name.startswith(TMP_SWAP_PREFIX):
                found.append(str(entry))
    except OSError:
        pass
    return sorted(found)


def _default_verify(bundle_dir: str) -> Optional[str]:
    try:
        from core.training.resume import is_resume_checkpoint_valid

        valid = is_resume_checkpoint_valid(Path(bundle_dir))
    except Exception as exc:  # noqa: BLE001 - validator itself failed.
        return f"checkpoint validator crashed: {exc}"
    return None if valid else f"{bundle_dir} is not a resume-valid bundle"


def execute_stop_swap(
    *,
    working_dir: str,
    checkpoint_name: str,
    write_bundle: Callable[[str], None],
    upload_bundle: Callable[[str], object],
    verify_bundle: Optional[Callable[[str], Optional[str]]] = None,
    prune_old: Optional[Callable[[], list]] = None,
    move_in: Optional[Callable[[str, str], str]] = None,
    make_tmp_parent: Optional[Callable[[], str]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> StopSwapResult:
    """Run the tmp-staged stop-save sequence. Never raises.

    Order (the safety contract): write tmp -> verify tmp -> upload tmp ->
    prune old -> move in -> re-verify working.  Prune runs ONLY after a
    successful upload, so the previous checkpoint is intact locally at every
    earlier failure, and on Kaggle versions afterwards.
    """
    say = log or (lambda message: logger.info("%s", message))
    verify = verify_bundle or _default_verify
    do_prune = prune_old or (lambda: prune_superseded_checkpoints(working_dir, keep = {checkpoint_name}))
    do_move = move_in or move_bundle_into_place

    def _mkparent() -> str:
        if make_tmp_parent is not None:
            return make_tmp_parent()
        return tempfile.mkdtemp(prefix = TMP_SWAP_PREFIX)

    try:
        orphans = orphaned_tmp_bundles()
    except Exception:
        orphans = []
    if orphans:
        say(f"Previous failed stop-save staging still present (harmless): {orphans}")

    try:
        tmp_parent = _mkparent()
    except Exception as exc:
        return StopSwapResult("write_failed", f"Could not create tmp staging: {exc}")
    tmp_ckpt = os.path.join(tmp_parent, checkpoint_name)
    try:
        write_bundle(tmp_ckpt)
    except Exception as exc:
        cleanup_path(tmp_parent)
        return StopSwapResult(
            "write_failed",
            f"Stop checkpoint could not be written ({exc}); previous checkpoint untouched in {working_dir}.",
        )
    reason = None
    try:
        reason = verify(tmp_ckpt)
    except Exception as exc:  # noqa: BLE001 - total validator.
        reason = f"checkpoint validator crashed: {exc}"
    if reason:
        cleanup_path(tmp_parent)
        return StopSwapResult(
            "verify_failed",
            f"Fresh stop checkpoint failed validation ({reason}); previous checkpoint untouched in {working_dir}.",
        )
    ok, url, error = safe_upload_bundle(upload_bundle, tmp_ckpt)
    if not ok:
        return StopSwapResult(
            "upload_failed",
            f"Stop checkpoint upload failed ({error or 'unknown error'}); previous checkpoint untouched in "
            f"{working_dir}, fresh bundle kept at {tmp_parent} for manual recovery.",
            tmp_path = tmp_parent,
        )
    try:
        pruned = list(do_prune())
    except Exception as exc:
        return StopSwapResult(
            "prune_failed",
            f"Uploaded ({url}), but pruning old checkpoints failed ({exc}); new bundle kept at {tmp_parent}, "
            f"previous checkpoints partially intact -- resolve manually before resuming.",
            tmp_path = tmp_parent,
            upload_url = url,
        )
    new_path = os.path.join(working_dir, checkpoint_name)
    try:
        do_move(tmp_ckpt, new_path)
    except Exception as exc:
        return StopSwapResult(
            "move_failed",
            f"Uploaded ({url}), but moving the bundle into place failed ({exc}); re-download dataset {url} "
            f"or copy from {tmp_parent}; partial destination was cleaned.",
            tmp_path = tmp_parent,
            pruned = tuple(pruned),
            upload_url = url,
        )
    reason = None
    try:
        reason = verify(new_path)
    except Exception as exc:  # noqa: BLE001 - total validator.
        reason = f"checkpoint validator crashed: {exc}"
    cleanup_path(tmp_parent)
    if reason:
        return StopSwapResult(
            "reverify_failed",
            f"Moved bundle failed validation in place ({reason}); re-download dataset {url}; "
            f"pruned locally: {list(pruned)}.",
            new_path = new_path,
            pruned = tuple(pruned),
            upload_url = url,
        )
    say(f"Stop checkpoint swapped into place: {new_path} (pruned {list(pruned)}, uploaded {url})")
    return StopSwapResult(
        "swapped", "ok", new_path = new_path, pruned = tuple(pruned), upload_url = url
    )
