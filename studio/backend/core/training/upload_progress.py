# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Heartbeat + watchdog-exemption policy for multi-GB storage uploads.

A full-finetune checkpoint bundle (full weights + optimizer states) can take
tens of minutes to zip and upload.  Without signs of life the UI looks hung
and the parent's stop watchdog (15s grace after ``complete``, 600s absolute
backstop from the stop request) can SIGKILL the worker mid-upload.  This
module holds the pieces both sides share:

- the worker heartbeat loop (emits UI-visible ``status`` beats plus a machine
  ``upload_progress`` event the parent uses below),
- the parent's exemption predicate (never kill while beats are fresh, bounded
  by an absolute cap so a wedged upload can't hold the GPU forever).

Stdlib-only on purpose: imported by the heavy worker/parent and by unit
tests alike.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

# Heartbeat cadence inside the worker during a blocking upload.
UPLOAD_HEARTBEAT_INTERVAL_S = 45.0
# Upper bound on beats per upload (45 * ~45s ~= 30min of signs of life).
UPLOAD_HEARTBEAT_MAX_BEATS = 40
# Parent side: beats fresher than this exempt the worker from watchdog kills.
UPLOAD_EXEMPT_FRESH_S = 120.0
# ... but total exemption never exceeds this from the first beat.
UPLOAD_EXEMPT_CAP_S = 3600.0


def format_bytes(num_bytes: float) -> str:
    """Human-readable byte count for status lines (``4.2 GB``)."""
    try:
        size = float(num_bytes)
    except (TypeError, ValueError):
        return "unknown size"
    if size < 0:
        return "unknown size"
    units = ("B", "KB", "MB", "GB", "TB")
    unit = 0
    while size >= 1024.0 and unit < len(units) - 1:
        size /= 1024.0
        unit += 1
    if unit == 0:
        return f"{int(size)} {units[unit]}"
    return f"{size:.1f} {units[unit]}"


def upload_heartbeat_loop(
    put,
    *,
    label: str,
    size_str: str,
    stop,
    interval_s: float = UPLOAD_HEARTBEAT_INTERVAL_S,
    max_beats: int = UPLOAD_HEARTBEAT_MAX_BEATS,
) -> None:
    """Emit periodic signs of life while a blocking upload runs.

    ``put`` receives event dicts (the worker's queue put or the MLX ``_send``
    closure adapted to it).  Each beat is a UI-visible ``status`` plus a
    machine ``upload_progress`` carrying elapsed seconds for the parent's
    watchdog exemption.  Exits promptly when ``stop`` is set; never raises.
    """
    start = time.monotonic()
    try:
        beats = max(0, int(max_beats))
    except (TypeError, ValueError):
        beats = 0
    try:
        interval = float(interval_s)
    except (TypeError, ValueError):
        interval = UPLOAD_HEARTBEAT_INTERVAL_S
    if interval <= 0:
        interval = UPLOAD_HEARTBEAT_INTERVAL_S
    for _ in range(beats):
        try:
            if stop.wait(interval):
                return
        except Exception:
            return
        elapsed = time.monotonic() - start
        try:
            put(
                {
                    "type": "status",
                    "message": f"Uploading to {label}... ({_format_elapsed(elapsed)}, {size_str})",
                    "ts": time.time(),
                }
            )
            put({"type": "upload_progress", "elapsed_s": elapsed, "ts": time.time()})
        except Exception:
            return


def _format_elapsed(seconds: float) -> str:
    try:
        total = max(0, int(seconds))
    except (TypeError, ValueError):
        return "?s"
    minutes, secs = divmod(total, 60)
    if minutes <= 0:
        return f"{secs}s"
    hours, minutes = divmod(minutes, 60)
    if hours <= 0:
        return f"{minutes}m {secs:02d}s"
    return f"{hours}h {minutes:02d}m"


def upload_exempts_kill(
    started_at: Optional[float],
    last_beat: Optional[float],
    now: Optional[float] = None,
) -> bool:
    """Whether an in-flight upload exempts the worker from watchdog kills.

    True only while beats are fresh (``UPLOAD_EXEMPT_FRESH_S``) AND the whole
    upload is under ``UPLOAD_EXEMPT_CAP_S``.  Stale/absent markers exempt
    nothing, so normal watchdog behavior is unchanged outside uploads.
    """
    if started_at is None or last_beat is None:
        return False
    try:
        current = time.monotonic() if now is None else float(now)
        start = float(started_at)
        beat = float(last_beat)
    except (TypeError, ValueError):
        return False
    if current - start >= UPLOAD_EXEMPT_CAP_S:
        return False
    return current - beat < UPLOAD_EXEMPT_FRESH_S


def note_upload_beat(state: Any, now: Optional[float] = None) -> None:
    """Record an ``upload_progress`` beat on parent state (or any namespace)."""
    current = time.monotonic() if now is None else now
    if getattr(state, "_upload_started_at", None) is None:
        state._upload_started_at = current
    state._upload_last_beat = current


def reset_upload_beats(state: Any) -> None:
    """Clear upload-exemption markers (new run start / terminal event)."""
    state._upload_started_at = None
    state._upload_last_beat = None
