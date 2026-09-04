# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for :mod:`core.training.upload_progress` (stdlib-only)."""

import importlib.util
import threading
import time
from pathlib import Path


_BACKEND = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "training_upload_progress_under_test",
        _BACKEND / "core" / "training" / "upload_progress.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


up = _load_module()


def test_format_bytes():
    assert up.format_bytes(0) == "0 B"
    assert up.format_bytes(512) == "512 B"
    assert up.format_bytes(1024) == "1.0 KB"
    assert up.format_bytes(46.68 * 1024 * 1024) == "46.7 MB"
    assert up.format_bytes(12 * 1024**3) == "12.0 GB"
    assert up.format_bytes(-1) == "unknown size"
    assert up.format_bytes("nope") == "unknown size"


def test_heartbeat_emits_status_and_progress_then_stops():
    events = []
    stop = threading.Event()
    thread = threading.Thread(
        target=up.upload_heartbeat_loop,
        kwargs={
            "put": events.append,
            "label": "Kaggle Dataset",
            "size_str": "4.2 GB",
            "stop": stop,
            "interval_s": 0.01,
            "max_beats": 50,
        },
        daemon=True,
    )
    thread.start()
    time.sleep(0.12)
    stop.set()
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    kinds = [e["type"] for e in events]
    assert "status" in kinds and "upload_progress" in kinds
    first_status = next(e for e in events if e["type"] == "status")
    assert "Kaggle Dataset" in first_status["message"] and "4.2 GB" in first_status["message"]
    assert all(e["elapsed_s"] >= 0 for e in events if e["type"] == "upload_progress")


def test_heartbeat_stops_immediately_when_already_set():
    events = []
    stop = threading.Event()
    stop.set()
    up.upload_heartbeat_loop(
        events.append, label="x", size_str="1 B", stop=stop, interval_s=0.01, max_beats=5
    )
    assert events == []


def test_exemption_only_while_fresh_and_capped():
    assert up.upload_exempts_kill(None, None, 100.0) is False
    assert up.upload_exempts_kill(0.0, None, 10.0) is False
    # Fresh beat, young upload -> exempt.
    assert up.upload_exempts_kill(0.0, 90.0, 100.0) is True
    # Stale beat -> no exemption (normal watchdog behavior resumes).
    assert up.upload_exempts_kill(0.0, 0.0, 500.0) is False
    # Fresh beat but past the absolute cap -> no exemption.
    assert up.upload_exempts_kill(0.0, 3599.0, 3600.0) is False
    assert up.upload_exempts_kill("bad", "data", 1.0) is False


def test_note_and_reset_beats():
    class _State:
        pass

    state = _State()
    up.note_upload_beat(state, now=10.0)
    assert state._upload_started_at == 10.0
    assert state._upload_last_beat == 10.0
    up.note_upload_beat(state, now=20.0)
    assert state._upload_started_at == 10.0
    assert state._upload_last_beat == 20.0
    up.reset_upload_beats(state)
    assert state._upload_started_at is None
    assert state._upload_last_beat is None
