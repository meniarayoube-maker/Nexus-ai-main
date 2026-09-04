# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for stop-save checkpoint verification/backfill.

Loads ``resume.py`` standalone (same pattern as test_training_resume.py) so no
GPU stack is needed.  ``torch`` is stubbed with a minimal writer: the validity
gate only inspects zip structure (``archive/data.pkl`` ending in a STOP op,
plus a tensor record for model files), so stdlib ``zipfile``/``pickle``
exercise the exact production checks.
"""

import importlib.util
import io
import json
import pickle
import sys
import zipfile
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parents[1]


def _load_resume_module():
    spec = importlib.util.spec_from_file_location(
        "training_resume_bundle_under_test",
        _BACKEND / "core" / "training" / "resume.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


resume = _load_resume_module()


class _FakeTorch:
    """Minimal ``torch.save``: a zip whose data.pkl is a STOP-terminated pickle."""

    @staticmethod
    def save(obj, path):
        payload = obj if isinstance(obj, dict) else {}
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("archive/data.pkl", pickle.dumps(payload))


@pytest.fixture(autouse=True)
def _stub_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())


class _FakeState:
    def __init__(self, step):
        self.global_step = step

    def save_to_json(self, path):
        Path(path).write_text(json.dumps({"global_step": self.global_step}))


class _FakeOpt:
    def __init__(self, payload=None):
        self._payload = payload if payload is not None else {"state": {}, "groups": []}

    def state_dict(self):
        return self._payload


class _FakeTrainer:
    def __init__(self, step):
        self.state = _FakeState(step)
        self.optimizer = _FakeOpt()
        self.scheduler = _FakeOpt({"last_epoch": step})

    def save_model(self, path):
        _write_adapter(Path(path))


def _write_adapter(checkpoint: Path) -> None:
    """A model file the gate accepts: data.pkl + one non-empty tensor record."""
    with zipfile.ZipFile(checkpoint / "adapter_model.bin", "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps({"weight": "x"}))
        archive.writestr("archive/data/0", b"0123456789")


def _write_full_bundle(checkpoint: Path, trainer) -> None:
    _write_adapter(checkpoint)
    _FakeTorch.save(trainer.optimizer.state_dict(), checkpoint / "optimizer.pt")
    _FakeTorch.save(trainer.scheduler.state_dict(), checkpoint / "scheduler.pt")
    trainer.state.save_to_json(str(checkpoint / "trainer_state.json"))


def test_complete_bundle_is_untouched_and_ok(tmp_path):
    trainer = _FakeTrainer(7)
    ckpt = tmp_path / "checkpoint-7"
    ckpt.mkdir(parents=True)
    _write_full_bundle(ckpt, trainer)

    assert resume.ensure_stop_checkpoint_bundle(trainer, tmp_path) is None
    assert resume.is_resume_checkpoint_valid(ckpt) is True


def test_missing_trainer_state_is_backfilled(tmp_path):
    trainer = _FakeTrainer(7)
    ckpt = tmp_path / "checkpoint-7"
    ckpt.mkdir(parents=True)
    _write_adapter(ckpt)

    assert resume.ensure_stop_checkpoint_bundle(trainer, tmp_path) is None
    assert (ckpt / "optimizer.pt").is_file()
    assert (ckpt / "scheduler.pt").is_file()
    assert (ckpt / "trainer_state.json").is_file()
    assert resume.is_resume_checkpoint_valid(ckpt) is True


def test_missing_weights_are_saved_by_trainer(tmp_path):
    trainer = _FakeTrainer(7)
    ckpt = tmp_path / "checkpoint-7"
    ckpt.mkdir(parents=True)

    assert resume.ensure_stop_checkpoint_bundle(trainer, tmp_path) is None
    assert resume.is_resume_checkpoint_valid(ckpt) is True


def test_failed_backfill_returns_reason_instead_of_raising(tmp_path):
    class _BadOpt:
        def state_dict(self):
            raise RuntimeError("nope")

    trainer = _FakeTrainer(7)
    trainer.optimizer = _BadOpt()
    ckpt = tmp_path / "checkpoint-7"
    ckpt.mkdir(parents=True)
    _write_adapter(ckpt)

    reason = resume.ensure_stop_checkpoint_bundle(trainer, tmp_path)
    assert isinstance(reason, str) and "nope" in reason
    assert resume.is_resume_checkpoint_valid(ckpt) is False
