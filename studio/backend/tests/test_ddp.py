# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Stub-level tests for the DDP helpers (``core.training.ddp``).

No torch/CUDA needed: rank identity is env-based and torch is read out of
``sys.modules`` (or stubbed there), mirroring the production access pattern.
"""

import os
import sys
import types

import pytest

from core.training.ddp import (
    NullEvents,
    current_rank,
    ddp_device_map,
    ddp_requested,
    distributed_active,
    effective_batch_str,
    is_main_process,
    set_rank_env,
    should_use_ddp,
    world_size,
)


@pytest.fixture(autouse=True)
def _clean_ddp_env(monkeypatch):
    for key in (
        "STUDIO_DDP_RANK",
        "STUDIO_DDP_WORLD",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
    ):
        monkeypatch.delenv(key, raising=False)


def test_no_rank_env_means_single_process():
    assert current_rank() is None
    assert world_size() == 1
    assert is_main_process() is True
    assert distributed_active() is False


def test_rank_env_parsing(monkeypatch):
    monkeypatch.setenv("STUDIO_DDP_RANK", "1")
    monkeypatch.setenv("STUDIO_DDP_WORLD", "2")
    assert current_rank() == 1
    assert world_size() == 2
    assert is_main_process() is False
    assert distributed_active() is True


def test_rank_zero_is_main(monkeypatch):
    monkeypatch.setenv("STUDIO_DDP_RANK", "0")
    assert is_main_process() is True
    assert distributed_active() is True


def test_malformed_rank_env_is_single_process(monkeypatch):
    monkeypatch.setenv("STUDIO_DDP_RANK", "banana")
    assert current_rank() is None
    assert is_main_process() is True


def test_set_rank_env_publishes_standard_vars(monkeypatch):
    set_rank_env(1, 2)
    assert os.environ["STUDIO_DDP_RANK"] == "1"
    assert os.environ["STUDIO_DDP_WORLD"] == "2"
    assert os.environ["LOCAL_RANK"] == "1"
    assert current_rank() == 1


def test_ddp_requested_flag():
    assert ddp_requested({}) is False
    assert ddp_requested({"distributed_ddp": False}) is False
    assert ddp_requested({"distributed_ddp": True}) is True


def test_should_use_ddp_matrix():
    cfg = {"distributed_ddp": True}
    assert should_use_ddp({}) == (False, "DDP not requested")
    ok, reason = should_use_ddp(cfg, path_supported=False, cuda_count=2)
    assert ok is False and "single-process" in reason
    ok, reason = should_use_ddp(cfg, cuda_count=1, dist_initialized=False)
    assert ok is False and "1 visible" in reason
    ok, reason = should_use_ddp(cfg, cuda_count=2, dist_initialized=True)
    assert ok is False and "already" in reason
    ok, reason = should_use_ddp(cfg, cuda_count=2, dist_initialized=False)
    assert ok is True and "2 GPUs" in reason

def test_should_use_ddp_without_torch(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    # Block the lazy import fallback: torch truly absent on this host.
    import builtins

    real_import = builtins.__import__

    def _no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_torch)
    ok, reason = should_use_ddp({"distributed_ddp": True}, cuda_count=None)
    assert ok is False and "torch" in reason.lower()


def test_should_use_ddp_reads_stub_cuda(monkeypatch):
    torch_stub = types.ModuleType("torch")
    cuda = types.SimpleNamespace(device_count=lambda: 2)
    distributed = types.SimpleNamespace(
        is_available=lambda: True, is_initialized=lambda: False
    )
    torch_stub.cuda = cuda
    torch_stub.distributed = distributed
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    ok, reason = should_use_ddp({"distributed_ddp": True})
    assert ok is True and "2 GPUs" in reason


def test_ddp_device_map_is_rank_local():
    assert ddp_device_map(0) == {"": "cuda:0"}
    assert ddp_device_map(1) == {"": "cuda:1"}


def test_effective_batch_str():
    assert effective_batch_str(2, 4, 2) == (
        "DDP x2: global batch = 2 (per device) x 4 (accum) x 2 (GPUs) = 16"
    )
    assert "GPUs" in effective_batch_str("x", None, 2)


def test_null_events_discards():
    sink = NullEvents()
    assert sink.put({"type": "status", "message": "x"}) is None
