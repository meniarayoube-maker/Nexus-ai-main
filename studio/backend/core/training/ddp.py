# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in single-node Distributed Data Parallel (DDP) for CUDA text training.

The default multi-GPU path shards the model across cards in ONE process
(``device_map="unsloth_balanced"``), so layers execute sequentially and one
card idles while the other computes.  DDP instead replicates the model, one
process per GPU, with NCCL gradient sync: both cards compute simultaneously.

This module holds ONLY dependency-light helpers (stdlib + loggers; torch is
read out of ``sys.modules``, never imported) so unit tests can stub it and
CPU-only hosts never pay for CUDA imports.  The spawn/supervise machinery
lives in ``core.training.worker``; rank gating consults these helpers.

Rank identity is env-based: the rank entry sets ``STUDIO_DDP_RANK`` /
``STUDIO_DDP_WORLD`` (plus the standard ``RANK``/``WORLD_SIZE``/
``LOCAL_RANK`` for transformers/Unsloth readers).  Absence of
``STUDIO_DDP_RANK`` means single-process: every helper below degrades to a
no-op and default behavior is byte-identical with the flag off.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional, Tuple

from loggers import get_logger

logger = get_logger(__name__)

RANK_ENV = "STUDIO_DDP_RANK"
WORLD_ENV = "STUDIO_DDP_WORLD"


def current_rank() -> Optional[int]:
    """This process's DDP rank, or None outside a DDP spawn."""
    raw = os.environ.get(RANK_ENV)
    if raw is None or not str(raw).strip():
        return None
    try:
        rank = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return rank if rank >= 0 else None


def world_size() -> int:
    """DDP world size, or 1 outside a DDP spawn."""
    raw = os.environ.get(WORLD_ENV)
    try:
        size = int(str(raw).strip()) if raw is not None else 1
    except (TypeError, ValueError):
        return 1
    return size if size >= 1 else 1


def is_main_process() -> bool:
    """True on rank 0 and in every single-process run."""
    rank = current_rank()
    return rank is None or rank == 0


def distributed_active() -> bool:
    """True inside a DDP-spawned rank (any rank)."""
    return current_rank() is not None


def ddp_requested(config: dict) -> bool:
    """Whether the run asked for DDP in the UI/API."""
    return bool((config or {}).get("distributed_ddp", False))


def should_use_ddp(
    config: dict,
    *,
    path_supported: bool = True,
    cuda_count: Optional[int] = None,
    dist_initialized: Optional[bool] = None,
) -> "Tuple[bool, str]":
    """Decide whether to launch the DDP spawn for this run.

    ``path_supported`` is False for trainers this version does not distribute
    (MLX, embedding); ``cuda_count``/``dist_initialized`` are injectable for
    tests (resolved from torch when None).  Returns ``(use, reason)`` where
    ``reason`` is a short human explanation used in the fallback warning.
    """
    if not ddp_requested(config):
        return False, "DDP not requested"
    if not path_supported:
        return False, "this trainer runs single-process in DDP v1"
    if dist_initialized is None or cuda_count is None:
        torch_module = sys.modules.get("torch")
        if torch_module is None:
            try:
                import torch as torch_module  # noqa: PLC0415
            except Exception:
                return False, "torch is not installed"
        if dist_initialized is None:
            try:
                distributed = getattr(torch_module, "distributed", None)
                dist_initialized = bool(
                    distributed is not None
                    and distributed.is_available()
                    and distributed.is_initialized()
                )
            except Exception:
                dist_initialized = False
        if cuda_count is None:
            try:
                cuda_count = int(torch_module.cuda.device_count())
            except Exception:
                cuda_count = 0
    if dist_initialized:
        return False, "already running distributed"
    if (cuda_count or 0) < 2:
        return False, f"only {cuda_count or 0} visible GPU(s); DDP needs 2+"
    return True, f"{cuda_count} GPUs"


def ddp_device_map(rank: int) -> dict:
    """Rank-local single-device map: DDP forbids sharded ``balanced`` maps.

    Each rank loads the full model onto its own card; NCCL (not layer
    pipelining) is the parallelism.
    """
    return {"": f"cuda:{int(rank)}"}


def set_rank_env(rank: int, world: int) -> None:
    """Publish rank identity for this process (helpers + transformers/Unsloth)."""
    os.environ[RANK_ENV] = str(int(rank))
    os.environ[WORLD_ENV] = str(int(world))
    os.environ.setdefault("RANK", str(int(rank)))
    os.environ.setdefault("WORLD_SIZE", str(int(world)))
    os.environ.setdefault("LOCAL_RANK", str(int(rank)))


def ensure_dist_env() -> "Tuple[str, str]":
    """Ensure the env rendezvous variables for the NCCL group.

    ``torch.multiprocessing.spawn`` (unlike torchrun) does NOT set
    ``MASTER_ADDR``/``MASTER_PORT``: without them ``init_process_group``
    fails with "environment variable MASTER_ADDR expected, but not set".
    Called once in the spawn supervisor so every rank inherits the same
    endpoint.  Pre-set values are respected (multi-node / custom fabrics).
    Returns ``(addr, port)``.
    """
    addr = os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    port = os.environ.get("MASTER_PORT")
    if not port:
        import socket as _socket

        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = str(sock.getsockname()[1])
        os.environ["MASTER_PORT"] = port
    return addr, port


def effective_batch_str(per_device: Any, accum: Any, world: int) -> str:
    """Human-readable global batch line for the startup log."""
    try:
        global_batch = int(per_device) * int(accum) * int(world)
    except (TypeError, ValueError):
        return f"per_device={per_device} x accum={accum} x {world} GPUs"
    return (
        f"DDP x{world}: global batch = {per_device} (per device) x {accum} "
        f"(accum) x {world} (GPUs) = {global_batch}"
    )


class NullEvents:
    """Event sink that discards everything (non-main DDP ranks).

    Only ``put`` is needed: ranks never read the event queue (the parent
    drains it).  Stop signaling keeps using the real ``stop_queue``.
    """

    def put(self, *args: Any, **kwargs: Any) -> None:
        return None
