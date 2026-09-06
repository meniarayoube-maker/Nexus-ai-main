# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the huggingface branch of ``resolve_storage_target_write_dir``.

Regression coverage: an explicit absolute output dir under an ACTIVE cloud
root must be honored as-is for the huggingface target (a restored run keeps
training where its checkpoint lives), while alien absolute paths must still
be rejected by strict containment.
"""

import importlib.util
import os
import sys
import types
from pathlib import Path


def _load_module(monkeypatch):
    # Load storage_targets with light stdlib shims so the suite runs without
    # the full backend dependency stack (same pattern as the kaggle tests).
    loggers = types.ModuleType("loggers")

    class _Logger:
        def __getattr__(self, name):
            def _method(*args, **kwargs):
                return None

            return _method

    loggers.get_logger = lambda name=None: _Logger()
    monkeypatch.setitem(sys.modules, "loggers", loggers)

    structlog = types.ModuleType("structlog")
    structlog.__getattr__ = lambda name: (lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "structlog", structlog)

    for name in [
        "utils",
        "utils.paths",
        "utils.paths.path_utils",
        "utils.paths.storage_roots",
    ]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    real_roots = importlib.util.spec_from_file_location(
        "storage_roots_under_test",
        Path(__file__).resolve().parents[1]
        / "utils"
        / "paths"
        / "storage_roots.py",
    )
    roots_module = importlib.util.module_from_spec(real_roots)
    assert real_roots.loader is not None
    real_roots.loader.exec_module(roots_module)

    pkg = types.ModuleType("utils")
    paths_pkg = types.ModuleType("utils.paths")
    path_utils = types.ModuleType("utils.paths.path_utils")
    path_utils.host_normalize_path = lambda p: p
    roots_shim = types.ModuleType("utils.paths.storage_roots")
    for _name in dir(roots_module):
        if not _name.startswith("__"):
            setattr(roots_shim, _name, getattr(roots_module, _name))
    pkg.paths = paths_pkg
    paths_pkg.path_utils = path_utils
    paths_pkg.storage_roots = roots_shim
    monkeypatch.setitem(sys.modules, "utils", pkg)
    monkeypatch.setitem(sys.modules, "utils.paths", paths_pkg)
    monkeypatch.setitem(sys.modules, "utils.paths.path_utils", path_utils)
    monkeypatch.setitem(sys.modules, "utils.paths.storage_roots", roots_shim)

    spec = importlib.util.spec_from_file_location(
        "storage_targets_under_test",
        Path(__file__).resolve().parents[1]
        / "utils"
        / "paths"
        / "storage_targets.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _with_fake_kaggle_root(module, monkeypatch, kaggle_root):
    real_root = Path(os.path.realpath(kaggle_root))
    monkeypatch.setattr(
        module,
        "storage_target_override_root",
        lambda target: real_root if target == "kaggle" else None,
    )


def test_hf_honors_explicit_path_under_active_kaggle_root(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch)
    kaggle_root = tmp_path / "kaggle"
    wanted = kaggle_root / "unsloth-outputs" / "my-run"
    wanted.mkdir(parents=True)
    _with_fake_kaggle_root(mod, monkeypatch, kaggle_root)

    target, resolved = mod.resolve_storage_target_write_dir(
        "huggingface", str(wanted), "my-run"
    )

    assert target == "huggingface"
    assert Path(resolved) == wanted


def test_hf_rejects_alien_absolute_path(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch)
    _with_fake_kaggle_root(mod, monkeypatch, tmp_path / "kaggle")

    try:
        mod.resolve_storage_target_write_dir(
            "huggingface", str(tmp_path / "elsewhere" / "x"), "x"
        )
    except ValueError:
        return
    raise AssertionError("alien absolute path must not resolve for huggingface")


def test_hf_unmounted_cloud_path_never_escapes(monkeypatch, tmp_path):
    # Kaggle not mounted: a cloud-looking absolute path must never resolve
    # (no fallback that escapes the root, no silent outside write) -- the
    # containment guard fires loudly, on every platform.
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "storage_target_override_root", lambda target: None)
    monkeypatch.setattr(mod, "outputs_root", lambda: tmp_path / "outputs")

    try:
        mod.resolve_storage_target_write_dir(
            "huggingface", "/kaggle/working/unsloth-outputs/run", "run"
        )
    except ValueError:
        return
    raise AssertionError("unmounted cloud path must not resolve for huggingface")


def test_hf_auto_names_under_local_root_without_explicit_dir(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "storage_target_override_root", lambda target: None)
    monkeypatch.setattr(mod, "outputs_root", lambda: tmp_path / "outputs")

    target, resolved = mod.resolve_storage_target_write_dir(
        "huggingface", None, "my run!"
    )

    assert target == "huggingface"
    assert Path(resolved).parent == tmp_path / "outputs"
    assert Path(resolved).is_dir()
