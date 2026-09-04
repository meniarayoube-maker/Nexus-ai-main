# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Stub-client tests for :mod:`utils.paths.kaggle_push`.

The real ``kaggle`` package is never installed in CI, so a fake
``kaggle.api.kaggle_api_extended`` module is injected into ``sys.modules``.
Two client shapes are covered: the documented real-style ``folder``-based
signatures, and a legacy ``folder_path``-based shape (the one that caused the
``unexpected keyword argument 'folder_path'`` production failure) to prove the
version-tolerant caller adapts instead of crashing.
"""

import json
import sys
import types

import pytest

from utils.paths.kaggle_push import push_output_to_kaggle


def _install_fake_kaggle(monkeypatch, api_cls):
    pkg = types.ModuleType("kaggle")
    api_pkg = types.ModuleType("kaggle.api")
    ext = types.ModuleType("kaggle.api.kaggle_api_extended")
    ext.KaggleApi = api_cls
    pkg.api = api_pkg
    api_pkg.kaggle_api_extended = ext
    monkeypatch.setitem(sys.modules, "kaggle", pkg)
    monkeypatch.setitem(sys.modules, "kaggle.api", api_pkg)
    monkeypatch.setitem(sys.modules, "kaggle.api.kaggle_api_extended", ext)
    return api_cls


class RealStyleApi:
    """Mirrors the documented kaggle-api signatures (folder-based)."""

    created = []

    def __init__(self):
        self.calls = []
        RealStyleApi.created.append(self)

    def authenticate(self):
        self.calls.append(("authenticate",))

    def dataset_create_new(self, folder, public=False, quiet=True, convert_to_csv=True, dir_mode="zip"):
        self.calls.append(("create_new", folder, public, quiet))

    def dataset_create_version(
        self, folder, version_notes="", quiet=True, convert_to_csv=True,
        delete_old_versions=False, dir_mode="zip",
    ):
        self.calls.append(("create_version", folder, version_notes))


class LegacyFolderPathApi(RealStyleApi):
    """Alternate build using ``folder_path``-style parameter names."""

    def dataset_create_new(self, folder_path, title=None, slug=None, is_private=True):
        self.calls.append(("create_new", folder_path, title, slug, is_private))

    def dataset_create_version(self, folder_path, version_notes="", force=False):
        self.calls.append(("create_version", folder_path, version_notes, force))


def _make_run_dir(tmp_path, name="my_run_123"):
    run_dir = tmp_path / name
    run_dir.mkdir()
    (run_dir / "adapter_model.safetensors").write_bytes(b"fake-weights")
    return run_dir


def test_create_new_success_writes_full_metadata_and_url(monkeypatch, tmp_path):
    RealStyleApi.created.clear()
    _install_fake_kaggle(monkeypatch, RealStyleApi)
    monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
    monkeypatch.setenv("KAGGLE_KEY", "testkey")
    run_dir = _make_run_dir(tmp_path)

    ok, url, error = push_output_to_kaggle(str(run_dir))

    assert error is None
    assert ok is True
    assert url == "https://www.kaggle.com/datasets/testuser/my-run-123"
    metadata = json.loads((run_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"] == "testuser/my-run-123"
    assert metadata["isPrivate"] is True
    assert metadata["licenses"] == [{"name": "cc-by-sa-4.0"}]
    api = RealStyleApi.created[-1]
    assert ("authenticate",) in api.calls
    assert ("create_new", str(run_dir), False, True) in api.calls


def test_public_flag_flows_to_api_and_metadata(monkeypatch, tmp_path):
    RealStyleApi.created.clear()
    _install_fake_kaggle(monkeypatch, RealStyleApi)
    monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
    monkeypatch.setenv("KAGGLE_KEY", "testkey")
    run_dir = _make_run_dir(tmp_path)

    ok, url, error = push_output_to_kaggle(str(run_dir), is_private=False)

    assert (ok, error) == (True, None)
    metadata = json.loads((run_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert metadata["isPrivate"] is False
    api = RealStyleApi.created[-1]
    assert ("create_new", str(run_dir), True, True) in api.calls


def test_already_exists_falls_back_to_new_version(monkeypatch, tmp_path):
    class ExistsApi(RealStyleApi):
        def dataset_create_new(self, folder, public=False, quiet=True, **kwargs):
            raise Exception("That dataset already exists (409 Conflict)")

    ExistsApi.created.clear()
    _install_fake_kaggle(monkeypatch, ExistsApi)
    monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
    monkeypatch.setenv("KAGGLE_KEY", "testkey")
    run_dir = _make_run_dir(tmp_path)

    ok, url, error = push_output_to_kaggle(str(run_dir))

    assert (ok, error) == (True, None)
    assert url == "https://www.kaggle.com/datasets/testuser/my-run-123"
    api = RealStyleApi.created[-1]
    kinds = [c[0] for c in api.calls]
    assert "create_version" in kinds


def test_genuine_create_failure_returns_precise_reason(monkeypatch, tmp_path):
    class BoomApi(RealStyleApi):
        def dataset_create_new(self, folder, public=False, quiet=True, **kwargs):
            raise RuntimeError("boom 500 internal")

    RealStyleApi.created.clear()
    _install_fake_kaggle(monkeypatch, BoomApi)
    monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
    monkeypatch.setenv("KAGGLE_KEY", "testkey")
    run_dir = _make_run_dir(tmp_path)

    ok, url, error = push_output_to_kaggle(str(run_dir))

    assert ok is False
    assert url is None
    assert error is not None and "boom 500" in error


def test_legacy_folder_path_signature_is_tolerated(monkeypatch, tmp_path):
    LegacyFolderPathApi.created.clear()
    _install_fake_kaggle(monkeypatch, LegacyFolderPathApi)
    monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
    monkeypatch.setenv("KAGGLE_KEY", "testkey")
    run_dir = _make_run_dir(tmp_path)

    ok, url, error = push_output_to_kaggle(str(run_dir))

    assert (ok, error) == (True, None)
    assert url == "https://www.kaggle.com/datasets/testuser/my-run-123"
    api = LegacyFolderPathApi.created[-1]
    assert api.calls[1][0] == "create_new"
    assert api.calls[1][1] == str(run_dir)


def test_missing_output_dir_is_skipped(monkeypatch, tmp_path):
    _install_fake_kaggle(monkeypatch, RealStyleApi)
    ok, url, error = push_output_to_kaggle(str(tmp_path / "does-not-exist"))
    assert (ok, url, error) == (False, None, None)


def test_owner_falls_back_to_client_username_attribute(monkeypatch, tmp_path):
    class AttrApi(RealStyleApi):
        username = "attruser"

    RealStyleApi.created.clear()
    _install_fake_kaggle(monkeypatch, AttrApi)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    run_dir = _make_run_dir(tmp_path)

    ok, url, error = push_output_to_kaggle(str(run_dir))

    assert (ok, error) == (True, None)
    assert url == "https://www.kaggle.com/datasets/attruser/my-run-123"
    metadata = json.loads((run_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"] == "attruser/my-run-123"


def test_no_owner_anywhere_still_reports_success_without_url(monkeypatch, tmp_path):
    RealStyleApi.created.clear()
    _install_fake_kaggle(monkeypatch, RealStyleApi)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    run_dir = _make_run_dir(tmp_path)

    ok, url, error = push_output_to_kaggle(str(run_dir))

    assert (ok, url, error) == (True, None, None)
