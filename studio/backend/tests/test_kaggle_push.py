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
from pathlib import Path

import pytest

from utils.paths.kaggle_push import (
    _checkpoint_dir_step,
    _validate_dataset_slug,
    download_output_from_kaggle,
    push_output_to_kaggle,
    upload_source_bytes,
)


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

    def dataset_create_new(self, folder, public=False, quiet=True, convert_to_csv=True, dir_mode="skip"):
        self.calls.append(("create_new", folder, public, quiet, dir_mode))

    def dataset_create_version(
        self, folder, version_notes="", quiet=True, convert_to_csv=True,
        delete_old_versions=False, dir_mode="skip",
    ):
        self.calls.append(("create_version", folder, version_notes, dir_mode))


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
    # Directories must ride along: dir_mode='zip', never the client's 'skip' default.
    assert ("create_new", str(run_dir), False, True, "zip") in api.calls


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
    assert ("create_new", str(run_dir), True, True, "zip") in api.calls


def test_version_upload_zips_directories(monkeypatch, tmp_path):
    class ExistsApi(RealStyleApi):
        def dataset_create_new(self, folder, public=False, quiet=True, **kwargs):
            raise Exception("That dataset already exists (409 Conflict)")

    RealStyleApi.created.clear()
    _install_fake_kaggle(monkeypatch, ExistsApi)
    monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
    monkeypatch.setenv("KAGGLE_KEY", "testkey")
    run_dir = _make_run_dir(tmp_path)

    ok, url, error = push_output_to_kaggle(str(run_dir))

    assert (ok, error) == (True, None)
    api = RealStyleApi.created[-1]
    version_calls = [c for c in api.calls if c[0] == "create_version"]
    assert version_calls and version_calls[0][3] == "zip"


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


def test_no_owner_anywhere_fails_fast_with_precise_reason(monkeypatch, tmp_path):
    RealStyleApi.created.clear()
    _install_fake_kaggle(monkeypatch, RealStyleApi)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    run_dir = _make_run_dir(tmp_path)

    ok, url, error = push_output_to_kaggle(str(run_dir))

    assert ok is False
    assert url is None
    assert error is not None and "username" in error.lower()
    # Refused before touching the API (no ownerless metadata id).
    api = RealStyleApi.created[-1]
    assert all(call[0] == "authenticate" for call in api.calls)


def test_username_without_key_keeps_owner(monkeypatch, tmp_path):
    # Regression: a known username must survive even when the key comes from
    # elsewhere (restored rows sanitize the key away). The old code nulled
    # both halves, producing an ownerless id that crashed the client.
    RealStyleApi.created.clear()
    _install_fake_kaggle(monkeypatch, RealStyleApi)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    run_dir = _make_run_dir(tmp_path)

    ok, url, error = push_output_to_kaggle(str(run_dir), username="loneuser")

    assert (ok, error) == (True, None)
    assert url == "https://www.kaggle.com/datasets/loneuser/my-run-123"
    metadata = json.loads((run_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"] == "loneuser/my-run-123"


# ---------------------------------------------------------------------------
# download_output_from_kaggle
# ---------------------------------------------------------------------------


class DownloadApi:
    """Real-style ``dataset_download`` signature."""

    created = []

    def __init__(self):
        self.calls = []
        DownloadApi.created.append(self)

    def authenticate(self):
        self.calls.append(("authenticate",))

    def dataset_download(self, dataset, path=None, force=False, quiet=True, unzip=False):
        self.calls.append(("download", dataset, path, force, quiet, unzip))
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "adapter_model.safetensors").write_bytes(b"fake-weights")


class LegacyDownloadApi(DownloadApi):
    """Alternate build with different parameter names."""

    def dataset_download(self, dataset_slug, download_dir=None, unzip_archive=False):
        self.calls.append(("download", dataset_slug, download_dir, unzip_archive))
        target = Path(download_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "adapter_model.safetensors").write_bytes(b"fake-weights")


class FilesDownloadApi:
    """Current kaggle-api shape: whole-dataset download is ``dataset_download_files``."""

    created = []

    def __init__(self):
        self.calls = []
        FilesDownloadApi.created.append(self)

    def authenticate(self):
        self.calls.append(("authenticate",))

    def dataset_download_files(self, dataset, path=None, force=False, quiet=True, unzip=False):
        self.calls.append(("download_files", dataset, path, force, quiet, unzip))
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "adapter_model.safetensors").write_bytes(b"fake-weights")


class NoDownloadApi:
    """Client build with no dataset download method at all."""

    created = []

    def __init__(self):
        self.calls = []
        NoDownloadApi.created.append(self)

    def authenticate(self):
        self.calls.append(("authenticate",))


def test_validate_dataset_slug():
    assert _validate_dataset_slug(" owner/slug ") == "owner/slug"
    assert _validate_dataset_slug("owner/slug/") == "owner/slug"
    assert _validate_dataset_slug("owner/slug?select=checkpoint-5") == "owner/slug"
    assert (
        _validate_dataset_slug("https://www.kaggle.com/datasets/owner/slug?select=x")
        == "owner/slug"
    )
    assert _validate_dataset_slug("not-a-slug") is None
    assert _validate_dataset_slug("a/b/c") is None
    assert _validate_dataset_slug("") is None
    assert _validate_dataset_slug(None) is None


def test_download_success(monkeypatch, tmp_path):
    import tempfile as _tempfile

    DownloadApi.created.clear()
    _install_fake_kaggle(monkeypatch, DownloadApi)
    monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
    monkeypatch.setenv("KAGGLE_KEY", "testkey")
    dest = tmp_path / "restored_run"

    ok, path, error = download_output_from_kaggle("owner/my-data", str(dest))

    assert (ok, error) == (True, None)
    assert path == str(dest)
    assert (dest / "adapter_model.safetensors").is_file()
    api = DownloadApi.created[-1]
    # Downloaded into system-temp staging first, then moved into place.
    assert api.calls[1][0] == "download"
    assert api.calls[1][1] == "owner/my-data"
    assert Path(api.calls[1][2]).parent.parent == Path(_tempfile.gettempdir())
    assert Path(api.calls[1][2]).name == "payload"


def test_download_legacy_signature_is_tolerated(monkeypatch, tmp_path):
    import tempfile as _tempfile

    LegacyDownloadApi.created.clear()
    _install_fake_kaggle(monkeypatch, LegacyDownloadApi)
    monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
    monkeypatch.setenv("KAGGLE_KEY", "testkey")
    dest = tmp_path / "restored_run"

    ok, path, error = download_output_from_kaggle("owner/my-data", str(dest))

    assert (ok, error) == (True, None)
    assert path == str(dest)
    assert (dest / "adapter_model.safetensors").is_file()
    api = LegacyDownloadApi.created[-1]
    assert api.calls[1][0] == "download"
    assert api.calls[1][1] == "owner/my-data"
    assert Path(api.calls[1][2]).parent.parent == Path(_tempfile.gettempdir())


def test_download_bad_slug_never_touches_api(monkeypatch, tmp_path):
    DownloadApi.created.clear()
    _install_fake_kaggle(monkeypatch, DownloadApi)

    ok, path, error = download_output_from_kaggle("not-a-slug", str(tmp_path / "x"))

    assert ok is False
    assert path is None
    assert error is not None and "owner/slug" in error
    assert DownloadApi.created == []


def test_download_stages_in_system_temp_and_cleans_up(monkeypatch, tmp_path):
    import tempfile as _tempfile

    DownloadApi.created.clear()
    _install_fake_kaggle(monkeypatch, DownloadApi)
    monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
    monkeypatch.setenv("KAGGLE_KEY", "testkey")
    dest = tmp_path / "restored_run"
    before = {p.name for p in Path(_tempfile.gettempdir()).iterdir() if p.name.startswith(".kaggle-restore-")}

    ok, path, error = download_output_from_kaggle("owner/my-data", str(dest))

    assert (ok, error) == (True, None)
    assert path == str(dest)
    assert (dest / "adapter_model.safetensors").is_file()
    after = {p.name for p in Path(_tempfile.gettempdir()).iterdir() if p.name.startswith(".kaggle-restore-")}
    assert after == before


def test_failed_download_leaves_no_partials_behind(monkeypatch, tmp_path):
    class PartialDownloadApi(DownloadApi):
        def dataset_download(self, dataset, path=None, **kwargs):
            target = Path(path)
            target.mkdir(parents=True, exist_ok=True)
            (target / "partial.bin").write_bytes(b"incomplete")
            raise RuntimeError("boom mid-download")

    DownloadApi.created.clear()
    _install_fake_kaggle(monkeypatch, PartialDownloadApi)
    monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
    monkeypatch.setenv("KAGGLE_KEY", "testkey")
    dest = tmp_path / "restored_run"

    ok, path, error = download_output_from_kaggle("owner/my-data", str(dest))

    assert ok is False
    assert path is None
    assert error is not None and "boom" in error
    # No partial files linger to trip the "already restored" guard next time.
    assert dest.is_dir()
    assert list(dest.iterdir()) == []


def test_download_api_failure_returns_reason(monkeypatch, tmp_path):
    class BoomDownloadApi(DownloadApi):
        def dataset_download(self, dataset, path=None, **kwargs):
            raise RuntimeError("404 not found")

    DownloadApi.created.clear()
    _install_fake_kaggle(monkeypatch, BoomDownloadApi)
    monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
    monkeypatch.setenv("KAGGLE_KEY", "testkey")

    ok, path, error = download_output_from_kaggle("owner/missing", str(tmp_path / "x"))

    assert ok is False
    assert path is None
    assert error is not None and "not found" in error.lower()


def test_download_prefers_download_files_method(monkeypatch, tmp_path):
    import tempfile as _tempfile

    FilesDownloadApi.created.clear()
    _install_fake_kaggle(monkeypatch, FilesDownloadApi)
    monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
    monkeypatch.setenv("KAGGLE_KEY", "testkey")
    dest = tmp_path / "restored_run"

    ok, path, error = download_output_from_kaggle("owner/my-data", str(dest))

    assert (ok, error) == (True, None)
    assert path == str(dest)
    assert (dest / "adapter_model.safetensors").is_file()
    api = FilesDownloadApi.created[-1]
    assert api.calls[1][0] == "download_files"
    assert api.calls[1][1] == "owner/my-data"
    assert Path(api.calls[1][2]).parent.parent == Path(_tempfile.gettempdir())


def test_download_without_any_method_is_precise(monkeypatch, tmp_path):
    NoDownloadApi.created.clear()
    _install_fake_kaggle(monkeypatch, NoDownloadApi)
    monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
    monkeypatch.setenv("KAGGLE_KEY", "testkey")

    ok, path, error = download_output_from_kaggle("owner/my-data", str(tmp_path / "x"))

    assert ok is False
    assert path is None
    assert error is not None and "no" in error.lower() and "download" in error.lower()


def test_upload_source_bytes_counts_recursively(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    ckpt = run_dir / "checkpoint-5"
    ckpt.mkdir()
    (ckpt / "optimizer.pt").write_bytes(b"0" * 100)
    # 12-byte adapter file + 100-byte optimizer file.
    assert upload_source_bytes(run_dir) == 112
    assert upload_source_bytes(tmp_path / "missing") == 0


def test_upload_refused_when_disk_too_full(monkeypatch, tmp_path):
    import shutil as _shutil

    RealStyleApi.created.clear()
    _install_fake_kaggle(monkeypatch, RealStyleApi)
    monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
    monkeypatch.setenv("KAGGLE_KEY", "testkey")
    run_dir = _make_run_dir(tmp_path)

    class _Usage:
        total = 10**12
        used = 10**12 - 1
        free = 1

    monkeypatch.setattr(_shutil, "disk_usage", lambda _path: _Usage())

    ok, url, error = push_output_to_kaggle(str(run_dir))

    assert ok is False
    assert url is None
    assert error is not None and "free disk" in error.lower()
    # Refused before touching the API.
    assert RealStyleApi.created == []


def _write_checkpoint_bundle(checkpoint, step, *, valid=True):
    import pickle
    import zipfile

    checkpoint.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(checkpoint / "adapter_model.bin", "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps({"weight": "x"}))
        archive.writestr("archive/data/0", b"0123456789")
    if valid:
        for name in ("optimizer.pt", "scheduler.pt"):
            with zipfile.ZipFile(checkpoint / name, "w") as archive:
                archive.writestr("archive/data.pkl", pickle.dumps({}))
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": step}), encoding="utf-8"
        )


def test_staging_uploads_latest_checkpoint_only(monkeypatch, tmp_path):
    class StagingApi(RealStyleApi):
        snapshots = []

        def dataset_create_new(self, folder, public=False, quiet=True, convert_to_csv=True, dir_mode="zip"):
            self.calls.append(("create_new", folder, public, quiet, dir_mode))
            root = Path(folder)
            StagingApi.snapshots.append(
                sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
            )

    RealStyleApi.created.clear()
    StagingApi.snapshots.clear()
    _install_fake_kaggle(monkeypatch, StagingApi)
    monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
    monkeypatch.setenv("KAGGLE_KEY", "testkey")
    run_dir = _make_run_dir(tmp_path)
    _write_checkpoint_bundle(run_dir / "checkpoint-5", 5)
    _write_checkpoint_bundle(run_dir / "checkpoint-9", 9)

    ok, url, error = push_output_to_kaggle(str(run_dir))

    assert (ok, error) == (True, None)
    assert url == "https://www.kaggle.com/datasets/testuser/my-run-123"
    api = RealStyleApi.created[-1]
    uploaded = Path(api.calls[1][1])
    # Staged elsewhere (hardlinks), never the live output dir itself.
    assert uploaded != run_dir
    assert uploaded.name.startswith(".upload-stage-")
    snapshot = StagingApi.snapshots[-1]
    assert "adapter_model.safetensors" in snapshot
    assert "dataset-metadata.json" in snapshot
    assert "checkpoint-9/trainer_state.json" in snapshot
    assert not any(entry.startswith("checkpoint-5/") for entry in snapshot)
    # Staging is cleaned afterwards (links only; sources untouched).
    leftovers = [p for p in run_dir.parent.iterdir() if p.name.startswith(".upload-stage-")]
    assert leftovers == []
    assert (run_dir / "checkpoint-5" / "trainer_state.json").is_file()


def test_flat_output_dir_uploads_root_directly(monkeypatch, tmp_path):
    RealStyleApi.created.clear()
    _install_fake_kaggle(monkeypatch, RealStyleApi)
    monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
    monkeypatch.setenv("KAGGLE_KEY", "testkey")
    run_dir = _make_run_dir(tmp_path)

    ok, _url, _error = push_output_to_kaggle(str(run_dir))

    assert ok is True
    api = RealStyleApi.created[-1]
    assert api.calls[1][1] == str(run_dir)


class ListableApi(RealStyleApi):
    """Fake client exposing dataset_list (search by slug, ref-shaped rows)."""

    listed_with = []
    existing_refs = []

    def dataset_list(self, search=None):
        ListableApi.listed_with.append(search)
        return [{"ref": ref} for ref in ListableApi.existing_refs]

    def dataset_create_new(self, folder, public=False, quiet=True, convert_to_csv=True, dir_mode="skip"):
        raise AssertionError("create must not be called for an existing dataset")


def test_existing_dataset_goes_straight_to_version(monkeypatch, tmp_path):
    # Regression for the silent-no-op create: some client releases accept
    # create on an existing slug as success without creating a version, which
    # the old try/except fallback could not detect.  With an existence probe
    # the version path is taken directly and create is never attempted.
    RealStyleApi.created.clear()
    ListableApi.existing_refs = ["testuser/my-run-123"]
    ListableApi.listed_with.clear()
    _install_fake_kaggle(monkeypatch, ListableApi)
    monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
    monkeypatch.setenv("KAGGLE_KEY", "testkey")
    run_dir = _make_run_dir(tmp_path)

    ok, url, error = push_output_to_kaggle(str(run_dir))

    assert (ok, error) == (True, None)
    assert url == "https://www.kaggle.com/datasets/testuser/my-run-123"
    assert ListableApi.listed_with == ["testuser/my-run-123"]
    api = RealStyleApi.created[-1]
    kinds = [c[0] for c in api.calls]
    assert "create_version" in kinds
    assert "create_new" not in kinds


def test_missing_dataset_goes_to_create(monkeypatch, tmp_path):
    class MissingApi(ListableApi):
        def dataset_create_new(self, folder, public=False, quiet=True, convert_to_csv=True, dir_mode="skip"):
            self.calls.append(("create_new", folder, public, quiet, dir_mode))

    RealStyleApi.created.clear()
    ListableApi.existing_refs = []
    _install_fake_kaggle(monkeypatch, MissingApi)
    monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
    monkeypatch.setenv("KAGGLE_KEY", "testkey")
    run_dir = _make_run_dir(tmp_path)

    ok, url, error = push_output_to_kaggle(str(run_dir))

    assert (ok, error) == (True, None)
    api = RealStyleApi.created[-1]
    kinds = [c[0] for c in api.calls]
    assert "create_new" in kinds
    assert "create_version" not in kinds


def test_list_unsupported_or_failing_falls_back_to_legacy(monkeypatch, tmp_path):
    # No dataset_list at all -> legacy try-create-first (existing behavior).
    RealStyleApi.created.clear()
    _install_fake_kaggle(monkeypatch, RealStyleApi)
    monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
    monkeypatch.setenv("KAGGLE_KEY", "testkey")

    first = tmp_path / "first"
    first.mkdir()
    ok, _url, error = push_output_to_kaggle(str(_make_run_dir(first)))
    assert (ok, error) == (True, None)

    # A raising list is equally inconclusive -> legacy path, still succeeds.
    class FlakyListApi(ListableApi):
        def dataset_list(self, search=None):
            raise RuntimeError("search backend down")

        def dataset_create_new(self, folder, public=False, quiet=True, convert_to_csv=True, dir_mode="skip"):
            self.calls.append(("create_new", folder, public, quiet, dir_mode))

    RealStyleApi.created.clear()
    _install_fake_kaggle(monkeypatch, FlakyListApi)
    second = tmp_path / "second"
    second.mkdir()
    ok, _url, error = push_output_to_kaggle(str(_make_run_dir(second)))
    assert (ok, error) == (True, None)
    api = RealStyleApi.created[-1]
    assert api.calls[1][0] == "create_new"


def test_list_object_shape_with_datasets_attr(monkeypatch, tmp_path):
    class ObjectListApi(ListableApi):
        def dataset_list(self, search=None):
            class _Result:
                datasets = [{"ref": "testuser/my-run-123"}]

            return _Result()

        def dataset_create_new(self, folder, public=False, quiet=True, convert_to_csv=True, dir_mode="skip"):
            raise AssertionError("create must not be called for an existing dataset")

    RealStyleApi.created.clear()
    _install_fake_kaggle(monkeypatch, ObjectListApi)
    monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
    monkeypatch.setenv("KAGGLE_KEY", "testkey")
    run_dir = _make_run_dir(tmp_path)

    ok, url, error = push_output_to_kaggle(str(run_dir))

    assert (ok, error) == (True, None)
    api = RealStyleApi.created[-1]
    assert [c[0] for c in api.calls].count("create_version") == 1


def test_checkpoint_step_sorts_numerically_not_lexicographically(tmp_path):
    # Regression: lexicographic order reads checkpoint-5 as newer than
    # checkpoint-19 ('5' > '1') and once shipped a stale bundle as "latest".
    assert _checkpoint_dir_step(tmp_path / "checkpoint-5") == 5
    assert _checkpoint_dir_step(tmp_path / "checkpoint-19") == 19
    assert _checkpoint_dir_step(tmp_path / "checkpoint-100") == 100
    assert _checkpoint_dir_step(tmp_path / "other") == -1
    assert _checkpoint_dir_step(tmp_path / "checkpoint-abc") == -1


def test_staging_keeps_numerically_latest_checkpoint(monkeypatch, tmp_path):
    class StagingApi(RealStyleApi):
        snapshots = []

        def dataset_create_new(self, folder, public=False, quiet=True, convert_to_csv=True, dir_mode="zip"):
            self.calls.append(("create_new", folder, public, quiet, dir_mode))
            root = Path(folder)
            StagingApi.snapshots.append(
                sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
            )

    RealStyleApi.created.clear()
    StagingApi.snapshots.clear()
    _install_fake_kaggle(monkeypatch, StagingApi)
    monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
    monkeypatch.setenv("KAGGLE_KEY", "testkey")
    run_dir = _make_run_dir(tmp_path)
    _write_checkpoint_bundle(run_dir / "checkpoint-5", 5)
    _write_checkpoint_bundle(run_dir / "checkpoint-19", 19)
    _write_checkpoint_bundle(run_dir / "checkpoint-100", 100)

    ok, _url, error = push_output_to_kaggle(str(run_dir))

    assert (ok, error) == (True, None)
    snapshot = StagingApi.snapshots[-1]
    assert "checkpoint-100/trainer_state.json" in snapshot
    assert not any(entry.startswith("checkpoint-5/") for entry in snapshot)
    assert not any(entry.startswith("checkpoint-19/") for entry in snapshot)
    leftovers = [p for p in run_dir.parent.iterdir() if p.name.startswith(".upload-stage-")]
    assert leftovers == []


def test_disk_gate_measures_system_temp_not_source(monkeypatch, tmp_path):
    import shutil as _shutil
    import tempfile as _tempfile

    RealStyleApi.created.clear()
    _install_fake_kaggle(monkeypatch, RealStyleApi)
    monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
    monkeypatch.setenv("KAGGLE_KEY", "testkey")
    run_dir = _make_run_dir(tmp_path)

    seen = {}

    class _Usage:
        total = 10**12
        used = 0
        free = 10**12

    def _fake_usage(path):
        seen["path"] = str(path)
        return _Usage()

    monkeypatch.setattr(_shutil, "disk_usage", _fake_usage)

    ok, _url, _error = push_output_to_kaggle(str(run_dir))

    assert ok is True
    # The temp archive is built by the client in the system temp dir, so the
    # gate must measure that filesystem -- never the source dir.
    assert Path(seen["path"]) == Path(_tempfile.gettempdir())
