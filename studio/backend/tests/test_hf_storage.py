# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Stub-client tests for Hugging Face storage (push privacy + pull restore).

``huggingface_hub`` is stubbed in ``sys.modules`` so no network or package is
needed.  Mirrors the ``test_kaggle_push.py`` style: real-shaped signatures,
legacy variants, and failure mapping.
"""

import sys
import types
from pathlib import Path

from utils.paths.hf_pull import (
    _select_sparse_patterns,
    _validate_repo_id,
    download_output_from_huggingface,
)
from utils.paths.storage_push import push_output_to_huggingface


def _install_fake_hub(monkeypatch, api_cls, errors_cls=None):
    pkg = types.ModuleType("huggingface_hub")
    errors = types.ModuleType("huggingface_hub.errors")

    class HfHubHTTPError(Exception):
        def __init__(self, message="", response=None):
            super().__init__(message)
            self.response = response

    errors.HfHubHTTPError = errors_cls or HfHubHTTPError
    pkg.errors = errors
    pkg.HfApi = api_cls
    monkeypatch.setitem(sys.modules, "huggingface_hub", pkg)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    return api_cls


def _make_output_dir(tmp_path, name="run_abc"):
    run_dir = tmp_path / name
    run_dir.mkdir()
    (run_dir / "adapter_model.safetensors").write_bytes(b"fake-weights")
    return run_dir


class FakeHfApi:
    """Real-shaped HfApi surface used across tests."""

    created = []

    def __init__(self, token=None):
        self.calls = []
        self.token = token
        FakeHfApi.created.append(self)

    def repo_info(self, repo_id, token=None):
        self.calls.append(("repo_info", repo_id))
        raise AssertionError("repo_info behavior set per test")

    def create_repo(self, repo_id, private=False, exist_ok=True, token=None):
        self.calls.append(("create_repo", repo_id, private, exist_ok))
        return None

    def upload_folder(self, repo_id, folder_path, commit_message=None, token=None):
        self.calls.append(("upload_folder", repo_id, folder_path))


class PrivateExistingApi(FakeHfApi):
    def repo_info(self, repo_id, token=None):
        self.calls.append(("repo_info", repo_id))
        return types.SimpleNamespace(private=True)


class PublicExistingApi(FakeHfApi):
    def repo_info(self, repo_id, token=None):
        self.calls.append(("repo_info", repo_id))
        return types.SimpleNamespace(private=False)


def _http_error(status):
    pkg = sys.modules["huggingface_hub"]

    class _Resp:
        status_code = status

    class _Err(pkg.errors.HfHubHTTPError):
        pass

    return _Err("http error", response=_Resp())


def test_push_refuses_unset_privacy_without_touching_api(monkeypatch, tmp_path):
    _install_fake_hub(monkeypatch, FakeHfApi)
    FakeHfApi.created.clear()
    run_dir = _make_output_dir(tmp_path)

    ok, url, error = push_output_to_huggingface(str(run_dir), "owner/model", private=None)

    assert ok is False
    assert url is None
    assert error is not None and "privacy" in error.lower()
    assert FakeHfApi.created == []


def test_push_private_and_public_flags(monkeypatch, tmp_path):
    _install_fake_hub(monkeypatch, PrivateExistingApi)
    PrivateExistingApi.created.clear()
    run_dir = _make_output_dir(tmp_path)

    ok, url, error = push_output_to_huggingface(
        str(run_dir), "owner/model", hf_token="tok", private=True
    )

    assert (ok, error) == (True, None)
    assert url == "https://huggingface.co/owner/model"
    api = PrivateExistingApi.created[-1]
    assert ("create_repo", "owner/model", True, True) in api.calls
    assert any(c[0] == "upload_folder" and c[1] == "owner/model" for c in api.calls)

    # Public request against a public repo proceeds with private=False.
    _install_fake_hub(monkeypatch, PublicExistingApi)
    PublicExistingApi.created.clear()

    ok, url, error = push_output_to_huggingface(
        str(run_dir), "owner/model", hf_token="tok", private=False
    )

    assert (ok, error) == (True, None)
    api = PublicExistingApi.created[-1]
    assert ("create_repo", "owner/model", False, True) in api.calls


def test_push_refuses_visibility_mismatch(monkeypatch, tmp_path):
    _install_fake_hub(monkeypatch, PublicExistingApi)
    PublicExistingApi.created.clear()
    run_dir = _make_output_dir(tmp_path)

    ok, url, error = push_output_to_huggingface(
        str(run_dir), "owner/model", hf_token="tok", private=True
    )

    assert ok is False
    assert url is None
    assert error is not None and "public" in error.lower()
    api = PublicExistingApi.created[-1]
    assert all(c[0] != "upload_folder" for c in api.calls)


def test_push_missing_repo_proceeds_to_create(monkeypatch, tmp_path):
    class GoneApi(FakeHfApi):
        def repo_info(self, repo_id, token=None):
            self.calls.append(("repo_info", repo_id))
            raise _http_error(404)

    # Install once so the raised error shares the client's error class.
    _install_fake_hub(monkeypatch, GoneApi)
    GoneApi.created.clear()
    run_dir = _make_output_dir(tmp_path)

    ok, url, error = push_output_to_huggingface(
        str(run_dir), "owner/model", hf_token="tok", private=True
    )

    assert (ok, error) == (True, None)
    api = GoneApi.created[-1]
    assert ("create_repo", "owner/model", True, True) in api.calls


def test_push_auth_failure_maps_precisely(monkeypatch, tmp_path):
    class DeniedApi(FakeHfApi):
        def repo_info(self, repo_id, token=None):
            self.calls.append(("repo_info", repo_id))
            raise _http_error(401)

    _install_fake_hub(monkeypatch, DeniedApi)
    run_dir = _make_output_dir(tmp_path)

    ok, url, error = push_output_to_huggingface(str(run_dir), "owner/model", private=False)

    assert ok is False
    assert url is None
    assert error is not None and "authentication" in error.lower()


def test_validate_repo_id():
    assert _validate_repo_id(" owner/model ") == "owner/model"
    assert _validate_repo_id("https://huggingface.co/owner/model?x=1") == "owner/model"
    assert _validate_repo_id("bare-name") is None
    assert _validate_repo_id("a/b/c") is None
    assert _validate_repo_id("") is None
    assert _validate_repo_id(None) is None


def _install_snapshot_stub(monkeypatch, behavior):
    pkg = types.ModuleType("huggingface_hub")

    def _snapshot_download(repo_id, revision=None, local_dir=None, token=None):
        return behavior(repo_id=repo_id, revision=revision, local_dir=local_dir, token=token)

    pkg.snapshot_download = _snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", pkg)


def test_download_success_and_revision_passthrough(monkeypatch, tmp_path):
    seen = {}

    def _ok(**kwargs):
        seen.update(kwargs)
        target = Path(kwargs["local_dir"])
        (target / "adapter_model.safetensors").write_bytes(b"fake-weights")
        return str(target)

    _install_snapshot_stub(monkeypatch, _ok)
    dest = tmp_path / "restored"

    ok, path, error = download_output_from_huggingface(
        "owner/model", str(dest), revision="abc123", hf_token="tok"
    )

    assert (ok, error) == (True, None)
    assert path == str(dest)
    assert seen["revision"] == "abc123"
    assert seen["token"] == "tok"
    assert (dest / "adapter_model.safetensors").is_file()


def test_download_failure_cleans_partials(monkeypatch, tmp_path):
    def _boom(**kwargs):
        target = Path(kwargs["local_dir"])
        (target / "partial.bin").write_bytes(b"incomplete")
        raise RuntimeError("boom mid-download")

    _install_snapshot_stub(monkeypatch, _boom)
    dest = tmp_path / "restored"

    ok, path, error = download_output_from_huggingface("owner/model", str(dest))

    assert ok is False
    assert path is None
    assert error is not None and "boom" in error
    assert dest.is_dir()
    assert list(dest.iterdir()) == []


def test_download_bad_repo_never_calls_api(monkeypatch, tmp_path):
    def _must_not_run(**kwargs):
        raise AssertionError("must not be called")

    _install_snapshot_stub(monkeypatch, _must_not_run)

    ok, path, error = download_output_from_huggingface("not-a-repo", str(tmp_path / "x"))

    assert ok is False
    assert path is None
    assert error is not None and "owner/name" in error


def _sibling(name, size=10):
    return types.SimpleNamespace(rfilename=name, size=size)


def _install_sparse_hub(monkeypatch, siblings, seen, fail_listing=False):
    pkg = types.ModuleType("huggingface_hub")

    def _repo_info(repo_id=None, revision=None, token=None):
        seen["listing"] = {"repo_id": repo_id, "revision": revision, "token": token}
        if fail_listing:
            raise RuntimeError("listing unavailable")
        return types.SimpleNamespace(siblings=list(siblings))

    def _snapshot_download(repo_id, revision=None, local_dir=None, token=None, **kwargs):
        seen["download"] = {
            "repo_id": repo_id,
            "revision": revision,
            "local_dir": local_dir,
            "token": token,
            **kwargs,
        }
        target = Path(local_dir)
        for name in kwargs.get("allow_patterns") or []:
            entry = target / name
            entry.parent.mkdir(parents=True, exist_ok=True)
            entry.write_bytes(b"fake")
        return str(target)

    pkg.repo_info = _repo_info
    pkg.snapshot_download = _snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", pkg)


def test_select_sparse_prefers_newest_checkpoint_numerically():
    siblings = [
        _sibling("checkpoint-5/optimizer.pt", 100),
        _sibling("checkpoint-5/model.safetensors", 200),
        _sibling("checkpoint-19/optimizer.pt", 100),
        _sibling("checkpoint-19/model.safetensors", 200),
        _sibling("checkpoint-9/optimizer.pt", 100),
        _sibling("config.json", 5),
        _sibling("run-config.json", 6),
        _sibling("tokenizer.json", 11),
        # Stop-save duplicates: same bytes at root and inside the bundle.
        _sibling("optimizer.pt", 100),
        _sibling("model.safetensors", 300),
        _sibling("stray-dir/notes.txt", 7),
    ]

    patterns, newest_step, omitted, omitted_bytes = _select_sparse_patterns(siblings)

    # Lexicographic order would pick checkpoint-5 or 9; numeric picks 19.
    assert newest_step == 19
    assert "checkpoint-19/optimizer.pt" in patterns
    assert "checkpoint-19/model.safetensors" in patterns
    assert "config.json" in patterns
    assert "run-config.json" in patterns
    assert "tokenizer.json" in patterns
    assert not any(p.startswith("checkpoint-5/") for p in patterns)
    assert not any(p.startswith("checkpoint-9/") for p in patterns)
    # Root twins of bundled files carry zero resume value: omitted.
    assert "optimizer.pt" not in patterns
    assert "model.safetensors" not in patterns
    assert "stray-dir/notes.txt" not in patterns
    assert omitted == 6  # ckpt-5 x2 + ckpt-9 x1 + root dups x2 + stray x1
    assert omitted_bytes == 100 + 200 + 100 + 100 + 300 + 7


def test_select_sparse_adapter_only_repo_keeps_all_roots():
    siblings = [
        _sibling("adapter_model.safetensors", 50),
        _sibling("adapter_config.json", 2),
        _sibling("run-config.json", 3),
    ]

    patterns, newest_step, omitted, omitted_bytes = _select_sparse_patterns(siblings)

    assert newest_step is None
    assert sorted(patterns) == ["adapter_config.json", "adapter_model.safetensors", "run-config.json"]
    assert (omitted, omitted_bytes) == (0, 0)


def test_download_sparse_passes_exact_patterns(monkeypatch, tmp_path):
    seen = {}
    siblings = [
        _sibling("checkpoint-5/optimizer.pt", 100),
        _sibling("checkpoint-19/optimizer.pt", 100),
        _sibling("optimizer.pt", 100),
        _sibling("config.json", 5),
        _sibling("run-config.json", 6),
    ]
    _install_sparse_hub(monkeypatch, siblings, seen)
    dest = tmp_path / "restored"

    ok, path, error = download_output_from_huggingface(
        "owner/model", str(dest), revision="abc123", hf_token="tok"
    )

    assert (ok, error) == (True, None)
    assert path == str(dest)
    assert seen["listing"] == {"repo_id": "owner/model", "revision": "abc123", "token": "tok"}
    requested = seen["download"]["allow_patterns"]
    assert sorted(requested) == ["checkpoint-19/optimizer.pt", "config.json", "run-config.json"]
    assert (dest / "checkpoint-19" / "optimizer.pt").is_file()
    assert not (dest / "checkpoint-5").exists()


def test_download_sparse_falls_back_to_full_when_listing_fails(monkeypatch, tmp_path):
    seen = {}
    _install_sparse_hub(monkeypatch, [], seen, fail_listing=True)
    dest = tmp_path / "restored"

    ok, path, error = download_output_from_huggingface("owner/model", str(dest))

    assert (ok, error) == (True, None)
    # Fallback keeps the historical call shape: no allow_patterns at all.
    assert "allow_patterns" not in seen["download"]


class SparsePushApi(PrivateExistingApi):
    """Fake client with modern upload_folder + prune surface."""

    deleted = []

    def upload_folder(self, repo_id, folder_path, commit_message=None, token=None,
                      allow_patterns=None):
        self.calls.append(("upload_folder", repo_id, folder_path, allow_patterns))

    def list_repo_files(self, repo_id=None, token=None):
        self.calls.append(("list_repo_files", repo_id))
        return list(getattr(self, "remote_files", []))

    def delete_folder(self, repo_id=None, folder_path=None, token=None):
        self.calls.append(("delete_folder", repo_id, folder_path))
        SparsePushApi.deleted.append(folder_path)


def _make_checkpoint_run(tmp_path):
    run_dir = tmp_path / "run_ckpt"
    run_dir.mkdir()
    for ckpt in ("checkpoint-5", "checkpoint-19"):
        d = run_dir / ckpt
        d.mkdir()
        (d / "optimizer.pt").write_bytes(b"fake-opt")
        (d / "model.safetensors").write_bytes(b"fake-weights")
    (run_dir / "config.json").write_bytes(b"{}")
    (run_dir / "run-config.json").write_bytes(b"{}")
    return run_dir


def test_push_sparse_uploads_newest_bundle_and_prunes_old(monkeypatch, tmp_path):
    _install_fake_hub(monkeypatch, SparsePushApi)
    SparsePushApi.created.clear()
    SparsePushApi.deleted.clear()
    SparsePushApi.remote_files = [
        "checkpoint-5/optimizer.pt",
        "checkpoint-19/optimizer.pt",
        "config.json",
    ]
    run_dir = _make_checkpoint_run(tmp_path)

    ok, url, error = push_output_to_huggingface(
        str(run_dir), "owner/model", hf_token="tok", private=True
    )

    assert (ok, error) == (True, None)
    assert url == "https://huggingface.co/owner/model"
    api = SparsePushApi.created[-1]
    upload = next(c for c in api.calls if c[0] == "upload_folder")
    assert sorted(upload[3]) == [
        "README.md",
        "checkpoint-19/model.safetensors",
        "checkpoint-19/optimizer.pt",
        "config.json",
        "run-config.json",
    ]
    # Only the older remote bundle is pruned; newest is never deleted.
    assert SparsePushApi.deleted == ["checkpoint-5"]


def test_push_prune_never_touches_newer_remote_bundles(monkeypatch, tmp_path):
    _install_fake_hub(monkeypatch, SparsePushApi)
    SparsePushApi.created.clear()
    SparsePushApi.deleted.clear()
    SparsePushApi.remote_files = [
        "checkpoint-5/optimizer.pt",
        "checkpoint-19/optimizer.pt",
        "checkpoint-25/optimizer.pt",
    ]
    run_dir = _make_checkpoint_run(tmp_path)

    ok, _, error = push_output_to_huggingface(
        str(run_dir), "owner/model", hf_token="tok", private=True
    )

    assert (ok, error) == (True, None)
    assert SparsePushApi.deleted == ["checkpoint-5"]


def test_push_adapter_only_uploads_roots_and_prunes_nothing(monkeypatch, tmp_path):
    _install_fake_hub(monkeypatch, SparsePushApi)
    SparsePushApi.created.clear()
    SparsePushApi.deleted.clear()
    SparsePushApi.remote_files = ["checkpoint-5/optimizer.pt", "config.json"]
    run_dir = _make_output_dir(tmp_path)

    ok, _, error = push_output_to_huggingface(
        str(run_dir), "owner/model", hf_token="tok", private=True
    )

    assert (ok, error) == (True, None)
    api = SparsePushApi.created[-1]
    upload = next(c for c in api.calls if c[0] == "upload_folder")
    assert sorted(upload[3]) == ["README.md", "adapter_model.safetensors"]
    assert SparsePushApi.deleted == []


def test_push_prune_skipped_when_listing_fails(monkeypatch, tmp_path):
    class NoListApi(SparsePushApi):
        def list_repo_files(self, repo_id=None, token=None):
            raise RuntimeError("listing down")

    _install_fake_hub(monkeypatch, NoListApi)
    NoListApi.created.clear()
    NoListApi.deleted.clear()
    run_dir = _make_checkpoint_run(tmp_path)

    ok, _, error = push_output_to_huggingface(
        str(run_dir), "owner/model", hf_token="tok", private=True
    )

    assert (ok, error) == (True, None)
    assert NoListApi.deleted == []


def test_push_legacy_client_falls_back_to_full_upload(monkeypatch, tmp_path):
    # Legacy upload_folder without allow_patterns: historical full upload.
    _install_fake_hub(monkeypatch, PrivateExistingApi)
    PrivateExistingApi.created.clear()
    run_dir = _make_checkpoint_run(tmp_path)

    ok, _, error = push_output_to_huggingface(
        str(run_dir), "owner/model", hf_token="tok", private=True
    )

    assert (ok, error) == (True, None)
    api = PrivateExistingApi.created[-1]
    upload = next(c for c in api.calls if c[0] == "upload_folder")
    assert upload == ("upload_folder", "owner/model", str(run_dir))
