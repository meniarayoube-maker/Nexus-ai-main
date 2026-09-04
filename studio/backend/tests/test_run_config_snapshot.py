# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for :mod:`core.training.run_config_snapshot` (stdlib-only)."""

from pathlib import Path

from core.training.run_config_snapshot import (
    RUN_CONFIG_FILENAME,
    build_restored_config,
    load_run_config_snapshot,
    sanitize_run_config,
    write_run_config_snapshot,
)


def test_sanitize_drops_secrets_case_insensitively():
    cleaned = sanitize_run_config(
        {
            "model_name": "unsloth/qwen2.5-0.5b",
            "hf_token": "secret-token",
            "KAGGLE_KEY": "secret-key",
            "wandb_token": "secret",
            "api_secret": "x",
            "db_password": "x",
            "kaggle_username": "walzoomtech",
            "approved_remote_code_fingerprint": "abc123",
            "max_steps": 30,
        }
    )
    assert cleaned["model_name"] == "unsloth/qwen2.5-0.5b"
    assert cleaned["kaggle_username"] == "walzoomtech"
    assert cleaned["approved_remote_code_fingerprint"] == "abc123"
    assert cleaned["max_steps"] == 30
    for key in cleaned:
        lowered = key.lower()
        assert "token" not in lowered
        assert "secret" not in lowered
        assert "password" not in lowered
        assert key != "KAGGLE_KEY"


def test_sanitize_coerces_unserializable_values():
    cleaned = sanitize_run_config({"output_dir": Path("/tmp/x"), "tags": {"a", "b"}})
    assert cleaned["output_dir"] == str(Path("/tmp/x"))
    # Sets degrade through str() so the snapshot stays JSON-safe.
    assert isinstance(cleaned["tags"], str)


def test_write_then_load_roundtrip(tmp_path):
    target = write_run_config_snapshot(
        tmp_path, {"model_name": "m", "training_type": "LoRA/QLoRA", "hf_token": "nope"}
    )
    assert target == str(tmp_path / RUN_CONFIG_FILENAME)
    loaded = load_run_config_snapshot(tmp_path)
    assert loaded is not None
    assert loaded["model_name"] == "m"
    assert loaded["training_type"] == "LoRA/QLoRA"
    assert "hf_token" not in loaded


def test_load_absent_or_corrupt_returns_none(tmp_path):
    assert load_run_config_snapshot(tmp_path / "missing") is None
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / RUN_CONFIG_FILENAME).write_text("{not json", encoding="utf-8")
    assert load_run_config_snapshot(bad) is None
    wrong = tmp_path / "wrong"
    wrong.mkdir()
    (wrong / RUN_CONFIG_FILENAME).write_text("[1, 2]", encoding="utf-8")
    assert load_run_config_snapshot(wrong) is None


def test_build_prefers_snapshot_file_and_scrubs_stale_paths():
    config = build_restored_config(
        file_config={
            "model_name": "unsloth/qwen2.5-0.5b-bnb-4bit",
            "training_type": "LoRA/QLoRA",
            "format_type": "alpaca",
            "hf_dataset": "unsloth/alpaca-cleaned",
            "output_dir": "/root/.unsloth/studio/outputs/old",
            "resume_from_checkpoint": "/root/.unsloth/studio/outputs/old/checkpoint-5",
            "start_request_id": "stale-id",
            "max_steps": 30,
        },
        inferred_model="/kaggle/working/unsloth-outputs/restored",
        inferred_training_type="Full Finetuning",
        manual_hf_dataset="other/dataset",
        slug="owner/name",
        storage_target="kaggle",
    )
    assert config["model_name"] == "unsloth/qwen2.5-0.5b-bnb-4bit"
    assert config["training_type"] == "LoRA/QLoRA"
    assert config["hf_dataset"] == "unsloth/alpaca-cleaned"
    assert config["max_steps"] == 30
    assert config["restored_from_kaggle"] == "owner/name"
    assert config["storage_target"] == "kaggle"
    for stale in ("output_dir", "resume_from_checkpoint", "start_request_id"):
        assert stale not in config


def test_build_fills_gaps_from_inference_and_manual():
    config = build_restored_config(
        file_config=None,
        inferred_model="/kaggle/working/unsloth-outputs/restored",
        inferred_training_type="Full Finetuning",
        manual_hf_dataset="unsloth/alpaca-cleaned",
        slug="owner/name",
        storage_target="kaggle",
    )
    assert config["model_name"] == "/kaggle/working/unsloth-outputs/restored"
    assert config["training_type"] == "Full Finetuning"
    assert config["format_type"] == "alpaca"
    assert config["hf_dataset"] == "unsloth/alpaca-cleaned"
