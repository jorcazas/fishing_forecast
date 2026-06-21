"""Tests del fetch de artefactos S3 legacy (cliente boto3 mockeado; sin tocar AWS)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fishing_forecast.config import Settings
from fishing_forecast.etl.extract import s3_legacy


def _settings_with_keys(tmp_path: Path, keys: dict | None) -> Settings:
    keys_file = tmp_path / "keys.json"
    if keys is not None:
        keys_file.write_text(json.dumps(keys))
    return Settings(keys_file=keys_file)


def test_load_keys_reads_file(tmp_path: Path) -> None:
    s = _settings_with_keys(tmp_path, {"aws_access_key_id": "AKIA", "aws_secret_access_key": "x"})
    assert s.load_keys()["aws_access_key_id"] == "AKIA"


def test_load_keys_absent_returns_empty(tmp_path: Path) -> None:
    s = _settings_with_keys(tmp_path, None)
    assert s.load_keys() == {}


def test_resolve_bucket_priority(tmp_path: Path) -> None:
    s = _settings_with_keys(tmp_path, {"bucket": "from-keys"})
    assert s3_legacy.resolve_bucket(s) == "from-keys"
    assert s3_legacy.resolve_bucket(s, override="explicit") == "explicit"


def test_resolve_bucket_missing_raises(tmp_path: Path) -> None:
    s = _settings_with_keys(tmp_path, {"aws_access_key_id": "AKIA"})
    with pytest.raises(ValueError, match="bucket"):
        s3_legacy.resolve_bucket(s)


def test_list_artifacts_paginates() -> None:
    client = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {"Contents": [{"Key": "models/xgb.joblib", "Size": 10}]},
        {"Contents": [{"Key": "metrics/exp.json", "Size": 20}]},
    ]
    client.get_paginator.return_value = paginator
    objs = s3_legacy.list_artifacts(client, "bucket", "")
    assert [(o.key, o.size) for o in objs] == [
        ("models/xgb.joblib", 10),
        ("metrics/exp.json", 20),
    ]


def test_download_artifact_skips_when_same_size(tmp_path: Path) -> None:
    target = tmp_path / "models" / "xgb.joblib"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"abc")  # 3 bytes
    client = MagicMock()
    client.head_object.return_value = {"ContentLength": 3}
    s3_legacy.download_artifact(client, "bucket", "models/xgb.joblib", tmp_path)
    client.download_file.assert_not_called()  # mismo tamaño → skip


def test_download_artifact_downloads_when_absent(tmp_path: Path) -> None:
    client = MagicMock()
    out = s3_legacy.download_artifact(client, "bucket", "metrics/exp.json", tmp_path)
    client.download_file.assert_called_once_with("bucket", "metrics/exp.json", str(out))
    assert out == tmp_path / "metrics" / "exp.json"


def test_sync_downloads_all(tmp_path: Path) -> None:
    client = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {"Contents": [{"Key": "a.json", "Size": 1}, {"Key": "b.joblib", "Size": 2}]}
    ]
    client.get_paginator.return_value = paginator
    paths = s3_legacy.sync(client, "bucket", "", tmp_path)
    assert len(paths) == 2
    assert client.download_file.call_count == 2
