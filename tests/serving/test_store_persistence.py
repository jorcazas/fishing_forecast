"""Tests de la persistencia del store de producción (B7)."""

from __future__ import annotations

import json

import pytest

from fishing_forecast.serving.forecast import (
    STORE_FORMAT_VERSION,
    ForecastStore,
    get_store,
    load_store,
    save_store,
)


def _store() -> ForecastStore:
    return ForecastStore(
        cut_date="2024-06-01",
        series={
            "lobster_red@litoral_bc_sur": {
                "series": "lobster_red@litoral_bc_sur",
                "dates": ["2024-09-15", "2024-09-16"],
                "y_obs": [120.0, 0.0],
                "median": [98.5, 1.2],
                "coverage90": 0.9,
                "n_test": 2,
            }
        },
        units={"litoral_bc_sur": {"name": "Litoral de BC Sur", "region": "sq", "lat": 30.4}},
    )


def test_save_load_roundtrip_preserves_series_and_units(tmp_path):
    path = save_store(_store(), tmp_path / "store.json")
    loaded = load_store(path)
    assert loaded.cut_date == "2024-06-01"
    assert loaded.series == _store().series
    assert loaded.units == _store().units


def test_manifest_records_cut_series_and_format_version(tmp_path):
    path = save_store(_store(), tmp_path / "store.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format_version"] == STORE_FORMAT_VERSION
    assert payload["cut_date"] == "2024-06-01"
    assert payload["n_series"] == 1
    assert payload["created_at"]


def test_load_store_rejects_a_different_format_version(tmp_path):
    path = save_store(_store(), tmp_path / "store.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["format_version"] = STORE_FORMAT_VERSION + 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="format_version"):
        load_store(path)


def test_get_store_prefers_the_artifact_over_training(tmp_path, monkeypatch):
    path = save_store(_store(), tmp_path / "store.json")

    def _fail(*_a, **_k):  # pragma: no cover - debe no llamarse
        raise AssertionError("get_store entrenó teniendo artefacto disponible")

    monkeypatch.setattr("fishing_forecast.serving.forecast.build_store", _fail)
    assert get_store(path).cut_date == "2024-06-01"
