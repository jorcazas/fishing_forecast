"""Tests del extractor NOAA OISST: construcción de URLs e idempotencia del downloader."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fishing_forecast.etl.extract.sst_oisst import (
    PSL_BASE_URL,
    OISSTFileSpec,
    build_specs,
    download_file,
)


def test_build_specs_one_per_year_sorted() -> None:
    specs = build_specs([2011, 1982, 1982])
    assert [s.year for s in specs] == [1982, 2011]  # ordenado y deduplicado
    assert specs[0].filename == "sst.day.mean.1982.nc"
    assert specs[0].url == f"{PSL_BASE_URL}/sst.day.mean.1982.nc"


def test_build_specs_rejects_pre_1982() -> None:
    with pytest.raises(ValueError, match="empieza en 1982"):
        build_specs([1981])


def _mock_session_with_body(body: bytes, *, etag: str = '"abc"') -> MagicMock:
    session = MagicMock()
    response = MagicMock()
    response.headers = {"ETag": etag, "Content-Length": str(len(body))}
    response.iter_content = lambda chunk_size: [body]
    response.raise_for_status = lambda: None
    response.__enter__ = lambda self: response
    response.__exit__ = lambda self, *a: None
    session.get.return_value = response
    return session


def test_download_writes_file_and_meta(tmp_path: Path) -> None:
    spec = OISSTFileSpec(2000, f"{PSL_BASE_URL}/sst.day.mean.2000.nc", "sst.day.mean.2000.nc")
    session = _mock_session_with_body(b"netcdf-bytes")
    out = download_file(spec, tmp_path, session=session)
    assert out.read_bytes() == b"netcdf-bytes"
    meta = json.loads((tmp_path / "sst.day.mean.2000.nc.meta.json").read_text())
    assert meta["etag"] == '"abc"'
    session.get.assert_called_once()


def test_download_is_idempotent_when_etag_matches(tmp_path: Path) -> None:
    spec = OISSTFileSpec(2000, f"{PSL_BASE_URL}/sst.day.mean.2000.nc", "sst.day.mean.2000.nc")
    session = _mock_session_with_body(b"netcdf-bytes")
    download_file(spec, tmp_path, session=session)

    # Segunda llamada: HEAD reporta el mismo ETag → no debe re-descargar (get no se llama de nuevo).
    head = MagicMock()
    head.ok = True
    head.headers = {"ETag": '"abc"'}
    session.head.return_value = head
    session.get.reset_mock()
    download_file(spec, tmp_path, session=session)
    session.get.assert_not_called()


def test_force_redownloads(tmp_path: Path) -> None:
    spec = OISSTFileSpec(2000, f"{PSL_BASE_URL}/sst.day.mean.2000.nc", "sst.day.mean.2000.nc")
    session = _mock_session_with_body(b"v1")
    download_file(spec, tmp_path, session=session)
    session.get.reset_mock()
    download_file(spec, tmp_path, session=session, force=True)
    session.get.assert_called_once()
