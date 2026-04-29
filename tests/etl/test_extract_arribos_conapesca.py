"""Tests del extractor de CONAPESCA.

Cubre la función pura ``parse_index_html`` (sin red) y la idempotencia del
downloader (con HTTP mockeado a través de un ``requests.Session`` falsa).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fishing_forecast.etl.extract.arribos_conapesca import (
    FileSpec,
    download_file,
    parse_index_html,
)


@pytest.fixture
def index_html(fixtures_dir: Path) -> str:
    return (fixtures_dir / "conapesca_index_sample.html").read_text(encoding="utf-8")


def test_parse_index_finds_all_known_files(index_html: str) -> None:
    specs = parse_index_html(index_html)
    # 4 arribo (2018, 2024, 2025, 2026) + 3 produccion (2018, 2024, 2025) = 7
    assert len(specs) == 7
    years_arribo = sorted(s.year for s in specs if s.kind == "arribo_cosecha")
    assert years_arribo == [2018, 2024, 2025, 2026]
    years_prod = sorted(s.year for s in specs if s.kind == "produccion")
    assert years_prod == [2018, 2024, 2025]


def test_parse_index_handles_url_irregularities(index_html: str) -> None:
    specs = parse_index_html(index_html)
    # Pre-2025 arribo: literal space en el filename, debe haberse decodificado
    pre_2025 = next(s for s in specs if s.year == 2024 and s.kind == "arribo_cosecha")
    assert " " in pre_2025.filename, "El filename debe conservar el espacio literal"
    assert pre_2025.filename == "AVISOS_ MAYORES_MENORES_COSECHA_2024.csv"
    # Post-2025: path anidado
    post_2025 = next(s for s in specs if s.year == 2025 and s.kind == "arribo_cosecha")
    assert "/2025/aviso_arribo/" in post_2025.url


def test_parse_index_ignores_unmatched_csvs(index_html: str) -> None:
    specs = parse_index_html(index_html)
    # SOMETHING_ELSE_2020.csv no debe aparecer
    assert all("SOMETHING_ELSE" not in s.url for s in specs)


def test_parse_index_results_are_sorted(index_html: str) -> None:
    specs = parse_index_html(index_html)
    keys = [(s.year, s.kind) for s in specs]
    assert keys == sorted(keys)


def test_filespec_relative_path() -> None:
    spec = FileSpec(
        year=2024,
        kind="arribo_cosecha",
        url="https://example/AVISOS_ MAYORES_MENORES_COSECHA_2024.csv",
        filename="AVISOS_ MAYORES_MENORES_COSECHA_2024.csv",
    )
    assert spec.relative_path() == Path("arribo_cosecha/AVISOS_ MAYORES_MENORES_COSECHA_2024.csv")


# ---------- Downloader: idempotencia ----------


def _make_session_with_response(content: bytes, headers: dict[str, str]) -> MagicMock:
    session = MagicMock()
    response_ctx = MagicMock()
    response_ctx.__enter__.return_value = response_ctx
    response_ctx.__exit__.return_value = False
    response_ctx.raise_for_status.return_value = None
    response_ctx.headers = headers
    response_ctx.iter_content.return_value = iter([content])
    session.get.return_value = response_ctx
    head_response = MagicMock()
    head_response.ok = True
    head_response.headers = headers
    session.head.return_value = head_response
    return session


def test_download_file_writes_target_and_meta(tmp_path: Path) -> None:
    spec = FileSpec(
        year=2024,
        kind="arribo_cosecha",
        url="https://example/file.csv",
        filename="file.csv",
    )
    payload = b"col_a,col_b\n1,2\n3,4\n"
    headers = {
        "ETag": '"abc123"',
        "Last-Modified": "Mon, 01 Jan 2024 00:00:00 GMT",
        "Content-Length": str(len(payload)),
    }
    session = _make_session_with_response(payload, headers)

    target = download_file(spec, tmp_path, session=session)

    assert target.exists()
    assert target.read_bytes() == payload
    meta = json.loads((target.with_suffix(target.suffix + ".meta.json")).read_text())
    assert meta["etag"] == '"abc123"'
    assert meta["content_length"] == len(payload)
    session.get.assert_called_once()


def test_download_file_skips_when_etag_matches(tmp_path: Path) -> None:
    spec = FileSpec(
        year=2024,
        kind="arribo_cosecha",
        url="https://example/file.csv",
        filename="file.csv",
    )
    headers = {
        "ETag": '"unchanged"',
        "Last-Modified": "Mon, 01 Jan 2024 00:00:00 GMT",
        "Content-Length": "10",
    }
    # Pre-cargamos archivo + meta como si ya hubiéramos descargado
    target = tmp_path / spec.relative_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"x" * 10)
    target.with_suffix(target.suffix + ".meta.json").write_text(
        json.dumps(
            {
                "url": spec.url,
                "etag": '"unchanged"',
                "last_modified": "Mon, 01 Jan 2024 00:00:00 GMT",
                "content_length": 10,
            }
        )
    )
    session = _make_session_with_response(b"new content", headers)

    result = download_file(spec, tmp_path, session=session)

    assert result == target
    # No debió llamar GET (hit de cache)
    session.get.assert_not_called()
    session.head.assert_called_once()


def test_download_file_force_redownloads(tmp_path: Path) -> None:
    spec = FileSpec(
        year=2024,
        kind="arribo_cosecha",
        url="https://example/file.csv",
        filename="file.csv",
    )
    target = tmp_path / spec.relative_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"old")
    target.with_suffix(target.suffix + ".meta.json").write_text(
        json.dumps({"etag": '"x"', "last_modified": None, "content_length": 3})
    )
    headers = {"ETag": '"x"', "Content-Length": "5"}
    session = _make_session_with_response(b"fresh", headers)

    download_file(spec, tmp_path, session=session, force=True)

    assert target.read_bytes() == b"fresh"
    session.get.assert_called_once()
