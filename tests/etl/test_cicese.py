"""Tests del extractor y la transformación de CICESE."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from fishing_forecast.etl.extract.cicese import (
    Station,
    build_index_url,
    parse_index_html,
)
from fishing_forecast.etl.transform.cicese import (
    CICESE_COLUMNS,
    read_dat,
    to_daily,
    transform,
)


@pytest.fixture
def index_html(fixtures_dir: Path) -> str:
    return (fixtures_dir / "cicese_index_sample.html").read_text(encoding="utf-8")


@pytest.fixture
def dat_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "cicese_sample.dat"


def test_build_index_url() -> None:
    url = build_index_url("GRON", 2021)
    assert url == "http://redmar.cicese.mx/emmc/DATA/GRON/MIN/2021/"


def test_parse_index_finds_only_dat_files(index_html: str) -> None:
    urls = parse_index_html(index_html, "http://redmar.cicese.mx/emmc/DATA/GRON/MIN/2021/")
    assert len(urls) == 2  # ignora ../ y readme.txt
    assert all(u.endswith(".dat") for u in urls)
    assert urls[0].endswith("/GRON2021_01.dat")  # URL absoluta


def test_station_dataclass() -> None:
    s = Station(name="guerrero_negro", code="GRON")
    assert s.code == "GRON"


def test_read_dat_assigns_23_columns(dat_path: Path) -> None:
    df = read_dat(dat_path)
    assert list(df.columns) == list(CICESE_COLUMNS)
    assert len(df) == 4


def test_to_daily_medians_and_renames(dat_path: Path) -> None:
    df = read_dat(dat_path)
    daily = to_daily(df, station="guerrero_negro", region="vizcaino")
    assert list(daily.columns[:3]) == ["ds", "station", "region"]
    assert "water_temperature" in daily.columns
    assert "temperatura_agua" not in daily.columns  # renombrado
    assert len(daily) == 2  # dos días
    by_day = daily.set_index("ds")
    assert by_day.loc[date(2021, 3, 1), "water_temperature"] == pytest.approx(19.0)
    assert by_day.loc[date(2021, 3, 2), "water_temperature"] == pytest.approx(22.0)
    assert (daily["station"] == "guerrero_negro").all()
    assert (daily["region"] == "vizcaino").all()


def test_to_daily_respects_aggregates_filter(dat_path: Path) -> None:
    daily = to_daily(
        read_dat(dat_path),
        station="guerrero_negro",
        region="vizcaino",
        aggregates=["water_temperature"],
    )
    assert list(daily.columns) == ["ds", "station", "region", "water_temperature"]


def test_transform_writes_parquet(dat_path: Path, tmp_path: Path) -> None:
    out_path = tmp_path / "cicese" / "guerrero_negro.parquet"
    df = transform(
        [dat_path, dat_path],  # dos archivos: mismas fechas → medianas iguales, no duplica días
        station="guerrero_negro",
        region="vizcaino",
        out_path=out_path,
    )
    assert out_path.exists()
    roundtrip = pd.read_parquet(out_path)
    assert len(roundtrip) == len(df) == 2
