"""Tests del extractor Copernicus (subset mockeado; sin red ni credenciales)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fishing_forecast.config import Settings
from fishing_forecast.etl.extract.copernicus import (
    ProductSpec,
    build_subset_kwargs,
    download_product,
    extract,
    load_products,
)

PRODUCT = ProductSpec(
    dataset_id="cmems_obs-sst_glo_phy_my_l4_P1D-m",
    short_name="sst_l4",
    variables=["analysed_sst"],
    region={"lon_min": -117.0, "lon_max": -112.5, "lat_min": 28.0, "lat_max": 32.0},
)


def test_load_products_real_config() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    products = load_products(repo_root / "configs" / "copernicus_vars.yaml")
    assert any(p.short_name == "sst_l4" for p in products)


def test_build_subset_kwargs_maps_region_and_dates() -> None:
    kw = build_subset_kwargs(PRODUCT, start=date(2017, 1, 1), end=date(2020, 6, 30), output_dir=Path("/tmp/cop"))
    assert kw["dataset_id"] == PRODUCT.dataset_id
    assert kw["variables"] == ["analysed_sst"]
    assert kw["minimum_longitude"] == -117.0
    assert kw["maximum_longitude"] == -112.5
    assert kw["minimum_latitude"] == 28.0
    assert kw["maximum_latitude"] == 32.0
    assert kw["start_datetime"] == "2017-01-01T00:00:00"
    assert kw["end_datetime"] == "2020-06-30T00:00:00"
    assert kw["output_filename"] == "sst_l4.nc"


def test_download_product_calls_subset_with_credentials(tmp_path: Path) -> None:
    subset = MagicMock()
    settings = Settings(copernicus_user="u", copernicus_pass="p")
    out = download_product(
        PRODUCT,
        start=date(2017, 1, 1),
        end=date(2017, 1, 31),
        output_dir=tmp_path,
        settings=settings,
        subset_fn=subset,
    )
    assert out == tmp_path / "sst_l4.nc"
    subset.assert_called_once()
    kwargs = subset.call_args.kwargs
    assert kwargs["username"] == "u" and kwargs["password"] == "p"
    assert kwargs["overwrite"] is True


def test_download_product_omits_credentials_when_absent(tmp_path: Path) -> None:
    subset = MagicMock()
    settings = Settings(copernicus_user="", copernicus_pass="")
    download_product(
        PRODUCT, start="2017-01-01", end="2017-01-31", output_dir=tmp_path, settings=settings, subset_fn=subset
    )
    kwargs = subset.call_args.kwargs
    assert "username" not in kwargs and "password" not in kwargs  # usa el login file del SDK


def test_download_product_idempotent_skip(tmp_path: Path) -> None:
    (tmp_path / "sst_l4.nc").write_bytes(b"x")  # ya existe
    subset = MagicMock()
    settings = Settings(copernicus_user="u", copernicus_pass="p")
    download_product(
        PRODUCT, start="2017-01-01", end="2017-01-31", output_dir=tmp_path, settings=settings, subset_fn=subset
    )
    subset.assert_not_called()  # no re-descarga


def test_download_product_force_redownloads(tmp_path: Path) -> None:
    (tmp_path / "sst_l4.nc").write_bytes(b"x")
    subset = MagicMock()
    settings = Settings(copernicus_user="u", copernicus_pass="p")
    download_product(
        PRODUCT,
        start="2017-01-01",
        end="2017-01-31",
        output_dir=tmp_path,
        force=True,
        settings=settings,
        subset_fn=subset,
    )
    subset.assert_called_once()


def test_extract_iterates_products(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    subset = MagicMock()
    settings = Settings(copernicus_user="u", copernicus_pass="p")
    paths = extract(
        config_path=repo_root / "configs" / "copernicus_vars.yaml",
        start="2017-01-01",
        end="2017-12-31",
        output_dir=tmp_path,
        settings=settings,
        subset_fn=subset,
    )
    assert len(paths) >= 1
    assert subset.call_count == len(paths)
