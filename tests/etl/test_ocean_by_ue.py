"""Tests de la agregación SST por bbox de UE (puros sobre xarray sintético + roundtrip netCDF)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fishing_forecast.etl.aggregate.mhw import MHWParams
from fishing_forecast.etl.aggregate.ocean_by_ue import (
    open_oisst,
    sst_bbox_mean,
    sst_mhw_for_bbox,
)

# bbox tipo San Quintín (convención -180..180).
SQ_BBOX = {"lon_min": -117.0, "lon_max": -115.0, "lat_min": 30.0, "lat_max": 31.5}


def _make_dataset(*, lon_0_360: bool, n_days: int = 5) -> xr.Dataset:
    """Dataset OISST-like con 4 lats y 4 lons; dos celdas caen dentro del bbox SQ."""
    lats = np.array([29.0, 30.5, 31.0, 33.0])  # 30.5 y 31.0 dentro de [30, 31.5]
    lons_180 = np.array([-118.0, -116.0, -115.5, -113.0])  # -116 y -115.5 dentro de [-117,-115]
    lons = (lons_180 % 360) if lon_0_360 else lons_180
    times = pd.date_range("2000-06-01", periods=n_days, freq="D")
    # SST = índice de celda, para que el promedio de las celdas in-box sea predecible.
    base = np.arange(len(lats) * len(lons), dtype=float).reshape(len(lats), len(lons))
    data = np.broadcast_to(base, (n_days, len(lats), len(lons))).copy()
    return xr.Dataset(
        {"sst": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )


def test_bbox_mean_selects_inbox_cells_180() -> None:
    ds = _make_dataset(lon_0_360=False)
    df = sst_bbox_mean(ds, SQ_BBOX)
    # In-box: lat idx 1,2 (30.5, 31.0) × lon idx 1,2 (-116, -115.5) → valores base 5,6,9,10.
    expected = np.mean([5.0, 6.0, 9.0, 10.0])
    assert len(df) == 5
    assert list(df.columns) == ["ds", "sst"]
    assert df["sst"].iloc[0] == pytest.approx(expected)
    assert (df["sst"] == df["sst"].iloc[0]).all()  # serie constante en el tiempo


def test_bbox_mean_handles_0_360_longitude() -> None:
    ds = _make_dataset(lon_0_360=True)
    df = sst_bbox_mean(ds, SQ_BBOX)
    expected = np.mean([5.0, 6.0, 9.0, 10.0])
    assert df["sst"].iloc[0] == pytest.approx(expected)


def test_bbox_outside_grid_warns_and_nans() -> None:
    ds = _make_dataset(lon_0_360=False)
    far = {"lon_min": 10.0, "lon_max": 11.0, "lat_min": 10.0, "lat_max": 11.0}
    df = sst_bbox_mean(ds, far)
    assert df["sst"].isna().all()


def test_open_oisst_roundtrip(tmp_path: Path) -> None:
    ds = _make_dataset(lon_0_360=True)
    nc_path = tmp_path / "sst.day.mean.2000.nc"
    ds.to_netcdf(nc_path)
    with open_oisst([nc_path]) as reopened:
        df = sst_bbox_mean(reopened, SQ_BBOX)
    assert df["sst"].iloc[0] == pytest.approx(np.mean([5.0, 6.0, 9.0, 10.0]))


def test_sst_mhw_for_bbox_appends_mhw_columns(tmp_path: Path) -> None:
    # 3 años diarios para tener baseline (2 años) + serie; inyecta una ola en 2002.
    times = pd.date_range("2000-01-01", "2002-12-31", freq="D")
    lats = np.array([30.5, 31.0])
    lons = np.array([-116.0, -115.5])
    doy = times.dayofyear.to_numpy()
    seasonal = 18.0 + 5.0 * np.sin(2 * np.pi * (doy - 80) / 365.0)
    data = np.broadcast_to(seasonal[:, None, None], (len(times), 2, 2)).copy()
    hw = (times >= "2002-07-01") & (times <= "2002-07-12")
    data[hw, :, :] += 6.0
    ds = xr.Dataset(
        {"sst": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    nc_path = tmp_path / "sst.day.mean.test.nc"
    ds.to_netcdf(nc_path)

    params = MHWParams(
        baseline_start=date(2000, 1, 1),
        baseline_end=date(2001, 12, 31),
        smooth_window=11,
    )
    df = sst_mhw_for_bbox([nc_path], SQ_BBOX, params)
    assert {"sst_anomaly", "mhw_category", "mhw_intensity"} <= set(df.columns)
    event = (df["ds"] >= "2002-07-01") & (df["ds"] <= "2002-07-12")
    assert (df.loc[event, "mhw_category"] >= 1).all()
