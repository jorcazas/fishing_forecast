"""Agregación oceanográfica por unidad económica: promedio espacial sobre el bbox TURF.

Por ahora cubre la SST de NOAA OISST: abre los netCDF anuales, recorta al bounding box
de la UE y promedia espacialmente para obtener **un escalar de SST por día por UE**
(granularidad que pide `docs/etl_design.md` §5.3). Sobre esa serie diaria se calcula el
índice MHW (`etl/aggregate/mhw.py`).

Las funciones de cálculo (`sst_bbox_mean`) son puras sobre un `xarray.Dataset`, así que
se testean con datasets sintéticos sin descargar nada. `open_oisst` aísla la lectura de
disco.

Nota sobre longitud: OISST usa convención **0-360**, mientras que los bbox de
`configs/economic_units.yaml` están en **-180..180**. `sst_bbox_mean` detecta la
convención del dataset y convierte el bbox en consecuencia (incluyendo el wrap en el
antimeridiano).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import xarray as xr
from loguru import logger

from fishing_forecast.etl.aggregate.mhw import MHWParams, add_mhw

_LAT_NAMES = ("lat", "latitude", "nav_lat", "y")
_LON_NAMES = ("lon", "longitude", "nav_lon", "x")
_TIME_NAMES = ("time", "ds", "date", "t")


def _find_coord(dataset: xr.Dataset | xr.DataArray, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in dataset.coords or name in getattr(dataset, "dims", ()):
            return name
    raise KeyError(
        f"No encontré ninguna coordenada en {candidates}; presentes: {list(dataset.coords)}"
    )


def _select_sst_var(dataset: xr.Dataset, sst_var: str | None) -> xr.DataArray:
    if sst_var is not None:
        return dataset[sst_var]
    if "sst" in dataset.data_vars:
        return dataset["sst"]
    data_vars = list(dataset.data_vars)
    if len(data_vars) == 1:
        return dataset[data_vars[0]]
    raise ValueError(f"No pude inferir la variable de SST entre {data_vars}; especifica `sst_var`.")


def _to_celsius(da: xr.DataArray) -> xr.DataArray:
    """Devuelve la SST en °C. OISST viene en °C; OSTIA/Copernicus en Kelvin.

    Detecta por el atributo `units`; si falta, usa la magnitud (Kelvin ronda 270-310).
    """
    units = str(da.attrs.get("units", "")).strip().lower()
    if units in ("kelvin", "k", "degk", "degrees_kelvin"):
        return da - 273.15
    if units in ("celsius", "c", "degc", "degree_celsius", "degrees_celsius", "deg_c"):
        return da
    # Sin units fiable: heurística por magnitud (una SST de >100 solo tiene sentido en K).
    if float(da.max()) > 100.0:
        return da - 273.15
    return da


def sst_bbox_mean(
    dataset: xr.Dataset,
    bbox: dict[str, float],
    *,
    sst_var: str | None = None,
) -> pd.DataFrame:
    """Promedio espacial diario de SST dentro de `bbox`. Pura sobre un `xarray.Dataset`.

    `bbox` = ``{lon_min, lon_max, lat_min, lat_max}`` en convención -180..180.
    Devuelve un DataFrame con columnas ``ds`` (datetime) y ``sst`` (float), ordenado por
    fecha. Promedia ignorando NaN (celdas de tierra / fuera de máscara).
    """
    da = _to_celsius(_select_sst_var(dataset, sst_var))
    time_name = _find_coord(da, _TIME_NAMES)
    series = _bbox_spatial_mean(da, bbox)
    df = series.to_dataframe(name="sst").reset_index()[[time_name, "sst"]]
    df = df.rename(columns={time_name: "ds"})
    df["ds"] = pd.to_datetime(df["ds"])
    return df.sort_values("ds").reset_index(drop=True)


def _bbox_spatial_mean(da: xr.DataArray, bbox: dict[str, float]) -> xr.DataArray:
    """Promedia una `DataArray` sobre el bbox (ignora NaN). Maneja convención 0-360 y wrap."""
    lat_name = _find_coord(da, _LAT_NAMES)
    lon_name = _find_coord(da, _LON_NAMES)
    lat = da[lat_name]
    lon = da[lon_name]
    is_360 = float(lon.max()) > 180.0

    def to_conv(value: float) -> float:
        return value % 360 if is_360 else value

    lon_lo, lon_hi = to_conv(bbox["lon_min"]), to_conv(bbox["lon_max"])
    lat_mask = (lat >= bbox["lat_min"]) & (lat <= bbox["lat_max"])
    if lon_lo <= lon_hi:
        lon_mask = (lon >= lon_lo) & (lon <= lon_hi)
    else:  # cruza el antimeridiano tras convertir a 0-360
        lon_mask = (lon >= lon_lo) | (lon <= lon_hi)

    if int((lat_mask & lon_mask).sum()) == 0:
        logger.warning(f"El bbox {bbox} no cae sobre ninguna celda del grid; será NaN.")
    return da.where(lat_mask & lon_mask).mean(dim=[lat_name, lon_name], skipna=True)


def bbox_means(dataset: xr.Dataset, bbox: dict[str, float]) -> pd.DataFrame:
    """Promedio espacial diario de **todas las variables** del dataset dentro de `bbox`.

    Para productos multi-variable (color del océano: CHL, KD490, SPM, ...). Devuelve un
    DataFrame con `ds` + una columna por variable (nombre en minúsculas). Ignora variables
    auxiliares de incertidumbre/flags.
    """
    data_vars = [
        v for v in dataset.data_vars if not str(v).lower().endswith(("_uncertainty", "flags"))
    ]
    if not data_vars:
        raise ValueError("El dataset no tiene variables de datos utilizables.")
    time_name = _find_coord(dataset[data_vars[0]], _TIME_NAMES)
    out = None
    for var in data_vars:
        series = _bbox_spatial_mean(dataset[var], bbox)
        col = pd.to_datetime(series[time_name].to_index()).rename("ds")
        s = pd.DataFrame({"ds": col, str(var).lower(): series.to_numpy()})
        out = s if out is None else out.merge(s, on="ds", how="outer")
    return out.sort_values("ds").reset_index(drop=True)


def oc_series_for_bbox(paths: Iterable[Path], bbox: dict[str, float]) -> pd.DataFrame:
    """Lee los netCDF de color del océano y devuelve un DataFrame diario (ds + variables)."""
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("Sin netCDF de color del océano.")
    merged = None
    for p in paths:
        with xr.open_dataset(p) as ds:
            df = bbox_means(ds, bbox)
        merged = df if merged is None else merged.merge(df, on="ds", how="outer")
    return merged.sort_values("ds").reset_index(drop=True)


def open_oisst(paths: Iterable[Path]) -> xr.Dataset:
    """Abre uno o varios netCDF de OISST como un único Dataset (concatenado en tiempo)."""
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("No se proporcionaron archivos netCDF de OISST.")
    if len(paths) == 1:
        return xr.open_dataset(paths[0])
    return xr.open_mfdataset(paths, combine="by_coords")


def sst_series_for_bbox(
    paths: Iterable[Path],
    bbox: dict[str, float],
    *,
    sst_var: str | None = None,
) -> pd.DataFrame:
    """Lee los netCDF de OISST y devuelve la serie diaria de SST (ds, sst) para el bbox."""
    with open_oisst(paths) as dataset:
        return sst_bbox_mean(dataset, bbox, sst_var=sst_var)


def sst_mhw_for_bbox(
    paths: Iterable[Path],
    bbox: dict[str, float],
    mhw_params: MHWParams,
    *,
    sst_var: str | None = None,
) -> pd.DataFrame:
    """Serie diaria de SST por UE + columnas MHW (sst_anomaly, mhw_category, mhw_intensity)."""
    daily = sst_series_for_bbox(paths, bbox, sst_var=sst_var)
    return add_mhw(daily, mhw_params)
