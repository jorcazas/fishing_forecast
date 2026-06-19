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
    da = _select_sst_var(dataset, sst_var)
    lat_name = _find_coord(da, _LAT_NAMES)
    lon_name = _find_coord(da, _LON_NAMES)
    time_name = _find_coord(da, _TIME_NAMES)

    lat = da[lat_name]
    lon = da[lon_name]
    is_360 = float(lon.max()) > 180.0

    def to_conv(value: float) -> float:
        return value % 360 if is_360 else value

    lon_lo = to_conv(bbox["lon_min"])
    lon_hi = to_conv(bbox["lon_max"])
    lat_mask = (lat >= bbox["lat_min"]) & (lat <= bbox["lat_max"])
    if lon_lo <= lon_hi:
        lon_mask = (lon >= lon_lo) & (lon <= lon_hi)
    else:  # bbox cruza el antimeridiano tras la conversión a 0-360
        lon_mask = (lon >= lon_lo) | (lon <= lon_hi)

    masked = da.where(lat_mask & lon_mask)
    n_cells = int((lat_mask & lon_mask).sum())
    if n_cells == 0:
        logger.warning(
            f"El bbox {bbox} no cae sobre ninguna celda del grid; SST será NaN. "
            "Revisar coordenadas de la UE."
        )

    series = masked.mean(dim=[lat_name, lon_name], skipna=True)
    df = series.to_dataframe(name="sst").reset_index()[[time_name, "sst"]]
    df = df.rename(columns={time_name: "ds"})
    df["ds"] = pd.to_datetime(df["ds"])
    return df.sort_values("ds").reset_index(drop=True)


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
