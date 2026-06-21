"""Transformación de los ``.dat`` CICESE crudo → interim diario por estación.

Los ``.dat`` de REDMAR **no tienen header**, vienen delimitados por espacios y traen
23 columnas en orden fijo (los nombres salen de la metadata de CICESE, ver el script
legacy `etl/cicese.py`). Esta transformación:

1. Lee y concatena los ``.dat`` de una estación (muestras por minuto).
2. Agrega a granularidad **diaria** por mediana (robusta a outliers de los sensores).
3. Renombra las columnas crudas (español) a los códigos en inglés de
   `configs/cicese_stations.yaml: daily_aggregates`.
4. Construye `ds` y etiqueta `station` / `region`.

Salida: `data/interim/cicese/<station>.parquet` (ancho: `ds, station, region, <vars>`).

**Pendiente de verificar con datos reales**: el valor centinela de "dato faltante" de
REDMAR (¿9999? ¿-99999?). Hasta confirmarlo, `na_values` es un parámetro explícito
(default None = no se sustituye nada) para no contaminar la mediana con supuestos. Ver
`PENDINGS.md`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from loguru import logger

#: Orden fijo de las 23 columnas de los ``.dat`` (sin header). Fuente: metadata CICESE.
CICESE_COLUMNS: tuple[str, ...] = (
    "anio",
    "mes",
    "dia",
    "hora",
    "minuto",
    "segundo",
    "id_estacion",
    "voltaje_sistema",
    "nivel_mar_leveltrol",
    "nivel_mar_burbujeador",
    "sw_1",
    "sw_2",
    "temperatura_agua",
    "nivel_mar_ott_rsl",
    "radiacion_solar",
    "direccion_viento",
    "magnitud_viento",
    "temperatura_aire",
    "humedad_relativa",
    "presion_atmosferica",
    "precipitacion",
    "voltaje_estacion_met",
    "nivel_mar_sutron",
)

#: Mapeo columna cruda (español) → código en inglés (`daily_aggregates` del YAML).
RAW_TO_AGGREGATE: dict[str, str] = {
    "nivel_mar_leveltrol": "sea_level_leveltrol",
    "nivel_mar_burbujeador": "sea_level_burbujeador",
    "nivel_mar_ott_rsl": "sea_level_ott_rsl",
    "nivel_mar_sutron": "sea_level_sutron",
    "temperatura_agua": "water_temperature",
    "radiacion_solar": "solar_radiation",
    "direccion_viento": "wind_direction",
    "magnitud_viento": "wind_magnitude",
    "temperatura_aire": "air_temperature",
    "humedad_relativa": "relative_humidity",
    "presion_atmosferica": "atmospheric_pressure",
    "precipitacion": "precipitation",
}


def read_dat(path: Path, *, na_values: Sequence[str | float] | None = None) -> pd.DataFrame:
    """Lee un ``.dat`` CICESE (sin header, separado por espacios) con nombres de columna."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=CICESE_COLUMNS,
        na_values=list(na_values) if na_values is not None else None,
        engine="python",
    )
    return df


def to_daily(
    df: pd.DataFrame,
    *,
    station: str,
    region: str | None,
    aggregates: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Agrega las muestras a mediana diaria y renombra a los códigos en inglés.

    `aggregates` filtra qué variables (en código inglés) conservar; None = todas las
    mapeadas en `RAW_TO_AGGREGATE` que existan en el DataFrame.
    """
    present_raw = [c for c in RAW_TO_AGGREGATE if c in df.columns]
    numeric = df[["anio", "mes", "dia", *present_raw]].apply(pd.to_numeric, errors="coerce")
    daily = numeric.groupby(["anio", "mes", "dia"], as_index=False).median()

    daily["ds"] = pd.to_datetime(
        dict(year=daily["anio"], month=daily["mes"], day=daily["dia"]), errors="coerce"
    ).dt.date
    daily = daily[daily["ds"].notna()].drop(columns=["anio", "mes", "dia"])
    daily = daily.rename(columns=RAW_TO_AGGREGATE)

    keep = list(aggregates) if aggregates is not None else list(RAW_TO_AGGREGATE.values())
    keep = [c for c in keep if c in daily.columns]
    daily["station"] = station
    daily["region"] = region
    ordered = ["ds", "station", "region", *keep]
    return daily[ordered].sort_values("ds").reset_index(drop=True)


def transform(
    dat_paths: Iterable[Path],
    *,
    station: str,
    region: str | None = None,
    aggregates: Iterable[str] | None = None,
    na_values: Sequence[str | float] | None = None,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """Lee todos los ``.dat`` de una estación, agrega a diario y (opcional) escribe parquet."""
    dat_paths = [Path(p) for p in dat_paths]
    if not dat_paths:
        raise ValueError(f"Sin archivos .dat para la estación {station!r}.")

    frames = [read_dat(p, na_values=na_values) for p in dat_paths]
    combined = pd.concat(frames, ignore_index=True)
    daily = to_daily(combined, station=station, region=region, aggregates=aggregates)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        daily.to_parquet(out_path, compression="zstd", index=False)
        logger.info(f"{station}: {len(daily)} días → {out_path}")
    return daily
