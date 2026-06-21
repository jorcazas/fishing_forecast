"""Consolidación interim → `data/processed/dataset_v1.parquet`.

Join final del esquema de `docs/etl_design.md` §4.1: **una fila por
`(ds, species, economic_unit)`** sobre el rango de fechas completo. El "spine" es la
tabla de arribos (`data/interim/arribos.parquet`); a ella se le pega la SST/MHW por UE
(`data/interim/ocean_<ue>.parquet`), que se *broadcastea* a todas las especies de esa
UE (la oceanografía es por UE, no por especie).

No descarga ni re-agrega: solo joinea interim + deriva columnas calendario y de
metadatos. Las `x1..x16` de GlobColour y la SST de Copernicus aún no existen (fuentes
bloqueadas por credenciales); cuando lleguen se suman aquí con el mismo patrón que el
océano OISST. Ver `PENDINGS.md`.

Manejo de `y` faltante (§4.4):
- fuera de temporada (`in_season=False`) sin registro → `y=0`, `is_imputed_y=False`.
- dentro de temporada sin registro → `y=NaN` (no se imputa).
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from fishing_forecast.utils.dates import in_season, season_id

#: Columnas oceanográficas que aporta `aggregate/ocean_by_ue.py`.
OCEAN_COLUMNS = ["sst", "sst_anomaly", "mhw_category", "mhw_intensity"]

#: Orden canónico de columnas del dataset consolidado (§4.1).
SCHEMA_COLUMNS = [
    "ds",
    "y",
    "species",
    "economic_unit",
    "region",
    "sst",
    "sst_anomaly",
    "mhw_category",
    "mhw_intensity",
    "season",
    "in_season",
    "is_imputed_y",
    "is_imputed_x",
    "ocean_impute_method",
    "source_globcolour_files",
    "etl_run_id",
]


def _parse_month_day(value: str) -> tuple[int, int]:
    month, day = value.split("-")
    return int(month), int(day)


def build_season_lookup(season_calendars: dict) -> dict[tuple[str, str], tuple[int, int, int, int]]:
    """`{(species, economic_unit): (start_month, start_day, end_month, end_day)}`."""
    lookup: dict[tuple[str, str], tuple[int, int, int, int]] = {}
    for species, by_ue in (season_calendars or {}).items():
        for ue, cal in by_ue.items():
            sm, sd = _parse_month_day(cal["start"])
            em, ed = _parse_month_day(cal["end"])
            lookup[(species, ue)] = (sm, sd, em, ed)
    return lookup


def build_grid(arribos: pd.DataFrame, date_start: date, date_end: date) -> pd.DataFrame:
    """Rejilla completa `(ds, species, economic_unit, region)` sobre el rango de fechas.

    Las combinaciones `(species, economic_unit, region)` salen de las presentes en los
    arribos (las series que realmente tenemos).
    """
    pairs = arribos[["species", "economic_unit", "region"]].drop_duplicates()
    dates = pd.DataFrame(
        {"ds": pd.date_range(date_start, date_end, freq="D").date}
    )
    grid = pairs.merge(dates, how="cross")
    return grid


def _derive_season(grid: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    """Añade `season` (str) e `in_season` (bool) por grupo `(species, economic_unit)`."""
    out = grid.reset_index(drop=True).copy()
    season = pd.Series(index=out.index, dtype=object)
    in_s = pd.Series(index=out.index, dtype=bool)
    for (sp, ue), sub in out.groupby(["species", "economic_unit"], observed=True):
        cal = lookup.get((sp, ue))
        if cal is None:
            logger.warning(
                f"Sin calendario de temporada para ({sp!r}, {ue!r}); in_season=True por defecto."
            )
            season.loc[sub.index] = [f"{d.year}_{d.year}" for d in sub["ds"]]
            in_s.loc[sub.index] = True
        else:
            sm, sd, em, ed = cal
            season.loc[sub.index] = [season_id(d, sm, sd, em, ed) for d in sub["ds"]]
            in_s.loc[sub.index] = [in_season(d, sm, sd, em, ed) for d in sub["ds"]]
    out["season"] = season
    out["in_season"] = in_s.astype(bool)
    return out


def consolidate(
    arribos: pd.DataFrame,
    *,
    season_calendars: dict | None = None,
    date_start: date,
    date_end: date,
    ocean_by_ue: dict[str, pd.DataFrame] | None = None,
    etl_run_id: str | None = None,
) -> pd.DataFrame:
    """Produce el DataFrame consolidado con el esquema §4.1.

    `arribos` debe tener columnas `ds, y, species, economic_unit, region`.
    `ocean_by_ue` mapea `economic_unit → DataFrame(ds, sst, sst_anomaly, mhw_category,
    mhw_intensity)` (salida de `aggregate/ocean_by_ue.py`). Puede ser None/parcial.
    """
    etl_run_id = etl_run_id or datetime.now().strftime("%Y-%m-%d_%H%M%S")

    arribos = arribos.copy()
    arribos["ds"] = pd.to_datetime(arribos["ds"]).dt.date

    grid = build_grid(arribos, date_start, date_end)
    grid = _derive_season(grid, build_season_lookup(season_calendars or {}))

    merged = grid.merge(
        arribos[["ds", "species", "economic_unit", "y"]],
        on=["ds", "species", "economic_unit"],
        how="left",
    )

    # §4.4: fuera de temporada sin registro → y=0; dentro de temporada → NaN.
    out_of_season_missing = (~merged["in_season"]) & merged["y"].isna()
    merged.loc[out_of_season_missing, "y"] = 0.0
    merged["is_imputed_y"] = False  # nunca imputamos; el 0 fuera de temporada es definicional

    # Oceanografía por UE (broadcast a todas las especies de la UE).
    merged = _attach_ocean(merged, ocean_by_ue)

    merged["is_imputed_x"] = False
    merged["ocean_impute_method"] = "none"
    merged["source_globcolour_files"] = np.int32(0)  # GlobColour aún no integrado
    merged["etl_run_id"] = etl_run_id

    merged = merged[SCHEMA_COLUMNS]
    return merged.sort_values(["species", "economic_unit", "ds"]).reset_index(drop=True)


def _attach_ocean(
    merged: pd.DataFrame, ocean_by_ue: dict[str, pd.DataFrame] | None
) -> pd.DataFrame:
    if not ocean_by_ue:
        for col in OCEAN_COLUMNS:
            merged[col] = np.nan
        merged["mhw_category"] = np.int8(0)
        return merged

    frames: list[pd.DataFrame] = []
    for ue, df in ocean_by_ue.items():
        sub = df.copy()
        sub["ds"] = pd.to_datetime(sub["ds"]).dt.date
        sub["economic_unit"] = ue
        keep = ["ds", "economic_unit"] + [c for c in OCEAN_COLUMNS if c in sub.columns]
        frames.append(sub[keep])
    ocean = pd.concat(frames, ignore_index=True)
    merged = merged.merge(ocean, on=["ds", "economic_unit"], how="left")
    merged["mhw_category"] = merged["mhw_category"].fillna(0).astype(np.int8)
    for col in OCEAN_COLUMNS:
        if col not in merged.columns:
            merged[col] = np.nan
    return merged


def write_dataset(df: pd.DataFrame, out_path: Path) -> Path:
    """Escribe el dataset consolidado a un único Parquet (zstd). Crea el directorio."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression="zstd", index=False)
    logger.info(f"Escrito {len(df)} filas → {out_path}")
    return out_path


def write_dataset_partitioned(df: pd.DataFrame, root_dir: Path) -> Path:
    """Escribe el dataset particionado por `species` y `year(ds)` (§4.2).

    Layout: ``root_dir/species=<sp>/year=<yyyy>/*.parquet`` — los lectores que filtran por
    especie no leen bytes irrelevantes. Devuelve `root_dir`.
    """
    root_dir.mkdir(parents=True, exist_ok=True)
    partitioned = df.copy()
    partitioned["year"] = pd.to_datetime(partitioned["ds"]).dt.year
    partitioned.to_parquet(
        root_dir,
        compression="zstd",
        index=False,
        partition_cols=["species", "year"],
    )
    logger.info(f"Escrito {len(df)} filas particionadas (species×year) → {root_dir}")
    return root_dir


#: Columnas oceanográficas GlobColour del borrador (presentes solo cuando se integre).
_LSTM_X_COLUMNS = [f"x{i}" for i in range(1, 17)]


def export_lstm_csv(
    df: pd.DataFrame,
    out_path: Path,
    *,
    species: str = "lobster_red",
    economic_unit: str = "litoral_bc_sur",
) -> Path:
    """Export de compatibilidad con los scripts del borrador (`ds, y, x1..x16`).

    Filtra a una sola serie (`species` × `economic_unit`) y escribe `ds, y` más las
    `x1..x16` que existan en el dataset (hoy ninguna, hasta integrar GlobColour). Sirve
    para regression tests contra los modelos del borrador 2024.
    """
    subset = df[(df["species"] == species) & (df["economic_unit"] == economic_unit)]
    if subset.empty:
        raise ValueError(f"Sin filas para ({species!r}, {economic_unit!r}); nada que exportar.")
    cols = ["ds", "y"] + [c for c in _LSTM_X_COLUMNS if c in subset.columns]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subset[cols].sort_values("ds").to_csv(out_path, index=False)
    logger.info(f"Export LSTM ({species}×{economic_unit}): {len(subset)} filas → {out_path}")
    return out_path
