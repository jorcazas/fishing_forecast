"""Tests de la consolidación interim → dataset_v1."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fishing_forecast.etl.consolidate import (
    SCHEMA_COLUMNS,
    build_season_lookup,
    consolidate,
    export_lstm_csv,
    write_dataset_partitioned,
)

SEASONS = {"lobster_red": {"litoral_bc_sur": {"start": "09-15", "end": "02-15"}}}


@pytest.fixture
def arribos() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ds": [date(2018, 11, 1), date(2018, 11, 2), date(2019, 3, 1)],
            "y": [100.0, 50.0, 20.0],
            "species": ["lobster_red", "lobster_red", "abalone_blue"],
            "economic_unit": ["litoral_bc_sur", "litoral_bc_sur", "litoral_bc_sur"],
            "region": ["san_quintin", "san_quintin", "san_quintin"],
        }
    )


def test_build_season_lookup() -> None:
    lookup = build_season_lookup(SEASONS)
    assert lookup[("lobster_red", "litoral_bc_sur")] == (9, 15, 2, 15)


def test_consolidate_builds_full_grid(arribos: pd.DataFrame) -> None:
    df = consolidate(
        arribos,
        season_calendars=SEASONS,
        date_start=date(2018, 1, 1),
        date_end=date(2018, 12, 31),
    )
    assert list(df.columns) == SCHEMA_COLUMNS
    # 2 series (lobster_red, abalone_blue) × 365 días.
    assert len(df) == 2 * 365
    assert not df.duplicated(["ds", "species", "economic_unit"]).any()


def test_consolidate_y_merge_and_out_of_season_zero(arribos: pd.DataFrame) -> None:
    df = consolidate(
        arribos,
        season_calendars=SEASONS,
        date_start=date(2018, 1, 1),
        date_end=date(2018, 12, 31),
    )
    lob = df[df["species"] == "lobster_red"].set_index("ds")
    # Registro real conservado.
    assert lob.loc[date(2018, 11, 1), "y"] == 100.0
    # Día fuera de temporada (1-jul) sin registro → y=0, in_season False.
    assert lob.loc[date(2018, 7, 1), "y"] == 0.0
    assert not lob.loc[date(2018, 7, 1), "in_season"]
    # Día dentro de temporada (1-oct) sin registro → y NaN.
    assert np.isnan(lob.loc[date(2018, 10, 1), "y"])
    assert lob.loc[date(2018, 10, 1), "in_season"]


def test_consolidate_undeclared_species_defaults_in_season(arribos: pd.DataFrame) -> None:
    df = consolidate(
        arribos,
        season_calendars=SEASONS,  # abalone_blue no declarado
        date_start=date(2018, 1, 1),
        date_end=date(2018, 12, 31),
    )
    ab = df[df["species"] == "abalone_blue"]
    assert ab["in_season"].all()  # sin calendario → siempre in_season


def test_consolidate_attaches_ocean_broadcast_across_species(arribos: pd.DataFrame) -> None:
    ocean = pd.DataFrame(
        {
            "ds": [date(2018, 11, 1)],
            "sst": [21.0],
            "sst_anomaly": [2.0],
            "mhw_category": [np.int8(2)],
            "mhw_intensity": [2.0],
        }
    )
    df = consolidate(
        arribos,
        season_calendars=SEASONS,
        date_start=date(2018, 1, 1),
        date_end=date(2018, 12, 31),
        ocean_by_ue={"litoral_bc_sur": ocean},
    )
    on_day = df[df["ds"] == date(2018, 11, 1)]
    # La SST se broadcastea a ambas especies de la UE.
    assert set(on_day["species"]) == {"lobster_red", "abalone_blue"}
    assert (on_day["sst"] == 21.0).all()
    assert (on_day["mhw_category"] == 2).all()
    # Días sin océano → mhw_category=0 (relleno), sst NaN.
    off = df[df["ds"] == date(2018, 11, 2)]
    assert (off["mhw_category"] == 0).all()
    assert off["sst"].isna().all()


def test_write_dataset_partitioned(arribos: pd.DataFrame, tmp_path) -> None:
    df = consolidate(
        arribos,
        season_calendars=SEASONS,
        date_start=date(2018, 1, 1),
        date_end=date(2019, 1, 31),  # cruza dos años → partición year=2018 y year=2019
    )
    root = tmp_path / "dataset_v1"
    write_dataset_partitioned(df, root)
    assert (root / "species=lobster_red" / "year=2018").exists()
    assert (root / "species=abalone_blue" / "year=2019").exists()
    # Roundtrip: leer todo el dataset particionado conserva el total de filas.
    assert len(pd.read_parquet(root)) == len(df)


def test_export_lstm_csv(arribos: pd.DataFrame, tmp_path) -> None:
    df = consolidate(
        arribos,
        season_calendars=SEASONS,
        date_start=date(2018, 1, 1),
        date_end=date(2018, 12, 31),
    )
    out = export_lstm_csv(df, tmp_path / "lstm_data.csv")
    csv = pd.read_csv(out)
    assert list(csv.columns) == ["ds", "y"]  # sin x1..x16 aún (GlobColour no integrado)
    assert len(csv) == 365  # solo lobster_red × litoral_bc_sur


def test_export_lstm_csv_raises_when_empty(arribos: pd.DataFrame) -> None:
    df = consolidate(
        arribos, season_calendars=SEASONS, date_start=date(2018, 1, 1), date_end=date(2018, 1, 31)
    )
    with pytest.raises(ValueError, match="Sin filas"):
        export_lstm_csv(df, Path("/tmp/never.csv"), species="urchin_red")


def test_consolidate_no_ocean_fills_defaults(arribos: pd.DataFrame) -> None:
    df = consolidate(
        arribos,
        season_calendars=SEASONS,
        date_start=date(2018, 1, 1),
        date_end=date(2018, 1, 31),
    )
    assert (df["mhw_category"] == 0).all()
    assert df["sst"].isna().all()
    assert (df["source_globcolour_files"] == 0).all()
    assert (df["ocean_impute_method"] == "none").all()
    assert df["etl_run_id"].nunique() == 1
