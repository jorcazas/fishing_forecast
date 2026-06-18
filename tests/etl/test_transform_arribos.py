"""Tests de la transformación de arribos CONAPESCA crudo → interim long-tidy.

Cubre normalización de texto, construcción de lookups, lectura del CSV crudo
(ISO-8859-1 con preámbulo), mapeo/filtrado de especies y UEs, y agregación diaria.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml

from fishing_forecast.etl.transform.arribos import (
    build_species_lookup,
    build_ue_lookup,
    clean_arribos,
    normalize_text,
    read_conapesca_csv,
    transform,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SPECIES_MAPPING = REPO_ROOT / "configs" / "species_mapping.yaml"
ECONOMIC_UNITS = REPO_ROOT / "configs" / "economic_units.yaml"

DATASET_V1_SPECIES = [
    "lobster_red",
    "abalone_blue",
    "abalone_red",
    "abalone_black",
    "urchin_red",
]


@pytest.fixture
def sample_csv(fixtures_dir: Path) -> Path:
    return fixtures_dir / "conapesca_arribos_sample.csv"


def test_normalize_text_strips_accents_and_case() -> None:
    assert normalize_text("ABULÓN  AZUL ") == "ABULON AZUL"
    assert normalize_text("Langosta roja") == "LANGOSTA ROJA"


def test_build_species_lookup_maps_accented_and_plain_aliases() -> None:
    lookup = build_species_lookup(
        {
            "mappings": [
                {"code": "abalone_blue", "aliases": ["ABULON AZUL ENT. FCO.", "ABULÓN AZUL"]},
            ]
        }
    )
    assert lookup[normalize_text("ABULON AZUL ENT. FCO.")] == "abalone_blue"
    assert lookup[normalize_text("ABULÓN AZUL")] == "abalone_blue"


def test_build_ue_lookup_carries_region() -> None:
    lookup = build_ue_lookup(
        {
            "litoral_bc_sur": {
                "name": "LITORAL DE BAJA CALIFORNIA S DE PR DE RL",
                "region": "san_quintin",
            }
        }
    )
    code, region = lookup[normalize_text("LITORAL DE BAJA CALIFORNIA S DE PR DE RL")]
    assert code == "litoral_bc_sur"
    assert region == "san_quintin"


def test_read_conapesca_csv_skips_preamble(sample_csv: Path) -> None:
    raw = read_conapesca_csv(sample_csv)
    # 7 filas de datos, header reconocido
    assert len(raw) == 7
    assert "NOMBRE ESPECIE" in raw.columns
    assert "PERIODO FIN" in raw.columns


def test_read_conapesca_csv_raises_on_wrong_separator(sample_csv: Path) -> None:
    with pytest.raises(ValueError, match="faltan columnas"):
        read_conapesca_csv(sample_csv, sep=";")


def test_clean_arribos_maps_filters_and_aggregates(sample_csv: Path) -> None:
    raw = read_conapesca_csv(sample_csv)
    species_lookup = build_species_lookup(
        yaml.safe_load(SPECIES_MAPPING.read_text(encoding="utf-8"))
    )
    ue_lookup = build_ue_lookup(yaml.safe_load(ECONOMIC_UNITS.read_text(encoding="utf-8")))
    out = clean_arribos(
        raw,
        species_lookup=species_lookup,
        ue_lookup=ue_lookup,
        keep_species=DATASET_V1_SPECIES,
        keep_units=["litoral_bc_sur"],
    )

    # TIBURON (sin mapeo) y OTRA COOPERATIVA (UE sin mapeo) se descartan → 4 filas.
    assert list(out.columns) == ["ds", "y", "species", "economic_unit", "region"]
    assert len(out) == 4
    assert set(out["species"]) == {"lobster_red", "abalone_blue", "urchin_red"}
    assert (out["economic_unit"] == "litoral_bc_sur").all()
    assert (out["region"] == "san_quintin").all()

    # Las dos filas de langosta del 15/11/2024 se suman: 120.5 + 80.0 = 200.5
    lobster = out[(out["species"] == "lobster_red") & (out["ds"] == date(2024, 11, 15))]
    assert lobster["y"].iloc[0] == pytest.approx(200.5)

    # abalone_blue aparece en dos días distintos (10 y 11 de marzo), no se colapsan.
    abalone_days = out[out["species"] == "abalone_blue"]["ds"].tolist()
    assert sorted(abalone_days) == [date(2024, 3, 10), date(2024, 3, 11)]


def test_clean_arribos_no_filter_keeps_all_mapped(sample_csv: Path) -> None:
    raw = read_conapesca_csv(sample_csv)
    out = clean_arribos(
        raw,
        species_lookup=build_species_lookup(
            yaml.safe_load(SPECIES_MAPPING.read_text(encoding="utf-8"))
        ),
        ue_lookup=build_ue_lookup(yaml.safe_load(ECONOMIC_UNITS.read_text(encoding="utf-8"))),
    )
    # Sin filtro de UE, la fila OTRA COOPERATIVA sigue sin mapeo de UE → descartada igual.
    # Sin filtro de especie, TIBURON sigue sin mapeo → descartada igual.
    assert set(out["species"]) <= {"lobster_red", "abalone_blue", "urchin_red"}
    assert "OTRA COOPERATIVA NO MAPEADA" not in out["economic_unit"].tolist()


def test_transform_writes_parquet(sample_csv: Path, tmp_path: Path) -> None:
    out_path = tmp_path / "interim" / "arribos.parquet"
    df = transform(
        [sample_csv],
        species_mapping_path=SPECIES_MAPPING,
        economic_units_path=ECONOMIC_UNITS,
        keep_species=DATASET_V1_SPECIES,
        keep_units=["litoral_bc_sur"],
        out_path=out_path,
    )
    assert out_path.exists()
    roundtrip = pd.read_parquet(out_path)
    assert len(roundtrip) == len(df) == 4


def test_transform_deduplicates_across_files(sample_csv: Path) -> None:
    # Pasar el mismo archivo dos veces simula solapamiento de años: y debe sumarse, no duplicar filas.
    df = transform(
        [sample_csv, sample_csv],
        species_mapping_path=SPECIES_MAPPING,
        economic_units_path=ECONOMIC_UNITS,
        keep_species=DATASET_V1_SPECIES,
        keep_units=["litoral_bc_sur"],
    )
    assert len(df) == 4  # mismas 4 combinaciones (ds, species, ue)
    lobster = df[(df["species"] == "lobster_red") & (df["ds"] == date(2024, 11, 15))]
    assert lobster["y"].iloc[0] == pytest.approx(401.0)  # 200.5 sumado dos veces
