"""Transformación de avisos de arribo (CONAPESCA) crudo → interim long-tidy.

Toma los CSV que descarga ``etl/extract/arribos_conapesca.py`` y produce una tabla
limpia con granularidad **una fila por `(ds, species, economic_unit)`** (volumen
diario sumado sobre todos los permisionarios de la UE), lista para `aggregate/`
y `consolidate.py`.

Particularidades del crudo CONAPESCA (ver bitácora 2026-04-29 §4):

- Encoding **ISO-8859-1**.
- 4 líneas de título/disclaimer antes del header (header en la línea 5).
- ~35 columnas; nosotros seleccionamos por nombre, así que columnas extra se ignoran.
- Columnas de interés:
    ``PERIODO FIN``                     → ``ds``
    ``PESO DESEMBARCADO_KILOGRAMOS``    → ``y``  (kg)
    ``NOMBRE ESPECIE``                  → ``species`` (texto crudo, se mapea)
    ``UNIDAD ECONOMICA``                → ``economic_unit`` (texto crudo, se mapea)
    ``NOMBRE ESTADO`` / ``LITORAL``     → contexto geográfico (no usado aún; la
                                          ``region`` canónica se deriva del mapping UE)

El separador real del CSV no se ha podido verificar contra el archivo de 150 MB
(no está en el repo). Se asume coma; es overridable vía ``sep``. Si CONAPESCA
publica con otro separador, el QC de columnas faltantes lo detecta de inmediato.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml
from loguru import logger

#: Número de líneas de preámbulo antes del header en los CSV de CONAPESCA.
CONAPESCA_PREAMBLE_LINES = 4
CONAPESCA_ENCODING = "iso-8859-1"

#: Nombres crudos de las columnas de interés en el CSV de CONAPESCA.
COL_DS = "PERIODO FIN"
COL_Y = "PESO DESEMBARCADO_KILOGRAMOS"
COL_SPECIES = "NOMBRE ESPECIE"
COL_UE = "UNIDAD ECONOMICA"

#: Columnas de salida de la tabla interim long-tidy.
OUTPUT_COLUMNS = ["ds", "y", "species", "economic_unit", "region"]


def normalize_text(text: str) -> str:
    """Normaliza texto para comparación robusta entre fuentes/años.

    Mayúsculas + sin acentos (NFKD) + espacios internos colapsados + strip.
    Así ``"ABULÓN AZUL"`` y ``"ABULON  AZUL "`` colapsan al mismo valor.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    no_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(no_accents.upper().split())


def build_species_lookup(species_mapping: dict) -> dict[str, str]:
    """Construye ``{alias_normalizado: code}`` desde el contenido de species_mapping.yaml."""
    lookup: dict[str, str] = {}
    for entry in species_mapping.get("mappings", []):
        code = entry["code"]
        for alias in entry.get("aliases", []):
            key = normalize_text(alias)
            if key in lookup and lookup[key] != code:
                logger.warning(
                    f"Alias de especie duplicado {alias!r}: ya mapea a "
                    f"{lookup[key]!r}, se ignora {code!r}."
                )
                continue
            lookup[key] = code
    return lookup


def build_ue_lookup(economic_units: dict) -> dict[str, tuple[str, str | None]]:
    """Construye ``{nombre_UE_normalizado: (code, region)}`` desde economic_units.yaml."""
    lookup: dict[str, tuple[str, str | None]] = {}
    for code, entry in economic_units.items():
        name = entry.get("name")
        if not name:
            logger.warning(f"UE {code!r} sin campo 'name'; no se podrá mapear desde el crudo.")
            continue
        lookup[normalize_text(name)] = (code, entry.get("region"))
    return lookup


def read_conapesca_csv(
    path: Path,
    *,
    sep: str = ",",
    encoding: str = CONAPESCA_ENCODING,
    preamble_lines: int = CONAPESCA_PREAMBLE_LINES,
) -> pd.DataFrame:
    """Lee un CSV crudo de CONAPESCA respetando preámbulo y encoding.

    Pura respecto a la lógica de negocio: solo lee bytes → DataFrame crudo (sin mapear).
    """
    df = pd.read_csv(
        path,
        sep=sep,
        encoding=encoding,
        skiprows=preamble_lines,
        dtype=str,
        low_memory=False,
    )
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in (COL_DS, COL_Y, COL_SPECIES, COL_UE) if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name}: faltan columnas esperadas {missing}. "
            f"Columnas presentes: {list(df.columns)[:10]}... "
            "Revisar separador/encoding/preámbulo."
        )
    return df


def clean_arribos(
    raw: pd.DataFrame,
    *,
    species_lookup: dict[str, str],
    ue_lookup: dict[str, tuple[str, str | None]],
    keep_species: Iterable[str] | None = None,
    keep_units: Iterable[str] | None = None,
    dayfirst: bool = True,
) -> pd.DataFrame:
    """Limpia un DataFrame crudo de CONAPESCA a la tabla long-tidy de salida.

    Pasos:
      1. Mapea ``NOMBRE ESPECIE`` → ``species`` y ``UNIDAD ECONOMICA`` → ``economic_unit``
         usando los lookups normalizados. Filas sin mapeo se descartan (con conteo).
      2. Filtra a ``keep_species`` / ``keep_units`` si se proveen (None = sin filtro).
      3. Parsea ``ds`` a fecha y ``y`` a float (kg).
      4. Agrega sumando ``y`` por ``(ds, species, economic_unit, region)``.

    No imputa nada ni introduce ceros. ``region`` se deriva del mapping UE.
    """
    df = raw.copy()
    df["species"] = df[COL_SPECIES].map(lambda s: species_lookup.get(normalize_text(str(s))))
    ue_mapped = df[COL_UE].map(lambda s: ue_lookup.get(normalize_text(str(s))))
    df["economic_unit"] = ue_mapped.map(lambda v: v[0] if v is not None else None)
    df["region"] = ue_mapped.map(lambda v: v[1] if v is not None else None)

    n_total = len(df)
    df = df[df["species"].notna() & df["economic_unit"].notna()]
    logger.debug(
        f"Filas con especie+UE mapeadas: {len(df)}/{n_total} "
        f"({n_total - len(df)} descartadas sin mapeo)."
    )

    if keep_species is not None:
        df = df[df["species"].isin(set(keep_species))]
    if keep_units is not None:
        df = df[df["economic_unit"].isin(set(keep_units))]

    df["ds"] = pd.to_datetime(df[COL_DS], dayfirst=dayfirst, errors="coerce").dt.date
    df["y"] = pd.to_numeric(df[COL_Y], errors="coerce")

    bad_dates = df["ds"].isna().sum()
    if bad_dates:
        logger.warning(f"{bad_dates} fila(s) con fecha no parseable; se descartan.")
    df = df[df["ds"].notna()]

    grouped = (
        df.groupby(["ds", "species", "economic_unit", "region"], dropna=False, observed=True)["y"]
        .sum(min_count=1)
        .reset_index()
    )
    return (
        grouped[OUTPUT_COLUMNS]
        .sort_values(["species", "economic_unit", "ds"])
        .reset_index(drop=True)
    )


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def transform(
    csv_paths: Iterable[Path],
    *,
    species_mapping_path: Path,
    economic_units_path: Path,
    keep_species: Iterable[str] | None = None,
    keep_units: Iterable[str] | None = None,
    out_path: Path | None = None,
    sep: str = ",",
) -> pd.DataFrame:
    """Orquestador: lee varios CSV crudos, los limpia y consolida en una tabla long-tidy.

    Si ``out_path`` se provee, escribe el resultado a Parquet (zstd) y crea el directorio.
    Devuelve siempre el DataFrame consolidado.
    """
    species_lookup = build_species_lookup(_load_yaml(species_mapping_path))
    ue_lookup = build_ue_lookup(_load_yaml(economic_units_path))

    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        logger.info(f"Transformando {path.name}")
        raw = read_conapesca_csv(path, sep=sep)
        frames.append(
            clean_arribos(
                raw,
                species_lookup=species_lookup,
                ue_lookup=ue_lookup,
                keep_species=keep_species,
                keep_units=keep_units,
            )
        )

    if not frames:
        raise ValueError("No se proporcionaron CSV para transformar.")

    combined = pd.concat(frames, ignore_index=True)
    # Un mismo (ds, species, ue) puede aparecer en >1 archivo (solapamiento de años); re-agregar.
    consolidated = (
        combined.groupby(["ds", "species", "economic_unit", "region"], dropna=False, observed=True)[
            "y"
        ]
        .sum(min_count=1)
        .reset_index()
        .sort_values(["species", "economic_unit", "ds"])
        .reset_index(drop=True)
    )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        consolidated.to_parquet(out_path, compression="zstd", index=False)
        logger.info(f"Escrito {len(consolidated)} filas → {out_path}")

    return consolidated
