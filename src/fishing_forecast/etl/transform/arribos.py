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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml
from loguru import logger

#: Columnas de salida de la tabla interim long-tidy.
OUTPUT_COLUMNS = ["ds", "y", "species", "economic_unit", "region"]


@dataclass(frozen=True)
class ArribosDialect:
    """Cómo leer un CSV de arribos de una fuente concreta.

    Las dos fuentes (CONAPESCA descargado vs export COBI) traen el mismo esquema lógico
    con distinto encoding, casing de columnas, preámbulo y formato de fecha.
    """

    name: str
    col_ds: str
    col_y: str
    col_species: str
    col_ue: str
    encoding: str = "utf-8"
    preamble_lines: int = 0
    sep: str = ","
    dayfirst: bool = False


#: CSV crudo descargado de CONAPESCA: ISO-8859-1, 4 líneas de preámbulo, fechas DD/MM/YYYY.
CONAPESCA_DIALECT = ArribosDialect(
    name="conapesca",
    col_ds="PERIODO FIN",
    col_y="PESO DESEMBARCADO_KILOGRAMOS",
    col_species="NOMBRE ESPECIE",
    col_ue="UNIDAD ECONOMICA",
    encoding="iso-8859-1",
    preamble_lines=4,
    sep=",",
    dayfirst=True,
)

#: Export COBI (`Arribos2017-2021.csv`): UTF-8, sin preámbulo, columnas snake_case,
#: fechas ISO `YYYY-MM-DD`. Mismo esquema lógico que CONAPESCA, ya pre-parseado.
COBI_DIALECT = ArribosDialect(
    name="cobi",
    col_ds="periodo_fin",
    col_y="peso_desembarcado",
    col_species="especie",
    col_ue="unidad_economica",
    encoding="utf-8",
    preamble_lines=0,
    sep=",",
    dayfirst=False,
)

#: Dialectos disponibles por nombre (para el CLI `--source`).
DIALECTS: dict[str, ArribosDialect] = {
    CONAPESCA_DIALECT.name: CONAPESCA_DIALECT,
    COBI_DIALECT.name: COBI_DIALECT,
}


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


def read_source_csv(path: Path, dialect: ArribosDialect) -> pd.DataFrame:
    """Lee un CSV crudo de arribos según ``dialect`` (encoding/preámbulo/separador).

    Pura respecto a la lógica de negocio: solo lee bytes → DataFrame crudo (sin mapear).
    Revienta con mensaje claro si faltan las columnas clave (separador/encoding malos).
    """
    df = pd.read_csv(
        path,
        sep=dialect.sep,
        encoding=dialect.encoding,
        skiprows=dialect.preamble_lines,
        dtype=str,
        low_memory=False,
    )
    df.columns = [c.strip() for c in df.columns]
    required = (dialect.col_ds, dialect.col_y, dialect.col_species, dialect.col_ue)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name}: faltan columnas esperadas {missing} (dialecto {dialect.name!r}). "
            f"Columnas presentes: {list(df.columns)[:10]}... "
            "Revisar separador/encoding/preámbulo."
        )
    return df


def read_conapesca_csv(
    path: Path,
    *,
    sep: str = ",",
    encoding: str = CONAPESCA_DIALECT.encoding,
    preamble_lines: int = CONAPESCA_DIALECT.preamble_lines,
) -> pd.DataFrame:
    """Wrapper de compatibilidad: lee un CSV con el dialecto CONAPESCA (overrides opcionales)."""
    dialect = replace(
        CONAPESCA_DIALECT, sep=sep, encoding=encoding, preamble_lines=preamble_lines
    )
    return read_source_csv(path, dialect)


def clean_arribos(
    raw: pd.DataFrame,
    *,
    species_lookup: dict[str, str],
    ue_lookup: dict[str, tuple[str, str | None]],
    keep_species: Iterable[str] | None = None,
    keep_units: Iterable[str] | None = None,
    dialect: ArribosDialect = CONAPESCA_DIALECT,
) -> pd.DataFrame:
    """Limpia un DataFrame crudo de arribos a la tabla long-tidy de salida.

    Pasos:
      1. Mapea la especie y la UE crudas → códigos internos con los lookups normalizados.
         Filas sin mapeo se descartan (con conteo).
      2. Filtra a ``keep_species`` / ``keep_units`` si se proveen (None = sin filtro).
      3. Parsea ``ds`` a fecha y ``y`` a float (kg) según ``dialect``.
      4. Agrega sumando ``y`` por ``(ds, species, economic_unit, region)``.

    No imputa nada ni introduce ceros. ``region`` se deriva del mapping UE.
    """
    df = raw.copy()
    df["species"] = df[dialect.col_species].map(
        lambda s: species_lookup.get(normalize_text(str(s)))
    )
    ue_mapped = df[dialect.col_ue].map(lambda s: ue_lookup.get(normalize_text(str(s))))
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

    df["ds"] = pd.to_datetime(df[dialect.col_ds], dayfirst=dialect.dayfirst, errors="coerce").dt.date
    df["y"] = pd.to_numeric(df[dialect.col_y], errors="coerce")

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
    dialect: ArribosDialect = CONAPESCA_DIALECT,
) -> pd.DataFrame:
    """Orquestador: lee varios CSV crudos, los limpia y consolida en una tabla long-tidy.

    ``dialect`` selecciona la fuente (CONAPESCA o COBI). Si ``out_path`` se provee, escribe
    el resultado a Parquet (zstd). Devuelve siempre el DataFrame consolidado.
    """
    species_lookup = build_species_lookup(_load_yaml(species_mapping_path))
    ue_lookup = build_ue_lookup(_load_yaml(economic_units_path))

    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        logger.info(f"Transformando {path.name} (dialecto {dialect.name})")
        raw = read_source_csv(path, dialect)
        frames.append(
            clean_arribos(
                raw,
                species_lookup=species_lookup,
                ue_lookup=ue_lookup,
                keep_species=keep_species,
                keep_units=keep_units,
                dialect=dialect,
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
