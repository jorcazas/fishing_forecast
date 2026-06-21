"""Quality checks sobre el dataset consolidado (`docs/etl_design.md` §4, §6.1).

Filosofía: **fallar ruidosamente** en problemas estructurales (duplicados, tipos,
rangos imposibles) y **advertir** en problemas blandos (cobertura oceanográfica baja,
filas fuera de temporada con `y≠0`). Nada de `try/except: pass`.

`check_dataset` es pura: devuelve la lista de incidencias. `run_quality_checks` aplica
la política (levanta `QualityCheckError` si hay errores, o también si hay warnings y
`fail_on_warning=True`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import pandas as pd
from loguru import logger

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class QCIssue:
    """Una incidencia detectada por los checks."""

    level: Severity
    check: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.level.upper()}] {self.check}: {self.detail}"


class QualityCheckError(RuntimeError):
    """Se levanta cuando los quality checks no pasan según la política configurada."""


_GROUP_KEYS = ["ds", "species", "economic_unit"]


def check_dataset(
    df: pd.DataFrame,
    *,
    known_species: Iterable[str] | None = None,
    known_units: Iterable[str] | None = None,
    ocean_coverage_min: float = 0.80,
) -> list[QCIssue]:
    """Corre todos los checks y devuelve la lista de incidencias (pura)."""
    issues: list[QCIssue] = []

    missing_cols = [k for k in _GROUP_KEYS if k not in df.columns]
    if missing_cols:
        issues.append(QCIssue("error", "schema", f"faltan columnas clave {missing_cols}"))
        return issues  # sin las claves no tiene sentido seguir

    # Duplicados de la clave primaria.
    dup = int(df.duplicated(_GROUP_KEYS).sum())
    if dup:
        issues.append(QCIssue("error", "duplicates", f"{dup} fila(s) con clave (ds,species,UE) repetida"))

    # y no negativa (NaN permitido).
    if "y" in df.columns:
        neg = int((df["y"] < 0).sum())
        if neg:
            issues.append(QCIssue("error", "y_range", f"{neg} fila(s) con y<0"))

    # mhw_category en 0..4.
    if "mhw_category" in df.columns:
        bad_cat = int((~df["mhw_category"].isin([0, 1, 2, 3, 4])).sum())
        if bad_cat:
            issues.append(QCIssue("error", "mhw_category", f"{bad_cat} fila(s) fuera de 0..4"))

    # Especies / UEs conocidas.
    if known_species is not None and "species" in df.columns:
        unknown = sorted(set(df["species"]) - set(known_species))
        if unknown:
            issues.append(QCIssue("error", "species_domain", f"especies no reconocidas: {unknown}"))
    if known_units is not None and "economic_unit" in df.columns:
        unknown = sorted(set(df["economic_unit"]) - set(known_units))
        if unknown:
            issues.append(QCIssue("error", "unit_domain", f"UEs no reconocidas: {unknown}"))

    # season / in_season presentes y bien tipadas.
    if "in_season" in df.columns and df["in_season"].dtype != bool:
        issues.append(QCIssue("error", "in_season_type", f"in_season no es bool ({df['in_season'].dtype})"))
    if "season" in df.columns and int(df["season"].isna().sum()):
        issues.append(QCIssue("error", "season_null", f"{int(df['season'].isna().sum())} season nulas"))

    # Fuera de temporada → y debe ser 0 (warning: puede haber excepciones legítimas).
    if {"in_season", "y"} <= set(df.columns):
        offending = df[(~df["in_season"]) & (df["y"].fillna(0) != 0)]
        if len(offending):
            issues.append(
                QCIssue("warning", "y_out_of_season", f"{len(offending)} fila(s) fuera de temporada con y≠0")
            )

    # Cobertura oceanográfica (fracción de SST no nula).
    if "sst" in df.columns:
        coverage = float(df["sst"].notna().mean())
        if coverage < ocean_coverage_min:
            issues.append(
                QCIssue(
                    "warning",
                    "ocean_coverage",
                    f"cobertura SST {coverage:.1%} < umbral {ocean_coverage_min:.0%}",
                )
            )

    return issues


def check_sst_correlation(
    oisst: pd.DataFrame,
    cicese: pd.DataFrame,
    *,
    min_corr: float = 0.70,
    oisst_col: str = "sst",
    cicese_col: str = "water_temperature",
    ds_col: str = "ds",
    min_overlap: int = 30,
) -> QCIssue | None:
    """Valida que la SST de OISST correlacione con la temperatura in-situ de CICESE.

    Cross-check entre fuentes (no opera sobre el dataset consolidado, que no trae CICESE).
    Alinea ambas series por `ds`, descarta NaN y calcula la correlación de Pearson.
    Devuelve un `QCIssue` (warning) si el solape es insuficiente o la correlación cae por
    debajo de `min_corr`; `None` si todo bien.
    """
    left = oisst[[ds_col, oisst_col]].rename(columns={oisst_col: "_oisst"})
    right = cicese[[ds_col, cicese_col]].rename(columns={cicese_col: "_cicese"})
    merged = left.merge(right, on=ds_col, how="inner").dropna()
    n = len(merged)
    if n < min_overlap:
        return QCIssue(
            "warning",
            "sst_correlation",
            f"solape OISST∩CICESE insuficiente ({n} días < {min_overlap}); no se evalúa.",
        )
    corr = float(merged["_oisst"].corr(merged["_cicese"]))
    if not pd.notna(corr) or corr < min_corr:
        return QCIssue(
            "warning",
            "sst_correlation",
            f"correlación SST OISST vs CICESE {corr:.2f} < umbral {min_corr:.2f} (n={n}).",
        )
    return None


def run_quality_checks(
    df: pd.DataFrame,
    *,
    known_species: Iterable[str] | None = None,
    known_units: Iterable[str] | None = None,
    ocean_coverage_min: float = 0.80,
    fail_on_warning: bool = False,
) -> list[QCIssue]:
    """Corre los checks, loguea cada incidencia y aplica la política de fallo.

    Levanta `QualityCheckError` si hay errores (siempre) o warnings (si
    `fail_on_warning`). Devuelve la lista de incidencias si pasa.
    """
    issues = check_dataset(
        df,
        known_species=known_species,
        known_units=known_units,
        ocean_coverage_min=ocean_coverage_min,
    )
    for issue in issues:
        (logger.error if issue.level == "error" else logger.warning)(str(issue))

    errors = [i for i in issues if i.level == "error"]
    warnings = [i for i in issues if i.level == "warning"]
    if errors or (fail_on_warning and warnings):
        raise QualityCheckError(
            f"Quality checks fallaron: {len(errors)} error(es), {len(warnings)} warning(s)."
        )
    if not issues:
        logger.info("Quality checks: todo en orden.")
    return issues
