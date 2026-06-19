"""Índice de olas de calor marinas (MHW) según Hobday et al. (2016, 2018).

Implementación propia (la librería `marineHeatWaves` no instala en el entorno y es
poco mantenida). Es **pura respecto a la fuente de SST**: recibe una serie diaria de
SST y devuelve las columnas del esquema (`sst_anomaly`, `mhw_category`,
`mhw_intensity`). Quién produce la SST (NOAA OISST v2.1 promediada por bbox de UE, o
Copernicus L4) es responsabilidad de la capa de extracción/agregación oceanográfica.

Algoritmo (ver `docs/etl_design.md` §5.3):

1. **Climatología diaria** sobre el baseline (por defecto 1982-2011, 30 años):
   para cada día-del-año se agrupan los valores de SST dentro de una ventana de
   ±`window_half_width` días (a través de todos los años del baseline) y se calcula
   la media (`clim`) y el percentil 90 (`thresh`). Ambas curvas se suavizan con una
   media móvil circular de `smooth_window` días.
2. **Detección de eventos**: días con `SST ≥ thresh`; un evento requiere ≥
   `min_duration` días consecutivos. Eventos separados por huecos ≤ `max_gap` días se
   fusionan en uno solo.
3. **Categorización** (Hobday et al. 2018) por múltiplos de `(thresh - clim)`:
   I/Moderado [1,2), II/Fuerte [2,3), III/Severo [3,4), IV/Extremo ≥4.

Salida por día:
- `sst_anomaly`  = `SST - clim`  (siempre, incluso negativa).
- `mhw_category` = 0 sin evento; 1..4 dentro de un evento (decisión §5.4).
- `mhw_intensity` = `sst_anomaly` dentro de un evento; `NaN` fuera.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from loguru import logger

#: Año bisiesto de referencia para alinear el día-del-año en una rejilla fija de 366.
#: Así el 1-mar siempre es el día 61 y el 29-feb el día 60, sin desfasarse entre
#: años bisiestos y no bisiestos (el problema del `dayofyear` crudo de pandas).
_REF_LEAP_YEAR = 2000
_N_YEAR_DAYS = 366

#: Nombres de columnas de salida (esquema de dataset_v1, §4.1).
COL_ANOMALY = "sst_anomaly"
COL_CATEGORY = "mhw_category"
COL_INTENSITY = "mhw_intensity"


@dataclass(frozen=True)
class MHWParams:
    """Parámetros del detector de MHW."""

    baseline_start: date
    baseline_end: date
    percentile: float = 90.0
    window_half_width: int = 5
    smooth_window: int = 31
    min_duration: int = 5
    max_gap: int = 2

    @classmethod
    def from_config(cls, mhw_cfg: dict) -> MHWParams:
        """Construye los parámetros desde el bloque `mhw:` de `configs/etl.yaml`."""
        baseline = mhw_cfg.get("baseline", {})
        return cls(
            baseline_start=_as_date(baseline["start"]),
            baseline_end=_as_date(baseline["end"]),
            percentile=float(mhw_cfg.get("percentile", 90.0)),
            window_half_width=int(mhw_cfg.get("window_half_width_days", 5)),
            smooth_window=int(mhw_cfg.get("smoothing_window_days", 31)),
            min_duration=int(mhw_cfg.get("min_duration_days", 5)),
            max_gap=int(mhw_cfg.get("max_gap_days", 2)),
        )


def _as_date(value: object) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def year_day(d: date) -> int:
    """Día-del-año en rejilla fija de 366 (29-feb=60, 1-mar=61 siempre)."""
    return (date(_REF_LEAP_YEAR, d.month, d.day) - date(_REF_LEAP_YEAR, 1, 1)).days + 1


def category_from_ratio(ratio: float) -> int:
    """Categoría MHW (0-4) dado el ratio `(SST - clim) / (thresh - clim)`.

    `ratio < 1` → 0 (no alcanza el umbral). [1,2)→1, [2,3)→2, [3,4)→3, ≥4→4.
    NaN → 0.
    """
    if not np.isfinite(ratio) or ratio < 1.0:
        return 0
    return int(min(np.floor(ratio), 4))


def _circular_running_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Media móvil circular (wrap en los extremos del año). `window` debe ser impar."""
    if window <= 1:
        return values
    half = window // 2
    padded = np.concatenate([values[-half:], values, values[:half]])
    kernel = np.ones(window) / window
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: len(values)]


def compute_climatology(sst: pd.Series, params: MHWParams) -> pd.DataFrame:
    """Climatología diaria (media y umbral p90) sobre el baseline.

    `sst` es una serie indexada por fecha (DatetimeIndex o índice de `date`).
    Devuelve un DataFrame de 366 filas indexado por `year_day` (1..366) con columnas
    `clim` (media) y `thresh` (percentil).
    """
    idx = pd.to_datetime(sst.index)
    mask = (idx.date >= params.baseline_start) & (idx.date <= params.baseline_end)
    base = sst[mask]
    if base.empty:
        raise ValueError(
            f"No hay SST dentro del baseline {params.baseline_start}..{params.baseline_end}."
        )

    by_yd: dict[int, list[float]] = defaultdict(list)
    for ts, val in zip(pd.to_datetime(base.index), base.to_numpy(), strict=True):
        if np.isfinite(val):
            by_yd[year_day(ts.date())].append(float(val))

    n_years = params.baseline_end.year - params.baseline_start.year + 1
    if n_years < 30:
        logger.warning(
            f"Baseline de {n_years} años (<30): la climatología MHW puede ser inestable."
        )

    clim = np.full(_N_YEAR_DAYS, np.nan)
    thresh = np.full(_N_YEAR_DAYS, np.nan)
    w = params.window_half_width
    for target in range(1, _N_YEAR_DAYS + 1):
        pool: list[float] = []
        for off in range(-w, w + 1):
            src = ((target - 1 + off) % _N_YEAR_DAYS) + 1
            pool.extend(by_yd.get(src, []))
        if pool:
            clim[target - 1] = float(np.mean(pool))
            thresh[target - 1] = float(np.percentile(pool, params.percentile))

    if np.isnan(clim).any():
        logger.warning(
            f"{int(np.isnan(clim).sum())} día(s)-del-año sin datos de baseline; "
            "se interpolan circularmente."
        )
        clim = _fill_circular(clim)
        thresh = _fill_circular(thresh)

    clim = _circular_running_mean(clim, params.smooth_window)
    thresh = _circular_running_mean(thresh, params.smooth_window)
    return pd.DataFrame(
        {"clim": clim, "thresh": thresh},
        index=pd.RangeIndex(1, _N_YEAR_DAYS + 1, name="year_day"),
    )


def _fill_circular(values: np.ndarray) -> np.ndarray:
    """Rellena NaN por interpolación lineal circular (para días-del-año sin baseline)."""
    n = len(values)
    if not np.isnan(values).any():
        return values
    extended = np.concatenate([values, values, values])
    s = pd.Series(extended).interpolate(limit_direction="both").to_numpy()
    return s[n : 2 * n]


def _merge_runs(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    """Fusiona corridas (start, end) inclusivas separadas por huecos ≤ `max_gap`."""
    if not runs:
        return []
    merged = [runs[0]]
    for start, end in runs[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end - 1 <= max_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def _exceedance_runs(exceed: np.ndarray) -> list[tuple[int, int]]:
    """Corridas maximales de índices consecutivos donde `exceed` es True."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(exceed):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(exceed) - 1))
    return runs


def add_mhw(
    daily: pd.DataFrame,
    params: MHWParams,
    *,
    ds_col: str = "ds",
    sst_col: str = "sst",
    return_diagnostics: bool = False,
) -> pd.DataFrame:
    """Agrega columnas MHW a un DataFrame diario de SST de **una sola serie** (una UE).

    `daily` debe tener una columna de fecha (`ds_col`) y una de SST (`sst_col`). Se
    reindexa internamente a un rango diario continuo para que la detección de días
    consecutivos sea correcta aunque haya huecos. Devuelve `daily` (orden y filas
    originales) con `sst_anomaly`, `mhw_category` (int8) y `mhw_intensity` añadidas.

    Si `return_diagnostics=True`, incluye además `clim`, `thresh` e `in_mhw` (útil para
    `reports/figures/mhw_timeline.png`).
    """
    if daily.empty:
        raise ValueError("DataFrame de SST vacío.")

    series = (
        daily[[ds_col, sst_col]]
        .assign(**{ds_col: pd.to_datetime(daily[ds_col])})
        .set_index(ds_col)[sst_col]
        .astype(float)
    )
    if series.index.has_duplicates:
        raise ValueError("Fechas duplicadas en la serie de SST; agrega antes de llamar.")

    clim_df = compute_climatology(series, params)

    full_idx = pd.date_range(series.index.min(), series.index.max(), freq="D")
    sst = series.reindex(full_idx)
    yds = np.array([year_day(ts.date()) for ts in full_idx])
    clim = clim_df["clim"].to_numpy()[yds - 1]
    thresh = clim_df["thresh"].to_numpy()[yds - 1]

    anomaly = sst.to_numpy() - clim
    exceed = np.nan_to_num(sst.to_numpy() >= thresh, nan=False).astype(bool)

    runs = _merge_runs(_exceedance_runs(exceed), params.max_gap)
    in_mhw = np.zeros(len(full_idx), dtype=bool)
    for start, end in runs:
        if end - start + 1 >= params.min_duration:
            in_mhw[start : end + 1] = True

    denom = thresh - clim
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denom > 0, anomaly / denom, np.nan)
    category = np.zeros(len(full_idx), dtype=np.int8)
    for i in np.flatnonzero(in_mhw):
        category[i] = max(category_from_ratio(ratio[i]), 1)
    intensity = np.where(in_mhw, anomaly, np.nan)

    result = pd.DataFrame(
        {
            COL_ANOMALY: anomaly,
            COL_CATEGORY: category,
            COL_INTENSITY: intensity,
        },
        index=full_idx,
    )
    if return_diagnostics:
        result["clim"] = clim
        result["thresh"] = thresh
        result["in_mhw"] = in_mhw

    # Reindexar a las fechas originales (descarta los días rellenados del rango continuo).
    out = daily.copy()
    aligned = result.reindex(pd.to_datetime(daily[ds_col]).to_numpy())
    for col in result.columns:
        out[col] = aligned[col].to_numpy()
    out[COL_CATEGORY] = out[COL_CATEGORY].astype(np.int8)
    return out
