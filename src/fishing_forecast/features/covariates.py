"""Features para el modelo con covariables oceanográficas (Fase 1.4 / adelanto de Fase 2).

Construye una matriz tabular para predecir el volumen diario `y` de **una sola serie**
(`species x economic_unit`) a partir de:

- **Calendario** (determinista en `t`, no es leakage): día-del-año (sin/cos), `in_season`.
- **Lags de `y`**: misma fecha de años previos (365, 730 días) — captura el ciclo anual.
- **Oceanografía desplazada `shift_days`** (convención del proyecto X(t)→Y(t+90d)): la SST,
  anomalía y MHW de ~3 meses antes, más medias rodantes (estado pre-temporada / calor
  acumulado del año previo). El desplazamiento es el mecanismo ecológico: las condiciones
  del océano meses antes condicionan la captura de la temporada.

**Sin leakage**: toda feature oceanográfica es un `shift(>= shift_days)` (de un valor o de
una media rodante que termina en `t` y luego se desplaza), así que nunca usa información de
fechas `> t - shift_days`. Las features de calendario son funciones de la fecha en `t`. Los
lags de `y` son estrictamente del pasado. Hay un test que lo verifica.

Asume que el DataFrame de entrada es **una serie diaria continua** ordenada por fecha (lo
que produce `consolidate` para una `(species, economic_unit)`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

#: Columnas oceanográficas que se desplazan/agregan (SST/MHW + color del océano).
#: Las que no estén en el DataFrame se ignoran, así que es seguro listarlas todas.
OCEAN_COLS = (
    "sst",
    "sst_anomaly",
    "mhw_category",
    "mhw_intensity",
    "chl",
    "kd490",
    "spm",
    "zsd",
    "bbp",
    "cdm",
)

#: Lags de `y` (días). 365/730 = misma época de 1 y 2 años atrás.
Y_LAGS = (365, 730)


def build_covariate_features(
    df: pd.DataFrame,
    *,
    shift_days: int = 90,
    rolling_windows: tuple[int, ...] = (90, 365),
) -> pd.DataFrame:
    """Construye features + target para una serie diaria `(ds, y, in_season, <ocean>, season)`.

    Devuelve un DataFrame con `ds`, `season`, `y` (target) y las columnas de features.
    No imputa `y` (se decide en el experimento). No elimina filas con features NaN
    (los modelos de árbol manejan `NaN`); el experimento decide qué filas usar.
    """
    out = df.sort_values("ds").reset_index(drop=True).copy()
    ds = pd.to_datetime(out["ds"])

    # --- Calendario (conocido en t) ---
    doy = ds.dt.dayofyear.to_numpy()
    feat = pd.DataFrame(index=out.index)
    feat["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    feat["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    feat["in_season"] = out["in_season"].astype(int) if "in_season" in out else 0

    # --- Lags de y (pasado) ---
    for lag in Y_LAGS:
        feat[f"y_lag{lag}"] = out["y"].shift(lag)

    # --- Oceanografía desplazada shift_days ---
    for col in OCEAN_COLS:
        if col not in out.columns:
            continue
        feat[f"{col}_lag{shift_days}"] = out[col].shift(shift_days)
        for w in rolling_windows:
            roll = out[col].rolling(window=w, min_periods=max(5, w // 3)).mean()
            feat[f"{col}_roll{w}_lag{shift_days}"] = roll.shift(shift_days)

    feat["ds"] = out["ds"]
    feat["season"] = out.get("season")
    feat["y"] = out["y"]
    return feat


def feature_columns(feat: pd.DataFrame) -> list[str]:
    """Nombres de columnas de features (todo menos `ds`, `season`, `y` y las de grupo)."""
    return [c for c in feat.columns if c not in ("ds", "season", "y", "species", "economic_unit")]


def build_multiseries_features(
    df: pd.DataFrame,
    *,
    group_col: str | list[str] = "species",
    shift_days: int = 90,
    rolling_windows: tuple[int, ...] = (90, 365),
) -> pd.DataFrame:
    """Construye features para **varias series** apiladas, sin cruzar lags entre grupos.

    `group_col` puede ser una columna (`"species"`) o varias (`["species", "economic_unit"]`)
    para definir la serie. Aplica `build_covariate_features` a cada serie por separado (así
    los lags/rolling nunca cruzan series) y concatena, conservando las columnas de grupo. El
    one-hot de los grupos se hace en el experimento.
    """
    group_cols = [group_col] if isinstance(group_col, str) else list(group_col)
    frames: list[pd.DataFrame] = []
    for key, sub in df.groupby(group_cols, observed=True):
        feat = build_covariate_features(sub, shift_days=shift_days, rolling_windows=rolling_windows)
        key_tuple = key if isinstance(key, tuple) else (key,)
        for col, val in zip(group_cols, key_tuple, strict=True):
            feat[col] = val
        frames.append(feat)
    if not frames:
        raise ValueError(f"Sin grupos en {group_cols}.")
    return pd.concat(frames, ignore_index=True)


# --- Fase 2: anomalías climatológicas, interacciones y configuración por YAML ------------------


@dataclass(frozen=True)
class FeatureConfig:
    """Configuración del feature engineering (ver `configs/features.yaml`)."""

    shift_days: int = 90
    rolling_windows: tuple[int, ...] = (90, 365)
    anomalies: bool = False
    anomaly_columns: tuple[str, ...] = ()
    anomaly_smooth_window: int = 15
    interactions: tuple[tuple[str, str], ...] = ()


def load_feature_config(path: str | Path | None = None) -> FeatureConfig:
    """Lee `configs/features.yaml` (o `path`) y devuelve la configuración tipada."""
    import yaml

    from fishing_forecast.config import get_settings

    src = Path(path) if path else get_settings().configs_root / "features.yaml"
    cfg = yaml.safe_load(src.read_text("utf-8")) or {}
    anom = cfg.get("anomalies") or {}
    inter = cfg.get("interactions") or {}
    pairs = tuple(tuple(p) for p in (inter.get("pairs") or [])) if inter.get("enabled") else ()
    return FeatureConfig(
        shift_days=int(cfg.get("shift_days", 90)),
        rolling_windows=tuple(cfg.get("rolling_windows") or (90, 365)),
        anomalies=bool(anom.get("enabled", False)),
        anomaly_columns=tuple(anom.get("columns") or ()),
        anomaly_smooth_window=int(anom.get("smooth_window", 15)),
        interactions=pairs,
    )


def fit_climatology(
    df: pd.DataFrame,
    columns: tuple[str, ...] | list[str],
    *,
    train_end: pd.Timestamp | str,
    smooth_window: int = 15,
) -> pd.DataFrame:
    """Climatología por día-del-año estimada **solo con datos anteriores a `train_end`**.

    Devuelve un DataFrame indexado por `doy` (1-366) con una columna por variable: la media
    histórica de ese día del año, suavizada con una ventana circular de `smooth_window` días
    (la media cruda por doy con pocas temporadas es ruidosa).

    `train_end` es obligatorio y explícito justamente para que el anti-leakage sea auditable:
    la climatología es un estadístico agregado y calcularlo con años de test contaminaría todas
    las anomalías. Devuelve solo las columnas presentes en `df`.
    """
    train = df[pd.to_datetime(df["ds"]) < pd.Timestamp(train_end)]
    cols = [c for c in columns if c in df.columns]
    if train.empty or not cols:
        return pd.DataFrame(index=pd.RangeIndex(1, 367, name="doy"))
    doy = pd.to_datetime(train["ds"]).dt.dayofyear
    clim = train[cols].groupby(doy).mean()
    clim = clim.reindex(pd.RangeIndex(1, 367, name="doy"))
    # Suavizado circular: se concatena el año consigo mismo para que diciembre y enero se vean.
    tripled = pd.concat([clim, clim, clim])
    smoothed = tripled.rolling(window=smooth_window, center=True, min_periods=1).mean()
    out = smoothed.iloc[len(clim) : 2 * len(clim)]
    out.index = clim.index
    return out


def add_climatology_anomalies(
    feat: pd.DataFrame,
    raw: pd.DataFrame,
    climatology: pd.DataFrame,
    *,
    shift_days: int = 90,
) -> pd.DataFrame:
    """Añade `{col}_anom_lag{shift}` = (valor menos la climatología de su día-del-año), desplazado.

    La anomalía se calcula sobre la serie cruda y **después** se desplaza `shift_days`, así que
    la feature en `t` solo usa observaciones de `t - shift_days` o antes.
    """
    if climatology.empty:
        return feat
    doy = pd.to_datetime(raw["ds"]).dt.dayofyear.to_numpy()
    for col in climatology.columns:
        if col not in raw.columns:
            continue
        expected = climatology[col].reindex(doy).to_numpy()
        anom = pd.Series(raw[col].to_numpy() - expected, index=raw.index)
        feat[f"{col}_anom_lag{shift_days}"] = anom.shift(shift_days).to_numpy()
    return feat


def add_interactions(
    feat: pd.DataFrame, pairs: tuple[tuple[str, str], ...] | list[list[str]]
) -> pd.DataFrame:
    """Añade el producto `a * b` como `a__x__b` para cada par presente en `feat`.

    Un par cuyas columnas no existan (p. ej. una UE sin color del océano) se omite y se anota en
    el log: es un dato faltante legítimo, no un error, pero no se silencia.
    """
    for a, b in pairs:
        if a not in feat.columns or b not in feat.columns:
            logger.warning(f"interacción {a}*{b} omitida: falta {a if a not in feat else b}")
            continue
        feat[f"{a}__x__{b}"] = feat[a].astype(float) * feat[b].astype(float)
    return feat


def build_features_v2(
    df: pd.DataFrame,
    *,
    config: FeatureConfig | None = None,
    train_end: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """`build_covariate_features` + anomalías climatológicas + interacciones (Fase 2).

    Es aditiva sobre el builder original —las columnas de aquel no cambian— para que los
    experimentos existentes sigan comparándose contra la misma matriz. Las anomalías requieren
    `train_end` (la climatología solo puede ver train); si no se pasa, se omiten con aviso.
    """
    cfg = config or load_feature_config()
    feat = build_covariate_features(
        df, shift_days=cfg.shift_days, rolling_windows=cfg.rolling_windows
    )
    raw = df.sort_values("ds").reset_index(drop=True)
    if cfg.anomalies:
        if train_end is None:
            logger.warning("anomalías omitidas: se requiere `train_end` para la climatología.")
        else:
            clim = fit_climatology(
                raw,
                cfg.anomaly_columns,
                train_end=train_end,
                smooth_window=cfg.anomaly_smooth_window,
            )
            feat = add_climatology_anomalies(feat, raw, clim, shift_days=cfg.shift_days)
    if cfg.interactions:
        feat = add_interactions(feat, cfg.interactions)
    return feat
