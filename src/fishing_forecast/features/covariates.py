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

import numpy as np
import pandas as pd

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
