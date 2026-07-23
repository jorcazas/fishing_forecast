"""Construcción del *store* de pronóstico calibrado por serie (especie × unidad económica).

Reutiliza los bloques puros de la CQR de producción (Exp 4): rejilla de XGBoost cuantílico en
espacio log, reordenamiento monótono de cuantiles (Chernozhukov 2010) y conformalización
*por serie* (Mondrian, en temporada, escala log). Entrena una vez y devuelve, por serie, los
arreglos diarios (observado, mediana, bandas 80/90%) del periodo de prueba `ds >= cut_date`,
más un resumen por temporada. El servidor cachea el resultado.

La inferencia es sobre el periodo de prueba con oceanografía disponible: no se pronostica la
oceanografía futura, así que el horizonte llega hasta la última fecha con covariables. Para las
series recientes (corte 2024-06-01) eso incluye las temporadas 2024-2026.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from fishing_forecast.config import get_settings
from fishing_forecast.evaluation.conformal import mondrian_cqr, sorted_quantile_preds
from fishing_forecast.evaluation.metrics import coverage
from fishing_forecast.features.covariates import build_multiseries_features, feature_columns

SPECIES = ("lobster_red", "abalone_blue", "abalone_red", "abalone_black", "urchin_red")
GROUP = ["species", "economic_unit"]
CONF_FRAC = 0.25
MIN_CATCH_DAYS = 20
CONF_LEVELS = (0.80, 0.90)
CRPS_GRID = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
SEED = 42

#: Corte de prueba por defecto para el servicio: 2024-06-01 deja el bache post-MHW dentro del
#: entrenamiento → calibración estable y horizonte reciente. Override: `FF_SERVE_CUT`.
DEFAULT_CUT = os.environ.get("FF_SERVE_CUT", "2024-06-01")

QUANTILE_XGB = dict(
    objective="reg:quantileerror",
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=4,
)

#: Nombres comunes en español para el front (código interno → etiqueta).
SPECIES_LABELS = {
    "lobster_red": "Langosta roja",
    "abalone_blue": "Abulón azul",
    "abalone_red": "Abulón rojo",
    "abalone_black": "Abulón negro",
    "urchin_red": "Erizo rojo",
}


@dataclass
class ForecastStore:
    """Pronóstico calibrado cacheado, indexado por serie ``especie@unidad``."""

    cut_date: str
    series: dict[str, dict] = field(default_factory=dict)
    units: dict[str, dict] = field(default_factory=dict)  # metadata de UE (nombre, región, lat)

    def list_series(self) -> list[dict]:
        """Resumen ligero de cada serie para poblar los selectores del front."""
        out = []
        for key, rec in sorted(self.series.items()):
            out.append(
                {
                    "series": key,
                    "species": rec["species"],
                    "species_label": rec["species_label"],
                    "unit": rec["unit"],
                    "unit_name": rec["unit_name"],
                    "region": rec["region"],
                    "lat": rec["lat"],
                    "coverage90": rec["coverage90"],
                    "n_test": rec["n_test"],
                    "last_date": rec["dates"][-1] if rec["dates"] else None,
                }
            )
        return out


def _load_units() -> dict[str, dict]:
    """Metadata de unidades económicas (nombre, región, latitud central del bbox)."""
    settings = get_settings()
    cfg = yaml.safe_load((settings.configs_root / "economic_units.yaml").read_text("utf-8")) or {}
    units: dict[str, dict] = {}
    for code, entry in cfg.items():
        bbox = entry.get("bbox") or {}
        lat = None
        if "lat_min" in bbox and "lat_max" in bbox:
            lat = round((float(bbox["lat_min"]) + float(bbox["lat_max"])) / 2.0, 2)
        units[code] = {
            "name": entry.get("name", code),
            "region": entry.get("region", ""),
            "lat": lat,
        }
    return units


def _load_series(species: tuple[str, ...]) -> pd.DataFrame:
    """Series diarias `(especie, UE)` con captura suficiente; NaN→0, recorte a último catch."""
    settings = get_settings()
    df = pd.read_parquet(settings.processed_dir / "dataset_v1.parquet")
    df = df[df["species"].isin(species)].copy()
    df["ds"] = pd.to_datetime(df["ds"])
    frames = []
    for _key, s in df.groupby(GROUP, observed=True):
        s = s.sort_values("ds").copy()
        s["y"] = s["y"].fillna(0.0)
        if int((s["y"] > 0).sum()) < MIN_CATCH_DAYS:
            continue
        last_catch = s.loc[s["y"] > 0, "ds"].max()
        frames.append(s[s["ds"] <= last_catch])
    if not frames:
        raise ValueError("Ninguna serie supera MIN_CATCH_DAYS.")
    return pd.concat(frames, ignore_index=True)


def _fit_quantile_grid(feat, cols, mask, levels):
    import xgboost as xgb

    models = {}
    for a in levels:
        m = xgb.XGBRegressor(quantile_alpha=a, **QUANTILE_XGB)
        m.fit(feat.loc[mask, cols], feat.loc[mask, "y_log"])
        models[a] = m
    return models


def _season_year(ds: pd.Timestamp) -> int:
    """Año de inicio de temporada (15-sep–15-feb): meses >= julio → año actual; si no, año-1."""
    return int(ds.year if ds.month >= 7 else ds.year - 1)


def _in_lobster_season(ds: pd.Timestamp) -> bool:
    """Ventana canónica de langosta roja en Baja California: 15-sep a 15-feb.

    El dataset solo trae calendario para ``litoral_bc_sur``; el resto de las series de langosta
    quedan con ``in_season=True`` todo el año (calendario no declarado). Para el resumen por
    temporada aplicamos aquí la ventana reglamentaria, común a toda la especie.
    """
    md = (ds.month, ds.day)
    return md >= (9, 15) or md <= (2, 15)


def _season_summaries(sdf: pd.DataFrame, species: str) -> list[dict]:
    """Panel de *backtest* por temporada (solo días en temporada).

    El total por temporada se representa como el **observado** (histórico) y la **cobertura
    empírica del intervalo del 90%** ---la métrica honesta de confianza---, no como un punto:
    con distribuciones de captura muy concentradas en pocos días, la suma de medianas diarias
    es un mal estimador del total (subestima las series con picos, sobreestima las bajas). La
    estimación central se conserva solo como referencia, claramente etiquetada.
    """
    ins = sdf.copy()
    ins["ds_dt"] = pd.to_datetime(ins["ds"])
    if species == "lobster_red":
        # Ventana reglamentaria común a toda la especie (el flag del dataset no es fiable fuera
        # de litoral_bc_sur). Etiqueta cruzada de año, p.ej. "2024-25".
        ins = ins[ins["ds_dt"].map(_in_lobster_season)].copy()
        ins["sy"] = ins["ds_dt"].map(_season_year)
        label = lambda sy: f"{sy}-{str(sy + 1)[-2:]}"  # noqa: E731
    else:
        # Sin calendario declarado para abulón/erizo: se agrupa por año calendario.
        ins = ins[ins["in_season"] == 1].copy()
        ins["sy"] = ins["ds_dt"].dt.year
        label = str  # noqa: E731
    if ins.empty:
        return []
    out = []
    for sy, g in ins.groupby("sy"):
        obs = float(g["y_obs"].sum())
        has_obs = bool((g["y_obs"] > 0).any())
        y, lo, hi = g["y_obs"].to_numpy(), g["lo90"].to_numpy(), g["hi90"].to_numpy()
        covered = int(((y >= lo) & (y <= hi)).sum())
        out.append(
            {
                "label": label(sy),
                "obs_total_kg": round(obs, 1) if has_obs else None,
                "coverage90": round(covered / len(g), 3) if len(g) else None,
                "central_est_kg": round(float(g["median"].sum()), 1),  # referencia, no fiable
                "n_days": int(len(g)),
            }
        )
    return sorted(out, key=lambda r: r["label"])


def build_store(cut_date: str | None = None, species: tuple[str, ...] = SPECIES) -> ForecastStore:
    """Entrena la CQR de producción y cachea el pronóstico calibrado por serie.

    Mismo pipeline que Exp 4: features multiserie, XGBoost cuantílico en log, conformalización
    Mondrian normalizada por serie (calibrada en temporada), inversión a kg. Devuelve un
    :class:`ForecastStore` con arreglos diarios y resúmenes por temporada por serie.
    """
    cut = pd.Timestamp(cut_date or DEFAULT_CUT)
    units = _load_units()
    raw = _load_series(species)
    feat = build_multiseries_features(raw, group_col=GROUP)

    base_cols = feature_columns(feat)
    feat["_series"] = feat["species"].astype(str) + "@" + feat["economic_unit"].astype(str)
    onehot = pd.get_dummies(feat[GROUP], columns=GROUP)
    feat = pd.concat([feat, onehot], axis=1)
    cols = base_cols + list(onehot.columns)
    feat["y_log"] = np.log1p(feat["y"])

    pre_cut = feat[feat["ds"] < cut]
    if pre_cut.empty:
        raise ValueError(f"No hay datos antes del corte {cut.date()}.")
    conf_start = pre_cut["ds"].quantile(1.0 - CONF_FRAC)
    train_mask = feat["ds"] < conf_start
    conf_mask = (feat["ds"] >= conf_start) & (feat["ds"] < cut) & (feat["in_season"] == 1)
    test = feat[feat["ds"] >= cut].copy()
    logger.info(
        f"[serving] corte={cut.date()} train={int(train_mask.sum())} "
        f"conf={int(conf_mask.sum())} test={len(test)} series={test['_series'].nunique()}"
    )

    grid_models = _fit_quantile_grid(feat, cols, train_mask, CRPS_GRID)
    X_conf, y_conf_log = feat.loc[conf_mask, cols], feat.loc[conf_mask, "y_log"].to_numpy()
    conf_series = feat.loc[conf_mask, "_series"].to_numpy()
    X_test, test_series = test[cols], test["_series"].to_numpy()

    grid_log_conf = sorted_quantile_preds(grid_models, X_conf)
    grid_log_test = sorted_quantile_preds(grid_models, X_test)
    median = np.clip(np.expm1(grid_log_test[0.5]), 0.0, None)

    def interval(cl: float):
        alpha = 1.0 - cl
        a_lo, a_hi = round(alpha / 2.0, 4), round(1.0 - alpha / 2.0, 4)
        lo_log, hi_log = mondrian_cqr(
            conf_series, test_series,
            grid_log_conf[a_lo], grid_log_conf[a_hi], y_conf_log,
            grid_log_test[a_lo], grid_log_test[a_hi], alpha, normalize=True,
        )
        return np.clip(np.expm1(lo_log), 0.0, None), np.clip(np.expm1(hi_log), 0.0, None)

    lo80, hi80 = interval(0.80)
    lo90, hi90 = interval(0.90)
    y_test = test["y"].to_numpy()

    store = ForecastStore(cut_date=str(cut.date()), units=units)
    test = test.reset_index(drop=True)
    for label, idx in test.groupby("_series", observed=True).groups.items():
        pos = np.asarray(idx)
        order = np.argsort(test.loc[pos, "ds"].to_numpy())
        pos = pos[order]
        sp, ue = label.split("@", 1)
        meta = units.get(ue, {"name": ue, "region": "", "lat": None})
        sdf = pd.DataFrame(
            {
                "ds": test.loc[pos, "ds"].dt.strftime("%Y-%m-%d").to_numpy(),
                "y_obs": np.round(y_test[pos], 1),
                "median": np.round(median[pos], 1),
                "lo80": np.round(lo80[pos], 1),
                "hi80": np.round(hi80[pos], 1),
                "lo90": np.round(lo90[pos], 1),
                "hi90": np.round(hi90[pos], 1),
                "in_season": test.loc[pos, "in_season"].astype(int).to_numpy(),
            }
        )
        # Cobertura de confianza = **en temporada** (métrica operativa honesta): fuera de
        # temporada la captura es 0 y el intervalo la cubre trivialmente, inflando el número.
        ds_pos = test.loc[pos, "ds"]
        if sp == "lobster_red":
            m = ds_pos.map(_in_lobster_season).to_numpy()
        else:
            m = test.loc[pos, "in_season"].astype(bool).to_numpy()
        cov_pos = pos[m] if m.any() else pos
        cov90 = round(float(coverage(y_test[cov_pos], lo90[cov_pos], hi90[cov_pos])), 3)
        store.series[label] = {
            "series": label,
            "species": sp,
            "species_label": SPECIES_LABELS.get(sp, sp),
            "unit": ue,
            "unit_name": meta["name"],
            "region": meta["region"],
            "lat": meta["lat"],
            "coverage90": cov90,
            "n_test": int(len(pos)),
            "dates": sdf["ds"].tolist(),
            "y_obs": sdf["y_obs"].tolist(),
            "median": sdf["median"].tolist(),
            "lo80": sdf["lo80"].tolist(),
            "hi80": sdf["hi80"].tolist(),
            "lo90": sdf["lo90"].tolist(),
            "hi90": sdf["hi90"].tolist(),
            "in_season": sdf["in_season"].tolist(),
            "seasons": _season_summaries(sdf, sp),
        }
    logger.info(f"[serving] store listo: {len(store.series)} series")
    return store
