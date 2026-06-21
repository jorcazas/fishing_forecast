"""Exp 1 — Re-entrenamiento del baseline estadístico (ARIMA, Prophet) en langosta-SQ.

Reproduce el baseline del borrador 2024 con los datos reales actuales (`dataset_v1`,
arribos COBI). Por ahora solo los modelos que usan **solo `y`** (ARIMA, Prophet); los que
necesitan covariables oceanográficas (LGBM/XGBoost/LSTM con `x1..x16`/SST) esperan a la
ingesta de OISST/GlobColour.

Decisiones (documentadas; ver `docs/etl_design.md` §4.4 y ADR-0001):
- Serie diaria de `y` (kg) para `lobster_red × litoral_bc_sur`.
- Los `y=NaN` dentro de temporada se rellenan con **0** *solo para modelar* (un día en
  temporada sin registro = sin captura ese día). El ETL conserva el NaN; el relleno vive
  en esta capa de modelado.
- La serie se recorta al último día con captura > 0 (la UE deja de reportar langosta tras
  la temporada 2021-2022; modelar el vacío posterior no aporta).
- Corte de test canónico **2020-07-01** (comparable con el borrador). El corte adicional
  2024-06-01 no aplica: no hay langosta-SQ después de 2022.

Uso:
    uv run python experiments/exp1_baseline_retrain/baseline.py
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from fishing_forecast.config import get_settings
from fishing_forecast.evaluation.metrics import all_metrics, season_sum_errors

EXP_ID = "exp1_baseline"
SPECIES = "lobster_red"
ECONOMIC_UNIT = "litoral_bc_sur"
CUT_DATE = pd.Timestamp("2020-07-01")
SEED = 42

# Rejilla pequeña y principista para ARIMA (AIC en train), no la 50×50×50 del borrador.
ARIMA_P = range(0, 4)
ARIMA_D = range(0, 2)
ARIMA_Q = range(0, 4)


@dataclass
class SeriesBundle:
    """Serie diaria y su partición temporal, con la temporada por día."""

    full: pd.DataFrame  # columnas: ds (index), y, season
    train: pd.DataFrame
    test: pd.DataFrame


def load_series(species: str = SPECIES, economic_unit: str = ECONOMIC_UNIT) -> SeriesBundle:
    """Carga `dataset_v1`, filtra la serie, rellena NaN in-season=0 y recorta el vacío final."""
    settings = get_settings()
    path = settings.processed_dir / "dataset_v1.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Falta {path}. Corre `fishing-etl transform arribos --source cobi` y `consolidate`."
        )
    df = pd.read_parquet(path)
    sub = df[(df["species"] == species) & (df["economic_unit"] == economic_unit)].copy()
    if sub.empty:
        raise ValueError(f"Sin filas para ({species}, {economic_unit}) en {path}.")

    sub["ds"] = pd.to_datetime(sub["ds"])
    sub = sub.sort_values("ds").set_index("ds")
    sub["y"] = sub["y"].fillna(0.0)  # NaN in-season → 0 solo para modelar (ver docstring)

    last_catch = sub.index[sub["y"] > 0].max()
    sub = sub.loc[:last_catch]
    logger.info(f"Serie {species}×{economic_unit}: {len(sub)} días, hasta {last_catch.date()}.")

    full = sub[["y", "season"]]
    return SeriesBundle(
        full=full,
        train=full[full.index < CUT_DATE],
        test=full[full.index >= CUT_DATE],
    )


def fit_arima(train_y: pd.Series) -> tuple[object, tuple[int, int, int]]:
    """Ajusta ARIMA eligiendo (p,d,q) por mínimo AIC en train. Devuelve (modelo, orden)."""
    from statsmodels.tsa.arima.model import ARIMA

    best_aic, best_order, best_fit = np.inf, None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in ARIMA_P:
            for d in ARIMA_D:
                for q in ARIMA_Q:
                    try:
                        fit = ARIMA(train_y.to_numpy(), order=(p, d, q)).fit()
                    except Exception:  # combinaciones no convergentes: se saltan
                        continue
                    if fit.aic < best_aic:
                        best_aic, best_order, best_fit = fit.aic, (p, d, q), fit
    if best_fit is None:
        raise RuntimeError("Ningún ARIMA convergió en la rejilla.")
    logger.info(f"ARIMA mejor orden={best_order} (AIC={best_aic:.1f})")
    return best_fit, best_order


def forecast_arima(fit: object, n: int) -> np.ndarray:
    """Pronóstico puntual de ARIMA a `n` pasos, recortado a no-negativos (kg ≥ 0)."""
    fc = np.asarray(fit.forecast(steps=n), dtype=float)
    return np.clip(fc, 0.0, None)


def fit_forecast_prophet(train: pd.DataFrame, test_index: pd.DatetimeIndex) -> np.ndarray | None:
    """Prophet con estacionalidad anual. Devuelve None si Prophet no está instalado."""
    try:
        from prophet import Prophet
    except ImportError:
        logger.warning("Prophet no instalado; se omite (uv pip install prophet).")
        return None

    dfp = train.reset_index().rename(columns={"ds": "ds", "y": "y"})[["ds", "y"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(dfp)
        future = pd.DataFrame({"ds": test_index})
        fc = m.predict(future)
    return np.clip(fc["yhat"].to_numpy(), 0.0, None)


def evaluate_model(name: str, test: pd.DataFrame, y_pred: np.ndarray) -> dict:
    """Calcula métricas diarias + error de suma de temporada para un modelo."""
    daily = all_metrics(test["y"].to_numpy(), y_pred)
    season_df = pd.DataFrame(
        {"season": test["season"].to_numpy(), "y_true": test["y"].to_numpy(), "y_pred": y_pred}
    )
    seasons = season_sum_errors(season_df)
    return {
        "model": name,
        "cut_date": str(CUT_DATE.date()),
        "daily": daily,
        "season_sum": {
            s: {
                "true_sum": round(float(r.true_sum), 1),
                "pred_sum": round(float(r.pred_sum), 1),
                "pct_error": None if pd.isna(r.pct_error) else round(float(r.pct_error), 1),
            }
            for s, r in seasons.iterrows()
        },
    }


def _plot(test: pd.DataFrame, preds: dict[str, np.ndarray], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(test.index, test["y"], color="#222", lw=0.8, label="real")
    for name, yp in preds.items():
        ax.plot(test.index, yp, lw=1.2, label=f"pred {name}")
    ax.axvline(CUT_DATE, color="grey", ls=":", lw=1)
    ax.set_title(f"Exp 1 — baseline langosta-SQ (test desde {CUT_DATE.date()})")
    ax.set_ylabel("y (kg/día)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    np.random.seed(SEED)
    settings = get_settings()
    bundle = load_series()
    logger.info(f"train={len(bundle.train)} días, test={len(bundle.test)} días")

    results: list[dict] = []
    preds: dict[str, np.ndarray] = {}

    # --- ARIMA ---
    arima_fit, order = fit_arima(bundle.train["y"])
    arima_pred = forecast_arima(arima_fit, len(bundle.test))
    preds["arima"] = arima_pred
    res = evaluate_model("arima", bundle.test, arima_pred)
    res["order"] = list(order)
    results.append(res)

    # --- Prophet (si está instalado) ---
    prophet_pred = fit_forecast_prophet(bundle.train, bundle.test.index)
    if prophet_pred is not None:
        preds["prophet"] = prophet_pred
        results.append(evaluate_model("prophet", bundle.test, prophet_pred))

    # --- Artefactos ---
    metrics_dir = settings.reports_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for res in results:
        out = metrics_dir / f"{EXP_ID}_{res['model']}_{res['cut_date']}.json"
        out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        logger.info(f"Métricas → {out}")

    _plot(bundle.test, preds, settings.reports_root / "figures" / f"{EXP_ID}_pred_vs_real.png")
    _write_summary(results, settings.reports_root / f"{EXP_ID}_summary.md")
    print(_summary_table(results))


def _summary_table(results: list[dict]) -> str:
    lines = ["", f"{'modelo':<10}{'MAE':>12}{'RMSE':>12}{'sMAPE%':>10}"]
    for r in results:
        d = r["daily"]
        lines.append(f"{r['model']:<10}{d['mae']:>12.1f}{d['rmse']:>12.1f}{d['smape']:>10.1f}")
    return "\n".join(lines)


def _write_summary(results: list[dict], out_path: Path) -> None:
    rows = ["# Exp 1 — Baseline estadístico (langosta-SQ)", ""]
    rows.append(f"Corte de test: **{results[0]['cut_date']}**. Solo modelos sobre `y`.")
    rows.append("")
    rows.append("| modelo | MAE | RMSE | sMAPE% | error suma temporada (por temporada) |")
    rows.append("|---|---|---|---|---|")
    for r in results:
        d = r["daily"]
        seas = "; ".join(
            f"{s}: {v['pct_error']}%" for s, v in r["season_sum"].items() if v["pct_error"] is not None
        )
        rows.append(
            f"| {r['model']} | {d['mae']:.1f} | {d['rmse']:.1f} | {d['smape']:.1f} | {seas or '—'} |"
        )
    rows.append("")
    rows.append(
        "> Nota: ARIMA/Prophet son baselines débiles para una serie estacional y dispersa; "
        "el desempeño fuerte del borrador venía del ensamble XGBoost+LSTM con covariables "
        "oceanográficas (pendiente de la ingesta OISST/GlobColour)."
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows))
    logger.info(f"Resumen → {out_path}")


if __name__ == "__main__":
    main()
