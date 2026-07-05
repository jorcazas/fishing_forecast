"""Exp 2 — Modelo con covariables oceanográficas (SST + MHW) vs el piso ARIMA/Prophet.

Pregunta: ¿añadir SST/MHW desplazadas 3 meses reduce la **sobrepredicción de la temporada
2021-2022** (el crash post-ola de calor) que cometían ARIMA/Prophet (+291% / +418%)?

Modelo: XGBoost sobre la serie diaria de langosta-SQ, features de `features/covariates.py`
(calendario + lags de `y` + SST/MHW desplazadas 90 días, **sin leakage**). Misma partición
y serie que Exp 1 (corte 2020-07-01, recorte al último día con captura) para comparar peras
con peras.

**Limitación honesta**: el train (2017-01 → 2020-06) tiene solo ~3 temporadas completas, así
que la relación "MHW del año previo → menor captura" tiene poquísimos ejemplos. Este Exp
establece el marco y el baseline con covariables; el poder real llegará con más temporadas
(datos nuevos) y el modelo global multi-UE de Fase 3.

Uso:
    uv run python experiments/exp2_covariates/covariate_model.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from fishing_forecast.config import get_settings
from fishing_forecast.evaluation.metrics import all_metrics, season_sum_errors
from fishing_forecast.features.covariates import build_covariate_features, feature_columns

EXP_ID = "exp2_covariates"
SPECIES = "lobster_red"
ECONOMIC_UNIT = "litoral_bc_sur"
CUT_DATE = pd.Timestamp("2020-07-01")
SEED = 42

XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=4,
)


def load_lobster_sq() -> pd.DataFrame:
    """Serie diaria de langosta-SQ con SST/MHW; NaN in-season de `y`→0; recorte a último catch."""
    settings = get_settings()
    path = settings.processed_dir / "dataset_v1.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}. Corre el ETL (transform/aggregate/consolidate).")
    df = pd.read_parquet(path)
    sub = df[(df["species"] == SPECIES) & (df["economic_unit"] == ECONOMIC_UNIT)].copy()
    sub["ds"] = pd.to_datetime(sub["ds"])
    sub = sub.sort_values("ds").reset_index(drop=True)
    sub["y"] = sub["y"].fillna(0.0)  # ver ADR-0001 / Exp 1
    last_catch = sub.loc[sub["y"] > 0, "ds"].max()
    sub = sub[sub["ds"] <= last_catch].reset_index(drop=True)
    if sub["sst"].notna().mean() < 0.5:
        logger.warning("SST mayormente nula: ¿corriste `aggregate ocean` + `consolidate`?")
    return sub


def _load_floor(settings) -> dict[str, dict]:
    """Lee los errores de suma de temporada de Exp 1 (ARIMA/Prophet) si existen."""
    floor: dict[str, dict] = {}
    for model in ("arima", "prophet"):
        p = settings.reports_root / "metrics" / f"exp1_baseline_{model}_{CUT_DATE.date()}.json"
        if p.exists():
            floor[model] = json.loads(p.read_text())["season_sum"]
    return floor


def main() -> None:
    import xgboost as xgb

    settings = get_settings()
    df = load_lobster_sq()
    feat = build_covariate_features(df, shift_days=90)
    cols = feature_columns(feat)

    train = feat[feat["ds"] < CUT_DATE]
    test = feat[feat["ds"] >= CUT_DATE]
    logger.info(f"train={len(train)} días, test={len(test)} días, {len(cols)} features")

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(train[cols], train["y"])
    pred = np.clip(model.predict(test[cols]), 0.0, None)

    daily = all_metrics(test["y"].to_numpy(), pred)
    season_df = pd.DataFrame(
        {"season": test["season"].to_numpy(), "y_true": test["y"].to_numpy(), "y_pred": pred}
    )
    seasons = season_sum_errors(season_df)

    importance = (
        pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False).head(8)
    )

    result = {
        "model": "xgboost_covariates",
        "cut_date": str(CUT_DATE.date()),
        "n_features": len(cols),
        "daily": daily,
        "season_sum": {
            s: {
                "true_sum": round(float(r.true_sum), 1),
                "pred_sum": round(float(r.pred_sum), 1),
                "pct_error": None if pd.isna(r.pct_error) else round(float(r.pct_error), 1),
            }
            for s, r in seasons.iterrows()
        },
        "top_features": {k: round(float(v), 4) for k, v in importance.items()},
    }

    metrics_dir = settings.reports_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{EXP_ID}_{CUT_DATE.date()}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )

    _plot(test, pred, settings.reports_root / "figures" / f"{EXP_ID}_pred_vs_real.png")
    floor = _load_floor(settings)
    _write_summary(result, floor, settings.reports_root / f"{EXP_ID}_summary.md")
    print(_console_table(result, floor))


def _plot(test: pd.DataFrame, pred: np.ndarray, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(test["ds"], test["y"], color="#222", lw=0.8, label="real")
    ax.plot(test["ds"], pred, color="#c0392b", lw=1.3, label="XGBoost (SST+MHW)")
    ax.set_title(f"Exp 2 — langosta-SQ con covariables (test desde {CUT_DATE.date()})")
    ax.set_ylabel("y (kg/día)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _season_str(season_sum: dict) -> str:
    return "; ".join(
        f"{s}: {v['pct_error']:+.0f}%" for s, v in season_sum.items() if v["pct_error"] is not None
    )


def _console_table(result: dict, floor: dict) -> str:
    lines = ["", "Error de suma de temporada (la métrica que importa a COBI):"]
    for m, ss in floor.items():
        lines.append(f"  {m:<22} {_season_str(ss)}")
    lines.append(f"  {'xgboost (SST+MHW)':<22} {_season_str(result['season_sum'])}")
    d = result["daily"]
    lines.append(
        f"\n  XGBoost diario: MAE={d['mae']:.1f}  RMSE={d['rmse']:.1f}  sMAPE={d['smape']:.1f}%"
    )
    lines.append("  top features: " + ", ".join(list(result["top_features"])[:5]))
    return "\n".join(lines)


def _write_summary(result: dict, floor: dict, out_path: Path) -> None:
    rows = [
        "# Exp 2 — Covariables oceanográficas (langosta-SQ)",
        "",
        f"Corte de test **{result['cut_date']}**, {result['n_features']} features "
        "(calendario + lags de y + SST/MHW desplazadas 90 días).",
        "",
        "## Error de suma de temporada vs el piso (Exp 1)",
        "",
        "| modelo | 2020_2021 | 2021_2022 (crash) |",
        "|---|---|---|",
    ]

    def cell(ss: dict, key: str) -> str:
        v = ss.get(key, {}).get("pct_error")
        return "—" if v is None else f"{v:+.0f}%"

    for m, ss in floor.items():
        rows.append(f"| {m} | {cell(ss, '2020_2021')} | {cell(ss, '2021_2022')} |")
    ss = result["season_sum"]
    rows.append(f"| **xgboost (SST+MHW)** | {cell(ss, '2020_2021')} | {cell(ss, '2021_2022')} |")
    rows += [
        "",
        f"Diario: MAE={result['daily']['mae']:.1f}, RMSE={result['daily']['rmse']:.1f}, "
        f"sMAPE={result['daily']['smape']:.1f}%.",
        "",
        "Top features (importancia XGBoost): " + ", ".join(result["top_features"]),
        "",
        "> Limitación: solo ~3 temporadas en train, así que la señal MHW→captura tiene pocos "
        "ejemplos. Marco listo; el poder llega con más temporadas y el modelo global de Fase 3.",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows))
    logger.info(f"Resumen → {out_path}")


if __name__ == "__main__":
    main()
