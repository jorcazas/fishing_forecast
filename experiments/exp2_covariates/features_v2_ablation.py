"""Exp 2.5 (B4) — ¿Suman las features de Fase 2 (anomalías + interacciones)?

`build_features_v2` añade sobre la matriz original:

- **Anomalías climatológicas** por día-del-año (`{col}_anom_lag90`), con la climatología
  estimada **solo con años de train** (`fit_climatology(train_end=corte)`), y
- **interacciones** con justificación ecológica declaradas en `configs/features.yaml`.

Aquí se mide si eso mejora al modelo global de producción (`pooled_log`, Exp 3.2) o si repite
el patrón de Exp 2 —más features, más sobreajuste con pocas temporadas—. Se compara serie por
serie con el mismo pool, mismo objetivo `log1p(y)`, mismos hiperparámetros y misma semilla: la
única diferencia es la matriz de features.

Uso:
    uv run python -m experiments.exp2_covariates.features_v2_ablation
    FF_CUT_DATE=2024-06-01 uv run python -m experiments.exp2_covariates.features_v2_ablation
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from experiments.exp3_global_model.pooled_ynorm import GROUP, XGB_PARAMS, load_series
from loguru import logger

from fishing_forecast.config import get_settings
from fishing_forecast.evaluation.metrics import all_metrics
from fishing_forecast.features.covariates import (
    build_features_v2,
    build_multiseries_features,
    feature_columns,
    load_feature_config,
)

EXP_ID = "exp2_features_v2"
CUT_DATE = pd.Timestamp(os.environ.get("FF_CUT_DATE", "2024-06-01"))


def _multiseries_v2(raw: pd.DataFrame, cut: pd.Timestamp) -> pd.DataFrame:
    """`build_features_v2` aplicado serie por serie (los lags nunca cruzan series).

    La climatología se ajusta **dentro de cada serie** con `train_end=cut`, así que no mezcla
    unidades económicas ni ve el periodo de prueba.
    """
    cfg = load_feature_config()
    frames = []
    for key, sub in raw.groupby(GROUP, observed=True):
        feat = build_features_v2(sub, config=cfg, train_end=cut)
        for col, val in zip(GROUP, key, strict=True):
            feat[col] = val
        frames.append(feat)
    return pd.concat(frames, ignore_index=True)


def _prepare(feat: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One-hot de identidad + objetivo en log; devuelve `(feat, columnas del pool)`."""
    base_cols = feature_columns(feat)
    feat = feat.copy()
    feat["_series"] = feat["species"].astype(str) + "@" + feat["economic_unit"].astype(str)
    onehot = pd.get_dummies(feat[GROUP], columns=GROUP)
    feat = pd.concat([feat, onehot], axis=1)
    feat["y_log"] = np.log1p(feat["y"])
    return feat, base_cols + list(onehot.columns)


def _fit_predict(feat: pd.DataFrame, cols: list[str], cut: pd.Timestamp) -> pd.DataFrame:
    """Entrena el pool en log antes del corte y devuelve las predicciones en kg del test."""
    import xgboost as xgb

    train = feat["ds"] < cut
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(feat.loc[train, cols], feat.loc[train, "y_log"])
    test = feat[~train].copy()
    test["pred"] = np.clip(np.expm1(model.predict(test[cols])), 0.0, None)
    return test[["ds", "_series", "y", "pred"]]


def main() -> None:
    settings = get_settings()
    raw = load_series()

    feat_v1, cols_v1 = _prepare(build_multiseries_features(raw, group_col=GROUP))
    feat_v2, cols_v2 = _prepare(_multiseries_v2(raw, CUT_DATE))
    added = sorted(set(cols_v2) - set(cols_v1))
    logger.info(f"v1={len(cols_v1)} features, v2={len(cols_v2)} (+{len(added)}): {added}")

    pred_v1 = _fit_predict(feat_v1, cols_v1, CUT_DATE)
    pred_v2 = _fit_predict(feat_v2, cols_v2, CUT_DATE)

    per_series: dict[str, dict] = {}
    wins = 0
    for label, g1 in pred_v1.groupby("_series", observed=True):
        g2 = pred_v2[pred_v2["_series"] == label]
        if g1.empty or g2.empty or (g1["y"] > 0).sum() == 0:
            continue
        m1 = all_metrics(g1["y"].to_numpy(), g1["pred"].to_numpy())
        m2 = all_metrics(g2["y"].to_numpy(), g2["pred"].to_numpy())
        better = m2["rmse"] <= m1["rmse"]
        wins += int(better)
        per_series[label] = {"v1": m1, "v2": m2, "v2_better_rmse": bool(better)}

    n = len(per_series)
    result = {
        "cut_date": str(CUT_DATE.date()),
        "n_series": n,
        "n_features_v1": len(cols_v1),
        "n_features_v2": len(cols_v2),
        "added_features": added,
        "v2_wins": wins,
        "v2_win_rate": round(wins / n, 2) if n else None,
        "rmse_mean_v1": round(float(np.mean([r["v1"]["rmse"] for r in per_series.values()])), 1),
        "rmse_mean_v2": round(float(np.mean([r["v2"]["rmse"] for r in per_series.values()])), 1),
        "per_series": per_series,
    }
    metrics_dir = settings.reports_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{EXP_ID}_{CUT_DATE.date()}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )
    _write_summary(result, settings.reports_root / f"{EXP_ID}_summary_{CUT_DATE.date()}.md")
    print(_console(result))


def _console(r: dict) -> str:
    lines = [
        "",
        f"Features v2 (anomalías + interacciones) vs v1 — pooled_log, corte {r['cut_date']}:",
    ]
    lines.append(f"  features: v1={r['n_features_v1']}  v2={r['n_features_v2']}")
    lines.append(f"  {'serie':<34}{'RMSE v1':>10}{'RMSE v2':>10}   mejor")
    for label, m in r["per_series"].items():
        mark = "v2" if m["v2_better_rmse"] else "v1"
        lines.append(f"  {label:<34}{m['v1']['rmse']:>10.1f}{m['v2']['rmse']:>10.1f}   {mark}")
    lines.append(
        f"\n  v2 gana/empata en {r['v2_wins']}/{r['n_series']} series ({r['v2_win_rate']}); "
        f"RMSE medio {r['rmse_mean_v1']} → {r['rmse_mean_v2']}"
    )
    return "\n".join(lines)


def _write_summary(r: dict, out_path) -> None:
    rows = [
        f"# Exp 2.5 — Features de Fase 2 sobre el modelo global (corte {r['cut_date']})",
        "",
        f"Mismo pool `log1p(y)`, mismos hiperparámetros y semilla; solo cambia la matriz: "
        f"**{r['n_features_v1']} → {r['n_features_v2']}** features.",
        "",
        "Añadidas: " + ", ".join(f"`{c}`" for c in r["added_features"]),
        "",
        f"**v2 gana o empata en RMSE en {r['v2_wins']}/{r['n_series']} series "
        f"({r['v2_win_rate']}).** RMSE medio entre series: {r['rmse_mean_v1']} → "
        f"{r['rmse_mean_v2']} kg/día.",
        "",
        "| serie | RMSE v1 | RMSE v2 | sMAPE v1 | sMAPE v2 |",
        "|---|---|---|---|---|",
    ]
    for label, m in r["per_series"].items():
        rows.append(
            f"| {label} | {m['v1']['rmse']:.1f} | {m['v2']['rmse']:.1f} | "
            f"{m['v1']['smape']:.1f} | {m['v2']['smape']:.1f} |"
        )
    rows += [
        "",
        "> La climatología de las anomalías se estima solo con datos anteriores al corte "
        "(`fit_climatology(train_end=corte)`) y dentro de cada serie; las interacciones están "
        "declaradas con justificación ecológica en `configs/features.yaml`.",
    ]
    out_path.write_text("\n".join(rows))
    logger.info(f"Resumen → {out_path}")


if __name__ == "__main__":
    main()
