"""Exp 3 — Modelo global multi-especie / multi-UE vs modelos específicos.

Pregunta: ¿un solo XGBoost entrenado sobre **todas las series `(especie, UE)`** (con one-hot
de especie y de UE + oceanografía por UE) predice mejor las series cortas que un modelo por
serie? Es la respuesta al cuello de botella de Exp 2 (pocos datos por serie): agrupar para
tomar prestada fuerza estadística, ahora también **entre unidades económicas** a lo largo del
gradiente biogeográfico (San Quintín ~30.5°N vs Isla Cedros ~28°N).

Diseño en `docs/hierarchical_design.md`. Corte 2020-07-01 (igual a Exp 1/2). Serie =
`(species, economic_unit)` con ≥ `MIN_CATCH_DAYS` días de captura; se recorta cada serie a su
último día con captura.

Uso:
    uv run python experiments/exp3_global_model/global_model.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from loguru import logger

from fishing_forecast.config import get_settings
from fishing_forecast.evaluation.metrics import all_metrics, season_sum_errors
from fishing_forecast.features.covariates import build_multiseries_features, feature_columns

EXP_ID = "exp3_global_model"
SPECIES = ("lobster_red", "abalone_blue", "abalone_red", "abalone_black", "urchin_red")
GROUP = ["species", "economic_unit"]
CUT_DATE = pd.Timestamp("2020-07-01")
MIN_CATCH_DAYS = 20  # mínimo de días con captura para incluir una serie
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


def load_series() -> pd.DataFrame:
    """Series diarias `(especie, UE)` con captura suficiente; NaN in-season→0, recorte a último catch."""
    settings = get_settings()
    df = pd.read_parquet(settings.processed_dir / "dataset_v1.parquet")
    df = df[df["species"].isin(SPECIES)].copy()
    df["ds"] = pd.to_datetime(df["ds"])
    frames = []
    for (sp, ue), s in df.groupby(["species", "economic_unit"], observed=True):
        s = s.sort_values("ds").copy()
        s["y"] = s["y"].fillna(0.0)
        catch_days = int((s["y"] > 0).sum())
        if catch_days < MIN_CATCH_DAYS:
            continue
        last_catch = s.loc[s["y"] > 0, "ds"].max()
        frames.append(s[s["ds"] <= last_catch])
        logger.info(f"serie {sp}@{ue}: {catch_days} días con captura")
    if not frames:
        raise ValueError("Ninguna serie supera MIN_CATCH_DAYS.")
    return pd.concat(frames, ignore_index=True)


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, seasons: np.ndarray) -> dict:
    ss = season_sum_errors(pd.DataFrame({"season": seasons, "y_true": y_true, "y_pred": y_pred}))
    return {"daily": all_metrics(y_true, y_pred), "season_sum": ss["pct_error"].round(1).to_dict()}


def main() -> None:
    import xgboost as xgb

    settings = get_settings()
    raw = load_series()
    feat = build_multiseries_features(raw, group_col=GROUP)

    base_cols = feature_columns(feat)  # antes del one-hot
    onehot = pd.get_dummies(
        feat[["species", "economic_unit"]], columns=["species", "economic_unit"]
    )
    feat = pd.concat([feat, onehot], axis=1)
    global_cols = base_cols + list(onehot.columns)

    train_mask = feat["ds"] < CUT_DATE
    gmodel = xgb.XGBRegressor(**XGB_PARAMS)
    gmodel.fit(feat.loc[train_mask, global_cols], feat.loc[train_mask, "y"])
    g_importance = (
        pd.Series(gmodel.feature_importances_, index=global_cols)
        .sort_values(ascending=False)
        .head(8)
    )

    results: dict[str, dict] = {}
    wins = 0
    for (sp, ue), sp_all in feat.groupby(["species", "economic_unit"], observed=True):
        test = sp_all[sp_all["ds"] >= CUT_DATE]
        if test.empty or (test["y"] > 0).sum() == 0:
            logger.warning(f"{sp}@{ue}: sin captura en test; se omite de la comparación.")
            continue
        label = f"{sp}@{ue}"

        g_pred = np.clip(gmodel.predict(test[global_cols]), 0.0, None)
        g_eval = _evaluate(test["y"].to_numpy(), g_pred, test["season"].to_numpy())

        sp_train = sp_all[sp_all["ds"] < CUT_DATE]
        smodel = xgb.XGBRegressor(**XGB_PARAMS)
        smodel.fit(sp_train[base_cols], sp_train["y"])
        s_pred = np.clip(smodel.predict(test[base_cols]), 0.0, None)
        s_eval = _evaluate(test["y"].to_numpy(), s_pred, test["season"].to_numpy())

        global_wins = g_eval["daily"]["rmse"] <= s_eval["daily"]["rmse"]
        wins += int(global_wins)
        results[label] = {
            "n_test": len(test),
            "n_test_catch": int((test["y"] > 0).sum()),
            "global": g_eval,
            "specific": s_eval,
            "global_wins_rmse": bool(global_wins),
        }

    n = len(results)
    summary = {
        "cut_date": str(CUT_DATE.date()),
        "n_series": n,
        "global_wins": wins,
        "global_win_rate": round(wins / n, 2) if n else None,
        "success_criterion_0.60": (wins / n >= 0.60) if n else None,
        "global_top_features": {k: round(float(v), 4) for k, v in g_importance.items()},
        "per_series": results,
    }
    metrics_dir = settings.reports_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{EXP_ID}_{CUT_DATE.date()}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    _write_summary(summary, settings.reports_root / f"{EXP_ID}_summary.md")
    print(_console(summary))


def _console(s: dict) -> str:
    lines = ["", f"Global vs específico por serie (RMSE diario), corte {s['cut_date']}:"]
    lines.append(f"  {'serie':<32}{'n_catch':>8}{'global':>10}{'específ.':>11}  gana")
    for label, r in s["per_series"].items():
        g, sp_ = r["global"]["daily"]["rmse"], r["specific"]["daily"]["rmse"]
        win = "global" if r["global_wins_rmse"] else "específico"
        lines.append(f"  {label:<32}{r['n_test_catch']:>8}{g:>10.1f}{sp_:>11.1f}  {win}")
    lines.append(
        f"\n  Global gana/empata en {s['global_wins']}/{s['n_series']} series "
        f"(rate {s['global_win_rate']}); criterio ≥0.60: {s['success_criterion_0.60']}"
    )
    lines.append("  top features globales: " + ", ".join(list(s["global_top_features"])[:5]))
    return "\n".join(lines)


def _write_summary(s: dict, out_path) -> None:
    rows = [
        "# Exp 3 — Modelo global multi-especie / multi-UE vs específico",
        "",
        f"Corte **{s['cut_date']}**. {s['n_series']} series `(especie, UE)`.",
        "",
        "| serie (especie@UE) | días captura test | RMSE global | RMSE específico | gana |",
        "|---|---|---|---|---|",
    ]
    for label, r in s["per_series"].items():
        g, sp_ = r["global"]["daily"]["rmse"], r["specific"]["daily"]["rmse"]
        win = "global" if r["global_wins_rmse"] else "específico"
        rows.append(f"| {label} | {r['n_test_catch']} | {g:.1f} | {sp_:.1f} | {win} |")
    rows += [
        "",
        f"**Global gana/empata en {s['global_wins']}/{s['n_series']} series** "
        f"(rate {s['global_win_rate']}); criterio ≥0.60 (PLAN §3.1): **{s['success_criterion_0.60']}**.",
        "",
        "Top features del modelo global: " + ", ".join(s["global_top_features"]),
        "",
        "> UEs: San Quintín (~30.5°N) e Isla Cedros (~28°N) — gradiente biogeográfico. Bboxes "
        "aproximados (oficina de arribo), pendiente polígono TURF. Test de abulón diminuto.",
    ]
    out_path.write_text("\n".join(rows))
    logger.info(f"Resumen → {out_path}")


if __name__ == "__main__":
    main()
