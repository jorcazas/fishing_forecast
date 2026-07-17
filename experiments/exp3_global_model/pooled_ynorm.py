"""Exp 3.2 — Pooling con `y` normalizada por serie: ¿evita que la escala domine el loss?

Exp 3 mostró que el pooling todo-junto (un XGBoost sobre todas las series `(especie, UE)`) queda
**confundido por escala**: langosta (cientos de kg) domina el loss cuadrático sobre abulón/erizo
(unidades–decenas), así que el modelo global aprende sobre todo langosta y ayuda poco a las series
pequeñas. Aquí normalizamos el objetivo **por serie antes de agrupar** y comparamos:

- `specific`   — un modelo por serie sobre `y` cruda (referencia, = Exp 3 específico).
- `pooled_raw` — pooling sobre `y` cruda (= Exp 3 global; el que sufre por escala).
- `pooled_log` — pooling sobre `log1p(y)` (transformación global, sin stats, invertible).
- `pooled_z`   — pooling sobre z-score **por serie con media/desv de TRAIN** (sin leakage).

Para todos los pooled, el modelo ve el one-hot de especie/UE + oceanografía por UE. La predicción
se **invierte a kg** antes de medir, y se compara serie por serie contra el específico.

Hipótesis: normalizar sube el win-rate del pooling frente al específico, sobre todo en las series
de escala chica (abulón/erizo), sin perjudicar a langosta.

Uso:
    uv run python experiments/exp3_global_model/pooled_ynorm.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from loguru import logger

from fishing_forecast.config import get_settings
from fishing_forecast.evaluation.metrics import all_metrics, season_sum_errors
from fishing_forecast.features.covariates import build_multiseries_features, feature_columns

EXP_ID = "exp3_pooled_ynorm"
SPECIES = ("lobster_red", "abalone_blue", "abalone_red", "abalone_black", "urchin_red")
GROUP = ["species", "economic_unit"]
CUT_DATE = pd.Timestamp("2020-07-01")
MIN_CATCH_DAYS = 20
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


def _series_key(feat: pd.DataFrame) -> pd.Series:
    return feat["species"].astype(str) + "@" + feat["economic_unit"].astype(str)


def _train_stats(feat: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """Media/desv de `y` por serie usando **solo** train (para z-score sin leakage)."""
    stats = (
        feat.loc[train_mask]
        .groupby("_series", observed=True)["y"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "mu", "std": "sigma"})
    )
    stats["sigma"] = stats["sigma"].replace(0.0, 1.0).fillna(1.0)  # serie constante → sigma=1
    return stats


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, seasons: np.ndarray) -> dict:
    ss = season_sum_errors(pd.DataFrame({"season": seasons, "y_true": y_true, "y_pred": y_pred}))
    return {"daily": all_metrics(y_true, y_pred), "season_sum": ss["pct_error"].round(1).to_dict()}


def _fit_pool(feat, cols, train_mask, target: str):
    import xgboost as xgb

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(feat.loc[train_mask, cols], feat.loc[train_mask, target])
    return model


def main() -> None:
    settings = get_settings()
    raw = load_series()
    feat = build_multiseries_features(raw, group_col=GROUP)

    base_cols = feature_columns(feat)  # antes de añadir _series/one-hot/objetivos derivados
    feat["_series"] = _series_key(feat)
    onehot = pd.get_dummies(feat[GROUP], columns=GROUP)
    feat = pd.concat([feat, onehot], axis=1)
    pool_cols = base_cols + list(onehot.columns)

    train_mask = feat["ds"] < CUT_DATE

    # Objetivos normalizados por serie (stats de train para z-score → sin leakage).
    stats = _train_stats(feat, train_mask)
    feat["y_log"] = np.log1p(feat["y"])
    mu = feat["_series"].map(stats["mu"])
    sigma = feat["_series"].map(stats["sigma"])
    feat["y_z"] = (feat["y"] - mu) / sigma

    # Modelos pooled (uno por variante de objetivo).
    m_raw = _fit_pool(feat, pool_cols, train_mask, "y")
    m_log = _fit_pool(feat, pool_cols, train_mask, "y_log")
    m_z = _fit_pool(feat, pool_cols, train_mask, "y_z")

    import xgboost as xgb

    results: dict[str, dict] = {}
    wins = {"pooled_raw": 0, "pooled_log": 0, "pooled_z": 0}
    for (sp, ue), sp_all in feat.groupby(GROUP, observed=True):
        test = sp_all[sp_all["ds"] >= CUT_DATE]
        if test.empty or (test["y"] > 0).sum() == 0:
            logger.warning(f"{sp}@{ue}: sin captura en test; se omite.")
            continue
        label = f"{sp}@{ue}"
        y_true = test["y"].to_numpy()
        seasons = test["season"].to_numpy()
        s_mu = float(stats.loc[label, "mu"])
        s_sigma = float(stats.loc[label, "sigma"])

        # Específico (referencia): un modelo por serie, y cruda.
        sp_train = sp_all[sp_all["ds"] < CUT_DATE]
        smodel = xgb.XGBRegressor(**XGB_PARAMS)
        smodel.fit(sp_train[base_cols], sp_train["y"])
        s_pred = np.clip(smodel.predict(test[base_cols]), 0.0, None)
        s_eval = _evaluate(y_true, s_pred, seasons)

        # Pooled: predecir en espacio transformado → invertir a kg → clip.
        p_raw = np.clip(m_raw.predict(test[pool_cols]), 0.0, None)
        p_log = np.clip(np.expm1(m_log.predict(test[pool_cols])), 0.0, None)
        p_z = np.clip(m_z.predict(test[pool_cols]) * s_sigma + s_mu, 0.0, None)

        variants = {
            "specific": s_eval,
            "pooled_raw": _evaluate(y_true, p_raw, seasons),
            "pooled_log": _evaluate(y_true, p_log, seasons),
            "pooled_z": _evaluate(y_true, p_z, seasons),
        }
        ref = s_eval["daily"]["rmse"]
        win_flags = {}
        for name in ("pooled_raw", "pooled_log", "pooled_z"):
            w = variants[name]["daily"]["rmse"] <= ref
            wins[name] += int(w)
            win_flags[name] = bool(w)
        results[label] = {
            "n_test_catch": int((y_true > 0).sum()),
            "train_scale_mu": round(s_mu, 1),
            "variants": variants,
            "beats_specific_rmse": win_flags,
        }

    n = len(results)
    summary = {
        "cut_date": str(CUT_DATE.date()),
        "n_series": n,
        "win_rate_vs_specific": {k: (round(v / n, 2) if n else None) for k, v in wins.items()},
        "wins_vs_specific": wins,
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
    lines = ["", f"Pooling con y normalizada vs específico (RMSE diario), corte {s['cut_date']}:"]
    lines.append(f"  {'serie':<30}{'específ.':>10}{'p_raw':>9}{'p_log':>9}{'p_z':>9}")
    for label, r in s["per_series"].items():
        v = r["variants"]
        lines.append(
            f"  {label:<30}"
            f"{v['specific']['daily']['rmse']:>10.1f}"
            f"{v['pooled_raw']['daily']['rmse']:>9.1f}"
            f"{v['pooled_log']['daily']['rmse']:>9.1f}"
            f"{v['pooled_z']['daily']['rmse']:>9.1f}"
        )
    wr = s["win_rate_vs_specific"]
    lines.append(
        f"\n  Gana/empata vs específico:  raw {wr['pooled_raw']}  "
        f"log {wr['pooled_log']}  z {wr['pooled_z']}  (de {s['n_series']} series)"
    )
    return "\n".join(lines)


def _write_summary(s: dict, out_path) -> None:
    rows = [
        "# Exp 3.2 — Pooling con `y` normalizada por serie",
        "",
        f"Corte **{s['cut_date']}**. {s['n_series']} series `(especie, UE)`. RMSE diario (kg), "
        "tras invertir la transformación. `p_raw`=pool sobre y cruda, `p_log`=pool sobre log1p(y), "
        "`p_z`=pool sobre z-score por serie (stats de train).",
        "",
        "| serie (especie@UE) | escala µ_train | específico | p_raw | p_log | p_z |",
        "|---|---|---|---|---|---|",
    ]
    for label, r in s["per_series"].items():
        v = r["variants"]
        rows.append(
            f"| {label} | {r['train_scale_mu']} | {v['specific']['daily']['rmse']:.1f} | "
            f"{v['pooled_raw']['daily']['rmse']:.1f} | {v['pooled_log']['daily']['rmse']:.1f} | "
            f"{v['pooled_z']['daily']['rmse']:.1f} |"
        )
    wr = s["win_rate_vs_specific"]
    rows += [
        "",
        f"**Gana/empata vs específico (RMSE):** p_raw {s['wins_vs_specific']['pooled_raw']}/"
        f"{s['n_series']} ({wr['pooled_raw']}), p_log {s['wins_vs_specific']['pooled_log']}/"
        f"{s['n_series']} ({wr['pooled_log']}), p_z {s['wins_vs_specific']['pooled_z']}/"
        f"{s['n_series']} ({wr['pooled_z']}).",
        "",
        "> Normalizar el objetivo por serie evita que la escala de langosta (cientos de kg) domine "
        "el loss del pool sobre abulón/erizo (unidades). Sin leakage: las stats del z-score salen "
        "solo de train.",
    ]
    out_path.write_text("\n".join(rows))
    logger.info(f"Resumen → {out_path}")


if __name__ == "__main__":
    main()
