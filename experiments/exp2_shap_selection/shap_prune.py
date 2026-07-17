"""Exp 2.3 — Selección de features con SHAP: ¿podar reduce el sobreajuste?

Exp 2 mostró que añadir las ~35 features (SST/MHW + color del océano, con lags/rolling) al
modelo de una sola serie **empeora** el desempeño (sobreajuste con pocas temporadas). Aquí:

1. Entrena XGBoost con el set completo (langosta-SQ, corte 2020-07-01).
2. Calcula SHAP (`TreeExplainer`) sobre train → ranking por `mean(|SHAP|)`.
3. Poda: conserva las features con `mean(|SHAP|)` ≥ `THRESHOLD_FRAC` de la suma total.
4. Re-entrena solo con las podadas y compara **completo vs podado** en test.

Hipótesis: el modelo podado generaliza mejor (menos varianza) que el completo.

Uso:
    uv run python experiments/exp2_shap_selection/shap_prune.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from loguru import logger

from fishing_forecast.config import get_settings
from fishing_forecast.evaluation.metrics import all_metrics, season_sum_errors
from fishing_forecast.features.covariates import build_covariate_features, feature_columns

EXP_ID = "exp2_shap_selection"
SPECIES = "lobster_red"
ECONOMIC_UNIT = "litoral_bc_sur"
CUT_DATE = pd.Timestamp("2020-07-01")
THRESHOLD_FRAC = 0.01  # conservar features con mean|SHAP| >= 1% del total (PLAN §2.3)
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


def _load() -> pd.DataFrame:
    settings = get_settings()
    df = pd.read_parquet(settings.processed_dir / "dataset_v1.parquet")
    sub = df[(df["species"] == SPECIES) & (df["economic_unit"] == ECONOMIC_UNIT)].copy()
    sub["ds"] = pd.to_datetime(sub["ds"])
    sub = sub.sort_values("ds").reset_index(drop=True)
    sub["y"] = sub["y"].fillna(0.0)
    last_catch = sub.loc[sub["y"] > 0, "ds"].max()
    return sub[sub["ds"] <= last_catch].reset_index(drop=True)


def _fit_eval(train, test, cols):
    import xgboost as xgb

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(train[cols], train["y"])
    pred = np.clip(model.predict(test[cols]), 0.0, None)
    daily = all_metrics(test["y"].to_numpy(), pred)
    ss = (
        season_sum_errors(
            pd.DataFrame(
                {
                    "season": test["season"].to_numpy(),
                    "y_true": test["y"].to_numpy(),
                    "y_pred": pred,
                }
            )
        )["pct_error"]
        .round(1)
        .to_dict()
    )
    return model, daily, ss


def main() -> None:
    import shap

    settings = get_settings()
    feat = build_covariate_features(_load(), shift_days=90)
    cols = feature_columns(feat)
    train = feat[feat["ds"] < CUT_DATE]
    test = feat[feat["ds"] >= CUT_DATE]
    logger.info(f"train={len(train)}, test={len(test)}, features={len(cols)}")

    # 1) Modelo completo
    model_full, daily_full, ss_full = _fit_eval(train, test, cols)

    # 2) SHAP sobre train
    explainer = shap.TreeExplainer(model_full)
    sv = explainer.shap_values(train[cols])
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=cols).sort_values(ascending=False)
    frac = mean_abs / mean_abs.sum()

    # 3) Poda por umbral
    kept = frac[frac >= THRESHOLD_FRAC].index.tolist()
    logger.info(f"SHAP: {len(kept)}/{len(cols)} features sobre el umbral {THRESHOLD_FRAC:.0%}")

    # 4) Modelo podado
    _, daily_pruned, ss_pruned = _fit_eval(train, test, kept)

    result = {
        "cut_date": str(CUT_DATE.date()),
        "n_features_full": len(cols),
        "n_features_pruned": len(kept),
        "kept_features": kept,
        "shap_ranking": {k: round(float(v), 4) for k, v in mean_abs.items()},
        "full": {"daily": daily_full, "season_sum": ss_full},
        "pruned": {"daily": daily_pruned, "season_sum": ss_pruned},
    }
    metrics_dir = settings.reports_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{EXP_ID}_{CUT_DATE.date()}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )
    _shap_bar(mean_abs, settings.reports_root / "figures" / f"{EXP_ID}_shap_bar.png")
    _write_summary(result, settings.reports_root / f"{EXP_ID}_summary.md")
    print(_console(result))


def _shap_bar(mean_abs: pd.Series, out_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = mean_abs.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.index, top.to_numpy(), color="#2c7fb8")
    ax.set_title("SHAP mean(|valor|) — top 15 features (langosta-SQ)")
    ax.set_xlabel("mean(|SHAP|)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _console(r: dict) -> str:
    lines = ["", f"SHAP feature selection (langosta-SQ, corte {r['cut_date']}):"]
    lines.append(f"  features: completo={r['n_features_full']}  podado={r['n_features_pruned']}")
    lines.append(f"  conservadas: {', '.join(r['kept_features'])}")
    f, p = r["full"]["daily"], r["pruned"]["daily"]
    lines.append(f"\n  {'':10}{'MAE':>10}{'RMSE':>10}{'sMAPE%':>10}   suma temp. 2021-22")
    lines.append(
        f"  {'completo':10}{f['mae']:>10.1f}{f['rmse']:>10.1f}{f['smape']:>10.1f}   {r['full']['season_sum'].get('2021_2022')}%"
    )
    lines.append(
        f"  {'podado':10}{p['mae']:>10.1f}{p['rmse']:>10.1f}{p['smape']:>10.1f}   {r['pruned']['season_sum'].get('2021_2022')}%"
    )
    return "\n".join(lines)


def _write_summary(r: dict, out_path) -> None:
    f, p = r["full"]["daily"], r["pruned"]["daily"]
    rows = [
        "# Exp 2.3 — Selección de features con SHAP (langosta-SQ)",
        "",
        f"Corte **{r['cut_date']}**. Features: completo **{r['n_features_full']}** → "
        f"podado **{r['n_features_pruned']}** (mean|SHAP| ≥ 1% del total).",
        "",
        "| modelo | n features | MAE | RMSE | sMAPE% | error temp. 2021-22 |",
        "|---|---|---|---|---|---|",
        f"| completo | {r['n_features_full']} | {f['mae']:.1f} | {f['rmse']:.1f} | {f['smape']:.1f} | {r['full']['season_sum'].get('2021_2022')}% |",
        f"| podado | {r['n_features_pruned']} | {p['mae']:.1f} | {p['rmse']:.1f} | {p['smape']:.1f} | {r['pruned']['season_sum'].get('2021_2022')}% |",
        "",
        "Features conservadas (por SHAP): " + ", ".join(r["kept_features"]),
        "",
        "> Figura: `reports/figures/exp2_shap_selection_shap_bar.png`. La poda por SHAP prueba "
        "si reducir features mitiga el sobreajuste observado en Exp 2 (pocas temporadas).",
    ]
    out_path.write_text("\n".join(rows))
    logger.info(f"Resumen → {out_path}")


if __name__ == "__main__":
    main()
