"""Figura comparativa — langosta@SQ observado vs. pronosticado en los dos cortes.

Acompaña a la §6.10 de la tesis (Tabla `extension_comparativa`): un panel por corte
(2020-07-01 con ~3 temporadas de train; 2024-06-01 con ~7, el bache dentro del train),
mostrando el observado contra el XGBoost completo (35 vars) y el podado por SHAP. Visualiza
por qué más temporadas de entrenamiento reducen el error a la mitad: con 3 temporadas el
modelo extrapola un régimen pre-bache y sobre-predice las temporadas post-2021; con 7 sigue
la recuperación.

Reutiliza las funciones de Exp 2 y la lista de features podadas que Exp 2.3 ya guardó por
corte (`reports/metrics/exp2_shap_selection_{cut}.json`). Corre Exp 2.3 antes si falta.

Uso:
    uv run python -m experiments.exp2_covariates.compare_cuts
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from loguru import logger

from fishing_forecast.config import get_settings
from fishing_forecast.evaluation.metrics import all_metrics
from fishing_forecast.features.covariates import build_covariate_features, feature_columns

from .covariate_model import XGB_PARAMS, load_lobster_sq

CUTS = ("2020-07-01", "2024-06-01")
Y_CAP_FACTOR = 3.0  # recorte del eje y a 3x el máximo observado (el pronóstico se dispara)


def _pruned_features(cut: str, all_cols: list[str]) -> list[str]:
    """Lee las features podadas por SHAP del JSON de Exp 2.3; cae al set completo si falta."""
    p = get_settings().reports_root / "metrics" / f"exp2_shap_selection_{cut}.json"
    if not p.exists():
        logger.warning(f"Falta {p.name}; corre exp2_shap_selection. Uso el set completo.")
        return all_cols
    return json.loads(p.read_text())["kept_features"]


def _fit_predict(train, test, cols):
    import xgboost as xgb

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(train[cols], train["y"])
    return np.clip(model.predict(test[cols]), 0.0, None)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    settings = get_settings()
    feat = build_covariate_features(load_lobster_sq(), shift_days=90)
    cols = feature_columns(feat)

    fig, axes = plt.subplots(len(CUTS), 1, figsize=(13, 8), squeeze=False)
    for ax, cut in zip(axes.ravel(), CUTS, strict=True):
        cut_ts = pd.Timestamp(cut)
        train = feat[feat["ds"] < cut_ts]
        test = feat[feat["ds"] >= cut_ts]
        train_span = f"{train['ds'].min():%Y-%m} → {train['ds'].max():%Y-%m}"

        pred_full = _fit_predict(train, test, cols)
        kept = _pruned_features(cut, cols)
        pred_pruned = _fit_predict(train, test, kept)
        mae_full = all_metrics(test["y"].to_numpy(), pred_full)["mae"]
        mae_pruned = all_metrics(test["y"].to_numpy(), pred_pruned)["mae"]

        ax.plot(test["ds"], test["y"], color="#222", lw=0.9, label="observado", zorder=5)
        ax.plot(test["ds"], pred_full, color="#c0392b", lw=1.1, alpha=0.9,
                label=f"XGBoost 35 vars (MAE {mae_full:.0f})")
        ax.plot(test["ds"], pred_pruned, color="#1a9850", lw=1.1, alpha=0.9,
                label=f"XGBoost SHAP {len(kept)} vars (MAE {mae_pruned:.0f})")
        obs_max = float(test["y"].max()) or 1.0
        ax.set_ylim(-0.05 * obs_max, Y_CAP_FACTOR * obs_max)
        ax.set_title(
            f"Corte {cut} — entrenamiento {train_span} "
            f"(test n={len(test)}; eje y recortado a {Y_CAP_FACTOR:.0f}x el máx. observado)",
            fontsize=10,
        )
        ax.set_ylabel("captura diaria (kg)")
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Langosta @ San Quintín: observado vs. pronosticado según temporadas de entrenamiento",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = settings.reports_root / "figures" / "exp2_comparativa_cuts.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info(f"Figura → {out}")


if __name__ == "__main__":
    main()
