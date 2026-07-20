"""Exp 4 — Pronóstico probabilístico con Conformalized Quantile Regression (CQR).

Envuelve el modelo global ganador de Exp 3.2 (**pool sobre `log1p(y)`**) en una CQR (Romano et
al. 2019) para entregar **intervalos calibrados** en vez de un punto. Motivación: con ~3
temporadas el punto es ruidoso, así que lo honesto —y lo útil para COBI— es cuantificar la
incertidumbre. Se usa `mapie.regression.ConformalizedQuantileRegressor` con regresores cuantílicos
de XGBoost (objetivo `reg:quantileerror`) ajustados en **espacio log** y luego invertidos a kg.

Partición temporal estricta (sin leakage):
- **test**: `ds >= CUT_DATE` (2020-07-01, corte canónico).
- dentro de train: **proper-train** (más antiguo) + **conformalización** (el `CONF_FRAC` final
  de fechas pre-corte) — la calibración conformal solo ve datos anteriores al test.

Métricas (CLAUDE.md Fase 4): cobertura empírica vs nominal (80/90%), ancho medio de intervalo,
CRPS (descomposición pinball sobre rejilla de cuantiles), y **cobertura condicional durante MHW**
(¿el intervalo se mantiene honesto en las temporadas anómalas que rompieron el modelo puntual?).

Uso:
    uv run python experiments/exp4_cqr/cqr_intervals.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from loguru import logger

from fishing_forecast.config import get_settings
from fishing_forecast.evaluation.metrics import (
    coverage,
    crps_from_quantiles,
    mean_interval_width,
)
from fishing_forecast.features.covariates import build_multiseries_features, feature_columns

EXP_ID = "exp4_cqr"
SPECIES = ("lobster_red", "abalone_blue", "abalone_red", "abalone_black", "urchin_red")
GROUP = ["species", "economic_unit"]
CUT_DATE = pd.Timestamp("2020-07-01")
CONF_FRAC = 0.25  # fracción final del periodo pre-corte reservada a conformalización
MIN_CATCH_DAYS = 20
CONF_LEVELS = (0.80, 0.90)  # niveles nominales de los intervalos
CRPS_GRID = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
FOCUS_SERIES = "lobster_red@litoral_bc_sur"  # serie insignia para el fan chart
SEED = 42

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


def load_series() -> pd.DataFrame:
    """Series diarias `(especie, UE)` con captura suficiente; NaN in-season→0, recorte a último catch."""
    settings = get_settings()
    df = pd.read_parquet(settings.processed_dir / "dataset_v1.parquet")
    df = df[df["species"].isin(SPECIES)].copy()
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


def _quantile_model(alpha: float):
    import xgboost as xgb

    return xgb.XGBRegressor(quantile_alpha=alpha, **QUANTILE_XGB)


def _fit_quantile_grid(feat, cols, mask, levels):
    """Ajusta un XGBoost cuantílico por nivel sobre `y_log` (espacio log)."""
    models = {}
    for a in levels:
        m = _quantile_model(a)
        m.fit(feat.loc[mask, cols], feat.loc[mask, "y_log"])
        models[a] = m
    return models


def _mhw_mask(test: pd.DataFrame) -> np.ndarray:
    """Días de ola de calor marina en test (categoría Hobday >= 1)."""
    if "mhw_category" not in test.columns:
        return np.zeros(len(test), dtype=bool)
    return test["mhw_category"].fillna(0).to_numpy() >= 1


def main() -> None:
    from mapie.regression import ConformalizedQuantileRegressor as CQR

    settings = get_settings()
    raw = load_series()
    feat = build_multiseries_features(raw, group_col=GROUP)

    base_cols = feature_columns(feat)  # antes de añadir _series/one-hot/y_log
    feat["_series"] = feat["species"].astype(str) + "@" + feat["economic_unit"].astype(str)
    onehot = pd.get_dummies(feat[GROUP], columns=GROUP)
    feat = pd.concat([feat, onehot], axis=1)
    cols = base_cols + list(onehot.columns)
    feat["y_log"] = np.log1p(feat["y"])

    # Partición temporal: proper-train | conformalización | test.
    pre_cut = feat[feat["ds"] < CUT_DATE]
    conf_start = pre_cut["ds"].quantile(1.0 - CONF_FRAC)
    train_mask = feat["ds"] < conf_start
    conf_mask = (feat["ds"] >= conf_start) & (feat["ds"] < CUT_DATE)
    test = feat[feat["ds"] >= CUT_DATE].copy()
    logger.info(
        f"proper-train={int(train_mask.sum())}, conf={int(conf_mask.sum())} "
        f"(desde {conf_start.date()}), test={len(test)}, features={len(cols)}"
    )

    # Rejilla de cuantiles (para CRPS) + modelos por nivel, en espacio log.
    grid_models = _fit_quantile_grid(feat, cols, train_mask, CRPS_GRID)
    median_model = grid_models[0.5]
    X_conf, y_conf = feat.loc[conf_mask, cols], feat.loc[conf_mask, "y"]
    X_test = test[cols]
    y_test = test["y"].to_numpy()

    # Predicciones de la rejilla en kg (invertir log) → CRPS y punto (mediana).
    grid_pred_kg = {
        q: np.clip(np.expm1(m.predict(X_test)), 0.0, None) for q, m in grid_models.items()
    }
    crps_overall = crps_from_quantiles(y_test, grid_pred_kg)

    # CQR por nivel: [lower, upper, median] prefit en log; conformalizar en kg tras invertir.
    # mapie conformaliza sobre la escala en la que se entrega y_conf/predicción; para mantener
    # todo consistente envolvemos los modelos en un wrapper que invierte a kg al predecir.
    class _InvExpm1:
        def __init__(self, model):
            self.model = model

        def fit(self, *a, **k):  # prefit=True → no se vuelve a ajustar
            return self

        def predict(self, X):
            return np.clip(np.expm1(self.model.predict(X)), 0.0, None)

    intervals: dict[str, dict] = {}
    ci_bounds = {}  # nivel → (lo_arr, hi_arr) en kg para test
    for cl in CONF_LEVELS:
        a_lo, a_hi = (1.0 - cl) / 2.0, 1.0 - (1.0 - cl) / 2.0
        lo_m = _InvExpm1(
            grid_models[a_lo] if a_lo in grid_models else _fit_one(feat, cols, train_mask, a_lo)
        )
        hi_m = _InvExpm1(
            grid_models[a_hi] if a_hi in grid_models else _fit_one(feat, cols, train_mask, a_hi)
        )
        med_m = _InvExpm1(median_model)
        cqr = CQR(estimator=[lo_m, hi_m, med_m], confidence_level=cl, prefit=True)
        cqr.conformalize(X_conf, y_conf)
        _, iv = cqr.predict_interval(X_test)
        lo, hi = iv[:, 0, 0], iv[:, 1, 0]
        lo = np.clip(lo, 0.0, None)
        ci_bounds[cl] = (lo, hi)
        intervals[f"{cl:.2f}"] = {
            "nominal": cl,
            "coverage": round(coverage(y_test, lo, hi), 3),
            "mean_width": round(mean_interval_width(lo, hi), 1),
        }

    # Cobertura condicional MHW vs no-MHW (usando el intervalo al 90%).
    mhw = _mhw_mask(test)
    lo90, hi90 = ci_bounds[0.90]
    cond = {
        "n_mhw_days": int(mhw.sum()),
        "coverage_mhw": round(coverage(y_test[mhw], lo90[mhw], hi90[mhw]), 3)
        if mhw.any()
        else None,
        "coverage_non_mhw": round(coverage(y_test[~mhw], lo90[~mhw], hi90[~mhw]), 3),
    }

    # Cobertura por serie (intervalo al 90%).
    per_series = {}
    for label, idx in test.groupby("_series", observed=True).groups.items():
        pos = test.index.get_indexer(idx)
        per_series[label] = {
            "n_test": len(pos),
            "coverage_0.90": round(coverage(y_test[pos], lo90[pos], hi90[pos]), 3),
            "crps": round(
                crps_from_quantiles(y_test[pos], {q: p[pos] for q, p in grid_pred_kg.items()}), 2
            ),
        }

    summary = {
        "cut_date": str(CUT_DATE.date()),
        "conf_start": str(conf_start.date()),
        "n_test": len(test),
        "crps_overall": round(crps_overall, 2),
        "intervals": intervals,
        "mhw_conditional_0.90": cond,
        "per_series": per_series,
    }
    metrics_dir = settings.reports_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{EXP_ID}_{CUT_DATE.date()}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    _fan_chart(
        test,
        y_test,
        grid_pred_kg,
        ci_bounds,
        settings.reports_root / "figures" / f"{EXP_ID}_fan_chart.png",
    )
    _write_summary(summary, settings.reports_root / f"{EXP_ID}_summary.md")
    print(_console(summary))


def _fit_one(feat, cols, mask, alpha):
    m = _quantile_model(alpha)
    m.fit(feat.loc[mask, cols], feat.loc[mask, "y_log"])
    return m


def _fan_chart(test, y_test, grid_pred_kg, ci_bounds, out_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = test["_series"] == FOCUS_SERIES
    if not sub.any():
        return
    pos = np.where(sub.to_numpy())[0]
    ds = pd.to_datetime(test.loc[sub, "ds"]).to_numpy()
    order = np.argsort(ds)
    ds = ds[order]
    pos = pos[order]
    median = grid_pred_kg[0.5][pos]

    fig, ax = plt.subplots(figsize=(11, 5))
    for cl, color in zip((0.90, 0.80), ("#c6dbef", "#6baed6"), strict=True):
        lo, hi = ci_bounds[cl]
        ax.fill_between(ds, lo[pos], hi[pos], color=color, label=f"intervalo {int(cl * 100)}%")
    ax.plot(ds, median, color="#08519c", lw=1.2, label="mediana pronosticada")
    ax.scatter(ds, y_test[pos], s=8, color="#252525", label="observado", zorder=5)
    ax.set_title(f"CQR — {FOCUS_SERIES} (test desde {CUT_DATE.date()})")
    ax.set_ylabel("captura diaria (kg)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _console(s: dict) -> str:
    lines = ["", f"CQR intervalos calibrados (pool log1p), corte {s['cut_date']}:"]
    lines.append(f"  conformalización desde {s['conf_start']}, test n={s['n_test']}")
    lines.append(f"\n  {'nominal':>8}{'cobertura':>11}{'ancho medio (kg)':>18}")
    for _, iv in s["intervals"].items():
        lines.append(f"  {iv['nominal']:>8.0%}{iv['coverage']:>11.1%}{iv['mean_width']:>18.1f}")
    c = s["mhw_conditional_0.90"]
    lines.append(
        f"\n  cobertura 90% en MHW: {c['coverage_mhw']} ({c['n_mhw_days']} días) "
        f"vs no-MHW: {c['coverage_non_mhw']}"
    )
    lines.append(f"  CRPS global: {s['crps_overall']}")
    lines.append("\n  cobertura 90% por serie:")
    for label, r in s["per_series"].items():
        lines.append(
            f"    {label:<32}{r['coverage_0.90']:>7.1%}  (n={r['n_test']}, CRPS={r['crps']})"
        )
    return "\n".join(lines)


def _write_summary(s: dict, out_path) -> None:
    rows = [
        "# Exp 4 — Pronóstico probabilístico con CQR",
        "",
        f"Corte **{s['cut_date']}**; conformalización desde **{s['conf_start']}**; test n={s['n_test']}. "
        "Modelo: pool global sobre `log1p(y)` (Exp 3.2) envuelto en Conformalized Quantile "
        "Regression (`mapie`), con regresores cuantílicos XGBoost en espacio log invertidos a kg.",
        "",
        "| intervalo nominal | cobertura empírica | ancho medio (kg) |",
        "|---|---|---|",
    ]
    for _, iv in s["intervals"].items():
        rows.append(f"| {iv['nominal']:.0%} | {iv['coverage']:.1%} | {iv['mean_width']:.1f} |")
    c = s["mhw_conditional_0.90"]
    rows += [
        "",
        f"**Cobertura condicional (intervalo 90%)**: durante MHW **{c['coverage_mhw']}** "
        f"({c['n_mhw_days']} días) vs. fuera de MHW **{c['coverage_non_mhw']}**. "
        "Cuanto más cerca del 0.90 nominal en ambos regímenes, más honesto el intervalo en las "
        "temporadas anómalas.",
        "",
        f"**CRPS global**: {s['crps_overall']} (menor es mejor).",
        "",
        "| serie (especie@UE) | n test | cobertura 90% | CRPS |",
        "|---|---|---|---|",
    ]
    for label, r in s["per_series"].items():
        rows.append(f"| {label} | {r['n_test']} | {r['coverage_0.90']:.1%} | {r['crps']} |")
    rows += [
        "",
        "> Figura: `reports/figures/exp4_cqr_fan_chart.png` (bandas 80/90% + mediana + observado "
        f"para {FOCUS_SERIES}). La CQR da la garantía de cobertura marginal sin asumir la forma de "
        "la distribución; es el producto operativo para COBI (rango esperado de captura).",
    ]
    out_path.write_text("\n".join(rows))
    logger.info(f"Resumen → {out_path}")


if __name__ == "__main__":
    main()
