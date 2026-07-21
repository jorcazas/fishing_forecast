"""Exp 4b — Endurecer los modelos cuantílicos base de la CQR (tightening del ancho en días pico).

En Exp 4 la corrección conformal resultó casi nula: el ancho ---y su explosión en días pico---
lo fija el **modelo cuantílico base** (q0.95 en escala log, amplificado al invertir con expm1),
no el envoltorio conformal. Aquí se **afinan los cuantílicos con Optuna** minimizando la pérdida
pinball (regla de puntaje propia que penaliza tanto intervalos demasiado anchos como demasiado
angostos), y se comprueba si el ancho p90 (días pico) baja **sin** perder cobertura.

Partición (sin tocar test):
- **tuning**: se entrena en `ds < TUNE_VAL_START` y se valida la pinball en el `TUNE_VAL_FRAC`
  final del periodo pre-corte (solo días en temporada). Los hiperparámetros se eligen aquí.
- **evaluación final**: partición canónica de Exp 4 (proper-train | conformalización en
  temporada | test ≥ 2020-07-01). Se compara **default vs afinado** en cobertura y ancho.

Uso (como módulo, desde la raíz del repo):
    uv run python -m experiments.exp4_cqr.tune_quantiles
"""

from __future__ import annotations

import json

import experiments.exp4_cqr.cqr_intervals as C
import numpy as np
import optuna
import pandas as pd
from loguru import logger

from fishing_forecast.config import get_settings
from fishing_forecast.evaluation.conformal import mondrian_cqr, sorted_quantile_preds
from fishing_forecast.evaluation.metrics import coverage, crps_from_quantiles, pinball_loss
from fishing_forecast.features.covariates import build_multiseries_features, feature_columns

EXP_ID = "exp4_cqr_tuned"
N_TRIALS = 40
TUNE_VAL_FRAC = 0.30  # 30% final del pre-corte (en temporada) para validar la pinball

#: Hiperparámetros de producción de Exp 4 (subconjunto ajustable de `C.QUANTILE_XGB`).
DEFAULT_PARAMS = dict(
    max_depth=3,
    learning_rate=0.05,
    n_estimators=200,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=1.0,
)


def _xgb_quantile(alpha: float, params: dict):
    import xgboost as xgb

    return xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=alpha,
        random_state=C.SEED,
        n_jobs=4,
        **params,
    )


def _fit_grid(feat, cols, mask, levels, params: dict) -> dict:
    return {
        a: _xgb_quantile(a, params).fit(feat.loc[mask, cols], feat.loc[mask, "y_log"])
        for a in levels
    }


def _mean_pinball(feat, cols, grid_models, mask, levels) -> float:
    """Pinball media sobre la rejilla de cuantiles, en espacio log (consistente con el ajuste)."""
    preds = sorted_quantile_preds(grid_models, feat.loc[mask, cols])
    y = feat.loc[mask, "y_log"].to_numpy()
    return float(np.mean([pinball_loss(y, preds[a], a) for a in levels]))


def _build_features():
    settings = get_settings()
    raw = C.load_series()
    feat = build_multiseries_features(raw, group_col=C.GROUP)
    base_cols = feature_columns(feat)
    feat["_series"] = feat["species"].astype(str) + "@" + feat["economic_unit"].astype(str)
    onehot = pd.get_dummies(feat[C.GROUP], columns=C.GROUP)
    feat = pd.concat([feat, onehot], axis=1)
    cols = base_cols + list(onehot.columns)
    feat["y_log"] = np.log1p(feat["y"])
    return settings, feat, cols


def _objective(trial, feat, cols, tune_train, tune_val) -> float:
    params = dict(
        max_depth=trial.suggest_int("max_depth", 2, 5),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
        min_child_weight=trial.suggest_float("min_child_weight", 1.0, 30.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
    )
    grid = _fit_grid(feat, cols, tune_train, C.CRPS_GRID, params)
    return _mean_pinball(feat, cols, grid, tune_val, C.CRPS_GRID)


def _evaluate(feat, cols, params: dict, conf_start) -> dict:
    """Pipeline CQR de Exp 4 (Mondrian, normalizado, calibrado en temporada) con `params` dados."""
    train_mask = feat["ds"] < conf_start
    conf_mask = (feat["ds"] >= conf_start) & (feat["ds"] < C.CUT_DATE) & (feat["in_season"] == 1)
    test = feat[feat["ds"] >= C.CUT_DATE]
    grid = _fit_grid(feat, cols, train_mask, C.CRPS_GRID, params)
    gc = sorted_quantile_preds(grid, feat.loc[conf_mask, cols])
    gt = sorted_quantile_preds(grid, test[cols])
    ycl = feat.loc[conf_mask, "y_log"].to_numpy()
    cser = feat.loc[conf_mask, "_series"].to_numpy()
    tser = test["_series"].to_numpy()
    y_test = test["y"].to_numpy()

    out: dict = {"intervals": {}, "per_series": {}}
    ci = {}
    for cl in C.CONF_LEVELS:
        alpha = 1.0 - cl
        a_lo, a_hi = round(alpha / 2.0, 4), round(1.0 - alpha / 2.0, 4)
        lo_log, hi_log = mondrian_cqr(
            cser, tser, gc[a_lo], gc[a_hi], ycl, gt[a_lo], gt[a_hi], alpha, normalize=True
        )
        lo = np.clip(np.expm1(lo_log), 0.0, None)
        hi = np.clip(np.expm1(hi_log), 0.0, None)
        w = hi - lo
        ci[cl] = (lo, hi)
        out["intervals"][f"{cl:.2f}"] = {
            "nominal": cl,
            "coverage": round(coverage(y_test, lo, hi), 3),
            "median_width": round(float(np.median(w)), 1),
            "p90_width": round(float(np.percentile(w, 90)), 1),
        }
    grid_kg = {a: np.clip(np.expm1(gt[a]), 0.0, None) for a in C.CRPS_GRID}
    out["crps"] = round(crps_from_quantiles(y_test, grid_kg), 2)
    lo90, hi90 = ci[0.90]
    for s in np.unique(tser):
        m = tser == s
        out["per_series"][s] = {
            "coverage_0.90": round(coverage(y_test[m], lo90[m], hi90[m]), 3),
            "p90_width": round(float(np.percentile((hi90 - lo90)[m], 90)), 1),
        }
    return out


def main() -> None:
    settings, feat, cols = _build_features()

    pre_cut = feat[feat["ds"] < C.CUT_DATE]
    tune_val_start = pre_cut["ds"].quantile(1.0 - TUNE_VAL_FRAC)
    conf_start = pre_cut["ds"].quantile(1.0 - C.CONF_FRAC)
    tune_train = feat["ds"] < tune_val_start
    tune_val = (feat["ds"] >= tune_val_start) & (feat["ds"] < C.CUT_DATE) & (feat["in_season"] == 1)
    logger.info(
        f"tune_train={int(tune_train.sum())}, tune_val(en temporada)={int(tune_val.sum())} "
        f"(desde {tune_val_start.date()}); conf_start={conf_start.date()}"
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=C.SEED)
    )
    study.optimize(lambda t: _objective(t, feat, cols, tune_train, tune_val), n_trials=N_TRIALS)
    best_params = study.best_params
    logger.info(f"best pinball(val)={study.best_value:.4f}  params={best_params}")

    # Pinball de validación del default, para contexto.
    default_grid = _fit_grid(feat, cols, tune_train, C.CRPS_GRID, DEFAULT_PARAMS)
    default_val_pinball = _mean_pinball(feat, cols, default_grid, tune_val, C.CRPS_GRID)

    # Evaluación final (test) default vs afinado.
    eval_default = _evaluate(feat, cols, DEFAULT_PARAMS, conf_start)
    eval_tuned = _evaluate(feat, cols, best_params, conf_start)

    result = {
        "cut_date": str(C.CUT_DATE.date()),
        "n_trials": N_TRIALS,
        "val_pinball_default": round(default_val_pinball, 4),
        "val_pinball_tuned": round(study.best_value, 4),
        "best_params": best_params,
        "default": eval_default,
        "tuned": eval_tuned,
    }
    metrics_dir = settings.reports_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{EXP_ID}_{C.CUT_DATE.date()}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )
    _write_summary(result, settings.reports_root / f"{EXP_ID}_summary.md")
    print(_console(result))


def _fmt_intervals(ev: dict) -> list[str]:
    return [
        f"    {iv['nominal']:>4.0%}  cob={iv['coverage']:>6.1%}  "
        f"ancho med={iv['median_width']:>7.1f}  ancho p90={iv['p90_width']:>9.1f}"
        for iv in ev["intervals"].values()
    ]


def _console(r: dict) -> str:
    lines = ["", f"Afinado de cuantílicos (Optuna, {r['n_trials']} trials), corte {r['cut_date']}:"]
    lines.append(
        f"  pinball(val, log): default={r['val_pinball_default']}  tuned={r['val_pinball_tuned']}"
    )
    lines.append(f"  best params: {r['best_params']}")
    lines.append(f"\n  DEFAULT (Exp 4)   CRPS={r['default']['crps']}")
    lines += _fmt_intervals(r["default"])
    lines.append(f"\n  AFINADO           CRPS={r['tuned']['crps']}")
    lines += _fmt_intervals(r["tuned"])
    lines.append("\n  cobertura 90% / ancho p90 por serie (default → afinado):")
    for s in r["default"]["per_series"]:
        d, t = r["default"]["per_series"][s], r["tuned"]["per_series"][s]
        lines.append(
            f"    {s:<32}{d['coverage_0.90']:>6.1%}→{t['coverage_0.90']:<6.1%}  "
            f"p90 {d['p90_width']:>8.1f}→{t['p90_width']:<8.1f}"
        )
    return "\n".join(lines)


def _write_summary(r: dict, out_path) -> None:
    rows = [
        "# Exp 4b — Endurecer los cuantílicos base de la CQR (Optuna sobre pinball)",
        "",
        f"Corte **{r['cut_date']}**. Optuna ({r['n_trials']} trials) minimizando la pérdida "
        f"pinball de validación (log): **{r['val_pinball_default']}** (default) → "
        f"**{r['val_pinball_tuned']}** (afinado).",
        "",
        f"`best_params`: `{r['best_params']}`",
        "",
        "| variante | nivel | cobertura | ancho mediano (kg) | ancho p90 (kg) | CRPS |",
        "|---|---|---|---|---|---|",
    ]
    for name, ev in (("default", r["default"]), ("afinado", r["tuned"])):
        for iv in ev["intervals"].values():
            rows.append(
                f"| {name} | {iv['nominal']:.0%} | {iv['coverage']:.1%} | "
                f"{iv['median_width']:.1f} | {iv['p90_width']:.1f} | {ev['crps']} |"
            )
    rows += [
        "",
        "**Cobertura 90% y ancho p90 por serie (default → afinado):**",
        "",
        "| serie | cobertura 90% | ancho p90 (kg) |",
        "|---|---|---|",
    ]
    for s in r["default"]["per_series"]:
        d, t = r["default"]["per_series"][s], r["tuned"]["per_series"][s]
        rows.append(
            f"| {s} | {d['coverage_0.90']:.1%} → {t['coverage_0.90']:.1%} | "
            f"{d['p90_width']:.1f} → {t['p90_width']:.1f} |"
        )
    rows += [
        "",
        "> El ancho p90 (días pico) es el objetivo: afinar los cuantílicos con pinball busca "
        "estrecharlo sin perder cobertura. El objetivo se valida en el 30% final del pre-corte; "
        "test intacto (≥ corte).",
    ]
    out_path.write_text("\n".join(rows))
    logger.info(f"Resumen → {out_path}")


if __name__ == "__main__":
    main()
