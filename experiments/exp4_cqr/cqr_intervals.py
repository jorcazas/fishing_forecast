"""Exp 4 — Pronóstico probabilístico con Conformalized Quantile Regression (CQR).

Envuelve el modelo global ganador de Exp 3.2 (**pool sobre `log1p(y)`**) en una CQR (Romano et
al. 2019) para entregar **intervalos calibrados** en vez de un punto. Motivación: con ~3
temporadas el punto es ruidoso, así que lo honesto —y lo útil para COBI— es cuantificar la
incertidumbre. Se usan regresores cuantílicos de XGBoost (objetivo `reg:quantileerror`) en
**espacio log** y una CQR de conformalización dividida **por serie (Mondrian)**, con las cotas
invertidas a kg. (Se prefirió la CQR explícita sobre `mapie` porque con modelos cuantílicos
independientes + inversión `expm1` producía intervalos no anidados; la versión propia garantiza
anidamiento y calibración por serie, en línea con la normalización de `y` de Exp 3.2.)

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
from fishing_forecast.evaluation.metrics import coverage, crps_from_quantiles
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


def _sorted_quantile_preds(grid_models: dict, X) -> dict[float, np.ndarray]:
    """Predice cada cuantil de la rejilla y reordena por fila para eliminar el cruce de cuantiles.

    Regresores cuantílicos independientes pueden cruzarse (q0.1 > q0.9 en algunas filas). Se
    ordena cada fila de forma ascendente (rearrangement, Chernozhukov et al. 2010), que preserva
    la cobertura y garantiza monotonía → intervalos anidados entre niveles.
    """
    levels = sorted(grid_models)
    preds = np.column_stack([grid_models[q].predict(X) for q in levels])
    preds = np.sort(preds, axis=1)
    return {q: preds[:, i] for i, q in enumerate(levels)}


def _mhw_mask(test: pd.DataFrame) -> np.ndarray:
    """Días de ola de calor marina en test (categoría Hobday contemporánea >= 1)."""
    if "mhw_now" not in test.columns:
        return np.zeros(len(test), dtype=bool)
    return test["mhw_now"].fillna(0).to_numpy() >= 1


def _split_cqr(
    lo_conf: np.ndarray,
    hi_conf: np.ndarray,
    y_conf: np.ndarray,
    lo_test: np.ndarray,
    hi_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Split-CQR (Romano et al. 2019): ajusta las cotas cuantílicas con la corrección conformal.

    Trabaja en la escala en que se le pasan las cotas (aquí, espacio log). El score de
    conformidad es `E_i = max(lo(x_i) - y_i, y_i - hi(x_i))`; se toma el cuantil finito-muestral
    `ceil((n+1)(1-alpha))/n` de los scores y se ensancha simétricamente. Garantiza cobertura
    marginal >= 1-alpha e intervalos **anidados** entre niveles (alpha mayor → Q menor).
    """
    scores = np.maximum(lo_conf - y_conf, y_conf - hi_conf)
    n = len(scores)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)  # clip al rango válido
    q = np.sort(scores)[k - 1]
    return lo_test - q, hi_test + q


def _mondrian_cqr(
    conf_series: np.ndarray,
    test_series: np.ndarray,
    lo_conf: np.ndarray,
    hi_conf: np.ndarray,
    y_conf: np.ndarray,
    lo_test: np.ndarray,
    hi_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Split-CQR **por serie** (Mondrian): una corrección conformal por `(especie, UE)`.

    Un único `Q` global lo fija la serie peor calibrada y, al invertir el log, infla las series
    de gran escala (langosta). Calibrar por serie da cobertura ~nominal *dentro* de cada serie y
    anchos acordes a su escala — el análogo conformal de la normalización de `y` de Exp 3.2.
    """
    lo_out = np.empty_like(lo_test, dtype=float)
    hi_out = np.empty_like(hi_test, dtype=float)
    for s in np.unique(test_series):
        c, t = conf_series == s, test_series == s
        if not c.any():  # sin puntos de conformalización → sin corrección (cae al cuantil crudo)
            lo_out[t], hi_out[t] = lo_test[t], hi_test[t]
            continue
        lo_out[t], hi_out[t] = _split_cqr(
            lo_conf[c], hi_conf[c], y_conf[c], lo_test[t], hi_test[t], alpha
        )
    return lo_out, hi_out


def main() -> None:
    settings = get_settings()
    raw = load_series()
    feat = build_multiseries_features(raw, group_col=GROUP)

    base_cols = feature_columns(feat)  # antes de añadir _series/one-hot/y_log
    feat["_series"] = feat["species"].astype(str) + "@" + feat["economic_unit"].astype(str)
    # MHW contemporáneo (no es feature, solo para la cobertura condicional): el builder de
    # covariables solo deja los MHW *desplazados*, así que traemos el estado del día objetivo.
    mhw_now = raw[[*GROUP, "ds", "mhw_category"]].rename(columns={"mhw_category": "mhw_now"})
    feat = feat.merge(mhw_now, on=[*GROUP, "ds"], how="left")
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
    X_conf = feat.loc[conf_mask, cols]
    y_conf_log = feat.loc[conf_mask, "y_log"].to_numpy()  # se conformaliza en espacio log
    conf_series = feat.loc[conf_mask, "_series"].to_numpy()
    X_test = test[cols]
    y_test = test["y"].to_numpy()
    test_series = test["_series"].to_numpy()

    # Predicciones cuantílicas en espacio log (conf y test). Se corrige el cruce de cuantiles
    # (regresores independientes) reordenando cada fila de forma monótona (Chernozhukov 2010).
    grid_log_conf = _sorted_quantile_preds(grid_models, X_conf)
    grid_log_test = _sorted_quantile_preds(grid_models, X_test)

    # CRPS en kg desde la rejilla (invertir log), y punto = mediana.
    grid_pred_kg = {q: np.clip(np.expm1(p), 0.0, None) for q, p in grid_log_test.items()}
    crps_overall = crps_from_quantiles(y_test, grid_pred_kg)

    # CQR por nivel, **conformalizando en espacio log y por serie (Mondrian)**: la corrección
    # es aditiva en log = multiplicativa en kg, y separada por `(especie, UE)` para que la escala
    # de langosta no infle a abulón. Se invierten las cotas con expm1 (monótona).
    intervals: dict[str, dict] = {}
    ci_bounds = {}  # nivel → (lo_arr, hi_arr) en kg para test
    for cl in CONF_LEVELS:
        alpha = 1.0 - cl
        a_lo, a_hi = round(alpha / 2.0, 4), round(1.0 - alpha / 2.0, 4)
        lo_log, hi_log = _mondrian_cqr(
            conf_series,
            test_series,
            grid_log_conf[a_lo],
            grid_log_conf[a_hi],
            y_conf_log,
            grid_log_test[a_lo],
            grid_log_test[a_hi],
            alpha,
        )
        lo = np.clip(np.expm1(lo_log), 0.0, None)
        hi = np.clip(np.expm1(hi_log), 0.0, None)
        ci_bounds[cl] = (lo, hi)
        # El ancho *medio* en kg no tiene sentido aquí: al exponenciar el espacio log unos pocos
        # días pico inflan la media (media >> mediana). Se reporta la **mediana** (robusta).
        intervals[f"{cl:.2f}"] = {
            "nominal": cl,
            "coverage": round(coverage(y_test, lo, hi), 3),
            "median_width": round(float(np.median(hi - lo)), 1),
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
            "median_width_0.90": round(float(np.median(hi90[pos] - lo90[pos])), 1),
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
    # La cota superior al 90% se dispara en días pico (expansión exponencial del espacio log);
    # se recorta el eje para que observado/mediana/80% sean legibles (el 90% se sale por arriba).
    obs_max = float(np.nanmax(y_test[pos])) if len(pos) else 1.0
    ax.set_ylim(-0.05 * obs_max, max(3.0 * obs_max, 1.0))
    ax.set_title(f"CQR — {FOCUS_SERIES} (test desde {CUT_DATE.date()}; eje y recortado a 3x el máx. observado)")
    ax.set_ylabel("captura diaria (kg)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _console(s: dict) -> str:
    lines = ["", f"CQR intervalos calibrados (pool log1p), corte {s['cut_date']}:"]
    lines.append(f"  conformalización desde {s['conf_start']}, test n={s['n_test']}")
    lines.append(f"\n  {'nominal':>8}{'cobertura':>11}{'ancho mediano (kg)':>20}")
    for _, iv in s["intervals"].items():
        lines.append(f"  {iv['nominal']:>8.0%}{iv['coverage']:>11.1%}{iv['median_width']:>20.1f}")
    c = s["mhw_conditional_0.90"]
    lines.append(
        f"\n  cobertura 90% en MHW: {c['coverage_mhw']} ({c['n_mhw_days']} días) "
        f"vs no-MHW: {c['coverage_non_mhw']}"
    )
    lines.append(f"  CRPS global: {s['crps_overall']}")
    lines.append("\n  cobertura 90% por serie (ancho mediano kg):")
    for label, r in s["per_series"].items():
        lines.append(
            f"    {label:<32}{r['coverage_0.90']:>7.1%}  (n={r['n_test']}, "
            f"ancho~{r['median_width_0.90']}, CRPS={r['crps']})"
        )
    return "\n".join(lines)


def _write_summary(s: dict, out_path) -> None:
    rows = [
        "# Exp 4 — Pronóstico probabilístico con CQR",
        "",
        f"Corte **{s['cut_date']}**; conformalización desde **{s['conf_start']}**; test n={s['n_test']}. "
        "Modelo: pool global sobre `log1p(y)` (Exp 3.2) con Conformalized Quantile Regression "
        "**por serie (Mondrian)** en espacio log, invertida a kg. Se reporta el ancho **mediano** "
        "(la media en kg la inflan unos pocos días pico al exponenciar el espacio log).",
        "",
        "| intervalo nominal | cobertura empírica marginal | ancho mediano (kg) |",
        "|---|---|---|",
    ]
    for _, iv in s["intervals"].items():
        rows.append(f"| {iv['nominal']:.0%} | {iv['coverage']:.1%} | {iv['median_width']:.1f} |")
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
        "| serie (especie@UE) | n test | cobertura 90% | ancho mediano (kg) | CRPS |",
        "|---|---|---|---|---|",
    ]
    for label, r in s["per_series"].items():
        rows.append(
            f"| {label} | {r['n_test']} | {r['coverage_0.90']:.1%} | "
            f"{r['median_width_0.90']} | {r['crps']} |"
        )
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
