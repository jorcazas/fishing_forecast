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
  de fechas pre-corte, **solo días en temporada**) — la calibración solo ve datos anteriores al
  test. Se calibra en temporada porque los días fuera (captura 0, cuantiles ≈ 0) tienen score
  conformal 0 y, siendo mayoría, anulaban la corrección (fijaban `Q`=0).

Se comparan dos variantes conformal: `split` (corrección constante) y `normalized` (localmente
adaptativa, ensanche ∝ dispersión local) — la normalizada es la de producción.

Métricas (CLAUDE.md Fase 4): cobertura empírica vs nominal (80/90%), ancho de intervalo, CRPS
(descomposición pinball sobre rejilla de cuantiles), y **cobertura condicional durante MHW**
(¿el intervalo se mantiene honesto en las temporadas anómalas que rompieron el modelo puntual?).

Uso:
    uv run python experiments/exp4_cqr/cqr_intervals.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from loguru import logger

from fishing_forecast.config import get_settings
from fishing_forecast.evaluation.conformal import mondrian_cqr, sorted_quantile_preds
from fishing_forecast.evaluation.metrics import coverage, crps_from_quantiles
from fishing_forecast.features.covariates import build_multiseries_features, feature_columns

EXP_ID = "exp4_cqr"
SPECIES = ("lobster_red", "abalone_blue", "abalone_red", "abalone_black", "urchin_red")
GROUP = ["species", "economic_unit"]
#: Corte de test. Canónico 2020-07-01; con los datos 2022-2026 conviene también 2024-06-01
#: (crash + recuperación quedan en train/calibración). Override: `FF_CUT_DATE=2024-06-01`.
CUT_DATE = pd.Timestamp(os.environ.get("FF_CUT_DATE", "2020-07-01"))
CONF_FRAC = 0.25  # fracción final del periodo pre-corte reservada a conformalización
MIN_CATCH_DAYS = 20
CONF_LEVELS = (0.80, 0.90)  # niveles nominales de los intervalos
CRPS_GRID = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
FOCUS_SERIES = "lobster_red@litoral_bc_sur"  # serie insignia para el fan chart individual
GRID_SPECIES = "lobster_red"  # el grid muestra todas las series de esta especie (7 UEs)
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


#: Ablación de composición del pool (reproducibilidad de la progresión 7→10→14→18→21 series):
#: `FF_EXCLUDE_LOBSTER_UES=ue1,ue2,...` excluye esas UEs de langosta del pool (las demás
#: especies/UEs quedan intactas). Vacío → pool completo.
EXCLUDE_LOBSTER_UES = tuple(
    u.strip() for u in os.environ.get("FF_EXCLUDE_LOBSTER_UES", "").split(",") if u.strip()
)


def load_series() -> pd.DataFrame:
    """Series diarias `(especie, UE)` con captura suficiente; NaN in-season→0, recorte a último catch."""
    settings = get_settings()
    df = pd.read_parquet(settings.processed_dir / "dataset_v1.parquet")
    df = df[df["species"].isin(SPECIES)].copy()
    if EXCLUDE_LOBSTER_UES:
        drop = (df["species"] == "lobster_red") & (df["economic_unit"].isin(EXCLUDE_LOBSTER_UES))
        df = df[~drop].copy()
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
    """Días de ola de calor marina en test (categoría Hobday contemporánea >= 1)."""
    if "mhw_now" not in test.columns:
        return np.zeros(len(test), dtype=bool)
    return test["mhw_now"].fillna(0).to_numpy() >= 1


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
    # La conformalización se restringe a días **en temporada**: los días fuera de temporada
    # (captura 0, cuantiles base ≈ 0) tienen score de conformidad exactamente 0 y, al ser mayoría,
    # fijaban el cuantil conformal en 0 (la corrección quedaba anulada). Calibrar solo en
    # temporada hace que `Q` refleje los días de captura, que son los que importan.
    pre_cut = feat[feat["ds"] < CUT_DATE]
    conf_start = pre_cut["ds"].quantile(1.0 - CONF_FRAC)
    train_mask = feat["ds"] < conf_start
    conf_mask = (feat["ds"] >= conf_start) & (feat["ds"] < CUT_DATE) & (feat["in_season"] == 1)
    test = feat[feat["ds"] >= CUT_DATE].copy()
    logger.info(
        f"proper-train={int(train_mask.sum())}, conf(en temporada)={int(conf_mask.sum())} "
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
    grid_log_conf = sorted_quantile_preds(grid_models, X_conf)
    grid_log_test = sorted_quantile_preds(grid_models, X_test)

    # CRPS en kg desde la rejilla (invertir log), y punto = mediana.
    grid_pred_kg = {q: np.clip(np.expm1(p), 0.0, None) for q, p in grid_log_test.items()}
    crps_overall = crps_from_quantiles(y_test, grid_pred_kg)

    # CQR por nivel, **conformalizando en espacio log y por serie (Mondrian)**: la corrección
    # es aditiva en log = multiplicativa en kg, y separada por `(especie, UE)`. Se comparan dos
    # variantes: `split` (constante, ensancha todo por igual → sobre-cubre e infla días pico) y
    # `normalized` (adaptativa, ensanche ∝ dispersión local). La normalizada es la de producción.
    # Se invierten las cotas con expm1 (monótona → intervalos anidados).
    def _interval(cl: float, normalize: bool):
        alpha = 1.0 - cl
        a_lo, a_hi = round(alpha / 2.0, 4), round(1.0 - alpha / 2.0, 4)
        lo_log, hi_log = mondrian_cqr(
            conf_series,
            test_series,
            grid_log_conf[a_lo],
            grid_log_conf[a_hi],
            y_conf_log,
            grid_log_test[a_lo],
            grid_log_test[a_hi],
            alpha,
            normalize=normalize,
        )
        return np.clip(np.expm1(lo_log), 0.0, None), np.clip(np.expm1(hi_log), 0.0, None)

    # El ancho *medio* en kg no aplica (al exponenciar el log unos pocos días pico inflan la
    # media ≫ mediana); se reporta la **mediana** (robusta).
    methods = {"split": False, "normalized": True}
    method_intervals: dict[str, dict] = {m: {} for m in methods}
    prod_bounds = {}  # nivel → (lo, hi) del método de producción (normalized)
    for cl in CONF_LEVELS:
        for name, norm in methods.items():
            lo, hi = _interval(cl, norm)
            width = hi - lo
            method_intervals[name][f"{cl:.2f}"] = {
                "nominal": cl,
                "coverage": round(coverage(y_test, lo, hi), 3),
                "median_width": round(float(np.median(width)), 1),
                "p90_width": round(float(np.percentile(width, 90)), 1),  # días pico
            }
            if norm:
                prod_bounds[cl] = (lo, hi)
    intervals = method_intervals["normalized"]  # producción
    ci_bounds = prod_bounds

    # Cobertura condicional MHW vs no-MHW (usando el intervalo al 90% de producción).
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
        "method_comparison": method_intervals,
        "mhw_conditional_0.90": cond,
        "per_series": per_series,
    }
    metrics_dir = settings.reports_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{EXP_ID}_{CUT_DATE.date()}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    figures_dir = settings.reports_root / "figures"
    _fan_chart(test, y_test, grid_pred_kg, ci_bounds, figures_dir / f"{EXP_ID}_fan_chart.png")
    _fan_grid(
        test,
        y_test,
        grid_pred_kg,
        ci_bounds,
        per_series,
        figures_dir / f"{EXP_ID}_fan_grid_{CUT_DATE.date()}.png",
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
    ax.set_title(
        f"CQR — {FOCUS_SERIES} (test desde {CUT_DATE.date()}; eje y recortado a 3x el máx. observado)"
    )
    ax.set_ylabel("captura diaria (kg)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _fan_grid(test, y_test, grid_pred_kg, ci_bounds, per_series, out_path) -> None:
    """Grid observado-vs-pronosticado (mediana + bandas 80/90%) por serie de `GRID_SPECIES`.

    Muestra el nuevo alcance: las 7 series de langosta (2 → 7 UEs) sobre la línea de tiempo
    ampliada 2017-2026. Cada panel recorta el eje y a 3x el máximo observado (la cota superior
    al 90% se dispara en días pico al exponenciar el espacio log) y anota su cobertura empírica.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = sorted(lbl for lbl in per_series if str(lbl).startswith(GRID_SPECIES + "@"))
    if not labels:
        return
    ncols = 2 if len(labels) <= 4 else 3
    nrows = (len(labels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 3.0 * nrows), squeeze=False)
    lo90, hi90 = ci_bounds[0.90]
    lo80, hi80 = ci_bounds[0.80]

    for ax, label in zip(axes.ravel(), labels, strict=False):
        sub = (test["_series"] == label).to_numpy()
        pos = np.where(sub)[0]
        ds = pd.to_datetime(test.loc[sub, "ds"]).to_numpy()
        order = np.argsort(ds)
        ds, pos = ds[order], pos[order]
        ax.fill_between(ds, lo90[pos], hi90[pos], color="#c6dbef", label="90%")
        ax.fill_between(ds, lo80[pos], hi80[pos], color="#6baed6", label="80%")
        ax.plot(ds, grid_pred_kg[0.5][pos], color="#08519c", lw=1.0, label="mediana")
        ax.scatter(ds, y_test[pos], s=6, color="#252525", label="observado", zorder=5)
        obs_max = float(np.nanmax(y_test[pos])) if len(pos) else 1.0
        ax.set_ylim(-0.05 * obs_max, max(3.0 * obs_max, 1.0))
        cov = per_series[label]["coverage_0.90"]
        ue = label.split("@", 1)[1]
        ax.set_title(f"{ue}  —  cobertura 90%: {cov:.0%}", fontsize=9)
        ax.tick_params(labelsize=7)

    for ax in axes.ravel()[len(labels) :]:
        ax.set_visible(False)
    axes.ravel()[0].legend(loc="upper right", fontsize=7)
    fig.suptitle(
        f"CQR — langosta observada vs. pronosticada, {len(labels)} UEs "
        f"(test desde {CUT_DATE.date()}; eje y recortado a 3x el máx. observado)",
        fontsize=11,
    )
    fig.supylabel("captura diaria (kg)", fontsize=9)
    fig.tight_layout(rect=(0.01, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _console(s: dict) -> str:
    lines = ["", f"CQR intervalos calibrados (pool log1p), corte {s['cut_date']}:"]
    lines.append(f"  conformalización desde {s['conf_start']}, test n={s['n_test']}")
    lines.append(
        f"\n  {'método':>11}{'nominal':>9}{'cobertura':>11}{'ancho med.':>12}{'ancho p90':>12}"
    )
    for name, ivs in s["method_comparison"].items():
        for _, iv in ivs.items():
            lines.append(
                f"  {name:>11}{iv['nominal']:>9.0%}{iv['coverage']:>11.1%}"
                f"{iv['median_width']:>12.1f}{iv['p90_width']:>12.1f}"
            )
    lines.append("  (producción = normalized; ancho en kg)")
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
        "**Comparación de métodos conformal** (`split` = corrección constante; `normalized` = "
        "adaptativa, ensanche ∝ dispersión local; producción = normalized):",
        "",
        "| método | nominal | cobertura marginal | ancho mediano (kg) |",
        "|---|---|---|---|",
    ]
    for name, ivs in s["method_comparison"].items():
        for _, iv in ivs.items():
            rows.append(
                f"| {name} | {iv['nominal']:.0%} | {iv['coverage']:.1%} | {iv['median_width']:.1f} |"
            )
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
        "> Figuras: `reports/figures/exp4_cqr_fan_chart.png` (serie insignia " + FOCUS_SERIES + ") y "
        f"`reports/figures/exp4_cqr_fan_grid_{s['cut_date']}.png` (grid observado-vs-pronosticado de "
        "las 7 UEs de langosta, bandas 80/90% + mediana + observado). La CQR da la garantía de "
        "cobertura marginal sin asumir la forma de la distribución; es el producto operativo para "
        "COBI (rango esperado de captura).",
    ]
    out_path.write_text("\n".join(rows))
    logger.info(f"Resumen → {out_path}")


if __name__ == "__main__":
    main()
