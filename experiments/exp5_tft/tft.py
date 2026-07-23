"""Exp 5 (opcional) — Temporal Fusion Transformer: prueba de techo.

Pregunta (ver `docs/decisions/ADR-0002-tft.md`): con los datos actuales, ¿un TFT supera al
pool XGBoost + CQR (Exp 3.2 / Exp 4)? La hipótesis de trabajo es que **no** ---cada serie
tiene ~200-1 110 días de captura, uno o dos órdenes por debajo del umbral (~10 000 obs/grupo)
donde los Transformers empiezan a pagar---, y que ese resultado sea un hallazgo metodológico.

Modelo **global** sobre las mismas series `(especie, UE)` que Exp 4, con `pytorch-forecasting`:

- objetivo `y` (kg) con `GroupNormalizer(transformation="log1p")` por serie (como Exp 3.2);
- covariables **conocidas a futuro**: calendario (`doy_sin`, `doy_cos`, `in_season`);
- covariables **observadas pasadas** (solo el codificador las ve → sin fuga): SST, MHW,
  color del océano;
- estáticas: especie y unidad económica;
- salida por **cuantiles** (misma rejilla que la CQR) → mismas métricas (cobertura, CRPS),
  reutilizando `evaluation.metrics`. Salidas espejo de Exp 4 para comparar lado a lado.

Partición temporal por `FF_CUT_DATE` (2020-07-01 canónico; 2024-06-01 alterno), igual que Exp 4.

**IMPORTANTE — entrega de código, no de entrenamiento.** El entrenamiento del TFT es largo;
por la regla del proyecto NO se corre sin confirmación. La preparación de datos
(`build_tft_frame`) es pura y está testeada sin `torch`. `main()` importa `torch`/
`pytorch_forecasting` de forma perezosa y avisa si falta el extra `dl`.

Uso (tras confirmar):
    uv sync --extra dl --extra models
    uv run python -m experiments.exp5_tft.tft            # corte canónico 2020-07-01
    FF_CUT_DATE=2024-06-01 uv run python -m experiments.exp5_tft.tft
    FF_TFT_EPOCHS=50 uv run python -m experiments.exp5_tft.tft   # más épocas

Nota de compatibilidad: la API de `predict(..., mode="quantiles", return_index=True)` y el
retorno de `pytorch-forecasting` han variado entre versiones 1.x; la alineación de
predicciones (`_align_quantile_preds`) sigue el recipe canónico y puede requerir un ajuste
menor contra la versión instalada.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from loguru import logger

from fishing_forecast.config import get_settings
from fishing_forecast.evaluation.metrics import all_metrics, coverage, crps_from_quantiles

EXP_ID = "exp5_tft"
SPECIES = ("lobster_red", "abalone_blue", "abalone_red", "abalone_black", "urchin_red")
GROUP = ["species", "economic_unit"]
CUT_DATE = pd.Timestamp(os.environ.get("FF_CUT_DATE", "2020-07-01"))
MIN_CATCH_DAYS = 20
FOCUS_SERIES = "lobster_red@litoral_bc_sur"
SEED = 42

#: Covariables observadas (pasado) presentes en dataset_v1 que el codificador puede ver.
OBSERVED_CANDIDATES = ("sst", "sst_anomaly", "mhw_category", "mhw_intensity",
                       "bbp", "cdm", "chl", "kd490", "spm", "zsd")
KNOWN_REALS = ["doy_sin", "doy_cos", "in_season"]  # conocidas a futuro (calendario)

#: Misma rejilla de cuantiles que la CQR (Exp 4) → CRPS y cobertura comparables.
QUANTILES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
CONF_LEVELS = (0.80, 0.90)

ENC_LEN = 180          # ventana de codificador (días de historia)
PRED_LEN = 30          # horizonte de decodificador (días)
MAX_EPOCHS = int(os.environ.get("FF_TFT_EPOCHS", "30"))
BATCH_SIZE = int(os.environ.get("FF_TFT_BATCH", "256"))


def build_tft_frame(
    df: pd.DataFrame,
    species: tuple[str, ...] = SPECIES,
    min_catch_days: int = MIN_CATCH_DAYS,
) -> tuple[pd.DataFrame, list[str]]:
    """Prepara el frame long para `TimeSeriesDataSet` (PURO, sin `torch`).

    Por serie `(especie, UE)`: ordena por fecha, `y` in-season NaN→0 (ADR-0001), recorta al
    último día con captura, añade `time_idx` entero (días desde la fecha mínima global),
    calendario (`doy_sin`, `doy_cos`), y rellena las covariables observadas (ffill/bfill por
    serie → mediana de columna) para que no queden NaN (los reales del TFT no admiten NaN).

    Devuelve `(frame, observed_cols_presentes)`.
    """
    sub = df[df["species"].isin(species)].copy()
    sub["ds"] = pd.to_datetime(sub["ds"])
    observed = [c for c in OBSERVED_CANDIDATES if c in sub.columns]

    frames: list[pd.DataFrame] = []
    for _key, s in sub.groupby(GROUP, observed=True):
        s = s.sort_values("ds").copy()
        s["y"] = s["y"].fillna(0.0)
        catch = s.loc[s["y"] > 0, "ds"]
        if len(catch) < min_catch_days:
            continue
        s = s[s["ds"] <= catch.max()]
        # covariables sin NaN: ffill/bfill dentro de la serie
        for c in observed:
            s[c] = s[c].ffill().bfill()
        frames.append(s)
    if not frames:
        raise ValueError("Ninguna serie supera MIN_CATCH_DAYS.")
    frame = pd.concat(frames, ignore_index=True)

    # relleno final de cualquier NaN residual de covariable con la mediana de la columna
    for c in observed:
        med = frame[c].median()
        frame[c] = frame[c].fillna(med if pd.notna(med) else 0.0)

    frame["_series"] = frame["species"].astype(str) + "@" + frame["economic_unit"].astype(str)
    day0 = frame["ds"].min()
    frame["time_idx"] = (frame["ds"] - day0).dt.days.astype(int)
    doy = frame["ds"].dt.dayofyear.to_numpy()
    frame["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    frame["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    frame["in_season"] = frame.get("in_season", 0)
    frame["in_season"] = frame["in_season"].astype(float)
    # tipos categóricos para estáticas (pytorch-forecasting los espera como str)
    for c in ("_series", "species", "economic_unit"):
        frame[c] = frame[c].astype(str)
    return frame, observed


def _load() -> pd.DataFrame:
    settings = get_settings()
    path = settings.processed_dir / "dataset_v1.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}. Corre el ETL (transform/aggregate/consolidate).")
    return pd.read_parquet(path)


def _align_quantile_preds(raw, index: pd.DataFrame, frame: pd.DataFrame):
    """Expande las ventanas del decodificador a filas (serie, ds) con predicciones por cuantil.

    `raw` tiene forma [n_ventanas, PRED_LEN, n_cuantiles] (predicciones en kg tras invertir el
    normalizador). `index` (de `return_index=True`) trae, por ventana, la serie y el `time_idx`
    del PRIMER paso del decodificador. Se mapea cada paso k → `time_idx0 + k`, se deduplica
    quedándose con la última predicción por (serie, time_idx) y se une a `frame` por (serie,
    time_idx) para recuperar `ds`, `y` y `mhw_category`.
    """
    q_arr = np.asarray(raw)  # [W, PRED_LEN, Q]
    n_w, horizon, _ = q_arr.shape
    series = index["_series"].to_numpy() if "_series" in index else index.iloc[:, 0].to_numpy()
    t0 = index["time_idx"].to_numpy()
    rows: dict[tuple, np.ndarray] = {}
    for w in range(n_w):
        for k in range(horizon):
            rows[(series[w], int(t0[w] + k))] = q_arr[w, k, :]  # última gana (dedup)
    keys = list(rows.keys())
    pred_df = pd.DataFrame(
        {"_series": [k[0] for k in keys], "time_idx": [k[1] for k in keys]}
    )
    qmat = np.vstack([rows[k] for k in keys])
    # Reordenamiento monótono por fila (Chernozhukov 2010): QuantileLoss no garantiza cuantiles
    # no cruzados, y el cruce produce intervalos no anidados (p. ej. cobertura 80% > 90%).
    # Como QUANTILES está en orden ascendente, ordenar los valores los realinea a sus niveles.
    qmat = np.sort(qmat, axis=1)
    for j, q in enumerate(QUANTILES):
        pred_df[f"q{q}"] = np.clip(qmat[:, j], 0.0, None)
    merged = pred_df.merge(
        frame[["_series", "time_idx", "ds", "y", "mhw_category"]],
        on=["_series", "time_idx"], how="inner",
    )
    return merged


def _metrics(merged: pd.DataFrame) -> dict:
    """Métricas espejo de Exp 4: cobertura 80/90, ancho mediano, CRPS, condicional MHW, por serie."""
    y = merged["y"].to_numpy()
    qcols = {q: merged[f"q{q}"].to_numpy() for q in QUANTILES}
    median = qcols[0.5]
    daily = all_metrics(y, median)

    def interval(cl):
        a = round((1 - cl) / 2, 4)
        return qcols[a], qcols[round(1 - a, 4)]

    intervals = {}
    prod90 = None
    for cl in CONF_LEVELS:
        lo, hi = interval(cl)
        intervals[f"{cl:.2f}"] = {
            "nominal": cl,
            "coverage": round(coverage(y, lo, hi), 3),
            "median_width": round(float(np.median(hi - lo)), 1),
        }
        if cl == 0.90:
            prod90 = (lo, hi)
    lo90, hi90 = prod90
    mhw = merged["mhw_category"].fillna(0).to_numpy() >= 1
    cond = {
        "n_mhw_days": int(mhw.sum()),
        "coverage_mhw": round(coverage(y[mhw], lo90[mhw], hi90[mhw]), 3) if mhw.any() else None,
        "coverage_non_mhw": round(coverage(y[~mhw], lo90[~mhw], hi90[~mhw]), 3),
    }
    per_series = {}
    for label, idx in merged.groupby("_series").groups.items():
        pos = merged.index.get_indexer(idx)
        per_series[label] = {
            "n_test": len(pos),
            "coverage_0.90": round(coverage(y[pos], lo90[pos], hi90[pos]), 3),
            "crps": round(crps_from_quantiles(y[pos], {q: v[pos] for q, v in qcols.items()}), 2),
        }
    return {
        "daily": daily,
        "crps_overall": round(crps_from_quantiles(y, qcols), 2),
        "intervals": intervals,
        "mhw_conditional_0.90": cond,
        "per_series": per_series,
        "n_test": len(merged),
    }


def main() -> None:
    try:
        import lightning.pytorch as pl
        import torch
        from lightning.pytorch.callbacks import EarlyStopping
        from pytorch_forecasting import (
            GroupNormalizer,
            TemporalFusionTransformer,
            TimeSeriesDataSet,
        )
        from pytorch_forecasting.metrics import QuantileLoss
    except ImportError as e:
        raise SystemExit(
            "Faltan dependencias de deep learning. Instala el extra `dl`:\n"
            "    uv sync --extra dl --extra models\n"
            f"(detalle: {e})"
        ) from e

    pl.seed_everything(SEED, workers=True)
    settings = get_settings()
    frame, observed = build_tft_frame(_load())
    logger.info(
        f"{frame['_series'].nunique()} series, {len(frame)} filas, "
        f"covariables observadas: {observed}"
    )

    training_cutoff = int(frame.loc[frame["ds"] < CUT_DATE, "time_idx"].max())
    unknown_reals = ["y", *observed]
    kwargs = dict(
        time_idx="time_idx",
        target="y",
        group_ids=["_series"],
        max_encoder_length=ENC_LEN,
        max_prediction_length=PRED_LEN,
        static_categoricals=["species", "economic_unit"],
        time_varying_known_reals=["time_idx", *KNOWN_REALS],
        time_varying_unknown_reals=unknown_reals,
        target_normalizer=GroupNormalizer(groups=["_series"], transformation="log1p"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    training = TimeSeriesDataSet(frame[frame["time_idx"] <= training_cutoff], **kwargs)
    # Ventanas cuyo decodificador cae después del corte (todo el periodo de prueba).
    test = TimeSeriesDataSet.from_dataset(
        training, frame, min_prediction_idx=training_cutoff + 1, stop_randomization=True
    )
    train_dl = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
    val_dl = test.to_dataloader(train=False, batch_size=BATCH_SIZE * 2, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.01,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=QuantileLoss(quantiles=QUANTILES),
        optimizer="adam",
    )
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        gradient_clip_val=0.1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, mode="min")],
        enable_progress_bar=True,
        logger=False,
        enable_checkpointing=False,
    )
    logger.warning(f"Entrenando TFT ({MAX_EPOCHS} épocas máx.) — puede tardar varios minutos.")
    trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # Predicción por cuantiles en kg (el normalizador invierte log1p) + índice para alinear.
    with torch.no_grad():
        preds = tft.predict(val_dl, mode="quantiles", return_index=True)
    raw = preds.output if hasattr(preds, "output") else preds[0]
    index = preds.index if hasattr(preds, "index") else preds[1]
    raw = raw.cpu().numpy() if hasattr(raw, "cpu") else np.asarray(raw)

    merged = _align_quantile_preds(raw, index, frame)
    result = {"cut_date": str(CUT_DATE.date()), "model": "tft",
              "max_epochs": MAX_EPOCHS, **_metrics(merged)}

    metrics_dir = settings.reports_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{EXP_ID}_{CUT_DATE.date()}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )
    _fan_chart(merged, settings.reports_root / "figures" / f"{EXP_ID}_fan_chart_{CUT_DATE.date()}.png")
    _write_summary(result, settings.reports_root / f"{EXP_ID}_summary.md")
    print(_console(result))


def _fan_chart(merged: pd.DataFrame, out_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = merged[merged["_series"] == FOCUS_SERIES].sort_values("ds")
    if sub.empty:
        return
    ds = pd.to_datetime(sub["ds"]).to_numpy()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(ds, sub["q0.05"], sub["q0.95"], color="#c6dbef", label="intervalo 90%")
    ax.fill_between(ds, sub["q0.1"], sub["q0.9"], color="#6baed6", label="intervalo 80%")
    ax.plot(ds, sub["q0.5"], color="#08519c", lw=1.2, label="mediana TFT")
    ax.scatter(ds, sub["y"], s=8, color="#252525", label="observado", zorder=5)
    obs_max = float(sub["y"].max()) or 1.0
    ax.set_ylim(-0.05 * obs_max, max(3.0 * obs_max, 1.0))
    ax.set_title(f"TFT — {FOCUS_SERIES} (test desde {CUT_DATE.date()}; eje y recortado a 3x máx.)")
    ax.set_ylabel("captura diaria (kg)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _console(r: dict) -> str:
    d, i = r["daily"], r["intervals"]
    lines = [
        "", f"TFT (global, cuantílico), corte {r['cut_date']}, test n={r['n_test']}:",
        f"  MAE={d['mae']:.1f}  RMSE={d['rmse']:.1f}  sMAPE={d['smape']:.1f}%",
        f"  cobertura 80%={i['0.80']['coverage']:.1%}  90%={i['0.90']['coverage']:.1%}",
        f"  CRPS global={r['crps_overall']}",
        "  (comparar con Exp 4 CQR: mismas métricas y rejilla de cuantiles)",
    ]
    return "\n".join(lines)


def _write_summary(r: dict, out_path) -> None:
    d, i = r["daily"], r["intervals"]
    rows = [
        "# Exp 5 — Temporal Fusion Transformer (prueba de techo)",
        "",
        f"Modelo global cuantílico (`pytorch-forecasting`), corte **{r['cut_date']}**, "
        f"{r['max_epochs']} épocas máx., test n={r['n_test']}. Métricas y rejilla de cuantiles "
        "idénticas a Exp 4 (CQR) para comparación directa.",
        "",
        "| métrica | TFT |",
        "|---|---|",
        f"| MAE diario (kg) | {d['mae']:.1f} |",
        f"| RMSE diario (kg) | {d['rmse']:.1f} |",
        f"| sMAPE (%) | {d['smape']:.1f} |",
        f"| cobertura 80% | {i['0.80']['coverage']:.1%} |",
        f"| cobertura 90% | {i['0.90']['coverage']:.1%} |",
        f"| CRPS global | {r['crps_overall']} |",
        "",
        "> Comparar contra `reports/exp4_cqr_summary.md` (mismo corte). Si el TFT no gana, es un "
        "hallazgo metodológico: con ~200-1 100 obs/serie (≪ 10 000) la complejidad extra no paga; "
        "la palanca es más datos. Ver `docs/decisions/ADR-0002-tft.md`.",
    ]
    out_path.write_text("\n".join(rows))
    logger.info(f"Resumen → {out_path}")


if __name__ == "__main__":
    main()
