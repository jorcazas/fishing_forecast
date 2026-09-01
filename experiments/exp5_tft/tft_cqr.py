"""Exp 5b — TFT + CQR: comparación JUSTA de cobertura (TFT+conformal vs XGBoost+conformal).

Exp 5 mostró que los cuantiles crudos del TFT no están calibrados (cobertura ~6%). Pero la
CQR (Exp 4) tampoco usa los cuantiles crudos del XGBoost: los envuelve en una corrección
conformal por serie, calibrada en temporada. La comparación justa, entonces, es aplicar
**la misma** envoltura conformal a los cuantiles del TFT y medir:

- **cobertura** → debería quedar cerca del nominal para AMBOS (esa es la garantía conformal);
- **ancho del intervalo** → el discriminador real: a igual cobertura, cuantiles base más
  nítidos dan intervalos más estrechos. Si el TFT necesita intervalos más anchos que el
  XGBoost para la misma cobertura, sus cuantiles base son peores.

Reutiliza `evaluation.conformal.mondrian_cqr` (la misma de Exp 4) y la preparación/entrenamiento
de `exp5_tft.tft`. El TFT se entrena SOLO hasta el inicio de la conformalización (`conf_start`)
para que el conjunto de calibración sea held-out (validez conformal), igual que el proper-train
de Exp 4. Métricas y rejilla de cuantiles idénticas a Exp 4.

**IMPORTANTE**: reentrena el TFT (largo); no correr sin confirmar. Default corte 2024-06-01.

Uso (tras confirmar):
    uv run python -m experiments.exp5_tft.tft_cqr
    FF_CUT_DATE=2024-06-01 FF_TFT_EPOCHS=8 uv run python -m experiments.exp5_tft.tft_cqr
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from loguru import logger

from fishing_forecast.config import get_settings
from fishing_forecast.evaluation.conformal import mondrian_cqr
from fishing_forecast.evaluation.metrics import all_metrics, coverage, crps_from_quantiles

from .tft import (
    BATCH_SIZE,
    ENC_LEN,
    KNOWN_REALS,
    MAX_EPOCHS,
    PRED_LEN,
    QUANTILES,
    SEED,
    _align_quantile_preds,
    _load,
    build_tft_frame,
)

EXP_ID = "exp5_tft_cqr"
CUT_DATE = pd.Timestamp(os.environ.get("FF_CUT_DATE", "2024-06-01"))
CONF_FRAC = 0.25  # fracción final del pre-corte reservada a conformalización (como Exp 4)
CONF_LEVELS = (0.80, 0.90)
FOCUS_SERIES = "lobster_red@litoral_bc_sur"


def _conformalize(conf: pd.DataFrame, test: pd.DataFrame):
    """Aplica CQR Mondrian (en espacio log, normalizada) a los cuantiles del TFT, por nivel.

    Devuelve `(intervals_dict, (lo90, hi90))` con las cotas en kg del método de producción.
    """
    y_conf_log = np.log1p(conf["y"].to_numpy())
    conf_series = conf["_series"].to_numpy()
    test_series = test["_series"].to_numpy()
    y_test = test["y"].to_numpy()

    intervals: dict[str, dict] = {}
    prod90 = None
    for cl in CONF_LEVELS:
        a = round((1 - cl) / 2, 4)
        a_hi = round(1 - a, 4)
        lo_c, hi_c = np.log1p(conf[f"q{a}"].to_numpy()), np.log1p(conf[f"q{a_hi}"].to_numpy())
        lo_t, hi_t = np.log1p(test[f"q{a}"].to_numpy()), np.log1p(test[f"q{a_hi}"].to_numpy())
        lo_log, hi_log = mondrian_cqr(
            conf_series, test_series, lo_c, hi_c, y_conf_log, lo_t, hi_t, 1 - cl, normalize=True
        )
        lo, hi = np.clip(np.expm1(lo_log), 0.0, None), np.clip(np.expm1(hi_log), 0.0, None)
        intervals[f"{cl:.2f}"] = {
            "nominal": cl,
            "coverage": round(coverage(y_test, lo, hi), 3),
            "median_width": round(float(np.median(hi - lo)), 1),
            "p90_width": round(float(np.percentile(hi - lo, 90)), 1),
        }
        if cl == 0.90:
            prod90 = (lo, hi)
    return intervals, prod90


def _metrics(test: pd.DataFrame, intervals: dict, prod90) -> dict:
    y = test["y"].to_numpy()
    qcols = {q: test[f"q{q}"].to_numpy() for q in QUANTILES}
    daily = all_metrics(y, qcols[0.5])
    lo90, hi90 = prod90
    mhw = test["mhw_category"].fillna(0).to_numpy() >= 1
    cond = {
        "n_mhw_days": int(mhw.sum()),
        "coverage_mhw": round(coverage(y[mhw], lo90[mhw], hi90[mhw]), 3) if mhw.any() else None,
        "coverage_non_mhw": round(coverage(y[~mhw], lo90[~mhw], hi90[~mhw]), 3),
    }
    per_series = {}
    for label, idx in test.groupby("_series").groups.items():
        pos = test.index.get_indexer(idx)
        per_series[label] = {
            "n_test": len(pos),
            "coverage_0.90": round(coverage(y[pos], lo90[pos], hi90[pos]), 3),
            "median_width_0.90": round(float(np.median(hi90[pos] - lo90[pos])), 1),
            "crps": round(crps_from_quantiles(y[pos], {q: v[pos] for q, v in qcols.items()}), 2),
        }
    return {
        "daily": daily,
        "crps_overall": round(crps_from_quantiles(y, qcols), 2),
        "intervals": intervals,
        "mhw_conditional_0.90": cond,
        "per_series": per_series,
        "n_test": len(test),
        "n_conf": None,
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
            "Faltan dependencias de deep learning. Instala:\n"
            "    uv sync --extra dl --extra models\n"
            f"(detalle: {e})"
        ) from e

    pl.seed_everything(SEED, workers=True)
    settings = get_settings()
    frame, observed = build_tft_frame(_load())

    # Partición como Exp 4: proper-train < conf_start | conf en [conf_start, cut) | test >= cut.
    pre = frame[frame["ds"] < CUT_DATE]
    conf_start = pre["ds"].quantile(1.0 - CONF_FRAC)
    train_cutoff = int(frame.loc[frame["ds"] < conf_start, "time_idx"].max())
    logger.info(
        f"{frame['_series'].nunique()} series; TFT entrena <{conf_start.date()}; "
        f"conformaliza en [{conf_start.date()}, {CUT_DATE.date()}); test >= {CUT_DATE.date()}"
    )

    unknown_reals = ["y", *observed]
    kwargs = dict(
        time_idx="time_idx", target="y", group_ids=["_series"],
        max_encoder_length=ENC_LEN, max_prediction_length=PRED_LEN,
        static_categoricals=["species", "economic_unit"],
        time_varying_known_reals=["time_idx", *KNOWN_REALS],
        time_varying_unknown_reals=unknown_reals,
        target_normalizer=GroupNormalizer(groups=["_series"], transformation="log1p"),
        add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    training = TimeSeriesDataSet(frame[frame["time_idx"] <= train_cutoff], **kwargs)
    # Ventanas cuyo decodificador cae en el periodo de conformalización + prueba.
    predict_ds = TimeSeriesDataSet.from_dataset(
        training, frame, min_prediction_idx=train_cutoff + 1, stop_randomization=True
    )
    train_dl = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
    val_dl = predict_ds.to_dataloader(train=False, batch_size=BATCH_SIZE * 2, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training, learning_rate=0.01, hidden_size=16, attention_head_size=2, dropout=0.1,
        hidden_continuous_size=8, loss=QuantileLoss(quantiles=QUANTILES), optimizer="adam",
    )
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS, accelerator="auto", gradient_clip_val=0.1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, mode="min")],
        enable_progress_bar=True, logger=False, enable_checkpointing=False,
    )
    logger.warning(f"Entrenando TFT ({MAX_EPOCHS} épocas máx.) — puede tardar varios minutos.")
    trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

    with torch.no_grad():
        preds = tft.predict(val_dl, mode="quantiles", return_index=True)
    raw = preds.output if hasattr(preds, "output") else preds[0]
    index = preds.index if hasattr(preds, "index") else preds[1]
    raw = raw.cpu().numpy() if hasattr(raw, "cpu") else np.asarray(raw)

    merged = _align_quantile_preds(raw, index, frame).merge(
        frame[["_series", "time_idx", "in_season"]], on=["_series", "time_idx"], how="left"
    )
    conf = merged[
        (merged["ds"] >= conf_start) & (merged["ds"] < CUT_DATE) & (merged["in_season"] == 1)
    ].reset_index(drop=True)
    test = merged[merged["ds"] >= CUT_DATE].reset_index(drop=True)
    if conf.empty or test.empty:
        raise ValueError(f"conf={len(conf)} / test={len(test)} vacío; revisa el corte/ventanas.")

    intervals, prod90 = _conformalize(conf, test)
    result = {"cut_date": str(CUT_DATE.date()), "model": "tft_cqr", "max_epochs": MAX_EPOCHS,
              **_metrics(test, intervals, prod90)}
    result["n_conf"] = len(conf)

    metrics_dir = settings.reports_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{EXP_ID}_{CUT_DATE.date()}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )
    # Un resumen por corte: si no, la segunda corrida pisa la primera y se pierde la
    # comparación entre cortes (el JSON sí es por corte).
    _write_summary(result, settings.reports_root / f"{EXP_ID}_summary_{CUT_DATE.date()}.md")
    print(_console(result))


def _console(r: dict) -> str:
    i = r["intervals"]
    return "\n".join([
        "", f"TFT + CQR (conformal, mismo que Exp 4), corte {r['cut_date']}, "
        f"test n={r['n_test']}, conf n={r['n_conf']}:",
        f"  cobertura 80%={i['0.80']['coverage']:.1%} (ancho med. {i['0.80']['median_width']} kg)",
        f"  cobertura 90%={i['0.90']['coverage']:.1%} (ancho med. {i['0.90']['median_width']} kg, "
        f"p90 {i['0.90']['p90_width']} kg)",
        f"  CRPS global={r['crps_overall']}",
        "  → comparar cobertura Y ancho contra Exp 4 (XGBoost+conformal): a igual cobertura, "
        "el ancho revela qué cuantiles base son más nítidos.",
    ])


def _write_summary(r: dict, out_path) -> None:
    i = r["intervals"]
    rows = [
        "# Exp 5b — TFT + CQR (comparación justa de cobertura)",
        "",
        f"Cuantiles del TFT envueltos en la **misma** corrección conformal que Exp 4 (Mondrian por "
        f"serie, en temporada, espacio log). Corte **{r['cut_date']}**, test n={r['n_test']}, "
        f"conf n={r['n_conf']}. A igual cobertura, el **ancho** discrimina la nitidez del cuantil base.",
        "",
        "| método | cobertura 90% | ancho mediano 90% (kg) | ancho p90 90% (kg) | CRPS |",
        "|---|---|---|---|---|",
        f"| **TFT + conformal** | {i['0.90']['coverage']:.1%} | {i['0.90']['median_width']} | "
        f"{i['0.90']['p90_width']} | {r['crps_overall']} |",
        "| XGBoost + conformal (Exp 4) | *(ver `reports/exp4_cqr_summary.md`, mismo corte)* | | | |",
        "",
        "> Si el TFT necesita intervalos más anchos que el XGBoost para la misma cobertura, sus "
        "cuantiles base son menos nítidos → el XGBoost+CQR sigue siendo el producto. Ver "
        "`docs/decisions/ADR-0002-tft.md` y `reports/exp5_tft_summary.md`.",
    ]
    out_path.write_text("\n".join(rows))
    logger.info(f"Resumen → {out_path}")


if __name__ == "__main__":
    main()
