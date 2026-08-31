"""Exp 1b — Re-entrenamiento de los modelos ML/DL del borrador 2023 sobre datos unidos.

Cierra el hueco de la Tabla `extension_comparativa` (tesis §6.10): ahí se re-entrenaron
ARIMA, Prophet y XGBoost sobre el conjunto unido 2017-2026, pero **faltaban LGBM, LSTM y el
ensamble XGBoost→LSTM**, que era el mejor modelo del borrador (error de suma de temporada
8.7% / 12.9% agregado). La conclusión del borrador se apoya en ese ensamble, así que dejarlo
fuera de la comparativa refrescada deja la afirmación sin verificar.

Modelos:

- ``lgbm``            — LightGBM con los hiperparámetros de la Tabla `modelos_seleccionados`.
- ``lstm``            — LSTM apilada (2 capas + dropout 0.3), tamaño acorde a los datos.
- ``lstm_orig2023``   — la MISMA arquitectura con las 2700/800 unidades del borrador (~41 M
  de parámetros para ~10³ observaciones). Se corre por fidelidad, no porque se espere que
  gane: el contraste con ``lstm`` es en sí un resultado.
- ``xgb_lstm``        — el ensamble del borrador: "primero se hace la predicción con XGBoost
  y ese vector de y se introduce en LSTM como un regresor más" (tesis §5). Aquí la columna
  extra se construye **fuera de muestra** en train (validación expansiva, ver
  `oof_predictions`); usar la predicción en muestra haría que la LSTM aprendiera de una
  señal que en test no existe con esa calidad.
- ``xgboost``         — se recalcula para que el JSON sea autoconsistente; debe coincidir con
  el de Exp 2 (mismos datos, features, semilla y parámetros).

Comparabilidad: misma serie (langosta-SQ), mismo `load_lobster_sq()`, mismas
features de `build_covariate_features(shift_days=90)` y mismo corte que Exp 2, para que las
filas nuevas se puedan pegar en la tabla junto a las existentes.

Sin leakage: las features ya son puro pasado (ver `features/covariates.py`); el escalado de
la LSTM usa media/desviación **solo de train**; la columna del ensamble en train es
out-of-fold; y las ventanas de test miran hacia atrás, nunca hacia adelante.

Uso:
    uv run python -m experiments.exp1_baseline_retrain.legacy_ml
    FF_CUT_DATE=2024-06-01 uv run python -m experiments.exp1_baseline_retrain.legacy_ml
    FF_LSTM_ARCHS=lstm,lstm_orig2023 uv run python -m experiments.exp1_baseline_retrain.legacy_ml
"""

from __future__ import annotations

import json
import os
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from experiments.exp2_covariates.covariate_model import XGB_PARAMS, load_lobster_sq
from loguru import logger

from fishing_forecast.config import get_settings
from fishing_forecast.evaluation.metrics import all_metrics, season_sum_errors
from fishing_forecast.features.covariates import build_covariate_features, feature_columns

EXP_ID = "exp1_legacy_ml"
#: Corte de test. Canónico 2020-07-01; `FF_CUT_DATE=2024-06-01` deja el bache en train.
CUT_DATE = pd.Timestamp(os.environ.get("FF_CUT_DATE", "2020-07-01"))
SEED = 42

#: Ventana temporal de la LSTM (el borrador usaba (None, 10, 17): 10 pasos).
SEQ_LEN = 10
#: Fracción final de train reservada para early stopping (cronológica, no aleatoria).
VAL_FRACTION = 0.15
LSTM_EPOCHS = int(os.environ.get("FF_LSTM_EPOCHS", "150"))
LSTM_PATIENCE = 20
LSTM_BATCH = 32
LSTM_LR = 1e-3
LSTM_DROPOUT = 0.3

#: Arquitecturas de LSTM a correr. `lstm_orig2023` reproduce las unidades del borrador.
LSTM_ARCHS: dict[str, tuple[int, int]] = {
    "lstm": (128, 64),
    "lstm_orig2023": (2700, 800),
}
#: Cuáles correr (coma-separadas). El original es caro; se pide explícitamente.
ARCHS_TO_RUN = tuple(os.environ.get("FF_LSTM_ARCHS", "lstm").split(","))

#: Hiperparámetros LGBM de la Tabla `modelos_seleccionados` del borrador 2023.
LGBM_PARAMS = dict(
    colsample_bytree=0.64,
    learning_rate=0.10,
    max_depth=6,
    n_estimators=63,
    num_leaves=67,
    random_state=SEED,
    n_jobs=4,
    verbose=-1,
)


# --------------------------------------------------------------------------------------
# Funciones puras (testeables sin datos reales ni entrenamiento)
# --------------------------------------------------------------------------------------


def make_windows(
    x: np.ndarray, y: np.ndarray, seq_len: int = SEQ_LEN
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ventanas deslizantes para la LSTM.

    Devuelve `(windows, targets, end_index)` donde `windows[k]` son las filas
    `[i-seq_len+1, i]` de `x`, `targets[k] = y[i]` e `end_index[k] = i`. El índice de fin
    permite mapear cada ventana a su fecha y decidir train/test **por la fila objetivo**,
    de modo que una ventana de test puede mirar filas de train (pasado) pero nunca al revés.
    """
    if x.ndim != 2:
        raise ValueError(f"`x` debe ser 2D (n, features); llegó {x.shape}.")
    if len(x) != len(y):
        raise ValueError(f"`x` e `y` deben tener el mismo largo ({len(x)} vs {len(y)}).")
    if seq_len < 1 or len(x) < seq_len:
        raise ValueError(f"seq_len={seq_len} inválido para n={len(x)}.")
    idx = np.arange(seq_len - 1, len(x))
    windows = np.stack([x[i - seq_len + 1 : i + 1] for i in idx])
    return windows, y[idx], idx


def standardize(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estandariza con media/desviación **de train** y sustituye NaN por 0 (= la media).

    Las features oceanográficas tienen NaN en los primeros meses (rolling/lag) y la LSTM no
    los tolera; los árboles sí, por eso esto vive aquí y no en `features/`.
    """
    mean = np.nanmean(train, axis=0)
    std = np.nanstd(train, axis=0)
    std = np.where((std == 0) | ~np.isfinite(std), 1.0, std)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    out = [np.nan_to_num((a - mean) / std, nan=0.0, posinf=0.0, neginf=0.0) for a in (train, other)]
    return out[0], out[1]


def shape_diagnostics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """¿El pronóstico *sigue la forma* de la serie, o solo acierta el nivel?

    En una serie donde la mayoría de los días son 0 (veda) un predictor casi plano obtiene
    un MAE bajo sin informar nada. El borrador 2023 ya observó eso de su LSTM ("el ajuste no
    seguía la tendencia de los datos reales") pese a tener el menor error, así que la
    comparativa necesita medirlo, no solo describirlo:

    - `dispersion_ratio` = sd(pred)/sd(obs): ~0 es un pronóstico plano, ~1 tiene la
      variabilidad del observado.
    - `correlation`: Pearson contra el observado; ~0 significa que la variabilidad que hay
      no está en fase con la real.
    """
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    sd_obs = float(np.std(y_true))
    sd_pred = float(np.std(pred))
    corr = 0.0
    if sd_obs > 0 and sd_pred > 0:
        corr = float(np.corrcoef(y_true, pred)[0, 1])
    return {
        "dispersion_ratio": round(sd_pred / sd_obs, 3) if sd_obs > 0 else 0.0,
        "correlation": round(corr, 3),
        "pred_mean": round(float(np.mean(pred)), 1),
        "pred_max": round(float(np.max(pred)), 1),
        "obs_mean": round(float(np.mean(y_true)), 1),
        "obs_max": round(float(np.max(y_true)), 1),
    }


def oof_predictions(
    x: np.ndarray, y: np.ndarray, fit_predict, n_splits: int = 5, min_train: int = 90
) -> np.ndarray:
    """Predicción out-of-fold con ventana expansiva (el esquema de validación del proyecto).

    Divide `x` en `n_splits` bloques cronológicos: el bloque *k* se predice con un modelo
    ajustado en todo lo anterior. Las primeras `min_train` filas (sin modelo previo posible)
    quedan en NaN, y el llamador decide qué hacer con ellas.
    """
    pred = np.full(len(x), np.nan)
    bounds = np.linspace(min_train, len(x), n_splits + 1).astype(int)
    for start, end in pairwise(bounds):
        if end <= start or start == 0:
            continue
        pred[start:end] = fit_predict(x[:start], y[:start], x[start:end])
    return pred


# --------------------------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------------------------


def _xgb_fit_predict(x_train, y_train, x_test) -> np.ndarray:
    import xgboost as xgb

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(x_train, y_train)
    return np.clip(model.predict(x_test), 0.0, None)


def _lgbm_fit_predict(x_train, y_train, x_test) -> np.ndarray:
    import lightgbm as lgb

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(x_train, y_train)
    return np.clip(model.predict(x_test), 0.0, None)


def _build_lstm(n_features: int, units: tuple[int, int]):
    """LSTM apilada del borrador: 2 capas + dropout, densa de 1 unidad con salida ReLU."""
    import torch
    from torch import nn

    class StackedLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm1 = nn.LSTM(n_features, units[0], batch_first=True)
            self.drop1 = nn.Dropout(LSTM_DROPOUT)
            self.lstm2 = nn.LSTM(units[0], units[1], batch_first=True)
            self.drop2 = nn.Dropout(LSTM_DROPOUT)
            self.head = nn.Linear(units[1], 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h, _ = self.lstm1(x)  # devuelve secuencias
            h = self.drop1(h)
            h, _ = self.lstm2(h)
            h = self.drop2(h[:, -1, :])  # solo el estado final
            return torch.relu(self.head(h)).squeeze(-1)  # y >= 0

    return StackedLSTM()


def _device():
    import torch

    name = os.environ.get("FF_DEVICE")
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _train_lstm(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, units: tuple[int, int]
) -> np.ndarray:
    """Entrena la LSTM sobre ventanas y predice `x_test`. Early stopping en el tramo final.

    El objetivo se divide entre la desviación de train (escala invertible sin parámetros
    estimados fuera de train) para que la MSE no viva en kg²; la predicción se re-escala.
    """
    import torch
    from torch import nn

    torch.manual_seed(SEED)
    # En macOS, xgboost/lightgbm ya cargaron su propio libomp; dejar que torch abra además su
    # pool de OpenMP cuelga el proceso en la barrera de fork (se queda bloqueado en un copy_
    # sin consumir CPU). Un solo hilo evita el conflicto y no cuesta nada: el cómputo pesado
    # va en MPS y las series son de ~10³ filas.
    torch.set_num_threads(1)
    device = _device()

    scale = float(np.std(y_train)) or 1.0
    n_val = max(1, int(len(x_train) * VAL_FRACTION))
    xt = torch.tensor(x_train[:-n_val], dtype=torch.float32)
    yt = torch.tensor(y_train[:-n_val] / scale, dtype=torch.float32)
    xv = torch.tensor(x_train[-n_val:], dtype=torch.float32, device=device)
    yv = torch.tensor(y_train[-n_val:] / scale, dtype=torch.float32, device=device)

    model = _build_lstm(x_train.shape[2], units).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"LSTM {units} → {n_params:,} parámetros para {len(xt)} ventanas de entrenamiento "
        f"(device={device.type})"
    )
    opt = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
    loss_fn = nn.MSELoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xt, yt), batch_size=LSTM_BATCH, shuffle=True
    )

    best_loss, best_state, stale = float("inf"), None, 0
    for epoch in range(LSTM_EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(xv), yv))
        if val_loss < best_loss - 1e-6:
            best_loss, stale = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= LSTM_PATIENCE:
                logger.info(f"early stopping en época {epoch} (val MSE {best_loss:.4f})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(x_test, dtype=torch.float32, device=device)).cpu().numpy()
    return np.clip(pred * scale, 0.0, None)


# --------------------------------------------------------------------------------------
# Orquestación
# --------------------------------------------------------------------------------------


def _evaluate(name: str, test: pd.DataFrame, pred: np.ndarray) -> dict:
    daily = all_metrics(test["y"].to_numpy(), pred)
    seasons = season_sum_errors(
        pd.DataFrame(
            {"season": test["season"].to_numpy(), "y_true": test["y"].to_numpy(), "y_pred": pred}
        )
    )
    return {
        "model": name,
        "daily": daily,
        "shape": shape_diagnostics(test["y"].to_numpy(), pred),
        "season_sum": {
            s: {
                "true_sum": round(float(r.true_sum), 1),
                "pred_sum": round(float(r.pred_sum), 1),
                "pct_error": None if pd.isna(r.pct_error) else round(float(r.pct_error), 1),
            }
            for s, r in seasons.iterrows()
        },
    }


def main() -> None:
    settings = get_settings()
    feat = build_covariate_features(load_lobster_sq(), shift_days=90)
    cols = feature_columns(feat)

    is_train = feat["ds"] < CUT_DATE
    train, test = feat[is_train], feat[~is_train]
    logger.info(f"corte {CUT_DATE.date()}: train={len(train)} días, test={len(test)} días")

    x_train, y_train = train[cols].to_numpy(float), train["y"].to_numpy(float)
    x_test = test[cols].to_numpy(float)

    results: list[dict] = []
    preds: dict[str, np.ndarray] = {}  # para la figura

    # --- Modelos tabulares -------------------------------------------------------------
    pred_xgb = _xgb_fit_predict(x_train, y_train, x_test)
    preds["xgboost"] = pred_xgb
    preds["lgbm"] = _lgbm_fit_predict(x_train, y_train, x_test)
    results.append(_evaluate("xgboost", test, pred_xgb))
    results.append(_evaluate("lgbm", test, preds["lgbm"]))

    # --- LSTM (ventanas sobre la matriz completa; el objetivo decide train/test) --------
    x_all, y_all = feat[cols].to_numpy(float), feat["y"].to_numpy(float)
    _, x_std_all = standardize(x_all[is_train.to_numpy()], x_all)
    windows, targets, end_idx = make_windows(x_std_all, y_all)
    win_is_train = is_train.to_numpy()[end_idx]

    for arch in ARCHS_TO_RUN:
        if arch not in LSTM_ARCHS:
            raise ValueError(f"Arquitectura desconocida {arch!r}; opciones: {list(LSTM_ARCHS)}")
        pred = _train_lstm(
            windows[win_is_train], targets[win_is_train], windows[~win_is_train], LSTM_ARCHS[arch]
        )
        preds[arch] = pred
        results.append(_evaluate(arch, test, pred))

    # --- Ensamble XGBoost → LSTM (la predicción de XGB como regresor extra) -------------
    xgb_col = np.empty(len(feat))
    xgb_col[is_train.to_numpy()] = oof_predictions(x_train, y_train, _xgb_fit_predict)
    xgb_col[~is_train.to_numpy()] = pred_xgb
    xgb_col = np.nan_to_num(xgb_col, nan=float(np.nanmedian(y_train)))

    x_ens = np.column_stack(
        [
            x_std_all,
            (xgb_col - xgb_col[is_train.to_numpy()].mean())
            / (np.std(xgb_col[is_train.to_numpy()]) or 1.0),
        ]
    )
    win_ens, targets_ens, _ = make_windows(x_ens, y_all)
    pred_ens = _train_lstm(
        win_ens[win_is_train], targets_ens[win_is_train], win_ens[~win_is_train], LSTM_ARCHS["lstm"]
    )
    preds["xgb_lstm"] = pred_ens
    results.append(_evaluate("xgb_lstm", test, pred_ens))

    payload = {
        "exp_id": EXP_ID,
        "cut_date": str(CUT_DATE.date()),
        "n_features": len(cols),
        "n_train_days": len(train),
        "n_test_days": len(test),
        "seed": SEED,
        "seq_len": SEQ_LEN,
        "models": {r["model"]: r for r in results},
    }
    metrics_dir = settings.reports_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out = metrics_dir / f"{EXP_ID}_{CUT_DATE.date()}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info(f"Métricas → {out}")

    _plot(test, preds, settings.reports_root / "figures" / f"{EXP_ID}_{CUT_DATE.date()}.png")
    _write_summary(metrics_dir, settings.reports_root / f"{EXP_ID}_summary.md")
    print(_console_table(payload))


def _plot(test: pd.DataFrame, preds: dict[str, np.ndarray], out_path: Path) -> None:
    """Observado vs. cada modelo. Es la figura que delata al pronóstico plano."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "xgboost": "#c0392b",
        "lgbm": "#e67e22",
        "lstm": "#2980b9",
        "lstm_orig2023": "#8e44ad",
        "xgb_lstm": "#1a9850",
    }
    fig, axes = plt.subplots(len(preds), 1, figsize=(13, 2.2 * len(preds)), sharex=True)
    obs_max = float(test["y"].max()) or 1.0
    for ax, (name, pred) in zip(np.atleast_1d(axes), preds.items(), strict=True):
        ax.plot(test["ds"], test["y"], color="#222", lw=0.8, label="observado")
        ax.plot(test["ds"], pred, color=colors.get(name, "#555"), lw=1.1, label=name)
        ax.set_ylim(-0.05 * obs_max, 3.0 * obs_max)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylabel("kg/día", fontsize=8)
    fig.suptitle(
        f"Exp 1b — langosta-SQ, modelos del borrador 2023 (corte {CUT_DATE.date()}; "
        "eje y recortado a 3x el máx. observado)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"Figura → {out_path}")


def _console_table(payload: dict) -> str:
    lines = [
        "",
        f"Corte {payload['cut_date']} — langosta @ San Quintín "
        f"(train {payload['n_train_days']} d, test {payload['n_test_days']} d)",
        f"  {'modelo':<16} {'MAE':>8} {'RMSE':>9} {'sMAPE':>8} {'sd/sd_obs':>10} {'corr':>7}",
    ]
    for name, r in payload["models"].items():
        d, sh = r["daily"], r["shape"]
        lines.append(
            f"  {name:<16} {d['mae']:>8.1f} {d['rmse']:>9.1f} {d['smape']:>7.1f}% "
            f"{sh['dispersion_ratio']:>10.2f} {sh['correlation']:>7.2f}"
        )
    return "\n".join(lines)


def _write_summary(metrics_dir: Path, out_path: Path) -> None:
    """Resumen con **todos** los cortes ya calculados, para pegar en la tesis §6.10.

    Lee cada `exp1_legacy_ml_<corte>.json` presente, de modo que correr el segundo corte no
    borra el primero: la comparación que importa es justamente entre cortes.
    """
    payloads = [json.loads(p.read_text()) for p in sorted(metrics_dir.glob(f"{EXP_ID}_*.json"))]
    rows = [
        "# Exp 1b — Modelos ML/DL del borrador 2023 sobre datos unidos (langosta-SQ)",
        "",
        "Completa la Tabla `extension_comparativa` de la tesis (§6.10), que solo tenía ARIMA, "
        "Prophet y XGBoost: aquí están **LGBM, LSTM y el ensamble XGBoost→LSTM**, el mejor "
        "modelo del borrador 2023. Mismos datos, features, partición y semilla que Exp 2 (la "
        "fila `xgboost` debe coincidir con la suya).",
    ]
    for payload in payloads:
        rows += [
            "",
            f"## Corte {payload['cut_date']}",
            "",
            f"{payload['n_features']} features · train {payload['n_train_days']} días / "
            f"test {payload['n_test_days']} días · semilla {payload['seed']} · "
            f"ventana LSTM {payload['seq_len']} pasos.",
            "",
            "| modelo | MAE | RMSE | sMAPE (%) | sd(pred)/sd(obs) | corr |",
            "|---|---|---|---|---|---|",
        ]
        for name, r in payload["models"].items():
            d, sh = r["daily"], r["shape"]
            rows.append(
                f"| {name} | {d['mae']:.1f} | {d['rmse']:.1f} | {d['smape']:.1f} | "
                f"{sh['dispersion_ratio']:.2f} | {sh['correlation']:.2f} |"
            )
        rows += ["", "| modelo | temporada | error de suma (%) |", "|---|---|---|"]
        for name, r in payload["models"].items():
            for season, ss in r["season_sum"].items():
                if ss["pct_error"] is not None:
                    rows.append(f"| {name} | {season} | {ss['pct_error']:+.1f} |")
    rows += [
        "",
        "## Notas de lectura",
        "",
        "- `sd(pred)/sd(obs)` y `corr` miden si el pronóstico **sigue la forma** de la serie. "
        "Un cociente cercano a 0 delata un pronóstico casi plano (que en una serie con mayoría "
        "de días en veda obtiene buen MAE sin informar nada) y uno muy por encima de 1, un "
        "modelo que sobre-reacciona.",
        "- `xgb_lstm` es el ensamble del borrador (la predicción de XGBoost como regresor extra "
        "de la LSTM), con esa columna construida **fuera de muestra** en train (ventana "
        "expansiva); usar la predicción en muestra le daría a la LSTM una señal que en test no "
        "existe con esa calidad.",
        "- `lstm_orig2023` (2700/800 unidades, ~41 M de parámetros) solo corre si se pide: "
        "`FF_LSTM_ARCHS=lstm,lstm_orig2023`.",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + "\n")
    logger.info(f"Resumen → {out_path}")


if __name__ == "__main__":
    main()
