"""Métricas de evaluación de pronósticos (CLAUDE.md → métricas obligatorias).

Funciones puras sobre arrays/Series, más un helper de error en **suma de temporada**
(la métrica que más le importa a COBI: ¿cuánto volumen total esperar la próxima
temporada?). Las métricas diarias (MAE, RMSE, sMAPE) miden el ajuste fino.

`smape` usa la convención simétrica acotada en [0, 200]%, con 0 cuando ambos valores
son 0 (día sin captura bien predicho), para no dividir entre cero ni explotar con `y`
chicos — robusto a la escala, por eso se agrega esta vez.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _aligned(y_true: object, y_pred: object) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"Formas distintas: y_true={yt.shape}, y_pred={yp.shape}.")
    mask = ~(np.isnan(yt) | np.isnan(yp))
    return yt[mask], yp[mask]


def mae(y_true: object, y_pred: object) -> float:
    """Error absoluto medio (kg)."""
    yt, yp = _aligned(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: object, y_pred: object) -> float:
    """Raíz del error cuadrático medio (kg)."""
    yt, yp = _aligned(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def smape(y_true: object, y_pred: object) -> float:
    """sMAPE simétrico en porcentaje [0, 200]. Términos 0/0 cuentan como 0."""
    yt, yp = _aligned(y_true, y_pred)
    denom = np.abs(yt) + np.abs(yp)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(denom == 0, 0.0, 2.0 * np.abs(yp - yt) / denom)
    return float(np.mean(terms) * 100.0)


def season_sum_percentage_error(true_sum: float, pred_sum: float) -> float:
    """Error porcentual (con signo) de la suma de temporada: 100*(pred-true)/true.

    Negativo = subestima. NaN si `true_sum == 0` (no hay base para el porcentaje).
    """
    if true_sum == 0:
        return float("nan")
    return float(100.0 * (pred_sum - true_sum) / true_sum)


def season_sum_errors(
    df: pd.DataFrame,
    *,
    season_col: str = "season",
    true_col: str = "y_true",
    pred_col: str = "y_pred",
) -> pd.DataFrame:
    """Error en suma de temporada por temporada.

    `df` tiene una fila por día con la temporada, el real y el predicho. Devuelve un
    DataFrame indexado por temporada con `true_sum`, `pred_sum` y `pct_error`.
    """
    grouped = df.groupby(season_col, observed=True)[[true_col, pred_col]].sum(min_count=1)
    grouped.columns = ["true_sum", "pred_sum"]
    grouped["pct_error"] = [
        season_sum_percentage_error(t, p)
        for t, p in zip(grouped["true_sum"], grouped["pred_sum"], strict=True)
    ]
    return grouped


def all_metrics(y_true: object, y_pred: object) -> dict[str, float]:
    """Empaqueta las métricas diarias en un dict (para serializar a JSON)."""
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "n": len(_aligned(y_true, y_pred)[0]),
    }


# ---------------------------------------------------------------------------
# Métricas probabilísticas (Fase 4: CQR / cuantiles).
# ---------------------------------------------------------------------------


def coverage(y_true: object, lower: object, upper: object) -> float:
    """Cobertura empírica: fracción de observaciones dentro de `[lower, upper]` (0-1).

    Debe acercarse al nivel nominal del intervalo (p.ej. 0.90 para un intervalo al 90%).
    """
    yt = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    mask = ~(np.isnan(yt) | np.isnan(lo) | np.isnan(hi))
    if not mask.any():
        return float("nan")
    inside = (yt[mask] >= lo[mask]) & (yt[mask] <= hi[mask])
    return float(np.mean(inside))


def mean_interval_width(lower: object, upper: object) -> float:
    """Ancho medio del intervalo (kg). Menor es mejor a igualdad de cobertura."""
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    mask = ~(np.isnan(lo) | np.isnan(hi))
    if not mask.any():
        return float("nan")
    return float(np.mean(hi[mask] - lo[mask]))


def pinball_loss(y_true: object, y_pred_q: object, quantile: float) -> float:
    """Pérdida pinball (quantile loss) para un cuantil `quantile` en (0, 1)."""
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile debe estar en (0,1), no {quantile}.")
    yt, yp = _aligned(y_true, y_pred_q)
    diff = yt - yp
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1.0) * diff)))


def crps_from_quantiles(
    y_true: object, quantile_preds: dict[float, object]
) -> float:
    """CRPS aproximado a partir de predicciones por cuantil (descomposición pinball).

    CRPS = 2 ∫₀¹ QL_τ dτ ≈ 2 · promedio de la pérdida pinball sobre una rejilla de
    cuantiles τ. `quantile_preds` mapea nivel τ → vector de predicciones de ese cuantil.
    Cuantos más niveles, mejor la aproximación.
    """
    if not quantile_preds:
        raise ValueError("quantile_preds vacío.")
    losses = [pinball_loss(y_true, yp, q) for q, yp in quantile_preds.items()]
    return float(2.0 * np.mean(losses))
