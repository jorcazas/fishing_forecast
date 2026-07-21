"""Regresión cuantílica conformalizada (CQR) — utilidades puras reusables.

Bloques de la CQR (Romano et al. 2019) que usan tanto el experimento base (Exp 4) como el de
afinado de cuantílicos (Exp 4b). Trabajan sobre arrays en la escala en que se les pasan las cotas
(en los experimentos, escala logarítmica; luego se invierten a kg). Se separan aquí ---en vez de
vivir en un script--- para poder testearlos y compartirlos sin duplicar código.
"""

from __future__ import annotations

import numpy as np


def sorted_quantile_preds(grid_models: dict, x) -> dict[float, np.ndarray]:
    """Predice cada cuantil de la rejilla y reordena por fila para eliminar el cruce de cuantiles.

    Regresores cuantílicos independientes pueden cruzarse (q0.1 > q0.9 en algunas filas). Se ordena
    cada fila de forma ascendente (rearrangement, Chernozhukov et al. 2010), que preserva la
    cobertura y garantiza monotonía → intervalos anidados entre niveles. `grid_models` mapea nivel
    de cuantil → modelo con método `.predict(x)`.
    """
    levels = sorted(grid_models)
    preds = np.column_stack([grid_models[q].predict(x) for q in levels])
    preds = np.sort(preds, axis=1)
    return {q: preds[:, i] for i, q in enumerate(levels)}


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Cuantil conformal finito-muestral `ceil((n+1)(1-alpha))/n` de los scores."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(np.sort(scores)[k - 1])


def split_cqr(
    lo_conf: np.ndarray,
    hi_conf: np.ndarray,
    y_conf: np.ndarray,
    lo_test: np.ndarray,
    hi_test: np.ndarray,
    alpha: float,
    *,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Split-CQR (Romano et al. 2019): ajusta las cotas cuantílicas con la corrección conformal.

    Trabaja en la escala en que se le pasan las cotas. El score de conformidad es
    `E_i = max(lo(x_i) - y_i, y_i - hi(x_i))`; se toma su cuantil conformal y se ensancha
    simétricamente. Garantiza cobertura marginal >= 1-alpha e intervalos **anidados** entre niveles
    (alpha mayor → Q menor).

    Con `normalize=True` (CQR **localmente adaptativo**): el score se divide por el ancho base del
    intervalo `w = hi - lo` (con un piso) y el ensanche en test es `Q · w(x)`, proporcional a la
    incertidumbre local en vez de una constante.
    """
    if normalize:
        w_conf = hi_conf - lo_conf
        floor = max(1e-3, 0.1 * float(np.median(w_conf)))  # evita dividir entre ~0
        w_conf = np.maximum(w_conf, floor)
        scores = np.maximum(lo_conf - y_conf, y_conf - hi_conf) / w_conf
        q = conformal_quantile(scores, alpha)
        w_test = np.maximum(hi_test - lo_test, floor)
        return lo_test - q * w_test, hi_test + q * w_test
    scores = np.maximum(lo_conf - y_conf, y_conf - hi_conf)
    q = conformal_quantile(scores, alpha)
    return lo_test - q, hi_test + q


def mondrian_cqr(
    conf_series: np.ndarray,
    test_series: np.ndarray,
    lo_conf: np.ndarray,
    hi_conf: np.ndarray,
    y_conf: np.ndarray,
    lo_test: np.ndarray,
    hi_test: np.ndarray,
    alpha: float,
    *,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Split-CQR **por serie** (Mondrian): una corrección conformal por grupo/serie.

    Un único `Q` global lo fija la serie peor calibrada y, al invertir el log, infla las series de
    gran escala. Calibrar por serie da cobertura ~nominal *dentro* de cada serie y anchos acordes a
    su escala. `normalize` propaga la variante localmente adaptativa a cada serie.
    """
    lo_out = np.empty_like(lo_test, dtype=float)
    hi_out = np.empty_like(hi_test, dtype=float)
    for s in np.unique(test_series):
        c, t = conf_series == s, test_series == s
        if not c.any():  # sin puntos de conformalización → sin corrección (cae al cuantil crudo)
            lo_out[t], hi_out[t] = lo_test[t], hi_test[t]
            continue
        lo_out[t], hi_out[t] = split_cqr(
            lo_conf[c], hi_conf[c], y_conf[c], lo_test[t], hi_test[t], alpha, normalize=normalize
        )
    return lo_out, hi_out
