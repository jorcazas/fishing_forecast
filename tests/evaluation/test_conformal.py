"""Tests de las utilidades de CQR (conformal.py)."""

from __future__ import annotations

import numpy as np

from fishing_forecast.evaluation.conformal import (
    conformal_quantile,
    mondrian_cqr,
    sorted_quantile_preds,
)


class _ConstModel:
    """Modelo de juguete que predice un valor constante (para `sorted_quantile_preds`)."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, x) -> np.ndarray:
        return np.full(len(x), self.value)


def test_conformal_quantile_finite_sample() -> None:
    scores = np.arange(1.0, 11.0)  # 1..10
    # ceil((10+1)*0.9)=10 → el mayor score.
    assert conformal_quantile(scores, 0.1) == 10.0
    # ceil((10+1)*0.5)=6 → el 6º menor.
    assert conformal_quantile(scores, 0.5) == 6.0


def test_sorted_quantile_preds_fixes_crossing() -> None:
    # Modelos "cruzados": el nivel 0.1 predice más alto que el 0.9.
    grid = {0.1: _ConstModel(5.0), 0.5: _ConstModel(3.0), 0.9: _ConstModel(1.0)}
    x = np.zeros((4, 2))
    out = sorted_quantile_preds(grid, x)
    # Tras reordenar por fila, el nivel más bajo tiene el valor más bajo.
    assert np.all(out[0.1] <= out[0.5])
    assert np.all(out[0.5] <= out[0.9])


def test_mondrian_cqr_nested_intervals() -> None:
    # Dos series; cotas base y datos de calibración simples.
    rng = np.random.default_rng(0)
    n = 200
    conf_series = np.array(["a"] * n + ["b"] * n)
    lo_conf = np.zeros(2 * n)
    hi_conf = np.concatenate([np.full(n, 2.0), np.full(n, 10.0)])
    y_conf = np.concatenate([rng.uniform(-1, 3, n), rng.uniform(-2, 12, n)])
    test_series = np.array(["a", "a", "b", "b"])
    lo_test = np.zeros(4)
    hi_test = np.array([2.0, 2.0, 10.0, 10.0])
    lo80, hi80 = mondrian_cqr(
        conf_series, test_series, lo_conf, hi_conf, y_conf, lo_test, hi_test, 0.2
    )
    lo90, hi90 = mondrian_cqr(
        conf_series, test_series, lo_conf, hi_conf, y_conf, lo_test, hi_test, 0.1
    )
    # El intervalo al 90% (alpha menor) contiene al del 80% (anidamiento).
    assert np.all(lo90 <= lo80 + 1e-9)
    assert np.all(hi90 >= hi80 - 1e-9)


def test_mondrian_cqr_marginal_coverage() -> None:
    # Cobertura empírica en calibración >= nivel nominal (garantía conformal, mismo conjunto).
    rng = np.random.default_rng(1)
    n = 500
    series = np.array(["s"] * n)
    lo = np.zeros(n)
    hi = np.full(n, 3.0)
    y = rng.uniform(-2, 6, n)
    lo_c, hi_c = mondrian_cqr(series, series, lo, hi, y, lo, hi, 0.1)
    cov = np.mean((y >= lo_c) & (y <= hi_c))
    assert cov >= 0.9


def test_mondrian_normalized_differs_when_heteroscedastic() -> None:
    # Con anchos base variables, la variante normalizada difiere de la constante.
    rng = np.random.default_rng(2)
    n = 400
    series = np.array(["s"] * n)
    lo = np.zeros(n)
    hi = rng.uniform(1.0, 10.0, n)  # ancho base muy variable
    y = rng.uniform(-2, 12, n)
    test_series = np.array(["s"] * 5)
    lo_t = np.zeros(5)
    hi_t = np.array([1.0, 3.0, 5.0, 8.0, 10.0])
    _, hi_split = mondrian_cqr(series, test_series, lo, hi, y, lo_t, hi_t, 0.1, normalize=False)
    _, hi_norm = mondrian_cqr(series, test_series, lo, hi, y, lo_t, hi_t, 0.1, normalize=True)
    assert not np.allclose(hi_split, hi_norm)
