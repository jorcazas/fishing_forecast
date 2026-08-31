"""Tests de las funciones puras de Exp 1b (ventanas, escalado, out-of-fold)."""

from __future__ import annotations

import numpy as np
import pytest
from experiments.exp1_baseline_retrain.legacy_ml import (
    make_windows,
    oof_predictions,
    shape_diagnostics,
    standardize,
)


def test_make_windows_shapes_and_alignment():
    x = np.arange(20, dtype=float).reshape(10, 2)
    y = np.arange(10, dtype=float)

    windows, targets, end_idx = make_windows(x, y, seq_len=3)

    assert windows.shape == (8, 3, 2)  # n - seq_len + 1 ventanas
    assert end_idx[0] == 2 and end_idx[-1] == 9
    np.testing.assert_array_equal(targets, y[2:])
    # La ventana k termina en su fila objetivo y solo mira hacia atrás.
    np.testing.assert_array_equal(windows[0], x[0:3])
    np.testing.assert_array_equal(windows[-1], x[7:10])


def test_make_windows_rejects_bad_input():
    with pytest.raises(ValueError):
        make_windows(np.zeros(5), np.zeros(5))  # 1D
    with pytest.raises(ValueError):
        make_windows(np.zeros((5, 2)), np.zeros(4))  # largos distintos
    with pytest.raises(ValueError):
        make_windows(np.zeros((3, 2)), np.zeros(3), seq_len=10)  # ventana mayor que n


def test_standardize_uses_train_stats_only():
    train = np.array([[0.0], [2.0]])  # media 1, desviación 1
    other = np.array([[100.0]])

    train_std, other_std = standardize(train, other)

    np.testing.assert_allclose(train_std.ravel(), [-1.0, 1.0])
    # La fila de test se escala con la media/desviación de train, no con la suya.
    np.testing.assert_allclose(other_std.ravel(), [99.0])


def test_standardize_handles_nan_and_constant_columns():
    train = np.array([[1.0, np.nan], [1.0, 5.0]])  # col 0 constante, col 1 con NaN

    train_std, _ = standardize(train, train)

    assert np.isfinite(train_std).all()
    np.testing.assert_allclose(train_std[:, 0], [0.0, 0.0])  # sin división por cero
    assert train_std[0, 1] == 0.0  # NaN → 0 = la media estandarizada


def test_oof_predictions_never_uses_future_rows():
    x = np.arange(200, dtype=float).reshape(-1, 1)
    y = x.ravel()
    seen_max: list[float] = []

    def fit_predict(x_tr, y_tr, x_te):
        seen_max.append(float(x_tr.max()))
        return np.full(len(x_te), x_tr[-1, 0])  # último valor visto

    pred = oof_predictions(x, y, fit_predict, n_splits=4, min_train=50)

    assert np.isnan(pred[:50]).all()  # sin modelo previo posible
    assert np.isfinite(pred[50:]).all()
    # Cada fold entrena solo con el pasado: su predicción nunca alcanza la fila predicha.
    assert all(pred[i] < x[i, 0] for i in range(50, len(x)))
    assert seen_max == sorted(seen_max)  # ventana expansiva


def test_shape_diagnostics_detects_flat_forecast():
    y = np.concatenate([np.zeros(90), np.full(10, 500.0)])  # veda + días pico
    flat = np.full(100, y.mean())

    d = shape_diagnostics(y, flat)

    assert d["dispersion_ratio"] == 0.0  # sin variabilidad: pronóstico plano
    assert d["correlation"] == 0.0


def test_shape_diagnostics_rewards_a_forecast_that_follows_the_shape():
    y = np.concatenate([np.zeros(90), np.full(10, 500.0)])

    d = shape_diagnostics(y, y * 0.9)

    assert d["correlation"] == 1.0
    assert d["dispersion_ratio"] == pytest.approx(0.9, abs=1e-3)
    assert d["obs_max"] == 500.0
