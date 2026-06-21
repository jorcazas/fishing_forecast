"""Tests de las métricas de evaluación."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fishing_forecast.evaluation.metrics import (
    all_metrics,
    mae,
    rmse,
    season_sum_errors,
    season_sum_percentage_error,
    smape,
)


def test_perfect_prediction_is_zero_error() -> None:
    y = [1.0, 2.0, 3.0]
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert smape(y, y) == 0.0


def test_mae_rmse_known_values() -> None:
    y_true = [0.0, 0.0, 0.0]
    y_pred = [1.0, 1.0, 1.0]
    assert mae(y_true, y_pred) == pytest.approx(1.0)
    assert rmse([0.0, 0.0], [3.0, 4.0]) == pytest.approx(np.sqrt(12.5))


def test_smape_bounded_and_symmetric() -> None:
    # Predecir el doble del real: 2*|d|/(a+f) = 2*100/300 -> 66.67%
    assert smape([100.0], [200.0]) == pytest.approx(2 * 100 / 300 * 100)
    # 0/0 cuenta como 0 (día sin captura bien predicho).
    assert smape([0.0, 50.0], [0.0, 50.0]) == 0.0
    # Acotado: predicción totalmente opuesta -> 200%.
    assert smape([100.0], [0.0]) == pytest.approx(200.0)


def test_nan_pairs_are_dropped() -> None:
    assert mae([1.0, np.nan, 3.0], [1.0, 5.0, 3.0]) == 0.0
    assert all_metrics([1.0, np.nan], [1.0, 9.0])["n"] == 1


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="Formas distintas"):
        mae([1.0, 2.0], [1.0])


def test_season_sum_percentage_error() -> None:
    assert season_sum_percentage_error(100.0, 90.0) == pytest.approx(-10.0)
    assert season_sum_percentage_error(100.0, 130.0) == pytest.approx(30.0)
    assert np.isnan(season_sum_percentage_error(0.0, 5.0))


def test_season_sum_errors_per_season() -> None:
    df = pd.DataFrame(
        {
            "season": ["2019_2020", "2019_2020", "2020_2021"],
            "y_true": [100.0, 100.0, 50.0],
            "y_pred": [80.0, 80.0, 60.0],
        }
    )
    out = season_sum_errors(df)
    assert out.loc["2019_2020", "true_sum"] == 200.0
    assert out.loc["2019_2020", "pct_error"] == pytest.approx(-20.0)
    assert out.loc["2020_2021", "pct_error"] == pytest.approx(20.0)
