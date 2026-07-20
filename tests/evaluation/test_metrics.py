"""Tests de las métricas de evaluación."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fishing_forecast.evaluation.metrics import (
    all_metrics,
    coverage,
    crps_from_quantiles,
    mae,
    mean_interval_width,
    pinball_loss,
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


def test_coverage_counts_points_inside_interval() -> None:
    y = [1.0, 2.0, 3.0, 4.0]
    lo = [0.0, 0.0, 5.0, 3.5]  # 3.0 queda fuera (5>3), el resto dentro
    hi = [2.0, 1.5, 6.0, 4.5]  # 2.0 dentro, pero 2.0<=1.5 falso → fuera
    # dentro: y=1 (0..2 sí), y=2 (0..1.5 no), y=3 (5..6 no), y=4 (3.5..4.5 sí) → 2/4
    assert coverage(y, lo, hi) == pytest.approx(0.5)


def test_coverage_full_and_empty() -> None:
    assert coverage([1.0, 2.0], [0.0, 0.0], [3.0, 3.0]) == 1.0
    assert np.isnan(coverage([np.nan], [0.0], [1.0]))


def test_mean_interval_width() -> None:
    assert mean_interval_width([0.0, 1.0], [2.0, 5.0]) == pytest.approx(3.0)


def test_pinball_loss_is_asymmetric() -> None:
    # Subestimar penaliza más en un cuantil alto.
    under = pinball_loss([10.0], [8.0], 0.9)  # 0.9 * 2
    over = pinball_loss([10.0], [12.0], 0.9)  # 0.1 * 2
    assert under == pytest.approx(1.8)
    assert over == pytest.approx(0.2)
    with pytest.raises(ValueError, match="quantile"):
        pinball_loss([1.0], [1.0], 1.5)


def test_crps_perfect_forecast_is_zero() -> None:
    y = [5.0, 5.0]
    qp = {0.25: [5.0, 5.0], 0.5: [5.0, 5.0], 0.75: [5.0, 5.0]}
    assert crps_from_quantiles(y, qp) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="vac"):
        crps_from_quantiles(y, {})
