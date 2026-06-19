"""Tests del detector de olas de calor marinas (Hobday 2016/2018).

Se usan series sintéticas: una climatología sinusoidal estacional como baseline y
anomalías inyectadas con duración/intensidad controladas, para verificar detección,
umbral de duración, fusión de huecos y categorización sin depender de datos reales.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from fishing_forecast.etl.aggregate.mhw import (
    MHWParams,
    add_mhw,
    category_from_ratio,
    compute_climatology,
    year_day,
)


def _seasonal_sst(
    dates: pd.DatetimeIndex, amplitude: float = 5.0, mean: float = 18.0
) -> np.ndarray:
    """SST estacional determinista (sin ruido) → climatología limpia."""
    doy = np.array([year_day(ts.date()) for ts in dates])
    return mean + amplitude * np.sin(2 * np.pi * (doy - 80) / 365.0)


@pytest.fixture
def baseline_params() -> MHWParams:
    # Baseline corto (la advertencia de <30 años es esperada en tests).
    return MHWParams(
        baseline_start=date(2000, 1, 1),
        baseline_end=date(2009, 12, 31),
        smooth_window=11,
    )


def test_year_day_is_leap_aligned() -> None:
    assert year_day(date(2001, 1, 1)) == 1
    assert year_day(date(2001, 3, 1)) == 61  # no bisiesto, sigue siendo 61
    assert year_day(date(2000, 3, 1)) == 61  # bisiesto
    assert year_day(date(2000, 2, 29)) == 60
    assert year_day(date(1999, 12, 31)) == 366


def test_category_from_ratio_bands() -> None:
    assert category_from_ratio(0.5) == 0
    assert category_from_ratio(1.0) == 1
    assert category_from_ratio(1.9) == 1
    assert category_from_ratio(2.0) == 2
    assert category_from_ratio(3.5) == 3
    assert category_from_ratio(4.0) == 4
    assert category_from_ratio(99.0) == 4
    assert category_from_ratio(float("nan")) == 0


def test_climatology_threshold_above_mean(baseline_params: MHWParams) -> None:
    dates = pd.date_range("2000-01-01", "2009-12-31", freq="D")
    sst = pd.Series(_seasonal_sst(dates), index=dates)
    clim = compute_climatology(sst, baseline_params)
    assert len(clim) == 366
    # El umbral p90 nunca debe quedar por debajo de la media climatológica.
    assert (clim["thresh"] >= clim["clim"] - 1e-9).all()


def test_no_mhw_in_pure_climatology(baseline_params: MHWParams) -> None:
    # Sin anomalías, casi ningún día debe marcarse como MHW.
    dates = pd.date_range("2000-01-01", "2012-12-31", freq="D")
    df = pd.DataFrame({"ds": dates, "sst": _seasonal_sst(dates)})
    out = add_mhw(df, baseline_params)
    assert out["mhw_category"].max() <= 1  # ruido numérico a lo sumo categoría 1 esporádica
    assert (out["mhw_category"] > 0).mean() < 0.05


def test_injected_heatwave_is_detected(baseline_params: MHWParams) -> None:
    dates = pd.date_range("2000-01-01", "2012-12-31", freq="D")
    sst = _seasonal_sst(dates)
    df = pd.DataFrame({"ds": dates, "sst": sst.copy()})
    # Inyecta +6°C durante 10 días consecutivos en 2012.
    hw = (df["ds"] >= "2012-07-01") & (df["ds"] <= "2012-07-10")
    df.loc[hw, "sst"] += 6.0
    out = add_mhw(df, baseline_params, return_diagnostics=True)

    hw_rows = out[hw]
    assert (hw_rows["mhw_category"] >= 1).all()
    assert (hw_rows["mhw_intensity"] > 0).all()
    # Fuera del evento, intensity es NaN.
    assert out.loc[~hw, "mhw_intensity"].isna().all()
    # sst_anomaly existe siempre.
    assert out["sst_anomaly"].notna().all()


def test_short_spike_below_min_duration_is_ignored(baseline_params: MHWParams) -> None:
    dates = pd.date_range("2000-01-01", "2012-12-31", freq="D")
    df = pd.DataFrame({"ds": dates, "sst": _seasonal_sst(dates)})
    # Pico de 3 días (< min_duration=5) → no debe contar como MHW.
    spike = (df["ds"] >= "2012-07-01") & (df["ds"] <= "2012-07-03")
    df.loc[spike, "sst"] += 6.0
    out = add_mhw(df, baseline_params)
    assert (out.loc[spike, "mhw_category"] == 0).all()


def test_gap_within_max_gap_is_joined(baseline_params: MHWParams) -> None:
    dates = pd.date_range("2000-01-01", "2012-12-31", freq="D")
    df = pd.DataFrame({"ds": dates, "sst": _seasonal_sst(dates)})
    # Dos rachas de 3 días separadas por 1 día normal: 3 + (1 gap) + 3 = 7 días ≥ 5 → MHW.
    block1 = (df["ds"] >= "2012-07-01") & (df["ds"] <= "2012-07-03")
    block2 = (df["ds"] >= "2012-07-05") & (df["ds"] <= "2012-07-07")
    df.loc[block1 | block2, "sst"] += 6.0
    out = add_mhw(df, baseline_params, return_diagnostics=True)
    event = (out["ds"] >= "2012-07-01") & (out["ds"] <= "2012-07-07")
    assert out.loc[event, "in_mhw"].all()  # incluye el día-hueco del 4-jul
    # El día-hueco (debajo del umbral) sigue marcado in_mhw con categoría ≥1.
    gap_day = out["ds"] == pd.Timestamp("2012-07-04")
    assert (out.loc[gap_day, "mhw_category"] >= 1).all()


def test_handles_input_gaps_without_false_consecutive(baseline_params: MHWParams) -> None:
    # Serie con huecos de calendario: la reindexación no debe inventar consecutividad.
    dates = pd.date_range("2000-01-01", "2012-12-31", freq="D")
    df = pd.DataFrame({"ds": dates, "sst": _seasonal_sst(dates)})
    # Elimina filas para crear huecos; el output debe conservar exactamente las filas de entrada.
    df_gappy = df[df["ds"].dt.day != 15].reset_index(drop=True)
    out = add_mhw(df_gappy, baseline_params)
    assert len(out) == len(df_gappy)
    assert {"sst_anomaly", "mhw_category", "mhw_intensity"} <= set(out.columns)


def test_from_config_maps_etl_yaml_block() -> None:
    cfg = {
        "baseline": {"start": "1982-01-01", "end": "2011-12-31"},
        "percentile": 90,
        "window_half_width_days": 5,
        "min_duration_days": 5,
        "max_gap_days": 2,
        "smoothing_window_days": 31,
    }
    params = MHWParams.from_config(cfg)
    assert params.baseline_start == date(1982, 1, 1)
    assert params.baseline_end == date(2011, 12, 31)
    assert params.window_half_width == 5
    assert params.max_gap == 2
    assert params.smooth_window == 31
