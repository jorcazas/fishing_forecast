"""Tests del feature builder de covariables — sobre todo la garantía de NO leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fishing_forecast.features.covariates import (
    build_covariate_features,
    build_multiseries_features,
    feature_columns,
)


def _daily_df(n: int = 800) -> pd.DataFrame:
    ds = pd.date_range("2017-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "ds": ds,
            "y": rng.gamma(2.0, 50.0, n),
            "in_season": np.isin(ds.month, [9, 10, 11, 12, 1, 2]),
            "sst": 18 + 4 * np.sin(np.linspace(0, 12, n)),
            "sst_anomaly": rng.normal(0, 1, n),
            "mhw_category": rng.integers(0, 3, n),
            "mhw_intensity": rng.normal(0, 0.5, n),
            "season": "s",
        }
    )


def test_shapes_and_columns() -> None:
    feat = build_covariate_features(_daily_df())
    assert {"ds", "season", "y"} <= set(feat.columns)
    cols = feature_columns(feat)
    assert "doy_sin" in cols and "in_season" in cols
    assert "sst_anomaly_lag90" in cols and "mhw_intensity_roll365_lag90" in cols
    assert "y" not in cols  # el target no es feature


def test_ocean_features_are_shifted_no_leakage() -> None:
    """La feature oceanográfica en t debe ser el valor de t-shift (nunca futuro)."""
    df = _daily_df()
    feat = build_covariate_features(df, shift_days=90)
    s = df.sort_values("ds").reset_index(drop=True)
    # sst_lag90[t] == sst[t-90]
    np.testing.assert_allclose(
        feat["sst_lag90"].to_numpy()[90:], s["sst"].to_numpy()[:-90], rtol=1e-9
    )
    # Primeras shift filas son NaN (no hay pasado).
    assert feat["sst_lag90"].iloc[:90].isna().all()


def test_injected_future_spike_does_not_leak() -> None:
    """Un pico de SST en una fecha no debe aparecer en features de fechas anteriores."""
    df = _daily_df()
    spike_idx = 500
    df.loc[spike_idx, "sst_anomaly"] = 999.0
    feat = build_covariate_features(df, shift_days=90)
    # Ninguna feature derivada de sst_anomaly en t < spike debe reflejar el pico.
    cols = [c for c in feature_columns(feat) if "sst_anomaly" in c]
    before = feat.loc[: spike_idx - 1, cols]
    assert not (before == 999.0).any().any()
    # El pico sí aparece desplazado 90 días después.
    assert feat.loc[spike_idx + 90, "sst_anomaly_lag90"] == 999.0


def test_y_lags_are_past() -> None:
    df = _daily_df()
    feat = build_covariate_features(df)
    s = df.sort_values("ds").reset_index(drop=True)
    np.testing.assert_allclose(
        feat["y_lag365"].to_numpy()[365:], s["y"].to_numpy()[:-365], rtol=1e-9
    )


def test_multiseries_no_cross_group_leakage() -> None:
    """Un pico de `y` en una especie no debe aparecer en los lags de otra."""
    a = _daily_df(500)
    a["species"] = "lobster_red"
    b = _daily_df(500)
    b["species"] = "abalone_blue"
    b["y"] = 1.0  # abalone: y constante conocido
    a.loc[100, "y"] = 88888.0  # pico solo en langosta
    feat = build_multiseries_features(pd.concat([a, b], ignore_index=True))

    assert set(feat["species"]) == {"lobster_red", "abalone_blue"}
    # Los lags de y de abalone deben venir solo de su propia serie (=1.0), nunca el pico.
    ab = feat[feat["species"] == "abalone_blue"]
    assert not (ab["y_lag365"] == 88888.0).any()
    assert set(ab["y_lag365"].dropna().unique()) <= {1.0}
    # El pico sí aparece en el lag de langosta 365 días después.
    lob = feat[feat["species"] == "lobster_red"].reset_index(drop=True)
    assert lob.loc[100 + 365, "y_lag365"] == 88888.0
