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


# --- Fase 2 (B4): anomalías climatológicas, interacciones y config YAML -----------------------


def test_climatology_uses_only_train_years() -> None:
    """Cambiar valores posteriores a `train_end` no puede mover la climatología."""
    from fishing_forecast.features.covariates import fit_climatology

    df = _daily_df()
    clim_a = fit_climatology(df, ["sst"], train_end="2018-06-01", smooth_window=1)
    contaminated = df.copy()
    future = pd.to_datetime(contaminated["ds"]) >= pd.Timestamp("2018-06-01")
    contaminated.loc[future, "sst"] = 999.0
    clim_b = fit_climatology(contaminated, ["sst"], train_end="2018-06-01", smooth_window=1)
    pd.testing.assert_frame_equal(clim_a, clim_b)


def test_climatology_is_the_mean_by_day_of_year() -> None:
    from fishing_forecast.features.covariates import fit_climatology

    df = _daily_df()
    clim = fit_climatology(df, ["sst"], train_end="2019-01-01", smooth_window=1)
    train = df[pd.to_datetime(df["ds"]) < pd.Timestamp("2019-01-01")]
    doy = pd.to_datetime(train["ds"]).dt.dayofyear
    expected = train.groupby(doy)["sst"].mean()
    np.testing.assert_allclose(clim.loc[expected.index, "sst"].to_numpy(), expected.to_numpy())


def test_anomaly_feature_is_shifted_and_does_not_leak() -> None:
    """`sst_anom_lag90[t]` solo puede depender de observaciones de `t-90` hacia atrás."""
    from fishing_forecast.features.covariates import (
        add_climatology_anomalies,
        fit_climatology,
    )

    df = _daily_df()
    clim = fit_climatology(df, ["sst"], train_end="2018-06-01", smooth_window=1)
    base = build_covariate_features(df, shift_days=90)
    feat = add_climatology_anomalies(base.copy(), df, clim, shift_days=90)
    assert feat["sst_anom_lag90"].iloc[:90].isna().all()

    spike = 500
    spiked = df.copy()
    spiked.loc[spike, "sst"] = 999.0
    feat_spiked = add_climatology_anomalies(
        build_covariate_features(spiked, shift_days=90), spiked, clim, shift_days=90
    )
    before = feat["sst_anom_lag90"].iloc[: spike + 90]
    after = feat_spiked["sst_anom_lag90"].iloc[: spike + 90]
    pd.testing.assert_series_equal(before, after)  # el pico solo aparece en t >= spike+90
    assert feat_spiked["sst_anom_lag90"].iloc[spike + 90] > 900


def test_interactions_are_exact_products_and_skip_missing_columns() -> None:
    from fishing_forecast.features.covariates import add_interactions

    feat = build_covariate_features(_daily_df(), shift_days=90)
    out = add_interactions(feat.copy(), [("sst_lag90", "in_season"), ("no_existe", "in_season")])
    np.testing.assert_allclose(
        out["sst_lag90__x__in_season"].to_numpy(),
        (feat["sst_lag90"] * feat["in_season"]).to_numpy(),
        equal_nan=True,
    )
    assert not [c for c in out.columns if c.startswith("no_existe")]


def test_build_features_v2_is_additive_over_the_original_builder() -> None:
    from fishing_forecast.features.covariates import FeatureConfig, build_features_v2

    df = _daily_df()
    cfg = FeatureConfig(
        anomalies=True,
        anomaly_columns=("sst",),
        interactions=(("sst_lag90", "in_season"),),
    )
    base = build_covariate_features(df, shift_days=90)
    v2 = build_features_v2(df, config=cfg, train_end="2018-06-01")
    assert set(base.columns) < set(v2.columns)
    pd.testing.assert_frame_equal(v2[base.columns], base)
    assert {"sst_anom_lag90", "sst_lag90__x__in_season"} <= set(v2.columns)


def test_build_features_v2_skips_anomalies_without_train_end() -> None:
    from fishing_forecast.features.covariates import FeatureConfig, build_features_v2

    cfg = FeatureConfig(anomalies=True, anomaly_columns=("sst",))
    v2 = build_features_v2(_daily_df(), config=cfg)
    assert "sst_anom_lag90" not in v2.columns


def test_project_feature_config_yaml_loads() -> None:
    from fishing_forecast.features.covariates import load_feature_config

    cfg = load_feature_config()
    assert cfg.shift_days == 90
    assert cfg.rolling_windows == (90, 365)
    assert cfg.anomalies and "sst" in cfg.anomaly_columns
    assert all(len(p) == 2 for p in cfg.interactions)
