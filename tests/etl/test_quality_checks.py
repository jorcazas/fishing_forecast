"""Tests de los quality checks del dataset consolidado."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from fishing_forecast.etl.quality_checks import (
    QualityCheckError,
    check_dataset,
    check_sst_correlation,
    run_quality_checks,
)


def _good_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ds": [date(2018, 11, 1), date(2018, 11, 2)],
            "y": [100.0, 0.0],
            "species": ["lobster_red", "lobster_red"],
            "economic_unit": ["litoral_bc_sur", "litoral_bc_sur"],
            "region": ["san_quintin", "san_quintin"],
            "sst": [21.0, 20.5],
            "mhw_category": np.array([2, 0], dtype=np.int8),
            "in_season": [True, True],
            "season": ["2018_2019", "2018_2019"],
        }
    )


def test_clean_dataset_has_no_errors() -> None:
    issues = check_dataset(
        _good_df(), known_species=["lobster_red"], known_units=["litoral_bc_sur"]
    )
    assert [i for i in issues if i.level == "error"] == []


def test_detects_duplicates() -> None:
    df = pd.concat([_good_df(), _good_df().head(1)], ignore_index=True)
    issues = check_dataset(df)
    assert any(i.check == "duplicates" and i.level == "error" for i in issues)


def test_detects_negative_y() -> None:
    df = _good_df()
    df.loc[0, "y"] = -5.0
    issues = check_dataset(df)
    assert any(i.check == "y_range" and i.level == "error" for i in issues)


def test_detects_bad_mhw_category() -> None:
    df = _good_df()
    df.loc[0, "mhw_category"] = 7
    issues = check_dataset(df)
    assert any(i.check == "mhw_category" and i.level == "error" for i in issues)


def test_detects_unknown_species() -> None:
    issues = check_dataset(_good_df(), known_species=["abalone_blue"])
    assert any(i.check == "species_domain" and i.level == "error" for i in issues)


def test_low_ocean_coverage_is_warning() -> None:
    df = _good_df()
    df["sst"] = [np.nan, np.nan]
    issues = check_dataset(df, ocean_coverage_min=0.8)
    cov = [i for i in issues if i.check == "ocean_coverage"]
    assert cov and cov[0].level == "warning"


def test_run_raises_on_error() -> None:
    df = _good_df()
    df.loc[0, "y"] = -1.0
    with pytest.raises(QualityCheckError):
        run_quality_checks(df)


def test_run_passes_clean_and_does_not_raise_on_warning_by_default() -> None:
    df = _good_df()
    df["sst"] = [np.nan, np.nan]  # genera warning de cobertura
    issues = run_quality_checks(df, ocean_coverage_min=0.8)  # no fail_on_warning
    assert any(i.level == "warning" for i in issues)


def test_run_fail_on_warning() -> None:
    df = _good_df()
    df["sst"] = [np.nan, np.nan]
    with pytest.raises(QualityCheckError):
        run_quality_checks(df, ocean_coverage_min=0.8, fail_on_warning=True)


def _sst_pair(n: int = 60, *, noise: float = 0.0, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2020-01-01", periods=n, freq="D").date
    base = 18.0 + 4.0 * np.sin(np.linspace(0, 6.28, n))
    oisst = pd.DataFrame({"ds": ds, "sst": base})
    cicese = pd.DataFrame({"ds": ds, "water_temperature": base + rng.normal(0, noise, n)})
    return oisst, cicese


def test_sst_correlation_passes_when_aligned() -> None:
    oisst, cicese = _sst_pair(noise=0.1)
    assert check_sst_correlation(oisst, cicese, min_corr=0.7) is None


def test_sst_correlation_warns_when_uncorrelated() -> None:
    oisst, cicese = _sst_pair(noise=0.1)
    rng = np.random.default_rng(1)
    cicese["water_temperature"] = rng.normal(18, 4, len(cicese))  # rompe la relación
    issue = check_sst_correlation(oisst, cicese, min_corr=0.7)
    assert issue is not None and issue.check == "sst_correlation" and issue.level == "warning"


def test_sst_correlation_warns_on_insufficient_overlap() -> None:
    oisst, cicese = _sst_pair(n=10)
    issue = check_sst_correlation(oisst, cicese, min_overlap=30)
    assert issue is not None and "insuficiente" in issue.detail
