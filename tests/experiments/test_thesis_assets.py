"""Tests de los insumos generados para el capítulo de Datos."""

from __future__ import annotations

import pandas as pd
import pytest
from experiments.thesis_assets import region_label, series_summary


@pytest.fixture
def frame() -> pd.DataFrame:
    """Dos series: una con captura y otra que nunca reporta nada."""
    days = pd.date_range("2020-09-14", periods=4, freq="D")
    return pd.DataFrame(
        {
            "ds": list(days) * 2,
            "y": [0.0, 100.0, None, 400.0, 0.0, 0.0, None, 0.0],
            "species": ["lobster_red"] * 4 + ["urchin_red"] * 4,
            "economic_unit": ["sq"] * 8,
            "region": ["san_quintin"] * 8,
            "season": ["2019_2020", "2020_2021", "2020_2021", "2020_2021"] * 2,
            "sst": [15.0, 16.0, None, 17.0] * 2,
            "chl": [1.0, None, None, 2.0] * 2,
        }
    )


def test_series_summary_counts_only_days_with_catch(frame):
    out = series_summary(frame)

    assert len(out) == 1  # la serie sin captura no aparece
    row = out.iloc[0]
    assert row["species"] == "lobster_red"
    assert row["catch_days"] == 2  # solo los días con y > 0
    assert row["seasons"] == 1  # ambas capturas caen en 2020_2021
    assert row["tonnes"] == pytest.approx(0.5)  # 500 kg


def test_series_summary_reports_covariate_coverage(frame):
    row = series_summary(frame).iloc[0]

    assert row["cov_sst"] == pytest.approx(0.75)  # 3 de 4 días con SST
    assert row["cov_chl"] == pytest.approx(0.5)
    assert row["first"] == pd.Timestamp("2020-09-15")
    assert row["last"] == pd.Timestamp("2020-09-17")


def test_region_label_falls_back_to_a_readable_form():
    assert region_label("san_quintin") == "San Quintín"
    assert region_label("una_zona_nueva") == "Una Zona Nueva"
