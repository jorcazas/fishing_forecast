"""Test de la figura de línea de tiempo MHW (smoke con datos sintéticos)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fishing_forecast.etl.aggregate.mhw import MHWParams, add_mhw
from fishing_forecast.viz.mhw_plot import plot_mhw_timeline


def _diagnostics() -> pd.DataFrame:
    dates = pd.date_range("2000-01-01", "2003-12-31", freq="D")
    doy = dates.dayofyear.to_numpy()
    sst = 18.0 + 5.0 * np.sin(2 * np.pi * (doy - 80) / 365.0)
    df = pd.DataFrame({"ds": dates, "sst": sst})
    hw = (df["ds"] >= "2003-07-01") & (df["ds"] <= "2003-07-12")
    df.loc[hw, "sst"] += 6.0
    params = MHWParams(
        baseline_start=date(2000, 1, 1), baseline_end=date(2002, 12, 31), smooth_window=11
    )
    return add_mhw(df, params, return_diagnostics=True)


def test_plot_writes_png(tmp_path: Path) -> None:
    out = plot_mhw_timeline(_diagnostics(), tmp_path / "mhw_timeline.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_requires_diagnostic_columns(tmp_path: Path) -> None:
    bad = pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=3), "sst": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="diagnósticas"):
        plot_mhw_timeline(bad, tmp_path / "x.png")
