"""Test de la preparación de datos del Exp 5 (TFT), sin `torch`.

`build_tft_frame` es una función pura (pandas) que arma el frame para `TimeSeriesDataSet`;
se valida el recorte al último día con captura, el filtro por `MIN_CATCH_DAYS`, el relleno
de covariables y la construcción de `time_idx`/calendario. No importa `torch` (vive dentro
de `main()`), así que corre en el CI unitario normal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# `experiments` es un paquete de espacio de nombres; asegurar la raíz del repo en sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.exp5_tft.tft import build_tft_frame


def _synthetic() -> pd.DataFrame:
    """dataset_v1 sintético: serie A con 30 días de captura + ceros de cola; serie B corta."""
    dates = pd.date_range("2019-01-01", periods=120, freq="D")
    rng = np.random.default_rng(0)

    # Serie A: captura (y>0) en los días 20-49, ceros antes/después; última captura en el día 49.
    ya = np.zeros(120)
    ya[20:50] = rng.uniform(50, 500, 30)
    in_season_a = np.zeros(120)
    in_season_a[15:55] = 1
    sst_a = np.full(120, 18.0)
    sst_a[0] = np.nan  # debe rellenarse (ffill/bfill)
    a = pd.DataFrame({
        "ds": dates, "y": ya, "species": "lobster_red", "economic_unit": "ue_a",
        "in_season": in_season_a, "sst": sst_a, "mhw_category": 0,
    })

    # Serie B: solo 5 días de captura → por debajo de MIN_CATCH_DAYS, debe descartarse.
    yb = np.zeros(120)
    yb[30:35] = rng.uniform(50, 200, 5)
    b = pd.DataFrame({
        "ds": dates, "y": yb, "species": "lobster_red", "economic_unit": "ue_b",
        "in_season": 1, "sst": 19.0, "mhw_category": 0,
    })
    return pd.concat([a, b], ignore_index=True)


def test_build_tft_frame_shapes_and_cleaning() -> None:
    frame, observed = build_tft_frame(_synthetic())

    # Serie B (5 < 20 días de captura) descartada; solo queda A.
    assert set(frame["_series"]) == {"lobster_red@ue_a"}

    # Recorte al último día con captura (índice 49 → 50 filas, días 0..49).
    assert len(frame) == 50
    assert pd.to_datetime(frame["ds"]).max() == pd.Timestamp("2019-02-19")  # 2019-01-01 + 49d

    # time_idx entero, contiguo desde 0.
    assert frame["time_idx"].tolist() == list(range(50))
    assert frame["time_idx"].dtype.kind == "i"

    # Covariables observadas sin NaN (el NaN de sst[0] se rellenó); sst presente en la lista.
    assert "sst" in observed
    assert frame["sst"].notna().all()

    # Calendario construido y en rango.
    for c in ("doy_sin", "doy_cos"):
        assert c in frame.columns
        assert frame[c].between(-1, 1).all()

    # y sin NaN (in-season NaN→0 ya venía a 0); estáticas como str.
    assert frame["y"].notna().all()
    assert frame["_series"].map(type).eq(str).all()


def test_build_tft_frame_raises_when_no_series_qualify() -> None:
    dates = pd.date_range("2019-01-01", periods=10, freq="D")
    tiny = pd.DataFrame({
        "ds": dates, "y": 0.0, "species": "lobster_red", "economic_unit": "ue_x",
        "in_season": 1, "sst": 18.0, "mhw_category": 0,
    })
    tiny.loc[0, "y"] = 10.0  # 1 día de captura, < MIN_CATCH_DAYS
    import pytest

    with pytest.raises(ValueError, match="MIN_CATCH_DAYS"):
        build_tft_frame(tiny)
