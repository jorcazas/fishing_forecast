"""Tests de las funciones puras del *store* de pronóstico (sin entrenar XGBoost).

`build_store` entrena el modelo (lento, cubierto por el smoke-test manual del servidor); aquí
se cubren los ayudantes puros de agrupación por temporada y ventana reglamentaria.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fishing_forecast.serving.forecast import (
    _in_lobster_season,
    _season_summaries,
    _season_year,
)


@pytest.mark.parametrize(
    "date,expected",
    [
        ("2024-09-20", True),   # dentro (después del 15-sep)
        ("2024-12-31", True),   # dentro (invierno)
        ("2025-02-10", True),   # dentro (antes del 15-feb)
        ("2025-02-20", False),  # fuera (después del 15-feb)
        ("2024-07-01", False),  # fuera (veda de verano)
        ("2024-09-14", False),  # borde: un día antes de abrir
    ],
)
def test_in_lobster_season(date, expected):
    assert _in_lobster_season(pd.Timestamp(date)) is expected


@pytest.mark.parametrize(
    "date,sy",
    [
        ("2024-09-20", 2024),  # otoño → año actual
        ("2025-01-15", 2024),  # enero → temporada que abrió el año anterior
        ("2024-06-01", 2023),  # mes < julio → año-1
    ],
)
def test_season_year(date, sy):
    assert _season_year(pd.Timestamp(date)) == sy


def _frame(dates, y_obs, lo90, hi90, in_season=1):
    n = len(dates)
    return pd.DataFrame(
        {
            "ds": list(dates),
            "y_obs": y_obs,
            "median": [0.0] * n,
            "lo80": lo90,
            "hi80": hi90,
            "lo90": lo90,
            "hi90": hi90,
            "in_season": [in_season] * n,
        }
    )


def test_season_summaries_lobster_uses_regulatory_window():
    # Días dentro y fuera de la ventana 15-sep–15-feb; solo los de dentro cuentan.
    df = _frame(
        dates=["2024-07-01", "2024-10-01", "2024-11-01", "2025-03-01"],
        y_obs=[500.0, 100.0, 120.0, 800.0],
        lo90=[0.0, 90.0, 110.0, 0.0],
        hi90=[0.0, 110.0, 130.0, 0.0],
    )
    out = _season_summaries(df, "lobster_red")
    assert len(out) == 1
    s = out[0]
    assert s["label"] == "2024-25"
    assert s["n_days"] == 2  # solo oct y nov (jul y mar quedan fuera de la ventana)
    assert s["obs_total_kg"] == pytest.approx(220.0)  # 100 + 120
    assert s["coverage90"] == 1.0  # ambos observados dentro de su banda


def test_season_summaries_non_lobster_groups_by_calendar_year():
    # Abulón/erizo no tienen calendario → se agrupa por año calendario usando in_season.
    df = _frame(
        dates=["2024-03-01", "2024-08-01", "2025-04-01"],
        y_obs=[10.0, 20.0, 30.0],
        lo90=[0.0, 0.0, 0.0],
        hi90=[100.0, 100.0, 100.0],
    )
    out = _season_summaries(df, "abalone_blue")
    labels = {s["label"] for s in out}
    assert labels == {"2024", "2025"}


def test_season_summaries_marks_missing_observed():
    # Temporada en curso sin captura observada → obs_total_kg None.
    df = _frame(
        dates=["2024-10-01", "2024-10-02"],
        y_obs=[0.0, 0.0],
        lo90=[0.0, 0.0],
        hi90=[5.0, 5.0],
    )
    out = _season_summaries(df, "lobster_red")
    assert out[0]["obs_total_kg"] is None
