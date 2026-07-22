"""Figura de línea de tiempo de olas de calor marinas (MHW) para San Quintín.

Cierra la Fase 1.3 (PENDINGS §4): genera `reports/figures/mhw_timeline.png` con la SST
diaria del bbox de San Quintín contra su climatología y umbral p90 (baseline 1982-2011,
Hobday 2016), sombreando los eventos MHW por categoría. Debe mostrar con nitidez el ``Blob''
de 2014-2016 y el régimen cálido de 2019-2021 que precede al colapso de la captura de langosta.

Reutiliza el detector de MHW del ETL (`aggregate.mhw.add_mhw`) y el ploteo de `viz.mhw_plot`.
La ventana se recorta a 2013-2025 (periodo operativo relevante); cambia `WINDOW_START` a None
para la serie completa 1982-2025.

Uso:
    uv run python experiments/mhw_timeline_figure.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

from fishing_forecast.config import get_settings
from fishing_forecast.etl.aggregate.mhw import MHWParams, add_mhw
from fishing_forecast.etl.aggregate.ocean_by_ue import sst_series_for_bbox
from fishing_forecast.viz.mhw_plot import plot_mhw_timeline

ECONOMIC_UNIT = "litoral_bc_sur"
WINDOW_START = pd.Timestamp("2013-01-01")  # None → serie completa 1982-2025


def main() -> None:
    settings = get_settings()
    repo = Path(__file__).resolve().parents[1]

    econ = yaml.safe_load((repo / "configs" / "economic_units.yaml").read_text(encoding="utf-8"))
    bbox = econ[ECONOMIC_UNIT]["bbox"]
    mhw_cfg = yaml.safe_load((repo / "configs" / "etl.yaml").read_text(encoding="utf-8"))["mhw"]
    params = MHWParams.from_config(mhw_cfg)

    sst_path = settings.raw_dir / "copernicus" / "sst_l4.nc"
    if not sst_path.exists():
        raise FileNotFoundError(f"Falta {sst_path} (SST OSTIA de Copernicus).")

    daily = sst_series_for_bbox([sst_path], bbox)
    diag = add_mhw(daily, params, return_diagnostics=True)
    diag["ds"] = pd.to_datetime(diag["ds"])
    n_events_days = int(diag["in_mhw"].sum())
    logger.info(f"SST {diag['ds'].min().date()}→{diag['ds'].max().date()}, {n_events_days} días MHW.")

    if WINDOW_START is not None:
        diag = diag[diag["ds"] >= WINDOW_START]
    span = f"{diag['ds'].min():%Y}-{diag['ds'].max():%Y}"

    out = settings.reports_root / "figures" / "mhw_timeline.png"
    plot_mhw_timeline(
        diag,
        out,
        title=f"Olas de calor marinas — San Quintín ({span}); baseline 1982-2011, umbral p90",
    )
    logger.info(f"Figura → {out}")


if __name__ == "__main__":
    main()
