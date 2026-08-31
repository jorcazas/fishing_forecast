"""Figura del producto operativo: lo que la API de inferencia sirve, tal cual.

Consulta la API en marcha (`/api/series`, `/api/forecast`) y grafica tres series
representativas con la banda calibrada del 80 y 90 % y la cobertura empírica anotada. Se
dibuja desde la respuesta de la API ---no desde los objetos del experimento--- para que la
figura del Capítulo del producto muestre exactamente lo que consulta una cooperativa.

Uso (con la API corriendo en local o en Docker):
    FF_API=http://127.0.0.1:8000 uv run python -m experiments.serving_figure
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

API = os.environ.get("FF_API", "http://127.0.0.1:8000")
#: Series elegidas por contraste: la insignia, una de historia larga y una de historia corta.
#: Ventana graficada: una temporada completa, para que el detalle diario sea legible.
WINDOW = ("2024-09-01", "2025-02-28")
SERIES = (
    ("lobster_red@litoral_bc_sur", "insignia (San Quintín)"),
    ("lobster_red@er_scpp_ensenada", "historia larga (El Rosario)"),
    ("lobster_red@vizcaino_tortugas", "historia corta (Vizcaíno)"),
)


def fetch(path: str) -> dict:
    with urllib.request.urlopen(f"{API}{path}", timeout=120) as resp:
        return json.load(resp)


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "final_work" / "images" / "producto_api.png"
    fig, axes = plt.subplots(len(SERIES), 1, figsize=(12, 8))
    for ax, (series, note) in zip(axes, SERIES, strict=True):
        rec = fetch(f"/api/forecast?series={series}")
        cols = ("y_obs", "median", "lo80", "hi80", "lo90", "hi90")
        frame = pd.DataFrame({"ds": pd.to_datetime(rec["dates"]), **{c: rec[c] for c in cols}})
        frame = frame[(frame["ds"] >= WINDOW[0]) & (frame["ds"] <= WINDOW[1])]
        ds = frame["ds"]
        rec = {**rec, **{c: frame[c].tolist() for c in cols}}
        ax.fill_between(ds, rec["lo90"], rec["hi90"], color="#9ecae1", alpha=0.55, label="90 %")
        ax.fill_between(ds, rec["lo80"], rec["hi80"], color="#4292c6", alpha=0.55, label="80 %")
        ax.plot(ds, rec["median"], color="#08519c", lw=1.0, label="mediana")
        ax.plot(ds, rec["y_obs"], ".", color="#111", ms=2.4, label="observado")
        obs_max = max(rec["y_obs"]) or 1.0
        ax.set_ylim(-0.05 * obs_max, 3.0 * obs_max)
        ax.set_title(
            f"{rec['species_label']} — {rec['unit_name'][:34]} · {note} · "
            f"cobertura 90 % en temporada = {rec['coverage90']:.1%}",
            fontsize=9,
        )
        ax.set_ylabel("kg/día", fontsize=8)
        ax.set_xlim(pd.Timestamp(WINDOW[0]), pd.Timestamp(WINDOW[1]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.legend(loc="upper right", fontsize=7, ncol=4)
        logger.info(f"{series}: cobertura {rec['coverage90']:.3f}")
    fig.suptitle(
        "Producto operativo: intervalos calibrados servidos por la API, temporada 2024-2025",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    logger.info(f"Figura → {out}")


if __name__ == "__main__":
    main()
