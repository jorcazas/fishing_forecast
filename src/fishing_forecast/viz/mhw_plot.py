"""Figura de línea de tiempo de olas de calor marinas (MHW).

Toma el DataFrame diagnóstico que produce
``aggregate.ocean_by_ue.sst_mhw_for_bbox`` / ``aggregate.mhw.add_mhw(..., return_diagnostics=True)``
(columnas ``ds, sst, clim, thresh, in_mhw, mhw_category``) y dibuja la SST contra su
climatología y umbral, sombreando los eventos MHW. Pensada para
``reports/figures/mhw_timeline.png`` (Fase 1.3): sobre datos reales debe mostrar
claramente el Blob 2014-2016 y el régimen 2019-2021.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend no interactivo (sirve en CI / sin display)

import matplotlib.pyplot as plt
import pandas as pd

_REQUIRED = {"clim", "thresh", "in_mhw"}

#: Colores por categoría MHW (Hobday 2018) para el sombreado.
_CATEGORY_COLORS = {
    1: "#ffd29e",  # moderado
    2: "#ff9b3f",  # fuerte
    3: "#e8480c",  # severo
    4: "#7a0a0a",  # extremo
}


def plot_mhw_timeline(
    df: pd.DataFrame,
    out_path: Path,
    *,
    ds_col: str = "ds",
    sst_col: str = "sst",
    title: str = "Olas de calor marinas (MHW)",
    figsize: tuple[float, float] = (14.0, 5.0),
    dpi: int = 120,
) -> Path:
    """Dibuja SST + climatología + umbral con eventos MHW sombreados; guarda PNG.

    Requiere las columnas diagnósticas `clim`, `thresh`, `in_mhw` (de `add_mhw(...,
    return_diagnostics=True)`). Devuelve la ruta escrita.
    """
    missing = _REQUIRED - set(df.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas diagnósticas {sorted(missing)}. "
            "Genera el DataFrame con add_mhw(..., return_diagnostics=True)."
        )

    data = df.sort_values(ds_col)
    ds = pd.to_datetime(data[ds_col])
    sst = data[sst_col].to_numpy()
    clim = data["clim"].to_numpy()
    thresh = data["thresh"].to_numpy()
    in_mhw = data["in_mhw"].to_numpy(dtype=bool)
    category = (
        data["mhw_category"].to_numpy() if "mhw_category" in data.columns else in_mhw.astype(int)
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ds, clim, color="#3b6ea5", lw=1.0, label="Climatología")
    ax.plot(ds, thresh, color="#3b6ea5", lw=1.0, ls="--", label="Umbral p90")
    ax.plot(ds, sst, color="#222222", lw=0.8, label="SST")

    # Sombrea entre umbral y SST durante eventos, coloreando por categoría máxima del tramo.
    for cat, color in _CATEGORY_COLORS.items():
        mask = in_mhw & (category == cat)
        if mask.any():
            ax.fill_between(ds, thresh, sst, where=mask, color=color, alpha=0.9, linewidth=0)
    # Para días-hueco fusionados (in_mhw pero categoría no listada) usa el color moderado.
    other = in_mhw & ~pd.Series(category).isin(list(_CATEGORY_COLORS)).to_numpy()
    if other.any():
        ax.fill_between(
            ds, thresh, sst, where=other, color=_CATEGORY_COLORS[1], alpha=0.9, linewidth=0
        )

    ax.set_title(title)
    ax.set_ylabel("SST (°C)")
    ax.set_xlabel("Fecha")
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
