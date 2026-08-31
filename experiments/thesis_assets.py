"""Genera los insumos del capítulo de Datos de la tesis a partir del dataset real.

Dos artefactos, ambos derivados de `data/processed/dataset_v1.parquet` (y del `.nc` de SST
para el mapa), de modo que la tesis nunca cite cifras escritas a mano:

- ``final_work/tables/series_summary.tex``: una fila por serie (especie x unidad económica)
  con temporadas disponibles, días con captura, tonelaje y cobertura de las covariables.
- ``final_work/images/mapa_ues.png``: mapa de las unidades económicas sobre la SST media,
  que muestra el gradiente térmico del rango (San Quintín al norte, Bahía Magdalena al sur).

Uso:
    uv run python -m experiments.thesis_assets
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from fishing_forecast.config import get_settings

#: Etiquetas legibles para la tesis (el código interno es snake_case).
SPECIES_LABEL = {
    "lobster_red": "Langosta roja",
    "abalone_blue": "Abulón azul",
    "abalone_black": "Abulón negro",
    "abalone_red": "Abulón rojo",
    "urchin_red": "Erizo rojo",
}
REGION_LABEL = {
    "san_quintin": "San Quintín",
    "el_rosario": "El Rosario",
    "punta_canoas": "Punta Canoas",
    "vizcaino": "Vizcaíno",
    "pacifico_bcs": "Pacífico BCS",
    "bahia_magdalena": "Bahía Magdalena",
}


def region_label(region: str) -> str:
    """Etiqueta legible de una región; cae a la clave prettificada si no está mapeada."""
    return REGION_LABEL.get(region, region.replace("_", " ").title())


#: Covariables cuya cobertura se reporta (una de SST, una de color del océano).
COVERAGE_COLS = ("sst", "chl")


def series_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por serie (especie x unidad económica) con lo que un lector necesita saber.

    `temporadas` cuenta las temporadas con al menos un día de captura (una temporada sin
    captura registrada no aporta señal), y `toneladas` es el total desembarcado del periodo.
    """
    rows = []
    for (species, unit), sub in df.groupby(["species", "economic_unit"], observed=True):
        caught = sub[sub["y"].fillna(0) > 0]
        if caught.empty:
            continue
        rows.append(
            {
                "species": species,
                "economic_unit": unit,
                "region": sub["region"].iloc[0],
                "seasons": caught["season"].nunique(),
                "catch_days": len(caught),
                "tonnes": sub["y"].sum(skipna=True) / 1000.0,
                "first": caught["ds"].min(),
                "last": caught["ds"].max(),
                **{f"cov_{c}": sub[c].notna().mean() for c in COVERAGE_COLS},
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["species", "tonnes"], ascending=[True, False]).reset_index(drop=True)


def _latex_escape(text: str) -> str:
    return text.replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def write_series_table(summary: pd.DataFrame, units: dict, out_path: Path) -> None:
    """Escribe la tabla `longtable` de series lista para `\\input` desde la tesis."""
    lines = [
        "% Generado por experiments/thesis_assets.py — no editar a mano.",
        r"\begin{longtable}{llrrrrc}",
        r"\caption{Series (especie $\times$ unidad económica) del conjunto consolidado "
        r"2017-2026. `Temporadas' cuenta las que tienen al menos un día de captura; la "
        r"cobertura es la fracción de días con el valor oceanográfico disponible.}"
        r"\label{tab:series_resumen}\\",
        r"\toprule",
        r"Especie & Zona & Temporadas & Días con captura & Toneladas & Cobertura SST & "
        r"Cobertura CHL \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Especie & Zona & Temporadas & Días con captura & Toneladas & Cobertura SST & "
        r"Cobertura CHL \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for _, r in summary.iterrows():
        unit_name = units.get(r["economic_unit"], {}).get("name", r["economic_unit"])
        zone = region_label(r["region"])
        lines.append(
            f"{SPECIES_LABEL.get(r['species'], r['species'])} & "
            f"{_latex_escape(zone)} ({_latex_escape(unit_name[:22])}) & "
            f"{r['seasons']:d} & {r['catch_days']:d} & {r['tonnes']:,.1f} & "
            f"{r['cov_sst']:.0%} & {r['cov_chl']:.0%} \\\\".replace("%", r"\%")
        )
    lines.append(r"\end{longtable}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Tabla → {out_path} ({len(summary)} series)")


def plot_units_map(units: dict, summary: pd.DataFrame, sst_path: Path, out_path: Path) -> None:
    """Mapa de las UEs sobre la SST media climatológica (la tierra queda en blanco)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import xarray as xr

    fig, ax = plt.subplots(figsize=(7.5, 9))
    if sst_path.exists():
        ds = xr.open_dataset(sst_path)
        var = next(v for v in ds.data_vars if ds[v].ndim >= 3)
        # Una franja de un año basta para la media y evita cargar 40 años en memoria.
        field = ds[var].isel(time=slice(-365, None)).mean("time")
        if float(field.max()) > 200:  # Kelvin → °C
            field = field - 273.15
        mesh = ax.pcolormesh(
            field["longitude"], field["latitude"], field, cmap="RdYlBu_r", shading="auto"
        )
        fig.colorbar(mesh, ax=ax, label="SST media 2025 (°C)", shrink=0.6)
        ds.close()

    tonnes = summary.groupby("economic_unit")["tonnes"].sum()
    drawn: set[tuple] = set()
    for key, cfg in units.items():
        bbox = cfg.get("bbox")
        if not bbox or key not in tonnes.index:
            continue
        corner = (bbox["lon_min"], bbox["lat_min"], bbox["lon_max"], bbox["lat_max"])
        if corner in drawn:  # varias cooperativas comparten bbox costero
            continue
        drawn.add(corner)
        ax.add_patch(
            plt.Rectangle(
                (bbox["lon_min"], bbox["lat_min"]),
                bbox["lon_max"] - bbox["lon_min"],
                bbox["lat_max"] - bbox["lat_min"],
                fill=False,
                edgecolor="#111",
                lw=1.4,
            )
        )
        label = region_label(cfg.get("region", key))
        ax.annotate(
            label,
            (bbox["lon_max"], (bbox["lat_min"] + bbox["lat_max"]) / 2),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8,
            va="center",
        )
    ax.set_xlabel("longitud (°)")
    ax.set_ylabel("latitud (°)")
    ax.set_title(
        "Unidades económicas del conjunto consolidado\n"
        "(recuadros = zona costera asignada a cada clúster de cooperativas)",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info(f"Mapa → {out_path}")


def main() -> None:
    settings = get_settings()
    df = pd.read_parquet(settings.processed_dir / "dataset_v1.parquet")
    df["ds"] = pd.to_datetime(df["ds"])
    units = yaml.safe_load((settings.configs_root / "economic_units.yaml").read_text())

    summary = series_summary(df)
    repo_root = settings.reports_root.parent
    write_series_table(summary, units, repo_root / "final_work" / "tables" / "series_summary.tex")
    plot_units_map(
        units,
        summary,
        settings.data_root / "raw" / "copernicus" / "sst_l4.nc",
        repo_root / "final_work" / "images" / "mapa_ues.png",
    )

    total = summary.groupby("species").agg(
        series=("economic_unit", "nunique"), tonnes=("tonnes", "sum")
    )
    print(total.to_string(float_format=lambda v: f"{v:,.1f}"))
    print(f"\nSeries totales: {len(summary)} · toneladas: {summary['tonnes'].sum():,.0f}")
    print(f"Rango: {summary['first'].min():%Y-%m-%d} → {summary['last'].max():%Y-%m-%d}")
    print(f"Días con captura (mediana por serie): {np.median(summary['catch_days']):.0f}")


if __name__ == "__main__":
    main()
