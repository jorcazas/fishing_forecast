"""Genera los insumos del capítulo de Datos de la tesis a partir del dataset real.

Dos artefactos, ambos derivados de `data/processed/dataset_v1.parquet` (y del `.nc` de SST
para el mapa), de modo que la tesis nunca cite cifras escritas a mano:

- ``final_work/tables/series_summary.tex``: una fila por serie (especie x unidad económica)
  con temporadas disponibles, días con captura, tonelaje y cobertura de las covariables.
- ``final_work/images/mapa_ues.png``: mapa de las unidades económicas sobre la SST media,
  que muestra el gradiente térmico del rango (San Quintín al norte, Bahía Magdalena al sur).
- ``fig_splits.png``, ``fig_desplazamiento.png`` y ``fig_zona_sq.png``: figuras propias que
  sustituyen a las ilustraciones de terceros del borrador (esquema de ventana expansiva,
  ejemplo del desplazamiento de 90 días con SST real, y detalle del bbox de San Quintín).

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


#: Nombre corto de cada cooperativa para la tabla de series (el nombre legal completo
#: desborda el ancho de página; el mapeo es explícito para no truncar a ciegas).
COOP_SHORT = {
    "litoral_bc_sur": "Litoral de BC",
    "pabellon_sq": "El Pabellón",
    "rocas_san_martin": "Rocas de San Martín",
    "er_el_chute": "El Chute",
    "er_isla_san_geronimo": "Isla San Jerónimo",
    "er_mortera_leyva": "Mortera de Leyva",
    "er_regasa": "Regasa",
    "er_scpp_ensenada": "SCPP Ensenada",
    "punta_canoas": "Punta Canoas",
    "isla_cedros": r"Pesc.\ Nac.\ de Abulón",
    "vizcaino_asuncion": "Leyes de Reforma",
    "vizcaino_emancipacion": "Emancipación",
    "vizcaino_natividad": "Buzos y Pescadores",
    "vizcaino_tortugas": "Bahía Tortugas",
    "abreojos_progreso": "Progreso",
    "abreojos_punta": "Punta Abreojos",
    "abreojos_san_ignacio": r"Calif.\ de San Ignacio",
    "la_purisima": "La Purísima",
    "magdalena_bahia": "Bahía Magdalena",
    "magdalena_chale": "Puerto Chalé",
    "magdalena_san_carlos": "Puerto San Carlos",
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
    header = (
        r"Especie & Zona (cooperativa) & Temp. & Días & Ton. & "
        r"Cob.\ SST & Cob.\ CHL \\"
    )
    lines = [
        "% Generado por experiments/thesis_assets.py — no editar a mano.",
        r"\begingroup\footnotesize\setlength{\tabcolsep}{3.5pt}",
        r"\begin{longtable}{llrrrrc}",
        r"\caption{Series (especie $\times$ unidad económica) del conjunto consolidado "
        r"2017-2026. `Temp.' cuenta las temporadas con al menos un día de captura; `Días' son "
        r"los días con captura; `Ton.' las toneladas desembarcadas del periodo; la cobertura "
        r"es la fracción de días con el valor oceanográfico disponible.}"
        r"\label{tab:series_resumen}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for _, r in summary.iterrows():
        coop = COOP_SHORT.get(
            r["economic_unit"], units.get(r["economic_unit"], {}).get("name", r["economic_unit"])
        )
        zone = region_label(r["region"])
        # Si la cooperativa se llama igual que la zona, no hay nada que desambiguar.
        zona = zone if coop == zone else f"{zone} ({coop})"
        lines.append(
            f"{SPECIES_LABEL.get(r['species'], r['species'])} & "
            f"{_latex_escape(zona)} & "
            f"{r['seasons']:d} & {r['catch_days']:d} & {r['tonnes']:,.1f} & "
            f"{r['cov_sst']:.0%} & {r['cov_chl']:.0%} \\\\".replace("%", "\\%")
        )
    lines.append(r"\end{longtable}")
    lines.append(r"\endgroup")
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


def plot_expanding_splits(out_path: Path) -> None:
    """Diagrama propio de la validación con ventana expansiva (Figura del Cap. 3).

    Sustituye a la figura tomada de un artículo de terceros: azul = entrenamiento,
    naranja = bloque de validación, en español, con la misma semántica.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrow, Rectangle

    n_folds, total = 4, 10.0
    train_c, val_c, rest_c = "#4878b0", "#e8853d", "#f0f0f0"
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.add_patch(Rectangle((0, n_folds), total, 0.6, color="#9fc79f", ec="#555"))
    ax.text(
        total / 2,
        n_folds + 0.3,
        "Serie completa (orden cronológico)",
        ha="center",
        va="center",
        fontsize=9,
    )
    val_len = total / (n_folds + 1)
    for k in range(n_folds):
        y = n_folds - 1 - k
        train_end = val_len * (k + 1)
        ax.add_patch(Rectangle((0, y), train_end, 0.6, color=train_c, ec="#555"))
        ax.add_patch(Rectangle((train_end, y), val_len, 0.6, color=val_c, ec="#555"))
        if train_end + val_len < total:
            ax.add_patch(
                Rectangle(
                    (train_end + val_len, y),
                    total - train_end - val_len,
                    0.6,
                    color=rest_c,
                    ec="#bbb",
                )
            )
        ax.text(
            train_end / 2,
            y + 0.3,
            "entrenamiento",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
        )
        ax.text(
            train_end + val_len / 2,
            y + 0.3,
            "validación",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
        )
    ax.add_patch(
        FancyArrow(
            total + 0.35,
            n_folds + 0.3,
            0,
            -(n_folds + 0.1),
            width=0.02,
            head_width=0.14,
            head_length=0.18,
            color="#555",
        )
    )
    ax.text(
        total + 0.55,
        n_folds / 2 + 0.3,
        "iteraciones",
        rotation=90,
        va="center",
        fontsize=8,
        color="#555",
    )
    ax.set_xlim(-0.2, total + 1.0)
    ax.set_ylim(-0.3, n_folds + 0.9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    logger.info(f"Figura splits → {out_path}")


def plot_shift_example(df: pd.DataFrame, out_path: Path) -> None:
    """Ejemplo del desplazamiento de 90 días con la SST real de San Quintín.

    Sustituye a la figura ilustrativa tomada de internet: misma idea (serie original
    contra la serie desplazada +90 días) pero con datos del propio conjunto.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = (
        df[(df["economic_unit"] == "litoral_bc_sur") & (df["species"] == "lobster_red")]
        .sort_values("ds")
        .set_index("ds")["sst"]
    )
    sub = sub.loc["2019-01-01":"2020-12-31"]
    shifted = sub.copy()
    shifted.index = shifted.index + pd.Timedelta(days=90)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(sub.index, sub.values, color="#4878b0", lw=1.2, label="SST original ($t$)")
    ax.plot(
        shifted.index, shifted.values, color="#e8b13d", lw=1.2, label="SST desplazada ($t+90$ d)"
    )
    ax.set_ylabel("SST (°C)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    logger.info(f"Figura desplazamiento → {out_path}")


def plot_zone_sq(units: dict, sst_path: Path, out_path: Path) -> None:
    """Detalle de la zona costera (bounding box) de San Quintín sobre la SST media.

    Sustituye a la captura de Google Maps: mismo propósito (ubicar la zona de la UE
    insignia) pero generado del propio `.nc` de SST y del bbox de `economic_units.yaml`.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import xarray as xr

    bbox = units["litoral_bc_sur"]["bbox"]
    pad = 0.8
    pad_e = 0.15  # hacia el este solo un margen corto: más allá está el Golfo de California
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ds = xr.open_dataset(sst_path)
    var = next(v for v in ds.data_vars if ds[v].ndim >= 3)
    field = (
        ds[var]
        .sel(
            longitude=slice(bbox["lon_min"] - pad, bbox["lon_max"] + pad_e),
            latitude=slice(bbox["lat_min"] - pad, bbox["lat_max"] + pad),
        )
        .isel(time=slice(-365, None))
        .mean("time")
    )
    if float(field.max()) > 200:  # Kelvin → °C
        field = field - 273.15
    mesh = ax.pcolormesh(
        field["longitude"], field["latitude"], field, cmap="RdYlBu_r", shading="auto"
    )
    fig.colorbar(mesh, ax=ax, label="SST media 2025 (°C)", shrink=0.8)
    ds.close()
    ax.add_patch(
        plt.Rectangle(
            (bbox["lon_min"], bbox["lat_min"]),
            bbox["lon_max"] - bbox["lon_min"],
            bbox["lat_max"] - bbox["lat_min"],
            fill=False,
            edgecolor="#111",
            lw=1.8,
        )
    )
    ax.annotate(
        "bbox San Quintín",
        (bbox["lon_min"], bbox["lat_max"]),
        xytext=(4, 5),
        textcoords="offset points",
        fontsize=9,
    )
    ax.set_xlabel("longitud (°)")
    ax.set_ylabel("latitud (°)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    logger.info(f"Figura zona SQ → {out_path}")


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
    images = repo_root / "final_work" / "images"
    plot_expanding_splits(images / "fig_splits.png")
    plot_shift_example(df, images / "fig_desplazamiento.png")
    plot_zone_sq(
        units, settings.data_root / "raw" / "copernicus" / "sst_l4.nc", images / "fig_zona_sq.png"
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
