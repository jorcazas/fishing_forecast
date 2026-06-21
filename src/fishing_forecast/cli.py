"""CLI del paquete: `fishing-etl <comando>`.

Esqueleto inicial. Los comandos concretos se implementan en Fase 1.2.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich import print
from rich.table import Table

from fishing_forecast import __version__
from fishing_forecast.config import get_settings

app = typer.Typer(
    name="fishing-etl",
    help="ETL y orquestación del proyecto fishing_forecast.",
    no_args_is_help=True,
)

extract_app = typer.Typer(help="Descarga raw de cada fuente externa.")
app.add_typer(extract_app, name="extract")

transform_app = typer.Typer(help="Transforma raw → interim por fuente.")
app.add_typer(transform_app, name="transform")

aggregate_app = typer.Typer(help="Agregación espacial/temporal por UE + MHW.")
app.add_typer(aggregate_app, name="aggregate")


@app.command()
def info() -> None:
    """Muestra la versión y rutas configuradas."""
    settings = get_settings()
    print(f"[bold]fishing_forecast[/] v{__version__}")
    print(f"  data_root    = {settings.data_root}")
    print(f"  configs_root = {settings.configs_root}")
    print(f"  reports_root = {settings.reports_root}")
    print(f"  models_root  = {settings.models_root}")


@extract_app.command("conapesca")
def extract_conapesca(
    years: str = typer.Option(
        "all",
        help="Años separados por coma (ej. '2018,2019,2024') o 'all' para todos.",
    ),
    kinds: str = typer.Option(
        "arribo_cosecha",
        help="Tipos: 'arribo_cosecha', 'produccion', o 'all'. Coma-separados.",
    ),
    force: bool = typer.Option(False, help="Re-descargar aunque haya cache local."),
    list_only: bool = typer.Option(
        False,
        "--list-only",
        help="Solo listar los archivos descubiertos en el índice; no descargar nada.",
    ),
) -> None:
    """Descarga avisos de arribo/cosecha/producción desde CONAPESCA."""
    from fishing_forecast.etl.extract import arribos_conapesca

    settings = get_settings()
    dest = settings.raw_dir / "arribos" / "conapesca"

    if list_only:
        specs = arribos_conapesca.fetch_index()
        table = Table(title=f"CONAPESCA — {len(specs)} archivo(s) descubiertos")
        table.add_column("año", justify="right")
        table.add_column("kind")
        table.add_column("filename")
        for s in specs:
            table.add_row(str(s.year), s.kind, s.filename)
        print(table)
        return

    year_list = None if years == "all" else [int(y) for y in years.split(",")]
    kind_list = None if kinds == "all" else kinds.split(",")
    paths = arribos_conapesca.extract(
        dest_dir=dest,
        years=year_list,
        kinds=kind_list,  # type: ignore[arg-type]
        force=force,
    )
    print(f"[green]Descargados {len(paths)} archivo(s) en {dest}[/]")


@extract_app.command("oisst")
def extract_oisst(
    years: str = typer.Option(
        "1982-2011",
        help="Rango 'YYYY-YYYY' o lista coma-separada. Default = baseline climatológico MHW.",
    ),
    force: bool = typer.Option(False, help="Re-descargar aunque haya cache local."),
) -> None:
    """Descarga SST diaria NOAA OISST v2.1 (archivos anuales netCDF)."""
    from fishing_forecast.etl.extract import sst_oisst

    settings = get_settings()
    dest = settings.raw_dir / "sst" / "oisst"
    year_list = _parse_years(years)
    print(f"[yellow]OISST: {len(year_list)} año(s) (~150 MB c/u). Descargando a {dest}[/]")
    paths = sst_oisst.extract(dest_dir=dest, years=year_list, force=force)
    print(f"[green]Descargados {len(paths)} archivo(s) en {dest}[/]")


def _parse_years(years: str) -> list[int]:
    if "-" in years and "," not in years:
        lo, hi = (int(p) for p in years.split("-"))
        return list(range(lo, hi + 1))
    return [int(y) for y in years.split(",")]


@extract_app.command("cicese")
def extract_cicese(
    years: str = typer.Option("2011-2025", help="Rango 'YYYY-YYYY' o lista coma-separada."),
    force: bool = typer.Option(False, help="Re-descargar aunque haya cache local."),
) -> None:
    """Descarga los .dat de las estaciones CICESE (REDMAR) de cicese_stations.yaml."""
    import yaml

    from fishing_forecast.etl.extract import cicese

    settings = get_settings()
    cfg = yaml.safe_load((settings.configs_root / "cicese_stations.yaml").read_text("utf-8"))
    stations = [
        cicese.Station(name=name, code=meta["code"]) for name, meta in cfg["stations"].items()
    ]
    dest = settings.raw_dir / "cicese"
    result = cicese.extract(
        stations=stations, years=_parse_years(years), dest_dir=dest, force=force
    )
    total = sum(len(v) for v in result.values())
    print(f"[green]CICESE: {total} archivo(s) .dat en {dest}[/]")


@aggregate_app.command("ocean")
def aggregate_ocean(
    ue: str = typer.Option("litoral_bc_sur", help="Código de UE (clave en economic_units.yaml)."),
) -> None:
    """SST por UE (promedio bbox OISST) + índice MHW → data/interim/ocean_<ue>.parquet."""
    import yaml

    from fishing_forecast.etl.aggregate import ocean_by_ue
    from fishing_forecast.etl.aggregate.mhw import MHWParams

    settings = get_settings()
    oisst_dir = settings.raw_dir / "sst" / "oisst"
    paths = sorted(oisst_dir.glob("*.nc"))
    if not paths:
        raise typer.BadParameter(
            f"No hay netCDF en {oisst_dir}. Corre primero `fishing-etl extract oisst`."
        )

    ue_cfg = yaml.safe_load((settings.configs_root / "economic_units.yaml").read_text("utf-8"))
    if ue not in ue_cfg or "bbox" not in ue_cfg[ue]:
        raise typer.BadParameter(f"UE {ue!r} sin bbox en economic_units.yaml.")
    bbox = ue_cfg[ue]["bbox"]

    etl_cfg = yaml.safe_load((settings.configs_root / "etl.yaml").read_text("utf-8"))
    params = MHWParams.from_config(etl_cfg["mhw"])

    df = ocean_by_ue.sst_mhw_for_bbox(paths, bbox, params)
    out_path = settings.interim_dir / f"ocean_{ue}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression="zstd", index=False)
    print(f"[green]SST+MHW de {ue}: {len(df)} días → {out_path}[/]")


@transform_app.command("arribos")
def transform_arribos(
    source: str = typer.Option(
        "conapesca",
        help="Fuente de arribos: 'conapesca' (CSV descargados) o 'cobi' (export local).",
    ),
    all_species: bool = typer.Option(
        False,
        "--all-species",
        help="No filtrar por las especies de dataset_v1 (conserva todas las mapeadas).",
    ),
    all_units: bool = typer.Option(
        False,
        "--all-units",
        help="No filtrar por UE definidas; conserva todas las mapeadas en economic_units.yaml.",
    ),
) -> None:
    """Transforma los CSV crudos de arribos → data/interim/arribos.parquet.

    `--source conapesca` lee los CSV descargados; `--source cobi` lee el export local
    configurado en `etl.yaml: sources.arribos_cobi.csv_path`.
    """
    import yaml

    from fishing_forecast.etl.transform import arribos as tr_arribos

    settings = get_settings()
    etl_cfg = yaml.safe_load((settings.configs_root / "etl.yaml").read_text(encoding="utf-8"))

    if source not in tr_arribos.DIALECTS:
        raise typer.BadParameter(f"source debe ser uno de {sorted(tr_arribos.DIALECTS)}.")
    dialect = tr_arribos.DIALECTS[source]

    if source == "cobi":
        # csv_path en config es relativo a la raíz del repo (= configs_root.parent).
        csv_path = settings.configs_root.parent / etl_cfg["sources"]["arribos_cobi"]["csv_path"]
        if not csv_path.exists():
            raise typer.BadParameter(f"No existe el CSV legacy COBI en {csv_path}.")
        csv_paths = [csv_path]
    else:
        raw_dir = settings.raw_dir / "arribos" / "conapesca" / "arribo_cosecha"
        csv_paths = sorted(raw_dir.glob("*.csv"))
        if not csv_paths:
            raise typer.BadParameter(
                f"No hay CSV en {raw_dir}. Corre primero `fishing-etl extract conapesca`."
            )

    keep_species = None if all_species else etl_cfg.get("dataset_v1_species")
    economic_units_path = settings.configs_root / "economic_units.yaml"
    keep_units = None
    if not all_units:
        ue_cfg = yaml.safe_load(economic_units_path.read_text(encoding="utf-8")) or {}
        keep_units = list(ue_cfg.keys())

    out_path = settings.interim_dir / "arribos.parquet"
    df = tr_arribos.transform(
        csv_paths,
        species_mapping_path=settings.configs_root / "species_mapping.yaml",
        economic_units_path=economic_units_path,
        keep_species=keep_species,
        keep_units=keep_units,
        out_path=out_path,
        dialect=dialect,
    )
    print(
        f"[green]Transformadas {len(csv_paths)} CSV ({source}) → {len(df)} filas en {out_path}[/]"
    )


@transform_app.command("cicese")
def transform_cicese() -> None:
    """Transforma los .dat de cada estación CICESE → data/interim/cicese/<station>.parquet."""
    import yaml

    from fishing_forecast.etl.transform import cicese as tr_cicese

    settings = get_settings()
    cfg = yaml.safe_load((settings.configs_root / "cicese_stations.yaml").read_text("utf-8"))
    aggregates = cfg.get("daily_aggregates")

    written = 0
    for name, meta in cfg["stations"].items():
        dat_paths = sorted((settings.raw_dir / "cicese" / name).glob("*.dat"))
        if not dat_paths:
            print(f"[yellow]{name}: sin .dat; se omite (corre `extract cicese`).[/]")
            continue
        out_path = settings.interim_dir / "cicese" / f"{name}.parquet"
        df = tr_cicese.transform(
            dat_paths,
            station=name,
            region=meta.get("region"),
            aggregates=aggregates,
            out_path=out_path,
        )
        print(f"[green]{name}: {len(df)} días → {out_path}[/]")
        written += 1
    if not written:
        raise typer.BadParameter("Ninguna estación tenía .dat. Corre `fishing-etl extract cicese`.")


@app.command()
def consolidate(version: str = typer.Option("v1")) -> None:
    """Join interim → data/processed/dataset_{version}.parquet (esquema §4.1)."""
    import yaml

    from fishing_forecast.etl import consolidate as cons

    settings = get_settings()
    arribos_path = settings.interim_dir / "arribos.parquet"
    if not arribos_path.exists():
        raise typer.BadParameter(
            f"Falta {arribos_path}. Corre `fishing-etl transform arribos` primero."
        )

    etl_cfg = yaml.safe_load((settings.configs_root / "etl.yaml").read_text("utf-8"))
    seasons = yaml.safe_load((settings.configs_root / "season_calendars.yaml").read_text("utf-8"))
    date_range = etl_cfg["date_range"]

    # Recoge los interims oceanográficos disponibles (ocean_<ue>.parquet).
    ocean_by_ue = {
        p.stem.removeprefix("ocean_"): pd.read_parquet(p)
        for p in sorted(settings.interim_dir.glob("ocean_*.parquet"))
    }
    if not ocean_by_ue:
        print("[yellow]Sin interims oceanográficos (ocean_*.parquet); SST/MHW irán vacías.[/]")

    df = cons.consolidate(
        pd.read_parquet(arribos_path),
        season_calendars=seasons,
        date_start=date_range["start"],  # PyYAML parsea YYYY-MM-DD como datetime.date
        date_end=date_range["end"],
        ocean_by_ue=ocean_by_ue or None,
    )
    out_path = settings.processed_dir / f"dataset_{version}.parquet"
    cons.write_dataset(df, out_path)
    print(f"[green]Consolidado {len(df)} filas → {out_path}[/]")


@app.command()
def qc(
    dataset: str = typer.Option("data/processed/dataset_v1.parquet"),
    fail_on_warning: bool = typer.Option(False, "--fail-on-warning"),
) -> None:
    """Quality checks sobre el dataset consolidado."""
    import yaml

    from fishing_forecast.etl import quality_checks as qc_mod

    settings = get_settings()
    path = Path(dataset)
    if not path.exists():
        raise typer.BadParameter(f"No existe {path}. Corre `fishing-etl consolidate` primero.")

    etl_cfg = yaml.safe_load((settings.configs_root / "etl.yaml").read_text("utf-8"))
    species_cfg = yaml.safe_load(
        (settings.configs_root / "species_mapping.yaml").read_text("utf-8")
    )
    ue_cfg = yaml.safe_load((settings.configs_root / "economic_units.yaml").read_text("utf-8"))
    known_species = [m["code"] for m in species_cfg.get("mappings", [])]
    known_units = list(ue_cfg.keys())
    coverage_min = etl_cfg.get("quality_checks", {}).get("ocean_coverage_min", 0.80)

    issues = qc_mod.run_quality_checks(
        pd.read_parquet(path),
        known_species=known_species,
        known_units=known_units,
        ocean_coverage_min=coverage_min,
        fail_on_warning=fail_on_warning,
    )
    print(f"[green]QC OK[/] — {len(issues)} incidencia(s) (ninguna bloqueante).")


if __name__ == "__main__":
    app()
