"""CLI del paquete: `fishing-etl <comando>`.

Esqueleto inicial. Los comandos concretos se implementan en Fase 1.2.
"""

from __future__ import annotations

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


@app.command()
def transform(sources: str = typer.Option("all")) -> None:
    """Transforma raw → interim por fuente. Implementación pendiente (Fase 1.2)."""
    raise NotImplementedError(f"transform({sources=}): pendiente (Fase 1.2)")


@app.command()
def consolidate(version: str = typer.Option("v1")) -> None:
    """Produce data/processed/dataset_{version}.parquet. Pendiente (Fase 1.2)."""
    raise NotImplementedError(f"consolidate({version=}): pendiente (Fase 1.2)")


@app.command()
def qc(dataset: str = typer.Option("data/processed/dataset_v1.parquet")) -> None:
    """Quality checks sobre el dataset consolidado. Pendiente (Fase 1.2)."""
    raise NotImplementedError(f"qc({dataset=}): pendiente (Fase 1.2)")


if __name__ == "__main__":
    app()
