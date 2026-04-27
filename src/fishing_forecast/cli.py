"""CLI del paquete: `fishing-etl <comando>`.

Esqueleto inicial. Los comandos concretos se implementan en Fase 1.2.
"""

from __future__ import annotations

import typer
from rich import print

from fishing_forecast import __version__
from fishing_forecast.config import get_settings

app = typer.Typer(
    name="fishing-etl",
    help="ETL y orquestación del proyecto fishing_forecast.",
    no_args_is_help=True,
)


@app.command()
def info() -> None:
    """Muestra la versión y rutas configuradas."""
    settings = get_settings()
    print(f"[bold]fishing_forecast[/] v{__version__}")
    print(f"  data_root    = {settings.data_root}")
    print(f"  configs_root = {settings.configs_root}")
    print(f"  reports_root = {settings.reports_root}")
    print(f"  models_root  = {settings.models_root}")


@app.command()
def extract(
    sources: str = typer.Option(
        "all",
        help="Fuentes separadas por coma: globcolour,copernicus,cicese,arribos,all",
    ),
    date_from: str = typer.Option("2017-01-01", "--from"),
    date_to: str = typer.Option("today", "--to"),
) -> None:
    """Descarga los archivos crudos de cada fuente. Implementación pendiente (Fase 1.2)."""
    raise NotImplementedError("extract: pendiente de implementar en Fase 1.2")


@app.command()
def transform(sources: str = typer.Option("all")) -> None:
    """Transforma raw → interim por fuente. Implementación pendiente (Fase 1.2)."""
    raise NotImplementedError("transform: pendiente de implementar en Fase 1.2")


@app.command()
def consolidate(version: str = typer.Option("v1")) -> None:
    """Produce data/processed/dataset_{version}.parquet. Pendiente (Fase 1.2)."""
    raise NotImplementedError("consolidate: pendiente de implementar en Fase 1.2")


@app.command()
def qc(dataset: str = typer.Option("data/processed/dataset_v1.parquet")) -> None:
    """Quality checks sobre el dataset consolidado. Pendiente (Fase 1.2)."""
    raise NotImplementedError("qc: pendiente de implementar en Fase 1.2")


if __name__ == "__main__":
    app()
