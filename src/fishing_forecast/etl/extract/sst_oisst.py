"""Extractor de SST diaria NOAA OISST v2.1 (high-resolution, 0.25°).

Fuente: NOAA PSL — https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/
Un archivo netCDF por año: ``sst.day.mean.<YYYY>.nc`` (~150 MB/año, grid global 0.25°,
longitud en convención 0-360, variable ``sst``).

Se usa para el baseline climatológico de MHW (1982-2011, 30 años) y para la SST
operativa por UE. Licencia abierta, sin credenciales — por eso es el primer eslabón
oceanográfico que podemos implementar sin esperar a Copernicus/GlobColour.

El descargador es idempotente con el mismo patrón que el extractor CONAPESCA: cachea
ETag/Last-Modified/Content-Length en un ``.meta.json`` junto al archivo y omite la
descarga si el servidor reporta los mismos metadatos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
from loguru import logger

from fishing_forecast.utils import download as dl

#: Base PSL para los archivos anuales de OISST v2.1 high-res.
PSL_BASE_URL = "https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres"
#: Plantilla del nombre de archivo anual.
FILENAME_TEMPLATE = "sst.day.mean.{year}.nc"
#: Primer año disponible en OISST v2.1.
FIRST_YEAR = 1982


@dataclass(frozen=True)
class OISSTFileSpec:
    """Descriptor de un archivo anual de OISST."""

    year: int
    url: str
    filename: str

    def relative_path(self) -> Path:
        return Path(self.filename)


def build_specs(years: Iterable[int], *, base_url: str = PSL_BASE_URL) -> list[OISSTFileSpec]:
    """Construye los `OISSTFileSpec` para los años pedidos. Pura: no toca la red.

    Lanza `ValueError` si algún año es anterior a `FIRST_YEAR`.
    """
    specs: list[OISSTFileSpec] = []
    for year in sorted(set(years)):
        if year < FIRST_YEAR:
            raise ValueError(f"OISST v2.1 empieza en {FIRST_YEAR}; pediste {year}.")
        filename = FILENAME_TEMPLATE.format(year=year)
        specs.append(OISSTFileSpec(year=year, url=f"{base_url}/{filename}", filename=filename))
    return specs


def download_file(
    spec: OISSTFileSpec,
    dest_dir: Path,
    *,
    session: requests.Session | None = None,
    force: bool = False,
) -> Path:
    """Descarga ``spec`` a ``dest_dir`` idempotentemente.

    Delega en :func:`fishing_forecast.utils.download.download_file`. Timeout largo (300 s)
    porque los archivos anuales pesan ~150 MB.
    """
    return dl.download_file(
        spec.url, dest_dir / spec.relative_path(), session=session, force=force, timeout=300
    )


def extract(
    *,
    dest_dir: Path,
    years: Iterable[int],
    force: bool = False,
    base_url: str = PSL_BASE_URL,
    session: requests.Session | None = None,
) -> list[Path]:
    """Orquestador: descarga los archivos anuales de OISST de los años pedidos."""
    session = session or requests.Session()
    specs = build_specs(years, base_url=base_url)
    logger.info(f"Descargando {len(specs)} archivo(s) OISST a {dest_dir}")
    return [download_file(spec, dest_dir, session=session, force=force) for spec in specs]
