"""Extractor de datos meteomareográficos CICESE (red REDMAR).

Fuente: http://redmar.cicese.mx/emmc/DATA/{CODE}/MIN/{year}/
Cada año tiene un índice HTML que lista archivos ``.dat`` (uno por periodo, sin header,
delimitados por espacios). Códigos de estación en `configs/cicese_stations.yaml`
(Isla Cedros = ICDN, Guerrero Negro = GRON).

Estos datos sirven para **QC de la SST de OISST** (correlación SST in-situ vs satélite)
y como covariables locales (nivel del mar, viento, etc.). REDMAR es HTTP plano, sin
credenciales.

El descargador es idempotente con el mismo patrón que CONAPESCA/OISST (cache
ETag/Last-Modified/Content-Length en `.meta.json`). Reescrito desde el script legacy
`etl/cicese.py` (que no era idempotente y usaba parsing frágil del HTML).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from loguru import logger

from fishing_forecast.utils import download as dl

REDMAR_BASE_URL = "http://redmar.cicese.mx/emmc/DATA"


@dataclass(frozen=True)
class Station:
    """Estación CICESE (nombre interno + código REDMAR)."""

    name: str
    code: str


def build_index_url(code: str, year: int, *, base_url: str = REDMAR_BASE_URL) -> str:
    """URL del índice anual de una estación."""
    return f"{base_url}/{code}/MIN/{year}/"


def parse_index_html(html: str, index_url: str) -> list[str]:
    """Extrae las URLs absolutas de los archivos ``.dat`` del índice. Pura (sin red)."""
    soup = BeautifulSoup(html, "lxml")
    urls: list[str] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        if href.lower().endswith(".dat"):
            urls.append(urljoin(index_url, href))
    return urls


def fetch_dat_urls(
    code: str, year: int, *, session: requests.Session | None = None, base_url: str = REDMAR_BASE_URL
) -> list[str]:
    """Descarga el índice anual y devuelve las URLs de los ``.dat``."""
    session = session or requests.Session()
    index_url = build_index_url(code, year, base_url=base_url)
    response = session.get(index_url, timeout=30)
    response.raise_for_status()
    return parse_index_html(response.text, index_url)


def download_file(
    url: str,
    dest_dir: Path,
    *,
    session: requests.Session | None = None,
    force: bool = False,
) -> Path:
    """Descarga ``url`` a ``dest_dir/<filename>`` idempotentemente.

    Delega en :func:`fishing_forecast.utils.download.download_file`.
    """
    filename = url.rsplit("/", 1)[-1]
    return dl.download_file(url, dest_dir / filename, session=session, force=force)


def extract(
    *,
    stations: Iterable[Station],
    years: Iterable[int],
    dest_dir: Path,
    force: bool = False,
    base_url: str = REDMAR_BASE_URL,
    session: requests.Session | None = None,
) -> dict[str, list[Path]]:
    """Descarga los ``.dat`` de cada estación/año a ``dest_dir/<station>/``.

    Devuelve `{station_name: [rutas descargadas]}`.
    """
    session = session or requests.Session()
    out: dict[str, list[Path]] = {}
    for station in stations:
        station_dir = dest_dir / station.name
        paths: list[Path] = []
        for year in years:
            try:
                urls = fetch_dat_urls(station.code, year, session=session, base_url=base_url)
            except requests.RequestException as exc:
                logger.warning(f"{station.name} {year}: índice no accesible ({exc}); se omite.")
                continue
            logger.info(f"{station.name} {year}: {len(urls)} archivo(s) .dat")
            for url in urls:
                paths.append(download_file(url, station_dir, session=session, force=force))
        out[station.name] = paths
    return out
