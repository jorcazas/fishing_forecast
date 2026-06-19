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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
from loguru import logger

#: Base PSL para los archivos anuales de OISST v2.1 high-res.
PSL_BASE_URL = "https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres"
#: Plantilla del nombre de archivo anual.
FILENAME_TEMPLATE = "sst.day.mean.{year}.nc"
#: Primer año disponible en OISST v2.1.
FIRST_YEAR = 1982
#: Tamaño de chunk para streaming (1 MiB).
DOWNLOAD_CHUNK_SIZE = 1024 * 1024


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
    """Descarga ``spec`` a ``dest_dir`` idempotentemente (ver módulo CONAPESCA)."""
    session = session or requests.Session()
    target = dest_dir / spec.relative_path()
    meta_path = target.with_suffix(target.suffix + ".meta.json")
    target.parent.mkdir(parents=True, exist_ok=True)

    if not force and target.exists() and meta_path.exists():
        cached_meta = json.loads(meta_path.read_text())
        if _server_matches_cache(session, spec.url, cached_meta):
            logger.info(f"[skip] {spec.filename} ya está actualizado.")
            return target

    logger.info(f"[get ] {spec.filename}")
    with session.get(spec.url, timeout=300, stream=True) as response:
        response.raise_for_status()
        content_length = int(response.headers.get("Content-Length", 0))
        bytes_written = 0
        tmp_path = target.with_suffix(target.suffix + ".part")
        with tmp_path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if chunk:
                    fh.write(chunk)
                    bytes_written += len(chunk)
        tmp_path.replace(target)
        meta = {
            "url": spec.url,
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "content_length": content_length or bytes_written,
            "downloaded_bytes": bytes_written,
        }
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    logger.info(f"[done] {spec.filename} ({bytes_written / (1024 * 1024):.1f} MB)")
    return target


def _server_matches_cache(
    session: requests.Session, url: str, cached_meta: dict[str, str | int | None]
) -> bool:
    try:
        head = session.head(url, timeout=30, allow_redirects=True)
    except requests.RequestException as exc:
        logger.warning(f"HEAD falló ({exc}); asumiendo cache desactualizado.")
        return False
    if not head.ok:
        return False
    server_etag = head.headers.get("ETag")
    server_last_mod = head.headers.get("Last-Modified")
    server_size = head.headers.get("Content-Length")
    if server_etag and cached_meta.get("etag") and server_etag == cached_meta["etag"]:
        return True
    if (
        server_last_mod
        and cached_meta.get("last_modified")
        and server_last_mod == cached_meta["last_modified"]
    ):
        return True
    if (
        server_size
        and cached_meta.get("content_length")
        and int(server_size) == cached_meta["content_length"]
    ):
        return True
    return False


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
