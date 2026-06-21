"""Extractor de avisos de arribo y producción publicados por CONAPESCA.

Fuente: https://conapesca.gob.mx/wb/cona/avisos_arribo_cosecha_produccion

CONAPESCA publica dos CSVs anuales por año (2018 en adelante):

- ``arribo_cosecha``  → "Avisos de Arribo y Cosecha"  (capturas + acuacultura, ~150 MB)
- ``produccion``      → "Avisos de Producción"        (procesamiento, ~150 MB)

Para nuestro target ``y`` (peso desembarcado) usamos el archivo de **arribo_cosecha**.
El de producción se descarga también porque sirve para contrastar y para futuros
experimentos (cadena de valor).

Las URLs son irregulares:
    pre-2025: https://nube.conapesca.gob.mx/datosabiertos/AVISOS_ MAYORES_MENORES_COSECHA_<YYYY>.csv
              (note the literal space between ``AVISOS_`` and ``MAYORES_``)
              https://nube.conapesca.gob.mx/datosabiertos/AVISOS_PRODUCCION_<YYYY>.csv
    2025+:    https://nube.conapesca.gob.mx/datosabiertos/<YYYY>/aviso_arribo/...
              https://nube.conapesca.gob.mx/datosabiertos/<YYYY>/produccion/...

El extractor descarga el HTML del índice y descubre los archivos in vivo, así que
si CONAPESCA cambia el patrón de URL en años futuros lo seguimos automáticamente.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal
from urllib.parse import unquote, urljoin

import requests
from bs4 import BeautifulSoup
from loguru import logger

from fishing_forecast.utils import download as dl

INDEX_URL = "https://conapesca.gob.mx/wb/cona/avisos_arribo_cosecha_produccion"

FileKind = Literal["arribo_cosecha", "produccion"]

#: Detección del tipo de archivo a partir del nombre/URL.
_KIND_PATTERNS: tuple[tuple[FileKind, re.Pattern[str]], ...] = (
    ("arribo_cosecha", re.compile(r"AVISOS[ _]+MAYORES[_ ]MENORES[_ ]COSECHA", re.IGNORECASE)),
    ("produccion", re.compile(r"AVISOS[ _]+PRODUCCION", re.IGNORECASE)),
)

#: Año embebido en la URL/nombre del archivo (4 dígitos consecutivos).
_YEAR_PATTERN = re.compile(r"(?<!\d)(20\d{2})(?!\d)")


@dataclass(frozen=True)
class FileSpec:
    """Descriptor de un archivo descubierto en el índice."""

    year: int
    kind: FileKind
    url: str
    filename: str

    def relative_path(self) -> Path:
        """Ruta relativa donde guardar este archivo dentro de ``data/raw/arribos/conapesca/``."""
        return Path(self.kind) / self.filename


def parse_index_html(html: str, base_url: str = INDEX_URL) -> list[FileSpec]:
    """Extrae la lista de archivos descargables del HTML del índice de CONAPESCA.

    Pura: no toca la red, no lee archivos. Recibe HTML, devuelve lista ordenada por
    (año, kind). Si un enlace no encaja con ningún patrón conocido, se descarta con
    un debug log — la página tiene navegación que también incluye `<a>` no relevantes.
    """
    soup = BeautifulSoup(html, "lxml")
    specs: list[FileSpec] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        if not href.lower().endswith(".csv"):
            continue
        absolute = urljoin(base_url, href)
        filename = unquote(absolute.rsplit("/", 1)[-1])
        kind = _detect_kind(filename)
        if kind is None:
            logger.debug(f"Archivo CSV ignorado (no encaja con patrones): {filename}")
            continue
        year = _detect_year(absolute)
        if year is None:
            logger.warning(f"No se pudo extraer el año de {absolute!r}; ignorando.")
            continue
        specs.append(FileSpec(year=year, kind=kind, url=absolute, filename=filename))
    specs.sort(key=lambda s: (s.year, s.kind))
    return specs


def _detect_kind(filename: str) -> FileKind | None:
    for kind, pattern in _KIND_PATTERNS:
        if pattern.search(filename):
            return kind
    return None


def _detect_year(url: str) -> int | None:
    match = _YEAR_PATTERN.search(url)
    if match is None:
        return None
    return int(match.group(1))


def fetch_index(session: requests.Session | None = None, url: str = INDEX_URL) -> list[FileSpec]:
    """Descarga el HTML del índice y devuelve los `FileSpec` descubiertos."""
    session = session or requests.Session()
    response = session.get(url, timeout=30)
    response.raise_for_status()
    # La página declara ISO-8859-1, pero los `<a href=...>` son ASCII puro.
    response.encoding = response.encoding or "iso-8859-1"
    return parse_index_html(response.text, base_url=url)


def download_file(
    spec: FileSpec,
    dest_dir: Path,
    *,
    session: requests.Session | None = None,
    force: bool = False,
) -> Path:
    """Descarga ``spec`` a ``dest_dir/spec.relative_path()`` de manera idempotente.

    Delega en :func:`fishing_forecast.utils.download.download_file` (cache `.meta.json`
    con ETag/Last-Modified/Content-Length). ``force=True`` re-descarga siempre.
    """
    return dl.download_file(
        spec.url, dest_dir / spec.relative_path(), session=session, force=force
    )


def extract(
    *,
    dest_dir: Path,
    years: Iterable[int] | None = None,
    kinds: Iterable[FileKind] | None = None,
    force: bool = False,
    session: requests.Session | None = None,
) -> list[Path]:
    """Orquestador: descubre el índice y descarga los archivos solicitados.

    ``years`` y ``kinds`` filtran la lista descubierta. None → todos.
    """
    session = session or requests.Session()
    specs = fetch_index(session=session)
    if not specs:
        raise RuntimeError(
            "No se encontraron archivos en el índice de CONAPESCA. "
            "Probable cambio de formato; revisar manualmente."
        )

    selected = _filter_specs(specs, years=years, kinds=kinds)
    if not selected:
        raise ValueError(
            f"Filtro vacío: ningún archivo encaja con years={years!r}, kinds={kinds!r}. "
            f"Disponibles: {sorted({s.year for s in specs})} × {sorted({s.kind for s in specs})}"
        )

    logger.info(f"Descargando {len(selected)} archivo(s) a {dest_dir}")
    paths: list[Path] = []
    for spec in selected:
        paths.append(download_file(spec, dest_dir, session=session, force=force))
    return paths


def _filter_specs(
    specs: list[FileSpec],
    *,
    years: Iterable[int] | None,
    kinds: Iterable[FileKind] | None,
) -> list[FileSpec]:
    year_set = set(years) if years is not None else None
    kind_set = set(kinds) if kinds is not None else None
    return [
        s
        for s in specs
        if (year_set is None or s.year in year_set)
        and (kind_set is None or s.kind in kind_set)
    ]


def specs_summary(specs: list[FileSpec]) -> list[dict[str, str | int]]:
    """Helper para imprimir/loggear el resultado del descubrimiento."""
    return [asdict(s) for s in specs]
