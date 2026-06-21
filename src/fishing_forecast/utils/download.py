"""Descarga HTTP idempotente con cache de metadatos.

Patrón compartido por los extractores (CONAPESCA, OISST, CICESE): descarga un archivo
solo si cambió en el servidor, usando un manifiesto ``<archivo>.meta.json`` con
ETag/Last-Modified/Content-Length. Descarga por streaming a un ``.part`` y rename
atómico al final. No hace resume parcial (los archivos completos caben una vez en disco).
"""

from __future__ import annotations

import json
from pathlib import Path

import requests
from loguru import logger

#: Tamaño de chunk para streaming de descargas (1 MiB).
DOWNLOAD_CHUNK_SIZE = 1024 * 1024


def server_matches_cache(
    session: requests.Session, url: str, cached_meta: dict[str, str | int | None]
) -> bool:
    """True si el servidor (vía HEAD) reporta el mismo ETag/Last-Modified/Content-Length."""
    try:
        head = session.head(url, timeout=20, allow_redirects=True)
    except requests.RequestException as exc:
        logger.warning(f"HEAD falló ({exc}); asumiendo cache desactualizado.")
        return False
    if not head.ok:
        return False
    # Cualquier metadato que coincida basta para considerar el cache vigente.
    for header, meta_key in (("ETag", "etag"), ("Last-Modified", "last_modified")):
        server_val = head.headers.get(header)
        if server_val and cached_meta.get(meta_key) and server_val == cached_meta[meta_key]:
            return True
    server_size = head.headers.get("Content-Length")
    return bool(
        server_size
        and cached_meta.get("content_length")
        and int(server_size) == cached_meta["content_length"]
    )


def download_file(
    url: str,
    target: Path,
    *,
    session: requests.Session | None = None,
    force: bool = False,
    timeout: int = 120,
    chunk_size: int = DOWNLOAD_CHUNK_SIZE,
) -> Path:
    """Descarga ``url`` a ``target`` de forma idempotente. Devuelve ``target``.

    Si existe el manifiesto ``target.meta.json`` y el servidor reporta los mismos
    metadatos, omite la descarga. ``force=True`` re-descarga siempre.
    """
    session = session or requests.Session()
    target.parent.mkdir(parents=True, exist_ok=True)
    meta_path = target.with_suffix(target.suffix + ".meta.json")

    if not force and target.exists() and meta_path.exists():
        cached_meta = json.loads(meta_path.read_text())
        if server_matches_cache(session, url, cached_meta):
            logger.debug(f"[skip] {target.name} ya está actualizado.")
            return target

    logger.info(f"[get ] {target.name}")
    with session.get(url, timeout=timeout, stream=True) as response:
        response.raise_for_status()
        content_length = int(response.headers.get("Content-Length", 0))
        bytes_written = 0
        tmp_path = target.with_suffix(target.suffix + ".part")
        with tmp_path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)
                    bytes_written += len(chunk)
        tmp_path.replace(target)
        meta = {
            "url": url,
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "content_length": content_length or bytes_written,
            "downloaded_bytes": bytes_written,
        }
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    logger.debug(f"[done] {target.name} ({bytes_written / (1024 * 1024):.1f} MB)")
    return target
