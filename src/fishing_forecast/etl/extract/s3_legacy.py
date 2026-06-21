"""Descarga de artefactos del borrador 2024 desde S3 (modelos joblib, `.h5`, métricas).

Las credenciales AWS se leen de ``keys.json`` (gitignored) vía
``Settings.load_keys()`` — **nunca** se loguea su contenido. El bucket puede venir de
``keys.json`` (`"bucket"`) o de ``S3_BUCKET_LEGACY`` en ``.env``.

Se usa en Fase 1.4 para comparar las métricas reproducidas contra las del borrador. Las
funciones reciben el cliente boto3 inyectado, así que se testean con un cliente mockeado
sin tocar AWS.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from fishing_forecast.config import Settings, get_settings

#: Alias laxo para el cliente boto3 S3 (evita depender de los stubs mypy-boto3).
S3Client = Any


@dataclass(frozen=True)
class S3Object:
    """Objeto descubierto en el bucket."""

    key: str
    size: int


def build_client(settings: Settings | None = None):  # -> S3Client
    """Construye un cliente boto3 S3 con las credenciales de ``keys.json``.

    Si ``keys.json`` no trae credenciales, cae en la cadena default de boto3
    (``~/.aws/credentials``, variables de entorno, rol de instancia).
    """
    import boto3

    settings = settings or get_settings()
    keys = settings.load_keys()
    kwargs: dict[str, str] = {}
    if keys.get("aws_access_key_id") and keys.get("aws_secret_access_key"):
        kwargs["aws_access_key_id"] = keys["aws_access_key_id"]
        kwargs["aws_secret_access_key"] = keys["aws_secret_access_key"]
    if keys.get("region"):
        kwargs["region_name"] = keys["region"]
    return boto3.client("s3", **kwargs)


def resolve_bucket(settings: Settings | None = None, override: str | None = None) -> str:
    """Resuelve el bucket: ``override`` > ``keys.json['bucket']`` > ``S3_BUCKET_LEGACY``."""
    settings = settings or get_settings()
    bucket = override or settings.load_keys().get("bucket") or settings.s3_bucket_legacy
    if not bucket:
        raise ValueError(
            "Falta el bucket de S3. Define 'bucket' en keys.json, S3_BUCKET_LEGACY en .env, "
            "o pásalo explícito."
        )
    return bucket


def list_artifacts(client: S3Client, bucket: str, prefix: str = "") -> list[S3Object]:
    """Lista objetos bajo ``prefix`` (paginado). Devuelve [(key, size)]."""
    paginator = client.get_paginator("list_objects_v2")
    objects: list[S3Object] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            objects.append(S3Object(key=obj["Key"], size=int(obj.get("Size", 0))))
    return objects


def download_artifact(
    client: S3Client, bucket: str, key: str, dest_dir: Path, *, force: bool = False
) -> Path:
    """Descarga ``key`` a ``dest_dir/<key>`` preservando subrutas. Idempotente por tamaño.

    Si el archivo local ya existe con el mismo tamaño que el objeto S3, se omite.
    """
    target = dest_dir / key
    target.parent.mkdir(parents=True, exist_ok=True)

    if not force and target.exists():
        head = client.head_object(Bucket=bucket, Key=key)
        if target.stat().st_size == int(head.get("ContentLength", -1)):
            logger.debug(f"[skip] {key} ya está descargado.")
            return target

    logger.info(f"[get ] s3://{bucket}/{key}")
    client.download_file(bucket, key, str(target))
    return target


def sync(
    client: S3Client,
    bucket: str,
    prefix: str,
    dest_dir: Path,
    *,
    force: bool = False,
) -> list[Path]:
    """Descarga todos los objetos bajo ``prefix`` a ``dest_dir``. Devuelve las rutas."""
    objects = list_artifacts(client, bucket, prefix)
    if not objects:
        logger.warning(f"Sin objetos bajo s3://{bucket}/{prefix}")
    return [download_artifact(client, bucket, o.key, dest_dir, force=force) for o in objects]
