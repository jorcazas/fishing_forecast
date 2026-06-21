"""Configuración centralizada del proyecto.

Todo valor de I/O y credenciales se carga desde `.env` (vía pydantic-settings).
Ningún módulo debe llamar a `os.getenv` directamente — siempre `get_settings()`.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Defaults de las rutas; se usan cuando la variable de entorno viene vacía.
_PATH_DEFAULTS = {
    "data_root": REPO_ROOT / "data",
    "configs_root": REPO_ROOT / "configs",
    "reports_root": REPO_ROOT / "reports",
    "models_root": REPO_ROOT / "models",
}


class Settings(BaseSettings):
    """Configuración tipada del proyecto.

    Lee de `.env` en la raíz del repo. Las variables de entorno reales tienen prioridad.
    """

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Rutas
    data_root: Path = Field(default=REPO_ROOT / "data")
    configs_root: Path = Field(default=REPO_ROOT / "configs")
    reports_root: Path = Field(default=REPO_ROOT / "reports")
    models_root: Path = Field(default=REPO_ROOT / "models")

    @field_validator("data_root", "configs_root", "reports_root", "models_root", mode="before")
    @classmethod
    def _empty_path_to_default(cls, value: object, info) -> object:
        """Una variable de entorno vacía (`DATA_ROOT=`) no debe pisar el default.

        Pasa cuando alguien copia `.env.example` (que trae las rutas en blanco) a `.env`.
        """
        if value is None or (isinstance(value, str) and not value.strip()):
            return _PATH_DEFAULTS[info.field_name]
        return value

    # Credenciales (vacías por default; el código que las usa debe validar antes de llamar)
    globcolour_user: str = ""
    globcolour_pass: str = ""
    copernicus_user: str = ""
    copernicus_pass: str = ""

    # S3 con artefactos del borrador (opcional). Las credenciales AWS viven en
    # `keys.json` (gitignored), no en `.env`. El bucket puede venir de `.env` o de keys.json.
    s3_bucket_legacy: str = ""
    keys_file: Path = Field(default=REPO_ROOT / "keys.json")

    # Postgres (opcional, solo si se retoma la carga a DB)
    pg_host: str = "localhost"
    pg_db: str = "cobi"
    pg_user: str = "postgres"
    pg_password: str = ""

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_root / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_root / "processed"

    def load_keys(self) -> dict[str, str]:
        """Lee `keys.json` (credenciales AWS y opcionalmente bucket/region). {} si no existe.

        Nunca loguear el contenido: trae secretos. Formato esperado:
        ``{"aws_access_key_id": ..., "aws_secret_access_key": ..., "region"?: ..., "bucket"?: ...}``.
        """
        if not self.keys_file.exists():
            return {}
        data = json.loads(self.keys_file.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"{self.keys_file} debe ser un objeto JSON con credenciales.")
        return data


@lru_cache
def get_settings() -> Settings:
    """Singleton para evitar releer `.env` en cada llamada."""
    return Settings()
