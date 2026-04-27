"""Configuración centralizada del proyecto.

Todo valor de I/O y credenciales se carga desde `.env` (vía pydantic-settings).
Ningún módulo debe llamar a `os.getenv` directamente — siempre `get_settings()`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


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

    # Credenciales (vacías por default; el código que las usa debe validar antes de llamar)
    globcolour_user: str = ""
    globcolour_pass: str = ""
    copernicus_user: str = ""
    copernicus_pass: str = ""

    # S3 con artefactos del borrador (opcional)
    s3_bucket_legacy: str = ""

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


@lru_cache
def get_settings() -> Settings:
    """Singleton para evitar releer `.env` en cada llamada."""
    return Settings()
