"""Extractor de productos Copernicus Marine vía el SDK `copernicusmarine` (v2).

Reemplaza el ETL del borrador basado en `motuclient` (muerto desde mar-2024). Descarga un
subconjunto espaciotemporal de cada producto declarado en `configs/copernicus_vars.yaml`
(por ahora SST L4; luego corrientes/salinidad) recortado al bounding box de la zona de
estudio, a un netCDF por producto en `data/raw/copernicus/`.

Las credenciales salen de `.env` (`COPERNICUS_USER`/`PASS` vía `Settings`); si están vacías,
el SDK usa el archivo de login (`~/.copernicusmarine`). Idempotente: omite la descarga si el
netCDF ya existe (salvo `force`).

`build_subset_kwargs` es pura (mapea config → kwargs del SDK) y se testea sin red; la
descarga real se inyecta vía `subset_fn` para poder mockearla.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable

import yaml
from loguru import logger

from fishing_forecast.config import Settings, get_settings


@dataclass(frozen=True)
class ProductSpec:
    """Un producto Copernicus a descargar (de `copernicus_vars.yaml`)."""

    dataset_id: str
    short_name: str
    variables: list[str]
    region: dict[str, float]
    start: str | None = None  # override de fecha inicial (p.ej. color del océano: ~1997+)


def load_products(config_path: Path) -> list[ProductSpec]:
    """Lee `copernicus_vars.yaml` → lista de `ProductSpec`."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    products = []
    for p in cfg.get("products", []):
        products.append(
            ProductSpec(
                dataset_id=p["dataset_id"],
                short_name=p["short_name"],
                variables=list(p["variables"]),
                region=p["region"],
                start=str(p["start"]) if p.get("start") is not None else None,
            )
        )
    if not products:
        raise ValueError(f"Sin productos en {config_path}.")
    return products


def _iso(d: date | datetime | str) -> str:
    if isinstance(d, str):
        return d if "T" in d else f"{d}T00:00:00"
    if isinstance(d, datetime):
        return d.isoformat()
    return f"{d.isoformat()}T00:00:00"


def build_subset_kwargs(
    product: ProductSpec,
    *,
    start: date | datetime | str,
    end: date | datetime | str,
    output_dir: Path,
) -> dict:
    """Mapea un `ProductSpec` + rango de fechas a los kwargs de `copernicusmarine.subset`.

    Pura: no toca la red. `coordinates_selection_method="inside"` recorta estrictamente
    dentro del bbox.
    """
    r = product.region
    return {
        "dataset_id": product.dataset_id,
        "variables": list(product.variables),
        "minimum_longitude": float(r["lon_min"]),
        "maximum_longitude": float(r["lon_max"]),
        "minimum_latitude": float(r["lat_min"]),
        "maximum_latitude": float(r["lat_max"]),
        "start_datetime": _iso(start),
        "end_datetime": _iso(end),
        "coordinates_selection_method": "inside",
        "output_directory": str(output_dir),
        "output_filename": f"{product.short_name}.nc",
    }


def download_product(
    product: ProductSpec,
    *,
    start: date | datetime | str,
    end: date | datetime | str,
    output_dir: Path,
    force: bool = False,
    settings: Settings | None = None,
    subset_fn: Callable | None = None,
) -> Path:
    """Descarga un producto a `output_dir/<short_name>.nc`. Idempotente por existencia.

    `subset_fn` permite inyectar un mock en tests; por defecto usa `copernicusmarine.subset`.
    """
    settings = settings or get_settings()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{product.short_name}.nc"

    if target.exists() and not force:
        logger.info(f"[skip] {target.name} ya existe.")
        return target

    if subset_fn is None:
        import copernicusmarine

        subset_fn = copernicusmarine.subset

    # Un producto puede fijar su propia fecha inicial (p.ej. color del océano ~1997+),
    # distinta del baseline MHW de la SST.
    eff_start = product.start or start
    kwargs = build_subset_kwargs(product, start=eff_start, end=end, output_dir=output_dir)
    kwargs["overwrite"] = True
    # Pasa credenciales explícitas solo si están en .env; si no, el SDK usa su login file.
    if settings.copernicus_user.strip() and settings.copernicus_pass.strip():
        kwargs["username"] = settings.copernicus_user
        kwargs["password"] = settings.copernicus_pass

    logger.info(f"[get ] {product.dataset_id} → {target.name} ({eff_start}..{end})")
    subset_fn(**kwargs)
    return target


def extract(
    *,
    config_path: Path,
    start: date | datetime | str,
    end: date | datetime | str,
    output_dir: Path,
    force: bool = False,
    settings: Settings | None = None,
    subset_fn: Callable | None = None,
) -> list[Path]:
    """Descarga todos los productos de `copernicus_vars.yaml` al rango/bbox dados."""
    settings = settings or get_settings()
    products = load_products(config_path)
    logger.info(f"Copernicus: {len(products)} producto(s) → {output_dir}")
    return [
        download_product(
            p,
            start=start,
            end=end,
            output_dir=output_dir,
            force=force,
            settings=settings,
            subset_fn=subset_fn,
        )
        for p in products
    ]
