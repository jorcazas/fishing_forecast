"""API mínima de inferencia (FastAPI) sobre las zonas del pool de langosta/abulón/erizo.

Al arrancar carga el store de producción una vez (``get_store``: el artefacto serializado en
``models/final/store.json`` si existe, o entrenando la CQR en caliente si no) y lo cachea; los
endpoints solo leen el cache. Sirve además un front estático mínimo.

Ejecutar en local:
    uv run --extra serve --extra models uvicorn fishing_forecast.serving.api:app --reload
o vía Docker (ver Dockerfile / docker-compose.yml).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from fishing_forecast.serving.forecast import DEFAULT_CUT, DEFAULT_STORE_PATH, get_store

#: Directorio del front estático (montado en la imagen Docker junto al paquete).
FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"

_state: dict = {"store": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Si hay artefacto serializado (`models/final/store.json`, ver su README) se sirve ese —el
    # mismo pronóstico auditado— y el arranque es inmediato; si no, se entrena en caliente.
    logger.info(f"[serving] preparando store (artefacto esperado en {DEFAULT_STORE_PATH})…")
    _state["store"] = get_store()
    logger.info("[serving] listo para servir")
    yield
    _state["store"] = None


app = FastAPI(
    title="Pronóstico pesquero — inferencia por zona",
    description="Intervalos calibrados (CQR) de captura por especie y unidad económica.",
    version="1.0.0",
    lifespan=lifespan,
)


def _store():
    store = _state["store"]
    if store is None:
        raise HTTPException(status_code=503, detail="El modelo aún se está entrenando; reintenta.")
    return store


@app.get("/api/health")
def health() -> dict:
    ready = _state["store"] is not None
    return {"status": "ok" if ready else "loading", "ready": ready, "cut_date": DEFAULT_CUT}


@app.get("/api/series")
def list_series() -> JSONResponse:
    """Catálogo de series (especie × unidad económica) para poblar los selectores."""
    store = _store()
    return JSONResponse({"cut_date": store.cut_date, "series": store.list_series()})


@app.get("/api/forecast")
def forecast(series: str) -> JSONResponse:
    """Pronóstico calibrado (mediana + bandas 80/90%) y resumen por temporada de una serie."""
    store = _store()
    rec = store.series.get(series)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Serie desconocida: {series!r}")
    return JSONResponse({"cut_date": store.cut_date, **rec})


@app.get("/")
def index() -> FileResponse:
    idx = FRONTEND_DIR / "index.html"
    if not idx.exists():
        raise HTTPException(status_code=500, detail=f"Falta el front en {FRONTEND_DIR}.")
    return FileResponse(idx)


# Sirve cualquier asset estático adicional (favicon, etc.) si el directorio existe.
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
