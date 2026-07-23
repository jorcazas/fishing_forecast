# Imagen de inferencia (multi-stage): API FastAPI + front mínimo para pronóstico calibrado por zona.
# Instala solo el extra `serve` (FastAPI + uvicorn + XGBoost). El builder resuelve el entorno y
# elimina las libs CUDA de XGBoost (~400 MB, inútiles en CPU); la imagen final solo copia el venv
# ya limpio → sin cadena de build ni nvidia.

# ---------- builder ----------
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder
WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# 1) Dependencias (capa cacheable). README/LICENSE los pide hatchling para la metadata del wheel.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-install-project --extra serve

# 2) Código + install del paquete (editable) y limpieza de las libs CUDA en la MISMA capa final
#    del venv → no quedan en ninguna capa commit.
COPY src ./src
RUN uv sync --frozen --extra serve \
    && rm -rf /app/.venv/lib/python3.13/site-packages/nvidia* \
    && find /app/.venv -name "__pycache__" -type d -prune -exec rm -rf {} +

# ---------- runtime ----------
FROM python:3.13-slim-bookworm
# libgomp1: runtime de OpenMP que XGBoost necesita.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV FF_SERVE_CUT=2024-06-01 \
    PATH="/app/.venv/bin:$PATH"

# venv ya resuelto y limpio + código + datos + front. El install editable apunta a /app/src, que
# se copia a la misma ruta → el paquete resuelve y las rutas de `get_settings`/front (relativas al
# módulo) caen en /app. `dataset_v1.parquet` (~3 MB) se hornea; el resto de `data/` va en .dockerignore.
COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY configs ./configs
COPY frontend ./frontend
COPY data/processed/dataset_v1.parquet ./data/processed/dataset_v1.parquet

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=20 \
    CMD python -c "import urllib.request,sys,json; sys.exit(0 if json.load(urllib.request.urlopen('http://127.0.0.1:8000/api/health')).get('ready') else 1)"

CMD ["uvicorn", "fishing_forecast.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
