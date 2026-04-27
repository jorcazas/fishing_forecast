# fishing_forecast

Predicción del volumen de pesca de langosta (*Panulirus interruptus*) en San Quintín, Baja California, usando variables oceanográficas (GlobColour, Copernicus, CICESE) y arribos pesqueros (COBI, CONAPESCA). Tesis de licenciatura, ITAM Ciencia de Datos, en colaboración con COBI.

## Estructura

- `src/fishing_forecast/` — código del proyecto (ETL, features, modelos).
- `configs/` — YAMLs de configuración (UEs, especies, calendarios, vars).
- `data/{raw,interim,processed}/` — datos (gitignored).
- `experiments/` — scripts reproducibles por experimento.
- `models/` — artefactos entrenados (gitignored).
- `reports/{figures,metrics,sessions,etl}/` — métricas, figuras, sesiones.
- `notebooks/` — exploración (correr `nbstripout` antes de commitear).
- `docs/` — diseño (`repo_audit.md`, `etl_design.md`), ADRs, secciones de tesis.
- `legacy/` — código del borrador 2024 antes de archivarlo (vacío al inicio).
- `etl/` y `forecasting_models/` (raíz) — código del borrador 2024, deprecated; mover a `legacy/` cuando el pipeline nuevo esté validado.
- `PLAN.md` — plan de experimentos por fase.
- `CLAUDE.md` — instrucciones permanentes para asistencia con Claude Code.

## Setup

Requiere Python 3.11+ y [uv](https://github.com/astral-sh/uv).

```bash
# Instalar dependencias core + dev
uv sync --extra dev

# Instalar todos los extras (etl, models, prob, tracking, nb, dev)
uv sync --all-extras

# Configurar credenciales (copiar y rellenar)
cp .env.example .env

# Smoke test
uv run pytest

# CLI
uv run fishing-etl info
```

## Documentación

- [`docs/repo_audit.md`](docs/repo_audit.md) — auditoría inicial del repo.
- [`docs/etl_design.md`](docs/etl_design.md) — diseño del pipeline de ETL.
- [`PLAN.md`](PLAN.md) — fases del proyecto.

## Estado

En desarrollo. Fase 1.1 (diseño del ETL) cerrada; Fase 1.2 (implementación) en curso.
