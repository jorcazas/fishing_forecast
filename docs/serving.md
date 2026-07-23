# Front de inferencia por zona (Docker)

Aplicación web mínima para que las cooperativas consulten el **pronóstico calibrado de captura**
(intervalos CQR) por especie y unidad económica, sobre las zonas incorporadas al pool.

## Qué sirve

- **API** (FastAPI, `src/fishing_forecast/serving/`): al arrancar entrena una sola vez la CQR de
  producción (misma lógica que Exp 4: XGBoost cuantílico en log + conformalización Mondrian por
  serie) y cachea, por serie, el pronóstico diario calibrado (mediana + bandas 80/90 %).
- **Front** (`frontend/index.html`): selector de especie → zona, gráfico de abanico diario
  (observado vs. bandas calibradas) y panel de *backtest* por temporada (captura observada +
  cobertura empírica del intervalo del 90 % en temporada).

Endpoints: `GET /api/health`, `GET /api/series`, `GET /api/forecast?series=<especie@ue>`, `GET /`.

## Correr con Docker

```bash
docker compose up --build      # construye y levanta en http://localhost:8000
```

El **primer arranque entrena el modelo (~30-60 s)**; `GET /api/health` devuelve `ready:true`
cuando termina (el front muestra un spinner y hace *polling* mientras tanto).

Variable de entorno relevante:

- `FF_SERVE_CUT` (default `2024-06-01`): corte de prueba que define el horizonte servido. Con
  `2024-06-01` el bache post-MHW queda en el entrenamiento y el horizonte cubre 2024-2026.

Sin compose:

```bash
docker build -t fishing-forecast-serve:latest .
docker run -p 8000:8000 fishing-forecast-serve:latest
```

## Correr en local (sin Docker)

```bash
uv sync --extra serve --extra models
uv run --no-sync uvicorn fishing_forecast.serving.api:app --host 0.0.0.0 --port 8000
```

## Notas de honestidad (importante)

- El producto es el **intervalo diario calibrado**, no un total puntual. La suma de medianas
  diarias es un mal estimador del total por temporada (las capturas se concentran en pocos días):
  subestima las series de gran pico y sobreestima las bajas. Por eso el panel por temporada muestra
  la **captura observada** y la **cobertura empírica**, y la estimación central solo como referencia.
- La **cobertura de confianza** que muestra la app es **en temporada** (15-sep–15-feb para langosta):
  fuera de temporada la captura es 0 y el intervalo la cubre trivialmente, lo que infla el número.
  Las zonas de historia larga (El Rosario, costa norte) cubren ~92-99 %; las de historia corta
  (Vizcaíno, Abreojos–La Purísima, solo CONAPESCA 2022+) cubren ~63-85 %: su calibración aún depende
  de acumular temporadas (la tesis de fondo).
- La app **no pronostica la oceanografía futura**; el horizonte llega hasta la última fecha con
  covariables. Es una herramienta de pronóstico calibrado sobre el periodo con datos, no un
  oráculo de temporadas futuras arbitrarias.

## Notas de mantenimiento

- **Calendario de temporada (resuelto 2026-07-23)**: `configs/season_calendars.yaml` ahora declara
  la ventana reglamentaria (15-sep–15-feb) para **todas** las UEs de langosta (antes solo
  `litoral_bc_sur`), así que `in_season` del dataset ya es fiable. La app conserva
  `_in_lobster_season` como ventana reglamentaria autocontenida para el etiquetado por temporada,
  pero coincide con el flag del dataset. Corregir esto re-derivó las métricas de Exp 4 (ver tesis
  §6.9, progresión 7→21 series).
