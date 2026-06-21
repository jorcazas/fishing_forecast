# PENDINGS.md

Mapa de lo que **falta para terminar el plan** (`PLAN.md`), separando lo que está
bloqueado por insumos externos de lo que es trabajo de código ya desbloqueado. Última
actualización: **2026-06-21**.

Estado global:
- **Código del ETL completo y testeado** (extract → transform → aggregate → consolidate →
  quality_checks), 81 tests verdes.
- **Arribos reales ya fluyen**: el export COBI (2016-2025) se ingiere y produce un
  `dataset_v1.parquet` real para langosta-SQ (reproduce el bache post-MHW 2021-2022). →
  los **baselines estadísticos de Fase 1.4 (ARIMA/Prophet) ya se pueden correr** sobre
  arribos solos, sin esperar a la oceanografía.
- **Falta** para el dataset *enriquecido*: SST/MHW reales (OISST) y las `x1..x16`
  (GlobColour/Copernicus) — gated por descargas grandes y credenciales.
- Artefacto LSTM del borrador (`.h5`) ya descargado para comparación.

Lo que sigue se ordena por bloqueador.

---

## 1. Bloqueadores duros (necesitan a Javier / COBI / credenciales)

Ninguno de estos se puede resolver desde el código; requieren un insumo externo.

| # | Bloqueador | Qué desbloquea | Acción concreta |
|---|---|---|---|
| B1 | **Credenciales GlobColour (FTP)** | `extract/globcolour.py`, `transform/globcolour.py`, columnas `x1..x16` | Regenerar credenciales (las del borrador expiraron) y ponerlas en `.env` (`GLOBCOLOUR_USER`/`GLOBCOLOUR_PASS`, ya previstas en `config.py`). |
| B2 | **Credenciales Copernicus Marine** | `extract/copernicus.py` (SDK `copernicusmarine`), SST L4 alternativa | Crear cuenta nueva, `copernicusmarine login`, credenciales a `.env`. El ETL hay que escribirlo con el SDK nuevo (motuclient está muerto desde mar-2024). |
| B3 | **Coordenadas TURF reales de COBI (shapefile/polígono)** | Recorte espacial fino por UE en `aggregate/ocean_by_ue.py` | Hoy se usa un **bbox placeholder** de San Quintín en `economic_units.yaml`. Reemplazar por el polígono real (cambio solo de config, sin re-ETL de código). |
| ~~B4~~ | ~~CSV legacy COBI `Arribos2017-2021.csv`~~ | **RESUELTO (2026-06-21)** | Archivo entregado en `data/raw/arribos/Arribos2017-2021.csv` (97k filas, 2016-2025). Se ingiere con `fishing-etl transform arribos --source cobi` (dialecto COBI). `dataset_v1` real generado: langosta-SQ reproduce el bache post-MHW 2021-2022. |
| B5 | **Artefactos del borrador en S3** (joblib XGB, `.h5` LSTM, métricas) | Comparación de métricas en Fase 1.4 | **Parcial (2026-06-21)**: bucket en `keys.json`, listado OK (12 objetos ≈21 GB). Descargado **`lstm_model_23-005.h5`** (2.8 GB, HDF5 válido) en `models/legacy/`. El XGB joblib y los JSON de métricas están **dentro de 11 zips `Tesis-*.zip` (~18 GB, un dump de la carpeta de tesis)** — no descargados; decidir si vale la pena o si Javier los tiene local. |

---

## 2. Operacional: correr descargas grandes (desbloqueado, pero pide confirmación)

El código existe; falta **ejecutarlo** porque implica datos pesados (CLAUDE.md: confirmar
antes de operaciones largas).

- [ ] **Descargar CONAPESCA** (`fishing-etl extract conapesca --years all`): 9 años ×
  ~150 MB = ~1.4 GB de `arribo_cosecha`. Extractor listo e idempotente.
- [ ] **Descargar NOAA OISST** (`fishing-etl extract oisst --years 1982-2025`): baseline
  1982-2011 + operativo 2012-2025 ≈ 44 archivos × ~150 MB ≈ **6-7 GB**. Decidir rango.
- [x] Correr el pipeline real de **arribos** (vía COBI, sin descargas): `transform arribos
  --source cobi → consolidate → qc` ya genera `dataset_v1.parquet` real (langosta-SQ con
  el bache post-MHW 2021-2022). Falta sumar `aggregate ocean` cuando haya OISST real.

> Decisión de fuente de arribos: hoy el spine de `dataset_v1` es **COBI** (cubre
> 2016-2025 y está local). CONAPESCA queda como fuente alterna/validación. Si algún día
> se usan ambas hay que decidir la **estrategia de unión/dedup** por `(ds, species, UE)`
> (qué fuente gana en periodos solapados) — ver §3.

---

## 3. Código de ETL desbloqueado y aún pendiente

No requieren insumos externos más allá de conocer el formato; se pueden hacer ya.

- [x] **`extract/cicese.py` + `transform/cicese.py`** — hechos (reescritos del legacy
  `etl/cicese.py`): índice REDMAR → `.dat` idempotente → mediana diaria por estación.
  **Falta verificar con datos reales**: el valor centinela de dato faltante de REDMAR
  (¿9999? ¿-99999?). Hoy `read_dat` acepta `na_values` explícito (default None); confirmar
  el centinela real y fijarlo (en config o en la llamada del CLI) para no sesgar medianas.
- [x] **Check de correlación SST CICESE vs OISST** — `quality_checks.check_sst_correlation`
  (Pearson sobre el solape diario, warning bajo `sst_cicese_correlation_min`). *Falta*
  llamarlo en el flujo real (necesita `interim/cicese/*` + serie OISST por región).
- [x] **Particionado en disco** por `species × year(ds)` (§4.2) —
  `consolidate.write_dataset_partitioned`.
- [x] **ADR** `docs/decisions/ADR-0001-y-missing.md` (decisión §4.4).
- [x] **Refactor**: patrón de descarga idempotente factorizado a `utils/download.py`;
  los 3 extractores (CONAPESCA/OISST/CICESE) ahora son wrappers delgados. Tests verdes.
- [x] **Export de compatibilidad** `dataset_v1 → lstm_data.csv` —
  `consolidate.export_lstm_csv` (filtra `lobster_red × litoral_bc_sur`, columnas
  `ds, y` + `x1..x16` cuando existan).
- [x] **Figura MHW** `viz/mhw_plot.plot_mhw_timeline` (SST + clim + umbral + eventos
  sombreados por categoría). *Falta* correrla con OISST real para generar el PNG.

### Pendientes de datos surgidos al ingerir COBI (2026-06-21)

- [ ] **Estrategia de unión CONAPESCA + COBI** si se usan ambas fuentes: hoy
  `transform arribos --source` escribe `interim/arribos.parquet` (una fuente pisa a la
  otra). Definir dedup por `(ds, species, UE)` cuando haga falta combinarlas.
- [ ] **Formas de producto excluidas a propósito** del mapping: solo se mapean las formas
  "ENT. FCA./FCO." (entero). Variantes "S.C." (sin concha), "CONCHA DE", "COLAS DE",
  "CARNE DE", "COCIDA" se descartan para no mezclar bases de peso. Si COBI quiere el peso
  total por especie habría que sumar formas con factores de conversión (decisión de dominio).
- [ ] **Hueco de langosta 2022+ en SQ**: el export COBI llega a 2025 pero las temporadas de
  langosta-SQ con captura van hasta 2021-2022. Confirmar con COBI si faltan años o si la UE
  dejó de reportar (afecta cuántas temporadas nuevas hay para Fase 1.4).
- [ ] **1 fila de langosta fuera de temporada** (warning `y_out_of_season` en QC real):
  revisar si es un arribo tardío legítimo o un error de fecha en el crudo.

---

## 4. Fase 1.4 — Re-entrenamiento del baseline (parcialmente desbloqueada)

`dataset_v1.parquet` ya es real para langosta-SQ (arribos COBI). Los modelos que **solo
usan `y`** (ARIMA, Prophet) se pueden entrenar **ya**. Los que usan covariables
oceanográficas (LGBM/XGBoost/LSTM con `x1..x16`/SST) esperan al enriquecimiento (B1/B2 +
OISST). La comparación contra el `.h5` del borrador necesita además TensorFlow/Keras
(no está en deps; el borrador usaba tf 2.7) y el XGB joblib (dentro de los zips de B5).

- [ ] Scripts `experiments/exp1_baseline_retrain/` por modelo. **Empezar por ARIMA y
  Prophet** (solo `y`); LGBM/XGBoost/LSTM cuando haya oceanografía.
- [ ] Partición temporal canónica corte **`2020-07-01`** + partición adicional
  `2024-06-01`.
- [ ] Métricas (MAE, RMSE, sMAPE, error de temporada) → `reports/metrics/exp1_*.json`.
- [ ] Tabla comparativa contra los números del paper (criterio: ±10% en la partición
  antigua). Cargar el `.h5` requiere agregar TensorFlow (pin tf 2.x) — diferir hasta el
  modelo LSTM.
- [ ] **Figura MHW** `reports/figures/mhw_timeline.png` (función `viz/mhw_plot` lista;
  desbloqueada en cuanto haya OISST real).

---

## 5. Fases 2-5 (dependen de Fase 1 cerrada con datos reales)

Resumen de lo que queda; detalle completo en `PLAN.md`.

- [ ] **Fase 2 — Feature engineering + SHAP**: módulos `features/{lags,rolling_stats,
  anomalies,calendar,interactions,pipeline}.py` + análisis SHAP + poda. **Crítico**: el
  shift de 90 días vive aquí (no en ETL, §5.5), y los tests deben verificar no-leakage.
- [ ] **Fase 3 — Modelo global multi-especie/UE**: el ETL ya soporta la granularidad
  `(ds, species, economic_unit)`, así que no hay re-ETL; falta el modelado (skforecast o
  darts) y la comparación global vs específico.
- [ ] **Fase 4 — CQR (intervalos calibrados)**: `mapie`, coverage empírico, CRPS,
  calibración condicional durante MHW, producto operativo para COBI.
- [ ] **Fase 5 (opcional) — TFT**: ADR de justificación + `pytorch-forecasting`/`darts`.

---

## 6. Higiene heredada del borrador (de `docs/repo_audit.md`, aún sin tocar)

El código nuevo vive en `src/fishing_forecast/`; el legacy (`etl/`, `forecasting_models/`
en la raíz) sigue intocado. Antes de reusarlo:

- [ ] Mover credenciales Postgres hardcodeadas (`etl/load/globcolour_load.py`,
  `etl/load/cicese_load.py`) a `.env` o borrar esos loaders si no se reusan.
- [ ] Quitar rutas hardcodeadas de Windows/Colab en los scripts de modelos.
- [ ] Borrar/mover a `legacy/` el `forecasting_models/modeling/code_wandb.py` (es un
  ejemplo CIFAR10/ResNet ajeno al proyecto).
- [ ] Decidir qué scripts del borrador se reescriben para Fase 1.4 y cuáles se archivan.

---

## Ruta crítica recomendada

1. **Ya, sin esperar nada**: arrancar **Fase 1.4 baseline estadístico** (ARIMA/Prophet)
   sobre el `dataset_v1` de arribos COBI — da los primeros números reproducidos.
2. Javier resuelve **B3** (bbox/shapefile) y confirma rango OISST → correr la descarga
   OISST (§2) → `aggregate ocean` → SST/MHW en el dataset → figura MHW (Fase 1.3).
3. En paralelo, **B1/B2** (credenciales) habilitan `x1..x16` → completan los modelos con
   covariables (LGBM/XGBoost/LSTM) de Fase 1.4.
4. Para comparar el LSTM del borrador: agregar TensorFlow y (si hace falta) bajar el XGB
   joblib de los zips de B5.
5. Recién con el dataset enriquecido tienen pleno sentido las Fases 2-5.
