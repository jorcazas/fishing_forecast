# PENDINGS.md

Mapa de lo que **falta para terminar el plan** (`PLAN.md`), separando lo que está
bloqueado por insumos externos de lo que es trabajo de código ya desbloqueado. Última
actualización: **2026-06-19**.

Estado global: el **camino de ETL de código está completo y testeado** (extract →
transform → aggregate → consolidate → quality_checks), verificado end-to-end con la
fixture (`transform arribos → consolidate → qc` produce `dataset_v1.parquet`). Lo que
falta para que el dataset sea *real y útil* depende de insumos externos y de correr
descargas grandes. Las Fases 2-5 dependen de tener ese dataset real.

---

## 1. Bloqueadores duros (necesitan a Javier / COBI / credenciales)

Ninguno de estos se puede resolver desde el código; requieren un insumo externo.

| # | Bloqueador | Qué desbloquea | Acción concreta |
|---|---|---|---|
| B1 | **Credenciales GlobColour (FTP)** | `extract/globcolour.py`, `transform/globcolour.py`, columnas `x1..x16` | Regenerar credenciales (las del borrador expiraron) y ponerlas en `.env` (`GLOBCOLOUR_USER`/`GLOBCOLOUR_PASS`, ya previstas en `config.py`). |
| B2 | **Credenciales Copernicus Marine** | `extract/copernicus.py` (SDK `copernicusmarine`), SST L4 alternativa | Crear cuenta nueva, `copernicusmarine login`, credenciales a `.env`. El ETL hay que escribirlo con el SDK nuevo (motuclient está muerto desde mar-2024). |
| B3 | **Coordenadas TURF reales de COBI (shapefile/polígono)** | Recorte espacial fino por UE en `aggregate/ocean_by_ue.py` | Hoy se usa un **bbox placeholder** de San Quintín en `economic_units.yaml`. Reemplazar por el polígono real (cambio solo de config, sin re-ETL de código). |
| ~~B4~~ | ~~CSV legacy COBI `Arribos2017-2021.csv`~~ | **RESUELTO (2026-06-21)** | Archivo entregado en `data/raw/arribos/Arribos2017-2021.csv` (97k filas, 2016-2025). Se ingiere con `fishing-etl transform arribos --source cobi` (dialecto COBI). `dataset_v1` real generado: langosta-SQ reproduce el bache post-MHW 2021-2022. |
| B5 | **Artefactos del borrador en S3** (joblib XGB, `.h5` LSTM, métricas) | Comparación de métricas en Fase 1.4 | Confirmar bucket/credenciales S3 (`S3_BUCKET_LEGACY` en `config.py`). |

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

## 4. Fase 1.4 — Re-entrenamiento del baseline (bloqueada por dataset real)

Depende de tener `dataset_v1.parquet` con datos reales (Sección 2) y de B5 para comparar.

- [ ] Scripts `experiments/exp1_baseline_retrain/` por modelo (ARIMA, Prophet, LGBM,
  XGBoost, LSTM, XGBoost+LSTM).
- [ ] Partición temporal canónica corte **`2020-07-01`** + partición adicional
  `2024-06-01`.
- [ ] Métricas (MAE, RMSE, sMAPE, error de temporada) → `reports/metrics/exp1_*.json`.
- [ ] Tabla comparativa contra los números del paper (criterio: ±10% en la partición
  antigua).
- [ ] **Figura MHW** `reports/figures/mhw_timeline.png` (desbloqueada en cuanto haya
  OISST real; `add_mhw(..., return_diagnostics=True)` ya expone `clim/thresh/in_mhw`).

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

1. Javier resuelve **B3** (bbox/shapefile) y confirma rango OISST → correr Sección 2.
2. Con dataset real: cerrar **Fase 1.3** (figura MHW) y **Fase 1.4** (baseline).
3. En paralelo, B1/B2 (credenciales) habilitan `x1..x16` para enriquecer el dataset.
4. Recién entonces tienen sentido las Fases 2-5.
