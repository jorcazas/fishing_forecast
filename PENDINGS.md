# PENDINGS.md

Mapa de lo que **falta para terminar el plan** (`PLAN.md`), separando lo que está
bloqueado por insumos externos de lo que es trabajo de código ya desbloqueado. Última
actualización: **2026-07-21**.

## Qué falta ahora (snapshot 2026-07-21, ordenado por prioridad)

Fases 1-3 cerradas con datos reales; ya no hay bloqueadores de insumos. Lo que queda:

1. **Cuantiles base más apretados (Fase 4 residual)**: la CQR ya está **calibrada** (Exp 4 +
   afinado 2026-07-21: calibrar en temporada bajó la cobertura de 97.5% a ~86% en el nivel 90%,
   cerca de nominal; el conformal `normalized` mejora un poco a `split`). Lo que queda ancho es la
   **cota superior en días pico**, y eso lo fija el **modelo cuantílico base** (q0.95 en log +
   `expm1`), no el conformal → afinar requiere mejores cuantiles/más datos (regularizar, Optuna
   sobre los cuantílicos, o más temporadas), no otro envoltorio conformal. Además: leve
   sub-cobertura (86%<90%) por shift conf→test (el test cruza el crash post-MHW).
2. **Fase 5 (opcional) — TFT**: prueba de techo; ADR de justificación + `pytorch-forecasting`/`darts`.
3. **Endurecer `pooled_log`** (Fase 3 residual): Optuna sobre el pool log; investigar por qué
   langosta@Cedros (mayor escala) empeora con log; probar pesos por serie u objetivo por-grupo.
4. **Más datos** (mayor palanca real, según Exp 2/2.3): más UEs (El Rosario/Ensenada, en
   `economic_units.yaml`, confirmar bboxes) y más temporadas; unión CONAPESCA+COBI.
5. **Feature engineering residual (Fase 2)**: `anomalies`, `interactions`, `rolling` configurable.
6. **Figura MHW** `reports/figures/mhw_timeline.png` (ver §4) — producible, falta correr.
7. **Higiene heredada** del borrador (§6) y fix natbib/latexmk (error author-year no fatal a
   `pdflatex`); labels placeholder duplicados del borrador original (`fig:enter-label`,
   `tab:catch_comparison`) — renombrar por unicidad.

**Hecho (2026-07-21)** — documentación de tesis: **Exp 2.3 (SHAP)** y **Exp 4 (CQR)** integrados a
`final_work.tex` (nuevas subsecciones + Tablas `extension_shap`/`extension_cqr` + conclusión).
Añadidas refs `romano2019`, `lundberg2017`; corregido el `\bibitem{greff2017}` duplicado (el 2º era
Hyndman, re-etiquetado `hyndman2021`). Compila a 26 págs, citas resueltas.

Estado global:
- **Código del ETL completo y testeado** (extract → transform → aggregate → consolidate →
  quality_checks) + **métricas y script de baseline ARIMA/Prophet**. 88 tests verdes.
- **Arribos reales ya fluyen**: el export COBI (2016-2025) se ingiere y produce un
  `dataset_v1.parquet` real para langosta-SQ (reproduce el bache post-MHW 2021-2022). →
  los **baselines estadísticos de Fase 1.4 (ARIMA/Prophet) ya se pueden correr** sobre
  arribos solos, sin esperar a la oceanografía.
- **Credenciales/insumos resueltos**: GlobColour (FTP verificado), Copernicus (login
  válido), arribos COBI ingeridos, `.h5` del borrador descargado, polígono TURF decidido
  (bbox). CONAPESCA 2018-2026 descargado.
- **Falta** para el dataset *enriquecido*: **código** de los extractores oceanográficos
  (`extract/copernicus.py`, `extract/globcolour.py` + transforms) y **correr** la descarga
  OISST. Ya nada de esto está bloqueado por insumos externos.

Lo que sigue se ordena por bloqueador.

---

## 1. Bloqueadores duros (necesitan a Javier / COBI / credenciales)

**Estado (2026-06-21): todos resueltos o decididos.** B1/B2 credenciales OK y verificadas,
B4 (arribos COBI) ingerido, B3 (polígono TURF) decidido a favor del bbox, B5 (.h5) bajado.
Lo que queda ya es **código** (escribir los extractores oceanográficos) + **correr la
descarga OISST**, no insumos externos.

| # | Bloqueador | Qué desbloquea | Acción concreta |
|---|---|---|---|
| ~~B1~~ | ~~GlobColour / color del océano (`x1..x16`)~~ | **RESUELTO vía Copernicus (2026-07-05)** | El FTP GlobColour da archivos **globales sin subset** (4 km = 18 MB/var/día → ~630 GB para 16 vars × 6 años: inviable). Se optó por la **misma data GlobColour re-distribuida por Copernicus** (server-side subset, como la SST). Integradas 6 variables ópticas/biológicas: **CHL, KD490, SPM, ZSD, BBP, CDM** (4 km diario, 2015-2026) → columnas en `dataset_v1` (cobertura 88-100%). No en Copernicus (atmosféricas, omitidas): POC, PIC, PAR, aerosoles, nubosidad. |
| ~~B2~~ | ~~Credenciales Copernicus Marine~~ | **RESUELTO + SST/MHW en `dataset_v1` (2026-06-22)** | OSTIA SST **1982-01-01→2025-12-18** en `data/raw/copernicus/sst_l4.nc` (231 MB; el rango arranca en el baseline MHW). `aggregate ocean --source copernicus` → `interim/ocean_litoral_bc_sur.parquet` (SST °C + MHW Hobday con baseline 1982-2011) → `consolidate` → **`dataset_v1` ya trae `sst`/`sst_anomaly`/`mhw_category`/`mhw_intensity`** (langosta-SQ 96.5% SST no-nula). **Validado**: el Blob 2014-16 sale fuerte (2015: 293 días MHW). No hizo falta `transform/copernicus.py`: `aggregate/ocean_by_ue` lee el `.nc` directo (detecta `latitude`/`longitude` + var única, convierte Kelvin→°C). |
| B3 | **Polígono TURF de San Quintín** | Recorte fino por UE en `aggregate/ocean_by_ue.py` | **COBI ya no está disponible** para entregar el shapefile. Busqué en internet (2026-06-21): el dataset público de TURFs de Villaseñor-Derbez (`jcvdav/ReserveEffect`) cubre Quintana Roo + Isla Natividad (lat 18.7-28.0°N), **NO San Quintín** (~30.4°N); el polígono de la cooperativa de SQ no es descargable públicamente. **Decisión pragmática**: a resolución OISST (0.25°≈27 km) y aun GlobColour (4 km) un **bbox costero de SQ basta** (es lo que hizo el borrador). Se mantiene el bbox de `economic_units.yaml` (lon −117..−115, lat 30..31.5). Polígono exacto = nice-to-have, no bloqueante. |
| ~~B4~~ | ~~CSV legacy COBI `Arribos2017-2021.csv`~~ | **RESUELTO (2026-06-21)** | Archivo entregado en `data/raw/arribos/Arribos2017-2021.csv` (97k filas, 2016-2025). Se ingiere con `fishing-etl transform arribos --source cobi` (dialecto COBI). `dataset_v1` real generado: langosta-SQ reproduce el bache post-MHW 2021-2022. |
| B5 | **Artefactos del borrador en S3** (joblib XGB, `.h5` LSTM, métricas) | Comparación de métricas en Fase 1.4 | **Parcial (2026-06-21)**: bucket en `keys.json`, listado OK (12 objetos ≈21 GB). Descargado **`lstm_model_23-005.h5`** (2.8 GB, HDF5 válido) en `models/legacy/`. El XGB joblib y los JSON de métricas están **dentro de 11 zips `Tesis-*.zip` (~18 GB, un dump de la carpeta de tesis)** — no descargados; decidir si vale la pena o si Javier los tiene local. |

---

## 2. Operacional: correr descargas grandes (desbloqueado, pero pide confirmación)

El código existe; falta **ejecutarlo** porque implica datos pesados (CLAUDE.md: confirmar
antes de operaciones largas).

- [x] **Descargar CONAPESCA** (2026-06-21): `fishing-etl extract conapesca --years all`
  bajó los 9 CSV de `arribo_cosecha` 2018-2026 (~1.3 GB) a
  `data/raw/arribos/conapesca/arribo_cosecha/`. Falta **transformarlos/usarlos**:
  `transform arribos --source conapesca` los lleva a interim, pero el spine de `dataset_v1`
  hoy es COBI → decidir la unión/dedup CONAPESCA+COBI antes de combinarlos (ver §3). Útiles
  para validar langosta-SQ y para otras UEs/especies (Fase 3).
- [~] **NOAA OISST** (`extract oisst`): **ya no es necesario para el camino crítico** —
  Copernicus OSTIA cubre SST 1982-2025 (baseline + análisis) y la MHW ya está en
  `dataset_v1`. OISST queda como fuente **alternativa/validación** (~6-7 GB), opcional.
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
- [ ] **Copernicus REP vs cobertura reciente**: `METOFFICE-GLO-SST-L4-REP-OBS-SST` es
  reprocesado (cobertura hasta ~hace meses). Si la descarga completa no llega a 2026, sumar
  el NRT `METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2` para los meses recientes (otro `ProductSpec`).
- [ ] **`InsecureRequestWarning`** del SDK copernicusmarine (su backend S3 CloudFerro hace
  HTTPS sin verificar). Es interno del SDK, no de nuestro código; la descarga es válida.
  Revisar si una versión nueva lo arregla o si conviene silenciarlo.
- [ ] **Polígono de Isla Natividad disponible** (descargado en
  `data/raw/turf/reserveeffect/turfs.{shp,dbf,...}`, atributo `Coop="SCPP Buzos y Pescadores
  de la Baja California"`): si se agrega esa UE en Fase 3, usarlo como polígono real.
  Requiere soporte de polígono (no solo bbox) en `ocean_by_ue` + `geopandas`/`shapely`.

---

## 4. Fase 1.4 — Re-entrenamiento del baseline (parcialmente desbloqueada)

`dataset_v1.parquet` ya es real para langosta-SQ (arribos COBI). **Hecho (código):**
- [x] `src/fishing_forecast/evaluation/metrics.py` — MAE, RMSE, sMAPE, error de suma de
  temporada (puro, 7 tests).
- [x] `experiments/exp1_baseline_retrain/baseline.py` — script reproducible: carga
  `dataset_v1`, filtra langosta-SQ, serie diaria (NaN in-season→0, recorte al último día
  con captura), corte **2020-07-01**, ajusta **ARIMA** (rejilla chica por AIC) y **Prophet**
  (si está instalado), escribe `reports/metrics/exp1_*.json`, figura y `exp1_summary.md`.

**Corrido (2026-06-21):** `uv run python experiments/exp1_baseline_retrain/baseline.py`
con Prophet instalado. Resultados en `reports/metrics/exp1_baseline_*.json` y
`reports/exp1_baseline_summary.md`:

| modelo | MAE | RMSE | sMAPE% | suma temporada 2020-21 | suma temporada **2021-22** (crash) |
|---|---|---|---|---|---|
| ARIMA(3,0,3) | 345 | 463 | 138% | −26% | **+291%** |
| Prophet | 372 | 576 | 129% | +62% | **+418%** |

**Hallazgo**: ambos baselines (solo `y`) **sobrepredicen brutalmente la temporada
2021-2022** (real 31 t; predicen 120-160 t) porque no "ven" la ola de calor marina que
colapsó la captura. Esto **motiva todo el proyecto**: el error de suma de temporada que
importa solo baja con el índice MHW + covariables oceanográficas (el ensamble del borrador
lograba ~8.7%). Es el "piso" contra el cual comparar.

**Pendiente:**
- [x] **Deps en el lock**: resuelto. Las deps runtime del ETL (requests, beautifulsoup4,
  lxml, xarray, netCDF4, matplotlib) se movieron al **core** de `pyproject.toml` (estaban en
  el extra `etl`, lo que hacía que `uv sync --extra models` rompiera la CLI por falta de
  `lxml`). `statsmodels`/`prophet`/`joblib` quedan fijados vía extra `models`. Sync correcto:
  `uv sync --extra models --extra dev`.
- [ ] Corte adicional `2024-06-01`: **no aplica** (no hay langosta-SQ tras 2022); ver §3.
- [x] **Modelo con covariables corrido (Exp 2, 2026-06-22)**:
  `experiments/exp2_covariates/covariate_model.py` (XGBoost + `features/covariates.py`,
  SST/MHW desplazadas 90 días, sin leakage). Resultado de suma de temporada 2021-2022:
  XGBoost **+368%** vs ARIMA +291% / Prophet +418% — **las covariables NO arreglan el crash
  todavía** (el modelo sí usa `sst_roll90_lag90` y `mhw_category_roll365_lag90` en su top-5,
  pero con solo ~3 temporadas de train no hay ejemplos suficientes de la relación MHW→captura).
  MAE diario 327 (lig. mejor que ARIMA 345). **Hallazgo que motiva Fase 3**: el valor de las
  covariables necesita más temporadas y/o pooling entre especies/UEs (modelo global).
- [ ] Comparación formal vs el paper / LSTM del borrador: requiere **TensorFlow/Keras**
  (el `.h5` ya está en `models/legacy/`; el borrador usaba tf 2.7) y el XGB joblib (zips B5).
- [ ] **LGBM/LSTM** y tuning (Optuna) sobre el mismo setup — opcional; el cuello de botella
  es la cantidad de temporadas, no el modelo.
- [ ] **Figura MHW** `reports/figures/mhw_timeline.png`: ahora **producible** (ya hay SST
  real 1982-2025). Falta correr `add_mhw(..., return_diagnostics=True)` sobre la serie de SQ
  y pasarla a `viz/mhw_plot.plot_mhw_timeline` (el `aggregate ocean` actual no guarda
  `clim`/`thresh`/`in_mhw`; añadir un flag o un pequeño script).

---

## 5. Fases 2-5 (dependen de Fase 1 cerrada con datos reales)

Resumen de lo que queda; detalle completo en `PLAN.md`.

- [~] **Fase 2 — Feature engineering + SHAP**: arrancado. `features/covariates.py`
  (calendario + lags + SST/MHW + **6 vars de color del océano** CHL/KD490/SPM/ZSD/BBP/CDM,
  desplazadas 90 días, sin leakage, testeado). **Hallazgo (2026-07-05)**: al añadir las OC al
  modelo de una sola serie (Exp 2) el desempeño **empeora** (MAE 327→424) — sobreajuste con
  ~35 features y solo ~3 temporadas; el modelo SÍ usa las OC (`bbp/cdm/zsd_roll90_lag90` en el
  top-5) pero sin más datos no ayudan. **Exp 2.3 SHAP (2026-07-05)**: podar por SHAP
  (mean|SHAP| ≥ 1% → 35→16 features) **NO** arregla el sobreajuste (MAE 424→445, error temp.
  2021-22 399%→447%): un poco peor. Ranking SHAP coherente (calendario `doy_sin`/`in_season`
  domina, luego `bbp_roll90_lag90`, `y_lag365`, SST/OC), o sea las features son razonables — el
  cuello es **volumen de datos**, no cuáles features. → la selección de features sola no basta:
  se necesita **más datos (Fase 3)** y/o **regularización más fuerte / y normalizada**. Falta
  aún `anomalies`, `interactions`, `rolling` configurable.
- [~] **Fase 3 — Modelo global multi-especie/UE**: marco + **multi-UE** (Exp 3, 2026-07-03) +
  **normalización de `y` por serie (Exp 3.2, 2026-07-17)**. Se agregó **Isla Cedros** (2ª UE,
  ~28°N) → 5 series `(especie, UE)`; SST/MHW por UE. Exp 3: pooling todo-junto **confundido por
  escala** (2/5). **Exp 3.2 lo resuelve**: normalizar el objetivo por serie antes de agrupar.
  `pooled_log` (log1p) **gana/empata vs específico en 4/5 series (0.8)** — cumple criterio ≥0.60;
  rescata abulón (raw RMSE 33-35 → log 3.5-7) y mejora langosta@SQ (544 vs 659). `pooled_z`
  (z-score por serie) queda peor (0.2). **→ el modelo global de producción es el pool sobre
  `log1p(y)`.** **Pendiente**: (a) endurecer el ganador — Optuna sobre `pooled_log`, revisar por
  qué langosta@Cedros pierde (¿serie de mayor escala sobre-comprimida?), quizá objetivo por-grupo
  o pesos por serie; (b) **más UEs** — El Rosario/Ensenada (nombres en `economic_units.yaml`),
  confirmar bboxes por oficina de arribo; (c) SHAP condicional por grupo (3.4); (d) migrar a
  skforecast/darts si crecen las series; (e) llevar `pooled_log` a Fase 4 (CQR).
- [x] **Fase 4 — CQR (intervalos calibrados)**: **hecho (2026-07-21)**, `experiments/exp4_cqr/`.
  CQR sobre `pooled_log` (Exp 3.2): cuantiles XGBoost en log + conformal split **Mondrian (por
  serie)**, invertido a kg. Métricas nuevas en `metrics.py` (coverage, pinball, CRPS; con tests).
  **Afinado (2026-07-21)**: calibrar **solo en temporada** (los días fuera fijaban `Q`=0) llevó la
  cobertura de 97.5% a **~86% (nivel 90%) / ~81% (80%)**, cerca de nominal; se añadió la variante
  conformal **normalized** (adaptativa) que mejora un poco a `split`. Condicional MHW 0.891 vs
  0.854 fuera (honesto). **Residual (ver arriba, #1)**: el ancho en días pico lo fija el cuantil
  base, no el conformal. Nota: CQR propia (no `mapie`) porque `mapie` 1.3 con cuantiles prefit +
  `expm1` daba intervalos no anidados.
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

Ya no hay bloqueadores de insumos; todo lo que sigue es código + correr cosas.

1. ✅ Baseline estadístico (ARIMA/Prophet) corrido — fija el piso (ver §4).
2. **Oceanografía** (desbloqueada): correr descarga OISST (§2) → `aggregate ocean` →
   SST/MHW en el dataset → figura MHW (Fase 1.3). El bbox de SQ basta (B3 decidido).
3. **Escribir `extract/copernicus.py` y `extract/globcolour.py`** (+ transforms) con las
   credenciales ya válidas → columnas `x1..x16` → modelos con covariables (LGBM/XGBoost/
   LSTM) de Fase 1.4.
4. Comparar el LSTM del borrador: agregar TensorFlow y (si hace falta) el XGB joblib de B5.
5. Con el dataset enriquecido, Fases 2-5.
