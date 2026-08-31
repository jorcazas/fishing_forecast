# PENDINGS.md

Mapa de lo que **falta para terminar el plan** (`PLAN.md`), separando lo que está
bloqueado por insumos externos de lo que es trabajo de código ya desbloqueado. Última
actualización: **2026-08-31**.

## Qué falta ahora (snapshot 2026-07-21, ordenado por prioridad)

Fases 1-3 cerradas con datos reales; ya no hay bloqueadores de insumos. Lo que queda:

0. **Despliegue — front de inferencia (HECHO 2026-07-23)**: mini-app FastAPI + front vanilla/SVG
   (`src/fishing_forecast/serving/`, `frontend/`) que sirve el pronóstico calibrado CQR por especie ×
   UE, dockerizada (`Dockerfile`, `docker-compose.yml`, `docs/serving.md`). `docker compose up --build`
   → http://localhost:8000 (ahora sirve 21 series de langosta + abulón/erizo = 28). El total por
   temporada NO es fiable como punto (captura concentrada en pocos días); el producto es el intervalo
   diario + cobertura en temporada. **Deuda del calendario de temporada: RESUELTA (2026-07-23)** —
   `season_calendars.yaml` ahora declara la ventana 15-sep–15-feb para **todas** las UEs de langosta
   (antes solo `litoral_bc_sur`); `in_season` del dataset ya es fiable. Esto re-derivó Exp 4 (ver #4
   y tesis §6.9); el workaround `_in_lobster_season` se conserva como ventana autocontenida pero
   coincide con el flag del dataset.

1. **Más temporadas de calibración (Fase 4 — el techo real)**: intenté endurecer los cuantílicos
   base con Optuna sobre pinball (Exp 4b, 2026-07-21) para estrechar el ancho de días pico →
   **NO funciona** (empeora langosta: p90 36k→402k kg, sobreajusta validación; ver bitácora).
   Además, la cobertura **por serie** del Exp 4 es heterogénea: **langosta@SQ sub-cubre al 53%**
   (el crash post-MHW cae bajo la cota inferior calibrada con una sola temporada previa), el 86%
   marginal promedia con el abulón sobre-cubriendo. **Ambos problemas son el mismo cuello: pocas
   temporadas de calibración.** No hay palanca de código/tuning; la mejora real es **más datos**
   (ver #4). Opcional de bajo retorno: conformal adaptativo por régimen (MHW), pero con 1 temporada
   de calibración por serie es poco robusto.
2. **Fase 5 (opcional) — TFT**: **CERRADA (2026-07-23)**. Corrido en ambos cortes con comparación
   justa (`experiments/exp5_tft/tft_cqr.py`: cuantiles del TFT en la misma CQR de Exp 4, con
   reordenamiento monótono de cuantiles). Veredicto: TFT+conformal en **paridad marginal** (CRPS a
   la par/ligeramente mejor; calibrado 91.2% @2024) pero **inestable por serie** y sin ganancia con
   más épocas → NO supera al XGBoost+CQR, como anticipaba el ADR. Escrita la §"Prueba de techo" en
   la tesis (`sec:extension_tft`) y reescrito `reports/exp5_tft_cqr_summary.md` con números
   corregidos. Reproducir: `uv sync --extra dl --extra models`, luego
   `FF_CUT_DATE=<corte> uv run python -m experiments.exp5_tft.tft_cqr`.
3. **Endurecer `pooled_log`** (Fase 3 residual): Optuna sobre el pool log; investigar por qué
   langosta@Cedros (mayor escala) empeora con log; probar pesos por serie u objetivo por-grupo.
4. **Más datos** (mayor palanca real) — **GRAN AVANCE (2026-07-21)**:
   - **Clúster El Rosario**: 5 cooperativas de langosta añadidas → langosta de 2 a **7 series**.
     Impacto: langosta@SQ RMSE 659→313 (pooled), CQR marginal 90% 86%→93%, p90 9776→3812 kg.
   - **Temporadas 2022-2026 DESBLOQUEADAS**: los AVISOS COSECHA de CONAPESCA (Downloads) traen
     langosta de nuestras UEs 2022-2026. `transform arribos --source union` (COBI ≤2021 +
     CONAPESCA ≥2022, sin doble conteo) → **`dataset_v1` 2017-2026; langosta@SQ ~9 temporadas**.
     Impacto: en corte **2024-06-01** (crash en train) **langosta@SQ cobertura CQR 41%→95.8%** —
     arregla la sub-cobertura que ningún tuning logró. CRPS 188→158.
   **Pendiente**: (a) ~~reflejar en la tesis las temporadas nuevas, las 7 series y el corte
   2024-06-01 con el salto de cobertura~~ **HECHO (2026-07-21)**: `final_work.tex` §Datos-y-ETL
   actualizado (union COBI+CONAPESCA, 2017-2026); nueva subsección §La-hipótesis-de-datos con
   Tabla `extension_moredata` (langosta@SQ 40.6%→95.8% al corte 2024, marginal 94→95%, CRPS
   158→220); párrafo de cierre en Conclusión. Compila a 27 págs, refs resueltas; (b) ~~confirmar
   consistencia del empalme 2021 COBI↔CONAPESCA~~ **HECHO (2026-07-21) — LIMPIO**: sin fechas
   duplicadas; COBI termina 2021-12-31 (truncaba la temporada 21-22 a ~31 t), la unión recupera
   ene-feb 2022 (→ 39.5 t completa); transición continua (dic-2021 COBI 6.3 t ≈ ene-2022 CONAPESCA
   6.5 t, escalas compatibles). Hallazgo extra: la captura NO se recuperó (24→18→17 t en 2022-2025).
   Corregidas en la tesis las cifras (31→39.5 t, 82→77%) y "recuperación"→"captura reducida".
   Caveat: no hay periodo de solape COBI/CONAPESCA para validar escala de forma exhaustiva; el
   traspaso dic→ene es la evidencia disponible y es consistente;
   (c) ~~clústers Ensenada/Vizcaíno — aún más UEs~~ **PARCIAL (2026-07-21)**: añadido el clúster
   **costa-norte con historia completa 2017-2026** (Punta Canoas ~29.4°N + El Pabellón de SQ + Rocas
   de San Martín ~30.4°N) → langosta **7→10 series**. Mejora el corte 2024 (marginal 95→96.7%, CRPS
   220→131, ancho langosta@SQ 471→341). HALLAZGO: al re-correr, la cobertura langosta@SQ en el corte
   2020 saltó 40.6%→99.8% (por composición del pool) → el 40.6% era frágil; el corte 2020 es
   **inestable** (OOD), el 2024 estable (~96%). Tesis §6.9 reformulada en torno a la ESTABILIDAD.
   **Vizcaíno Tier A HECHO (2026-07-23)**: 4 cooperativas FEDECOOP de Punta Eugenia (~27.1-27.9°N,
   dentro del bbox SST actual → cero descargas) añadidas → langosta **10→14 series**:
   vizcaino_tortugas (767t), vizcaino_emancipacion (616t), vizcaino_asuncion (494t), vizcaino_natividad
   (362t). Resultado (Exp 4 CQR): **langosta@SQ ESTABLE** (99.8%/96.2% sin cambio en ambos cortes —
   ya no es frágil como con la adición costa-norte); marginal→nominal (2024 96.7→94.0%). CRPS marginal
   sube (131→246 @2024) por composición de ESCALA (series de gran tonelaje), NO regresión de calidad;
   nuevas series cubren 91-93% (2020) / 81-86% (2024, solo ~2 temporadas de train). Reflejado en tesis
   (§6.9 tabla + párrafo robustez + figura 14 paneles).
   **Vizcaíno Tier B HECHO (2026-07-23)**: descarga Copernicus extendida (lat_min 25.8, lon_max -112.0,
   ~2 GB re-descargados) + 4 cooperativas del Pacífico BCS (la_purisima 533t, abreojos_progreso 366t,
   abreojos_punta 260t, abreojos_san_ignacio 152t) → langosta **14→18 series**. Resultado: langosta@SQ
   INVARIANTE otra vez (99.8%/96.2%) Y su **nitidez mejora** (CRPS 2024 87→76, 2020 505→144 —
   transferencia positiva dentro de especie explícita); marginal cerca del nominal; CRPS marginal sube
   por escala. Series Tier B mejor calibradas (87-94%) que las de Vizcaíno.
   **Vizcaíno Tier C HECHO (2026-07-23)**: descarga Copernicus re-extendida al sur (lat_min 24.0,
   lon_max -111.5, ~2.4 GB) + 3 cooperativas de Bahía Magdalena (magdalena_chale 34t, magdalena_bahia
   25t, magdalena_san_carlos 19t; costa externa de Isla Magdalena/Santa Margarita ~24.1-25.0°N) →
   langosta **18→21 series**. **En la misma tanda se corrigió el calendario de temporada** (todas las
   UEs de langosta, antes solo la insignia) → se re-corrió Exp 4 en LOS 5 TRAMOS (7/10/14/18/21) ×
   2 cortes con calibración consistente (nuevo filtro `FF_EXCLUDE_LOBSTER_UES` en Exp 4). **Hallazgos
   con calibración corregida**: (a) langosta@SQ **estable a 96.2% (corte 2024) en 7→21 series** — la
   invariancia se sostiene; (b) el corte 2020 es **inestable** (se mantuvo en ~41% hasta 18 series y
   saltó a 99.9% con 21 — confirma OOD, corte poco fiable); (c) **la afirmación previa de "nitidez que
   mejora al sumar zonas" NO se sostiene**: el CRPS insignia rebota (124→95→131→90→123) y Bahía
   Magdalena (borde cálido) EMPEORA la nitidez de SQ (90→123). La transferencia positiva dentro de la
   especie opera entre regímenes parecidos, no desde el extremo cálido del rango. Tesis §6.9 reescrita
   a la progresión **7→21** (tabla + 5 párrafos + figura 21 paneles + conclusión), atenuando la
   afirmación de nitidez. Ensenada (~31.8°N) sin langosta en los datos; (d) ~~parametrizar
   el corte también en Exp 1-3~~ **HECHO (2026-07-21)**: Exp 1/2/2.3 aceptan `FF_CUT_DATE` (como Exp 4).
   Comparativa refrescada de modelos puntuales langosta@SQ sobre datos unidos (ambos cortes) en
   `final_work.tex` §6.10 (Tabla `extension_comparativa`): de ~3 a ~7 temporadas de train el MAE cae
   a la mitad (ARIMA 331→178, XGBoost 459→145); el orden se invierte (XGBoost supera a ARIMA con 7
   temporadas); la poda SHAP cambia de signo (empeoraba pre-unión 424→445, ahora ayuda 145→104).
4b. **Modelos ML/DL del borrador en la comparativa (HECHO 2026-08-31)**: `experiments/exp1_baseline_retrain/legacy_ml.py`
   (Exp 1b) añade **LGBM, LSTM y el ensamble XGBoost→LSTM** a la Tabla `extension_comparativa`
   (§6.10), que solo tenía ARIMA/Prophet/XGBoost — el ensamble era el mejor modelo del borrador,
   así que su ausencia dejaba la conclusión de 2023 sin verificar. Mismos datos, features,
   partición y semilla que Exp 2 (la fila `xgboost` reproduce 459.0/145.4 exactos). **Veredicto:
   el ensamble NO sobrevive al refresco.** Corte 2020 (~3 temporadas): encabeza (MAE 236.4 vs
   459.0 del XGBoost) pero por **amortiguar** la serie, no por seguir su forma — razón
   sd(pred)/sd(obs) 1.58 vs 2.37 con la misma correlación (0.61-0.67 los cuatro). Corte 2024
   (~7 temporadas): se derrumba a MAE 499.4 (LSTM) / 561.7 (ensamble) vs 103.5 del XGBoost+SHAP,
   ahora **sobre-reaccionando** (razón 3.26/4.06); error de suma de la temporada 2024-2025 de
   +626%/+704% vs +174%. El 12.9% del borrador era artefacto de una partición corta. Tesis §6.10
   ampliada (2 filas nuevas + 4ª lectura + figura `fig:extension_comparativa_legacy`) y párrafo
   correctivo en la Conclusión; compila a 35 págs sin warnings. Reproducir:
   `FF_CUT_DATE=<corte> uv run python -m experiments.exp1_baseline_retrain.legacy_ml`.
   **Pendiente opcional (bajo retorno)**: la variante fiel `lstm_orig2023` (2700/800 unidades,
   ~41 M de parámetros para ~10³ observaciones). No corrida por costo; el resultado esperado es
   que empeore respecto a `lstm` (128/64) y el contraste sería un dato más del argumento de
   volumen de datos. Comando:
   `FF_LSTM_ARCHS=lstm,lstm_orig2023 FF_CUT_DATE=<corte> uv run python -m experiments.exp1_baseline_retrain.legacy_ml`.
   Nota técnica: en macOS hay que dejar torch en un solo hilo (`torch.set_num_threads(1)`, ya en
   el script) o el proceso se cuelga en la barrera de OpenMP por el `libomp` que ya cargaron
   xgboost/lightgbm.

5. **Feature engineering residual (Fase 2)**: `anomalies`, `interactions`, `rolling` configurable.
6. ~~**Figura MHW** `reports/figures/mhw_timeline.png`~~ **HECHO (2026-07-21)**: generada
   (SST + climatología + umbral p90 + eventos MHW sombreados; el "Blob" 2014-2016 y el régimen
   cálido 2019-2021 se ven con nitidez) e incluida en la tesis (`final_work.tex` §Datos,
   `\ref{fig:mhw_timeline}`).
7. **Higiene heredada** del borrador (§6): credenciales Postgres hardcodeadas, rutas Windows/Colab,
   `code_wandb.py` ajeno — mover a `legacy/` (pendiente). ~~Labels placeholder duplicados
   (`fig:enter-label` ×11, `tab:catch_comparison` ×4)~~ **HECHO (2026-07-23)**: renombrados por
   unicidad (`fig:splits`, `fig:etl_globcolour`, `fig:zona_pesca`, `fig:pred_*`, `tab:metricas_*`,
   etc.); ninguno estaba `\ref`-eado. La tesis recompila a 33 págs con **0 warnings de
   "multiply defined"**. El mensaje natbib author-year sigue siendo no fatal (solo detiene con
   `-halt-on-error`; el build en `nonstopmode` produce el PDF).

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
- [x] **Figura MHW** `reports/figures/mhw_timeline.png` **HECHA (2026-07-21)**: `add_mhw(...,
  return_diagnostics=True)` sobre la SST de SQ (1982-2025) → `viz/mhw_plot.plot_mhw_timeline`.
  Muestra el "Blob" 2014-2016 y el régimen cálido 2019-2021. Incluida en `final_work.tex`
  (§Datos, `\ref{fig:mhw_timeline}`).

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
- [x] **Fase 5 (opcional) — TFT**: **CERRADA (2026-07-23)** — corrida en dos cortes, comparación
  justa (TFT+conformal), veredicto de paridad marginal / no supera al XGBoost+CQR, y sección
  escrita en la tesis (ver §2 arriba). ADR `docs/decisions/ADR-0002-tft.md`.

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
