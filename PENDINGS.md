# PENDINGS.md

**Lista única de lo que falta.** Es la fuente de verdad del proyecto: `PLAN.md` es el diseño
original de las fases (congelado, ya no se marca) y `bitacora.md` es el historial de lo hecho
con sus números. Si algo no está en este archivo, no está pendiente.

Última actualización: **2026-09-01** (ronda final de revisión del documento).

---

## Estado: qué ya está cerrado

- **ETL completo y probado**: extract → transform → aggregate → consolidate → quality_checks,
  con arribos unidos COBI (≤2021) + CONAPESCA (≥2022) y oceanografía de Copernicus
  (SST/OSTIA + 6 variables de color del océano). `dataset_v1` cubre **2017-2026, 44 series**
  (21 de langosta), SST hasta 2026-03-31.
- **Los cinco experimentos del plan**: baseline ARIMA/Prophet (Exp 1), modelos ML/DL del
  borrador incluido el ensamble XGBoost+LSTM (Exp 1b), covariables + SHAP (Exp 2/2.3), modelo
  global con `log1p` (Exp 3/3.2), CQR (Exp 4) y TFT como prueba de techo (Exp 5). Todos
  reproducibles con un comando y parametrizados por `FF_CUT_DATE`.
- **Producto operativo**: API FastAPI + front + Docker, sirviendo 28 series.
- **Documento de tesis**: reestructurado a formato de tesis (`report`, capítulos, front matter),
  96 págs, 0 referencias sin resolver, 0 labels duplicados, **0 overfull en el log** tras la
  ronda final de revisión (2026-09-01): tabla de series reencajada, 10 citas `\cite`→`\citep/\citet`
  corregidas, 3 figuras de terceros sustituidas por figuras propias, tablas heredadas de 2023
  colapsadas y anotadas, y notas de instantánea en las tablas de Exp 3.
- **157 tests verdes.** Sin bloqueadores de insumos externos: todo lo que sigue es decisión,
  redacción o mejora opcional.

---

## A. Para poder presentar la tesis (bloqueante)

Solo queda **A1**. A2, A3 y A4 ya se cerraron (2026-08-31).

| # | Qué falta | Quién | Detalle |
|---|---|---|---|
| **A1** | Datos de la portada y del front matter | **Javier** | Nombre y grado del **asesor**; confirmar con el asesor y Dirección Escolar el **formato oficial vigente** (y si se requiere hoja de firmas); verificar el **texto vigente de la declaración de derechos** (arts. 21 y 27 LFDA) con la Biblioteca Raúl Bailléres Jr.; escribir los **agradecimientos**. El checklist completo de los cuatro campos está al inicio de `final_work/front/portada.tex`; el asesor se llena en un solo lugar (`\asesor{...}`). |
| ~~A2~~ | ~~Decidir el título~~ | — | **DECIDIDO (2026-08-31)**: se mantiene el título nuevo, *"Pronóstico calibrado del volumen de captura de langosta roja en Baja California: olas de calor marinas, modelos globales multi-especie e intervalos conformalizados"*. Queda como parte de A1 tramitar el cambio ante Dirección Escolar si el título registrado es el anterior. |
| ~~A3~~ | ~~Bibliografía~~ | — | **HECHO (2026-08-31)**: migrada a BibTeX (`final_work/referencias.bib`) y ampliada de 28 a **51 referencias**, todas citadas en el texto y todas resueltas. Añadidas: la pesquería y su marco regulatorio (NOM-006-SAG/PESC-2016 DOF 07-09-2016, Carta Nacional Pesquera DOF 10-03-2025, biología de *P. interruptus* de Vega Velázquez 2003, y el antecedente de pronóstico de precio de langosta mexicana de Hernández-Casas et al. 2022), la literatura de olas de calor marinas (Hobday 2018, Oliver 2018, Frölicher 2018, Di Lorenzo y Mantua 2016, Cavole 2016, Free 2019, Smith 2021), los productos oceanográficos (OSTIA: Donlon 2012, Good 2020; datos abiertos de CONAPESCA), metodología de pronóstico (Hyndman y Koehler 2006, Bergmeir y Benítez 2012, Tashman 2000, Box y Jenkins 2015, Hochreiter y Schmidhuber 1997) y el software usado. Fechas del DOF y datos de publicación verificados contra la fuente. |
| ~~A4~~ | ~~Lectura de consistencia~~ | — | **HECHO (2026-08-31)**: ver el detalle de los tres hallazgos en `bitacora.md` (2026-08-31 d). Quedaron 0 referencias rotas, 0 labels duplicados, 0 números de tabla/figura escritos a mano, y todas las cifras del texto contrastadas contra `reports/metrics/*.json`. |

## B. Mejoras opcionales de calidad (no bloquean)

Cerradas el 2026-08-31 (e): **B2, B4, B5, B7 y B9**. Sigue abierto lo de abajo.

| # | Qué | Estado | Detalle |
|---|---|---|---|
| **B1** | Guionado en español del PDF | **Bloqueado (decisión de Javier)** | `tlmgr` no puede instalar `babel-spanish`/`setspace`: el TinyTeX local es TeX Live **2024** y el repo remoto es **2026** ("cross release updates are only supported with update-tlmgr-latest"); el repo histórico de 2024 responde "TeX Live 2024 is frozen… tlmgr itself needs to be updated… Terminating". Desbloquearlo exige correr `update-tlmgr-latest.sh` o reinstalar TinyTeX —modifica la instalación de TeX del usuario, así que no se hizo sin permiso—. El preámbulo redefine los nombres a mano y el PDF compila bien; el único costo es el guionado. |
| ~~B2~~ | ~~Variante fiel `lstm_orig2023` (2700/800 unidades)~~ | **HECHO** | Re-entrenada la arquitectura del borrador (40.8 M de parámetros) sobre las mismas 2 295 ventanas. Con el corte 2024 **colapsa a una constante**: predice 0 todos los días, lo que le da el *mejor* MAE de la tabla (79.8 = la media observada) con dispersión y correlación 0 y error de temporada −100 %; juzgada solo por MAE habría ganado. Con el corte 2020 no colapsa pero es peor que la red pequeña (MAE 435.3 vs 290.7, dispersión 2.99 vs 1.58). Es el caso de manual de por qué las métricas de forma son obligatorias. |
| **B3** | Endurecer el modelo global `pooled_log` | Pendiente | Optuna sobre el pool en log; investigar por qué langosta@Isla Cedros (la de mayor escala) empeora con log; probar pesos por serie u objetivo por grupo. |
| ~~B4~~ | ~~Feature engineering residual (Fase 2)~~ | **HECHO** | `configs/features.yaml` + `fit_climatology` / `add_climatology_anomalies` / `add_interactions` / `build_features_v2` en `features/covariates.py`, con 7 tests anti-fuga. Ablación `experiments/exp2_covariates/features_v2_ablation.py`: **no mejoran** (corte 2024: gana en 10/28 series, RMSE medio 633.4→633.7; corte 2020: 25/33, 435.0→434.1). Resultado negativo consistente con Exp 2: con pocas temporadas, más features no ayudan. |
| ~~B5~~ | ~~SHAP condicional por grupo~~ | **HECHO** | `experiments/exp2_shap_selection/shap_by_group.py` (+6 tests). La identidad (one-hot especie/UE) se lleva **38.4 %** de la atribución en el corte 2020 y **33.5 %** en el 2024: el pool no es un conjunto de submodelos, pero tampoco ignora la identidad, y su peso **baja al crecer los datos**. Divergencia Jensen-Shannon entre especies ≈0.28 y entre UEs de langosta ≈0.22 (máx 0.47 entre La Purísima y Litoral de BC Sur). |
| **B6** | Conformal adaptativo por régimen (MHW) | Pendiente | Calibrar por separado días con y sin ola de calor. Con 1-2 temporadas de calibración por serie aún no es robusto. |
| ~~B7~~ | ~~Serializar el modelo de producción~~ | **HECHO** | `fishing-etl serve-build` → `models/final/store.json` (0.8 MB, 28 series, corte 2024-06-01) con manifiesto (versión de formato, semilla, huella del parquet) y `models/final/README.md`. La API carga el artefacto si existe (`get_store`) y solo entrena si falta, avisando en el log. 4 tests. |
| **B8** | **Más temporadas** | Pendiente (externo) | **El mayor retorno de todos**: es la conclusión de la tesis. Conforme CONAPESCA publique, re-correr el ETL y los experimentos (un comando cada uno). No requiere trabajo de modelo. |
| ~~B9~~ | ~~Re-correr Exp 5 (TFT) sobre el pool final~~ | **HECHO** | Re-corrido sobre el mismo pool y los mismos cortes que la CQR de producción, así que **ya es comparable** y la advertencia de comparabilidad salió de la tesis. Corte 2020: CRPS 116.2 (TFT) vs 132.8 (XGBoost+CQR), cobertura 70.8 % vs 96.7 %. Corte 2024: CRPS 210.9 vs 216.8, cobertura 79.1 % vs 95.8 %. Lectura: el TFT es algo más nítido y claramente peor calibrado —sub-cubre en ambos cortes—, y su diagnóstico por serie cambia de extremo a extremo al re-entrenar sobre otro pool (San Quintín pasó de 40.6/62.0 % a 99.6/95.8 %). Sigue reventando la serie de erizo de El Regasa por desborde al invertir el logaritmo. |

## C. Calidad de datos (chico, real, sin urgencia)

- [ ] **117 filas con captura fuera de la ventana reglamentaria** (`y>0` con `in_season=False`);
  generan un warning de QC. Son avisos de CONAPESCA con fecha fuera de 15-sep–15-feb: decidir si
  se recortan, se reclasifican a la temporada contigua o se aceptan como arribos tardíos
  legítimos. Hoy se aceptan sin decisión explícita.
- [ ] **Centinela de faltantes de REDMAR/CICESE** sin confirmar (¿9999?, ¿-99999?). `read_dat`
  acepta `na_values` explícito pero no está fijado; sin confirmarlo, las medianas diarias podrían
  sesgarse si algún día se usa esa fuente.
- [ ] **`quality_checks.check_sst_correlation` no está conectada** al flujo real (necesita
  `interim/cicese/*` y la serie de SST por región). Está implementada y testeada.
- [ ] **Formas de producto excluidas**: el mapping solo toma “entero” (ENT. FCA./FCO.). Colas,
  carne, cocida y sin concha se descartan para no mezclar bases de peso. Si COBI necesita el peso
  total por especie hay que sumarlas con factores de conversión (decisión de dominio, no de código).
- [ ] **Polígonos TURF**: el de Isla Natividad ya está descargado en
  `data/raw/turf/reserveeffect/`. Usarlo requiere soporte de polígono (no solo bbox) en
  `aggregate/ocean_by_ue.py` + `geopandas`/`shapely`. Nice-to-have; el bbox basta a la resolución
  de los productos satelitales.
- [ ] **`InsecureRequestWarning`** del SDK `copernicusmarine` (su backend S3 no verifica TLS). Es
  interno del SDK, no del código propio; revisar si una versión nueva lo corrige.

---

## D. Higiene y seguridad

- [ ] **Credenciales de Postgres en claro** en `legacy/etl/load/globcolour_load.py` y
  `legacy/etl/load/cicese_load.py`, **versionadas en git**. Acción: borrar esos *loaders* si no se
  van a reusar (lo más probable) o parametrizarlos por `.env`; y **rotar la contraseña** si ese
  servidor sigue vivo, porque el historial de git ya la contiene y borrarla del árbol no la
  elimina de la historia.
- [ ] **Rutas de Windows/Colab** en los scripts de modelado heredados (`legacy/forecasting_models/`):
  decidir si se archivan como están o se limpian. Ya están fuera del camino de ejecución.
- [ ] **Deuda de lint preexistente**: 7 errores de `ruff` en `src/` y `tests/` (ninguno en el
  código nuevo). `uv run ruff check .` los lista.
- [ ] **Artefactos de LaTeX versionados** (`.aux`, `.log`, `.out`, `.toc`, `.lof`, `.lot`, `.pdf`):
  decidir si se ignoran en `.gitignore` o se conservan a propósito para tener el PDF en el repo.
- [ ] **Commits automáticos del 2026-08-31**: tres commits (`6b1632f`, `556b339`, `410a14d`)
  entraron a `main` sin haberse pedido, aparentemente por un hook. Decidir si se dejan, se
  reescriben con mensajes propios o se reorganizan.

---

## E. Decisiones tomadas: cosas que ya NO se van a hacer

Se listan para que no vuelvan a aparecer como pendientes.

- **NOAA OISST** — superado: Copernicus OSTIA cubre SST 1982-2026 y ya alimenta el índice MHW.
- **Zips `Tesis-*.zip` en S3 (~18 GB)** con el joblib de XGBoost y las métricas del borrador —
  ya no hacen falta: Exp 1b re-entrenó todos esos modelos sobre los datos unidos.
- **Comparación contra el `.h5` del LSTM del borrador con TensorFlow** — superada por la misma
  razón; el `.h5` sigue en `models/legacy/` por si acaso.
- **Polígono TURF de San Quintín** — no es de acceso público y COBI ya no puede proveerlo; el
  bbox costero es la aproximación aceptada y así se declara en la tesis.
- **MLflow** — superado: cada experimento escribe su JSON de métricas, sus figuras y su resumen
  en Markdown, versionados por corte.
- **`notebooks/paper_figures.ipynb`** — superado: cada experimento genera sus propias figuras.
- **`docs/thesis_sections/*.md`** — superado: la tesis se escribe directo en LaTeX en
  `final_work/sections/`.
- **Estrategia de dedup COBI/CONAPESCA** — resuelta: unión por prioridad temporal
  (`transform arribos --source union`), sin doble conteo en la costura de 2021.
- **Hueco de langosta 2022+ en San Quintín** — resuelto: eran datos que COBI no tenía; los avisos
  de cosecha de CONAPESCA los aportan.
- **Cobertura reciente de SST (REP vs NRT)** — resuelta: el producto reprocesado llega a
  2026-03-31, suficiente para el horizonte servido.
