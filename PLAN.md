# PLAN.md

Plan de experimentos para la fase de expansión de la tesis (2026). Ordenado por prioridad según la reunión de planeación. Cada fase tiene criterios de éxito explícitos y un entregable verificable.

**Uso**: marcar cada tarea con `[x]` al completarla. Claude Code debe ejecutar **fase por fase** en orden. No saltar fases. Si una fase revela problemas que bloquean la siguiente, documentarlos y consultar antes de continuar.

---

## Fase 0. Reconocimiento del repo (obligatoria, hacer primero)

**Objetivo**: entender qué hay antes de agregar nada.

**Instrucciones a Claude Code**:

- [ ] Ejecutar `ls -la` en la raíz y `tree -L 3 -I '__pycache__|.git|node_modules|data/raw'` (o equivalente) para mapear la estructura.
- [ ] Leer `README.md`, `pyproject.toml` o `requirements.txt`, `.gitignore`, y cualquier `Makefile` o `justfile`.
- [ ] Identificar dónde viven los modelos actuales (`forecasting_models/`?, `src/`?, `notebooks/`?) y listar cada uno con una línea de descripción.
- [ ] Identificar el formato de los datos procesados actuales: ¿CSV?, ¿Parquet?, ¿dónde están los splits train/val/test?
- [ ] Identificar cómo se cargan los datos de GlobColour y COBI en el pipeline actual.
- [ ] Verificar si hay tracking de experimentos (MLflow, Weights & Biases, o solo notebooks).
- [ ] Verificar si hay tests (`pytest`, `unittest`).
- [ ] Detectar la versión de Python y las librerías principales ya en uso (para no introducir duplicados ni incompatibilidades).

**Entregable**:
- [ ] Un archivo `docs/repo_audit.md` con: estructura, inventario de modelos, formato de datos, gaps detectados, convenciones de estilo inferidas, y una lista priorizada de cosas a arreglar o estandarizar antes de empezar los experimentos.
- [ ] Un comentario en el chat resumiendo hallazgos y proponiendo ajustes al `CLAUDE.md` si la realidad del repo contradice lo que dice ahí.

**Criterio de éxito**: un humano puede leer `repo_audit.md` y entender el proyecto en 10 minutos.

---

## Fase 1. ETL de datos nuevos y re-entrenamiento del baseline

**Objetivo**: reproducir los resultados del borrador con los datos nuevos (2022-2025, otras especies, otras regiones) para tener un baseline actualizado.

### 1.1. Diseño del ETL

- [ ] Revisar el ETL existente del borrador (el paper describe un pipeline FTP → `.nc` → `.csv` → tabla consolidada para GlobColour, y OneDrive → `.csv` para COBI).
- [ ] Listar los archivos crudos nuevos disponibles y dónde están: langosta San Quintín 2022-2025, otras especies (erizo, abulón, etc.), otras regiones/UE. **Preguntar al usuario** la ruta exacta si no es obvio.
- [ ] Diseñar (y documentar en `docs/etl_design.md`) el esquema de la tabla consolidada nueva, con columnas:
  - `ds` (date)
  - `y` (float, kg)
  - `species` (categorical: lobster, urchin, abalone, ...)
  - `economic_unit` (categorical: LITORAL_BC, ISLA_NATIVIDAD, ...)
  - `region` (categorical, si aplica)
  - `x1...x16` (GlobColour), más las nuevas de sensores COBI si existen
  - `season` (int, identificador de temporada: `2017_2018`, `2018_2019`, ...)
  - `in_season` (bool, True si la fecha está dentro de la temporada de esa especie)

### 1.2. Implementación del ETL

- [ ] Implementar/extender módulos en `src/etl/` (o el paquete que corresponda):
  - `extract_globcolour.py`: descarga por FTP con credenciales de `.env`, idempotente, con cache.
  - `extract_cobi.py`: lectura desde OneDrive o ruta local si ya está descargado.
  - `transform.py`: .nc → DataFrame, agregación espacial (promedio por zona TURF correspondiente), agregación temporal (diaria).
  - `consolidate.py`: join de arribos con oceanográficos sobre `ds` + `economic_unit`, con manejo de NaN.
  - `quality_checks.py`: asserts sobre rangos, tipos, duplicados; rechazar datos claramente inválidos.
- [ ] Tests unitarios: un test por módulo, con fixtures chicos (5-10 filas).
- [ ] Integrar con el pipeline existente sin romperlo (el antiguo y el nuevo deben coexistir durante la transición).

### 1.3. Construcción del índice MHW

- [ ] Implementar `src/features/mhw.py`: dada una serie de SST, calcular categoría de MHW por día según Hobday et al. 2016 (baseline climatológico de 30 años, percentil 90, duración mínima de 5 días).
  - Opción: usar la librería `marineHeatWaves` (port de Python) si está disponible; si no, implementación propia basada en el paper.
  - Output: columna `mhw_category` (0=sin, 1=moderado, 2=fuerte, 3=severo, 4=extremo) y `mhw_intensity` (float).
- [ ] Agregar al dataset consolidado.
- [ ] Visualizar: gráfica de SST + MHW sobre el período completo, salvar en `reports/figures/mhw_timeline.png`. Debe mostrar claramente 2014-2016 (Blob) y 2019-2021 (segundo régimen).

### 1.4. Re-entrenamiento del baseline

- [ ] Para cada modelo del borrador original (ARIMA, Prophet, LGBM, XGBoost, LSTM, XGBoost+LSTM) crear un script en `experiments/exp1_baseline_retrain/` que:
  - Carga el dataset expandido.
  - Filtra solo langosta + San Quintín (para ser comparable con el borrador).
  - Reproduce la partición temporal (corte en 01-jun-2021) **y** una partición adicional con corte en 01-jun-2024 (para usar datos nuevos).
  - Ajusta hiperparámetros (Grid Search / Optuna) con el mismo procedimiento.
  - Registra métricas (MAE, RMSE, sMAPE, error de temporada) en `reports/metrics/exp1_*.json`.
  - Guarda el modelo en `models/exp1/`.
  - Produce gráfica predicho vs real en `reports/figures/exp1_*.png`.
- [ ] Producir un notebook `notebooks/exp1_summary.ipynb` (o Markdown) con una tabla comparativa: modelo × partición × métricas. Comparar contra los números del paper original para detectar regresiones.

**Criterio de éxito**:
- [ ] Las métricas reproducidas en la partición antigua están dentro de ±10% de las del paper. Si hay diferencias mayores, documentar por qué (posiblemente cambios en los datos fuente, que vale la pena notar).
- [ ] Existe un baseline sólido con datos 2022-2025 contra el cual comparar los siguientes experimentos.

---

## Fase 2. Feature engineering sistemático con SHAP

**Objetivo**: construir un conjunto enriquecido de features y usar SHAP para identificar los que realmente agregan señal.

### 2.1. Construcción de features

Implementar en `src/features/`:

- [ ] `lags.py`: generador de lags configurables para `y` (1, 3, 7, 14, 30, 90 días) y para cada X (30, 60, 90, 180 días). Usar `pandas.shift` respetando `economic_unit` y `species` (no cruzar series).
- [ ] `rolling_stats.py`: medias y desviaciones rodantes de X e Y con ventanas de 7, 30, 90 días. Usar `min_periods` cuidadoso para no contaminar con valores futuros.
- [ ] `anomalies.py`: anomalías climatológicas. Para cada X_i(t), calcular `X_i(t) - mean(X_i en la misma semana-del-año a través de años anteriores)`. **Importante**: el "año anterior" debe ser solo años en el set de entrenamiento para evitar leakage.
- [ ] `calendar.py`: día del año codificado como (sin, cos), día de la temporada, días desde apertura, semana del año, año.
- [ ] `interactions.py`: producto de pares de variables relevantes ecológicamente: SST × chlorophyll, pH × oxygen, SST × wind (si está), SST × PAR. Documentar la justificación ecológica de cada par.
- [ ] `pipeline.py`: función `build_features(df, config)` que orquesta todo lo anterior y produce el DataFrame listo para modelar. El config es un YAML con los parámetros (ventanas, lags, interacciones).

- [ ] Tests: uno por módulo, verificando que no hay NaN en el output (excepto los esperados en los primeros lag-rows), que las dimensiones son correctas, y **crítico**: que no hay leakage (el feature en t=k no usa información de t>k).

### 2.2. Análisis SHAP

- [ ] Script `experiments/exp2_shap_fe/`:
  - Entrenar XGBoost sobre el conjunto completo de features enriquecidos (tentativamente 80-120 features).
  - Calcular SHAP values con `shap.TreeExplainer` sobre el conjunto de validación.
  - Generar figuras: summary plot (bar + beeswarm), dependence plots para los top 10 features.
  - Guardar en `reports/figures/exp2_shap_*.png`.

### 2.3. Poda y re-entrenamiento

- [ ] Podar features con `mean(|SHAP|) < umbral` (empezar con 1% de la suma total; iterar).
- [ ] Re-entrenar modelo con features podados.
- [ ] Comparar métricas: modelo completo vs podado. El podado no debería perder más de ~2% en métricas y debería ser más interpretable.
- [ ] Repetir el análisis SHAP sobre el modelo podado y producir las figuras definitivas.

### 2.4. Documentación de hallazgos

- [ ] Notebook `notebooks/exp2_summary.ipynb` con:
  - Tabla: feature, tipo (lag / rolling / anomalía / etc.), SHAP importance, retenido sí/no.
  - Discusión de los top 10 features: ¿qué dice la ecología al respecto? ¿tiene sentido que SST_lag_90 sea top? (sí, por el desplazamiento de 3 meses).
  - Hallazgos inesperados si los hay.

**Criterio de éxito**:
- [ ] El modelo con features enriquecidos supera al baseline (Fase 1) en al menos una métrica en la partición nueva (corte 01-jun-2024), idealmente en todas.
- [ ] Existe una narrativa clara de por qué cada feature retenido importa, apoyada en SHAP + literatura.

---

## Fase 3. Modelo jerárquico / global multi-especie y multi-región

**Objetivo**: entrenar un solo modelo global sobre todas las especies y unidades económicas disponibles, aprovechando transferencia de información entre series.

### 3.1. Diseño

- [ ] Documentar en `docs/hierarchical_design.md`:
  - Qué especies y UEs se incluyen (criterio de inclusión: mínimo N observaciones, disponibilidad de X).
  - Cómo se codifican especie y UE: target encoding, one-hot, o embeddings (si se usa una red).
  - Cómo se hace la partición temporal respetando la estructura jerárquica (no especie-por-especie).
  - Qué métricas se reportan: agregadas y por especie × UE.

### 3.2. Implementación

- [ ] Decidir framework: **skforecast** (`ForecasterRecursiveMultiSeries` o `ForecasterDirectMultiVariate`) o **darts** (`RegressionModel` con `series=list_of_series`). Elegir según lo que ya esté en el repo o sea más ligero. Documentar la decisión.
- [ ] Script `experiments/exp3_global_model/`:
  - Cargar dataset completo con todas las especies y UEs.
  - Aplicar el pipeline de features de Fase 2 respetando agrupamientos.
  - Entrenar un modelo global (XGBoost o LightGBM como base, por velocidad).
  - Backtesting con ventana expansiva.
  - Métricas por serie (especie × UE) y agregadas.

### 3.3. Comparación contra modelos específicos

- [ ] Para cada serie (especie × UE), entrenar también un modelo específico (XGBoost entrenado solo con esa serie).
- [ ] Tabla comparativa: modelo específico vs modelo global, por serie, en las métricas clave.
- [ ] Identificar en qué series el modelo global mejora, en cuáles empeora, y por qué (relacionar con la cantidad de datos de cada serie).

### 3.4. Análisis de transferencia

- [ ] Feature importance por grupo (especie, UE): SHAP condicional para ver si el modelo global aprendió a tratar cada especie diferente o si aplica la misma lógica a todas.
- [ ] Documentar el hallazgo central: ¿las dinámicas son compartidas entre especies/regiones o son idiosincráticas?

**Criterio de éxito**:
- [ ] El modelo global empata o supera al específico en al menos 60% de las series, especialmente en las más cortas (donde la transferencia debe ayudar más).
- [ ] Hay una respuesta defendible a la pregunta: "¿vale la pena un modelo global?".

---

## Fase 4. Pronóstico probabilístico con Conformalized Quantile Regression

**Objetivo**: agregar intervalos de predicción calibrados al mejor modelo de las fases anteriores.

### 4.1. Implementación

- [ ] Script `experiments/exp4_cqr/`:
  - Tomar el mejor modelo de Fase 2 o Fase 3 como base.
  - Instalar `mapie` si no está (`pip install mapie`).
  - Aplicar `MapieQuantileRegressor` con α=0.1 (90% de cobertura nominal).
  - Partir el set de entrenamiento en train propio (70%) y calibración (30%), respetando orden temporal.
  - Entrenar quantile regressors para τ=0.05 y τ=0.95.
  - Conformalizar usando el set de calibración.

### 4.2. Evaluación

- [ ] Métricas sobre test:
  - **Coverage empírico** (debe estar cerca de 90%; desviaciones >5pp indican mala calibración).
  - **Ancho promedio del intervalo** (más angosto es mejor, a igual cobertura).
  - **CRPS** (Continuous Ranked Probability Score).
  - **Calibración condicional**: ¿el coverage se mantiene en los días dentro de temporada? ¿durante MHW? Si falla en MHW, reportarlo — es un hallazgo importante.
- [ ] Figura: predicción puntual + banda de 90% vs valores reales, coloreando diferente si el real cae dentro o fuera del intervalo.

### 4.3. Producto operativo

- [ ] Notebook o script que genere un "reporte de temporada próxima" con:
  - Predicción puntual del volumen total esperado.
  - Intervalo de 90% del volumen total.
  - Predicción diaria + bandas durante toda la temporada.
  - Indicador de confianza (ancho relativo del intervalo).
- [ ] Pensar la presentación pensando en COBI / pescadores (no técnico, visual).

**Criterio de éxito**:
- [ ] Coverage empírico dentro de 90±3 puntos porcentuales en test.
- [ ] El ancho del intervalo es informativo (no tan ancho que sea inútil, no tan angosto que subestime la incertidumbre).
- [ ] Hay un entregable visual listo para compartir con COBI.

---

## Fase 5 (opcional). Temporal Fusion Transformer

**Objetivo**: explorar si una arquitectura moderna de deep learning supera al ensamble. Solo hacer si hay tiempo.

- [ ] Justificar en un ADR corto (`docs/decisions/ADR-tft.md`) por qué TFT y no otro Transformer. Puntos a cubrir: acepta covariables estáticas, conocidas futuras y observadas pasadas; produce cuantiles; tiene interpretabilidad vía variable selection networks.
- [ ] Implementación con `pytorch-forecasting` (abstracción más simple) o `darts` (`TFTModel`).
- [ ] Dataset debe estar en formato multi-series con group IDs (especie × UE).
- [ ] Entrenamiento con early stopping sobre el set de validación temporal.
- [ ] Reportar métricas y compararlas con XGBoost+LSTM ensamble + CQR.
- [ ] Discusión honesta: si TFT no gana, eso es un hallazgo metodológico (con estos datos, la complejidad extra no paga).

**Criterio de éxito**:
- [ ] Reporte claro de si TFT vale la pena con los datos actuales. Si no gana, documentar cuánta más data haría falta (regla de dedo: Transformers empiezan a pagar con >10k observaciones por grupo).

---

## Entregables finales de esta fase del proyecto

Al completar todas las fases, debe existir:

- [ ] Código modular y testeado en `src/`.
- [ ] Cinco experimentos reproducibles en `experiments/exp1..exp5/`.
- [ ] Dataset consolidado con todas las especies y UEs disponibles.
- [ ] Resultados en `reports/metrics/` y figuras en `reports/figures/`.
- [ ] Mejor modelo serializado en `models/final/` con un README que diga cómo cargarlo y usarlo.
- [ ] Un notebook `notebooks/paper_figures.ipynb` que genere **todas** las figuras del paper actualizado.
- [ ] Sección nueva de la tesis redactada (Markdown en `docs/thesis_sections/` que luego se traduce a LaTeX): revisión bibliográfica expandida, metodología actualizada, resultados, discusión incluyendo análisis MHW.

---

## Reglas de oro para Claude Code durante la ejecución

1. **Una fase a la vez**. No mezclar código de distintas fases en el mismo branch.
2. **Commit frecuente**. Cada sub-sección (1.1, 1.2, etc.) debería cerrar con al menos un commit.
3. **Si una fase tarda más de lo esperado**, parar y comentar con el usuario antes de seguir o refactorizar.
4. **No entrenar nada largo sin confirmar**. Antes de correr un grid search de 500 trials o un LSTM de 100 epochs, avisar y confirmar.
5. **Al final de cada fase**, escribir un resumen en `reports/sessions/YYYY-MM-DD_fase_N.md` con qué se hizo, qué salió bien, qué se rompió, qué sigue.
6. **No borrar trabajo previo**. Si algo del borrador original ya no se usa, moverlo a `legacy/` antes de eliminarlo.
