# CLAUDE.md

Instrucciones permanentes para trabajar en este proyecto. Léelas antes de cualquier tarea y respétalas a lo largo de la conversación.

## Contexto del proyecto

Este repositorio contiene el código de la tesis de licenciatura de Javier Orcazas (ITAM, Ciencia de Datos): **predicción del volumen de pesca de langosta (Panulirus interruptus) en San Quintín, Baja California**, usando variables oceanográficas (GlobColour + sensores COBI) y datos de arribos pesqueros.

El proyecto colabora con **COBI** (Comunidad y Biodiversidad, ONG mexicana) y busca dar una herramienta operativa a cooperativas pesqueras para anticipar el volumen de captura de la siguiente temporada y optimizar insumos.

### Estado inicial (borrador 2023)

- Modelos entrenados: ARIMA, Prophet, LGBM, XGBoost, LSTM, y un ensamble **XGBoost+LSTM** (mejor modelo, error en suma de temporada ~8.7% para 2020-2021, ~12.9% agregado).
- Datos: GlobColour (16 variables oceanográficas) + arribos pesqueros COBI (volumen diario). Desplazamiento de 3 meses de las X para predecir 3 meses al futuro.
- Zona: LITORAL DE BAJA CALIFORNIA S DE PR DE RL (San Quintín).
- Temporada de langosta: 15-sep a 15-feb.
- Limitación principal: datos escasos (~400 observaciones útiles para langosta + San Quintín).

### Expansión 2026

Hay datos nuevos pendientes de conseguir (al 2026-04-16 aún no están accesibles; hay que descargarlos/solicitarlos):
- Más temporadas de langosta en San Quintín (2022-2025).
- Otras especies en San Quintín (erizo, abulón, etc.).
- Otras regiones / unidades económicas.

Credenciales GlobColour (FTP) y Copernicus (MOTU/copernicusmarine) del borrador ya no son válidas; hay que regenerarlas. Copernicus migró de `motuclient` a `copernicusmarine` en marzo 2024, así que el ETL de esa fuente hay que reescribirlo.

Artefactos del borrador (joblib XGB, `.h5` LSTM, métricas) están en un bucket S3 — se consultarán más adelante para comparar métricas.

### Hallazgo bibliográfico crítico

Villaseñor-Derbez, Arafeh-Dalmau & Micheli (2024, *Communications Earth & Environment*) documenta que **las olas de calor marinas (MHW) redujeron capturas de langosta/erizo/pepino en Baja California entre 15-58%**, con impactos mayores cerca de quiebres biogeográficos como San Quintín. Esto **explica el bache de la temporada 2021-2022** (post-MHW 2019-2021). Por tanto **siempre incluir un índice MHW** (categorización Hobday et al. 2016) como variable explicativa en los nuevos experimentos.

## Objetivos de esta fase del proyecto

Cinco experimentos ordenados por prioridad (detalle en `PLAN.md`):

1. **ETL + re-entrenamiento del baseline** con datos 2022-2025.
2. **Feature engineering sistemático + SHAP** (prioridad alta).
3. **Modelo jerárquico / global multi-especie y multi-región** (prioridad alta, más ambicioso).
4. **Pronóstico probabilístico con Conformalized Quantile Regression**.
5. **Temporal Fusion Transformer** (opcional, prueba de techo).

## Convenciones de código

### Estilo y estructura
- **Python 3.11+**. Usar type hints en funciones públicas.
- **Gestión de entorno**: usar `uv` si ya está configurado; si no, `poetry` o `pip` con `requirements.txt`. Antes de agregar dependencias verificar qué sistema usa el repo.
- **Formateo**: `ruff format` (o `black` si ya está en el repo). No mezclar.
- **Linting**: `ruff check`. Arreglar warnings cuando sea trivial, no introducir nuevos.
- **Imports**: ordenados con `isort` o `ruff`.
- **Docstrings**: estilo Google o NumPy, consistente con lo que ya exista en el repo.
- **Naming**: `snake_case` para funciones/variables, `PascalCase` para clases, `SCREAMING_SNAKE_CASE` para constantes.
- **Funciones puras cuando se pueda**: separar I/O de lógica de transformación para facilitar tests.

### Estructura de carpetas esperada (ajustar a lo que ya exista)
```
data/
  raw/              # crudos, nunca editar
  interim/          # intermedios del ETL
  processed/        # listos para modelado
notebooks/          # EDA y prototipado
src/ o forecasting_models/
  etl/              # extracción y transformación
  features/         # feature engineering
  models/           # wrappers de cada modelo
  evaluation/       # métricas y backtesting
  utils/
experiments/        # scripts/notebooks de cada experimento
configs/            # YAML de configuración (hydra o plain)
tests/
reports/
  figures/
```

### Modelado
- **Partición temporal estricta**: cualquier split debe respetar el orden cronológico. Corte de test canónico: **01-julio de 2020** (`2020-07-01`) para comparabilidad con los scripts del borrador original (ARIMA, LGBM, XGBoost, LSTM). Considerar además partición adicional con corte `2024-06-01` para aprovechar los datos 2022-2025 cuando estén disponibles.
- **Validación**: time series split con ventana expansiva (misma lógica del borrador original, Figura 1 de la tesis).
- **No data leakage**: imputación, normalización y cualquier stat agregada se calcula **solo** sobre train y se aplica a val/test. Usar `Pipeline` de scikit-learn o funciones explícitas.
- **Seeds fijas** en todo lo estocástico. Registrar el seed en el config del experimento.
- **Hiperparámetros**: Grid Search para baseline estadísticos, Bayesian Search (Optuna preferido) para ML/DL. Limitar trials para que sea reproducible en tiempos razonables.

### Métricas obligatorias
- **MAE, RMSE** (diarios).
- **sMAPE** (robusto a escala, agregar esta vez).
- **Error porcentual en suma de temporada** (definición de temporada: 15-sep a 15-feb).
- **CRPS** si el modelo produce distribución/cuantiles (Experimento 4 en adelante).
- **Coverage empírico** de intervalos para CQR (debe estar cerca del nominal).

### Experimentos y tracking
- Usar **MLflow** si ya está configurado; si no, bastan tablas CSV/Parquet en `reports/` con un identificador por experimento (`exp_id = f"{fecha}_{nombre}"`).
- Cada experimento produce: (1) modelo serializado (joblib/pickle/`.pt`), (2) métricas en JSON, (3) figuras en PNG, (4) summary en Markdown.
- **Reproducibilidad**: cada experimento debe poder re-correrse con un solo comando (`python experiments/exp_N_xxx.py` o similar).

### Git
- **Branch por experimento**: `exp/N-nombre-corto` (p. ej. `exp/2-feature-engineering-shap`).
- **Commits descriptivos**: presente, español o inglés consistente con el repo.
- **No commitear datos crudos ni modelos pesados**. Usar `.gitignore` para `data/raw/`, `data/processed/`, `*.pkl`, `*.pt` (salvo artefactos pequeños de ejemplo).
- **No commitear secretos ni credenciales de FTP de GlobColour ni de OneDrive de COBI**. Usar `.env` + `python-dotenv`.

## Dependencias recomendadas para los nuevos experimentos

Antes de instalar, revisar qué ya está en `requirements.txt` / `pyproject.toml` / `environment.yml`.

| Propósito | Paquete |
|---|---|
| Forecasting clásico/ML | `scikit-learn`, `xgboost`, `lightgbm`, `statsmodels`, `prophet` |
| Deep learning | `pytorch`, `pytorch-forecasting` (para TFT), `pytorch-lightning` |
| Series de tiempo ML | `skforecast` o `darts` (global models, backtesting) |
| Interpretabilidad | `shap` |
| Probabilístico | `mapie` (CQR), opcionalmente `neuralprophet` |
| MHW | `marineHeatWaves` (port de Python de Hobday et al. 2016) o implementación propia |
| Oceanográficos | `xarray`, `netCDF4` (para `.nc` de GlobColour) |
| Tracking | `mlflow` u `hydra` + `omegaconf` |
| Testing | `pytest` |

## Guardrails de comportamiento

### Antes de escribir código
1. **Explorar primero**. Leer el repo: `README.md`, estructura de carpetas, `pyproject.toml`/`requirements.txt`, archivos de modelos existentes. **No asumir**.
2. **Resumir lo encontrado** antes de proponer cambios: "el repo tiene X estructura, usa Y para Z, el modelo LSTM está en archivo W".
3. **Confirmar** con el usuario si hay ambigüedad significativa (p. ej. dos archivos parecen hacer lo mismo, o la convención de nombres es inconsistente).

### Durante el desarrollo
- **Cambios mínimos y localizados**. No reescribir lo que ya funciona; extender, no reemplazar.
- **Respetar estilo existente**. Si el repo usa f-strings, mantenerlo; si usa `.format()`, mantenerlo.
- **Tests para código no trivial**. Al menos uno por función de feature engineering, uno por métrica, uno por transformador de ETL.
- **Commits atómicos**. Un commit = un cambio lógico.
- **Preguntar antes de correr entrenamientos largos** (>10 min). Confirmar que el setup es correcto.

### Manejo de secretos y datos
- **Nunca hardcodear** URLs con credenciales, tokens, rutas a OneDrive personales, claves FTP.
- Si encuentro un secreto hardcodeado en el repo: **no lo reproduzco en el output**, aviso al usuario en privado y sugiero moverlo a `.env`.
- Datos crudos: **no pegar contenido de tablas reales en el chat** (podría contener datos de pescadores). Resumir estadísticas agregadas.

### Cuando algo falla
- **Leer el traceback completo** antes de proponer fix.
- **Hipótesis primero, código después**: "creo que falla porque X; voy a verificar Y; luego arreglar Z".
- **No silenciar errores** con `try/except: pass`. Si hay un fallback, que esté documentado.

## Definición del dominio: glosario

| Término | Significado |
|---|---|
| **Temporada de langosta** | 15-sep a 15-feb del año siguiente. Fuera de ese rango = veda. |
| **Unidad económica (UE)** | Cooperativa pesquera con una zona TURF asignada. La del borrador original es LITORAL DE BAJA CALIFORNIA S DE PR DE RL. |
| **TURF** | Territorial Use-Rights for Fisheries. Zona exclusiva de una UE. |
| **MHW (Marine Heatwave)** | Evento de SST anómalamente alta >5 días consecutivos, según Hobday et al. 2016. Categorías: moderado, fuerte, severo, extremo. |
| **Variables X (regresoras)** | 16 variables oceanográficas de GlobColour + sensores COBI (ver Apéndice B del paper). |
| **Variable Y (objetivo)** | Volumen total de pesca diario en kg. |
| **Desplazamiento de 3 meses** | Convención del proyecto: X(t) se usa para predecir Y(t+90d). Acordado con COBI. |
| **ds, y** | Nombres de columnas para fecha y objetivo (convención de Prophet, reutilizada en todo el repo). |

## Al terminar cada sesión de trabajo

- Actualizar `PLAN.md` marcando las tareas completadas.
- Dejar un resumen en `reports/sessions/YYYY-MM-DD.md` con: qué se hizo, resultados principales, bloqueadores, próximo paso concreto.
- Si se introdujeron decisiones de diseño no triviales, documentarlas en `docs/decisions/` como ADRs cortos (Architecture Decision Records).
