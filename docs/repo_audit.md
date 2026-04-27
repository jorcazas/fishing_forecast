# repo_audit.md

Auditoría del repositorio `fishing_forecast` al inicio de la fase de expansión 2026 (Fase 0 del `PLAN.md`). Estado congelado al commit `1f3aa03` (main).

---

## 1. Estructura actual

```
fishing_forecast/
├── CLAUDE.md                       # instrucciones permanentes (nuevas, 2026)
├── PLAN.md                         # plan de experimentos (nuevo, 2026)
├── README.md                       # 2 líneas, placeholder
├── LICENSE
├── .gitignore                      # plantilla estándar de Python, sin entradas para datos ni modelos
├── etl/
│   ├── __init__.py
│   ├── run_etl.sh / run_etl.bat    # orquestadores (extract → transform → load)
│   ├── requirements.txt            # ETL-only (netCDF4, xarray, motuclient, psycopg2)
│   ├── globcolour.py               # lib: FTP + procesamiento de .nc
│   ├── cicese.py                   # lib: scraping HTTP de CICESE + lectura .dat
│   ├── copernicus.py               # lib: Copernicus MOTU API
│   ├── extract/   {cicese, copernicus, globcolour, google_earth}_extract.py
│   ├── transform/ {cicese, copernicus, globcolour, google_earth, dataset_merger}.py
│   └── load/      {cicese, copernicus, globcolour, google_earth}_load.py
└── forecasting_models/
    ├── README.md                   # descripción del paper (2024)
    ├── requirements.txt            # modelado legacy, versiones fijadas 2021 (tensorflow 2.7, pandas 1.3.3)
    ├── data_processing.ipynb       # concatena CSVs limpios de GlobColour en un X.csv único
    ├── baseline/
    │   ├── baseline.ipynb          # EDA + ARIMA baseline original
    │   └── requirements.txt        # otra pila, versiones 2023 (pandas 2.1, sklearn 1.3)
    ├── modeling/
    │   └── code_wandb.py           # NO relacionado al proyecto: script CIFAR10 + resnet18
    └── models/
        ├── README.md
        ├── model.py                # clase base vacía
        ├── arima.py                # script mango Tuner + statsmodels ARIMA
        ├── prophet.py              # script mango Tuner + Prophet
        ├── lgbm_xgboost.py         # script TimeSeriesSplit + hyperopt/BayesSearchCV
        ├── lstm.py                 # script TF/Keras + wandb para el ensamble XGB+LSTM
        └── nixtla/time_gpt.py      # esqueleto wrapper de TimeGPT
```

**Directorios `data/`, `data/raw/`, `data/processed/` no existen.** Los scripts asumen que `etl/data/...` se crea en ejecución. Tampoco existen `tests/`, `src/`, `experiments/`, `reports/`, `models/` (artefactos), `configs/`, `docs/decisions/`, `notebooks/`.

---

## 2. Inventario de modelos

| Archivo | Estado | Qué hace realmente |
|---|---|---|
| `models/model.py` | stub | Clase base `Model` con `train/predict/evaluate` no implementados (imprime y devuelve dummies). Todas las subclases hacen `super().__init__()` y nada más — nunca se usa la interfaz. |
| `models/arima.py` | script | ARIMA con `mango.Tuner` (10 iteraciones). `ROOT = "path_to_data/"` hardcodeado, lee `lstm_data.csv`. Corte temporal `"2020-07-01"`. Redefine `class ARIMA(Model)` y luego importa otro `ARIMA` de `statsmodels` → colisión de nombre. |
| `models/prophet.py` | script | Prophet con `mango.Tuner` + `loguniform`. Lee `Arribos2017-2021.csv`, filtra LANGOSTA + LITORAL BC. Corte `"2021-01-01"`. Mismo problema de colisión de nombre `class Prophet(Model)` vs `from prophet import Prophet`. |
| `models/lgbm_xgboost.py` | script | `set_weekly_data()` / `set_daily_data()` filtran por temporadas 2017-18…2021-22 (hardcoded). Shift de 90 días en `x1..x16`. Corte `"2020-07-01"`. Hyperopt + `BayesSearchCV` + `TimeSeriesSplit(5)`. Guarda joblib en `/content/drive/MyDrive/Tesis/`. |
| `models/lstm.py` | script | LSTM ensamble con columna `xgboost` como feature (ensamble XGB→LSTM). Filtros por temporada 2017-18…2021-22. `sequence_length=10`. Grid de hiperparámetros manual, `wandb.init(project="cobi-forecast-only-fishing-seasons")`. Guarda `.h5` en Drive. Split por razón `368/628`. |
| `models/nixtla/time_gpt.py` | plantilla | Wrapper genérico de `nixtla.TimeGPT`, sin datos reales. |
| `modeling/code_wandb.py` | no relacionado | Ejemplo CIFAR10 + ResNet18 de ITAM. **Sugiero mover a `legacy/` o borrar.** |
| `baseline/baseline.ipynb` | notebook | EDA del dataset Arribos 2017-2021 + primer ARIMA (mismo que `arima.py`). |
| `data_processing.ipynb` | notebook | Concatena CSVs limpios por variable de GlobColour en un `X.csv` único (307179 × 44). Rutas Windows hardcodeadas. |

**Resumen**: los modelos del paper existen como scripts independientes con duplicación importante de código (preprocessing replicado en cada uno). No hay una API común. El ensamble XGB+LSTM se hace vía columna `xgboost` en el CSV de entrada del LSTM, no como composición programática.

---

## 3. Inventario de ETL

**Fuentes externas configuradas**:
1. **GlobColour** (FTP `ftp.hermes.acri.fr`) — 16 variables oceanográficas satelitales. Descarga por día, resolución configurable (4/25/100 km). Coordenadas `[-117, -112.5, 28, 32]` (San Quintín + Isla Cedros). Rango 2017-01-01 → 2023-01-01.
2. **CICESE** (HTTP `http://redmar.cicese.mx/emmc/DATA/`) — estaciones Isla Cedros y Guerrero Negro. Archivos `.dat` con formato de columnas fijas (anio/mes/dia/hora/…/nivel_mar/temperatura_agua/radiacion_solar/viento/humedad/presion/precipitacion).
3. **Copernicus Marine** (MOTU API) — configurable vía `infra/copernicus_var_dict.csv` (archivo no incluido en repo). Coordenadas mismas que GlobColour.
4. **Google Earth Engine** — solo un esqueleto (`extract/google_earth_extract.py`), scripts de transform/load están **vacíos** (1 línea). No operativo.

**Pipeline**:
- `extract/` → descarga a `data/{fuente}/raw/` (ruta relativa al parent_dir de `etl/`).
- `transform/` → `.nc` / `.dat` → CSV en `data/{fuente}/processed/`. `dataset_merger.py` concatena los CSV de GlobColour y agrega coordenadas + valor medio.
- `load/` → inserta fila por fila a PostgreSQL `localhost:cobi` con credenciales hardcodeadas (`postgres`/`admin`). Google Earth y Copernicus load están vacíos.

**Orquestación**: `run_etl.sh` / `.bat` llaman a cada etapa como módulo (`python -m extract.cicese_extract`). Asume `cwd = etl/` y que existen `etl/input/variable_dict.csv`, `etl/config/config.ini`, `etl/infra/copernicus_var_dict.csv` — **ninguno de estos archivos existe** en el repo.

**El merge final (oceanográfico + arribos) NO está en `etl/`.** Vive implícito en `data_processing.ipynb` y en el filtrado hecho por cada script de modelo.

---

## 4. Formato de datos y splits

- **Formato**: CSV plano. Todos los scripts leen con `pd.read_csv`. No hay Parquet, HDF5, ni feather. Archivos crudos `.nc` de GlobColour, `.dat` de CICESE.
- **Columnas estándar del dataset de modelado**: `ds` (fecha), `y` (kg desembarcados), `x1`…`x16` (variables GlobColour), y en LSTM adicionalmente `xgboost` (predicción del modelo base, feature del ensamble).
- **Archivos que se referencian pero no están en el repo**:
  - `Arribos2017-2021.csv` — volumen de arribos pesqueros (proviene de COBI).
  - `lstm_data.csv`, `lstm_shifted_data.csv`, `lstm_only_fishing_season.csv`, `lstm_xgboost_data.csv` — datasets intermedios construidos en los scripts.
  - `X.csv` — dataset oceanográfico concatenado (se produce en `data_processing.ipynb`).
- **Splits**: cortes temporales hardcodeados, **inconsistentes entre modelos**:
  - ARIMA: `cut_date = "2020-07-01"`.
  - Prophet: `"2021-01-01"` (train `<=`, test `>=` → solapamiento en la frontera).
  - LGBM/XGBoost: `"2020-07-01"` (weekly y daily).
  - LSTM: `split_ratio = 368/628` (empíricamente ≈ corte previo a temporada 2021).
- `CLAUDE.md` dice "corte de test: 01-junio de 2021". **Ningún script usa esa fecha**. Hay que decidir: (a) re-correr baselines con el corte 2021-06-01 para consistencia con lo declarado, o (b) actualizar CLAUDE.md al `2020-07-01` real del paper.
- **Desplazamiento de 3 meses**: implementado como `df.shift(periods=90)` sobre `x1..x16` en `lgbm_xgboost.py` y equivalente en el preproceso LSTM.

---

## 5. Tracking, tests, estilo, Python version

| Ítem | Estado |
|---|---|
| **MLflow** | no |
| **Weights & Biases** | sí en `lstm.py` (proyecto `cobi-forecast-only-fishing-seasons`). También en `code_wandb.py` (no relacionado). |
| **Optuna** | no; se usa `mango` (ARIMA, Prophet), `hyperopt` (LGBM/XGB), `BayesSearchCV` (XGB). |
| **Tests (pytest/unittest)** | **no existen.** `tests/` no existe. En `baseline/requirements.txt` figura `pytest==7.4.2` pero sin suite. |
| **Formateo (black/ruff)** | no configurado; no hay `.ruff.toml`, `pyproject.toml` ni `.pre-commit-config.yaml`. El estilo es heterogéneo entre scripts. |
| **Docstrings** | estilo Google/reST mezclado. `globcolour.py` está bien documentado; `lgbm_xgboost.py` casi sin docstrings. |
| **Type hints** | solo en `etl/globcolour.py` y `etl/copernicus.py`. |
| **Python version** | no pinneada en ningún config. `pandas==2.1.1` en `baseline/requirements.txt` implica ≥3.9; `tensorflow==2.7.0` en `forecasting_models/requirements.txt` implica 3.7–3.10. Hay **incompatibilidad potencial entre los dos requirements.txt**. |
| **Gestión de entorno** | `pip install -r requirements.txt` según README (venv). No hay `uv`, `poetry`, `conda/environment.yml`. |
| **Config** | `etl/config/config.ini` esperado pero no existe; ruta en `globcolour.py` es **absoluta Windows** (`C:/Users/javi2/...`). |

---

## 6. Riesgos detectados (seguridad y reproducibilidad)

1. **Rutas absolutas de Windows y Google Drive** hardcodeadas en 4 scripts de modelos (`path_to_data/`, `C:/Users/javi2/...`, `/content/drive/MyDrive/Tesis/...`). Rompen la reproducibilidad fuera del entorno original. **No son secretos pero revelan la estructura local.**
2. **Credenciales DB hardcodeadas** en `etl/load/globcolour_load.py:62` y `etl/load/cicese_load.py:19` (`password="admin"`). Para una DB local de desarrollo en `localhost` es de bajo riesgo, pero debe moverse a `.env` antes de cualquier commit futuro.
3. **Archivo de config con credenciales externas** (`config/config.ini`, rutas con `javi2`) referenciado pero no incluido. Hay que asegurarse de que nunca se commitee y crear un `.env.example` equivalente.
4. **Dependencia `motuclient` está deprecada** — Copernicus Marine migró a `copernicusmarine` desde marzo 2024. `copernicus.py` probablemente falle contra la API actual.
5. **Loops inserción fila-por-fila a Postgres** (`cur.execute` en for) en `load/*.py` es lento (O(n) round-trips); trivial de reescribir con `execute_values` si se mantiene ese path.
6. **Notebooks commiteados con outputs** (errores de ejecución incluidos en `baseline.ipynb`) — inflan el diff y pueden contener datos. Recomiendo `nbstripout` antes de cualquier commit de notebook.
7. **`code_wandb.py` es CIFAR10/ResNet18**, no tiene relación con el proyecto.
8. **`dataset_merger.py` tiene un bug**: itera `dataset['date'].unique()` y dentro del loop reasigna `mean_value = subset[...]` (una Series), pero luego construye `pd.DataFrame({...})` con `date` como escalar y `value/lat/lon` como Series — pandas broadcasta pero el código es frágil y la salida probablemente esté mal.

---

## 7. Comparación con lo declarado en `CLAUDE.md`

| CLAUDE.md dice | Realidad del repo | Conflicto |
|---|---|---|
| Estructura `data/raw/`, `data/processed/` | No existe; hay `etl/data/{fuente}/{raw,processed}/` (generada en runtime, no commiteada). | Ajustar: definir estructura canónica y migrar. |
| `src/` o `forecasting_models/` con subpaquetes `etl/`, `features/`, `models/`, `evaluation/` | El código ya vive en `etl/` (raíz) y `forecasting_models/` (raíz, con `models/` dentro). No hay `features/` ni `evaluation/`. | Decidir si crear `src/fishing_forecast/` nuevo o seguir con la estructura doble actual. |
| Corte de test `01-junio-2021` | Scripts usan `2020-07-01` (LGBM/XGB/ARIMA) o `2021-01-01` (Prophet). | Aclarar cuál es el corte de referencia del paper publicado. |
| `ruff format` + `ruff check` | No configurado. | Introducir en Fase 1. |
| `pytest` como framework de tests | No hay tests. | Introducir en Fase 1 con ≥1 test por módulo de ETL/features. |
| MLflow recomendado | Se usa W&B en LSTM. | Decidir: mantener W&B o migrar a MLflow. |
| Python 3.11+ con type hints | Requirements sugieren 3.9/3.10; hints solo parciales. | Pinnear Python, migrar requirements. |
| Seeds fijas | Solo `tf.random.set_seed(1807)` en LSTM. ARIMA/LGBM/XGBoost sin seeds. | Introducir `seed=` en configs. |
| `.env` + `python-dotenv` | No existe. Se usa `config/config.ini`. | Migrar a `.env`. |

---

## 8. Lista priorizada de cosas a arreglar/estandarizar antes de empezar Fase 1

**P0 — bloquea cualquier experimento reproducible**

1. Consolidar dependencias en un solo `pyproject.toml` (o `requirements.txt` raíz si prefieres simple) con Python 3.11 pinneado. Las tres `requirements.txt` actuales son contradictorias.
2. Decidir corte de test canónico (`2020-07-01` o `2021-06-01`) y documentar la decisión en un ADR corto.
3. Eliminar rutas Windows/Colab hardcodeadas. Centralizar en una única función `get_data_root()` que lea de `.env`.
4. Preguntar al usuario (Javier) dónde están físicamente los datos nuevos 2022-2025 para langosta + otras especies + otras UEs.
5. Crear `.env.example` con las claves esperadas: `GLOBCOLOUR_USER`, `GLOBCOLOUR_PASS`, `COPERNICUS_USER`, `COPERNICUS_PASS`, `DATA_ROOT`, `PG_*`.
6. Mover credenciales Postgres hardcodeadas a `.env`.

**P1 — necesario para Fase 1 limpia**

7. Crear estructura `data/{raw,interim,processed}/` con `.gitkeep` y actualizar `.gitignore` para excluir crudos (`data/raw/**`, `data/interim/**`, `*.nc`, `*.dat`, `*.csv` en datos grandes, `*.joblib`, `*.h5`, `*.pt`, `*.pkl`).
8. Reescribir `dataset_merger.py` — bug de construcción del DataFrame y lentitud por iteración fila-por-fila.
9. Migrar `motuclient` → `copernicusmarine` o validar que la API vieja siga funcionando.
10. Mover `modeling/code_wandb.py` a `legacy/` (no pertenece al proyecto).
11. Quitar outputs de los notebooks antes de commitear (`nbstripout`).

**P2 — calidad de vida**

12. Crear `tests/` con un test inicial por módulo de ETL (fixtures chicos, ~5 filas de `.nc` sintético).
13. Agregar `ruff` + pre-commit hook.
14. Unificar API de modelos con una clase base real (la actual `Model` es un stub vacío).
15. Introducir `mlflow` o formalizar el uso de W&B con un proyecto por fase del plan.
16. Documentar en `docs/etl_design.md` el esquema canónico del dataset consolidado (columnas, tipos, granularidad espacial/temporal).

---

## 9. Huecos de información que requieren input del usuario

- Ubicación física de los datos nuevos 2022-2025 (langosta SQ, erizo, abulón, otras UEs).
- ¿Hay un archivo `infra/copernicus_var_dict.csv` o `input/variable_dict.csv` del borrador original? Si sí, ¿dónde?
- ¿Acceso a `config.ini` antiguo con credenciales FTP de GlobColour y MOTU de Copernicus? (Si no, hay que regenerarlas.)
- ¿Existen los artefactos de modelos del borrador (joblib XGB, `.h5` LSTM) accesibles para comparar métricas?
- ¿W&B del proyecto `cobi-forecast-only-fishing-seasons` sigue activo y consultable?
