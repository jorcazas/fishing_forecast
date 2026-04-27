# etl_design.md

Diseño del pipeline de ETL para la fase de expansión 2026 del proyecto `fishing_forecast`. Documento de arquitectura, no de implementación. La implementación vive en Fase 1.2 del `PLAN.md`.

**Alcance**: cómo se va a pasar de las fuentes crudas (GlobColour, CICESE, Copernicus, COBI-arribos) a una única tabla consolidada lista para los experimentos de las Fases 2-5.

**Fuera de alcance**: feature engineering derivado (lags, rolling, anomalías, interacciones) — eso vive en `src/features/` (Fase 2). El ETL solo produce features "crudas" de cada fuente, alineadas en tiempo y espacio.

---

## 1. Objetivos

1. **Una sola tabla consolidada** con granularidad `(ds × species × economic_unit)` que sirva de entrada a todos los modelos (baseline, jerárquico, probabilístico, TFT).
2. **Reproducible sin rutas hardcodeadas**: todo parámetro de I/O sale de `.env` + un YAML de configuración por fuente.
3. **Idempotente**: volver a correr el ETL con datos ya descargados no re-descarga ni re-procesa si no hay cambios.
4. **Testeable**: cada módulo tiene un test unitario con fixtures chicos (5-10 filas).
5. **Compatible con el borrador**: los nombres de columnas `ds`, `y`, `x1..x16` se conservan para no romper los scripts del borrador durante la transición.
6. **Preparado para expansión**: agregar una nueva especie o UE no debería requerir tocar el código core del ETL — solo el YAML de configuración.

---

## 2. Fuentes de datos

| Fuente | Tipo | Variables | Granularidad | Estado | Uso |
|---|---|---|---|---|---|
| **GlobColour** | FTP `ftp.hermes.acri.fr`, `.nc` diarios | 16 variables ópticas/biogeoquímicas (chlor_a, POC, KD490, PAR, CHL1, SST-like T865, …) | diaria, grid 4 km (o 25/100) | credenciales por regenerar | principal: X_1..X_16 del modelo |
| **Copernicus Marine** | API `copernicusmarine` (antes MOTU) | SST L4, corrientes, salinidad, etc. — definido en `configs/copernicus_vars.csv` | diaria, grid ~0.05° | migrar de `motuclient` → `copernicusmarine`; credenciales por regenerar | fuente primaria de SST para MHW + covariables adicionales |
| **CICESE `redmar.cicese.mx`** | HTTP directorio `.dat` | sensores in-situ: nivel de mar, temp agua/aire, viento, humedad, presión, precipitación | minuto → agregado diario | funcional pero estaciones descargadas incompletas (Guerrero Negro, Isla Cedros) | validación cruzada de SST costera; variables de viento/presión |
| **COBI arribos** | CSV desde OneDrive de COBI | `fecha`, `peso_desembarcado`, `especie`, `unidad_economica`, … | diaria, por pesca registrada | sólo 2017-2021 en el borrador; 2022-2025 **pendiente** | target `y` |
| **Google Earth Engine** | `ee` API | MODIS/VIIRS potencialmente | diaria, grid variable | esqueleto, no operativo | **no incluir en Fase 1**; evaluar como sustituto de GlobColour si sigue inaccesible |

---

## 3. Arquitectura

### 3.1. Flujo de alto nivel

```
           ┌──────────────────┐
           │  configs/*.yaml  │  (parámetros, rutas, coordenadas por UE,
           │      .env        │   calendarios por especie, credenciales)
           └────────┬─────────┘
                    │
   ┌────────────────┴──────────────────────────────────────┐
   │                                                       │
   ▼                                                       ▼
EXTRACT                                                 EXTRACT
├── extract_globcolour  ──►  data/raw/globcolour/{var}/{yyyy}/{mm}/*.nc
├── extract_copernicus  ──►  data/raw/copernicus/{product}/*.nc
├── extract_cicese      ──►  data/raw/cicese/{station}/*.dat
└── extract_arribos     ──►  data/raw/arribos/arribos_{yyyy-yyyy}.csv

                        │
                        ▼
TRANSFORM (por fuente, a formato largo tidy)
├── transform_globcolour ──►  data/interim/globcolour/{var}.parquet
│                              cols: ds, lat, lon, value, variable
├── transform_copernicus ──►  data/interim/copernicus/{product}.parquet
├── transform_cicese     ──►  data/interim/cicese/{station}.parquet
│                              cols: ds, var_i (agregado diario)
└── transform_arribos    ──►  data/interim/arribos.parquet
                               cols: ds, species, economic_unit, region, y

                        │
                        ▼
FEATURE-LIKE (agregación espacial / temporal por UE)
├── aggregate_ocean  ──►  data/interim/ocean_by_ue.parquet
│                          cols: ds, economic_unit, x1..x16, sst, [copernicus vars]
│                          (promedio dentro de bounding box TURF de cada UE)
└── derive_mhw       ──►  data/interim/mhw_by_ue.parquet
                           cols: ds, economic_unit, mhw_category, mhw_intensity

                        │
                        ▼
CONSOLIDATE
└── consolidate  ──►  data/processed/dataset_v1.parquet
                       cols: ds, species, economic_unit, region, y,
                             x1..x16, sst, mhw_category, mhw_intensity,
                             season, in_season,
                             [flags de calidad: is_imputed_*, source]

                        │
                        ▼
QUALITY CHECKS
└── quality_checks  ──►  reports/etl/qc_{run_id}.json
                          (asserts sobre rangos, duplicados, huecos por UE×año,
                           coverage de x_i, consistencia de season/in_season)
```

### 3.2. Principios

- **Raw es inmutable**. Nadie escribe en `data/raw/`. Si algo se re-descarga, se hace con idempotencia (hash de contenido o filename match).
- **Interim acumula resultados parciales**. Cada fuente produce un Parquet largo-tidy independiente antes de cualquier join.
- **Processed es la única tabla que consume modelado**. Todos los experimentos leen de `data/processed/dataset_vN.parquet`. Se versiona con `v1`, `v2`, … (semver ligero).
- **Separación E → T → L estricta**. `extract_*.py` no procesa, `transform_*.py` no descarga, `consolidate.py` no descarga ni re-agrega — solo joinea interim + feature-like.
- **No leakage en el ETL**. El ETL produce datos "crudos" por día. Cualquier rolling, lag, anomalía, escalado se hace en feature engineering, sobre train solamente.

---

## 4. Esquema de la tabla consolidada (`dataset_v1.parquet`)

Granularidad canónica: **una fila por `(ds, species, economic_unit)`**.

### 4.1. Columnas

| Columna | Tipo | Nullable | Descripción |
|---|---|---|---|
| `ds` | `date32` | no | Fecha calendario. Normalizada a UTC-8 (hora Pacífico) al momento de ingerir CICESE. |
| `y` | `float64` | sí | Volumen desembarcado en **kg** en esa fecha para esa `(species, economic_unit)`. NaN si no hubo registro de arribo; no confundir con 0 (que implicaría día laboral sin captura). Ver §4.4. |
| `species` | `category` (string) | no | Código interno normalizado: `lobster_red`, `lobster_spiny`, `urchin_red`, `urchin_purple`, `abalone_blue`, `abalone_red`, `abalone_black`, `abalone_yellow`, `crab_king`, `fish_other`. El mapping desde el texto original (`LANGOSTA ROJA ENT. FCA.`, …) vive en `configs/species_mapping.yaml`. |
| `economic_unit` | `category` (string) | no | Código interno snake_case: `litoral_bc_sur` (= LITORAL DE BAJA CALIFORNIA S DE PR DE RL), `isla_natividad`, `ensenada`, … Mapping en `configs/economic_units.yaml`. |
| `region` | `category` (string) | sí | Agregado superior: `san_quintin`, `vizcaino`, `bahia_magdalena`, … Derivado del mapping UE→región. |
| `x1` … `x16` | `float64` | sí | 16 variables oceanográficas GlobColour promediadas sobre el bounding box TURF de la UE. **Sin shift**. Nombres reales (`chlor_a`, `POC`, …) van en `configs/globcolour_vars.yaml` con su mapeo a `x_i`. |
| `sst` | `float64` | sí | SST diaria L4 (Copernicus o AVHRR OI v2), promediada sobre el mismo bounding box. |
| `mhw_category` | `int8` | no | 0=sin MHW, 1=moderado, 2=fuerte, 3=severo, 4=extremo (Hobday et al. 2016). |
| `mhw_intensity` | `float64` | sí | Anomalía de SST sobre baseline climatológica 30 años (p90). NaN fuera de eventos MHW activos si prefieres; 0 si no. Decidir en §5.4. |
| `season` | `string` | no | Identificador de temporada calendario, p.ej. `2017_2018`. Null fuera de temporada solo si `in_season=False` y está entre temporadas — prefiero asignar siempre la próxima temporada para facilitar agrupación. |
| `in_season` | `bool` | no | True si `ds` está dentro del rango de temporada legal para **esa especie** (cada especie tiene su propio calendario, ver §5.2). |
| `is_imputed_y` | `bool` | no | True si `y` fue imputada (en general **no imputamos** `y`; este flag queda para días con `y=0` derivado de calendario in-season sin registro). |
| `is_imputed_x` | `bool` | no | True si cualquier `x_i` o `sst` se imputó (gap filling ≤3 días lineal). Si se imputó, también se registra el método en `ocean_impute_method`. |
| `ocean_impute_method` | `string` | sí | `none`, `linear_3d`, `spatial_neighbor`. |
| `source_globcolour_files` | `int32` | sí | Cuántos archivos `.nc` contribuyeron al promedio espacial de ese día (0 → todos los `x_i` son NaN). |
| `etl_run_id` | `string` | no | ID de la corrida (`YYYY-MM-DD_HHMMSS`) que produjo esta fila, para trazabilidad. |

### 4.2. Particionamiento en disco

Parquet particionado por `species` y `year(ds)`:

```
data/processed/dataset_v1/
├── species=lobster_red/
│   ├── year=2017/part-0.parquet
│   ├── year=2018/part-0.parquet
│   └── ...
├── species=urchin_red/
│   └── ...
```

**Por qué**: (a) pyarrow/polars filtran partición sin cargar bytes irrelevantes, (b) la mayoría de los experimentos filtran por especie, (c) un año cabe en memoria sin problemas. Re-serializar a CSV si se necesita compatibilidad estricta con el borrador.

### 4.3. Convenciones de tipos

- `ds` es `date32`, **no** `datetime`. Si alguien necesita timestamp, que lo derive.
- Todos los numéricos son `float64` salvo contadores (`int32`) y banderas (`bool`/`int8`).
- Las columnas categóricas usan el tipo `category` de pandas (o `dictionary` de arrow) para ahorrar RAM cuando hay decenas de miles de filas por especie.

### 4.4. Manejo de `y` faltantes

Tres casos posibles:

1. **Día fuera de temporada** (`in_season=False`): `y=0`, `is_imputed_y=False`. Asumimos que no hubo captura legal.
2. **Día dentro de temporada, sin registro COBI**: `y=NaN`, `is_imputed_y=False`. Podría ser un día sin pesca (clima, veda local, festivo) o un hueco en la captura de datos — no sabemos cuál. Los modelos deben decidir cómo tratarlo (Prophet y LSTM manejan NaN; XGBoost/LGBM también si se usa `missing=np.nan`).
3. **Imputación explícita** (p.ej. 0 en ausencia de registro con asunción de "no se pescó"): `is_imputed_y=True`. **No lo hacemos por default**; dejar el NaN para que sea un boolean-flag-friendly missing.

Esto es una **decisión no trivial**; queda como ADR corto (`docs/decisions/ADR-0001-y-missing.md` cuando se implemente).

---

## 5. Decisiones de diseño

### 5.1. Agregación espacial por unidad económica

Cada UE tiene una zona TURF con bounding box definido en `configs/economic_units.yaml`:

```yaml
litoral_bc_sur:
  name: "LITORAL DE BAJA CALIFORNIA S DE PR DE RL"
  region: san_quintin
  bbox: {lon_min: -117.0, lon_max: -115.0, lat_min: 30.0, lat_max: 31.5}
  season_calendars:
    lobster_red: {start: "09-15", end: "02-15"}
    abalone_blue: {start: "01-01", end: "05-31"}   # placeholder — confirmar
```

El bounding box se usa para promediar GlobColour, SST, corrientes. **Método inicial**: media aritmética de píxeles dentro del bbox, ignorando NaN. **Alternativa** (Fase 2): promedio ponderado por distancia al centroide, o recorte con polígono TURF real si COBI comparte el shapefile.

### 5.2. Calendarios de temporada

La temporada de **langosta roja** en Baja California es `09-15 → 02-15` (ya conocido). Para **otras especies** los calendarios pueden diferir y dependen de la UE (algunas cooperativas tienen vedas propias). Ver la lista de especies detectada en el dataset 2017-2021:

- ABULÓN AZUL, NEGRO, ROJO, AMARILLO — calendarios distintos entre sí y entre UEs.
- CANGREJO, ROCOTE, BACALAO, CABRILLA, MEROS — algunos sin temporada formal.
- ERIZO — no presente en LITORAL_BC_SUR pero sí en UEs del centro (Isla Natividad, p.ej.).

**Decisión**: los calendarios se declaran por `(economic_unit, species)` en el YAML. Si una especie aparece en una UE sin calendario declarado, `in_season=True` siempre (conservador) y se emite warning en QC.

### 5.3. Índice MHW

**Fuente de SST para MHW**: Copernicus L4 (e.g. `METOFFICE-GLO-SST-L4-REP-OBS-SST`, grid 0.05°) o NOAA OISST v2.1 (grid 0.25°). Preferencia por NOAA OISST por: licencia abierta, serie histórica 1982-presente sin interrupciones, baseline estable. **Decisión**: empezar con NOAA OISST para el baseline climatológico (1982-2011, 30 años); si queremos consistencia con GlobColour usar Copernicus.

**Algoritmo** (Hobday et al. 2016):
1. Baseline climatológica: p90 diario suavizado con ventana móvil de 11 días sobre 30 años (por defecto 1982-2011).
2. Evento = SST ≥ p90 durante ≥ 5 días consecutivos.
3. Categorías por múltiplo de (p90 - media climatológica):
   - Moderado (I): 1-2×
   - Fuerte (II): 2-3×
   - Severo (III): 3-4×
   - Extremo (IV): ≥ 4×

**Implementación**: librería `marineHeatWaves` (Oliver 2016, port Python de Hobday). Si no instala limpio, implementación propia en `src/features/mhw.py` con tests contra casos conocidos (p.ej. el Blob 2014-2016).

**Granularidad**: MHW se calcula **por UE** sobre la SST promediada del bbox. No por píxel — queremos un escalar por día por UE.

### 5.4. `mhw_intensity` fuera de eventos

Dos opciones:

- (a) `0` cuando `mhw_category=0`. Interpretable como "sin anomalía relevante".
- (b) `SST − climatología` siempre (incluso valores negativos). Preserva toda la señal de anomalía.

**Decisión**: guardar **la anomalía cruda** (opción b) porque es estrictamente más informativa, y el modelo puede derivar el 0 internamente. Renombrar a `sst_anomaly` y dejar `mhw_intensity` solo definido cuando `mhw_category>0` para evitar confusión. → **Dos columnas**: `sst_anomaly` (always) y `mhw_intensity` (NaN fuera de eventos).

### 5.5. Shift de 3 meses

**NO se hace en el ETL.** El shift de 90 días de `x_i` se aplica en la capa de features (Fase 2) como `x_i_lag90`. El ETL guarda los valores oceanográficos alineados con `ds` sin ningún desplazamiento. Esto evita duplicar la convención en dos lugares y permite experimentar con otros lags (30, 60, 180) sin re-correr el ETL.

### 5.6. Granularidad temporal

**Diaria** por defecto. Si un experimento necesita semanal (como el del borrador), se agrega en el preprocesamiento del modelo, no en el ETL.

### 5.7. Coexistencia con el borrador

Durante la transición, el pipeline nuevo convive con los scripts viejos:

- El código nuevo vive en `src/fishing_forecast/etl/` (o `etl/` si el usuario prefiere mantener la raíz actual — decisión pendiente).
- El código viejo en `forecasting_models/` y `etl/` (raíz) no se borra; se marca deprecated con un comentario arriba.
- Cuando el pipeline nuevo esté validado (criterios de éxito de Fase 1), el borrador se mueve a `legacy/`.
- Durante la transición el `dataset_v1.parquet` puede exportarse a `lstm_data.csv` con las mismas columnas (`ds, y, x1..x16`) filtrando `species=lobster_red & economic_unit=litoral_bc_sur` para regression tests contra los scripts viejos.

---

## 6. Módulos a implementar

Estructura propuesta (relativa a la raíz del repo):

```
src/fishing_forecast/
├── __init__.py
├── config.py                    # carga .env + YAMLs, expone Settings (pydantic)
├── etl/
│   ├── __init__.py
│   ├── extract/
│   │   ├── globcolour.py         # FTP, idempotente, cache por filename
│   │   ├── copernicus.py         # copernicusmarine SDK (migrado)
│   │   ├── cicese.py             # scraping HTTP
│   │   └── arribos.py            # lee CSVs COBI desde DATA_ROOT
│   ├── transform/
│   │   ├── globcolour.py         # .nc → parquet largo tidy
│   │   ├── copernicus.py
│   │   ├── cicese.py             # .dat → daily aggregate parquet
│   │   └── arribos.py            # limpieza nombres, mapping species/UE
│   ├── aggregate/
│   │   ├── ocean_by_ue.py        # bbox mean por UE, join multi-fuente
│   │   └── mhw.py                # cálculo MHW (Hobday 2016)
│   ├── consolidate.py            # join final → dataset_vN.parquet
│   ├── quality_checks.py         # pandera schemas + asserts custom
│   └── cli.py                    # `fishing-etl run --stage {extract,transform,consolidate}`
├── features/                     # Fase 2
└── utils/
    ├── io.py                     # wrappers parquet + logging
    └── dates.py                  # season_id, in_season, calendar helpers

configs/
├── etl.yaml                      # rutas, rangos de fechas, resolución
├── economic_units.yaml           # bbox + calendarios por UE
├── species_mapping.yaml          # texto COBI → código interno
├── globcolour_vars.yaml          # mapping x_i ↔ variable real
├── copernicus_vars.yaml
└── cicese_stations.yaml

tests/
├── fixtures/                     # .nc, .dat, .csv chicos (5-10 filas)
└── etl/
    ├── test_extract_globcolour.py
    ├── test_transform_globcolour.py
    ├── test_transform_arribos.py
    ├── test_aggregate_ocean.py
    ├── test_mhw.py
    ├── test_consolidate.py
    └── test_quality_checks.py
```

### 6.1. Contratos entre etapas

- **extract → transform**: contract file = archivo crudo en disco + filename convention (`YYYYMMDD` extraíble por regex). Ningún estado en memoria pasa entre etapas.
- **transform → aggregate**: parquet largo tidy, schema definido con `pandera`. Si cambia el schema, revienta explícitamente.
- **aggregate → consolidate**: parquet semi-ancho por fuente, una fila por `(ds, economic_unit)`.
- **consolidate → downstream**: `dataset_vN.parquet` con el schema de §4.1.

### 6.2. CLI

```
fishing-etl extract --sources globcolour,copernicus --from 2017-01-01 --to 2026-04-15
fishing-etl transform --sources all
fishing-etl aggregate --ues all
fishing-etl consolidate --version v1
fishing-etl run-all --config configs/etl.yaml
fishing-etl qc --dataset data/processed/dataset_v1.parquet
```

Implementado con `typer` o `click` (ya estándar en el ecosistema).

---

## 7. Configuración

### 7.1. `.env`

```
# Credenciales (no commitear; solo .env.example)
GLOBCOLOUR_USER=
GLOBCOLOUR_PASS=
COPERNICUS_USER=
COPERNICUS_PASS=

# Rutas
DATA_ROOT=/Users/javierorcazas/Documents/fishing_forecast/data
S3_BUCKET_LEGACY=     # bucket con artefactos del borrador

# Postgres (solo si se retoma la carga a DB)
PG_HOST=localhost
PG_DB=cobi
PG_USER=postgres
PG_PASSWORD=
```

### 7.2. `.env.example`

Igual que `.env` pero con valores vacíos o placeholders. **Sí se commitea**.

### 7.3. Cargador único

`src/fishing_forecast/config.py` expone `get_settings() -> Settings` (pydantic-settings). Todo el resto del código pide la Settings; no hay `os.getenv` disperso.

---

## 8. Persistencia y versionado

- **Formato**: Parquet (pyarrow). Compresión zstd por default.
- **Versión del dataset**: `dataset_v{N}.parquet` donde `N` se incrementa solo cuando cambia el schema. Cambios de contenido (nuevos datos, no nuevas columnas) mantienen `N` y se distinguen por `etl_run_id` en la columna.
- **Changelog**: `data/processed/CHANGELOG.md` con qué cambió en cada bump de versión.
- **Gitignore**: `data/raw/**`, `data/interim/**`, `data/processed/**` se ignoran completamente. El único rastro en git es el CHANGELOG.
- **Backups**: el `dataset_v{N}.parquet` final se sube a S3 (bucket por definir) manualmente al cerrar cada experimento.

---

## 9. Quality checks

Schema con `pandera`:

```python
# Ejemplo — no es el código final
DatasetSchema = pa.DataFrameSchema({
    "ds": pa.Column("datetime64[ns]", checks=pa.Check.in_range(date(2017,1,1), date.today())),
    "y": pa.Column(float, nullable=True, checks=pa.Check.ge(0)),
    "species": pa.Column(str, checks=pa.Check.isin(SPECIES_CODES)),
    "economic_unit": pa.Column(str, checks=pa.Check.isin(UE_CODES)),
    "mhw_category": pa.Column(int, checks=pa.Check.in_range(0, 4)),
    # ...
})
```

Asserts adicionales custom:
- Para cada `(species, economic_unit, year)`, el número de filas debe ser `365` o `366`; si es menor hay huecos — emitir warning con la lista de fechas faltantes.
- Si `in_season=True` y `y=NaN`, emitir warning (hueco de arribo en temporada).
- `x1..x16` cobertura ≥ 80% de días en temporada por UE (si no, escalar a decidir si se rellena con spatial neighbor o se descarta esa UE del experimento).
- Correlación entre `sst` de Copernicus y `temperatura_agua` de CICESE (cuando ambas existen) debe ser > 0.7. Si no, hay bug en alineación espacial.

Salida del QC: JSON en `reports/etl/qc_{run_id}.json` + summary Markdown en `reports/etl/qc_{run_id}.md`.

---

## 10. Deudas técnicas que se atacan en este ETL

De la lista priorizada del `docs/repo_audit.md`:

- [P0-1] Consolidación en `pyproject.toml` único (Python 3.11, deps modernas) — prerequisito.
- [P0-3] Eliminación de rutas hardcodeadas (vía `Settings`).
- [P0-5] `.env.example` con claves esperadas.
- [P0-6] Credenciales Postgres a `.env` (si se retoma la carga).
- [P1-7] Estructura `data/{raw,interim,processed}/` con `.gitkeep`.
- [P1-8] Reescribir `dataset_merger.py` — ya no existe como tal; lo reemplaza `aggregate/ocean_by_ue.py`.
- [P1-9] Migrar `motuclient` → `copernicusmarine`.

---

## 11. Decisiones tomadas (2026-04-27) y huecos restantes

Respuestas del usuario al cierre de Fase 1.1:

1. **Arribos 2022-2025**: se descargan desde CONAPESCA — `https://conapesca.gob.mx/wb/cona/avisos_arribo_cosecha_produccion`. **Implica**: agregar un nuevo extractor `extract/arribos_conapesca.py` además del lector COBI legacy, posiblemente con scraping si los archivos cambian de formato/URL. Verificar en Fase 1.2 que el esquema de CONAPESCA es compatible con el de COBI 2017-2021 (campos clave: fecha, especie, UE/permisionario, peso desembarcado).
2. **Credenciales GlobColour/Copernicus**: Javier las gestiona. Bloqueador propio del usuario, no del código.
3. **Coordenadas TURF**: las tiene COBI (Javier las puede compartir). Pendiente: ruta exacta del shapefile o tabla. Mientras tanto el código debe leer de `configs/economic_units.yaml` y soportar tanto bbox como polígono GeoJSON cuando esté disponible.
4. **Calendarios de temporada por `(especie, UE)`**: no existe una tabla oficial accesible. **Estrategia**: en `configs/season_calendars.yaml` declarar solo lo que sabemos con certeza (langosta roja en SQ: 09-15→02-15). Para el resto, default `in_season=True` siempre y emitir warning en QC. Ir afinando con el reglamento SINE/INAPESCA conforme avance la tesis.
5. **Especies en `dataset_v1`**: las 5 con ≥500 observaciones en el consolidado 2017-2021 (langosta_roja, abulón_azul, abulón_rojo, abulón_negro, erizo_rojo). Confirmado.
6. **SST para MHW**: **NOAA OISST v2.1** (decisión final). Razones: licencia abierta, baseline 1982-presente sin huecos, no depende de credenciales Copernicus, paquete `xarray` lo lee directo de NCEI THREDDS o ERDDAP.
7. **Estructura nueva**: `src/fishing_forecast/`. El código viejo (`etl/`, `forecasting_models/`) queda como legacy hasta validar el pipeline nuevo, luego se archiva en `legacy/`.

Decisiones pendientes que se documentarán como ADR cuando se implementen:

- **§4.4** `y=NaN` en temporada — mantener NaN (no imputar). ADR pendiente.
- **§8** Versionado del dataset — bump manual `v1→v2` + CHANGELOG. Sin DVC por ahora.

---

## 12. Criterios de éxito de Fase 1.1

El diseño está completo cuando:

- [x] Este documento cubre las fuentes, el esquema y el flujo.
- [ ] Las preguntas §11 #1-#7 están respondidas o aceptadas como limitación conocida.
- [ ] Un ingeniero nuevo puede leer este doc + `repo_audit.md` y entender qué hay que construir en Fase 1.2 sin leer el código del borrador.
