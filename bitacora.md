# bitacora.md

Registro cronológico del trabajo en `fishing_forecast`. Cada entrada referencia los commits y archivos tocados, con suficiente contexto para retomar sin tener que recorrer el historial completo de Claude.

---

## 2026-04-29 — Fase 0 (auditoría) + Fase 1.1 (diseño) + bootstrap + extractor CONAPESCA

### Sesión completa, 4 hitos

#### 1. Fase 0 — Reconocimiento del repo

**Entregable**: [`docs/repo_audit.md`](docs/repo_audit.md) (9 secciones, ~240 líneas).

Auditoría del estado heredado del borrador 2024 (commit base `1f3aa03`):

- Inventarié código existente en `etl/` (raíz) y `forecasting_models/` (raíz). Confirmé que no hay tests, ni `pyproject.toml`, ni `Makefile`, ni `.env`, y que los tres `requirements.txt` se contradicen (pandas 1.3 vs 1.5 vs 2.1; tensorflow 2.7).
- Detecté **rutas hardcodeadas de Windows/Colab** en 4 scripts de modelos (`path_to_data/`, `C:/Users/javi2/...`, `/content/drive/MyDrive/Tesis/...`).
- Detecté **credenciales Postgres "admin"** hardcodeadas en `etl/load/globcolour_load.py:62` y `etl/load/cicese_load.py:19`.
- Detecté que `etl/load/google_earth_load.py`, `etl/load/copernicus_load.py` y `etl/transform/google_earth_transform.py` **están vacíos** (1 línea).
- Detecté un **bug en `etl/transform/dataset_merger.py`**: itera `dataset['date'].unique()` y construye DataFrame con scalar+Series por broadcast frágil; salida probablemente incorrecta.
- Detecté que **`forecasting_models/modeling/code_wandb.py` no es del proyecto** (es un ejemplo CIFAR10/ResNet18 de ITAM).
- Detecté **inconsistencia en el corte de test**: CLAUDE.md decía `2021-06-01` pero ARIMA/LGBM/XGB usan `2020-07-01`, Prophet usa `2021-01-01`, LSTM usa un split por ratio.
- Detecté que `motuclient` está deprecado desde marzo 2024 (Copernicus migró a `copernicusmarine`).

**Decisiones tomadas con el usuario** (Javier):
- Corte canónico de test = **`2020-07-01`** (lo que realmente usaron los scripts).
- Datos 2022-2025 **aún no existen**; hay que descargarlos (ver §4 abajo).
- Credenciales GlobColour/Copernicus **hay que regenerarlas**.
- Artefactos del borrador (joblib XGB, `.h5` LSTM) viven en un bucket S3, se consultan después.

**Cambios en `CLAUDE.md`**:
- Sección "Estado inicial (borrador 2023)" → corte de test corregido a `2020-07-01`.
- Sección "Expansión 2026" → ahora refleja que datos 2022-2025 no existen, credenciales por regenerar, artefactos en S3.

**Memorias guardadas** (en `~/.claude/projects/-Users-javierorcazas-Documents-fishing-forecast/memory/`):
- `project_cut_date.md` — corte canónico `2020-07-01`.
- `project_data_state.md` — estado de datos/credenciales/artefactos a abr-2026.

---

#### 2. Fase 1.1 — Diseño del ETL

**Entregable**: [`docs/etl_design.md`](docs/etl_design.md) (~500 líneas, 12 secciones).

Pipeline diseñado: `raw/ (inmutable) → interim/ (long-tidy parquet por fuente) → aggregate/ (bbox-mean por UE + MHW) → processed/dataset_vN.parquet (particionado por species×year)`.

**Decisiones de diseño (todas reversibles, registradas en §11)**:
1. Una fila por `(ds, species, economic_unit)` — soporta el modelo jerárquico de Fase 3 sin re-ETL.
2. Shift de 3 meses **NO** en ETL; va en feature engineering. ETL guarda `x_i` alineado con `ds`.
3. `y=NaN` en temporada se mantiene (no se imputa) — flag-friendly para modelos que manejan missing.
4. SST anomaly siempre guardada; `mhw_intensity` solo durante eventos activos (dos columnas, no una sobrecargada).
5. Parquet zstd, particionado por `species × year(ds)`.
6. Migración a `copernicusmarine`.
7. Fuente recomendada de SST para MHW: **NOAA OISST v2.1** (abierto, baseline 30 años estable).

**Preguntas resueltas con el usuario**:
1. Arribos 2022-2025 → CONAPESCA (`https://conapesca.gob.mx/wb/cona/avisos_arribo_cosecha_produccion`).
2. Credenciales GlobColour/Copernicus → Javier las gestiona.
3. Coordenadas TURF → COBI las tiene; Javier las comparte cuando lleguemos a Fase 1.2 plena.
4. Calendarios de temporada → solo declaramos lo que conocemos (langosta-SQ); resto default `in_season=True` con warning en QC.
5. 5 especies en `dataset_v1`: lobster_red, abalone_blue, abalone_red, abalone_black, urchin_red.
6. SST para MHW: NOAA OISST v2.1.
7. Estructura nueva en `src/fishing_forecast/`; legacy queda intocado hasta validar pipeline nuevo.

**Memoria guardada**: `project_etl_decisions.md`.

---

#### 3. Bootstrap del paquete `fishing_forecast`

**Commit**: `e12e08e` — "Bootstrap del paquete fishing_forecast (Fase 1.1)" (44 archivos, +7749 líneas).

Estructura nueva creada:

```
fishing_forecast/
├── pyproject.toml              # hatchling, Python 3.11+, ruff, pytest, mypy
├── uv.lock                     # lockfile (commiteado para reproducibilidad)
├── .env.example                # plantilla de credenciales
├── README.md                   # actualizado: setup con uv, estructura, doc links
├── configs/                    # 7 YAMLs:
│   ├── etl.yaml                # orquestación, 5 especies, params MHW
│   ├── economic_units.yaml     # UEs con bbox (placeholder hasta shapefile COBI)
│   ├── species_mapping.yaml    # texto crudo COBI/CONAPESCA → código snake_case
│   ├── season_calendars.yaml   # solo langosta-SQ por ahora
│   ├── globcolour_vars.yaml    # mapeo x1..x16 → variable real
│   ├── cicese_stations.yaml    # Isla Cedros, Guerrero Negro
│   └── copernicus_vars.yaml    # producto SST L4 inicialmente
├── src/fishing_forecast/
│   ├── __init__.py             # version + truststore.inject_into_ssl()
│   ├── config.py               # Settings con pydantic-settings
│   ├── cli.py                  # `fishing-etl` con typer
│   ├── etl/{extract,transform,aggregate}/
│   ├── features/
│   └── utils/dates.py          # season_id + in_season
├── tests/
│   ├── conftest.py             # repo_root, fixtures_dir
│   └── test_smoke.py           # 5 tests del bootstrap
└── docs/{decisions,thesis_sections}/.gitkeep
```

**`.gitignore` actualizado**: ignora `data/{raw,interim,processed}/**`, `models/**`, `reports/{figures,metrics,etl}/**`, `*.nc`, `*.dat`, `*.joblib`, `*.h5`, `*.parquet`. Sí commitea `reports/sessions/**` y `uv.lock`.

**Bug encontrado y corregido en `utils/dates.py`**: la implementación inicial de `season_id` solo tomaba `start_month, start_day` y para días en el "gap" entre temporadas devolvía la temporada **anterior** (recién terminada), no la **próxima**. Para el desplazamiento de 90 días que usa el modelo (features de junio→temporada que arranca en septiembre), la semántica útil es "asignar a la próxima temporada". Refactoricé para tomar `start_month/day, end_month/day` y distinguir tres casos: (a) cross-year season en curso, (b) carry-over del año anterior, (c) gap entre temporadas. Test añadido para los tres.

**Verificación**:
- `uv sync --extra dev` → 41 paquetes instalados.
- `uv run pytest` → 5/5 verde.
- `uv run ruff check src tests` → All checks passed.
- `uv run fishing-etl info` → muestra rutas correctas.

---

#### 4. Fase 1.2 — Extractor CONAPESCA

**Commit**: `51796f2` — "Extractor CONAPESCA + fix SSL gob.mx (Fase 1.2)" (7 archivos, +532 líneas).

**Exploración previa**: descubrí que CONAPESCA publica **2 CSVs anuales × 9 años (2018-2026) = 18 archivos**:
- `arribo_cosecha` — capturas + acuacultura, ~150 MB c/u (los relevantes para `y`).
- `produccion` — procesamiento, ~150 MB c/u (secundarios, valor para fases futuras).

**Irregularidades de URL** que el extractor maneja:
- Pre-2025: filename con **espacio literal** (`AVISOS_ MAYORES_MENORES_COSECHA_2018.csv`).
- Post-2025: path anidado (`/2025/aviso_arribo/AVISOS_MAYORES_MENORES_COSECHA_2025.csv`).

**Schema confirmado en el CSV** (encoding ISO-8859-1, 35 columnas, header en línea 5 tras 4 líneas de título/disclaimer):
- `PERIODO FIN` → `ds`
- `PESO DESEMBARCADO_KILOGRAMOS` → `y`
- `NOMBRE ESPECIE` → `species` (raw, mapear con `species_mapping.yaml`)
- `UNIDAD ECONOMICA` + `RNPA UNIDAD ECONOMICA` → `economic_unit`
- `NOMBRE ESTADO` + `LITORAL` → `region`

**Implementación** — [`src/fishing_forecast/etl/extract/arribos_conapesca.py`]:
- `parse_index_html(html)` — pura, testeable sin red. Usa BeautifulSoup+lxml. Filtra por patrones regex `_KIND_PATTERNS`. Devuelve lista de `FileSpec` ordenada por (year, kind).
- `fetch_index()` — descarga el HTML del índice y lo parsea.
- `download_file(spec, dest_dir)` — idempotente:
  1. Si existe `<file>.meta.json` con etag/last-modified/content-length → HEAD al servidor; si coincide cualquier metadato, skip.
  2. Si no, descarga con stream (chunks de 1 MiB) a archivo `.part`, rename atómico al final, escribe `.meta.json`.
  3. `force=True` ignora cache.
- `extract(years, kinds, dest_dir, force)` — orquestador.

**Fix transversal de SSL** — [`src/fishing_forecast/__init__.py`]:
- Servidor `conapesca.gob.mx` manda cadena SSL incompleta (sin intermedio "GeoTrust TLS RSA CA G1" de DigiCert). `curl` resuelve eso vía AIA chasing del SO; `requests`+`certifi` no.
- Fix: `truststore.inject_into_ssl()` al cargar el paquete. Hace que `urllib3`/`requests` usen el trust store del SO (Keychain en macOS), que sí resuelve cadenas incompletas.
- `truststore` añadido como **dependencia core** porque es relevante para CICESE también.

**CLI** — `fishing-etl extract conapesca`:
```
--years all|2018,2019,...   # filtrar por año
--kinds arribo_cosecha|produccion|all
--force                     # ignorar cache
--list-only                 # solo descubrir, no descargar
```

**Tests** — 8 nuevos en `tests/etl/test_extract_arribos_conapesca.py`:
- `parse_index_html` con fixture HTML pequeña (8 anchors, incluye uno irrelevante para verificar filtrado).
- Idempotencia del downloader con `requests.Session` mockeada (skip cuando ETag matches, force re-download, escritura de `.meta.json`).

**Verificación contra servidor real**: `fishing-etl extract conapesca --list-only` descubre los 18 archivos esperados, con las irregularidades de URL bien manejadas.

**Total tests al cierre**: 13/13 verde, ruff limpio.

---

### Estado del proyecto al cierre del 2026-04-29

| Fase del PLAN | Estado |
|---|---|
| 0. Reconocimiento | ✅ completa |
| 1.1 Diseño del ETL | ✅ completa |
| 1.2 Implementación del ETL | 🟡 en curso — extractor CONAPESCA listo |
| 1.3 Índice MHW | ⏳ pendiente |
| 1.4 Re-entrenamiento baseline | ⏳ bloqueado por dataset_v1 |

### Bloqueadores al cierre

1. **Coordenadas TURF por UE** — Javier debe compartir el shapefile/tabla de COBI antes de implementar `aggregate/ocean_by_ue.py`.
2. **Credenciales GlobColour/Copernicus** — Javier debe regenerarlas antes de los extractores correspondientes.
3. **CSV legacy `Arribos2017-2021.csv`** — necesito la ruta local antes de implementar `extract/arribos_cobi.py` (el del borrador).

### Lo no bloqueado, listo para arrancar

- **`transform/arribos.py`** — parsea los CSVs Latin-1 que ya descarga el extractor; aplica species_mapping y filtro a las 5 especies + UE de SQ.
- **`aggregate/mhw.py`** — NOAA OISST es público; testeable contra el Blob 2014-2016 sin credenciales.
- **`extract/arribos_cobi.py`** — solo necesita la ruta local del CSV.

### Commits del día

```
51796f2 Extractor CONAPESCA + fix SSL gob.mx (Fase 1.2)
e12e08e Bootstrap del paquete fishing_forecast (Fase 1.1)
```

---

## 2026-06-18 — Fase 1.2: transformación de arribos CONAPESCA

### Hito: `transform/arribos.py` (raw CSV → interim long-tidy)

Retomé el primer pendiente "no bloqueado" del cierre anterior. Implementé la
transformación que limpia los CSV crudos que descarga el extractor CONAPESCA y
los lleva al parquet interim long-tidy.

**Entregable**: [`src/fishing_forecast/etl/transform/arribos.py`].

**Granularidad de salida**: una fila por `(ds, species, economic_unit)` con columnas
`ds, y, species, economic_unit, region` — coincide con §4.1 de `etl_design.md`.

**Funciones puras (testeables sin red ni archivos grandes)**:
- `normalize_text` — mayúsculas + sin acentos (NFKD) + espacios colapsados. Hace que
  `"ABULÓN AZUL"` y `"ABULON AZUL ENT. FCO."` casen aunque CONAPESCA y COBI escriban
  distinto entre años.
- `build_species_lookup` / `build_ue_lookup` — `{alias_normalizado → code}` desde
  `species_mapping.yaml` y `{nombre_UE_normalizado → (code, region)}` desde
  `economic_units.yaml`. La `region` canónica se deriva del mapping de UE (no de
  `NOMBRE ESTADO`/`LITORAL`), consistente con el diseño.
- `read_conapesca_csv` — lee ISO-8859-1 saltando 4 líneas de preámbulo; valida que
  existan las 4 columnas clave y revienta con mensaje claro si no (separador/encoding
  equivocado).
- `clean_arribos` — mapea, filtra a `keep_species`/`keep_units`, parsea `ds` (dayfirst)
  y `y` (float kg), descarta filas sin mapeo o sin fecha (con conteo), agrega sumando
  `y` por `(ds, species, economic_unit, region)`. **No imputa ni mete ceros.**
- `transform` — orquesta varios CSV, re-agrega para de-duplicar solapamientos entre
  años, y escribe parquet zstd a `data/interim/arribos.parquet`.

**Decisiones / supuestos pendientes de validar contra el archivo real (150 MB, no en repo)**:
- Separador asumido **coma** (`sep=","` overridable). Si CONAPESCA usa `;` o `|`, el
  validador de columnas faltantes lo detecta en la primera corrida.
- `PERIODO FIN` parseado con `dayfirst=True` (formato MX `DD/MM/YYYY`); también overridable.
- Filtro por defecto: `dataset_v1_species` (5 especies) + UEs definidas en
  `economic_units.yaml` (hoy solo `litoral_bc_sur`). Flags `--all-species` / `--all-units`
  para soltar el filtro cuando lleguen más UEs en Fase 3.

**CLI**: `fishing-etl transform arribos` (nuevo subgrupo `transform`), descubre los CSV
en `data/raw/arribos/conapesca/arribo_cosecha/`, carga configs y escribe el interim.

**Tests** — 9 nuevos en `tests/etl/test_transform_arribos.py` + fixture
`tests/fixtures/conapesca_arribos_sample.csv` (ISO-8859-1, preámbulo de 4 líneas,
incluye una especie sin mapeo `TIBURON` y una UE sin mapeo para verificar descarte;
dos filas mismo día/UE/especie para verificar agregación; alias con y sin acento).

**Verificación**: `uv run pytest` → 22/22 verde. `ruff check` + `ruff format` limpios.

### Estado al cierre del 2026-06-18

| Fase del PLAN | Estado |
|---|---|
| 0. Reconocimiento | ✅ |
| 1.1 Diseño del ETL | ✅ |
| 1.2 Implementación del ETL | 🟡 extractor CONAPESCA + transform/arribos listos |
| 1.3 Índice MHW | ⏳ no bloqueado (NOAA OISST público) |
| 1.4 Re-entrenamiento baseline | ⏳ bloqueado por dataset_v1 |

### Próximo paso concreto (no bloqueado)

- **`aggregate/mhw.py`** — NOAA OISST v2.1 es público; calcular categoría MHW (Hobday
  2016) y testear contra el Blob 2014-2016 sin credenciales. Es el siguiente eslabón
  que no depende de Javier (coordenadas TURF / credenciales / CSV legacy COBI).

---

## 2026-06-19 — Fase 1.3: índice MHW (Hobday 2016/2018)

### Hito: `etl/aggregate/mhw.py` (implementación propia, pura respecto a la fuente de SST)

`marineHeatWaves` (port de Oliver) no instala en el entorno → implementación propia
basada en el paper, como ya preveía `etl_design.md` §5.3. Vive en `etl/aggregate/`
(no en `features/` como decía el PLAN original) porque MHW es **columna del dataset
consolidado**, no un feature de modelado de Fase 2.

**Diseño clave**: la función pública `add_mhw(daily_df, params)` recibe una serie
diaria de SST de **una sola UE** y devuelve las columnas del esquema. No sabe de dónde
viene la SST → desacoplado de la extracción oceanográfica (que sigue bloqueada por
credenciales/coords). Esto permite implementar y testear MHW **ahora**.

**Algoritmo**:
- `compute_climatology` — climatología diaria sobre baseline (default 1982-2011):
  para cada día-del-año agrupa SST en ventana ±5d a través de los años, calcula media
  (`clim`) y p90 (`thresh`), y suaviza ambos con media móvil **circular** de 31d.
- Día-del-año en **rejilla fija de 366** anclada a un año bisiesto de referencia (2000),
  para que 1-mar=61 siempre y no se desfase entre años bisiestos/no-bisiestos (problema
  del `dayofyear` crudo de pandas). `year_day()` expuesto y testeado.
- `add_mhw` — reindexa a rango diario continuo (los huecos de calendario rompen la
  consecutividad correctamente), detecta corridas de `SST≥thresh`, **fusiona eventos
  separados por huecos ≤2d** (Hobday), descarta eventos < 5 días, y categoriza por
  `(SST-clim)/(thresh-clim)`: [1,2)→I, [2,3)→II, [3,4)→III, ≥4→IV.

**Columnas de salida** (esquema §4.1 + decisión §5.4):
- `sst_anomaly` — `SST-clim`, siempre (incluso negativa).
- `mhw_category` — int8 0..4; 0 fuera de evento, ≥1 dentro (los días-hueco fusionados,
  por debajo del umbral, quedan en categoría 1).
- `mhw_intensity` — `sst_anomaly` dentro de evento, `NaN` fuera.
- Con `return_diagnostics=True` añade `clim`, `thresh`, `in_mhw` (para la figura).

**Config**: agregué `window_half_width_days: 5` y `max_gap_days: 2` explícitos a
`configs/etl.yaml`, y **cambié `smoothing_window_days` de 11 → 31** (default de Hobday;
antes el 11 conflaba la ventana de pooling con la de suavizado). `MHWParams.from_config`
mapea el bloque `mhw:` del YAML.

**Tests** — 9 nuevos en `tests/etl/test_mhw.py` con series sintéticas (sinusoide
estacional determinista, sin datos reales): alineación de día-del-año (bisiestos),
bandas de categoría, umbral ≥ media, cero MHW en climatología pura, ola inyectada de
10 días detectada, pico de 3 días ignorado (< min_duration), fusión de hueco de 1 día,
preservación de filas con huecos de entrada, y `from_config`.

**Verificación**: `uv run pytest` → 31/31 verde. `ruff check` + `ruff format` limpios.

**Pendiente para cerrar Fase 1.3** (ambos requieren la SST real, fuera de lo no-bloqueado):
1. Wirear un extractor de NOAA OISST v2.1 + agregación bbox por UE (`aggregate/ocean_by_ue.py`).
2. `reports/figures/mhw_timeline.png` con el Blob 2014-2016 y el régimen 2019-2021.

### Estado al cierre del 2026-06-19

| Fase del PLAN | Estado |
|---|---|
| 0. Reconocimiento | ✅ |
| 1.1 Diseño del ETL | ✅ |
| 1.2 Implementación del ETL | 🟡 CONAPESCA extract + transform/arribos + algoritmo MHW |
| 1.3 Índice MHW | 🟡 algoritmo listo y testeado; falta SST real + figura |
| 1.4 Re-entrenamiento baseline | ⏳ bloqueado por dataset_v1 |

### Próximo paso (ya con dependencia externa)

El siguiente eslabón realista es **`extract/sst_oisst.py` + `aggregate/ocean_by_ue.py`**:
NOAA OISST es público (no necesita credenciales), pero implica descarga grande y, para
el promedio por UE, las **coordenadas TURF de COBI**. Confirmar con Javier si bajamos
OISST global (bbox SQ) o esperamos el shapefile. Lo de GlobColour/Copernicus sigue
bloqueado por credenciales.

---

## 2026-06-19 (cont.) — Vertical slice oceanográfico: OISST → SST por UE → MHW

Decidí avanzar con el bbox **placeholder** de San Quintín que ya vive en
`economic_units.yaml` (lon -117..-115, lat 30..31.5), en vez de esperar el shapefile
TURF de COBI: el promedio sobre ese bbox es una primera aproximación razonable y el
shapefile solo afina el recorte después. Así desbloqueo todo el camino OISST→MHW.

**No corrí la descarga real** (OISST son ~150 MB/año × 30+ años; CLAUDE.md pide
confirmar antes de operaciones largas). Todo quedó implementado y testeado con datos
sintéticos + un roundtrip netCDF chico.

#### `etl/extract/sst_oisst.py`

Extractor de NOAA OISST v2.1 high-res (PSL): un netCDF anual `sst.day.mean.<YYYY>.nc`.
- `build_specs(years)` — puro, ordena/deduplica, rechaza años < 1982.
- `download_file` / `extract` — mismo patrón idempotente que CONAPESCA (cache
  ETag/Last-Modified/Content-Length en `.meta.json`, descarga stream a `.part` + rename
  atómico). Reusa el `truststore` global (ya inyectado en `__init__`).

#### `etl/aggregate/ocean_by_ue.py`

- `sst_bbox_mean(dataset, bbox)` — **puro sobre un `xarray.Dataset`**. Recorta al bbox y
  promedia espacialmente (skipna, ignora celdas de tierra) → serie diaria `(ds, sst)`.
  **Maneja la convención de longitud**: OISST usa 0-360 y los bbox del repo son -180..180;
  detecta la convención del dataset y convierte el bbox (incluido el wrap en el
  antimeridiano). Detección flexible de nombres de coords (lat/latitude, lon/longitude,
  time/date).
- `open_oisst(paths)` — aísla la lectura de disco (`open_dataset` / `open_mfdataset`).
- `sst_series_for_bbox` / `sst_mhw_for_bbox` — encadenan lectura → bbox-mean → `add_mhw`.

#### CLI

- `fishing-etl extract oisst --years 1982-2011` (default = baseline climatológico MHW;
  acepta rango `YYYY-YYYY` o lista coma-separada). Avisa del tamaño antes de bajar.
- `fishing-etl aggregate ocean --ue litoral_bc_sur` — lee los netCDF descargados, toma el
  bbox de la UE y los params MHW de `etl.yaml`, y escribe
  `data/interim/ocean_<ue>.parquet` con `sst, sst_anomaly, mhw_category, mhw_intensity`.

#### Config

Agregué el bloque `sources.oisst` a `configs/etl.yaml` (base_url + download_dir).

#### Tests — 10 nuevos

- `test_extract_sst_oisst.py` (5): URLs anuales, rechazo de años < 1982, escritura de
  archivo+meta, idempotencia con HEAD/ETag mockeado, `--force` re-descarga.
- `test_ocean_by_ue.py` (5): bbox-mean selecciona las celdas correctas en convención
  -180..180 **y** 0-360, bbox fuera del grid → NaN + warning, roundtrip netCDF real
  (escribe `.nc` con xarray y reabre), y `sst_mhw_for_bbox` end-to-end (3 años sintéticos,
  ola inyectada en 2002 detectada como MHW).

**Verificación**: `uv run pytest` → 41/41 verde. `ruff check` + `ruff format` limpios.

### Estado al cierre

| Fase del PLAN | Estado |
|---|---|
| 0 / 1.1 | ✅ |
| 1.2 Implementación del ETL | 🟡 CONAPESCA + transform/arribos + OISST extract + ocean_by_ue (SST) |
| 1.3 Índice MHW | 🟡 algoritmo + pipeline SST→MHW listos; falta correr descarga real + figura |
| 1.4 Re-entrenamiento baseline | ⏳ bloqueado por dataset_v1 |

### Decisión que necesita a Javier

Para **correr de verdad** el pipeline oceanográfico hay que bajar OISST (decidir rango:
1982-2011 baseline + 2012-2025 operativo ≈ 44 archivos × ~150 MB). Confirmar antes de
disparar la descarga. El bbox usado es placeholder; el shapefile TURF de COBI lo afina
sin re-ETL (solo cambia `economic_units.yaml`).

### Pendientes no bloqueados que siguen

- `transform/cicese.py` — scraping de estaciones CICESE (Isla Cedros, Guerrero Negro);
  útil para validar SST contra OISST (QC `sst_cicese_correlation_min`).
- `extract/arribos_cobi.py` — lector del CSV legacy 2017-2021 (necesita la ruta local).
- `consolidate.py` + `quality_checks.py` — una vez haya ≥2 fuentes en interim.

---

## 2026-06-19 (cont.) — Cierre del camino de código del ETL: consolidate + quality_checks

Petición: "terminar el plan". El plan completo (Fases 1-5) no se puede *terminar* sin
datos reales y credenciales (bloqueadores externos), así que llevé el **código del ETL
hasta el final del camino** (extract→transform→aggregate→**consolidate→quality_checks**)
y dejé todo lo demás mapeado en `PENDINGS.md`.

#### `etl/consolidate.py`

Join final al esquema §4.1 (16 columnas). El spine es `interim/arribos.parquet`; se le
pega la SST/MHW por UE (`interim/ocean_<ue>.parquet`), **broadcasteada** a todas las
especies de la UE (la oceanografía es por UE, no por especie).
- `build_grid` — rejilla completa `(ds, species, economic_unit, region)` sobre el rango
  de fechas de `etl.yaml`, con las series presentes en arribos.
- `_derive_season` — `season` + `in_season` por grupo usando `season_calendars.yaml` y
  `utils/dates`; sin calendario declarado → `in_season=True` + warning.
- Manejo de `y` (§4.4): fuera de temporada sin registro → `y=0`; dentro de temporada sin
  registro → `NaN` (no se imputa). `is_imputed_y=False` siempre.
- Metadatos: `is_imputed_x`, `ocean_impute_method`, `source_globcolour_files=0` (GlobColour
  aún no integrado), `etl_run_id`.

#### `etl/quality_checks.py`

`check_dataset` (pura → lista de `QCIssue`) + `run_quality_checks` (aplica política,
levanta `QualityCheckError`). Checks: duplicados de clave primaria, `y≥0`,
`mhw_category∈0..4`, dominios species/UE, tipos `season`/`in_season`; **warnings** de
filas fuera de temporada con `y≠0` y de cobertura SST < umbral. Nada de `except: pass`.

#### CLI

Implementé los comandos que eran `NotImplementedError`: `fishing-etl consolidate` y
`fishing-etl qc [--fail-on-warning]`.

#### Verificación end-to-end (no solo unit tests)

Corrí el pipeline real con la fixture: `transform arribos → consolidate → qc`. Produjo
`dataset_v1.parquet` con **10176 filas** (3 especies × 3392 días, 2017-01-01→2026-04-15),
las 16 columnas del esquema, y el QC marcó correctamente el warning de cobertura SST 0%
(no hay OISST descargado). Limpié los artefactos de la prueba (gitignored).

**Tests**: 56/56 verde (15 nuevos: 7 consolidate + 8 quality_checks). `ruff` limpio.

#### `PENDINGS.md` (nuevo, en la raíz)

Mapa estructurado de lo que falta para terminar el plan: bloqueadores duros (credenciales
GlobColour/Copernicus, shapefile TURF de COBI, CSV legacy COBI, artefactos S3), descargas
grandes pendientes de confirmar (CONAPESCA ~1.4 GB, OISST ~6-7 GB), código de ETL aún
desbloqueado (CICESE, particionado, ADR §4.4, export de compatibilidad), y el detalle de
Fases 1.4-5 con sus dependencias. Incluye la ruta crítica recomendada.

### Estado al cierre

| Fase del PLAN | Estado |
|---|---|
| 0 / 1.1 | ✅ |
| 1.2 Implementación del ETL | ✅ **código completo y testeado** (falta correr con datos reales) |
| 1.3 Índice MHW | 🟡 algoritmo + pipeline listos; falta OISST real + figura |
| 1.4 Re-entrenamiento baseline | ⏳ bloqueado por dataset real + artefactos S3 |
| 2-5 | ⏳ dependen de Fase 1 cerrada con datos reales — ver `PENDINGS.md` |

### Próximo paso

Lo de mayor palanca ya no es código sino **insumos**: confirmar bbox/shapefile (B3) y
rango OISST para correr el pipeline real, y regenerar credenciales (B1/B2). El siguiente
código *desbloqueado* es `transform/cicese.py` (pendiente de verificar el formato `.dat`
real). Todo en `PENDINGS.md`.

---

## 2026-06-19 (cont.) — CICESE: extractor + transformación (reescritos del legacy)

Antes de escribir nada verifiqué el formato real leyendo el legacy `etl/cicese.py`
(CLAUDE.md: no asumir). Hallazgos: índice HTML de REDMAR por estación/año lista archivos
`.dat` **sin header**, separados por espacios, **23 columnas en orden fijo** (nombres de
la metadata CICESE), agregados a mediana diaria.

#### `etl/extract/cicese.py`

- `build_index_url` / `parse_index_html` (pura, BeautifulSoup) — reemplaza el parsing
  frágil del legacy (`line.split('href="')[1][:15]`) por extracción de `<a href>` `.dat`.
- `download_file` idempotente (mismo patrón meta.json que CONAPESCA/OISST). REDMAR es
  HTTP plano, sin credenciales.
- `extract(stations, years, dest_dir)` → `{station: [paths]}`, tolera años con índice
  inaccesible (warn + skip, sin reventar).

#### `etl/transform/cicese.py`

- `CICESE_COLUMNS` (23) + `RAW_TO_AGGREGATE` (español → códigos inglés de
  `cicese_stations.yaml`).
- `read_dat` (sep `\s+`, sin header), `to_daily` (mediana por `(anio,mes,dia)`, construye
  `ds`, renombra, filtra a `daily_aggregates`, etiqueta `station`/`region`), `transform`
  (concatena `.dat` → diario → parquet `interim/cicese/<station>.parquet`).
- **Decisión consciente**: el valor centinela de dato faltante de REDMAR (¿9999?) no se
  asume — `read_dat` toma `na_values` explícito (default None). Anotado en `PENDINGS.md`
  para fijarlo cuando haya datos reales (evita sesgar la mediana con un supuesto).

#### CLI

`fishing-etl extract cicese` y `fishing-etl transform cicese` (iteran las estaciones de
`cicese_stations.yaml`).

#### Tests — 7 nuevos (`tests/etl/test_cicese.py`) + fixtures

`cicese_index_sample.html` (2 `.dat` + `../` + `readme.txt` para verificar filtrado) y
`cicese_sample.dat` (23 cols, 2 muestras/día × 2 días). Cubren: URL del índice, parse
solo `.dat`, lectura de 23 columnas, mediana diaria (18+20→19, 21+23→22), renombrado,
filtro `aggregates`, y roundtrip parquet.

**Verificación**: `uv run pytest` → 63/63 verde. `ruff` limpio.

### Estado al cierre

| Fase del PLAN | Estado |
|---|---|
| 0 / 1.1 | ✅ |
| 1.2 Implementación del ETL | ✅ código completo (CONAPESCA, OISST, **CICESE**, consolidate, qc) |
| 1.3 Índice MHW | 🟡 algoritmo + pipeline listos; falta OISST real + figura |
| 1.4 / 2-5 | ⏳ ver `PENDINGS.md` |

Fuentes de código que quedan: GlobColour/Copernicus (bloqueadas por credenciales) y el
lector legacy COBI (bloqueado por la ruta del CSV). Pendientes finos de CICESE (centinela
NaN, check de correlación SST) y el refactor de los 3 descargadores idempotentes en
`PENDINGS.md`.

---

## 2026-06-20 — Runbook + Etapa 2 de PENDINGS (código desbloqueado)

Javier pidió (a) el paso a paso para conseguir credenciales/insumos y cerrar pendientes,
y (b) arrancar los items de código desbloqueados.

#### (a) `docs/SETUP_AND_RUNBOOK.md`

Runbook en dos partes: **A** = cómo conseguir cada insumo externo (GlobColour FTP en
hermes.acri.fr, Copernicus Marine + SDK, shapefile TURF de COBI, CSV legacy, S3) con las
variables de `.env.example`; **B** = orden de ejecución de los pendientes (descargas →
pipeline → código desbloqueado → enriquecimiento → modelado) marcando [tú] vs [claude].
Incluye ruta crítica.

#### (b) Etapa 2 — todo lo desbloqueable sin insumos externos

1. **Correlación SST CICESE vs OISST** — `quality_checks.check_sst_correlation` (Pearson
   sobre el solape diario; warning bajo el umbral o con solape < 30 días). 4 tests.
2. **Figura MHW** — `viz/mhw_plot.plot_mhw_timeline` (SST + climatología + umbral +
   eventos sombreados por categoría Hobday). Backend Agg. 2 tests (smoke PNG + validación
   de columnas diagnósticas). Movió `matplotlib` al extra `etl` (era solo `models`).
3. **Particionado** `consolidate.write_dataset_partitioned` (species×year), **export**
   `consolidate.export_lstm_csv` (compat borrador: `ds,y[,x1..x16]`), **ADR-0001**
   (y-missing), y **refactor**: el patrón de descarga idempotente se factorizó a
   `utils/download.py` y los 3 extractores quedaron como wrappers delgados.

#### Bug encontrado y corregido (a raíz del runbook)

Al copiar `.env.example` → `.env`, las rutas venían como `DATA_ROOT=` (vacías) y pisaban
los defaults (el smoke test reventó: `data_root.name == ''`). Mi propio runbook (`cp
.env.example .env`) habría brickeado la config. Fix: (1) `field_validator(mode="before")`
en `config.py` que trata string vacío como ausente y usa el default; (2) `.env.example`
ahora trae las rutas comentadas con la nota.

**Verificación**: `uv run pytest` → 71/71 verde. `ruff check` + `ruff format` limpios.

### Estado al cierre

| Fase del PLAN | Estado |
|---|---|
| 0 / 1.1 / 1.2 | ✅ |
| 1.3 Índice MHW | 🟡 algoritmo + pipeline + figura listos; falta OISST real para generar el PNG |
| 1.4 / 2-5 | ⏳ ver `PENDINGS.md` |

Toda la Etapa 2 de `PENDINGS.md` (código desbloqueado) está cerrada. Lo que sigue
requiere insumos externos: confirmar shapefile/bbox y rango OISST para correr el pipeline
real, y credenciales GlobColour/Copernicus para el enriquecimiento.

---

## 2026-06-21 — Ingesta del export COBI (B4 resuelto) + primer `dataset_v1` real

Javier entregó `data/raw/arribos/Arribos2017-2021.csv` (97k filas; **realmente 2016-2025**,
no solo 2017-2021). Inspeccioné estructura (sin pegar datos de pescadores): mismo esquema
lógico que CONAPESCA pero **snake_case minúsculas, UTF-8, sin preámbulo, fechas ISO**.
Es un export ya pre-parseado de CONAPESCA. UE objetivo presente (5594 filas), 368 UEs
distintas.

#### Generalización a dialectos (en vez de duplicar el módulo)

Refactoricé `transform/arribos.py` para soportar **dos dialectos** con la misma lógica:
- `ArribosDialect` (columnas + encoding + preámbulo + separador + dayfirst).
- `CONAPESCA_DIALECT` (ISO-8859-1, 4 líneas, `PERIODO FIN`, DD/MM/YYYY) y `COBI_DIALECT`
  (UTF-8, 0 preámbulo, `periodo_fin`, ISO).
- `read_conapesca_csv` quedó como wrapper de compatibilidad (tests viejos verdes);
  `read_source_csv(path, dialect)` es el lector general. `clean_arribos`/`transform`
  toman `dialect`. CLI: `transform arribos --source {conapesca,cobi}` (una sola salida
  `interim/arribos.parquet`, fuente seleccionable).

#### Bug de config corregido

`species_mapping.yaml` mapeaba erizo con el alias `"ERIZO ROJO"`, pero el crudo solo trae
`"ERIZO ROJO ENT. FCO."` → urchin_red habría mapeado **0 filas**. Agregué la forma
"ENT. FCO." (y la morada). Decisión de dominio anotada: solo se mapean formas "entero";
las formas de producto (S.C., COLAS DE, CARNE DE, COCIDA) se descartan para no mezclar
bases de peso.

#### Pipeline real corrido (rápido, 97k filas)

`transform arribos --source cobi` → 842 filas tidy (5 especies dataset_v1 × UE SQ) →
`consolidate` → `dataset_v1.parquet` (13568 filas de rejilla, 2017-01-01→2026-04-15) →
`qc` OK con 2 warnings no bloqueantes (cobertura SST 0% — sin OISST aún; 1 arribo de
langosta fuera de temporada). **Validación clave**: las sumas por temporada de langosta-SQ
reproducen el **bache post-MHW**: 2019_2020 ≈173 t → 2020_2021 ≈106 t → **2021_2022 ≈31 t**
(caída ~82% vs el pico), justo lo que documenta Villaseñor-Derbez 2024.

#### Tests

+2 (`COBI_DIALECT` lectura y end-to-end con fixture UTF-8 `cobi_arribos_sample.csv`).
**73/73 verde**, `ruff` limpio. Los artefactos reales en `data/` quedan (gitignored) para
que Javier los use.

### Estado al cierre

| Fase | Estado |
|---|---|
| 0 / 1.1 | ✅ |
| 1.2 ETL | ✅ código completo; **arribos reales ya fluyen (COBI)** |
| 1.3 MHW | 🟡 algoritmo+pipeline+figura listos; falta OISST real para el PNG |
| 1.4 baseline | ⏳ **desbloqueado en datos de arribos**: ya hay `dataset_v1` real langosta-SQ; falta enriquecer con oceanografía (OISST/GlobColour) y comparar vs S3 |

### Pendientes de datos nuevos (en `PENDINGS.md` §3)

Estrategia de unión CONAPESCA+COBI, formas de producto excluidas, hueco de langosta 2022+
en SQ (¿la UE dejó de reportar?), y la fila fuera de temporada del QC.

---

## 2026-06-21 (cont.) — S3 legacy vía keys.json (B5)

Javier pidió que S3 use `keys.json` (lo agregó al `.gitignore` como `*keys.json`). El
archivo ya existe en la raíz con `aws_access_key_id` / `aws_secret_access_key`.

- **`config.py`**: nuevo `keys_file` (default `keys.json`) + `Settings.load_keys()` (lee el
  JSON o `{}` si no existe; **nunca se loguea** — trae secretos). Centraliza la carga de
  credenciales AWS fuera de `.env`.
- **`etl/extract/s3_legacy.py`**: `build_client` (boto3 con creds de keys.json o cadena
  default), `resolve_bucket` (override > `keys.json['bucket']` > `S3_BUCKET_LEGACY`),
  `list_artifacts` (paginado), `download_artifact` (idempotente por tamaño), `sync`.
- **CLI**: `fishing-etl extract s3-legacy [--list-only] [--prefix ...] [--bucket ...]`
  (descarga a `models/legacy/`). Bucket faltante → `BadParameter` claro.
- **Dependencia**: `boto3>=1.34` declarado (ya venía transitivo por copernicusmarine).
- **Tests**: 8 nuevos con cliente boto3 **mockeado** (sin tocar AWS, sin imprimir llaves):
  load_keys, prioridad de bucket, paginación de listado, skip por tamaño, descarga, sync.

**Verificación**: `uv run pytest` → 81/81 verde. `ruff` limpio.

**Falta para usarlo de verdad**: el **bucket** (agregar `"bucket"` a `keys.json` o
`S3_BUCKET_LEGACY` en `.env`) y correr `extract s3-legacy --list-only`. No lo corrí porque
no conozco el bucket y no leo el `.env`/secretos del usuario.

**Actualización (mismo día)**: Javier agregó el bucket a `keys.json`. El listado funcionó:
12 objetos ≈ 21 GB — un `.h5` suelto (LSTM, 2.8 GB) y **11 zips `Tesis-*.zip` (~18 GB)** que
parecen un dump tipo Google Takeout de la carpeta de tesis (con el XGB joblib y las métricas
adentro). Por decisión de Javier descargué **solo el LSTM** `lstm_model_23-005.h5` a
`models/legacy/` (gitignored): 2795.98 MB, coincide exacto con S3, firma HDF5 válida. Los
zips quedan sin bajar (no vale la pena 18 GB para extraer un joblib; confirmar si los tiene
local). Para *cargar* el `.h5` hará falta TensorFlow/Keras (el borrador usaba tf 2.7; no está
en deps) — se resuelve al llegar a Fase 1.4.

---

## 2026-06-21 (cont.) — Fase 1.4 arrancada: métricas + script de baseline (no corrido)

Empecé el re-entrenamiento del baseline. Javier interrumpió la ejecución del experimento y
pidió **dejar lo pendiente en `PENDINGS.md`**, así que el código quedó listo y testeado pero
**no corrí el entrenamiento ni regeneré datos**.

- **`src/fishing_forecast/evaluation/metrics.py`**: `mae`, `rmse`, `smape` (simétrico,
  acotado [0,200], 0/0→0), `season_sum_percentage_error` y `season_sum_errors` (error de
  suma de temporada, la métrica que le importa a COBI). 7 tests.
- **`experiments/exp1_baseline_retrain/baseline.py`** (reproducible, un comando): carga
  `dataset_v1`, filtra langosta-SQ, arma la serie diaria (NaN in-season→0 *solo para
  modelar*, recorta al último día con captura), corte canónico **2020-07-01**, ajusta
  **ARIMA** (rejilla chica por AIC, no la 50x50x50 del borrador) y **Prophet** (si está
  instalado; si no, se omite con warning). Escribe métricas JSON, figura pred-vs-real y
  `exp1_summary.md`. Decisión de relleno NaN→0 documentada (consistente con ADR-0001 y con
  cómo el borrador sumaba la temporada).
- **statsmodels/joblib** instalados con `uv pip install` (⚠️ no quedaron en `uv.lock` —
  pendiente fijarlos con `uv sync --extra models`). Prophet aún no instalado.

**Verificación**: `ruff` limpio, `uv run pytest` → **88/88 verde** (sin correr training ni
tocar datos).

**Pendiente (en `PENDINGS.md` §4)**: correr el experimento, instalar Prophet, fijar deps en
el lock, comparar vs el paper, y los modelos con covariables (LGBM/XGBoost/LSTM) que esperan
la oceanografía. El corte 2024-06-01 no aplica (no hay langosta-SQ tras 2022).

### Estado al cierre

| Fase | Estado |
|---|---|
| 0 / 1.1 / 1.2 | ✅ |
| 1.3 MHW | 🟡 algoritmo+pipeline+figura listos; falta OISST real |
| 1.4 baseline | 🟡 métricas + script ARIMA/Prophet escritos y testeados; **falta correrlos** |
| 2-5 | ⏳ dependen del dataset enriquecido — ver `PENDINGS.md` |

### Actualización (mismo día): Javier corrió el baseline

`uv run python experiments/exp1_baseline_retrain/baseline.py` (con Prophet ya instalado).
train=1277 días, test=549 días (2020-07-01 →). ARIMA mejor orden (3,0,3) por AIC.

| modelo | MAE | RMSE | sMAPE% | suma temp. 2020-21 | suma temp. **2021-22** (crash) |
|---|---|---|---|---|---|
| ARIMA(3,0,3) | 345 | 463 | 138% | −26% | **+291%** |
| Prophet | 372 | 576 | 129% | +62% | **+418%** |

**Hallazgo central** (justifica el proyecto): ambos baselines solo-`y` sobrepredicen la
temporada 2021-2022 por 3-4x (real 31 t; predicen 120-160 t) porque no "ven" la MHW que
colapsó la captura. El error de suma de temporada que le importa a COBI solo bajará con el
índice MHW + covariables oceanográficas (el ensamble del borrador lograba ~8.7%). Estos
números son el **piso** contra el cual medir las fases siguientes.

Artefactos: `reports/metrics/exp1_baseline_{arima,prophet}_2020-07-01.json`,
`reports/figures/exp1_baseline_pred_vs_real.png`, `reports/exp1_baseline_summary.md`.
Pendiente: fijar `statsmodels`/`joblib`/`prophet` en `uv.lock` (se instalaron con `uv pip
install`). Fase 1.4 sigue 🟡 hasta tener los modelos con covariables.

### Fix de dependencias (mismo día)

Javier corrió `uv sync --extra models`, que **desinstaló el extra `etl`** (lxml, bs4,
xarray, netCDF4, requests) y rompió la CLI (`extract conapesca` → `FeatureNotFound: lxml`).
Causa raíz: las deps runtime del ETL vivían en un extra opcional, así que cualquier
`uv sync` sin `--extra etl` dejaba la CLI rota; además core tenía el shim `bs4` pero no el
parser `lxml`.

**Fix durable**: moví requests, beautifulsoup4, lxml, xarray, netCDF4, matplotlib al
**core** de `pyproject.toml` (el ETL es la función principal del paquete). El extra `etl`
queda solo con openpyxl/pandera; quité el matplotlib duplicado de `models` y el
`copernicusmarine` duplicado del etl. Re-sync `uv sync --extra models --extra dev` →
reconcilia env + `uv.lock` (también fija statsmodels/prophet/joblib). README actualizado.

**Verificación**: imports ETL OK, `fishing-etl extract conapesca --list-only` lista los 18
archivos, `uv run pytest` → 88/88 verde. Un `uv sync` pelón ya deja la CLI funcional.

> Nota: `uv sync` **no** edita `pyproject.toml`, solo el `.venv` y (si cambió pyproject)
> el `uv.lock`. Por eso correrlo no pisó mis ediciones.

---

## 2026-06-21 (cont.) — Credenciales resueltas + `extract/copernicus.py`

**Credenciales** (B1/B2): Javier pasó credenciales nuevas de GlobColour; las guardé en
`.env` (gitignored, sin imprimir valores) y **verifiqué login FTP a `ftp.hermes.acri.fr`
OK**. Copernicus ya estaba (`--check-credentials-valid` → válidas). Ambos blockers cerrados.

**B3 (polígono TURF)**: COBI ya no está disponible. Busqué en internet; el dataset público
de TURFs (`jcvdav/ReserveEffect`) cubre QRoo + Isla Natividad, **no San Quintín**. Decisión:
usar el **bbox** de SQ (suficiente a resolución OISST/GlobColour). Guardé el shapefile (tiene
Isla Natividad) en `data/raw/turf/reserveeffect/` para Fase 3.

**`extract/copernicus.py`** (SDK `copernicusmarine` v2.4.0, reemplaza motuclient):
- `ProductSpec` + `load_products` (de `copernicus_vars.yaml`), `build_subset_kwargs` (pura:
  config → kwargs del SDK; verifiqué los nombres reales de `subset()`: `minimum_longitude`,
  `start_datetime`, `output_filename`, `overwrite`, etc.), `download_product` (idempotente
  por existencia; credenciales de `.env` o login file; `subset_fn` inyectable para tests),
  `extract` (itera productos). CLI `fishing-etl extract copernicus`.
- **Tests**: 7 con `subset` mockeado (sin red): mapeo de región/fechas, credenciales
  presentes/ausentes, idempotencia, force, iteración. **95/95 verde**, ruff limpio.
- **NO corrí la descarga real** (preferencia de Javier: dejar la ejecución como pendiente).

**Smoke test real (aprobado por Javier)**: corrí un subset de 1 semana (2019-08-01..07,
durante la MHW). Falló primero con `DatasetNotFound`: el `dataset_id` de
`copernicus_vars.yaml` (`cmems_obs-sst_glo_phy_my_l4_P1D-m`) ya no existe. Consulté el
catálogo vía el SDK y lo corregí a **`METOFFICE-GLO-SST-L4-REP-OBS-SST`** (OSTIA REP, 0.05°
diario, `analysed_sst`; coincide con lo que sugería `etl_design.md` §5.3). Re-corrida OK:
netCDF de 7 días × 80×90, 0.13 MB, `analysed_sst` mean ≈ 24.1 °C (plausible). Smoke borrado.
Warning menor: `InsecureRequestWarning` del backend S3 del SDK (interno, no nuestro).

**Descarga completa (aprobada)**: `fishing-etl extract copernicus` → `data/raw/copernicus/
sst_l4.nc` (47 MB). El SDK recortó solo al rango disponible del REP: **2017-01-01 →
2025-12-18** (3274 días continuos), grid 80×90 sobre el bbox 28-32°N/−117..−112.5°W,
`analysed_sst` en Kelvin (mean ≈20.2°C, rango [10.1, 34.3]°C; ~40% NaN = máscara de tierra,
OK porque el promedio por-UE usa skipna). El REP termina 2025-12-18 → para ene-abr 2026
haría falta el NRT, pero la langosta-SQ termina en 2022, así que sobra cobertura.

**Pendiente** (`PENDINGS.md`): `transform/copernicus.py` (`.nc`→tidy) e integrarlo en
`aggregate/ocean_by_ue` (que ya detecta `latitude`/`longitude` y la var única
`analysed_sst`). Análogo para GlobColour (FTP).

### Estado: bloqueadores de insumos = 0

B1/B2 ✓ (verificados), B3 decidido (bbox), B4 ✓ (COBI ingerido), B5 ✓ (.h5). Lo que queda
es código (extractores oceanográficos + transforms) y correr descargas (OISST, Copernicus,
GlobColour). Ruta crítica actualizada en `PENDINGS.md`.

---

## 2026-06-22 — SST + MHW integrados a `dataset_v1` (Copernicus OSTIA)

Cerré el eslabón oceanográfico para SST. Hito grande: **`dataset_v1` ya trae las columnas
oceanográficas reales** (`sst`, `sst_anomaly`, `mhw_category`, `mhw_intensity`).

**Decisión clave**: el índice MHW (Hobday) necesita un baseline climatológico **pre-ola de
calor**. La SST de 2017-2025 no sirve de baseline (incluiría las MHW y las enmascararía).
Como OSTIA REP cubre 1981+, cambié `extract copernicus` para arrancar en el inicio del
baseline (`mhw.baseline.start`, 1982) y re-descargué: **`sst_l4.nc` 1982-01-01→2025-12-18,
16058 días, 231 MB**.

**Sin `transform/copernicus.py` aparte**: `aggregate/ocean_by_ue` lee el `.nc` directo
(ya detecta `latitude`/`longitude` y la var única `analysed_sst`). Agregué
`_to_celsius` (OSTIA viene en Kelvin; OISST en °C) por atributo `units` o heurística de
magnitud. CLI `aggregate ocean --source {copernicus,oisst}`.

**Pipeline corrido**: `aggregate ocean --source copernicus` (16058 días, MHW con baseline
1982-2011) → `interim/ocean_litoral_bc_sur.parquet` → `consolidate` → `dataset_v1`.

**Validación del índice MHW** (días MHW/año en el bbox de SQ): el **Blob 2014-2016 sale
fuerte** (2014: 231, 2015: 293, 2016: 102) y 2020: 100 — justo lo esperado. En langosta-SQ,
`sst` no-nula 96.5% (el hueco es 2025-12-19→2026, fuera del REP). dist `mhw_category` en SQ:
{0:2918, 1:429, 2:40, 3:5}.

**Tests**: +2 (Kelvin→°C por units y por magnitud). **97/97 verde**, ruff limpio.

**Implicaciones**:
- OISST ya **no es necesario** para el camino crítico (Copernicus da SST+baseline). Queda opcional.
- La **figura MHW** (Fase 1.3) ya es producible con SST real (falta exponer `clim/thresh/in_mhw`).
- **Fase 1.4 con covariables** ahora es posible: re-correr el baseline incluyendo MHW/SST
  para ver si baja la sobrepredicción de la temporada 2021-2022.

Pendientes finos en `PENDINGS.md`: NRT para ene-abr 2026 (irrelevante para langosta, que
acaba en 2022), figura MHW, y los modelos con covariables.

---

## 2026-06-22 (cont.) — Exp 2: modelo con covariables SST/MHW vs el piso

Primer modelo que usa la oceanografía recién integrada.

- **`features/covariates.py`** (puro, reutilizable Fase 2): calendario (doy sin/cos,
  in_season), lags de `y` (365/730), y SST/MHW **desplazadas 90 días** (convención
  X(t)→Y(t+90d)) más medias rodantes (estado pre-temporada / calor acumulado). **Sin
  leakage** garantizado por construcción (todo `shift(≥90)`); 4 tests lo verifican
  (incluido un pico inyectado que NO aparece en features previas).
- **`experiments/exp2_covariates/covariate_model.py`**: XGBoost, misma serie/partición que
  Exp 1 (langosta-SQ, corte 2020-07-01), 17 features. Métricas + suma de temporada +
  importancias; compara contra el piso ARIMA/Prophet; artefactos en `reports/`.

**Resultado (suma de temporada 2021-2022, el crash)**: XGBoost **+368%** vs ARIMA +291% /
Prophet +418%. **Las covariables NO arreglan el crash todavía.** Matices importantes: (1) el
modelo SÍ eligió las covariables oceanográficas (`sst_roll90_lag90`,
`mhw_category_roll365_lag90` en el top-5), (2) MAE diario 327 (lig. mejor que ARIMA 345).

**Por qué**: train = 2017-01→2020-06 = solo ~3 temporadas → casi no hay ejemplos de la
relación "MHW del año previo → menor captura". **Hallazgo metodológico que motiva Fase 3**:
el valor de las covariables requiere más temporadas y/o *pooling* entre especies/UEs (modelo
global que tome prestada fuerza estadística). Honesto y defendible para la tesis.

**Tests**: +4 (no-leakage). **101/101 verde**, ruff limpio. Artefactos:
`reports/metrics/exp2_covariates_2020-07-01.json`, `reports/figures/exp2_covariates_pred_vs_real.png`,
`reports/exp2_covariates_summary.md`.

### Estado

Fase 1 (ETL + baseline + covariables) esencialmente cerrada como marco. El cuello de botella
ya no es código ni datos oceanográficos, sino **cantidad de temporadas de captura**. Siguiente
paso natural: **Fase 3 (modelo global multi-especie/UE)** para pooling, y/o conseguir más
temporadas de arribos.

---

## 2026-07-02 — Fase 3: modelo global multi-especie (marco + primer resultado)

Diseño en `docs/hierarchical_design.md`. Con los datos actuales, "global" = **4 especies en
SQ** (langosta + 3 abulones; erizo excluido, 0 captura). Mono-UE todavía (solo hay arribos
de `litoral_bc_sur`).

- **`features/build_multiseries_features`**: agrupa por especie y aplica el builder por serie
  → sin cross-leakage entre especies (test dedicado: pico en langosta no aparece en features
  de abulón). One-hot de especie se hace en el experimento.
- **`experiments/exp3_global_model/`**: XGBoost **global** (4 especies, one-hot + oceanografía
  compartida) vs **específico** (uno por especie). Corte 2020-07-01. Compara RMSE diario por
  serie.

**Resultado** (RMSE diario, global vs específico):
| serie | global | específico | gana |
|---|---|---|---|
| abalone_black | **8.8** | 52.8 | global (6×) |
| abalone_red | **7.7** | 8.9 | global |
| abalone_blue | 8.9 | **6.0** | específico |
| lobster_red | 759.7 | **558.3** | específico |

**Global gana 2/4 (50%)**, bajo el criterio de 60% (PLAN §3.1). Pero el **patrón es el
esperado y valioso**: el pooling **ayuda a las series cortas** (abalone_black, que sola
sobreajusta feo: 52.8→8.8) y **estorba a la rica** (langosta se diluye: 558→760). Top
features del global: `in_season, doy_sin, y_lag365, sst_roll90_lag90, doy_cos` — el one-hot
de especie NO está en el top-5, así que el modelo explota estacionalidad/oceanografía
compartida, no solo separa especies.

**Conclusión metodológica**: el pooling total no es óptimo; el camino es **partial pooling /
jerárquico** (regularizar las series cortas hacia el grupo sin diluir la rica). Eso + multi-UE
(más UEs) son el siguiente paso real de Fase 3.

**Tests**: +1 (no cross-leakage multiserie). **102/102 verde**, ruff limpio. Artefactos:
`reports/metrics/exp3_global_model_2020-07-01.json`, `reports/exp3_global_model_summary.md`.

### Estado

Fases 1-3 existen como **marco reproducible y testeado** sobre datos reales (arribos COBI +
SST/MHW Copernicus). El cuello de botella persistente es la **cantidad de series/temporadas**
(4 especies, 1 UE, ~3 temporadas de train). Palancas de mayor impacto: multi-UE (definir
bboxes + procesar CONAPESCA nacional), más temporadas, y partial pooling.

---

## 2026-07-03 — Fase 3 multi-UE: se agrega Isla Cedros (gradiente biogeográfico)

Ataqué el cuello de botella "1 UE" agregando una 2ª unidad económica.

**Exploración**: en el crudo COBI, las cooperativas del Pacífico se agrupan por oficina de
arribo (Ensenada ~31.8°N, El Rosario ~30°N, **Isla Cedros ~28°N**) → esa dispersión de
latitud ES el gradiente biogeográfico de la tesis. Elegí **Isla Cedros** (SCPP Pescadores
Nacionales de Abulón): isla → ubicación clara, e ícono del abulón. Aporta lobster_red (~576 t,
343 días) + abalone_blue (~175 t) a ~28°N.

**Config**: agregué `isla_cedros` a `economic_units.yaml` (bbox aprox. de la isla, marcado
provisional). Cedros (27.7-28.4°N) caía debajo del bbox de descarga de Copernicus (lat_min
28) → **amplié `copernicus_vars.yaml` a lat_min 27** y re-descargué la SST (289 MB, 1982-2025,
lat 27-32).

**Código**: `build_multiseries_features` ahora acepta `group_col` multi-columna
(`["species","economic_unit"]`) sin romper el uso mono-columna (tests verdes). `exp3`
generalizado a series `(especie, UE)`: descubre series con ≥20 días de captura, one-hot de
especie **y** UE, compara global vs específico por serie.

**Pipeline multi-UE corrido**: transform (2 UEs) → aggregate ocean SQ + Cedros → consolidate.
`dataset_v1` ahora tiene **2 UEs**, SST 97% en ambas.

**Resultado Exp 3 multi-UE** (RMSE diario, 5 series):
| serie | días captura test | global | específico | gana |
|---|---|---|---|---|
| lobster_red@isla_cedros | 85 | **893.6** | 898.7 | global |
| lobster_red@litoral_bc_sur | 251 | 618.9 | 558.3 | específico |
| abalone_black@SQ | 3 | 24.6 | 52.8 | global |
| abalone_blue@SQ | 6 | 24.1 | 6.0 | específico |
| abalone_red@SQ | 2 | 24.0 | 8.9 | específico |

Global 2/5 (40%). **Hallazgos clave**: (1) **transferencia positiva dentro de especie entre
UEs** — la langosta de Cedros (menos datos) mejora al agruparse con la de SQ (893.6 < 898.7);
(2) el pooling **todo-junto está confundido por escala**: la langosta (cientos de kg) domina
el loss y **empeora** las predicciones de abulón (RMSE abulón-global saltó de ~8 a ~24 al
meter la langosta de Cedros al pool); (3) `species_lobster_red` sube al top-3 de importancias
(el modelo necesita distinguir escalas). **Conclusión de diseño**: no pooling todo-junto;
**agrupar por especie entre UEs** y/o **normalizar `y` por serie** (log/escala) para que la
escala no domine.

**Tests**: 102/102 verde, ruff limpio. Artefactos: `reports/metrics/exp3_global_model_2020-07-01.json`,
`reports/exp3_global_model_summary.md`.

### Estado

Multi-UE funciona end-to-end (mecánica lista para agregar más UEs: definir bbox por oficina).
La lección metodológica (transferencia intra-especie sí; pooling inter-escala no) apunta al
siguiente paso: **partial pooling / normalización de `y` por serie**, y sumar más UEs
(El Rosario, Ensenada) para robustecer la señal intra-especie.

---

## 2026-07-05 — Color del océano (GlobColour vía Copernicus) integrado a `dataset_v1`

Ataqué el gap más grande: las variables ópticas `x1..x16` que el borrador usaba y que aún
no estaban en el dataset nuevo.

**Decisión de fuente** (exploré el FTP de GlobColour primero): `ftp.hermes.acri.fr` sirve
archivos **globales, sin subset** — a 4 km son 18 MB/var/día → **~630 GB** para 16 vars × 6
años. Inviable. La misma data GlobColour se re-distribuye por **Copernicus Marine** con
subset server-side (como la SST), así que extendí `extract/copernicus.py` en vez de escribir
un extractor FTP. Presenté la disyuntiva a Javier; eligió Copernicus.

**Integradas 6 variables** ópticas/biológicas (4 km diario, 2015-2026, subset al bbox):
CHL (plankton L4 gap-free), KD490/SPM/ZSD (transp), BBP/CDM (optics). No están en Copernicus
(atmosféricas, se omiten): POC, PIC, PAR, aerosoles T865/T550, nubosidad.

**Código**: `copernicus_vars.yaml` +3 productos con `start` por-producto (2015, distinto del
baseline 1982 de la SST); `ProductSpec.start` + override en `download_product`.
`aggregate/ocean_by_ue`: refactoricé el masking a `_bbox_spatial_mean` y añadí `bbox_means`
(multi-variable, sin MHW) + `oc_series_for_bbox`. CLI `aggregate oceancolor`. `consolidate`
+ `_attach_oceancolor` (broadcast por UE, columnas extra al esquema). Fix: el glob
`ocean_*.parquet` excluía mal `oceancolor_*`; corregido. `features/covariates.OCEAN_COLS`
+= las 6 OC. Tests: +2 (per-product start, OC en consolidate). **104/104 verde**, ruff limpio.

**Pipeline corrido**: `extract copernicus` (OC ~1.3 GB) → `aggregate oceancolor` SQ + Cedros
→ `consolidate`. `dataset_v1` ahora trae `chl,kd490,spm,zsd,bbp,cdm` (cobertura 88-100%).
Validación de gradiente: **CHL San Quintín 0.94 vs Isla Cedros 0.62 mg/m³** (más surgencia
costera en SQ) — la data OC captura variación espacial real.

**Hallazgo de modelado (importante y honesto)**: re-corrí Exp 2 y Exp 3 con las OC. En la
serie única (Exp 2) el desempeño **empeora** (MAE 327→424; error de temporada 2021-22
+368%→+399%) — **sobreajuste**: ~35 features con solo ~3 temporadas. El modelo SÍ usa las OC
(`bbp/cdm/zsd_roll90_lag90` y `sst_roll90_lag90` en el top-5), pero sin más datos no ayudan.
En Exp 3 (pooled) sigue 2/5, con la transferencia intra-especie langosta@cedros aún positiva.

**Conclusión**: la data óptica ya está integrada y el pipeline es sólido; su valor requiere
**selección de features (SHAP, Fase 2.3)** + **más datos** (más UEs/temporadas) y/o
regularización. Objetivo #1 (recuperar las `x`) cumplido; ahora el cuello es data + selección,
no disponibilidad de variables.

## 2026-07-05 — Exp 2.3: selección de features con SHAP

`experiments/exp2_shap_selection/shap_prune.py`. Entrené XGBoost con las 35 features (langosta-SQ,
corte 2020-07-01), calculé SHAP (`TreeExplainer`) sobre train y podé a las 16 con
mean|SHAP| ≥ 1% del total, re-entrenando solo con esas. **Resultado honesto: la poda no ayuda.**

| modelo | n features | MAE | RMSE | sMAPE% | error temp. 2021-22 |
|---|---|---|---|---|---|
| completo | 35 | 423.6 | 659.7 | 116.0 | +398.7% |
| podado | 16 | 444.7 | 695.5 | 120.3 | +446.6% |

El ranking SHAP es **coherente** (calendario `doy_sin`/`in_season` primero, luego
`bbp_roll90_lag90`, `y_lag365`, `doy_cos`, `sst_lag90`, `y_lag730`, y varias OC rolling) — o sea
las features tienen sentido físico. Pero podar empeora un poco: **el cuello no es *cuáles*
features sino el volumen de datos** (~3 temporadas de langosta en SQ). Confirma el hallazgo de
Exp 2: la selección de features sola no arregla el sobreajuste; hace falta **más datos (Fase 3:
más UEs/temporadas, pooling con `y` normalizada)** y/o **regularización más fuerte**. Artefactos:
`reports/metrics/exp2_shap_selection_2020-07-01.json`, `reports/figures/exp2_shap_selection_shap_bar.png`,
`reports/exp2_shap_selection_summary.md`. ruff limpio.

## 2026-07-17 — Exp 3.2: pooling con `y` normalizada por serie (el hallazgo que destraba Fase 3)

`experiments/exp3_global_model/pooled_ynorm.py`. Exp 3 dejó el pooling todo-junto **confundido
por escala** (langosta cientos de kg domina el loss cuadrático sobre abulón/erizo unidades). Aquí
normalizo el objetivo **por serie antes de agrupar** y comparo, serie por serie, contra el modelo
específico (y cruda): `pooled_raw` (=Exp 3), `pooled_log` (log1p(y)), `pooled_z` (z-score por serie
con media/desv de **train** → sin leakage). Predicción invertida a kg antes de medir.

RMSE diario (kg), corte 2020-07-01:

| serie | específico | p_raw | p_log | p_z |
|---|---|---|---|---|
| abalone_black@SQ | 52.0 | 35.4 | **7.0** | 11.4 |
| abalone_blue@SQ | 6.7 | 32.7 | **4.6** | 15.6 |
| abalone_red@SQ | 7.3 | 34.9 | **3.5** | 10.6 |
| lobster@cedros | 914.1 | 889.0 | 1131.4 | 916.6 |
| lobster@SQ | 659.7 | 742.2 | **544.4** | 1013.9 |

**Gana/empata vs específico:** p_raw 0.4, **p_log 0.8**, p_z 0.2 (de 5 series). **`pooled_log`
cumple el criterio ≥0.60 de PLAN §3.1.** El log1p **rescata las series de escala chica** (abulón:
raw 33-35 → log 3.5-7.0, mejor que el específico) porque el loss deja de estar dominado por
langosta, y **mejora langosta@SQ** (544 vs 659). La única pérdida es langosta@Cedros (el log
comprime de más la serie de mayor escala). El z-score queda peor que el log (varias series con
pocos días de train → stats ruidosas).

**Conclusión (destraba Fase 3):** el pooling multi-serie **sí funciona** una vez que se quita el
confound de escala; **la transformación correcta es `log1p(y)`**, no el z-score. Este es el modelo
global a llevar a producción/CQR. Artefactos: `reports/metrics/exp3_pooled_ynorm_2020-07-01.json`,
`reports/exp3_pooled_ynorm_summary.md`. ruff limpio; sin cambios en `src/` (tests 104/104 intactos).

## 2026-07-21 — Fase 4: pronóstico probabilístico con CQR (intervalos calibrados)

`experiments/exp4_cqr/cqr_intervals.py`. Envuelve el modelo global `pooled_log` (Exp 3.2) en una
**Conformalized Quantile Regression**: regresores cuantílicos XGBoost (`reg:quantileerror`) en
espacio log + corrección conformal split. Partición temporal estricta: proper-train (más antiguo)
| conformalización (último 25% pre-corte, desde 2019-08-15) | test (≥ 2020-07-01). Métricas nuevas
en `evaluation/metrics.py` (con tests): `coverage`, `mean_interval_width`, `pinball_loss`,
`crps_from_quantiles`.

**Dos decisiones de diseño (aprendidas corrigiendo bugs):** (1) **conformalizar en espacio log,
no en kg** — la corrección aditiva en log es multiplicativa en kg, así escala por serie (un solo
`Q` en kg lo fijaba langosta e inflaba abulón); (2) **Mondrian (un `Q` por serie)** — es el
análogo conformal de la normalización de `y` de Exp 3.2. Además se reordenan los cuantiles por fila
(rearrangement) para quitar el cruce de regresores independientes → intervalos **anidados**. Se
prefirió CQR propia sobre `mapie` (1.3) porque `mapie` con modelos cuantílicos prefit + inversión
`expm1` daba intervalos **no anidados** (80% más ancho que 90%).

Resultados (corte 2020-07-01): cobertura marginal **80%→94.1%, 90%→97.5%** (conservador, sobre-
cubre). **Cobertura condicional durante MHW (int. 90%): 0.987 (313 días) vs 0.973 fuera** — el
intervalo **se mantiene honesto en las temporadas anómalas** que rompieron el modelo puntual (no
pierde calibración post-MHW). Por serie (cob. 90% / ancho mediano kg / CRPS): abulón
99.2-99.5% / ~4 / <1; langosta@Cedros **91.9%** / ~21 / 542; langosta@SQ 99.5% / ~23 / 329. Todas
≥ nominal. **Limitación honesta:** los intervalos **sobre-cubren y son anchos** (la cota superior
al 90% se dispara en días pico por la expansión exponencial del log + conservadurismo conformal con
pocas temporadas + shift conf→test que cruza el crash post-MHW). Artefactos:
`reports/metrics/exp4_cqr_2020-07-01.json`, `reports/exp4_cqr_summary.md`,
`reports/figures/exp4_cqr_fan_chart.png` (fan chart langosta@SQ, eje recortado 3x máx observado).
ruff limpio, **tests 109/109** (+5 métricas probabilísticas). **Próximo:** calibración más fina
(conformal adaptativo/normalizado, más datos de conformalización) para reducir el ancho sin perder
cobertura; entregar a COBI el rango por temporada.

## 2026-07-21 (tarde) — Afinar la CQR: calibrar en temporada + conformal normalizado

Objetivo: corregir la **sobre-cobertura** de Exp 4 (90% nominal → 97.5% empírico, intervalos
anchos). Añadí a `exp4_cqr`: (1) variante **normalized**
(localmente adaptativa, ensanche ∝ ancho base del intervalo) además de la `split` (constante),
comparadas lado a lado; (2) **conformalización solo en temporada**.

**Diagnóstico clave (por qué normalized *sola* no cambiaba nada):** el cuantil conformal `Q` salía
**exactamente 0** en todas las series → la corrección era nula y split==normalized. Causa: el set
de conformalización estaba dominado por **días fuera de temporada** (captura 0, cuantiles base ≈ 0,
score de conformidad = 0); esa masa de ceros fijaba el percentil-90 de los scores en 0. La
cobertura base de los cuantiles XGBoost ya era 93-100% (sobre-dispersos), así que el conformal no
tenía nada que ensanchar.

**Fix:** calibrar **solo en días en temporada** (`in_season==1`). Resultado: la cobertura pasó de
97.5% (muy conservadora) a **~86% (90% nominal) / ~81% (80% nominal)** — cerca de lo nominal — y
**normalized ahora sí mejora a split** (90%: cobertura 85.9% vs 84.9%; ancho p90 80%: 2249 vs 2440
kg, mantiene angostos los días fuera de temporada). MHW-condicional 0.891 (en MHW) vs 0.854 (fuera)
— honesto y cerca de nominal. Queda una **leve sub-cobertura** (86% < 90%) atribuible al shift
conf→test (el test cruza el crash post-MHW 2021-22, catches bajos). **El ancho grande en días pico
lo fija el modelo cuantílico base (q0.95 en log + expm1), no el conformal** → afinar más requiere
mejores cuantiles/más datos, no otro envoltorio conformal. Artefactos regenerados
(`exp4_cqr_*`). ruff limpio, tests 109/109.

## 2026-07-21 (noche) — Exp 4b: endurecer los cuantílicos base (Optuna) + corrección honesta

Objetivo: estrechar el ancho p90 (días pico) de la CQR afinando los **modelos cuantílicos base**
(que son los que fijan el ancho, no el conformal). `experiments/exp4_cqr/tune_quantiles.py`: Optuna
(40 trials, seed fija) minimizando la pérdida pinball en el 30% final del pre-corte (en temporada),
sin tocar test; luego compara default vs afinado con el pipeline de Exp 4.

**Resultado NEGATIVO (honesto).** La pinball de validación bajó (0.146→0.139) pero en test el
afinado **empeoró catastróficamente las series de gran escala**: langosta@SQ ancho p90 36,003 →
**401,985 kg**, CRPS 218→2825, cobertura 53%→45%; langosta@Cedros p90 6,966→83,732. El abulón
(escala chica) sí se estrechó (p90 ~17→~7) pero ya estaba bien. Causa: el afinado sobreajusta el
periodo de validación (pre-crash) y, en espacio log, un q0.95 apenas mayor **explota al exponenciar**
en los días pico del test. **Conclusión: HPO sobre pinball NO es la palanca; estrechar requiere más
temporadas / mejores cuantílicos, no otro tuning.** No se adoptan los params afinados.

**Corrección honesta de un reporte previo:** al revisar la cobertura **por serie** del Exp 4 actual
(conf en temporada) descubrí que **langosta@SQ sub-cubre al 53.4%**, no el ~99% que reporté antes en
la sesión (ese número era de la versión con conf en *todos* los días, previa al fix). O sea: calibrar
en temporada arregló la sobre-cobertura marginal (97.5%→86%) pero dejó a langosta@SQ **sub-cubierta**
—la caída post-MHW 2021-22 cae por debajo de la cota inferior calibrada con 2019-20—. El 86% marginal
promedia abulón sobre-cubriendo (99%) y langosta sub-cubriendo (53%). Es, de nuevo, el cuello de
datos: **una sola temporada de calibración no cubre un cambio de régimen**. Actualicé la tesis
(§CQR) con esta salvedad y con el resultado negativo del tuning.

**Refactor de higiene:** moví las utilidades conformal (`split_cqr`, `mondrian_cqr`,
`conformal_quantile`, `sorted_quantile_preds`) de `exp4_cqr.py` a **`src/fishing_forecast/evaluation/
conformal.py`** (reusables + testeables); Exp 4 y Exp 4b las importan de ahí (sin hacks de `sys.path`;
Exp 4b se corre como módulo `python -m ...`). +5 tests de conformal. **114/114 tests, ruff limpio.**
Artefactos: `reports/metrics/exp4_cqr_tuned_2020-07-01.json`, `reports/exp4_cqr_tuned_summary.md`.

## 2026-07-21 (noche, cont.) — Más UEs: clúster El Rosario (5 cooperativas de langosta)

Ataqué el cuello de datos por el lado factible. Exploré el crudo COBI: **"más temporadas" NO es
posible** (el export es 2017-2021; 0 filas 2022-2024, 2 en 2025 — las temporadas 2022-25 siguen
sin conseguirse). **"Más UEs" SÍ**: hay varias cooperativas con langosta roja en el Pacífico.
Usando `nombre_oficina`/`nombre_lugarcaptura` del crudo identifiqué el **clúster El Rosario**
(~29.6-30.4°N, justo al sur de San Quintín): SCPP ENSENADA SCL (758 t, ¡mayor que la UE del
paper!), MORTERA DE LEYVA (186 t), EL CHUTE (175 t), REGASA Nº2 (120 t), ISLA SAN GERONIMO (93 t).

**Implementación** (per-cooperativa, bbox El Rosario compartido — decisión del usuario): 5 UEs
nuevas en `economic_units.yaml` con bbox anclado (`&el_rosario_bbox`, dentro de la caja Copernicus
ya descargada). `transform arribos` (7812 filas mapeadas vs ~pocas antes) → `aggregate ocean`/
`oceancolor` una vez para el bbox compartido, copiado a los 5 códigos (evita releer el `.nc` 5x) →
`consolidate`. **`dataset_v1`: langosta pasa de 2 a 7 series** (185-384 días de captura c/u, SST 96.5%).

**Resultados (el pooling con más datos SÍ ayuda):**
- **Exp 3.2 (pooled)**: **langosta@SQ RMSE 659.7 → 313.5** (~52% menos) al agrupar con las UEs de
  langosta nuevas — el payoff directo de más series. Con 14 series (antes 5) el win-rate se
  reordena: **z-score gana 0.93** (13/14), raw 0.64, log 0.43 (antes log ganaba 0.8, z 0.2): con
  más series comparables, estandarizar por serie se vuelve la mejor normalización.
- **Exp 4 (CQR)**: cobertura marginal 90% **85.9% → 92.9%** (cerca de nominal), ancho p90 **9776 →
  3812 kg** (colas mucho más angostas), CRPS 218 → 188. **Pero langosta@SQ sigue sub-cubriendo
  (~47%)**: la CQR calibra por serie (Mondrian) y langosta@SQ sigue con **una sola temporada
  pre-crash** de calibración → el crash post-MHW queda fuera de distribución. Más UEs no añaden
  puntos de calibración a langosta@SQ; eso necesita más **temporadas** de langosta@SQ.

**Síntesis honesta:** más UEs mejoran mucho el **modelo puntual** (langosta@SQ a la mitad) y la
**calibración marginal + ancho de colas** de la CQR; el hueco específico de langosta@SQ en el crash
es un problema de temporadas, no de UEs. Confirma el diagnóstico: la palanca es volumen de datos, y
sumar UEs es la vía factible (las temporadas 2022-25 siguen bloqueadas). 114/114 tests, ruff limpio.
Artefactos exp3/exp4 regenerados.

## 2026-07-21 (noche, cont.) — ¡Desbloqueadas las temporadas 2022-2026! (unión COBI+CONAPESCA)

Javier señaló 5 CSV en Downloads: **AVISOS MAYORES/MENORES COSECHA 2022-2026** (CONAPESCA nacional,
~160 MB/año, iso-8859-1). Exploré: **SÍ traen langosta de nuestras UEs para 2022-2026** (el hueco
que llevábamos toda la sesión lamentando). Mismo esquema que el export COBI (columnas en MAYÚSCULAS),
`nombre_oficina`/`nombre_lugarcaptura` permiten filtrar BC Pacífico. Preámbulo variable (2 o 4 líneas
según el año).

**Integración** (decisión del usuario: COBI ≤2021-12-31 + CONAPESCA >2021-12-31, sin doble conteo):
- `read_source_csv` ahora **autodetecta la fila de encabezado** (`_detect_skiprows`, busca `col_ue`)
  → tolera el preámbulo variable sin romper COBI/CONAPESCA previos.
- `clean_arribos`/`transform` aceptan `date_min`/`date_max`.
- CLI `transform arribos --source union`: COBI acotado a ≤2021-12-31 + cada `cosecha_YYYY.csv`
  acotado a su propio año (→ ≥2022, sin solape de frontera), unidos. Archivos en
  `data/raw/arribos/cosecha_2022_2026/` (gitignored). +2 tests (autodetección de header, filtro de fechas).
- Pipeline: `transform arribos --source union` (5361 filas) → `consolidate`. **`dataset_v1` ahora
  2017→2026; langosta@SQ pasa de ~5 a ~9 temporadas** (1110 días de captura vs 693).

**Resultados — las temporadas SÍ arreglan lo que el tuning no pudo:**
- Corte canónico 2020-07-01: CRPS global 188→**158**, cobertura marginal 90% ~94%; pero langosta@SQ
  sigue sub-cubriendo (~41%) **porque las temporadas nuevas caen todas en test** (la calibración sigue
  siendo pre-2020).
- **Corte 2024-06-01** (crash + recuperación quedan en train/calibración; `FF_CUT_DATE` override):
  **langosta@SQ cobertura 90% pasa de ~41% a 95.8%**, todas las series de langosta 94-99%, marginal
  95%, MHW-condicional 0.90 vs 0.95. **Este es el arreglo real de la sub-cobertura**: con más
  temporadas el crash post-MHW ya no está fuera de la distribución aprendida.

**Síntesis:** el cuello de datos que atravesó toda la fase se **destrabó por el lado de las
temporadas**. La CQR queda bien calibrada en el corte 2024-06-01 (producto operativo real para COBI).
116/116 tests, ruff limpio. **Pendiente**: reflejar en la tesis (nuevas temporadas + corte 2024-06-01
+ el salto de cobertura de langosta@SQ); confirmar consistencia del empalme 2021 COBI↔CONAPESCA.
