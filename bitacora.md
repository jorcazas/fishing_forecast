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
