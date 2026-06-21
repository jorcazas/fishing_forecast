# SETUP_AND_RUNBOOK.md

Guía paso a paso para (A) conseguir las credenciales/insumos que faltan y (B) ejecutar
el orden de trabajo para cerrar los pendientes (`PENDINGS.md`). Los nombres de variables
de entorno corresponden a `.env.example`.

> Convención: **[tú]** = acción externa que solo puede hacer Javier; **[claude]** =
> implementación/ejecución que puede hacer Claude Code una vez desbloqueado.

---

## Parte A — Conseguir las llaves / insumos

Primero crea tu archivo de secretos local (está en `.gitignore`, nunca se commitea):

```bash
cp .env.example .env
```

Luego rellena cada bloque conforme lo obtengas.

### A1. GlobColour (FTP) → `GLOBCOLOUR_USER` / `GLOBCOLOUR_PASS`  [tú]
1. Entra a **https://hermes.acri.fr/** (ACRI-ST, distribuidor de GlobColour).
2. Regístrate y solicita acceso a los productos GlobColour 4 km (las 16 variables de
   `configs/globcolour_vars.yaml`). La aprobación llega **por email** (1-2 días).
3. Te mandan credenciales **FTP** de `ftp.hermes.acri.fr`.
4. Ponlas en `.env`: `GLOBCOLOUR_USER=...`, `GLOBCOLOUR_PASS=...`.
5. Verifica: `lftp -u "$GLOBCOLOUR_USER","$GLOBCOLOUR_PASS" ftp.hermes.acri.fr` y lista un
   directorio.

> Varias variables de GlobColour también están en Copernicus Marine (A2). Si la
> aprobación de Hermes se demora, podemos sacar algunas de ahí.

### A2. Copernicus Marine → `COPERNICUS_USER` / `COPERNICUS_PASS`  [tú]
1. Regístrate en **https://marine.copernicus.eu/** (gratis).
2. Instala el SDK y haz login (Claude agrega la dependencia):
   ```bash
   uv add copernicusmarine
   uv run copernicusmarine login        # guarda ~/.copernicusmarine
   ```
3. Pon también user/pass en `.env` (para correr sin interacción).
4. Verifica con el producto ya configurado en `configs/copernicus_vars.yaml`:
   ```bash
   uv run copernicusmarine describe --dataset-id cmems_obs-sst_glo_phy_my_l4_P1D-m
   ```

### A3. Shapefile TURF de COBI (B3 — afina el bbox placeholder)  [tú → claude]
1. Pide a COBI el **polígono TURF** de `LITORAL DE BAJA CALIFORNIA S DE PR DE RL` (y otras
   UEs) como **shapefile (.shp/.dbf/.shx) o GeoJSON**.
2. Colócalo en `data/raw/turf/` (gitignored).
3. Avísale a Claude: agregará `geopandas`, derivará el bbox/polígono y actualizará
   `configs/economic_units.yaml`. No requiere re-ETL de código.

### A4. CSV legacy de arribos COBI (B4 — detalle fino 2017-2021)  [tú → claude]
1. Localiza `Arribos2017-2021.csv` (máquina local u OneDrive de COBI).
2. Da la **ruta absoluta** (no pegues el contenido: puede traer datos de pescadores).
3. Claude implementa `extract/arribos_cobi.py` para leerlo.

### A5. Artefactos S3 del borrador 2024 (B5 — comparación de métricas)  [tú → claude]
Las credenciales AWS van en **`keys.json`** en la raíz (gitignored vía `*keys.json`):
```json
{
  "aws_access_key_id": "...",
  "aws_secret_access_key": "...",
  "region": "us-east-1",          // opcional
  "bucket": "mi-bucket-borrador"  // opcional; si no, usa S3_BUCKET_LEGACY de .env
}
```
1. Pega ahí las llaves (ya tienes `aws_access_key_id`/`aws_secret_access_key`).
2. Agrega `"bucket"` a `keys.json` **o** `S3_BUCKET_LEGACY=...` en `.env`.
3. Lista y descarga:
   ```bash
   uv run fishing-etl extract s3-legacy --list-only            # descubre
   uv run fishing-etl extract s3-legacy --prefix models/       # descarga a models/legacy/
   ```

---

## Parte B — Cerrar los pendientes (orden de ejecución)

### Etapa 1 — Dataset real (sin credenciales)
1. **[tú/claude] Descargar CONAPESCA** (~1.4 GB), idempotente:
   ```bash
   uv run fishing-etl extract conapesca --years all
   ```
2. **[tú] Decidir rango OISST** (baseline 1982-2011 + operativo 2012-2025 ≈ 6-7 GB) y
   **[tú/claude] descargar**:
   ```bash
   uv run fishing-etl extract oisst --years 1982-2025
   ```
3. **[claude] Confirmar el centinela NaN de CICESE** inspeccionando un `.dat` real
   (¿9999? ¿-99999?) y fijar `na_values` en la transformación.
4. **[tú/claude] Correr el pipeline:**
   ```bash
   uv run fishing-etl transform arribos
   uv run fishing-etl aggregate ocean --ue litoral_bc_sur
   uv run fishing-etl extract cicese && uv run fishing-etl transform cicese
   uv run fishing-etl consolidate
   uv run fishing-etl qc --fail-on-warning
   ```

### Etapa 2 — Cerrar Fase 1.3 / código desbloqueado (todo [claude])
5. **Check de correlación SST CICESE vs OISST** en `quality_checks.py`.
6. **`reports/figures/mhw_timeline.png`** (Blob 2014-16 + régimen 2019-21).
7. **Particionado** de `dataset_v1` por `species × year`; **ADR-0001** (y-missing);
   **export de compatibilidad** `lstm_data.csv`; **refactor** de los 3 descargadores a
   `utils/download.py`.

### Etapa 3 — Enriquecimiento oceanográfico (requiere A1/A2)
8. **[claude]** `extract/copernicus.py` (+ transform) con el SDK → SST/corrientes.
9. **[claude]** `extract/globcolour.py` (+ transform) → columnas `x1..x16`.
10. **[claude]** Integrar ambos en `aggregate/ocean_by_ue.py` y re-consolidar.

### Etapa 4 — Modelado (requiere Etapa 1; B5 ayuda)
11. **[claude] Fase 1.4** — `experiments/exp1_baseline_retrain/` (ARIMA/Prophet/LGBM/XGB/
    LSTM/ensamble), cortes `2020-07-01` + `2024-06-01`, métricas vs paper (±10%).
12. **[claude] Fases 2→5** — feature engineering+SHAP, modelo global, CQR, TFT opcional.

---

## Ruta crítica (camino más corto a valor)

**A3 (bbox/shapefile) + confirmar rango OISST → Etapa 1 → Etapa 2 (6,7) → Fase 1.4.**
Las credenciales (A1/A2) corren en paralelo y solo bloquean el *enriquecimiento* de la
Etapa 3.

Lo que Claude puede hacer **sin ningún insumo**: Etapa 2 items **5, 6, 7** (con SST
sintética como smoke test) y el refactor de descargadores. Esos ya están en curso.
