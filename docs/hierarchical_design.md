# Diseño del modelo global / jerárquico (Fase 3)

Objetivo: entrenar **un solo modelo** sobre varias series `(species, economic_unit)` para
que las series con pocos datos tomen prestada fuerza estadística de las demás (transfer),
en vez de un modelo por serie. Es la respuesta directa al cuello de botella de Exp 2: con
~3 temporadas por serie no hay señal suficiente; agrupando series sí puede haberla.

## 1. Alcance con los datos actuales (2026-06-22)

`dataset_v1` (arribos COBI, oceanografía Copernicus) tiene arribos **solo en la UE
`litoral_bc_sur`** (San Quintín). Especies con captura > 0 ahí:

| species | filas con y>0 | ~kg total |
|---|---|---|
| lobster_red | 693 | 621,243 |
| abalone_blue | 59 | 5,397 |
| abalone_red | 40 | 2,892 |
| abalone_black | 49 | 2,502 |
| urchin_red | 0 | — (excluida) |

→ **4 series** = 4 especies × 1 UE. Es un modelo **global multi-especie**, todavía
**mono-UE**. El multi-UE queda pendiente hasta tener arribos de más UEs (CONAPESCA nacional
ya descargado, pero requiere definir sus bboxes en `economic_units.yaml`) — ver `PENDINGS.md`.

Las **abalone** son las series objetivo del pooling: cortas (40-59 días con captura) y por sí
solas no modelables; la langosta (693) y la oceanografía compartida les aportan estructura.

## 2. Codificación de grupos

- **species**: one-hot (4 columnas binarias). Model-agnóstico, sin supuesto ordinal.
- **economic_unit**: one-hot cuando haya >1 UE (hoy constante → se omite).
- La oceanografía (SST/MHW) es **la misma para todas las especies de la UE** (mismo bbox);
  lo que distingue a cada serie es `y`, sus lags, y el one-hot de especie.

## 3. Partición temporal

Corte único **2020-07-01** aplicado **igual a todas las series** (no serie-por-serie): así
se respeta la cronología global y es comparable con Exp 1/Exp 2. Nada de train de una
especie que sea posterior al test de otra.

## 4. Features (reutiliza `features/covariates.py`)

Se construyen **por serie** (groupby species) para que lags/rolling **no crucen especies**,
luego se concatenan y se añade el one-hot de especie:

- Calendario: `doy_sin/cos`, `in_season`.
- Lags de `y`: 365, 730 días (dentro de cada especie).
- Oceanografía desplazada 90 días: `sst`, `sst_anomaly`, `mhw_category`, `mhw_intensity` +
  medias rodantes (compartidas entre especies de la UE).
- One-hot de especie.

`build_multiseries_features` garantiza que un evento en la serie A no aparezca en features
de la serie B (test dedicado).

## 5. Modelos y comparación

- **Global**: un XGBoost sobre las 4 series concatenadas (con one-hot de especie).
- **Específico**: un XGBoost por serie (solo sus filas).
- Se predice el test de cada serie con ambos y se comparan **por serie**.

## 6. Métricas

Por serie `(species, UE)` y agregadas: MAE, RMSE, sMAPE (diarias) + **error de suma de
temporada**. La comparación clave es **global vs específico por serie**.

**Criterio de éxito** (PLAN §3.1): el global empata o supera al específico en **≥60%** de las
series, sobre todo en las cortas (abalone), donde la transferencia debe ayudar más.

## 7. Análisis de transferencia (3.4)

Importancia de features del modelo global; en particular si el one-hot de especie domina
(el modelo trata cada especie aparte → poca transferencia real) o si las covariables
oceanográficas compartidas pesan (sí hay estructura común). SHAP condicional queda como
extensión.

## 8. Limitaciones y honestidad

- 4 series, 1 UE, ~3 temporadas de train: el pooling es el mecanismo correcto pero los datos
  siguen siendo escasos. Este experimento fija el **marco** y una primera medición; el valor
  crece con más UEs (multi-UE), más especies y más temporadas.
- No hay tuning extenso (baseline honesto); el cuello de botella es data, no hiperparámetros.

## 9. Framework

- `skforecast`/`darts` (multi-series nativo) se consideraron; para 4 series tabulares con
  covariables exógenas, un **XGBoost global con one-hot** es más simple, transparente y
  reusa la infraestructura de features/métricas. Si crece el número de series o se quiere
  backtesting con ventana expansiva "de fábrica", migrar a `skforecast.ForecasterRecursiveMultiSeries`.
