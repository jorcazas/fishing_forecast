# ADR-0001 — Manejo de `y` faltante en el dataset consolidado

- **Estado**: aceptada
- **Fecha**: 2026-06-19
- **Contexto del código**: `src/fishing_forecast/etl/consolidate.py`
- **Relacionado**: `docs/etl_design.md` §4.4

## Contexto

El dataset consolidado tiene granularidad **una fila por `(ds, species, economic_unit)`**
sobre el rango de fechas completo (no solo los días con registro de arribo). Para la
mayoría de las celdas no hay un aviso de arribo. Hay que decidir qué valor toma `y`
(volumen desembarcado, kg) en esos huecos, porque la semántica afecta directamente al
entrenamiento: un `0` afirma "no se pescó"; un `NaN` afirma "no sabemos".

## Decisión

Tres casos, distinguidos por la bandera `in_season`:

1. **Fuera de temporada** (`in_season=False`) sin registro → **`y = 0`**, `is_imputed_y=False`.
   Asumimos que no hubo captura legal (veda). El `0` es *definicional*, no una imputación.
2. **Dentro de temporada** sin registro → **`y = NaN`**, `is_imputed_y=False`.
   Puede ser un día sin pesca (clima, festivo, veda local) o un hueco de captura de datos:
   no lo sabemos, así que no inventamos un valor.
3. **Imputación explícita** (rellenar con 0 u otro valor bajo un supuesto) → solo si
   alguna vez se hace, con **`is_imputed_y=True`**. **No se hace por default.**

## Razón

- El `NaN` dentro de temporada es *boolean-flag-friendly*: Prophet y LSTM lo manejan
  nativamente, y XGBoost/LightGBM también con `missing=np.nan`. Mantenerlo preserva la
  distinción "sin pesca" vs "sin dato", que un `0` borraría.
- El `0` fuera de temporada no es una imputación: es la consecuencia lógica de la veda, y
  evita que los modelos vean huecos espurios en periodos donde por ley `y=0`.
- La bandera `is_imputed_y` deja la puerta abierta a imputaciones futuras sin perder
  trazabilidad.

## Consecuencias

- Cada modelo decide cómo tratar el `NaN` in-season (drop, máscara, o relleno propio en su
  preprocesamiento) — el ETL no impone esa decisión.
- Las métricas de suma de temporada deben tratar `NaN` con cuidado (sumar con `skipna` o
  filtrar), distinto de tratar `0`.
- Si en el futuro se confirma con COBI que ciertos huecos in-season **sí** son `y=0`
  (no falta de dato), se imputan marcando `is_imputed_y=True` y se actualiza este ADR.
