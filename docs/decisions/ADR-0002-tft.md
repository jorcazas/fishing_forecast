# ADR-0002 — Temporal Fusion Transformer como prueba de techo (Fase 5)

- **Estado**: aceptada (código entregado; entrenamiento pendiente de correr)
- **Fecha**: 2026-07-22
- **Contexto del código**: `experiments/exp5_tft/tft.py`
- **Relacionado**: `PLAN.md` §Fase 5; Exp 3.2 (`pooled_ynorm`), Exp 4 (`cqr_intervals`)

## Contexto

El PLAN contempla una prueba de techo (Fase 5, opcional) con un modelo de deep learning
de series de tiempo. La pregunta no es "¿podemos entrenar un Transformer?" sino
**"¿la complejidad extra paga con los datos que tenemos?"** — el diagnóstico transversal de
todo el proyecto es que el cuello de botella es el volumen de datos, no la sofisticación del
modelo (lo confirmaron el resultado nulo de SHAP y el de Optuna).

## Decisión

**Modelo: Temporal Fusion Transformer (TFT), vía `pytorch-forecasting`.**

Se eligió TFT (y no otro Transformer genérico) por tres propiedades que encajan con este
problema y permiten una comparación justa contra el pool XGBoost + CQR:

1. **Acepta los tres tipos de covariable** que ya modelamos: estáticas (especie, unidad
   económica), conocidas a futuro (calendario) y observadas pasadas (SST, MHW, color del
   océano). No hay que re-desplazar variables a mano: el codificador temporal aprende de la
   historia y el decodificador solo ve las covariables conocidas a futuro → sin fuga.
2. **Salida por cuantiles nativa** (`QuantileLoss`), comparable directamente con los
   intervalos de la CQR (misma rejilla de cuantiles → mismas métricas: cobertura, CRPS).
3. **Interpretabilidad** vía *variable selection networks* y *attention*, para contrastar
   la importancia de variables con el ranking SHAP de Exp 2.

Se prefiere `pytorch-forecasting` sobre `darts` porque su `TimeSeriesDataSet` expresa
explícitamente los tres tipos de covariable y el normalizador por grupo (`GroupNormalizer`
con `transformation="log1p"`), que replica el objetivo en escala log por serie de Exp 3.2.

**Setup para comparabilidad con Exp 4:** modelo *global* sobre las mismas series
`(especie, UE)`; objetivo `y` en kg con `log1p` + normalización por grupo; misma partición
temporal por `FF_CUT_DATE` (2020-07-01 canónico, 2024-06-01 alterno); misma rejilla de
cuantiles que la CQR; mismas métricas (MAE/RMSE/sMAPE, cobertura 80/90, CRPS), reutilizando
`evaluation.metrics`. Salidas espejo de Exp 4 (`reports/metrics/exp5_tft_<corte>.json`,
`reports/exp5_tft_summary.md`, fan chart) para poner las cifras lado a lado.

## Expectativa honesta (hipótesis a falsar)

Regla de dedo de la literatura: los Transformers de series empiezan a pagar con **>10 000
observaciones por grupo**. Aquí cada serie tiene ~200–1 110 días de captura (a lo sumo
~3 285 días-fila por serie sobre 2017-2026, incluyendo días fuera de temporada). Es **uno a
dos órdenes de magnitud por debajo** de ese umbral. La hipótesis de trabajo, por tanto, es
que **el TFT no supere al pool XGBoost + CQR**, y que ese resultado ---si se confirma--- sea
un hallazgo metodológico legítimo: con estos datos, la palanca es *más datos*, no más modelo.
El experimento está diseñado para poder falsar esto, no para lucir al TFT.

## Consecuencias

- Dependencias en el extra opcional `dl` (`torch`, `pytorch-lightning`,
  `pytorch-forecasting`), ya declarado en `pyproject.toml`; no se instalan por defecto.
- El entrenamiento es **largo** (varios minutos a decenas, según CPU/GPU y épocas): por la
  regla del proyecto no se corre sin confirmación. La preparación de datos
  (`build_tft_frame`) es pura y está testeada sin `torch`.
- Si el TFT no gana, se documenta cuánta más data haría falta; no se serializa como modelo
  final (sigue siendo el pool `log1p` + CQR el producto operativo).
