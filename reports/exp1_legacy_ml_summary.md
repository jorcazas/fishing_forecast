# Exp 1b — Modelos ML/DL del borrador 2023 sobre datos unidos (langosta-SQ)

Completa la Tabla `extension_comparativa` de la tesis (§6.10), que solo tenía ARIMA, Prophet y XGBoost: aquí están **LGBM, LSTM y el ensamble XGBoost→LSTM**, el mejor modelo del borrador 2023. Mismos datos, features, partición y semilla que Exp 2 (la fila `xgboost` debe coincidir con la suya).

## Corte 2020-07-01

35 features · train 1277 días / test 1644 días · semilla 42 · ventana LSTM 10 pasos.

| modelo | MAE | RMSE | sMAPE (%) | sd(pred)/sd(obs) | corr |
|---|---|---|---|---|---|
| xgboost | 459.0 | 712.3 | 153.0 | 2.37 | 0.66 |
| lgbm | 401.9 | 668.8 | 136.7 | 2.35 | 0.67 |
| lstm | 290.7 | 457.7 | 78.4 | 1.58 | 0.61 |
| xgb_lstm | 236.4 | 408.7 | 59.2 | 1.58 | 0.67 |

| modelo | temporada | error de suma (%) |
|---|---|---|
| xgboost | 2020_2021 | +101.4 |
| xgboost | 2021_2022 | +358.6 |
| xgboost | 2022_2023 | +767.3 |
| xgboost | 2023_2024 | +828.0 |
| xgboost | 2024_2025 | +960.4 |
| lgbm | 2020_2021 | +87.9 |
| lgbm | 2021_2022 | +333.4 |
| lgbm | 2022_2023 | +678.4 |
| lgbm | 2023_2024 | +671.9 |
| lgbm | 2024_2025 | +848.5 |
| lstm | 2020_2021 | +52.3 |
| lstm | 2021_2022 | +284.6 |
| lstm | 2022_2023 | +445.2 |
| lstm | 2023_2024 | +351.0 |
| lstm | 2024_2025 | +650.0 |
| xgb_lstm | 2020_2021 | +56.1 |
| xgb_lstm | 2021_2022 | +197.7 |
| xgb_lstm | 2022_2023 | +355.3 |
| xgb_lstm | 2023_2024 | +237.3 |
| xgb_lstm | 2024_2025 | +552.5 |

## Corte 2024-06-01

35 features · train 2708 días / test 213 días · semilla 42 · ventana LSTM 10 pasos.

| modelo | MAE | RMSE | sMAPE (%) | sd(pred)/sd(obs) | corr |
|---|---|---|---|---|---|
| xgboost | 145.4 | 212.0 | 97.1 | 1.61 | 0.69 |
| lgbm | 153.8 | 236.2 | 124.0 | 1.83 | 0.70 |
| lstm | 499.4 | 628.7 | 131.5 | 3.26 | 0.61 |
| xgb_lstm | 561.7 | 746.6 | 115.2 | 4.06 | 0.59 |

| modelo | temporada | error de suma (%) |
|---|---|---|
| xgboost | 2024_2025 | +173.7 |
| lgbm | 2024_2025 | +184.1 |
| lstm | 2024_2025 | +626.1 |
| xgb_lstm | 2024_2025 | +704.2 |

## Notas de lectura

- `sd(pred)/sd(obs)` y `corr` miden si el pronóstico **sigue la forma** de la serie. Un cociente cercano a 0 delata un pronóstico casi plano (que en una serie con mayoría de días en veda obtiene buen MAE sin informar nada) y uno muy por encima de 1, un modelo que sobre-reacciona.
- `xgb_lstm` es el ensamble del borrador (la predicción de XGBoost como regresor extra de la LSTM), con esa columna construida **fuera de muestra** en train (ventana expansiva); usar la predicción en muestra le daría a la LSTM una señal que en test no existe con esa calidad.
- `lstm_orig2023` (2700/800 unidades, ~41 M de parámetros) solo corre si se pide: `FF_LSTM_ARCHS=lstm,lstm_orig2023`.
