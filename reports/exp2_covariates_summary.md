# Exp 2 — Covariables oceanográficas (langosta-SQ)

Corte de test **2020-07-01**, 17 features (calendario + lags de y + SST/MHW desplazadas 90 días).

## Error de suma de temporada vs el piso (Exp 1)

| modelo | 2020_2021 | 2021_2022 (crash) |
|---|---|---|
| arima | -26% | +291% |
| prophet | +62% | +418% |
| **xgboost (SST+MHW)** | +53% | +368% |

Diario: MAE=327.0, RMSE=558.3, sMAPE=126.0%.

Top features (importancia XGBoost): in_season, doy_sin, sst_roll90_lag90, doy_cos, mhw_category_roll365_lag90, sst_lag90, y_lag365, y_lag730

> Limitación: solo ~3 temporadas en train, así que la señal MHW→captura tiene pocos ejemplos. Marco listo; el poder llega con más temporadas y el modelo global de Fase 3.