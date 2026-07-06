# Exp 2 — Covariables oceanográficas (langosta-SQ)

Corte de test **2020-07-01**, 35 features (calendario + lags de y + SST/MHW desplazadas 90 días).

## Error de suma de temporada vs el piso (Exp 1)

| modelo | 2020_2021 | 2021_2022 (crash) |
|---|---|---|
| arima | -26% | +291% |
| prophet | +62% | +418% |
| **xgboost (SST+MHW)** | +101% | +399% |

Diario: MAE=423.6, RMSE=659.7, sMAPE=116.0%.

Top features (importancia XGBoost): in_season, bbp_roll90_lag90, doy_sin, cdm_roll90_lag90, doy_cos, sst_roll90_lag90, mhw_category_roll365_lag90, sst_lag90

> Limitación: solo ~3 temporadas en train, así que la señal MHW→captura tiene pocos ejemplos. Marco listo; el poder llega con más temporadas y el modelo global de Fase 3.