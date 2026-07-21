# Exp 2 — Covariables oceanográficas (langosta-SQ)

Corte de test **2024-06-01**, 35 features (calendario + lags de y + SST/MHW desplazadas 90 días).

## Error de suma de temporada vs el piso (Exp 1)

| modelo | 2020_2021 | 2021_2022 (crash) |
|---|---|---|
| arima | — | — |
| prophet | — | — |
| **xgboost (SST+MHW)** | — | — |

Diario: MAE=145.4, RMSE=212.0, sMAPE=97.1%.

Top features (importancia XGBoost): in_season, doy_sin, sst_anomaly_roll90_lag90, sst_anomaly_roll365_lag90, bbp_roll90_lag90, doy_cos, cdm_roll90_lag90, sst_roll365_lag90

> Limitación: solo ~3 temporadas en train, así que la señal MHW→captura tiene pocos ejemplos. Marco listo; el poder llega con más temporadas y el modelo global de Fase 3.