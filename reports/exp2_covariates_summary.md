# Exp 2 — Covariables oceanográficas (langosta-SQ)

Corte de test **2020-07-01**, 35 features (calendario + lags de y + SST/MHW desplazadas 90 días).

## Error de suma de temporada vs el piso (Exp 1)

| modelo | 2020_2021 | 2021_2022 (crash) |
|---|---|---|
| arima | -26% | +251% |
| prophet | +62% | +378% |
| **xgboost (SST+MHW)** | +101% | +359% |

Diario: MAE=459.0, RMSE=712.3, sMAPE=153.0%.

Top features (importancia XGBoost): in_season, bbp_roll90_lag90, doy_sin, cdm_roll90_lag90, doy_cos, sst_roll90_lag90, mhw_category_roll365_lag90, sst_lag90

> Limitación: solo ~3 temporadas en train, así que la señal MHW→captura tiene pocos ejemplos. Marco listo; el poder llega con más temporadas y el modelo global de Fase 3.