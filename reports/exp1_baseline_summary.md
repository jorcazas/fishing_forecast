# Exp 1 — Baseline estadístico (langosta-SQ)

Corte de test: **2020-07-01**. Solo modelos sobre `y`.

| modelo | MAE | RMSE | sMAPE% | error suma temporada (por temporada) |
|---|---|---|---|---|
| arima | 345.0 | 463.1 | 138.0 | 2020_2021: -26.4%; 2021_2022: 291.3% |
| prophet | 371.9 | 576.2 | 128.9 | 2020_2021: 62.4%; 2021_2022: 417.6% |

> Nota: ARIMA/Prophet son baselines débiles para una serie estacional y dispersa; el desempeño fuerte del borrador venía del ensamble XGBoost+LSTM con covariables oceanográficas (pendiente de la ingesta OISST/GlobColour).