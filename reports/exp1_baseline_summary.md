# Exp 1 — Baseline estadístico (langosta-SQ)

Corte de test: **2024-06-01**. Solo modelos sobre `y`.

| modelo | MAE | RMSE | sMAPE% | error suma temporada (por temporada) |
|---|---|---|---|---|
| arima | 177.8 | 191.5 | 147.1 | 2024_2025: 174.6% |
| prophet | 190.3 | 332.0 | 64.4 | 2024_2025: 238.4% |

> Nota: ARIMA/Prophet son baselines débiles para una serie estacional y dispersa; el desempeño fuerte del borrador venía del ensamble XGBoost+LSTM con covariables oceanográficas (pendiente de la ingesta OISST/GlobColour).