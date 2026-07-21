# Exp 1 — Baseline estadístico (langosta-SQ)

Corte de test: **2020-07-01**. Solo modelos sobre `y`.

| modelo | MAE | RMSE | sMAPE% | error suma temporada (por temporada) |
|---|---|---|---|---|
| arima | 330.8 | 383.8 | 153.3 | 2020_2021: -26.4%; 2021_2022: 250.8%; 2022_2023: 467.7%; 2023_2024: 646.3%; 2024_2025: 612.0% |
| prophet | 454.7 | 685.1 | 158.8 | 2020_2021: 62.4%; 2021_2022: 378.4%; 2022_2023: 707.4%; 2023_2024: 1005.8%; 2024_2025: 966.3% |

> Nota: ARIMA/Prophet son baselines débiles para una serie estacional y dispersa; el desempeño fuerte del borrador venía del ensamble XGBoost+LSTM con covariables oceanográficas (pendiente de la ingesta OISST/GlobColour).