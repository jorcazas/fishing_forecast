# Exp 2.3 — Selección de features con SHAP (langosta-SQ)

Corte **2020-07-01**. Features: completo **35** → podado **16** (mean|SHAP| ≥ 1% del total).

| modelo | n features | MAE | RMSE | sMAPE% | error temp. 2021-22 |
|---|---|---|---|---|---|
| completo | 35 | 459.0 | 712.3 | 153.0 | 358.6% |
| podado | 16 | 387.5 | 673.0 | 132.1 | 390.3% |

Features conservadas (por SHAP): doy_sin, in_season, bbp_roll90_lag90, y_lag365, doy_cos, sst_lag90, y_lag730, cdm_roll90_lag90, bbp_lag90, zsd_roll90_lag90, spm_roll90_lag90, sst_roll90_lag90, mhw_category_roll365_lag90, chl_lag90, mhw_intensity_lag90, cdm_lag90

> Figura: `reports/figures/exp2_shap_selection_shap_bar.png`. La poda por SHAP prueba si reducir features mitiga el sobreajuste observado en Exp 2 (pocas temporadas).