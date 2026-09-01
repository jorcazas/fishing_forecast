# Exp 2.5 — Features de Fase 2 sobre el modelo global (corte 2024-06-01)

Mismo pool `log1p(y)`, mismos hiperparámetros y semilla; solo cambia la matriz: **61 → 68** features.

Añadidas: `bbp_anom_lag90`, `chl_anom_lag90`, `mhw_intensity_roll365_lag90__x__in_season`, `sst_anom_lag90`, `sst_anomaly_lag90__x__chl_lag90`, `y_lag365__x__sst_roll365_lag90`, `zsd_anom_lag90`

**v2 gana o empata en RMSE en 10/28 series (0.36).** RMSE medio entre series: 633.4 → 633.7 kg/día.

| serie | RMSE v1 | RMSE v2 | sMAPE v1 | sMAPE v2 |
|---|---|---|---|---|
| abalone_blue@er_scpp_ensenada | 58.6 | 58.6 | 200.0 | 200.0 |
| abalone_blue@la_purisima | 173.6 | 173.6 | 56.9 | 44.0 |
| abalone_blue@punta_canoas | 36.4 | 36.4 | 167.5 | 163.8 |
| lobster_red@abreojos_progreso | 1074.3 | 1074.3 | 134.5 | 135.2 |
| lobster_red@abreojos_punta | 793.7 | 793.6 | 115.4 | 116.6 |
| lobster_red@abreojos_san_ignacio | 781.6 | 781.6 | 114.9 | 115.7 |
| lobster_red@er_el_chute | 369.4 | 369.5 | 199.8 | 196.1 |
| lobster_red@er_isla_san_geronimo | 208.2 | 208.2 | 199.8 | 196.0 |
| lobster_red@er_mortera_leyva | 286.7 | 286.9 | 199.6 | 198.0 |
| lobster_red@er_regasa | 239.5 | 240.2 | 199.2 | 199.3 |
| lobster_red@er_scpp_ensenada | 813.7 | 817.4 | 197.5 | 197.9 |
| lobster_red@isla_cedros | 1189.2 | 1188.4 | 119.8 | 136.3 |
| lobster_red@la_purisima | 1707.4 | 1707.4 | 109.0 | 102.9 |
| lobster_red@litoral_bc_sur | 109.5 | 117.1 | 137.0 | 136.3 |
| lobster_red@magdalena_bahia | 41.1 | 41.1 | 163.4 | 147.2 |
| lobster_red@magdalena_chale | 187.1 | 187.1 | 121.3 | 116.9 |
| lobster_red@magdalena_san_carlos | 90.6 | 90.6 | 153.8 | 136.4 |
| lobster_red@pabellon_sq | 128.2 | 128.1 | 150.1 | 146.9 |
| lobster_red@punta_canoas | 114.3 | 114.0 | 130.2 | 119.8 |
| lobster_red@rocas_san_martin | 119.9 | 119.9 | 150.1 | 146.9 |
| lobster_red@vizcaino_asuncion | 1574.9 | 1574.9 | 141.0 | 136.6 |
| lobster_red@vizcaino_emancipacion | 1620.6 | 1620.5 | 130.4 | 130.4 |
| lobster_red@vizcaino_natividad | 1076.1 | 1076.0 | 130.5 | 130.5 |
| lobster_red@vizcaino_tortugas | 2270.7 | 2270.6 | 130.5 | 130.5 |
| urchin_red@er_mortera_leyva | 352.0 | 352.0 | 199.8 | 199.8 |
| urchin_red@er_regasa | 902.9 | 902.9 | 199.9 | 199.9 |
| urchin_red@er_scpp_ensenada | 1381.4 | 1381.5 | 199.9 | 199.9 |
| urchin_red@punta_canoas | 32.5 | 32.5 | 200.0 | 200.0 |

> La climatología de las anomalías se estima solo con datos anteriores al corte (`fit_climatology(train_end=corte)`) y dentro de cada serie; las interacciones están declaradas con justificación ecológica en `configs/features.yaml`.