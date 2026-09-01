# Exp 2.5 — Features de Fase 2 sobre el modelo global (corte 2020-07-01)

Mismo pool `log1p(y)`, mismos hiperparámetros y semilla; solo cambia la matriz: **61 → 68** features.

Añadidas: `bbp_anom_lag90`, `chl_anom_lag90`, `mhw_intensity_roll365_lag90__x__in_season`, `sst_anom_lag90`, `sst_anomaly_lag90__x__chl_lag90`, `y_lag365__x__sst_roll365_lag90`, `zsd_anom_lag90`

**v2 gana o empata en RMSE en 25/33 series (0.76).** RMSE medio entre series: 435.0 → 434.1 kg/día.

| serie | RMSE v1 | RMSE v2 | sMAPE v1 | sMAPE v2 |
|---|---|---|---|---|
| abalone_black@er_scpp_ensenada | 16.7 | 16.7 | 168.3 | 168.3 |
| abalone_black@litoral_bc_sur | 5.0 | 5.0 | 126.6 | 109.3 |
| abalone_blue@er_scpp_ensenada | 98.6 | 98.5 | 200.0 | 200.0 |
| abalone_blue@la_purisima | 126.1 | 126.1 | 13.0 | 14.8 |
| abalone_blue@litoral_bc_sur | 4.1 | 4.1 | 178.6 | 180.1 |
| abalone_blue@punta_canoas | 33.9 | 33.9 | 186.0 | 189.4 |
| abalone_red@litoral_bc_sur | 2.8 | 2.8 | 182.8 | 180.3 |
| lobster_red@abreojos_progreso | 783.4 | 783.4 | 89.5 | 80.7 |
| lobster_red@abreojos_punta | 664.2 | 664.2 | 85.0 | 76.7 |
| lobster_red@abreojos_san_ignacio | 499.1 | 499.1 | 83.0 | 74.7 |
| lobster_red@er_el_chute | 323.3 | 322.4 | 192.9 | 190.0 |
| lobster_red@er_isla_san_geronimo | 187.6 | 187.3 | 191.7 | 188.3 |
| lobster_red@er_mortera_leyva | 295.3 | 295.3 | 191.6 | 187.2 |
| lobster_red@er_regasa | 246.2 | 245.0 | 199.3 | 198.9 |
| lobster_red@er_scpp_ensenada | 850.8 | 847.9 | 197.9 | 197.6 |
| lobster_red@isla_cedros | 1002.7 | 1001.0 | 199.8 | 199.7 |
| lobster_red@la_purisima | 1667.4 | 1667.4 | 79.4 | 74.4 |
| lobster_red@litoral_bc_sur | 256.9 | 236.6 | 149.8 | 149.5 |
| lobster_red@magdalena_bahia | 108.4 | 108.4 | 126.5 | 137.9 |
| lobster_red@magdalena_chale | 143.9 | 143.9 | 102.3 | 101.1 |
| lobster_red@magdalena_san_carlos | 75.3 | 75.3 | 120.9 | 132.8 |
| lobster_red@pabellon_sq | 149.4 | 149.3 | 105.3 | 112.6 |
| lobster_red@punta_canoas | 114.8 | 114.6 | 158.5 | 141.9 |
| lobster_red@rocas_san_martin | 84.2 | 84.1 | 128.4 | 129.2 |
| lobster_red@vizcaino_asuncion | 994.9 | 994.8 | 83.6 | 84.2 |
| lobster_red@vizcaino_emancipacion | 1085.6 | 1085.4 | 98.6 | 101.1 |
| lobster_red@vizcaino_natividad | 714.9 | 714.9 | 98.4 | 101.1 |
| lobster_red@vizcaino_tortugas | 1435.1 | 1435.0 | 98.4 | 101.0 |
| urchin_red@er_el_chute | 14.4 | 14.4 | 190.8 | 199.2 |
| urchin_red@er_mortera_leyva | 370.6 | 370.8 | 199.4 | 199.5 |
| urchin_red@er_regasa | 584.6 | 584.7 | 199.7 | 199.7 |
| urchin_red@er_scpp_ensenada | 1387.7 | 1387.9 | 199.7 | 199.8 |
| urchin_red@punta_canoas | 25.5 | 25.5 | 196.0 | 197.0 |

> La climatología de las anomalías se estima solo con datos anteriores al corte (`fit_climatology(train_end=corte)`) y dentro de cada serie; las interacciones están declaradas con justificación ecológica en `configs/features.yaml`.