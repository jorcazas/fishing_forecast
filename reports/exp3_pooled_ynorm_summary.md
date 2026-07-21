# Exp 3.2 — Pooling con `y` normalizada por serie

Corte **2020-07-01**. 16 series `(especie, UE)`. RMSE diario (kg), tras invertir la transformación. `p_raw`=pool sobre y cruda, `p_log`=pool sobre log1p(y), `p_z`=pool sobre z-score por serie (stats de train).

| serie (especie@UE) | escala µ_train | específico | p_raw | p_log | p_z |
|---|---|---|---|---|---|
| abalone_black@er_scpp_ensenada | 0.6 | 16.6 | 94.1 | 16.7 | 16.6 |
| abalone_black@litoral_bc_sur | 1.8 | 65.3 | 20.2 | 5.0 | 5.2 |
| abalone_blue@er_scpp_ensenada | 27.2 | 295.8 | 138.2 | 98.5 | 108.2 |
| abalone_blue@litoral_bc_sur | 4.1 | 7.5 | 36.1 | 4.1 | 5.6 |
| abalone_red@litoral_bc_sur | 2.2 | 9.4 | 32.0 | 2.9 | 3.6 |
| lobster_red@er_el_chute | 87.9 | 340.6 | 289.8 | 323.2 | 291.6 |
| lobster_red@er_isla_san_geronimo | 48.6 | 179.5 | 180.7 | 186.8 | 170.1 |
| lobster_red@er_mortera_leyva | 96.9 | 403.8 | 283.0 | 294.9 | 288.3 |
| lobster_red@er_regasa | 54.8 | 269.1 | 221.5 | 246.4 | 212.4 |
| lobster_red@er_scpp_ensenada | 371.1 | 786.8 | 694.5 | 856.0 | 738.8 |
| lobster_red@isla_cedros | 273.4 | 1076.8 | 864.0 | 1002.2 | 834.5 |
| lobster_red@litoral_bc_sur | 379.2 | 712.3 | 269.4 | 190.7 | 487.2 |
| urchin_red@er_el_chute | 14.6 | 66.4 | 90.2 | 14.4 | 36.3 |
| urchin_red@er_mortera_leyva | 130.9 | 390.9 | 359.6 | 370.2 | 362.7 |
| urchin_red@er_regasa | 204.0 | 752.8 | 564.0 | 584.7 | 578.3 |
| urchin_red@er_scpp_ensenada | 354.2 | 1443.8 | 1284.3 | 1387.3 | 1308.0 |

**Gana/empata vs específico (RMSE):** p_raw 11/16 (0.69), p_log 13/16 (0.81), p_z 16/16 (1.0).

> Normalizar el objetivo por serie evita que la escala de langosta (cientos de kg) domine el loss del pool sobre abulón/erizo (unidades). Sin leakage: las stats del z-score salen solo de train.