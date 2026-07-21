# Exp 3.2 — Pooling con `y` normalizada por serie

Corte **2020-07-01**. 14 series `(especie, UE)`. RMSE diario (kg), tras invertir la transformación. `p_raw`=pool sobre y cruda, `p_log`=pool sobre log1p(y), `p_z`=pool sobre z-score por serie (stats de train).

| serie (especie@UE) | escala µ_train | específico | p_raw | p_log | p_z |
|---|---|---|---|---|---|
| abalone_black@litoral_bc_sur | 1.8 | 52.0 | 47.5 | 6.9 | 6.9 |
| abalone_blue@er_scpp_ensenada | 27.2 | 370.3 | 210.5 | 164.4 | 162.1 |
| abalone_blue@litoral_bc_sur | 4.1 | 6.7 | 43.8 | 4.5 | 5.3 |
| abalone_red@litoral_bc_sur | 2.2 | 7.3 | 46.0 | 3.6 | 3.7 |
| lobster_red@er_el_chute | 87.9 | 321.7 | 316.0 | 358.8 | 312.6 |
| lobster_red@er_isla_san_geronimo | 48.6 | 169.8 | 175.7 | 182.1 | 163.7 |
| lobster_red@er_mortera_leyva | 96.9 | 329.7 | 324.0 | 357.1 | 323.5 |
| lobster_red@er_regasa | 54.8 | 270.6 | 237.0 | 283.4 | 241.9 |
| lobster_red@er_scpp_ensenada | 371.1 | 821.2 | 810.4 | 1074.6 | 770.3 |
| lobster_red@isla_cedros | 273.4 | 914.1 | 920.1 | 1126.0 | 891.8 |
| lobster_red@litoral_bc_sur | 379.2 | 659.7 | 313.5 | 328.2 | 393.4 |
| urchin_red@er_mortera_leyva | 130.9 | 402.1 | 400.9 | 415.7 | 381.5 |
| urchin_red@er_regasa | 204.0 | 661.2 | 460.8 | 453.4 | 466.2 |
| urchin_red@er_scpp_ensenada | 354.2 | 1732.5 | 1748.9 | 1947.7 | 1756.8 |

**Gana/empata vs específico (RMSE):** p_raw 9/14 (0.64), p_log 6/14 (0.43), p_z 13/14 (0.93).

> Normalizar el objetivo por serie evita que la escala de langosta (cientos de kg) domine el loss del pool sobre abulón/erizo (unidades). Sin leakage: las stats del z-score salen solo de train.