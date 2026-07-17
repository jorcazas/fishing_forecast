# Exp 3.2 — Pooling con `y` normalizada por serie

Corte **2020-07-01**. 5 series `(especie, UE)`. RMSE diario (kg), tras invertir la transformación. `p_raw`=pool sobre y cruda, `p_log`=pool sobre log1p(y), `p_z`=pool sobre z-score por serie (stats de train).

| serie (especie@UE) | escala µ_train | específico | p_raw | p_log | p_z |
|---|---|---|---|---|---|
| abalone_black@litoral_bc_sur | 1.8 | 52.0 | 35.4 | 7.0 | 11.4 |
| abalone_blue@litoral_bc_sur | 4.1 | 6.7 | 32.7 | 4.6 | 15.6 |
| abalone_red@litoral_bc_sur | 2.2 | 7.3 | 34.9 | 3.5 | 10.6 |
| lobster_red@isla_cedros | 273.4 | 914.1 | 889.0 | 1131.4 | 916.6 |
| lobster_red@litoral_bc_sur | 379.2 | 659.7 | 742.2 | 544.4 | 1013.9 |

**Gana/empata vs específico (RMSE):** p_raw 2/5 (0.4), p_log 4/5 (0.8), p_z 1/5 (0.2).

> Normalizar el objetivo por serie evita que la escala de langosta (cientos de kg) domine el loss del pool sobre abulón/erizo (unidades). Sin leakage: las stats del z-score salen solo de train.