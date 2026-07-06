# Exp 3 — Modelo global multi-especie / multi-UE vs específico

Corte **2020-07-01**. 5 series `(especie, UE)`.

| serie (especie@UE) | días captura test | RMSE global | RMSE específico | gana |
|---|---|---|---|---|
| abalone_black@litoral_bc_sur | 3 | 35.4 | 52.0 | global |
| abalone_blue@litoral_bc_sur | 6 | 32.7 | 6.7 | específico |
| abalone_red@litoral_bc_sur | 2 | 34.9 | 7.3 | específico |
| lobster_red@isla_cedros | 85 | 889.0 | 914.1 | global |
| lobster_red@litoral_bc_sur | 251 | 742.2 | 659.7 | específico |

**Global gana/empata en 2/5 series** (rate 0.4); criterio ≥0.60 (PLAN §3.1): **False**.

Top features del modelo global: sst_roll90_lag90, species_lobster_red, y_lag365, doy_sin, zsd_roll90_lag90, bbp_roll90_lag90, doy_cos, sst_lag90

> UEs: San Quintín (~30.5°N) e Isla Cedros (~28°N) — gradiente biogeográfico. Bboxes aproximados (oficina de arribo), pendiente polígono TURF. Test de abulón diminuto.