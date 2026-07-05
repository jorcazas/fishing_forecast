# Exp 3 — Modelo global multi-especie / multi-UE vs específico

Corte **2020-07-01**. 5 series `(especie, UE)`.

| serie (especie@UE) | días captura test | RMSE global | RMSE específico | gana |
|---|---|---|---|---|
| abalone_black@litoral_bc_sur | 3 | 24.6 | 52.8 | global |
| abalone_blue@litoral_bc_sur | 6 | 24.1 | 6.0 | específico |
| abalone_red@litoral_bc_sur | 2 | 24.0 | 8.9 | específico |
| lobster_red@isla_cedros | 85 | 893.6 | 898.7 | global |
| lobster_red@litoral_bc_sur | 251 | 618.9 | 558.3 | específico |

**Global gana/empata en 2/5 series** (rate 0.4); criterio ≥0.60 (PLAN §3.1): **False**.

Top features del modelo global: doy_sin, y_lag365, species_lobster_red, doy_cos, mhw_category_roll90_lag90, sst_roll90_lag90, sst_lag90, sst_anomaly_roll90_lag90

> UEs: San Quintín (~30.5°N) e Isla Cedros (~28°N) — gradiente biogeográfico. Bboxes aproximados (oficina de arribo), pendiente polígono TURF. Test de abulón diminuto.