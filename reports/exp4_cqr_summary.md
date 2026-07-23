# Exp 4 — Pronóstico probabilístico con CQR

Corte **2024-06-01**; conformalización desde **2022-05-02**; test n=15685. Modelo: pool global sobre `log1p(y)` (Exp 3.2) con Conformalized Quantile Regression **por serie (Mondrian)** en espacio log, invertida a kg. Se reporta el ancho **mediano** (la media en kg la inflan unos pocos días pico al exponenciar el espacio log).

**Comparación de métodos conformal** (`split` = corrección constante; `normalized` = adaptativa, ensanche ∝ dispersión local; producción = normalized):

| método | nominal | cobertura marginal | ancho mediano (kg) |
|---|---|---|---|
| split | 80% | 90.9% | 2.9 |
| split | 90% | 95.6% | 53.6 |
| normalized | 80% | 90.9% | 2.9 |
| normalized | 90% | 95.8% | 29.4 |

**Cobertura condicional (intervalo 90%)**: durante MHW **0.925** (1431 días) vs. fuera de MHW **0.961**. Cuanto más cerca del 0.90 nominal en ambos regímenes, más honesto el intervalo en las temporadas anómalas.

**CRPS global**: 216.84 (menor es mejor).

| serie (especie@UE) | n test | cobertura 90% | ancho mediano (kg) | CRPS |
|---|---|---|---|---|
| abalone_blue@er_scpp_ensenada | 632 | 98.9% | 23.3 | 6.25 |
| abalone_blue@la_purisima | 654 | 97.1% | 0.6 | 24.24 |
| abalone_blue@punta_canoas | 425 | 99.1% | 6.8 | 3.51 |
| lobster_red@abreojos_progreso | 636 | 99.8% | 271301.3 | 355.58 |
| lobster_red@abreojos_punta | 487 | 100.0% | 1088980.1 | 206.04 |
| lobster_red@abreojos_san_ignacio | 484 | 94.2% | 10.2 | 168.94 |
| lobster_red@er_el_chute | 488 | 98.6% | 1.0 | 81.85 |
| lobster_red@er_isla_san_geronimo | 483 | 100.0% | 1.0 | 53.34 |
| lobster_red@er_mortera_leyva | 482 | 99.8% | 1.0 | 56.58 |
| lobster_red@er_regasa | 625 | 99.5% | 9.0 | 78.34 |
| lobster_red@er_scpp_ensenada | 624 | 97.6% | 11.4 | 276.08 |
| lobster_red@isla_cedros | 495 | 91.7% | 77.7 | 316.77 |
| lobster_red@la_purisima | 488 | 89.1% | 15.6 | 508.05 |
| lobster_red@litoral_bc_sur | 213 | 96.2% | 364.9 | 122.76 |
| lobster_red@magdalena_bahia | 678 | 97.3% | 9.5 | 6.52 |
| lobster_red@magdalena_chale | 503 | 96.8% | 10.5 | 29.39 |
| lobster_red@magdalena_san_carlos | 527 | 95.1% | 4.5 | 14.26 |
| lobster_red@pabellon_sq | 622 | 99.5% | 22.1 | 34.76 |
| lobster_red@punta_canoas | 487 | 98.6% | 7.3 | 25.67 |
| lobster_red@rocas_san_martin | 623 | 99.7% | 19.6 | 28.85 |
| lobster_red@vizcaino_asuncion | 637 | 88.5% | 264.1 | 550.05 |
| lobster_red@vizcaino_emancipacion | 624 | 97.0% | 651.7 | 628.12 |
| lobster_red@vizcaino_natividad | 625 | 90.9% | 166.9 | 374.07 |
| lobster_red@vizcaino_tortugas | 625 | 96.6% | 608.3 | 837.04 |
| urchin_red@er_mortera_leyva | 635 | 91.0% | 385.2 | 130.41 |
| urchin_red@er_regasa | 638 | 81.5% | 447.5 | 370.72 |
| urchin_red@er_scpp_ensenada | 638 | 91.4% | 1401.4 | 512.85 |
| urchin_red@punta_canoas | 607 | 99.8% | 155.7 | 3.03 |

> Figuras: `reports/figures/exp4_cqr_fan_chart.png` (serie insignia lobster_red@litoral_bc_sur) y `reports/figures/exp4_cqr_fan_grid_2024-06-01.png` (grid observado-vs-pronosticado de las 7 UEs de langosta, bandas 80/90% + mediana + observado). La CQR da la garantía de cobertura marginal sin asumir la forma de la distribución; es el producto operativo para COBI (rango esperado de captura).