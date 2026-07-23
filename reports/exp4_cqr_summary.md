# Exp 4 — Pronóstico probabilístico con CQR

Corte **2020-07-01**; conformalización desde **2019-08-16**; test n=53677. Modelo: pool global sobre `log1p(y)` (Exp 3.2) con Conformalized Quantile Regression **por serie (Mondrian)** en espacio log, invertida a kg. Se reporta el ancho **mediano** (la media en kg la inflan unos pocos días pico al exponenciar el espacio log).

**Comparación de métodos conformal** (`split` = corrección constante; `normalized` = adaptativa, ensanche ∝ dispersión local; producción = normalized):

| método | nominal | cobertura marginal | ancho mediano (kg) |
|---|---|---|---|
| split | 80% | 94.1% | 0.2 |
| split | 90% | 95.9% | 4.8 |
| normalized | 80% | 94.0% | 0.2 |
| normalized | 90% | 95.9% | 4.8 |

**Cobertura condicional (intervalo 90%)**: durante MHW **0.952** (4307 días) vs. fuera de MHW **0.96**. Cuanto más cerca del 0.90 nominal en ambos regímenes, más honesto el intervalo en las temporadas anómalas.

**CRPS global**: 135.57 (menor es mejor).

| serie (especie@UE) | n test | cobertura 90% | ancho mediano (kg) | CRPS |
|---|---|---|---|---|
| abalone_black@er_scpp_ensenada | 986 | 99.2% | 76.6 | 4.28 |
| abalone_black@litoral_bc_sur | 728 | 99.3% | 4.6 | 1.72 |
| abalone_blue@er_scpp_ensenada | 2063 | 98.4% | 81.6 | 15.77 |
| abalone_blue@la_purisima | 2085 | 98.7% | 0.5 | 12.25 |
| abalone_blue@litoral_bc_sur | 727 | 99.0% | 4.6 | 1.29 |
| abalone_blue@punta_canoas | 1856 | 98.7% | 5.2 | 4.62 |
| abalone_red@litoral_bc_sur | 721 | 99.4% | 4.5 | 1.32 |
| lobster_red@abreojos_progreso | 2067 | 93.6% | 0.7 | 176.7 |
| lobster_red@abreojos_punta | 1918 | 94.5% | 0.1 | 135.11 |
| lobster_red@abreojos_san_ignacio | 1915 | 96.5% | 0.1 | 79.51 |
| lobster_red@er_el_chute | 1919 | 96.4% | 245.2 | 82.67 |
| lobster_red@er_isla_san_geronimo | 1914 | 97.3% | 249.8 | 54.73 |
| lobster_red@er_mortera_leyva | 1913 | 96.5% | 239.5 | 74.53 |
| lobster_red@er_regasa | 2056 | 95.8% | 410.2 | 74.38 |
| lobster_red@er_scpp_ensenada | 2055 | 95.7% | 791.5 | 267.02 |
| lobster_red@isla_cedros | 1926 | 92.4% | 2.5 | 290.37 |
| lobster_red@la_purisima | 1919 | 95.9% | 0.1 | 275.58 |
| lobster_red@litoral_bc_sur | 1644 | 99.8% | 6.6 | 143.85 |
| lobster_red@pabellon_sq | 2053 | 94.8% | 1.8 | 40.66 |
| lobster_red@punta_canoas | 1918 | 96.1% | 21.4 | 29.17 |
| lobster_red@rocas_san_martin | 2054 | 96.6% | 1.8 | 20.62 |
| lobster_red@vizcaino_asuncion | 2068 | 92.8% | 0.2 | 236.58 |
| lobster_red@vizcaino_emancipacion | 2055 | 91.5% | 0.1 | 295.51 |
| lobster_red@vizcaino_natividad | 2056 | 93.1% | 0.1 | 174.26 |
| lobster_red@vizcaino_tortugas | 2056 | 91.0% | 0.1 | 366.73 |
| urchin_red@er_el_chute | 763 | 100.0% | 115.4 | 2.1 |
| urchin_red@er_mortera_leyva | 2066 | 97.5% | 1286.0 | 115.96 |
| urchin_red@er_regasa | 2069 | 97.8% | 2304.0 | 179.35 |
| urchin_red@er_scpp_ensenada | 2069 | 91.7% | 1087.4 | 476.0 |
| urchin_red@punta_canoas | 2038 | 99.6% | 17.9 | 3.28 |

> Figuras: `reports/figures/exp4_cqr_fan_chart.png` (serie insignia lobster_red@litoral_bc_sur) y `reports/figures/exp4_cqr_fan_grid_2020-07-01.png` (grid observado-vs-pronosticado de las 7 UEs de langosta, bandas 80/90% + mediana + observado). La CQR da la garantía de cobertura marginal sin asumir la forma de la distribución; es el producto operativo para COBI (rango esperado de captura).