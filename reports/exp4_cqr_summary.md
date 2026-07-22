# Exp 4 — Pronóstico probabilístico con CQR

Corte **2020-07-01**; conformalización desde **2019-08-16**; test n=35538. Modelo: pool global sobre `log1p(y)` (Exp 3.2) con Conformalized Quantile Regression **por serie (Mondrian)** en espacio log, invertida a kg. Se reporta el ancho **mediano** (la media en kg la inflan unos pocos días pico al exponenciar el espacio log).

**Comparación de métodos conformal** (`split` = corrección constante; `normalized` = adaptativa, ensanche ∝ dispersión local; producción = normalized):

| método | nominal | cobertura marginal | ancho mediano (kg) |
|---|---|---|---|
| split | 80% | 92.1% | 4.0 |
| split | 90% | 97.9% | 152.2 |
| normalized | 80% | 92.1% | 4.0 |
| normalized | 90% | 97.9% | 149.3 |

**Cobertura condicional (intervalo 90%)**: durante MHW **0.981** (2289 días) vs. fuera de MHW **0.979**. Cuanto más cerca del 0.90 nominal en ambos regímenes, más honesto el intervalo en las temporadas anómalas.

**CRPS global**: 114.8 (menor es mejor).

| serie (especie@UE) | n test | cobertura 90% | ancho mediano (kg) | CRPS |
|---|---|---|---|---|
| abalone_black@er_scpp_ensenada | 986 | 99.4% | 76.9 | 4.19 |
| abalone_black@litoral_bc_sur | 728 | 99.3% | 18.5 | 0.93 |
| abalone_blue@er_scpp_ensenada | 2063 | 98.1% | 10.1 | 13.36 |
| abalone_blue@litoral_bc_sur | 727 | 98.8% | 3.7 | 0.7 |
| abalone_blue@punta_canoas | 1856 | 98.7% | 1.2 | 4.04 |
| abalone_red@litoral_bc_sur | 721 | 99.6% | 49.1 | 1.37 |
| lobster_red@er_el_chute | 1919 | 98.2% | 322.7 | 80.45 |
| lobster_red@er_isla_san_geronimo | 1914 | 99.8% | 322.7 | 53.36 |
| lobster_red@er_mortera_leyva | 1913 | 98.6% | 322.7 | 72.17 |
| lobster_red@er_regasa | 2056 | 99.8% | 322.7 | 73.36 |
| lobster_red@er_scpp_ensenada | 2055 | 98.2% | 322.7 | 268.13 |
| lobster_red@isla_cedros | 1926 | 95.1% | 76.2 | 267.18 |
| lobster_red@litoral_bc_sur | 1644 | 99.8% | 48.1 | 268.04 |
| lobster_red@pabellon_sq | 2053 | 95.6% | 41.6 | 46.25 |
| lobster_red@punta_canoas | 1918 | 96.4% | 34.4 | 29.87 |
| lobster_red@rocas_san_martin | 2054 | 99.5% | 59.4 | 23.67 |
| urchin_red@er_el_chute | 763 | 100.0% | 635.4 | 6.1 |
| urchin_red@er_mortera_leyva | 2066 | 98.0% | 1307.0 | 124.98 |
| urchin_red@er_regasa | 2069 | 96.5% | 1492.9 | 184.41 |
| urchin_red@er_scpp_ensenada | 2069 | 91.8% | 2042.1 | 547.87 |
| urchin_red@punta_canoas | 2038 | 99.7% | 125.8 | 3.87 |

> Figuras: `reports/figures/exp4_cqr_fan_chart.png` (serie insignia lobster_red@litoral_bc_sur) y `reports/figures/exp4_cqr_fan_grid_2020-07-01.png` (grid observado-vs-pronosticado de las 7 UEs de langosta, bandas 80/90% + mediana + observado). La CQR da la garantía de cobertura marginal sin asumir la forma de la distribución; es el producto operativo para COBI (rango esperado de captura).