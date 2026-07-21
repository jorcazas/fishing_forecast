# Exp 4 — Pronóstico probabilístico con CQR

Corte **2020-07-01**; conformalización desde **2019-08-16**; test n=25619. Modelo: pool global sobre `log1p(y)` (Exp 3.2) con Conformalized Quantile Regression **por serie (Mondrian)** en espacio log, invertida a kg. Se reporta el ancho **mediano** (la media en kg la inflan unos pocos días pico al exponenciar el espacio log).

**Comparación de métodos conformal** (`split` = corrección constante; `normalized` = adaptativa, ensanche ∝ dispersión local; producción = normalized):

| método | nominal | cobertura marginal | ancho mediano (kg) |
|---|---|---|---|
| split | 80% | 91.3% | 74.6 |
| split | 90% | 94.0% | 529.2 |
| normalized | 80% | 91.3% | 74.6 |
| normalized | 90% | 94.0% | 515.8 |

**Cobertura condicional (intervalo 90%)**: durante MHW **0.941** (1586 días) vs. fuera de MHW **0.94**. Cuanto más cerca del 0.90 nominal en ambos regímenes, más honesto el intervalo en las temporadas anómalas.

**CRPS global**: 158.13 (menor es mejor).

| serie (especie@UE) | n test | cobertura 90% | ancho mediano (kg) | CRPS |
|---|---|---|---|---|
| abalone_black@er_scpp_ensenada | 986 | 98.9% | 35.7 | 2.57 |
| abalone_black@litoral_bc_sur | 728 | 99.3% | 4.1 | 0.51 |
| abalone_blue@er_scpp_ensenada | 2063 | 98.0% | 1.2 | 13.44 |
| abalone_blue@litoral_bc_sur | 727 | 98.8% | 0.9 | 0.46 |
| abalone_red@litoral_bc_sur | 721 | 99.4% | 4.4 | 0.43 |
| lobster_red@er_el_chute | 1919 | 99.0% | 549.1 | 79.26 |
| lobster_red@er_isla_san_geronimo | 1914 | 100.0% | 598.3 | 55.22 |
| lobster_red@er_mortera_leyva | 1913 | 99.3% | 598.3 | 72.74 |
| lobster_red@er_regasa | 2056 | 100.0% | 884.3 | 71.52 |
| lobster_red@er_scpp_ensenada | 2055 | 97.6% | 861.5 | 248.03 |
| lobster_red@isla_cedros | 1926 | 93.7% | 21.2 | 258.27 |
| lobster_red@litoral_bc_sur | 1644 | 40.6% | 21.0 | 130.78 |
| urchin_red@er_el_chute | 763 | 100.0% | 105.3 | 4.47 |
| urchin_red@er_mortera_leyva | 2066 | 98.1% | 1212.9 | 188.45 |
| urchin_red@er_regasa | 2069 | 96.6% | 3036.9 | 231.8 |
| urchin_red@er_scpp_ensenada | 2069 | 91.6% | 1430.6 | 667.64 |

> Figura: `reports/figures/exp4_cqr_fan_chart.png` (bandas 80/90% + mediana + observado para lobster_red@litoral_bc_sur). La CQR da la garantía de cobertura marginal sin asumir la forma de la distribución; es el producto operativo para COBI (rango esperado de captura).