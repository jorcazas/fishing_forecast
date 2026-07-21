# Exp 4 — Pronóstico probabilístico con CQR

Corte **2020-07-01**; conformalización desde **2019-08-16**; test n=6950. Modelo: pool global sobre `log1p(y)` (Exp 3.2) con Conformalized Quantile Regression **por serie (Mondrian)** en espacio log, invertida a kg. Se reporta el ancho **mediano** (la media en kg la inflan unos pocos días pico al exponenciar el espacio log).

**Comparación de métodos conformal** (`split` = corrección constante; `normalized` = adaptativa, ensanche ∝ dispersión local; producción = normalized):

| método | nominal | cobertura marginal | ancho mediano (kg) |
|---|---|---|---|
| split | 80% | 92.0% | 123.4 |
| split | 90% | 92.9% | 474.0 |
| normalized | 80% | 92.0% | 123.4 |
| normalized | 90% | 92.9% | 473.9 |

**Cobertura condicional (intervalo 90%)**: durante MHW **0.914** (546 días) vs. fuera de MHW **0.93**. Cuanto más cerca del 0.90 nominal en ambos regímenes, más honesto el intervalo en las temporadas anómalas.

**CRPS global**: 188.29 (menor es mejor).

| serie (especie@UE) | n test | cobertura 90% | ancho mediano (kg) | CRPS |
|---|---|---|---|---|
| abalone_black@litoral_bc_sur | 364 | 99.2% | 17.4 | 1.24 |
| abalone_blue@er_scpp_ensenada | 395 | 95.7% | 6.1 | 31.92 |
| abalone_blue@litoral_bc_sur | 364 | 98.4% | 2.7 | 0.46 |
| abalone_red@litoral_bc_sur | 364 | 100.0% | 17.4 | 0.99 |
| lobster_red@er_el_chute | 547 | 98.4% | 669.1 | 90.69 |
| lobster_red@er_isla_san_geronimo | 547 | 99.8% | 617.8 | 52.88 |
| lobster_red@er_mortera_leyva | 547 | 98.7% | 658.6 | 95.79 |
| lobster_red@er_regasa | 548 | 99.6% | 663.8 | 70.97 |
| lobster_red@er_scpp_ensenada | 548 | 97.3% | 1156.5 | 361.71 |
| lobster_red@isla_cedros | 545 | 91.0% | 29.0 | 358.63 |
| lobster_red@litoral_bc_sur | 549 | 46.8% | 32.7 | 171.25 |
| urchin_red@er_mortera_leyva | 540 | 97.2% | 1476.3 | 158.98 |
| urchin_red@er_regasa | 545 | 98.9% | 3992.8 | 163.76 |
| urchin_red@er_scpp_ensenada | 547 | 86.8% | 2477.7 | 845.35 |

> Figura: `reports/figures/exp4_cqr_fan_chart.png` (bandas 80/90% + mediana + observado para lobster_red@litoral_bc_sur). La CQR da la garantía de cobertura marginal sin asumir la forma de la distribución; es el producto operativo para COBI (rango esperado de captura).