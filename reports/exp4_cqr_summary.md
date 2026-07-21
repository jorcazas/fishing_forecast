# Exp 4 — Pronóstico probabilístico con CQR

Corte **2024-06-01**; conformalización desde **2022-02-03**; test n=5953. Modelo: pool global sobre `log1p(y)` (Exp 3.2) con Conformalized Quantile Regression **por serie (Mondrian)** en espacio log, invertida a kg. Se reporta el ancho **mediano** (la media en kg la inflan unos pocos días pico al exponenciar el espacio log).

**Comparación de métodos conformal** (`split` = corrección constante; `normalized` = adaptativa, ensanche ∝ dispersión local; producción = normalized):

| método | nominal | cobertura marginal | ancho mediano (kg) |
|---|---|---|---|
| split | 80% | 92.1% | 27.3 |
| split | 90% | 95.0% | 176.9 |
| normalized | 80% | 92.1% | 27.3 |
| normalized | 90% | 95.0% | 176.9 |

**Cobertura condicional (intervalo 90%)**: durante MHW **0.902** (132 días) vs. fuera de MHW **0.951**. Cuanto más cerca del 0.90 nominal en ambos regímenes, más honesto el intervalo en las temporadas anómalas.

**CRPS global**: 220.64 (menor es mejor).

| serie (especie@UE) | n test | cobertura 90% | ancho mediano (kg) | CRPS |
|---|---|---|---|---|
| abalone_blue@er_scpp_ensenada | 632 | 98.9% | 8.6 | 6.16 |
| lobster_red@er_el_chute | 488 | 96.7% | 2.6 | 76.05 |
| lobster_red@er_isla_san_geronimo | 483 | 99.4% | 2.5 | 45.41 |
| lobster_red@er_mortera_leyva | 482 | 98.3% | 2.6 | 54.37 |
| lobster_red@er_regasa | 625 | 99.5% | 193.3 | 69.85 |
| lobster_red@er_scpp_ensenada | 624 | 99.4% | 185.1 | 233.24 |
| lobster_red@isla_cedros | 495 | 94.5% | 1.9 | 294.11 |
| lobster_red@litoral_bc_sur | 213 | 95.8% | 471.2 | 95.66 |
| urchin_red@er_mortera_leyva | 635 | 92.9% | 292.6 | 193.49 |
| urchin_red@er_regasa | 638 | 80.4% | 483.0 | 398.5 |
| urchin_red@er_scpp_ensenada | 638 | 91.8% | 1706.5 | 771.24 |

> Figura: `reports/figures/exp4_cqr_fan_chart.png` (bandas 80/90% + mediana + observado para lobster_red@litoral_bc_sur). La CQR da la garantía de cobertura marginal sin asumir la forma de la distribución; es el producto operativo para COBI (rango esperado de captura).