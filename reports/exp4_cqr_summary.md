# Exp 4 — Pronóstico probabilístico con CQR

Corte **2020-07-01**; conformalización desde **2019-08-15**; test n=2186. Modelo: pool global sobre `log1p(y)` (Exp 3.2) con Conformalized Quantile Regression **por serie (Mondrian)** en espacio log, invertida a kg. Se reporta el ancho **mediano** (la media en kg la inflan unos pocos días pico al exponenciar el espacio log).

**Comparación de métodos conformal** (`split` = corrección constante; `normalized` = adaptativa, ensanche ∝ dispersión local; producción = normalized):

| método | nominal | cobertura marginal | ancho mediano (kg) |
|---|---|---|---|
| split | 80% | 80.9% | 0.3 |
| split | 90% | 84.9% | 7.7 |
| normalized | 80% | 80.8% | 0.3 |
| normalized | 90% | 85.9% | 7.7 |

**Cobertura condicional (intervalo 90%)**: durante MHW **0.891** (313 días) vs. fuera de MHW **0.854**. Cuanto más cerca del 0.90 nominal en ambos regímenes, más honesto el intervalo en las temporadas anómalas.

**CRPS global**: 217.88 (menor es mejor).

| serie (especie@UE) | n test | cobertura 90% | ancho mediano (kg) | CRPS |
|---|---|---|---|---|
| abalone_black@litoral_bc_sur | 364 | 99.2% | 3.6 | 0.6 |
| abalone_blue@litoral_bc_sur | 364 | 99.2% | 4.0 | 0.48 |
| abalone_red@litoral_bc_sur | 364 | 99.5% | 4.5 | 0.32 |
| lobster_red@isla_cedros | 545 | 91.9% | 20.7 | 541.73 |
| lobster_red@litoral_bc_sur | 549 | 53.4% | 21.4 | 328.85 |

> Figura: `reports/figures/exp4_cqr_fan_chart.png` (bandas 80/90% + mediana + observado para lobster_red@litoral_bc_sur). La CQR da la garantía de cobertura marginal sin asumir la forma de la distribución; es el producto operativo para COBI (rango esperado de captura).