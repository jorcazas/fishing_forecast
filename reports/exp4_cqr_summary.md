# Exp 4 — Pronóstico probabilístico con CQR

Corte **2020-07-01**; conformalización desde **2019-08-15**; test n=2186. Modelo: pool global sobre `log1p(y)` (Exp 3.2) envuelto en Conformalized Quantile Regression (`mapie`), con regresores cuantílicos XGBoost en espacio log invertidos a kg.

| intervalo nominal | cobertura empírica | ancho medio (kg) |
|---|---|---|
| 80% | 94.1% | 8539.4 |
| 90% | 96.8% | 932.0 |

**Cobertura condicional (intervalo 90%)**: durante MHW **None** (0 días) vs. fuera de MHW **0.968**. Cuanto más cerca del 0.90 nominal en ambos regímenes, más honesto el intervalo en las temporadas anómalas.

**CRPS global**: 291.14 (menor es mejor).

| serie (especie@UE) | n test | cobertura 90% | CRPS |
|---|---|---|---|
| abalone_black@litoral_bc_sur | 364 | 98.9% | 0.6 |
| abalone_blue@litoral_bc_sur | 364 | 98.9% | 0.48 |
| abalone_red@litoral_bc_sur | 364 | 99.2% | 0.32 |
| lobster_red@isla_cedros | 545 | 89.7% | 736.3 |
| lobster_red@litoral_bc_sur | 549 | 99.3% | 427.38 |

> Figura: `reports/figures/exp4_cqr_fan_chart.png` (bandas 80/90% + mediana + observado para lobster_red@litoral_bc_sur). La CQR da la garantía de cobertura marginal sin asumir la forma de la distribución; es el producto operativo para COBI (rango esperado de captura).