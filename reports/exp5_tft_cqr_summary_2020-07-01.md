# Exp 5b — TFT + CQR (comparación justa de cobertura)

Cuantiles del TFT envueltos en la **misma** corrección conformal que Exp 4 (Mondrian por serie, en temporada, espacio log). Corte **2020-07-01**, test n=59678, conf n=7382. A igual cobertura, el **ancho** discrimina la nitidez del cuantil base.

| método | cobertura 90% | ancho mediano 90% (kg) | ancho p90 90% (kg) | CRPS |
|---|---|---|---|---|
| **TFT + conformal** | 70.8% | 3.0 | 2953.6 | 116.18 |
| XGBoost + conformal (Exp 4) | *(ver `reports/exp4_cqr_summary.md`, mismo corte)* | | | |

> Si el TFT necesita intervalos más anchos que el XGBoost para la misma cobertura, sus cuantiles base son menos nítidos → el XGBoost+CQR sigue siendo el producto. Ver `docs/decisions/ADR-0002-tft.md` y `reports/exp5_tft_summary.md`.