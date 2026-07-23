# Exp 5b — TFT + CQR (comparación justa de cobertura, dos cortes)

Cuantiles del Temporal Fusion Transformer envueltos en la **misma** corrección conformal que
Exp 4 (Mondrian por serie, en temporada, espacio log), tras **reordenamiento monótono por fila**
(Chernozhukov et al. 2010) para eliminar el cruce de cuantiles. Esto mide *TFT+conformal* vs.
*XGBoost+conformal* manzana con manzana: a igual objetivo de cobertura, el **ancho** y el **CRPS**
discriminan la nitidez del cuantil base.

## Comparación de cortes (intervalo nominal 90%)

| | Corte 2020-07-01 (bache **fuera** del train) | | Corte 2024-06-01 (bache **dentro** del train) | |
|---|---|---|---|---|
| | **TFT+conformal** | XGBoost+CQR (Exp 4) | **TFT+conformal** | XGBoost+CQR (Exp 4) |
| cobertura 90% | 82.5% (sub-cubre) | 97.9% (sobre-cubre) | **91.2%** (≈nominal) | 96.7% (sobre-cubre) |
| cobertura 80% | 78.4% | 92.1% | 82.8% | 93.3% |
| ancho mediano 90% (kg) | 2.1 | 149.3 | 1.5 | 137.8 |
| **CRPS global** | **90.6** | 114.8 | **117.8** | 130.8 |
| n test | 35 538 | 35 538 | 8 717 | 8 717 |

## Lectura honesta (verdicto firme)

- **Marginalmente**, TFT+conformal es competitivo o ligeramente mejor en CRPS en ambos cortes
  (90.6 vs 114.8 @2020; 117.8 vs 130.8 @2024). Tras el reordenamiento de cuantiles queda
  **bien calibrado en distribución** (91.2% ≈ nominal 90% en el corte 2024), frente a la
  sobre-cobertura conservadora del XGBoost (96.7%). Fuera de distribución (corte 2020) **sub-cubre**
  (82.5%), mientras el XGBoost sobre-cubre (97.9%).
- **El reordenamiento de cuantiles fue material**: sin él, los cuantiles del TFT se cruzaban y la
  cobertura resultaba no-monótona (p. ej. 80% > 90%). `np.sort` por fila lo corrige y mejora la
  calibración marginal.
- **Por serie es inestable y a veces patológico.** La serie insignia langosta@San Quintín oscila
  40.6% (2020) → 62.0% (2024) de cobertura, con CRPS 114 → 213; el XGBoost se mantiene estable
  (99.8% → 96.2%, CRPS 268 → 85). Además `urchin_red@er_regasa` (corte 2020) reventó a un ancho
  mediano de ~9×10⁸ kg por desborde al exponenciar en escala log —un artefacto del cuantil base del
  TFT que la envoltura conformal no cura. Cada corrida entrena un TFT fresco, y el ruido corrida a
  corrida en las series pequeñas es alto.
- **Más cómputo no ayuda.** 8 vs 30 épocas no cambia el veredicto (firma de un problema limitado por
  datos, no por capacidad del modelo).

## Veredicto (criterio de éxito de Fase 5)

Con ~200-1 100 catch-days por serie ---uno a dos órdenes de magnitud por debajo del umbral
(~10 000 obs/grupo) donde los Transformers empiezan a pagar--- el TFT **no supera** al producto
XGBoost + CQR. Marginalmente están en **paridad** (CRPS a la par o ligeramente a favor del TFT), pero
el XGBoost+CQR es **estable, reproducible y sobre-cubre de forma segura** ---justo lo que necesita una
cooperativa---, mientras que el TFT es ruidoso por serie y ocasionalmente patológico. La capa
conformal, no la arquitectura, es la que entrega la calibración. La palanca sigue siendo *más datos*.
Ver `docs/decisions/ADR-0002-tft.md` y `reports/exp5_tft_summary.md`.
