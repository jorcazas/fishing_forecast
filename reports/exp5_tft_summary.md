# Exp 5 — Temporal Fusion Transformer (prueba de techo)

Modelo global cuantílico (`pytorch-forecasting` 1.7), corte **2024-06-01**, 8 épocas
(batch 256, `val_loss` aún bajando al detenerse por presupuesto de cómputo), test n=8717.
Métricas y rejilla de cuantiles **idénticas a Exp 4 (CQR)** para comparación directa.

## TFT vs. CQR (mismo corte, mismo test)

| métrica (intervalo 90% salvo nota) | TFT | CQR (pool log + conformal) |
|---|---|---|
| MAE diario (kg) | 148.7 | — (XGBoost-SHAP puntual: 103.5) |
| RMSE diario (kg) | 601.3 | — (XGBoost-SHAP: 155.6) |
| sMAPE (%) | 200.0 | — (XGBoost-SHAP: 64.1) |
| cobertura 80% | 4.2% | 93.3% |
| cobertura 90% | 5.7% | 96.7% |
| ancho mediano 90% (kg) | 1.2 | 137.8 |
| **CRPS global** | **128.9** | **130.8** |
| langosta@SQ — cobertura 90% | 41.8% | 96.2% |
| langosta@SQ — CRPS | 50.1 | 85.0 |

## Lectura honesta

- **El CRPS es prácticamente empatado** (128.9 vs 130.8; en langosta@SQ el TFT es incluso
  mejor, 50.1 vs 85.0). Es decir, la *nitidez* probabilística cruda del TFT es competitiva:
  el centro de la distribución es razonable (el fan chart de langosta@SQ muestra bandas
  sensatas **en temporada**).
- **La cobertura NO es comparable manzana con manzana.** Los cuantiles del TFT no están
  conformalizados ni calibrados solo-en-temporada (lo que sí hace la CQR). Con este
  presupuesto colapsan (ancho mediano del 90% = **1.2 kg**) y, sobre todo, los muchos días
  **fuera de temporada** con captura 0 caen apenas por debajo de una cota inferior
  ligeramente positiva → cobertura marginal 5.7%. La capa conformal-en-temporada de la CQR
  es exactamente lo que entrega el 96.7%.
- **En puntual**, el MAE es similar al XGBoost-SHAP (148.7 vs 103.5) pero el RMSE y el sMAPE
  son claramente peores (601 vs 156; 200% vs 64%): el TFT falla más en los días pico.

## Veredicto (criterio de éxito de Fase 5)

Con ~200-1 100 observaciones por serie ---uno a dos órdenes de magnitud por debajo del
umbral (~10 000 obs/grupo) donde los Transformers empiezan a pagar--- y 8 épocas de
entrenamiento, **el TFT no supera al pool XGBoost + CQR**: su nitidez (CRPS) es a lo sumo
equivalente, su desempeño puntual es peor en los días pico, y como producto probabilístico
calibrado queda muy por detrás porque carece de la envoltura conformal. Es el **hallazgo
metodológico esperado**: con estos datos la complejidad extra no paga; la palanca sigue
siendo *más datos*. Ver `docs/decisions/ADR-0002-tft.md`.

## Caveats y siguientes pasos posibles

- **8 épocas** (limitado por cómputo; ~1.75 min/época en esta máquina). Más épocas podrían
  mejorar algo la calibración, pero es improbable que inviertan el veredicto (CRPS ya empata).
- Comparación **justa de cobertura**: envolver los cuantiles del TFT en la MISMA CQR
  (conformal-en-temporada) — mediría "TFT+conformal" vs "XGBoost+conformal".
- Solo se corrió el **corte 2024** (el test de 2020 abarca 6 años → muchas más ventanas de
  predicción, bastante más lento). Figura: `reports/figures/exp5_tft_fan_chart_2024-06-01.png`.
