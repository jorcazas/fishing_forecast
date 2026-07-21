# Exp 4b — Endurecer los cuantílicos base de la CQR (Optuna sobre pinball)

Corte **2020-07-01**. Optuna (40 trials) minimizando la pérdida pinball de validación (log): **0.146** (default) → **0.139** (afinado).

`best_params`: `{'max_depth': 2, 'learning_rate': 0.12092920245034901, 'n_estimators': 250, 'min_child_weight': 19.167888492996873, 'reg_lambda': 0.5669772921893893, 'reg_alpha': 0.122540317768369, 'subsample': 0.6968662446992331, 'colsample_bytree': 0.9967770542573963}`

| variante | nivel | cobertura | ancho mediano (kg) | ancho p90 (kg) | CRPS |
|---|---|---|---|---|---|
| default | 80% | 80.8% | 0.3 | 2248.5 | 217.88 |
| default | 90% | 85.9% | 7.7 | 9776.1 | 217.88 |
| afinado | 80% | 83.2% | 0.1 | 4700.0 | 2825.25 |
| afinado | 90% | 84.7% | 4.0 | 47039.0 | 2825.25 |

**Cobertura 90% y ancho p90 por serie (default → afinado):**

| serie | cobertura 90% | ancho p90 (kg) |
|---|---|---|
| abalone_black@litoral_bc_sur | 99.2% → 99.7% | 17.0 → 7.6 |
| abalone_blue@litoral_bc_sur | 99.2% → 99.5% | 17.6 → 13.8 |
| abalone_red@litoral_bc_sur | 99.5% → 100.0% | 18.6 → 6.1 |
| lobster_red@isla_cedros | 91.9% → 94.1% | 6966.3 → 83731.7 |
| lobster_red@litoral_bc_sur | 53.4% → 45.4% | 36002.8 → 401984.9 |

> El ancho p90 (días pico) es el objetivo: afinar los cuantílicos con pinball busca estrecharlo sin perder cobertura. El objetivo se valida en el 30% final del pre-corte; test intacto (≥ corte).