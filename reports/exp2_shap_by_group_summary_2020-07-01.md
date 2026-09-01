# Exp 2.4 — SHAP condicional por grupo (corte 2020-07-01)

Modelo: el pool sobre `log1p(y)` de Exp 3.2 (producción), 34 series, 43406 filas de train, 61 features de las cuales 26 son one-hot de identidad.

**Cuota de atribución que se va a la identidad (especie/UE): 38.4%** del `mean(|SHAP|)` total. Es el indicador directo de si el pool usa la identidad como atajo (cuota alta ⇒ submodelos disfrazados) o si comparte estructura ambiental (cuota baja ⇒ el pooling sí transfiere).

| grupo | filas train | cuota identidad | top-3 features |
|---|---|---|---|
| abalone_black | 2554 | 57.8% | species_lobster_red, species_abalone_black, economic_unit_litoral_bc_sur |
| abalone_blue | 6373 | 47.7% | species_lobster_red, in_season, economic_unit_litoral_bc_sur |
| abalone_red | 1277 | 58.5% | species_lobster_red, economic_unit_litoral_bc_sur, in_season |
| lobster_red | 26817 | 32.7% | in_season, species_lobster_red, y_lag365 |
| urchin_red | 6385 | 37.3% | species_lobster_red, y_lag365, in_season |
| lobster_red@abreojos_progreso | 1277 | 30.2% | in_season, species_lobster_red, y_lag365 |
| lobster_red@abreojos_punta | 1277 | 30.2% | in_season, species_lobster_red, y_lag365 |
| lobster_red@abreojos_san_ignacio | 1277 | 30.2% | in_season, species_lobster_red, y_lag365 |
| lobster_red@er_el_chute | 1277 | 28.6% | in_season, species_lobster_red, y_lag365 |
| lobster_red@er_isla_san_geronimo | 1277 | 28.2% | in_season, species_lobster_red, y_lag365 |
| lobster_red@er_mortera_leyva | 1277 | 29.8% | in_season, species_lobster_red, y_lag365 |
| lobster_red@er_regasa | 1277 | 33.6% | in_season, economic_unit_er_regasa, y_lag365 |
| lobster_red@er_scpp_ensenada | 1277 | 40.5% | economic_unit_er_scpp_ensenada, in_season, y_lag365 |
| lobster_red@isla_cedros | 1277 | 30.2% | in_season, species_lobster_red, y_lag365 |
| lobster_red@la_purisima | 1277 | 30.9% | in_season, species_lobster_red, y_lag365 |
| lobster_red@litoral_bc_sur | 1277 | 41.1% | economic_unit_litoral_bc_sur, y_lag365, in_season |
| lobster_red@magdalena_bahia | 1277 | 28.8% | in_season, species_lobster_red, economic_unit_litoral_bc_sur |
| lobster_red@magdalena_chale | 1277 | 31.4% | in_season, species_lobster_red, y_lag365 |
| lobster_red@magdalena_san_carlos | 1277 | 28.8% | in_season, species_lobster_red, economic_unit_litoral_bc_sur |
| lobster_red@pabellon_sq | 1277 | 36.4% | in_season, species_lobster_red, economic_unit_pabellon_sq |
| lobster_red@punta_canoas | 1277 | 29.8% | in_season, species_lobster_red, y_lag365 |
| lobster_red@rocas_san_martin | 1277 | 29.4% | in_season, species_lobster_red, y_lag365 |
| lobster_red@vizcaino_asuncion | 1277 | 29.4% | in_season, species_lobster_red, y_lag365 |
| lobster_red@vizcaino_emancipacion | 1277 | 32.3% | in_season, species_lobster_red, y_lag365 |
| lobster_red@vizcaino_natividad | 1277 | 32.3% | in_season, species_lobster_red, y_lag365 |
| lobster_red@vizcaino_tortugas | 1277 | 32.3% | in_season, species_lobster_red, y_lag365 |

**Divergencia entre especies** (Jensen-Shannon sobre las cuotas por feature; 0 = misma lógica, 1 = disjuntas): media **0.281**, máx **0.378** (abalone_red vs urchin_red).

| par | JS | coseno |
|---|---|---|
| abalone_red vs urchin_red | 0.378 | 0.767 |
| abalone_black vs urchin_red | 0.343 | 0.813 |
| abalone_black vs lobster_red | 0.308 | 0.808 |
| abalone_red vs lobster_red | 0.3 | 0.805 |
| abalone_black vs abalone_red | 0.29 | 0.914 |
| abalone_blue vs abalone_red | 0.261 | 0.927 |
| abalone_black vs abalone_blue | 0.26 | 0.928 |
| lobster_red vs urchin_red | 0.242 | 0.905 |

**Divergencia entre UEs de langosta** (Jensen-Shannon sobre las cuotas por feature; 0 = misma lógica, 1 = disjuntas): media **0.227**, máx **0.443** (lobster_red@litoral_bc_sur vs lobster_red@pabellon_sq).

| par | JS | coseno |
|---|---|---|
| lobster_red@litoral_bc_sur vs lobster_red@pabellon_sq | 0.443 | 0.64 |
| lobster_red@litoral_bc_sur vs lobster_red@vizcaino_asuncion | 0.439 | 0.651 |
| lobster_red@litoral_bc_sur vs lobster_red@magdalena_bahia | 0.434 | 0.689 |
| lobster_red@litoral_bc_sur vs lobster_red@magdalena_san_carlos | 0.434 | 0.689 |
| lobster_red@abreojos_progreso vs lobster_red@litoral_bc_sur | 0.433 | 0.656 |
| lobster_red@abreojos_punta vs lobster_red@litoral_bc_sur | 0.433 | 0.656 |
| lobster_red@abreojos_san_ignacio vs lobster_red@litoral_bc_sur | 0.433 | 0.656 |
| lobster_red@isla_cedros vs lobster_red@litoral_bc_sur | 0.427 | 0.705 |

> Figura: `reports/figures/exp2_shap_by_group_2020-07-01.png`. El SHAP se calcula sobre train (interpretación de lo aprendido), no sobre test.