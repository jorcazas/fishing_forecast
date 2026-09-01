# Exp 2.4 — SHAP condicional por grupo (corte 2024-06-01)

Modelo: el pool sobre `log1p(y)` de Exp 3.2 (producción), 34 series, 87399 filas de train, 61 features de las cuales 26 son one-hot de identidad.

**Cuota de atribución que se va a la identidad (especie/UE): 33.5%** del `mean(|SHAP|)` total. Es el indicador directo de si el pool usa la identidad como atajo (cuota alta ⇒ submodelos disfrazados) o si comparte estructura ambiental (cuota baja ⇒ el pooling sí transfiere).

| grupo | filas train | cuota identidad | top-3 features |
|---|---|---|---|
| abalone_black | 4268 | 54.8% | species_lobster_red, species_abalone_black, in_season |
| abalone_blue | 11393 | 45.9% | species_lobster_red, in_season, y_lag365 |
| abalone_red | 1998 | 52.9% | species_lobster_red, economic_unit_litoral_bc_sur, in_season |
| lobster_red | 56868 | 27.5% | in_season, species_lobster_red, y_lag365 |
| urchin_red | 12872 | 37.1% | species_lobster_red, y_lag365, in_season |
| lobster_red@abreojos_progreso | 2708 | 25.4% | in_season, species_lobster_red, y_lag365 |
| lobster_red@abreojos_punta | 2708 | 25.2% | in_season, species_lobster_red, y_lag365 |
| lobster_red@abreojos_san_ignacio | 2708 | 25.4% | in_season, species_lobster_red, y_lag365 |
| lobster_red@er_el_chute | 2708 | 27.2% | in_season, species_lobster_red, y_lag365 |
| lobster_red@er_isla_san_geronimo | 2708 | 24.7% | in_season, y_lag365, species_lobster_red |
| lobster_red@er_mortera_leyva | 2708 | 25.6% | in_season, y_lag365, species_lobster_red |
| lobster_red@er_regasa | 2708 | 28.7% | in_season, y_lag365, species_lobster_red |
| lobster_red@er_scpp_ensenada | 2708 | 36.1% | in_season, economic_unit_er_scpp_ensenada, y_lag365 |
| lobster_red@isla_cedros | 2708 | 21.0% | in_season, y_lag365, species_lobster_red |
| lobster_red@la_purisima | 2708 | 25.4% | in_season, species_lobster_red, sst_roll365_lag90 |
| lobster_red@litoral_bc_sur | 2708 | 34.6% | y_lag365, economic_unit_litoral_bc_sur, y_lag730 |
| lobster_red@magdalena_bahia | 2708 | 24.1% | in_season, species_lobster_red, sst_roll365_lag90 |
| lobster_red@magdalena_chale | 2708 | 25.9% | in_season, species_lobster_red, sst_roll365_lag90 |
| lobster_red@magdalena_san_carlos | 2708 | 24.2% | in_season, species_lobster_red, sst_roll365_lag90 |
| lobster_red@pabellon_sq | 2708 | 26.1% | in_season, species_lobster_red, y_lag365 |
| lobster_red@punta_canoas | 2708 | 27.4% | in_season, species_lobster_red, y_lag365 |
| lobster_red@rocas_san_martin | 2708 | 25.8% | in_season, species_lobster_red, y_lag365 |
| lobster_red@vizcaino_asuncion | 2708 | 26.1% | in_season, species_lobster_red, y_lag365 |
| lobster_red@vizcaino_emancipacion | 2708 | 26.5% | in_season, species_lobster_red, y_lag365 |
| lobster_red@vizcaino_natividad | 2708 | 27.1% | in_season, species_lobster_red, y_lag365 |
| lobster_red@vizcaino_tortugas | 2708 | 26.4% | in_season, species_lobster_red, y_lag365 |

**Divergencia entre especies** (Jensen-Shannon sobre las cuotas por feature; 0 = misma lógica, 1 = disjuntas): media **0.274**, máx **0.331** (abalone_black vs lobster_red).

| par | JS | coseno |
|---|---|---|
| abalone_black vs lobster_red | 0.331 | 0.785 |
| abalone_red vs urchin_red | 0.309 | 0.873 |
| abalone_red vs lobster_red | 0.303 | 0.805 |
| abalone_black vs urchin_red | 0.296 | 0.875 |
| abalone_black vs abalone_red | 0.291 | 0.915 |
| abalone_black vs abalone_blue | 0.273 | 0.922 |
| abalone_blue vs abalone_red | 0.265 | 0.94 |
| abalone_blue vs lobster_red | 0.242 | 0.849 |

**Divergencia entre UEs de langosta** (Jensen-Shannon sobre las cuotas por feature; 0 = misma lógica, 1 = disjuntas): media **0.216**, máx **0.468** (lobster_red@la_purisima vs lobster_red@litoral_bc_sur).

| par | JS | coseno |
|---|---|---|
| lobster_red@la_purisima vs lobster_red@litoral_bc_sur | 0.468 | 0.59 |
| lobster_red@litoral_bc_sur vs lobster_red@magdalena_san_carlos | 0.466 | 0.603 |
| lobster_red@abreojos_san_ignacio vs lobster_red@litoral_bc_sur | 0.461 | 0.599 |
| lobster_red@litoral_bc_sur vs lobster_red@magdalena_bahia | 0.461 | 0.612 |
| lobster_red@abreojos_progreso vs lobster_red@litoral_bc_sur | 0.46 | 0.602 |
| lobster_red@litoral_bc_sur vs lobster_red@magdalena_chale | 0.459 | 0.6 |
| lobster_red@abreojos_punta vs lobster_red@litoral_bc_sur | 0.451 | 0.62 |
| lobster_red@litoral_bc_sur vs lobster_red@vizcaino_asuncion | 0.449 | 0.604 |

> Figura: `reports/figures/exp2_shap_by_group_2024-06-01.png`. El SHAP se calcula sobre train (interpretación de lo aprendido), no sobre test.