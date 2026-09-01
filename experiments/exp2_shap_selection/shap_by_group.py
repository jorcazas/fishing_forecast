"""Exp 2.4 (B5) — SHAP condicional por grupo: ¿el modelo global es uno o son muchos?

La tesis plantea una pregunta que hasta ahora no contestaba con evidencia: el pool sobre
`log1p(y)` (Exp 3.2, el modelo de producción) mete todas las series `(especie, UE)` en un solo
XGBoost. ¿Aprende **una sola lógica** que aplica a todas, o usa el one-hot de identidad para
partirse internamente en submodelos por especie/UE?

El experimento no introduce un modelo nuevo: reentrena exactamente la variante `pooled_log` de
Exp 3.2 —mismo objetivo `log1p(y)`, mismas features y mismos hiperparámetros que hereda la CQR
de producción— y descompone su atribución SHAP **condicionada al grupo**. El pool son todas las
series que superan el filtro de captura antes del corte (más de las que llegan a test, porque
aquí no se exige que la serie siga viva después del corte); el conteo exacto queda en el JSON.

1. `mean(|SHAP|)` global → ranking de referencia.
2. `mean(|SHAP|)` restringido a las filas de cada especie y de cada UE de langosta.
3. **Cuota de identidad**: fracción de `mean(|SHAP|)` que se va a las columnas one-hot de
   `species`/`economic_unit`. Alta ⇒ el pool usa la identidad como atajo (efectivamente
   submodelos); baja ⇒ la señal ambiental manda y el pooling sí comparte estructura.
4. **Divergencia entre grupos**: distancia de Jensen-Shannon y similitud coseno entre los
   vectores de cuota por feature de cada par de grupos. JS≈0 ⇒ misma lógica; JS alto ⇒ el
   modelo pondera distinto según el grupo.

Todo el SHAP se calcula sobre **train** (igual que `shap_prune.py`): es interpretación de lo que
el modelo aprendió, no una medición en test, y así no se toca la partición.

Uso:
    uv run python -m experiments.exp2_shap_selection.shap_by_group
    FF_CUT_DATE=2024-06-01 uv run python -m experiments.exp2_shap_selection.shap_by_group
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from experiments.exp3_global_model.pooled_ynorm import XGB_PARAMS, load_series
from loguru import logger

from fishing_forecast.config import get_settings
from fishing_forecast.features.covariates import build_multiseries_features, feature_columns

EXP_ID = "exp2_shap_by_group"
GROUP = ["species", "economic_unit"]
#: Corte de test; el SHAP se calcula sobre las filas anteriores (train del modelo de producción).
CUT_DATE = pd.Timestamp(os.environ.get("FF_CUT_DATE", "2020-07-01"))
#: Un grupo con muy pocas filas de train da un ranking SHAP puro ruido: se reporta aparte.
MIN_TRAIN_ROWS = 200
TOP_N = 12  # features mostradas en el mapa de calor y en el resumen


def _series_key(feat: pd.DataFrame) -> pd.Series:
    return feat["species"].astype(str) + "@" + feat["economic_unit"].astype(str)


def _shares(mean_abs: pd.Series) -> pd.Series:
    """Normaliza un vector de `mean(|SHAP|)` a cuotas que suman 1 (0 si el vector es nulo)."""
    total = float(mean_abs.sum())
    return mean_abs / total if total > 0 else mean_abs * 0.0


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """Distancia de Jensen-Shannon (raíz de la divergencia, base 2) entre dos distribuciones.

    En [0, 1]: 0 = idénticas, 1 = soportes disjuntos. Se implementa a mano —cuatro líneas— para
    no añadir una dependencia solo por esto.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_p = np.nansum(np.where(p > 0, p * np.log2(p / m), 0.0))
        kl_q = np.nansum(np.where(q > 0, q * np.log2(q / m), 0.0))
    return float(np.sqrt(max(0.5 * (kl_p + kl_q), 0.0)))


def cosine(p: np.ndarray, q: np.ndarray) -> float:
    """Similitud coseno entre dos vectores de cuotas (1 = mismo perfil de importancia)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    denom = float(np.linalg.norm(p) * np.linalg.norm(q))
    return float(p @ q / denom) if denom > 0 else 0.0


def group_shap(
    shap_values: np.ndarray,
    cols: list[str],
    labels: pd.Series,
    *,
    min_rows: int = MIN_TRAIN_ROWS,
) -> tuple[dict[str, pd.Series], dict[str, int]]:
    """`mean(|SHAP|)` por feature dentro de cada grupo de `labels`.

    Devuelve `(cuotas_por_grupo, n_filas_por_grupo)`. Los grupos con menos de `min_rows` filas
    se excluyen de las cuotas (ranking no confiable) pero sí aparecen en el conteo.
    """
    abs_sv = np.abs(shap_values)
    counts: dict[str, int] = {}
    shares: dict[str, pd.Series] = {}
    labels = labels.reset_index(drop=True)
    for label in labels.unique():
        mask = (labels == label).to_numpy()
        counts[str(label)] = int(mask.sum())
        if mask.sum() < min_rows:
            continue
        shares[str(label)] = _shares(pd.Series(abs_sv[mask].mean(axis=0), index=cols))
    return shares, counts


def identity_share(shares: pd.Series, identity_cols: list[str]) -> float:
    """Cuota de atribución que se va a las columnas one-hot de identidad (especie/UE)."""
    return float(shares.reindex(identity_cols).fillna(0.0).sum())


def pairwise_divergence(shares: dict[str, pd.Series]) -> list[dict]:
    """Divergencia JS y coseno entre todos los pares de grupos, ordenada de mayor a menor JS."""
    names = sorted(shares)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            va, vb = shares[a].to_numpy(), shares[b].reindex(shares[a].index).fillna(0.0).to_numpy()
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "jensen_shannon": round(jensen_shannon(va, vb), 3),
                    "cosine": round(cosine(va, vb), 3),
                }
            )
    return sorted(rows, key=lambda r: -r["jensen_shannon"])


def main() -> None:
    import shap
    import xgboost as xgb

    settings = get_settings()
    raw = load_series()
    feat = build_multiseries_features(raw, group_col=GROUP)

    base_cols = feature_columns(feat)
    feat["_series"] = _series_key(feat)
    onehot = pd.get_dummies(feat[GROUP], columns=GROUP)
    identity_cols = list(onehot.columns)
    feat = pd.concat([feat, onehot], axis=1)
    pool_cols = base_cols + identity_cols

    train = feat[feat["ds"] < CUT_DATE].reset_index(drop=True)
    train["y_log"] = np.log1p(train["y"])
    logger.info(
        f"corte {CUT_DATE.date()}: train={len(train)} filas, "
        f"{train['_series'].nunique()} series, {len(pool_cols)} features "
        f"({len(identity_cols)} de identidad)"
    )

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(train[pool_cols], train["y_log"])

    sv = shap.TreeExplainer(model).shap_values(train[pool_cols])
    global_shares = _shares(pd.Series(np.abs(sv).mean(axis=0), index=pool_cols)).sort_values(
        ascending=False
    )

    by_species, n_species = group_shap(sv, pool_cols, train["species"])
    lobster = train["_series"].where(train["species"] == "lobster_red", other="(otras especies)")
    by_unit, n_unit = group_shap(sv, pool_cols, lobster)
    by_unit.pop("(otras especies)", None)
    n_unit.pop("(otras especies)", None)

    result = {
        "cut_date": str(CUT_DATE.date()),
        "n_train_rows": len(train),
        "n_series": int(train["_series"].nunique()),
        "n_features": len(pool_cols),
        "n_identity_features": len(identity_cols),
        "identity_share_global": round(identity_share(global_shares, identity_cols), 3),
        "global_ranking": {k: round(float(v), 4) for k, v in global_shares.head(25).items()},
        "by_species": {
            g: {
                "n_train_rows": n_species[g],
                "identity_share": round(identity_share(s, identity_cols), 3),
                "top": {
                    k: round(float(v), 4)
                    for k, v in s.sort_values(ascending=False).head(TOP_N).items()
                },
            }
            for g, s in by_species.items()
        },
        "by_lobster_unit": {
            g: {
                "n_train_rows": n_unit[g],
                "identity_share": round(identity_share(s, identity_cols), 3),
                "top": {
                    k: round(float(v), 4)
                    for k, v in s.sort_values(ascending=False).head(TOP_N).items()
                },
            }
            for g, s in by_unit.items()
        },
        "divergence_species": pairwise_divergence(by_species),
        "divergence_lobster_units": pairwise_divergence(by_unit),
        "skipped_small_groups": {
            g: n for g, n in {**n_species, **n_unit}.items() if n < MIN_TRAIN_ROWS
        },
    }

    metrics_dir = settings.reports_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{EXP_ID}_{CUT_DATE.date()}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )
    _heatmap(
        global_shares,
        by_species,
        by_unit,
        settings.reports_root / "figures" / f"{EXP_ID}_{CUT_DATE.date()}.png",
    )
    _write_summary(result, settings.reports_root / f"{EXP_ID}_summary_{CUT_DATE.date()}.md")
    print(_console(result))


def _heatmap(global_shares, by_species, by_unit, out_path) -> None:
    """Mapa de calor: cuota de `mean(|SHAP|)` de las TOP_N features globales en cada grupo."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feats = global_shares.head(TOP_N).index.tolist()
    groups = {
        **{f"sp: {k}": v for k, v in by_species.items()},
        **{f"ue: {k.split('@')[-1]}": v for k, v in by_unit.items()},
    }
    if not groups:
        logger.warning("Ningún grupo supera MIN_TRAIN_ROWS; no se dibuja el mapa de calor.")
        return
    mat = np.array([[float(s.get(f, 0.0)) for f in feats] for s in groups.values()])

    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * len(feats), 1.2 + 0.42 * len(groups)))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(feats)), feats, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(groups)), list(groups), fontsize=8)
    ax.set_title(
        "Cuota de mean(|SHAP|) por grupo — modelo global pooled_log\n"
        "(filas iguales ⇒ el pool aplica la misma lógica a todos los grupos)",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax, label="cuota de mean(|SHAP|)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info(f"Figura → {out_path}")


def _console(r: dict) -> str:
    lines = ["", f"SHAP condicional por grupo (pooled_log, corte {r['cut_date']}):"]
    lines.append(
        f"  cuota de atribución a la identidad (one-hot): {r['identity_share_global']:.1%}"
    )
    lines.append(f"  top global: {', '.join(list(r['global_ranking'])[:6])}")
    lines.append(f"\n  {'grupo':<34}{'filas':>8}{'cuota identidad':>18}   top-3")
    for scope in ("by_species", "by_lobster_unit"):
        for g, d in r[scope].items():
            top3 = ", ".join(list(d["top"])[:3])
            lines.append(f"  {g:<34}{d['n_train_rows']:>8}{d['identity_share']:>17.1%}   {top3}")
    for name, key in (
        ("especies", "divergence_species"),
        ("UEs langosta", "divergence_lobster_units"),
    ):
        if r[key]:
            js = [d["jensen_shannon"] for d in r[key]]
            top = r[key][0]
            lines.append(
                f"\n  Divergencia entre {name}: JS media {np.mean(js):.3f}, "
                f"máx {max(js):.3f} ({top['a']} vs {top['b']})"
            )
    return "\n".join(lines)


def _write_summary(r: dict, out_path) -> None:
    rows = [
        f"# Exp 2.4 — SHAP condicional por grupo (corte {r['cut_date']})",
        "",
        f"Modelo: el pool sobre `log1p(y)` de Exp 3.2 (producción), {r['n_series']} series, "
        f"{r['n_train_rows']} filas de train, {r['n_features']} features de las cuales "
        f"{r['n_identity_features']} son one-hot de identidad.",
        "",
        f"**Cuota de atribución que se va a la identidad (especie/UE): "
        f"{r['identity_share_global']:.1%}** del `mean(|SHAP|)` total. "
        "Es el indicador directo de si el pool usa la identidad como atajo "
        "(cuota alta ⇒ submodelos disfrazados) o si comparte estructura ambiental "
        "(cuota baja ⇒ el pooling sí transfiere).",
        "",
        "| grupo | filas train | cuota identidad | top-3 features |",
        "|---|---|---|---|",
    ]
    for scope in ("by_species", "by_lobster_unit"):
        for g, d in r[scope].items():
            rows.append(
                f"| {g} | {d['n_train_rows']} | {d['identity_share']:.1%} | "
                f"{', '.join(list(d['top'])[:3])} |"
            )
    for name, key in (
        ("especies", "divergence_species"),
        ("UEs de langosta", "divergence_lobster_units"),
    ):
        if not r[key]:
            continue
        js = [d["jensen_shannon"] for d in r[key]]
        rows += [
            "",
            f"**Divergencia entre {name}** (Jensen-Shannon sobre las cuotas por feature; "
            f"0 = misma lógica, 1 = disjuntas): media **{np.mean(js):.3f}**, "
            f"máx **{max(js):.3f}** ({r[key][0]['a']} vs {r[key][0]['b']}).",
            "",
            "| par | JS | coseno |",
            "|---|---|---|",
        ]
        rows += [
            f"| {d['a']} vs {d['b']} | {d['jensen_shannon']} | {d['cosine']} |" for d in r[key][:8]
        ]
    if r["skipped_small_groups"]:
        rows += [
            "",
            f"> Grupos omitidos por tener < {MIN_TRAIN_ROWS} filas de train (ranking SHAP no "
            f"confiable): {', '.join(f'{g} ({n})' for g, n in r['skipped_small_groups'].items())}.",
        ]
    rows += [
        "",
        f"> Figura: `reports/figures/{EXP_ID}_{r['cut_date']}.png`. El SHAP se calcula sobre "
        "train (interpretación de lo aprendido), no sobre test.",
    ]
    out_path.write_text("\n".join(rows))
    logger.info(f"Resumen → {out_path}")


if __name__ == "__main__":
    main()
