"""Tests de las funciones puras de Exp 2.4 (SHAP condicional por grupo)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from experiments.exp2_shap_selection.shap_by_group import (
    cosine,
    group_shap,
    identity_share,
    jensen_shannon,
    pairwise_divergence,
)


def test_jensen_shannon_identical_is_zero_and_disjoint_is_one():
    p = np.array([0.5, 0.5, 0.0])
    assert jensen_shannon(p, p) == 0.0
    assert jensen_shannon(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 1.0


def test_cosine_is_scale_invariant():
    p = np.array([0.6, 0.3, 0.1])
    assert cosine(p, 3 * p) == pytest.approx(1.0)
    assert cosine(p, np.zeros(3)) == 0.0


def test_group_shap_averages_only_within_the_group_and_normalizes():
    # Grupo A atribuye todo a f0; grupo B, todo a f1.
    sv = np.array([[2.0, 0.0], [4.0, 0.0], [0.0, 1.0], [0.0, 3.0]])
    labels = pd.Series(["A", "A", "B", "B"])
    shares, counts = group_shap(sv, ["f0", "f1"], labels, min_rows=1)
    assert counts == {"A": 2, "B": 2}
    assert shares["A"].to_dict() == {"f0": 1.0, "f1": 0.0}
    assert shares["B"].to_dict() == {"f0": 0.0, "f1": 1.0}


def test_group_shap_skips_groups_below_min_rows_but_still_counts_them():
    sv = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    labels = pd.Series(["A", "A", "B"])
    shares, counts = group_shap(sv, ["f0", "f1"], labels, min_rows=2)
    assert counts["B"] == 1
    assert "B" not in shares
    assert "A" in shares


def test_identity_share_ignores_missing_columns():
    shares = pd.Series({"sst": 0.6, "species_lobster_red": 0.3, "mhw_now": 0.1})
    assert identity_share(shares, ["species_lobster_red", "economic_unit_x"]) == 0.3


def test_pairwise_divergence_is_sorted_by_descending_js():
    shares = {
        "a": pd.Series({"f0": 1.0, "f1": 0.0}),
        "b": pd.Series({"f0": 0.9, "f1": 0.1}),
        "c": pd.Series({"f0": 0.0, "f1": 1.0}),
    }
    rows = pairwise_divergence(shares)
    assert len(rows) == 3
    js = [r["jensen_shannon"] for r in rows]
    assert js == sorted(js, reverse=True)
    assert {rows[0]["a"], rows[0]["b"]} == {"a", "c"}  # el par disjunto es el más divergente
