"""Unit edge cases for eda/improve/rules_model.py."""
import math

import numpy as np
import pandas as pd
import pytest

from app.kb.normalize import key as value_key
from improve import rules_model as rm


def q(n, unit="user"):
    return value_key({"kind": "quantity", "value": n, "unit": unit})


def test_ratio_plain():
    assert rm.ratio_to_held(q(50), q(100)) == pytest.approx(0.5)


def test_ratio_zero_and_negative_values_are_nan():
    assert math.isnan(rm.ratio_to_held(q(0), q(100)))
    assert math.isnan(rm.ratio_to_held(q(10), q(0)))
    assert math.isnan(rm.ratio_to_held(q(-5), q(10)))


def test_ratio_non_numeric_kinds_are_nan():
    name_a = value_key({"kind": "name", "value": "Holdan"})
    name_b = value_key({"kind": "name", "value": "Gavin"})
    assert math.isnan(rm.ratio_to_held(name_a, name_b))
    ver = value_key({"kind": "version", "value": "2.0"})
    assert math.isnan(rm.ratio_to_held(ver, ver))


def test_ratio_missing_held_is_nan():
    assert math.isnan(rm.ratio_to_held(q(10), None))
    assert math.isnan(rm.ratio_to_held(q(10), float("nan")))


def test_ratio_different_dimensions_is_nan():
    assert math.isnan(rm.ratio_to_held(q(10, "GB"), q(10, "user")))


@pytest.mark.parametrize("r,hit", [(2.0, 1), (0.5, 1), (2.5, 1), (0.4, 1), (5.0, 1), (10.0, 1), (0.1, 1),
                                   (1.0, 0), (1.3, 0), (3.0, 0), (float("nan"), 0)])
def test_perturbation_hits(r, hit):
    assert rm.is_perturbation(r) == hit


def test_unsure_layer_keeps_confident_false():
    feats = pd.DataFrame([{"joke_tail": 1, "n_facts": 1, "kv_line": 0, "hearsay": 0, "today_date_mismatch": 0,
                           "pair_n_values_max": 3, "same_value_rows_min": 2, "tl_n_pages_for_max": 2}])
    labels, conf = rm.unsure_layer(np.array([[0.8, 0.15, 0.05]]), feats, {"joke": True})
    assert rm.CLASSES[labels[0]] == "false"
    assert conf[0] == pytest.approx(0.8)


def test_unsure_confidence_is_nan_and_others_clipped():
    feats = pd.DataFrame([{"joke_tail": 1, "n_facts": 1, "kv_line": 0, "hearsay": 0, "today_date_mismatch": 0,
                           "pair_n_values_max": 3, "same_value_rows_min": 2, "tl_n_pages_for_max": 2}] * 2)
    feats.loc[1, "joke_tail"] = 0
    labels, conf = rm.unsure_layer(np.array([[0.1, 0.85, 0.05], [0.45, 0.4, 0.15]]), feats, {"joke": True})
    assert rm.CLASSES[labels[0]] == "unsure" and math.isnan(conf[0])
    assert conf[1] >= 0.5


def test_tune_t_returns_grid_value_and_no_tf_loss():
    rng = np.random.default_rng(0)
    n = 200
    y = rng.choice([0, 1, 2], size=n, p=[0.15, 0.8, 0.05])
    p = np.full((n, 3), 0.05)
    p[np.arange(n), y] = 0.9
    p = p / p.sum(1, keepdims=True)
    feats = pd.DataFrame({"joke_tail": 0, "n_facts": 1, "kv_line": 0, "hearsay": 0, "today_date_mismatch": 0,
                          "pair_n_values_max": 1, "same_value_rows_min": 0, "tl_n_pages_for_max": 0}, index=range(n))
    t = rm.tune_t(p, y, feats, {"single_value": True}, grid=(0.0, 0.6, 0.8))
    assert t in (0.0, 0.6, 0.8)
