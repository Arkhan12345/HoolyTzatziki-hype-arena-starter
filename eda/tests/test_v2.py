"""v2 robustness pieces: referent check, plausibility, tolerance, renaming, confidence cap, unknown-subject offset."""
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

H300 = Path(__file__).resolve().parents[1] / "h300"
sys.path.insert(0, str(H300))

from lib import referent  # noqa: E402
from lib.feats import v2_fields  # noqa: E402


def _kb(values_by_attr):
    timeline = {("S%d" % i, a): [("2026-01-01", ("quantity", "count", v, "eq"), "true")]
                for a, vals in values_by_attr.items() for i, v in enumerate(vals)}
    return SimpleNamespace(kb=SimpleNamespace(timeline=timeline, by_subject={"S0": {"users"}}))


def test_confusable_pairs_are_symmetric():
    assert referent.confusable("venue", "city") and referent.confusable("city", "venue")
    assert referent.confusable("median_response_time", "p99_response_time")
    assert not referent.confusable("price", "venue")


def test_plausibility_flags_an_outlier_but_not_a_typical_value():
    rd = _kb({"users": [1000, 1200, 900, 1100, 1050]})
    z_typ, n = referent.plausibility(rd, "users", {"kind": "quantity", "value": 1000, "unit": "user"})
    z_out, _ = referent.plausibility(rd, "users", {"kind": "quantity", "value": 1_000_000, "unit": "user"})
    assert n == 5 and abs(z_typ) < 1 and z_out > 2.5


def test_plausibility_needs_three_values_and_numbers():
    rd = _kb({"users": [1000, 1200]})
    assert math.isnan(referent.plausibility(rd, "users", {"kind": "quantity", "value": 5, "unit": "user"})[0])
    rd = _kb({"users": [1, 2, 3]})
    assert math.isnan(referent.plausibility(rd, "users", {"kind": "name", "value": "Lisbon"})[0])


def test_v2_fields_without_a_reading_are_neutral():
    v = v2_fields(_kb({}), None, {"matches_latest": 0})
    assert v["v2_known_subject"] == 0 and v["v2_referent_weak"] == 1 and math.isnan(v["v2_plaus_z"])


def test_tolerance_accepts_a_hedged_close_value_only():
    rd = _kb({"users": [1000, 1100, 1200]})
    best = (1.0, None, "S0", {"users": 0.9, "price": 0.1}, {"users"}, "boolean", [])
    close = v2_fields(rd, best, {"matches_latest": 0, "log_ratio_latest": math.log(1.05), "misread_hedge": 1,
                                 "parse_conf": 0.9})
    far = v2_fields(rd, best, {"matches_latest": 0, "log_ratio_latest": math.log(1.5), "misread_hedge": 1,
                               "parse_conf": 0.9})
    plain = v2_fields(rd, best, {"matches_latest": 0, "log_ratio_latest": math.log(1.05), "misread_hedge": 0,
                                 "parse_conf": 0.9})
    assert close["v2_tol_match"] == 1 and far["v2_tol_match"] == 0 and plain["v2_tol_match"] == 0


def test_confusable_runner_up_makes_the_referent_weak():
    rd = _kb({})
    best = (1.0, None, "S0", {"venue": 0.55, "city": 0.45}, {"venue"}, "boolean", [])
    assert v2_fields(rd, best, {"parse_conf": 0.9})["v2_confusable"] == 1
    best = (1.0, None, "S0", {"venue": 0.9, "price": 0.1}, {"venue"}, "boolean", [])
    assert v2_fields(rd, best, {"parse_conf": 0.9})["v2_referent_weak"] == 0


def test_renaming_is_consistent_and_leaves_values_and_dates():
    import build_v2
    lex = json.loads((H300.parents[1] / "data" / "lexicon.json").read_text(encoding="utf-8"))
    mapping, aliases = build_v2.rename_map(lex)
    names = {s["name"].lower() for s in lex["subjects"]}
    assert all(v.lower() not in names for v in mapping.values()) and len(set(mapping.values())) == len(mapping)
    row = {"id": "p-1", "date": "2026-01-12", "label": "true",
           "text": "Hooly Cloud has been open since November 2025 and covers 3 regions so far.",
           "form": json.dumps([{"subject": "Hooly Cloud", "attribute": "regions", "value": {}}])}
    r = build_v2.rename_row(row, mapping, aliases)
    assert "Hooly Cloud" not in r["text"] and mapping["Hooly Cloud"] in r["text"]
    assert "3 regions" in r["text"] and "November 2025" in r["text"] and r["date"] == row["date"]
    assert build_v2.rename_row(row, mapping, aliases)["text"] == r["text"]


def test_confidence_cap_never_changes_the_label_and_caps_weak_rows():
    import ladder_v2

    class Stub:
        def predict_proba(self, F):
            return np.array([[0.9, 0.08, 0.02], [0.9, 0.08, 0.02]])

    m = ladder_v2.Model({**ladder_v2.BASE, "cap": True})
    m.m, m.off = Stub(), 0.0
    F = pd.DataFrame({"v2_referent_weak": [1, 0], "misread_hedge": [0, 0], "known_pair": [1, 1]})
    pred, P = m.predict(F)
    assert list(pred) == [0, 0]
    assert P[0, 0] == pytest.approx(0.6) and P[1, 0] == pytest.approx(0.9)
    assert np.allclose(P.sum(1), 1)


def test_unknown_offset_only_moves_unknown_rows():
    import ladder_v2
    P = np.array([[0.6, 0.4, 0.0]] * 4)
    y = np.array([1, 1, 0, 0])
    mask = np.array([True, True, False, False])
    off = ladder_v2.tune_unknown_offset(P, y, mask)
    assert off > 0
