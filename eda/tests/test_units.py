"""Unit tests for the research building blocks (eda/features.py, eda/improve/*, extractor, weak labels)."""
import importlib

import numpy as np
import pandas as pd
import pytest

from features import kb_features, parse_text_dates, text_flags
from improve.judge_nb import EvidenceJudge, apply_temperature, fit_temperature, with_false_prob
from improve.timeline import Timeline, anchors_for
from improve.trust import page_trust


# ------------------------------------------------------------------ features.py
def _rows(texts, date="2026-04-28"):
    return pd.DataFrame({"text": texts, "date": pd.to_datetime([date] * len(texts))})


def test_text_flags_first_person_joke_tail_hedge():
    X = text_flags(_rows([
        "Revenue was €68 million and I am not even tired.",
        "Hooly Docs has about 8,800 users a month.",
        "Hooly Docs 2.0",
    ]))
    assert X.loc[0, "first_person"] == 1 and X.loc[0, "joke_tail"] == 1
    assert X.loc[1, "hedge"] == 1 and X.loc[1, "first_person"] == 0
    assert X.loc[2, "no_period"] == 1 and X.loc[2, "hype_score"] == 0


def test_first_person_is_case_sensitive():
    # "I" must not match inside words, and lowercase "i" alone is not first person
    X = text_flags(_rows(["Inbox is fine.", "I said so."]))
    assert X.loc[0, "first_person"] == 0 and X.loc[1, "first_person"] == 1


def test_parse_text_dates_formats():
    assert parse_text_dates("starts on 12 January 2026") == [(2026, 1, 12)]
    assert parse_text_dates("opened on June 9, 2026") == [(2026, 6, 9)]
    assert parse_text_dates("open since November 2025") == [(2025, 11, None)]
    assert parse_text_dates("no date here") == []


def test_today_with_mismatched_date_is_flagged():
    X = text_flags(_rows(["Hooly Android 17 beta starts today, 3 September 2025"], date="2026-04-28"))
    assert X.loc[0, "today_date_mismatch"] == 1 and X.loc[0, "text_date_in_past"] == 1


# ------------------------------------------------------------------ kb_features
def _facts(rows):
    df = pd.DataFrame(rows, columns=["id", "date", "label", "value"])
    df["date"] = pd.to_datetime(df["date"])
    df["num"] = range(len(df))
    df["subject"], df["attribute"] = "Hooly Test", "price"
    return df


def test_kb_relations_on_a_small_timeline():
    F = _facts([
        ("a", "2026-01-01", "true", "10"),
        ("b", "2026-03-01", "true", "12"),
        ("t_match", "2026-04-01", "?", "12"),
        ("t_stale", "2026-04-01", "?", "10"),
        ("t_new", "2026-04-01", "?", "99"),
        ("t_future", "2025-12-01", "?", "10"),
    ])
    rel = kb_features(F, {"a", "b"}).set_index(F["id"])["kb_relation"]
    assert rel["t_match"] == "match_latest"
    assert rel["t_stale"] == "stale_older"
    assert rel["t_new"] == "contradicts"
    assert rel["t_future"] == "future_value"


def test_kb_ignores_labels_outside_the_training_ids():
    F = _facts([("a", "2026-01-01", "true", "10"), ("b", "2026-03-01", "true", "12"),
                ("t", "2026-04-01", "?", "12")])
    rel = kb_features(F, {"a"}).set_index(F["id"])["kb_relation"]
    assert rel["t"] == "contradicts"  # b's label is not training data, so 12 is unknown


# ------------------------------------------------------------------ 02_rules
def test_wilson_interval():
    rules = importlib.import_module("02_rules")
    lo, hi = rules.wilson(5, 10)
    assert 0 < lo < 0.5 < hi < 1
    lo0, hi0 = rules.wilson(0, 280)
    assert lo0 == pytest.approx(0, abs=1e-12) and hi0 < 0.02
    assert all(np.isnan(rules.wilson(0, 0)))


# ------------------------------------------------------------------ splits
def test_splits_columns_and_values(splits):
    assert {"id", "page", "date", "label", "holdout", "fold_group", "fold_row5", "time_split"} <= set(splits)
    assert set(splits["fold_row5"]) == {0, 1, 2, 3, 4}
    assert set(splits["time_split"]) == {"train", "test"}
    assert (splits.loc[splits["holdout"] == 1, "fold_group"] == -1).all()


def test_holdout_keeps_every_label_class(splits):
    assert set(splits.loc[splits["holdout"] == 1, "label"]) == {"true", "false", "unsure"}


# ------------------------------------------------------------------ timeline edge cases
def _tl(rows):
    df = pd.DataFrame(rows, columns=["id", "page", "date", "vkey"])
    df["date"] = pd.to_datetime(df["date"])
    df["subject"], df["attribute"] = "Hooly Test", "price"
    return Timeline(df)


def test_timeline_one_mention():
    tl = _tl([("x", 1, "2026-01-01", "A")])
    feats, agree = tl.run(np.zeros(1, int), np.ones(1), np.ones(1, bool))
    assert feats.loc[0, "tl_agree"] == 1 and feats.loc[0, "n_stairs"] == 1
    assert feats.loc[0, "support_for"] == 0 and np.isnan(agree[0])  # a lone pair page is not informative


def test_timeline_empty_frame_raises():
    """Documented behaviour: an empty fact frame is not a valid timeline (callers never build one)."""
    with pytest.raises(ValueError):
        _tl([])


def test_timeline_tie_is_deterministic():
    rows = [("a", 1, "2026-01-01", "A"), ("b", 2, "2026-01-01", "B")]
    runs = [_tl(rows).run(np.zeros(2, int), np.ones(2), np.ones(2, bool))[0]["tl_agree"].tolist()
            for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]
    assert sum(runs[0]) == 1  # exactly one of the tied values is held


def test_anchor_codes():
    tl = _tl([("t", 1, "2026-01-01", "A"), ("f1", 2, "2026-02-01", "B"), ("fm", 3, "2026-03-01", "C"),
              ("u", 4, "2026-04-01", "D"), ("v", 5, "2026-05-01", "E")])
    labels = {"t": "true", "f1": "false", "fm": "false", "u": "unsure"}
    a = anchors_for(tl, {"t", "f1", "fm", "u"}, labels, {"t": 1, "f1": 1, "fm": 2, "u": 1})
    assert a.tolist() == [1, -1, 9, 9, 0]


# ------------------------------------------------------------------ page trust EM
def test_page_trust_bounds_and_direction():
    # pair 1: page 1 anchored true "A"; pages 2-4 vote A, page 5 always votes the minority value
    rows, anchor = [], []
    for pair in range(4):
        base = pd.Timestamp("2026-01-01") + pd.Timedelta(days=30 * pair)
        for page in (1, 2, 3, 4):
            rows.append((f"p{pair}_{page}", page, base, "A", f"S{pair}"))
            anchor.append(1 if page == 1 else 0)
        rows.append((f"p{pair}_5", 5, base, "Z", f"S{pair}"))
        anchor.append(0)
    df = pd.DataFrame(rows, columns=["id", "page", "date", "vkey", "subject"])
    df["attribute"] = "price"
    tl = Timeline(df)
    prior = {p: 0.8 for p in range(1, 6)}
    trust, w = page_trust(tl, np.array(anchor), prior, rounds=5)
    assert all(0 <= v <= 1 for v in trust.values())
    assert trust[5] < trust[2]
    assert w.shape == (tl.n,)
    trust10, _ = page_trust(tl, np.array(anchor), prior, rounds=10)
    assert abs(trust10[5] - trust[5]) < 0.05  # converged after a few rounds


# ------------------------------------------------------------------ evidence-sum judge
def test_evidence_judge_monotone():
    items = pd.DataFrame({"status": ["outdated"] * 20 + ["holds"] * 80})
    is_false = np.array([True] * 16 + [False] * 4 + [True] * 4 + [False] * 76)
    j = EvidenceJudge().fit(items, is_false)
    lo = j.log_odds(pd.DataFrame({"status": ["outdated", "holds", "never_seen"]}))
    assert lo[0] > lo[2] > lo[1]  # unseen category falls back to the prior


def test_temperature_and_false_prob_keep_rows_normalised(rng):
    P = rng.dirichlet([2, 5, 1], size=200)
    y = rng.integers(0, 3, size=200)
    t = fit_temperature(P, y)
    assert 0.5 <= t <= 3.0
    Q = apply_temperature(P, t)
    assert np.allclose(Q.sum(1), 1)
    R = with_false_prob(P, np.full(200, 0.3))
    assert np.allclose(R.sum(1), 1) and np.allclose(R[:, 0], 0.3)
    ratio_before = P[:, 1] / P[:, 2]
    assert np.allclose(R[:, 1] / R[:, 2], ratio_before)


# ------------------------------------------------------------------ extractor
def test_extractor_reads_common_values():
    from extract_simple import parse_values
    vals = lambda s: [v for _, _, v in parse_values(s)]  # noqa: E731
    dur = [v for v in vals("the median time to recover was 52 min") if v["kind"] == "duration"]
    assert dur and dur[0]["value"] == 52 and dur[0]["unit"] == "min"
    pct = [v for v in vals("Last quarter Hooly Cloud was up 99.82% of the time.") if v["kind"] == "percent"]
    assert pct and float(pct[0]["value"]) == pytest.approx(99.82)
    ver = [v for v in vals("Hooly Gemini 2.1 ships today") if v["kind"] == "version"]
    assert ver and ver[0]["value"] == "2.1"


# ------------------------------------------------------------------ weak labels
def test_label_model_probabilities(rng):
    wl = importlib.import_module("weak_labels")
    n, m = 300, 6
    L = pd.DataFrame(rng.choice([-1, 0, 1, 2], size=(n, m), p=[0.6, 0.15, 0.2, 0.05]),
                     columns=[f"lf{i}" for i in range(m)])
    L = L.replace(-1, wl.ABST)
    P, prec, pi = wl.label_model(L, np.array([0.15, 0.83, 0.02]))
    assert P.shape == (n, 3)
    assert np.all((P >= 0) & (P <= 1)) and np.allclose(P.sum(1), 1)
    assert np.all((prec >= 0) & (prec <= 1))
