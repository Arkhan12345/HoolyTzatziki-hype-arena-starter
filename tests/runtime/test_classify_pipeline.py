"""The integrated classifier: KB rules, training-data generator, label-free KB, prod / debug modes."""
import json
import os

import pytest

from app.io.reader import read_practice
from app.kb import normalize as nz
from app.kb.store import KB

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEX = json.load(open(os.path.join(ROOT, "data", "lexicon.json"), encoding="utf-8"))
GOLD = read_practice(os.path.join(ROOT, "data", "practice.csv"))
SMALL = GOLD[:400]


def _gold_facts(rows):
    return [{"row_id": r["id"], "subject": f["subject"], "attribute": f["attribute"], "value": f["value"], "conf": 0.9}
            for r in rows for f in r["form"]]


def test_rows_of_one_date_sort_by_number_not_text():
    assert sorted(["p-1000", "p-999", "p-12"], key=nz.row_order) == ["p-12", "p-999", "p-1000"]
    kb = KB(LEX)
    v = lambda x: {"kind": "quantity", "value": x, "unit": "seat"}  # noqa: E731
    kb._add("Hooly Board", "seats", v(7), "2026-05-01", "gold", 1.0, "p-1000")
    kb._add("Hooly Board", "seats", v(5), "2026-05-01", "gold", 1.0, "p-999")
    value, _, _ = kb.lookup("Hooly Board", "seats", "2026-05-01")
    assert value["value"] == 7  # p-1000 comes after p-999 on that date


def test_a_bound_includes_its_own_value():
    sixty = {"kind": "quantity", "value": 60, "unit": "participant"}
    assert nz.key_inclusive({**sixty, "op": "le"}) == nz.key_inclusive(sixty) == nz.key_inclusive({**sixty, "op": "ge"})
    assert nz.key({**sixty, "op": "le"}) != nz.key(sixty)  # the claim validator stays strict
    assert nz.key_inclusive({"kind": "duration", "value": 2, "unit": "week"}) == \
        nz.key_inclusive({"kind": "duration", "value": 14, "unit": "day"})  # units convert


@pytest.fixture(scope="module")
def items():
    from app.classify.datagen import generate_items
    from app.validate.rules import norm_text

    rep = {}
    out = generate_items(KB(LEX).build([], GOLD, None), LEX, GOLD, {norm_text(r["text"]) for r in GOLD}, rep)
    return out, rep


def test_generator_covers_every_type_and_label(items):
    out, rep = items
    seen = {(i["type"], i["label"]) for i in out}
    for t in ("fact", "change", "compare", "at_event", "relation"):
        for lab in ("true", "false", "unsure"):
            assert (t, lab) in seen, (t, lab)
    assert rep["fact_restatements_skipped"] > 0


def test_flipped_items_really_differ(items):
    out, _ = items
    texts = {i["text"] for i in out if i["label"] == "true"}
    for f in (i for i in out if i["label"] == "false"):
        assert f["text"] not in texts
        if f["type"] == "compare":
            assert f["form"]["op"] in ("gt", "lt")


def test_unsure_items_are_about_pairs_no_gold_row_states(items):
    from app.kb.inventory import FactInventory

    out, _ = items
    known = set(FactInventory.from_kb(KB(LEX).build([], GOLD, None)).pairs())
    for i in (i for i in out if i["label"] == "unsure" and i["type"] == "fact"):
        assert (i["form"]["subject"], i["form"]["attribute"]) not in known


def test_label_free_kb_never_sees_a_label():
    from app.classify.model import World

    rows = [{"id": r["id"], "text": r["text"], "date": r["date"], "label": r["label"]} for r in SMALL]
    flipped = [dict(r, label={"true": "false", "false": "true"}.get(r["label"], "true")) for r in rows]
    a = World(rows, _gold_facts(SMALL), [], use_labels=False).F
    b = World(flipped, _gold_facts(SMALL), [], use_labels=False).F
    cols = ["tl_agree", "support_for", "support_against", "st_holds"]
    assert a[cols].fillna(-1).equals(b[cols].fillna(-1))


def test_a_labelled_row_is_scored_without_its_own_label():
    """Out-of-fold: flipping one labelled row's label changes nothing in that row's own features."""
    from app.classify.model import World

    rows = [{"id": r["id"], "text": r["text"], "date": r["date"], "label": r["label"]} for r in SMALL]
    forms = {r["id"]: r["form"] for r in SMALL}
    target = next(r["id"] for r in rows if r["label"] == "true" and forms[r["id"]])
    flipped = [dict(r, label="false") if r["id"] == target else r for r in rows]
    wa = World(rows, _gold_facts(SMALL), [], gold_forms=forms)
    wb = World(flipped, _gold_facts(SMALL), [], gold_forms=forms)
    fa = wa.row_features(target, "sentence", "", "")
    fb = wb.row_features(target, "sentence", "", "")
    assert fa == pytest.approx(fb, nan_ok=True)
    assert (wa.tl.anchor == 1).sum() > 0  # the labelled rows do pin the inference KB


@pytest.mark.parametrize("mode", ["prod", "debug"])
def test_classify_labels_every_unlabelled_row(mode):
    from app.classify import LAST, classify

    rows = [{"id": r["id"], "text": r["text"], "date": r["date"], "label": "" if i % 5 == 0 else r["label"]}
            for i, r in enumerate(SMALL)]
    rep = {}
    labels = classify(rows, _gold_facts(SMALL), KB(LEX), mode=mode, report=rep, gold_rows=GOLD)
    blank = [r["id"] for r in rows if not r["label"]]
    assert [lb.id for lb in labels] == blank
    for lb in labels:
        assert lb.label in ("true", "false", "unsure")
        assert (lb.confidence is None) == (lb.label == "unsure")
        assert lb.confidence is None or 0.5 <= lb.confidence <= 1
        assert 0 <= LAST["details"][lb.id]["trust_score"] <= 1
    assert rep["mode"] == mode
    assert (rep["debug"] is not None) == (mode == "debug")
    if mode == "debug":
        d = rep["debug"]
        assert d["test_items"] > 0 and d["accuracy"] is not None
        assert d["train_rows"] == d["train_items"] + rep["labelled_rows"]  # labelled rows + half the generated
        assert len(d["predictions"]) == d["test_items"]  # inference on every held-out sentence
        assert d["accuracy"] == pytest.approx(sum(x["correct"] for x in d["predictions"]) / d["test_items"], abs=1e-4)


def test_hidden_practice_rows_stay_out_of_the_truth_kb():
    from app.classify import _truth_gold

    rows = [{"id": "p-1", "label": ""}, {"id": "p-2", "label": "true"}]
    assert [g["id"] for g in _truth_gold(rows, GOLD[:3])] == ["p-2", "p-3"]


def test_a_fact_that_rewords_a_prose_sentence_is_not_new():
    from app.generate.restate import restates_prose

    v = {"kind": "quantity", "value": 4, "unit": "recruiter"}
    form = {"subject": "Hooly Hiring Programme", "attribute": "recruiters", "value": v}
    rows = {"p-1": {"text": "A team of 4 recruiters reads your file.", "form": [form]},
            "p-2": {"text": "Recruiters: 4", "form": [form]}}
    assert restates_prose(form, ["p-1"], rows)  # prose: only a rewording
    assert not restates_prose(form, ["p-2"], rows)  # a table line made into a sentence is new


def test_no_output_fact_claim_rewords_a_prose_sentence():
    from app.generate import generate
    from app.generate.restate import restates_prose
    from app.validate.rules import norm_text

    kb = KB(LEX).build([], GOLD, None)
    rep = {}
    claims = generate(kb, LEX, {norm_text(r["text"]) for r in GOLD}, 1000, rep)
    assert len(claims) == 1000
    facts = [c for c in claims if c.type == "fact"]
    assert facts and not [c for c in facts
                          if restates_prose(c.form, [s["row_id"] for s in c.meta.get("sources", [])], kb.rows)]
