"""Text -> facts for every universe row: the pipeline's `extract(rows, lexicon)` (app/run.py, app/api).

Two independent readers, validated on eda/splits.csv (pages held out, fact F1 by value kind):
  Arckan's ranker (extractor.py)   CV F1 0.809, holdout 0.838: the facts the pipeline uses
  X's simple extractor (simple.py) CV F1 0.723, holdout 0.736: a second opinion only
Arckan's wins on every value kind and no union beats it, but a fact both read is right 95% of the time
(holdout precision 0.953 vs 0.883), so agreement raises the fact's confidence (its vote in the KB).

Both readers learn from the practice rows' gold `form` (in the image). A row that is itself a practice row
is read by models fitted WITHOUT its page (5 page folds): the facts of labelled rows look exactly like the
facts of unlabelled rows, so the classifier trains on what it will see at inference.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import pandas as pd

from app.kb.normalize import key as vkey

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRACTICE = os.path.join(ROOT, "data", "practice.csv")
FOLDS = 5
AGREE_BOOST = 0.5  # conf -> 1 - (1 - conf) * 0.5 when both readers agree


def _pages(rows: list[dict]) -> list[int]:
    out, page, prev = [], 0, None
    for r in rows:
        if r.get("date") != prev:
            page += 1
            prev = r.get("date")
        out.append(page)
    return out


def _gold_rows(gold_rows):
    if gold_rows is None:
        from app.io.reader import read_rows

        gold_rows = read_rows(PRACTICE) if os.path.exists(PRACTICE) else []
    out = []
    for r in gold_rows:
        form = r.get("form")
        if isinstance(form, list):
            form = json.dumps(form)
        if form:
            out.append({"id": r["id"], "text": r.get("text", ""), "date": r.get("date", ""), "form": form})
    return out


def _arckan(train: list[dict], target: list[dict], lexicon: dict) -> list[dict]:
    from app.extract.extractor import FactExtractor, Lexicon

    ex = FactExtractor(Lexicon.from_dict(lexicon)).fit(train)
    return [{"row_id": f.row_id, "subject": f.subject, "attribute": f.attribute, "value": f.value,
             "conf": float(f.conf), "span": f.span, "anchored": bool(f.anchored)}
            for facts in ex.extract(target) for f in facts]


def _simple(train: list[dict], target: list[dict], lexicon: dict) -> set[tuple]:
    from app.extract import simple as XS

    if not train or not target:
        return set()
    T = pd.DataFrame(train)
    gaz = XS.Gazetteer(lexicon)
    picker = XS.AttributePicker(gaz).fit(XS.training_docs(T, XS.gold_facts(T), gaz))
    D = pd.DataFrame([{"id": r["id"], "text": r.get("text", ""), "date": r.get("date", "")} for r in target])
    D["page"] = _pages(target)
    return {(f["id"], f["subject"], f["attribute"], vkey(f["value"])) for f in XS.extract_rows(D, gaz, picker)}


def _read(train, target, lexicon, report):
    facts = _arckan(train, target, lexicon)
    try:
        agree = _simple(train, target, lexicon)
    except Exception as e:  # noqa: BLE001 - the second opinion is optional
        report.setdefault("errors", []).append(f"simple extractor: {e!r}"[:200])
        agree = set()
    for f in facts:
        f["agree"] = (f["row_id"], f["subject"], f["attribute"], vkey(f["value"])) in agree
        if f["agree"]:
            f["conf"] = 1 - (1 - f["conf"]) * AGREE_BOOST
    return facts


def extract(rows: list[dict], lexicon: dict, gold_rows: list[dict] | None = None,
            report: dict | None = None) -> list[dict]:
    """Facts of every row: dicts with row_id, subject, attribute, value, conf, span, anchored, agree."""
    report = report if report is not None else {}
    gold = _gold_rows(gold_rows)
    if not gold:
        report["extract"] = "no practice rows with forms: nothing to learn from"
        return []
    gold_ids = {g["id"] for g in gold}
    inside = [r for r in rows if r["id"] in gold_ids]
    outside = [r for r in rows if r["id"] not in gold_ids]
    facts = _read(gold, outside, lexicon, report) if outside else []
    if inside:  # cross-fitted: each practice row is read by models that never saw its page
        page_of = dict(zip([g["id"] for g in gold], _pages(gold)))
        fold_rows = defaultdict(list)
        for r in inside:
            fold_rows[page_of[r["id"]] % FOLDS].append(r)
        for k, target in fold_rows.items():
            train = [g for g in gold if page_of[g["id"]] % FOLDS != k]
            facts += _read(train, target, lexicon, report)
    report.update({"extracted_facts": len(facts), "rows_read": len(rows), "cross_fitted_rows": len(inside),
                   "agreeing_facts": sum(f["agree"] for f in facts)})
    return facts


__all__ = ["extract"]
