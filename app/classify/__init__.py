"""The pipeline's `classify(rows, facts, kb)`: train on labelled data, label the unlabelled universe rows.

Training data = the universe rows that come with a label + generated items in the 5 claim types
(app/classify/datagen.py: true from the claim generator, false by flipped logic, unsure from unstated pairs).
The classifier's KB (app/classify/timeline.py) is pinned only by the input's labelled rows (TZ_KB_LABELS=0: no
labels at all); a labelled row's own features come from a KB without its page's labels (out-of-fold). Generated items
never pin or vote: their labels are training targets only.

Modes (env TZ_MODE, or the `mode` argument; the API passes the frontend's choice):
  prod   train on the labelled rows + ALL generated items, label the unlabelled rows (the contest's task)
  debug  generated items split in half (stratified by type and label): train on the labelled rows + one half, run
         inference on the other half and measure accuracy there (every held-out prediction is reported); any
         unlabelled input rows are still labelled by that model so the run's table stays complete
Every label carries a confidence (the model's probability, 0.5..1, empty for unsure as the contract says) and,
in the auxiliary output (never in output.csv), a trust score and the rows that support or contradict it.
"""
from __future__ import annotations

import os
import random
import time
from collections import defaultdict

import numpy as np

from app.io.types import Label

LAST: dict = {}  # details of the latest run (per-row trust and sources, debug metrics), for the API / aux files
DEBUG_TEST_SHARE = 0.5
QUOTE_ROWS = 2000  # sources carry their sentence text up to this many labelled rows (the demo); ids only beyond


def _truth_gold(rows, gold_rows):
    """Practice rows the truth KB may use: never one whose label the input hides (no leak into the items)."""
    hidden = {r["id"] for r in rows if not r.get("label")}
    return [g for g in gold_rows if g["id"] not in hidden]


def _split(items, share, seed=0):
    by = defaultdict(list)
    for it in items:
        by[(it["type"], it["label"])].append(it)
    rng = random.Random(seed)
    train, test = [], []
    for k in sorted(by):
        grp = by[k][:]
        rng.shuffle(grp)
        cut = int(round(len(grp) * (1 - share)))
        train += grp[:cut]
        test += grp[cut:]
    return train, test


def _metrics(y_true, y_pred, types):
    from app.classify.model import CLASSES

    y_true, y_pred, types = (np.asarray([str(x) for x in a]) for a in (y_true, y_pred, types))
    CLASSES = [str(c) for c in CLASSES]
    out = {"n": int(len(y_true)), "accuracy": round(float((y_true == y_pred).mean()), 4) if len(y_true) else None}
    out["by_type"] = {t: {"n": int((types == t).sum()), "accuracy": round(float((y_true[types == t] == y_pred[types == t]).mean()), 4)}
                      for t in sorted(set(types))}
    out["by_label"] = {c: {"n": int((y_true == c).sum()),
                           "recall": round(float((y_pred[y_true == c] == c).mean()), 4) if (y_true == c).any() else None,
                           "precision": round(float((y_true[y_pred == c] == c).mean()), 4) if (y_pred == c).any() else None}
                       for c in CLASSES}
    out["confusion"] = {t: {p: int(((y_true == t) & (y_pred == p)).sum()) for p in CLASSES} for t in CLASSES}
    return out


def classify(rows, facts, kb, mode: str | None = None, report: dict | None = None, gold_rows=None,
             kb_labels: bool | None = None):
    from app.classify import model as Mdl
    from app.classify.datagen import generate_items
    from app.kb.store import KB
    from app.validate.rules import norm_text

    t0 = time.time()
    report = report if report is not None else {}
    mode = (mode or os.environ.get("TZ_MODE") or "prod").lower()
    lexicon = kb.lex.raw if kb is not None else {"subjects": [], "events": []}
    if gold_rows is None:
        from app.io.reader import read_practice
        from app.extract import PRACTICE

        gold_rows = read_practice(PRACTICE) if os.path.exists(PRACTICE) else []
    gold = _truth_gold(rows, gold_rows)
    truth = KB(lexicon).build([], gold, None)
    texts = {norm_text(r.get("text", "")) for r in rows} | {norm_text(g.get("text", "")) for g in gold}
    gen_rep: dict = {}
    items = generate_items(truth, lexicon, gold, texts, gen_rep)
    t_gen = time.time()

    use_labels = kb_labels if kb_labels is not None else os.environ.get("TZ_KB_LABELS", "1") != "0"
    gold_forms = {g["id"]: g.get("form") or [] for g in gold}  # pins only; `gold` already excludes hidden rows
    gold_text = {g["id"]: g.get("text", "") for g in gold}
    world = Mdl.World(rows, facts or [], items, use_labels=use_labels, gold_forms=gold_forms)
    t_kb = time.time()

    labelled = [r for r in rows if r.get("label") in ("true", "false", "unsure")]
    targets = [r for r in rows if not r.get("label")]
    if mode == "debug":  # train on the labelled rows + half the generated sentences; infer and score on the other half
        train_items, test_items = _split(items, DEBUG_TEST_SHARE)
    else:
        train_items, test_items = items, []
    train_rows = labelled
    refs = Mdl.sentence_refs(train_rows) + Mdl.item_refs(train_items)
    y = np.array([list(Mdl.CLASSES).index(r["label"]) for r in train_rows] +
                 [list(Mdl.CLASSES).index(it["label"]) for it in train_items])
    if len(set(y.tolist())) < 2:  # nothing (or one class) to learn from: the pipeline keeps its baseline labels
        report.update({"mode": mode, "skipped": f"{len(refs)} training rows, {len(set(y.tolist()))} classes"})
        return []
    X = world.matrix(refs)
    cols = list(X.columns)
    model = Mdl.fit(X, y)
    t_fit = time.time()

    debug = None
    if test_items:
        Xt = world.matrix(Mdl.item_refs(test_items)).reindex(columns=cols)
        Pt = Mdl.predict(model, Xt)
        pt = Mdl.CLASSES[Pt.argmax(1)]
        debug = _metrics([it["label"] for it in test_items], pt, [it["type"] for it in test_items])
        debug["train_items"], debug["test_items"] = len(train_items), len(test_items)
        debug["train_rows"] = len(refs)
        # the inference on the held-out half, every sentence (the frontend lists them)
        debug["predictions"] = [
            {"id": it["id"], "type": it["type"], "text": it["text"], "date": it["date"], "gold": it["label"],
             "pred": str(p), "correct": str(p) == it["label"],
             "confidence": None if str(p) == "unsure" else round(float(max(0.5, pp.max())), 3),
             "probabilities": {str(c): round(float(x), 3) for c, x in zip(Mdl.CLASSES, pp)},
             "origin": it["origin"], "sources": it["sources"],
             "source_texts": {s: world.text_of.get(s) or gold_text.get(s, "") for s in it["sources"]}}
            for it, p, pp in zip(test_items, pt, Pt)]

    labels, details = [], {}
    world.with_quotes = len(targets) <= QUOTE_ROWS
    if targets:
        Xu = world.matrix(Mdl.sentence_refs(targets)).reindex(columns=cols)
        P = Mdl.predict(model, Xu)
        for r, p in zip(targets, P):
            k = int(p.argmax())
            lab = str(Mdl.CLASSES[k])
            conf = None if lab == "unsure" else round(float(min(1.0, max(0.5, p[k]))), 3)
            labels.append(Label(id=r["id"], label=lab, confidence=conf))
            info = world.trust_and_sources(r["id"], r.get("date", ""))
            details[r["id"]] = {"label": lab, "confidence": conf,
                                "probabilities": {c: round(float(x), 3) for c, x in zip(Mdl.CLASSES, p)}, **info}
    t_pred = time.time()

    imp = sorted(zip(cols, getattr(model, "feature_importances_", [])), key=lambda t: -t[1])[:15]
    report.update({
        "mode": mode, "labelled_rows": len(labelled), "unlabelled_rows": len(targets),
        "generated": gen_rep, "train_rows": len(refs), "features": len(cols),
        "kb": {"labels": "input-labelled rows pin the KB (out-of-fold for their own features)" if use_labels
               else "label-free", "pins": int((world.tl.anchor == 1).sum()), "forbids": int((world.tl.anchor == -1).sum()),
               "votes": int((world.tl.anchor == 0).sum()), "queries": int((world.F.kind == "item").sum()),
               "pairs": len(world.tl.pairs), "pages": len(world.trust)},
        "debug": debug, "top_features": [{"feature": c, "importance": int(v)} for c, v in imp],
        "seconds": {"generate_items": round(t_gen - t0, 2), "kb_em": round(t_kb - t_gen, 2),
                    "fit": round(t_fit - t_kb, 2), "predict": round(t_pred - t_fit, 2)},
    })
    LAST.clear()
    LAST.update({"report": report, "details": details})
    return labels


__all__ = ["classify", "LAST"]
