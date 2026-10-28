"""Debug: claims from A (practice gold, production) vs B (X/Arckan's extraction + classification).

    python scripts/compare_generators.py                      # B from app.extract/app.classify, else the MOCK
    python scripts/compare_generators.py --structured b.json  # B from their dumped output (shape: app/generate/strategies.py)
    python scripts/compare_generators.py --mock --noise 0.1   # B = MOCK classifier that mislabels 10% of rows

Setting: practice pages split at random (--hide, default 50%). A sees the gold of the kept pages only;
B sees its classifier's output on every row (as in the arena, where it reads the whole universe).
Both go through the same generator, validator and selection, and both are judged with
app/generate/truth.py against all of practice. Files in --out: claims_A.jsonl, claims_B.jsonl
(every claim with its source sentences and verdict) and comparison.json.

The MOCK is not a model: it labels each practice row with its gold label (flipped with probability
--noise) and returns the gold forms as facts. It only shows how B behaves for a given error rate.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.generate import strategies  # noqa: E402
from app.generate.select import generate_from_inventory  # noqa: E402
from app.generate.truth import truth_problems  # noqa: E402
from app.io.reader import read_lexicon, read_practice  # noqa: E402
from app.kb.inventory import FactInventory  # noqa: E402
from app.kb.store import KB  # noqa: E402
from app.validate.rules import norm_text  # noqa: E402
from scripts.claim_holdout import pages  # noqa: E402

UNCONFIRMED = ("not held on", "no current value", "no trusted day")


def mock_structured(practice: list[dict], noise: float, seed: int) -> tuple[list[dict], list[dict]]:
    rnd = random.Random(seed)
    labels, facts = [], []
    for r in practice:
        label = r["label"]
        if label in ("true", "false") and rnd.random() < noise:
            label = "false" if label == "true" else "true"
        labels.append({"id": r["id"], "label": label, "confidence": round(rnd.uniform(0.9, 0.99), 3)})
        facts += [{"row_id": r["id"], "subject": f["subject"], "attribute": f["attribute"], "value": f["value"],
                   "span": r["text"], "conf": 0.95} for f in r["form"] if "op" not in f["value"]]
    return labels, facts


def judge(claims, full: FactInventory) -> list[dict]:
    out = []
    for c in claims:
        p = truth_problems(c.type, c.form, full)
        verdict = "holds" if not p else ("unconfirmed" if any(w in p[0] for w in UNCONFIRMED) else "contradicted")
        meta = getattr(c, "meta", {}) or {}
        out.append({"id": c.id, "type": c.type, "text": c.text, "form": c.form, "confidence": c.confidence,
                    "verdict": verdict, "problem": p[0] if p else None, "semantic_key": meta.get("semantic_key"),
                    "sources": meta.get("sources", [])})
    return out


def summary(recs: list[dict]) -> dict:
    by = {}
    for t in sorted({r["type"] for r in recs}) + ["all"]:
        rs = [r for r in recs if t in ("all", r["type"])]
        v = Counter(r["verdict"] for r in rs)
        by[t] = {"n": len(rs), "contradicted": v["contradicted"], "unconfirmed": v["unconfirmed"],
                 "wrong_rate": round((v["contradicted"] + v["unconfirmed"]) / len(rs), 4) if rs else None}
    return by


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--structured", help="JSON with B's labels + facts (shared-interface shape)")
    ap.add_argument("--mock", action="store_true", help="force the MOCK classifier for B")
    ap.add_argument("--noise", type=float, default=0.0, help="MOCK: share of rows whose label is flipped")
    ap.add_argument("--min-conf", type=float, default=strategies.MIN_CONF)
    ap.add_argument("--hide", type=float, default=0.5, help="share of practice pages A does not see")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "compare"))
    args = ap.parse_args()
    if not 0 <= args.hide < 1 or not 0 <= args.noise <= 1:
        ap.error("--hide in [0, 1), --noise in [0, 1]")

    practice = read_practice(os.path.join(ROOT, "data", "practice.csv"))
    lex = read_lexicon(os.path.join(ROOT, "data", "lexicon.json"))
    universe = [{"id": r["id"], "text": r["text"], "date": r["date"], "label": ""} for r in practice]
    texts = {norm_text(r["text"]) for r in practice}
    full = FactInventory.from_kb(KB(lex).build([], practice))

    rnd = random.Random(args.seed)
    kept = [r for p in pages(practice) if rnd.random() >= args.hide for r in p]

    if args.structured:
        b_labels, b_facts = strategies.load_structured(args.structured)
        b_source = f"file {args.structured}"
    else:
        real = None if args.mock else strategies.from_p3(universe, lex)
        if real is not None:
            b_labels, b_facts = real
            b_source = "app.extract + app.classify"
        else:
            b_labels, b_facts = mock_structured(practice, args.noise, args.seed)
            b_source = f"MOCK classifier (gold forms, {args.noise:.0%} labels flipped) - not a model"

    inv_a = strategies.inventory_gold(lex, kept, universe)
    inv_b = strategies.inventory_classified(lex, universe, b_labels, b_facts, args.min_conf)
    rep_a, rep_b = {}, {}
    claims_a = generate_from_inventory(inv_a, texts, 1000, rep_a)
    claims_b = generate_from_inventory(inv_b, texts, 1000, rep_b)
    ja, jb = judge(claims_a, full), judge(claims_b, full)

    keys_a, keys_b = {r["semantic_key"] for r in ja}, {r["semantic_key"] for r in jb}
    result = {
        "setting": {"hide": args.hide, "seed": args.seed, "a_gold_rows": len(kept), "b_source": b_source,
                    "b_min_conf": args.min_conf, "b_rows_labelled_true": sum(1 for lb in b_labels
                                                                            if strategies._get(lb, "label") == "true")},
        "A_gold": {"selected": len(claims_a), "valid": rep_a.get("valid"), "distinct": rep_a.get("distinct_semantic"),
                   "by_type": rep_a.get("by_type"), "judged": summary(ja)},
        "B_classified": {"selected": len(claims_b), "valid": rep_b.get("valid"), "distinct": rep_b.get("distinct_semantic"),
                         "by_type": rep_b.get("by_type"), "judged": summary(jb)},
        "overlap": {"same_claim_in_both": len(keys_a & keys_b), "only_A": len(keys_a - keys_b),
                    "only_B": len(keys_b - keys_a)},
    }
    os.makedirs(args.out, exist_ok=True)
    for name, recs in (("claims_A.jsonl", ja), ("claims_B.jsonl", jb)):
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
            for r in recs:
                r["in_other"] = r["semantic_key"] in (keys_b if name.endswith("A.jsonl") else keys_a)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)

    print(f"B = {b_source}; A sees {len(kept)} gold rows ({args.hide:.0%} of pages hidden)")
    for name in ("A_gold", "B_classified"):
        r = result[name]
        j = r["judged"]["all"]
        print(f"{name:13s} selected {r['selected']:4d}  distinct available {r['distinct']}  wrong {j['wrong_rate']:.1%} "
              f"({j['contradicted']} contradicted, {j['unconfirmed']} unconfirmed)  by type {r['by_type']}")
    print(f"overlap: {result['overlap']}   files: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
