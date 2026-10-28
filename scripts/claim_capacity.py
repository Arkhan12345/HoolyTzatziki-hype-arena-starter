"""How many true, valid, new claims can the KB support? (no selection limits applied)

    python scripts/claim_capacity.py                  # practice gold, bundled lexicon
    python scripts/claim_capacity.py --json out.json  # also write the full report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.generate.candidates import current  # noqa: E402
from app.generate.select import rank, screen  # noqa: E402
from app.io.reader import read_lexicon, read_practice  # noqa: E402
from app.kb import normalize as nz  # noqa: E402
from app.kb.inventory import FactInventory  # noqa: E402
from app.kb.store import KB  # noqa: E402
from app.validate.rules import norm_text  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--practice", default=os.path.join(ROOT, "data", "practice.csv"))
    ap.add_argument("--lexicon", default=os.path.join(ROOT, "data", "lexicon.json"))
    ap.add_argument("--json")
    args = ap.parse_args()

    rows, lex = read_practice(args.practice), read_lexicon(args.lexicon)
    kb = KB(lex).build([], rows)
    inv = FactInventory.from_kb(kb)
    texts = {norm_text(r["text"]) for r in rows}

    gold_facts = sum(len(r["form"]) for r in rows if r["label"] == "true")
    cur = {p: current(inv, *p) for p in inv.pairs()}
    valid_now = {p: c for p, c in cur.items() if c[0] is not None and c[1] >= 0.6}
    # compares need a current value on both sides (any recency: robust order across history lifts them)
    comparable = [p for p, c in cur.items() if c[0] is not None and nz.dimension(c[0]) is not None]
    by_dim = Counter(nz.dimension(cur[p][0]) for p in comparable)
    pair_count = sum(n * (n - 1) // 2 for n in by_dim.values())

    rep: dict = {}
    valid = screen(inv, texts, rep)
    distinct, dups = rank(valid)
    conf_hist = {t: Counter() for t in ("compare", "fact", "change", "at_event", "relation")}
    for c in distinct:
        conf_hist[c.type][f"{c.confidence:.2f}"] += 1
    distinct_compare = sorted((c.confidence for c in distinct if c.type == "compare"), reverse=True)

    out = {
        "gold_rows_true": sum(1 for r in rows if r["label"] == "true"),
        "gold_facts": gold_facts,
        "pairs_with_trusted_values": inv.stats()["pairs_with_trusted"],
        "temporally_valid_facts_on_as_of": len(valid_now),
        "comparable_facts": len(comparable),
        "comparable_fact_pairs_unordered": pair_count,
        "candidates_by_type": rep["candidates_by_type"],
        "rejected_by_rule": rep["rejected"],
        "valid_new_by_type": dict(Counter(c.type for c in valid)),
        "valid_ordered_compares": sum(1 for c in valid if c.type == "compare"),
        "distinct_semantic_by_type": dict(Counter(c.type for c in distinct)),
        "semantic_duplicates": len(dups),
        "total_valid_new": len(valid),
        "total_distinct_semantic": len(distinct),
        "confidence_histogram_distinct": {t: dict(sorted(h.items(), reverse=True)) for t, h in conf_hist.items()},
        "hypothesis_1000_compare": {
            "distinct_compares": len(distinct_compare),
            "feasible": len(distinct_compare) >= 1000,
            "min_confidence_of_top_1000": distinct_compare[999] if len(distinct_compare) >= 1000 else None,
            "robust_distinct_compares": sum(1 for c in distinct if c.type == "compare" and c.meta.get("robust")),
        },
    }
    print(json.dumps(out, indent=1, default=str))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
