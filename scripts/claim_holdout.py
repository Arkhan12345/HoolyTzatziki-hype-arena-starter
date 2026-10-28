"""Measured claim precision: generate from part of practice, judge on all of it.

    python scripts/claim_holdout.py                   # hide 50% and 90% of the pages, 3 seeds each
    python scripts/claim_holdout.py --hide 0.9 --seeds 5

A page is a run of consecutive ids with the same date. Claims are generated from the kept pages
only and judged with app/generate/truth.py against the inventory of the whole practice set: a claim
the hidden pages contradict (a newer value, a different value on the event day, ...) counts as wrong.
The full set is itself a sample of the universe, so this is a lower bound on the real error rate.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.generate.select import generate_from_inventory  # noqa: E402
from app.generate.truth import truth_problems  # noqa: E402
from app.io.reader import read_lexicon, read_practice  # noqa: E402
from app.kb.inventory import FactInventory  # noqa: E402
from app.kb.store import KB  # noqa: E402
from app.validate.rules import norm_text  # noqa: E402


def pages(rows):
    out, prev = [], None
    for r in sorted(rows, key=lambda r: int(re.sub(r"\D", "", r["id"]) or 0)):
        if prev is None or r["date"] != prev["date"]:
            out.append([])
        out[-1].append(r)
        prev = r
    return out


def band(c):
    return "0.95+" if c >= 0.95 else "0.90-0.95" if c >= 0.9 else "0.80-0.90" if c >= 0.8 else "<0.80"


def run(rows, lex, hide, seed, texts, full_inv):
    pg = pages(rows)
    rnd = random.Random(seed)
    kept = [r for p in pg if rnd.random() >= hide for r in p]
    inv = FactInventory.from_kb(KB(lex).build([], kept))
    claims = generate_from_inventory(inv, texts, 1000, {})
    stats = defaultdict(lambda: [0, 0])
    for c in claims:
        problems = truth_problems(c.type, c.form, full_inv)
        wrong = bool(problems)
        # "not held"/"no current value" = the full data cannot confirm it; anything else contradicts it
        unconfirmed = wrong and any(w in problems[0] for w in ("not held on", "no current value", "no trusted day"))
        meta = getattr(c, "meta", {}) or {}
        detail = (f"{c.type}/robust={meta.get('robust')}" if c.type == "compare" else
                  f"{c.type}/{band(c.confidence)}")
        for k in (("type", c.type), ("band", band(c.confidence)), ("detail", detail), ("all", "all")):
            stats[k][0] += 1
            stats[k][1] += wrong
            stats[k + ("unconfirmed",)][1] += unconfirmed
    return len(kept), claims, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hide", type=float, nargs="*", default=[0.5, 0.9])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--json")
    args = ap.parse_args()
    if args.seeds < 1:
        ap.error("--seeds must be at least 1")
    if any(not 0 <= h < 1 for h in args.hide):
        ap.error("--hide takes shares of pages in [0, 1)")
    rows, lex = read_practice(os.path.join(ROOT, "data", "practice.csv")), read_lexicon(os.path.join(ROOT, "data", "lexicon.json"))
    texts = {norm_text(r["text"]) for r in rows}
    full_inv = FactInventory.from_kb(KB(lex).build([], rows))
    report = {}
    for hide in args.hide:
        agg = defaultdict(lambda: [0, 0])
        kept_n, n_claims = [], []
        for seed in range(args.seeds):
            kept, claims, stats = run(rows, lex, hide, seed, texts, full_inv)
            kept_n.append(kept)
            n_claims.append(len(claims))
            for k, (n, w) in stats.items():
                agg[k][0] += n
                agg[k][1] += w
        key = f"hide_{hide:.2f}"
        report[key] = {
            "rows_kept_avg": round(sum(kept_n) / len(kept_n)), "claims_avg": round(sum(n_claims) / len(n_claims)),
            "wrong_rate": {f"{k[0]}:{k[1]}": {"n": n, "wrong": w, "unconfirmed": agg[k + ("unconfirmed",)][1],
                                              "rate": round(w / n, 4) if n else None}
                           for k, (n, w) in sorted(agg.items()) if len(k) == 2},
        }
        print(f"\nhide {hide:.0%} of pages, {args.seeds} seeds: kept ~{report[key]['rows_kept_avg']} rows, "
              f"~{report[key]['claims_avg']} claims")
        for k, v in report[key]["wrong_rate"].items():
            print(f"  {k:34s} n={v['n']:5d}  wrong={v['wrong']:4d} (unconfirmed {v['unconfirmed']:4d})  rate={v['rate']}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
