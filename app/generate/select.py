"""Claim selection: candidates -> truth check -> contract check -> novelty -> rank -> exactly n.

Confidence first. No quotas per type and no caps per subject: if the best n are all compares, the
output is n compares. Order of preference:
  1. truth-verified, contract-valid, new text;
  2. one claim per semantic key first (a paraphrase or the other side of a compare states nothing new),
     best confidence first; ties broken by spreading over (subject, attribute) pairs;
  3. semantic duplicates only if the distinct claims run out before n.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict

from app.generate.candidates import FACTORIES, Candidate
from app.generate.truth import truth_problems
from app.io.types import Claim
from app.kb.inventory import FactInventory
from app.validate.rules import norm_text, validate_claim


def all_candidates(inv: FactInventory, report: dict | None = None):
    counts = Counter()
    for factory in FACTORIES:
        try:
            for c in factory(inv):
                counts[c.type] += 1
                yield c
        except Exception as e:  # noqa: BLE001 - one broken factory must not stop the others
            print(f"[generate] {factory.__name__} failed: {e!r}", file=sys.stderr)
    if report is not None:
        report["candidates_by_type"] = dict(counts)


def screen(inv: FactInventory, input_texts: set[str], report: dict) -> list[Candidate]:
    """Every candidate that is true, contract-valid and new (unique text)."""
    valid, seen_text = [], set()
    rejected = defaultdict(Counter)
    t_truth = t_contract = 0.0
    n = 0
    for c in all_candidates(inv, report):
        n += 1
        t = time.perf_counter()
        tp = truth_problems(c.type, c.form, inv)
        t_truth += time.perf_counter() - t
        if tp:
            rejected[c.type][tp[0].split(":")[0] + ": " + tp[0].split(":", 1)[-1].strip()[:40]] += 1
            continue
        probe = Claim(id="tz-probe", type=c.type, text=c.text, form=c.form, confidence=min(1, max(0.5, c.confidence)))
        probe.meta = c.meta  # type: ignore[attr-defined]
        t = time.perf_counter()
        cp = validate_claim(probe, inv.lex, None)
        t_contract += time.perf_counter() - t
        if cp:
            rejected[c.type][cp[0].split(":")[0][:60]] += 1
            continue
        nt = norm_text(c.text)
        if nt in input_texts:
            rejected[c.type]["copy of an input row"] += 1
            continue
        if nt in seen_text:
            rejected[c.type]["duplicate text"] += 1
            continue
        seen_text.add(nt)
        valid.append(c)
    report.update({"candidates": n, "truth_s": round(t_truth, 3), "validate_s": round(t_contract, 3),
                   "rejected": {k: dict(v.most_common()) for k, v in rejected.items()}})
    return valid


def rank(valid: list[Candidate]) -> tuple[list[Candidate], list[Candidate]]:
    """(distinct, duplicates): best candidate per semantic key, and the rest; both best first."""
    groups: dict[tuple, list[Candidate]] = defaultdict(list)
    for c in valid:
        groups[c.semantic_key].append(c)
    distinct, dups = [], []
    for g in groups.values():
        g.sort(key=lambda c: (-c.confidence, c.meta.get("template", 0), c.text))
        distinct.append(g[0])
        dups.extend(g[1:])
    return _spread(distinct), _spread(dups)


def _spread(cs: list[Candidate]) -> list[Candidate]:
    """Sort by confidence; among equal confidence, interleave (subject, attribute) pairs."""
    cs = sorted(cs, key=lambda c: (-c.confidence, c.type, c.text))
    out, i = [], 0
    while i < len(cs):
        j = i
        while j < len(cs) and cs[j].confidence == cs[i].confidence:
            j += 1
        seen = Counter()
        bucket = []
        for c in cs[i:j]:
            p = _pair_of(c)
            bucket.append((seen[p], c))
            seen[p] += 1
        out.extend(c for _, c in sorted(bucket, key=lambda x: x[0]))
        i = j
    return out


def _pair_of(c: Candidate):
    f = c.form
    if c.type == "compare":
        return (f["left"]["subject"], f["left"]["attribute"])
    return (f["subject"], f.get("attribute"))


def select(valid: list[Candidate], n: int, report: dict) -> list[Candidate]:
    distinct, dups = rank(valid)
    chosen = distinct[:n]
    if len(chosen) < n:
        chosen += dups[: n - len(chosen)]
    report.update({
        "valid": len(valid), "distinct_semantic": len(distinct), "semantic_duplicates": len(dups),
        "valid_by_type": dict(Counter(c.type for c in valid)),
        "distinct_by_type": dict(Counter(c.type for c in distinct)),
        "selected_semantic_duplicates": max(0, len(chosen) - len(distinct[:n])),
        "min_confidence_selected": min((c.confidence for c in chosen), default=None),
    })
    return chosen


def generate_from_inventory(inv: FactInventory, input_texts: set[str], n: int = 1000,
                            report: dict | None = None) -> list[Claim]:
    rep: dict = {}
    valid = screen(inv, set(input_texts), rep)
    chosen = select(valid, n, rep)
    claims = []
    for i, c in enumerate(chosen):
        cl = Claim(id=f"tz-{i + 1:04d}", type=c.type, text=c.text, form=c.form,
                   confidence=round(min(0.99, max(0.5, c.confidence)), 3))
        try:  # provenance stays out of the submission: the writer only reads the official fields
            cl.meta = {"evidence": list(c.evidence), "semantic_key": json.dumps(c.semantic_key, default=str), **c.meta}
        except Exception:  # noqa: BLE001 - P1's Claim may be frozen/slotted
            pass
        claims.append(cl)
    rep.update({"by_type": dict(Counter(c.type for c in claims)), "n": len(claims), "inventory": inv.stats()})
    if report is not None:
        report.update(rep)
    return claims


def generate(kb, lexicon: dict, input_texts: set[str], n: int = 1000, report: dict | None = None,
             trust_extracted: bool = False) -> list[Claim]:
    """Pipeline entry (agreed interface): KB -> inventory -> claims."""
    return generate_from_inventory(FactInventory.from_kb(kb, trust_extracted=trust_extracted), input_texts, n, report)
