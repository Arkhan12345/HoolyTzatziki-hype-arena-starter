"""Claim selection: candidates -> independent validator -> semantic dedup -> rank -> exactly n.

Every candidate goes through app/validate/claims.py (contract, truth recomputed from the KB, novelty,
provenance); only PASS candidates can be selected, so a rejected one is replaced by the next best.
Confidence first, no quotas. Fallback ladder when the strict pool runs short: confident and clearly
new claims, then confident ones a sentence might weakly restate, then the less confident ones; never a
FAIL, an UNVERIFIED or a duplicate (fewer than n claims, logged, rather than that). Two rules on top:
  - one claim per piece of information (mentor's rule): paraphrases, inverse compares and aliases
    share a semantic key and only the best one can be selected, never two;
  - a soft preference for all five types: if a type has PASS candidates at confidence >= 0.85 but fewer
    than MIN_PER_TYPE were selected, its best ones replace the least confident selected claims.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict

from app.generate.candidates import FACTORIES, Candidate
from app.io.types import Claim
from app.kb.inventory import FactInventory
from app.validate.claims import PASS, ClaimValidator
from app.validate.claims import report as validation_report
from app.validate.rules import norm_text

MIN_PER_TYPE = 25
SOFT_MIN_CONF = 0.85


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


def screen(inv: FactInventory, input_texts: set[str], report: dict,
           validator: ClaimValidator | None = None) -> list[Candidate]:
    """Every candidate the independent validator passes, one per text."""
    validator = validator or ClaimValidator(inv, input_texts)
    valid, seen_text = [], set()
    rejected = defaultdict(Counter)
    verdicts = defaultdict(Counter)
    t0, n = time.perf_counter(), 0
    for c in all_candidates(inv, report):
        n += 1
        probe = Claim(id="tz-probe", type=c.type, text=c.text, form=c.form, confidence=min(1, max(0.5, c.confidence)))
        v = validator.check(probe, c.meta)
        verdicts[c.type][v.verdict] += 1
        if v.verdict != PASS:
            first = v.reasons[0] if v.reasons else v.verdict
            rejected[c.type][first.split(":")[0] + ": " + first.split(":", 1)[-1].strip()[:45]] += 1
            continue
        nt = norm_text(c.text)
        if nt in seen_text:
            rejected[c.type]["duplicate text"] += 1
            continue
        seen_text.add(nt)
        c.meta["vkey"] = v.semantic_key
        if "novelty_weak" in v.flags:
            c.meta["novelty_weak"] = True
        valid.append(c)
    report.update({"candidates": n, "validate_s": round(time.perf_counter() - t0, 3), "truth_s": 0.0,
                   "verdicts_by_type": {k: dict(v) for k, v in verdicts.items()},
                   "rejected": {k: dict(v.most_common()) for k, v in rejected.items()}})
    return valid


def rank(valid: list[Candidate]) -> tuple[list[Candidate], list[Candidate]]:
    """(distinct, duplicates): best candidate per (validator) semantic key, and the rest; both best first."""
    groups: dict[tuple, list[Candidate]] = defaultdict(list)
    for c in valid:
        groups[c.meta.get("vkey") or c.semantic_key].append(c)
    distinct, dups = [], []
    for g in groups.values():
        g.sort(key=lambda c: (_tier(c), -c.confidence, c.meta.get("template", 0), c.text))
        distinct.append(g[0])
        dups.extend(g[1:])
    # fallback ladder: confident and clearly new, then confident with a weak "a sentence might say it",
    # then the less confident ones in the same order. FAIL / UNVERIFIED / duplicates never enter.
    tiers = defaultdict(list)
    for c in distinct:
        tiers[_tier(c)].append(c)
    ordered = [c for t in sorted(tiers) for c in _spread(tiers[t])]
    return ordered, _spread(dups)


def _tier(c: Candidate) -> tuple[bool, bool]:
    """Fallback ladder position: (less confident, possibly restated); lower is better."""
    return (c.confidence < SOFT_MIN_CONF, bool(c.meta.get("novelty_weak")))


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


def _is_inverse(c: Candidate) -> bool:
    """A compare whose left side is not the canonical first side: the mirror of another wording."""
    key = c.meta.get("vkey") or ()
    return bool(key) and key[0] == "compare" and (c.form["left"]["subject"], c.form["left"]["attribute"]) != tuple(key[1])


def _prefer_all_types(distinct: list[Candidate], n: int) -> list[Candidate]:
    """Best n by confidence; then make room for types with strong PASS candidates but < MIN_PER_TYPE taken."""
    chosen = list(distinct[:n])
    rest = distinct[n:]
    counts = Counter(c.type for c in chosen)
    for t in ("fact", "compare", "change", "at_event", "relation"):
        need = MIN_PER_TYPE - counts[t]
        if need <= 0:
            continue
        for c in [c for c in rest if c.type == t and c.confidence >= SOFT_MIN_CONF][:need]:
            # drop the last-ranked claim of a type that stays above its minimum, never one from a
            # better fallback tier than the claim that replaces it
            for i in range(len(chosen) - 1, -1, -1):
                d = chosen[i]
                if counts[d.type] > MIN_PER_TYPE and _tier(d) >= _tier(c):
                    chosen.pop(i)
                    counts[d.type] -= 1
                    chosen.append(c)
                    counts[t] += 1
                    break
    return sorted(chosen, key=lambda c: (_tier(c), -c.confidence))


def select(valid: list[Candidate], n: int, report: dict, gold_rows: set[str] | None = None) -> list[Candidate]:
    distinct, dups = rank(valid)
    chosen = _prefer_all_types(distinct, n)
    inverse = sum(1 for c in dups if _is_inverse(c))
    report.update({
        "valid": len(valid), "distinct_semantic": len(distinct), "semantic_duplicates": len(dups),
        "semantic_duplicates_inverse_compare": inverse, "semantic_duplicates_paraphrase": len(dups) - inverse,
        "valid_by_type": dict(Counter(c.type for c in valid)),
        "distinct_by_type": dict(Counter(c.type for c in distinct)),
        "selected_semantic_duplicates": 0,  # never: one claim per piece of information
        "shortfall": max(0, n - len(chosen)),
        "fallback": {"selected_novelty_weak": sum(1 for c in chosen if c.meta.get("novelty_weak")),
                     "selected_below_soft_min_conf": sum(1 for c in chosen if c.confidence < SOFT_MIN_CONF),
                     "available_novelty_weak": sum(1 for c in distinct if c.meta.get("novelty_weak"))},
        "min_confidence_selected": min((c.confidence for c in chosen), default=None),
    })
    if gold_rows is not None:
        gold_only = lambda c: bool(c.evidence) and all(r in gold_rows for r in c.evidence)  # noqa: E731
        report.update({"valid_gold_only": sum(1 for c in valid if gold_only(c)),
                       "selected_gold_only": sum(1 for c in chosen if gold_only(c)),
                       "selected_kb_supported": sum(1 for c in chosen if not gold_only(c))})
    return chosen


def claim_pairs(ctype: str, form: dict) -> list[tuple[str, str]]:
    """The (subject, attribute) facts a claim is about."""
    if ctype == "compare":
        return [(form["left"]["subject"], form["left"]["attribute"]), (form["right"]["subject"], form["right"]["attribute"])]
    if ctype == "relation":
        return [(form["subject"], form["link"]), (form["target"], form["attribute"])]
    if ctype == "at_event":
        return [(form["subject"], form["attribute"]), (form["event"], "date")]
    return [(form["subject"], form["attribute"])]


def generate_from_inventory(inv: FactInventory, input_texts: set[str], n: int = 1000,
                            report: dict | None = None) -> list[Claim]:
    rep: dict = {}
    validator = ClaimValidator(inv, set(input_texts))
    valid = screen(inv, set(input_texts), rep, validator)
    chosen = select(valid, n, rep, inv.gold_rows)
    claims = []
    for i, c in enumerate(chosen):
        cl = Claim(id=f"tz-{i + 1:04d}", type=c.type, text=c.text, form=c.form,
                   confidence=round(min(0.99, max(0.5, c.confidence)), 3))
        try:  # provenance stays out of the submission: the writer only reads the official fields
            cl.meta = {"evidence": list(c.evidence), "sources": inv.sources(c.evidence, pairs=claim_pairs(c.type, c.form)),
                       "semantic_key": json.dumps(c.semantic_key, default=str),
                       **{k: v for k, v in c.meta.items() if k != "vkey"}}
        except Exception:  # noqa: BLE001 - P1's Claim may be frozen/slotted
            pass
        claims.append(cl)
    # the final set, re-validated as a whole (uniqueness across the 1,000)
    verdicts = validator.check_set(claims)
    for cl, v in zip(claims, verdicts):
        try:
            cl.meta["validation"] = v.as_dict()
        except Exception:  # noqa: BLE001
            pass
    vrep = validation_report(claims, verdicts, {
        "generated": rep.get("candidates"), "by_type": rep.get("candidates_by_type"),
        "verdicts_by_type": rep.get("verdicts_by_type"), "rejected": rep.get("rejected"),
        "passed_unique_text": rep.get("valid"), "distinct_semantic": rep.get("distinct_semantic"),
        "semantic_duplicates": {"inverse_compare": rep.get("semantic_duplicates_inverse_compare"),
                                "paraphrase_or_alias": rep.get("semantic_duplicates_paraphrase")}})
    rep.update({"by_type": dict(Counter(c.type for c in claims)), "n": len(claims), "inventory": inv.stats(),
                "validation": {**vrep["selected"], "validator_errors": validator.errors}})
    if report is not None:
        report.update(rep)
        report["_validation_report"] = vrep
    return claims


def generate(kb, lexicon: dict, input_texts: set[str], n: int = 1000, report: dict | None = None,
             trust_extracted: bool = False, trust_classified: bool | None = None) -> list[Claim]:
    """Pipeline entry (agreed interface): KB -> inventory -> claims.

    trust_classified (default: env TZ_TRUST_CLASSIFIED=1) also builds claims on facts of rows the sentence
    classifier labelled true with high confidence (KB.add_classified), not only on practice gold."""
    if trust_classified is None:
        trust_classified = os.environ.get("TZ_TRUST_CLASSIFIED") == "1"
    inv = FactInventory.from_kb(kb, trust_extracted=trust_extracted, trust_classified=trust_classified)
    return generate_from_inventory(inv, input_texts, n, report)
