"""Step 1 of the hand-written synthetic set: SPECS, built by script from the CV training data's KB only.

A spec fixes everything that decides the label (facts, date, case type, chain) and both labels (practice rule and
closed-world rule). The sentence is written later, by hand, in `written/`; it can never change the label.

Training KB = practice rows with holdout == 0 and fold_row5 != 0 (the rows check.sh hides are excluded too).
Held-out texts (guard for near-copies at assembly): practice holdout pages, check.sh rows, all h300 statements.

One shared pool serves both modes: each mode is sub-sampled to its own label mix in assemble.py, so the
pool holds, per case type, the larger of the two modes' needs.

    python eda/h300/claude_synth/plan.py            # writes specs/batch_NNN.jsonl + briefs/batch_NNN.txt
    python eda/h300/claude_synth/plan.py --top-up          # appends plain specs the first pool ran short of
    python eda/h300/claude_synth/plan.py --top-up-chains   # appends practice-mix chain specs
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
H300 = HERE.parent
ROOT = H300.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(H300))
from app.generate.render import attr_words, value_text  # noqa: E402
import chains  # noqa: E402
import synth  # noqa: E402

SEED = 2910
PER_MODE = 3000                 # rows wanted per mode after sub-sampling
CHAIN_SHARE = 0.17
BATCH = 200
TOP_UP = {"true_current": 260, "unseen_pair": 160, "joke": 40, "stale": 20, "perturbed": 20, "premature": 15,
          "swapped": 15}       # sized from the first assembly: the label quotas each mode could not fill
TOP_UP_CHAINS = 110                      # practice-mix chains, same reason
STYLES = ["headline", "spec_line", "release_note", "press_release", "interview_quote", "faq", "pronoun_followup"]


def training_rows():
    practice = pd.read_csv(ROOT / "data" / "practice.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    splits = pd.read_csv(ROOT / "eda" / "splits.csv", dtype=str)
    keep = splits[(splits.holdout == "0") & (splits.fold_row5 != "0")].id
    return practice[practice.id.isin(set(keep))].to_dict("records")


def pool_quotas():
    """Per case type, the larger of the two modes' needs (non-chain part)."""
    n_plain = round(PER_MODE * (1 - CHAIN_SHARE))
    q = {}
    for case in synth.LABEL["practice"]:
        q[case] = max(round(synth.PRACTICE_MIX.get(case, 0) * n_plain), round(synth.CLOSED_MIX.get(case, 0) * n_plain))
    return q


def brief(spec):
    """One compact line per spec for the writer."""
    parts = [spec["spec_id"], spec["style"]]
    for f in spec["facts"]:
        parts.append(f'{f["subject"]} / {attr_words(f["attribute"])} = {value_text(f["value"]) or f["value"]}')
    if spec["chain"]:
        parts.append(f'CHAIN {spec["chain"]["kind"]} ({spec["chain"]["style"]}; may leave link targets unnamed '
                     f'only if nested)')
    if spec["case_type"] == "joke":
        parts.append("ADD a first-person hype/joke tail (no numbers)")
    return " | ".join(parts)


def plain_specs(kb, quotas, rng, known, start):
    """Plain (non-chain) specs per case quota; `known` holds (facts, date) keys already taken."""
    specs = []
    for case, quota in quotas.items():
        got, budget = 0, quota * 15
        seen = set(known)
        while got < quota and budget > 0:
            budget -= 1
            r = synth.make_case(kb, case, rng)
            if r is None:
                continue
            facts = json.loads(r["form"])
            key = (json.dumps(facts, sort_keys=True), r["date"])
            if key in seen:
                continue
            seen.add(key)
            style = "blog_hype" if case == "joke" else STYLES[(start + len(specs)) % len(STYLES)]
            specs.append({"case_type": case, "date": r["date"], "facts": facts, "chain": None, "style": style,
                          "labels": {"practice": synth.LABEL["practice"][case],
                                     "closed_world": synth.LABEL["closed_world"][case]},
                          "chain_length": 0, "broken_link": 0})
            got += 1
    return specs


def write_batches(specs, first_batch):
    (HERE / "specs").mkdir(exist_ok=True)
    (HERE / "briefs").mkdir(exist_ok=True)
    for b in range(0, len(specs), BATCH):
        n = first_batch + b // BATCH
        chunk = specs[b:b + BATCH]
        (HERE / "specs" / f"batch_{n:03d}.jsonl").write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in chunk) + "\n", encoding="utf-8")
        (HERE / "briefs" / f"batch_{n:03d}.txt").write_text(
            "\n".join(brief(s) for s in chunk) + "\n", encoding="utf-8")
    counts = pd.Series([s["case_type"] for s in specs]).value_counts().to_dict()
    print(f"{len(specs)} specs in {(len(specs) + BATCH - 1) // BATCH} batches from batch {first_batch}; {counts}")


def chain_specs(rows, lex, n_per_mode, nrng, modes=("practice", "closed")):
    """Chain specs drawn per mode with that mode's type mix; each spec carries the labels of both modes."""
    facts = pd.DataFrame([{"id": r["id"], "date": str(r["date"])[:10], "label": r["label"],
                           "subject": f["subject"], "attribute": f["attribute"], "kind": f["value"]["kind"],
                           "value_raw": f["value"].get("value"), "unit": f["value"].get("unit"),
                           "op": f["value"].get("op"), "currency": f["value"].get("currency"),
                           "per": f["value"].get("per")}
                          for r in rows for f in json.loads(r.get("form") or "[]")])
    graph = chains.build_link_graph(facts, lex, {r["id"] for r in rows})
    specs = []
    for mode in modes:
        df = chains.chain_cases(graph, n_per_mode, nrng, mode, lex, style_mix=0.5,
                                length_mix={1: 0.5, 2: 0.35, 3: 0.15})
        for c in df.to_dict("records"):
            hops = json.loads(c["form"])
            if any(value_text(h["value"]) is None and h["value"].get("kind") != "boolean" for h in hops):
                continue
            ct = c["case_type"]
            specs.append({"case_type": "chain_" + ct, "date": c["date"], "facts": hops,
                          "chain": {"kind": ct, "style": c["style"], "path": [h["subject"] for h in hops]},
                          "style": "chain_" + c["style"],
                          "labels": {"practice": chains.LABELS["practice"][ct],
                                     "closed_world": chains.LABELS["closed"][ct]},
                          "chain_length": int(c["chain_length"]),
                          "broken_link": 0 if ct == "valid" else 1})
    return specs


def existing_specs():
    return [json.loads(line) for p in sorted((HERE / "specs").glob("batch_*.jsonl"))
            for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_specs(specs, old):
    for i, s in enumerate(specs):
        s["spec_id"] = f"cs-{len(old) + i + 1:05d}"
    write_batches(specs, (len(old) + BATCH - 1) // BATCH)


def top_up(kb):
    """Extra plain specs, deduplicated against the existing pool, numbered after it."""
    old = existing_specs()
    seen = {(json.dumps(s["facts"], sort_keys=True), s["date"]) for s in old if not s["chain"]}
    specs = plain_specs(kb, TOP_UP, random.Random(SEED + 1), seen, len(old))
    random.Random(SEED + 1).shuffle(specs)
    append_specs(specs, old)


def top_up_chains(rows, lex):
    """Extra practice-mix chains (the practice file leaves unknown ends out and fell under a 15% chain share)."""
    old = existing_specs()
    seen = {(json.dumps(s["facts"], sort_keys=True), s["date"]) for s in old if s["chain"]}
    specs = []
    for s in chain_specs(rows, lex, TOP_UP_CHAINS, np.random.default_rng(SEED + 2), modes=("practice",)):
        key = (json.dumps(s["facts"], sort_keys=True), s["date"])
        if key not in seen and s["case_type"] != "chain_unknown_end":
            seen.add(key)
            specs.append(s)
    append_specs(specs, old)


def main():
    rng = random.Random(SEED)
    lex = json.loads((ROOT / "data" / "lexicon.json").read_text(encoding="utf-8"))
    rows = training_rows()
    kb = synth.FoldKB(rows, lex)
    if "--top-up" in sys.argv:
        top_up(kb)
        return
    if "--top-up-chains" in sys.argv:
        top_up_chains(rows, lex)
        return
    specs = plain_specs(kb, pool_quotas(), rng, set(), 0)
    # chains: half drawn with the practice-rule type mix, half with the closed-world one
    specs += chain_specs(rows, lex, round(PER_MODE * CHAIN_SHARE) // 2 + 40, np.random.default_rng(SEED))
    rng.shuffle(specs)
    for i, s in enumerate(specs):
        s["spec_id"] = f"cs-{i + 1:05d}"
    write_batches(specs, 0)


if __name__ == "__main__":
    main()
