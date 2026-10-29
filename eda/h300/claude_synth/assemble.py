"""Step 3 of the hand-written synthetic set: validate every written sentence against its spec, then assemble.

A sentence is accepted only if
  - its subject (the chain's head for chains) is named, by name or alias, in the sentence;
  - every spec value is stated so the value parser reads it back (eda/extract_simple.parse_values, compared with
    app/kb/normalize.key); name values must appear by name; booleans are stated in words (not parsed);
    in a nested chain an intermediate link entity may stay unnamed;
  - it states no OTHER value (after masking lexicon names, which may contain numbers like "Pixel 10");
  - it is not a near-copy (rapidfuzz ratio >= 90) of any practice or h300 sentence, nor a duplicate.
The label always comes from the spec, never from the sentence.

Each mode is then sub-sampled to its own label mix (practice rule ~83/15/2, closed-world 1/3 each), with the
chain share kept at ~17%.

    python eda/h300/claude_synth/assemble.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

HERE = Path(__file__).resolve().parent
H300 = HERE.parent
ROOT = H300.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eda"))
sys.path.insert(0, str(H300))
from app.kb.normalize import key as vkey  # noqa: E402
import extract_simple as ex  # noqa: E402
import synth  # noqa: E402

OUT = H300 / "out"
COLUMNS = ["id", "text", "date", "label", "form", "case_type", "chain_length", "broken_link", "fold", "mode",
           "source"]
SEED = 2911


def load_jsonl(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def check(spec, text, gaz):
    """-> reason string if rejected, else None."""
    t = text.strip()
    if not t:
        return "empty"
    low = t.lower()
    names = {k: v for k, v in gaz.lower.items()}
    head = spec["facts"][0]["subject"]
    mentioned = {e for _, _, e in gaz.mentions(t)}
    if head not in mentioned and head.lower() not in low:
        return "subject not named"
    nested = bool(spec["chain"]) and spec["chain"]["style"] == "nested"
    link_entities = {f["subject"] for f in spec["facts"][1:]} if spec["chain"] else set()
    parsed = ex.parse_values(t)
    want = []
    for f in spec["facts"]:
        v = f["value"]
        k = v.get("kind")
        if k == "boolean":
            continue
        if k == "name":
            nm = str(v.get("value", ""))
            target = names.get(nm.lower(), nm)
            if nested and target in link_entities:
                continue
            if nm.lower() not in low and target not in mentioned:
                return f"name value missing: {nm}"
            continue
        want.append(vkey(v))
    got = [vkey(v) for _, _, v in parsed]
    for w in want:
        if w not in got:
            return "value not parsed back"
    # extras: parse again with lexicon names masked, so numbers inside names don't count
    masked = t
    for a, b, _ in sorted(gaz.mentions(t), key=lambda m: -m[0]):
        masked = masked[:a] + "Acme" + masked[b:]
    extra = [vkey(v) for _, _, v in ex.parse_values(masked) if vkey(v) not in want]
    if extra:
        return "extra value"
    return None


def main():
    lex = json.loads((ROOT / "data" / "lexicon.json").read_text(encoding="utf-8"))
    gaz = ex.Gazetteer(lex)
    known = {}                                  # subject or alias (lower) -> attributes it may carry
    for sub in lex["subjects"]:
        for n in [sub["name"]] + sub.get("aliases", []):
            known[n.lower()] = set(sub.get("attributes", []))
    for e in lex.get("events", []):
        known.setdefault((e["name"] if isinstance(e, dict) else e).lower(),
                         set(e.get("attributes", [])) if isinstance(e, dict) else set())
    held = pd.read_csv(ROOT / "data" / "practice.csv", encoding="utf-8-sig", dtype=str).text.tolist()
    held += pd.read_csv(ROOT / "data" / "hooly_statements_300.csv", encoding="utf-8-sig", dtype=str).text.tolist()
    held = [h.strip() for h in held if isinstance(h, str)]

    specs = {}
    for p in sorted((HERE / "specs").glob("batch_*.jsonl")):
        for s in load_jsonl(p):
            specs[s["spec_id"]] = s
    written = {}
    for p in sorted((HERE / "written").glob("batch_*.jsonl")):
        for w in load_jsonl(p):
            written[w["spec_id"]] = w["text"]

    accepted, reasons, seen = [], Counter(), set()
    by_style, by_case = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for sid, text in written.items():
        s = specs.get(sid)
        if s is None:
            reasons["unknown spec"] += 1
            continue
        why = check(s, text, gaz)
        if why is None and any(str(f["subject"]).lower() not in known
                               or (known[str(f["subject"]).lower()] and f["attribute"] not in known[str(f["subject"]).lower()])
                               for f in s["facts"]):
            why = "fact outside the lexicon"
        t = text.strip()
        if why is None and t.lower() in seen:
            why = "duplicate"
        if why is None:
            best = process.extractOne(t, held, scorer=fuzz.ratio)
            if best and round(best[1]) >= 90:     # rounded like the uint8 scores of the harsh checks
                why = "near-copy"
        by_style[s["style"]][1] += 1
        by_case[s["case_type"]][1] += 1
        if why:
            reasons[why] += 1
            continue
        seen.add(t.lower())
        by_style[s["style"]][0] += 1
        by_case[s["case_type"]][0] += 1
        accepted.append((s, t))

    rng = random.Random(SEED)
    stats = {"written": len(written), "accepted": len(accepted), "rejections": dict(reasons),
             "by_style": {k: f"{a}/{n}" for k, (a, n) in sorted(by_style.items())},
             "by_case": {k: f"{a}/{n}" for k, (a, n) in sorted(by_case.items())}, "modes": {}}
    for mode, mix in (("practice", synth.PRACTICE_MIX), ("closed_world", synth.CLOSED_MIX)):
        plain = [x for x in accepted if not x[0]["chain"]]
        # practice rule: every broken link is false, so an unknown chain end (unsure, broken_link 1) stays out
        chain = [x for x in accepted if x[0]["chain"]
                 and not (mode == "practice" and x[0]["case_type"] == "chain_unknown_end")]
        by = defaultdict(list)
        for x in plain:
            by[x[0]["case_type"]].append(x)
        # label quotas are exact; inside a label, case types follow the mix weights until a pool runs dry
        # (update is capped by the training KB) and the remainder goes to the other cases of that label
        lab_w = defaultdict(float)
        for c, w in mix.items():
            lab_w[synth.LABEL[mode][c]] += w
        lab_n = {lab: sum(len(by[c]) for c in mix if synth.LABEL[mode][c] == lab) for lab in lab_w}
        n_plain = min(int(lab_n[lab] / w) for lab, w in lab_w.items() if w > 0)
        rows = []
        for lab, lw in lab_w.items():
            cases = [c for c in mix if synth.LABEL[mode][c] == lab]
            want, take = round(lw * n_plain), {c: 0 for c in cases}
            while sum(take.values()) < want:
                open_ = [c for c in cases if take[c] < len(by[c])]
                left = want - sum(take.values())
                wsum = sum(mix[c] for c in open_)
                for c in open_:
                    take[c] = min(len(by[c]), take[c] + max(1, round(left * mix[c] / wsum)))
                    if sum(take.values()) >= want:
                        break
            for c in cases:
                pick = by[c][:]
                rng.shuffle(pick)
                rows += pick[:take[c]]
        n_chain = min(len(chain), round(len(rows) * 0.17 / 0.83))
        pick = chain[:]
        rng.shuffle(pick)
        rows += pick[:n_chain]
        rng.shuffle(rows)
        out = []
        for i, (s, t) in enumerate(rows):
            out.append({"id": f"cs10k-{mode[:2]}-{i + 1}", "text": t, "date": s["date"],
                        "label": s["labels"][mode], "form": json.dumps(s["facts"], ensure_ascii=False),
                        "case_type": s["case_type"], "chain_length": s["chain_length"],
                        "broken_link": s["broken_link"], "fold": "all",
                        "mode": "practice" if mode == "practice" else "closed_world", "source": "claude"})
        df = pd.DataFrame(out, columns=COLUMNS)
        name = "synth10k_claude_practice_rule.csv" if mode == "practice" else "synth10k_claude_closed_world.csv"
        df.to_csv(OUT / name, index=False, encoding="utf-8")
        stats["modes"][mode] = {"rows": len(df), "mix": df.label.value_counts(normalize=True).round(3).to_dict(),
                                "cases": df.case_type.value_counts().to_dict(),
                                "chain_share": round((df.chain_length.astype(int) > 0).mean(), 3)}
    (HERE / "assemble_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
