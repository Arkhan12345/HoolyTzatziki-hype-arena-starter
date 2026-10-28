"""Step 8: hand-off files for teammates -> eda/out/handoff/

  demo_failure_types.csv      one practice row per archetype / failure type, label blanked (for the live demo)
  demo_failure_types_key.csv  id, expected label, why (for the presenter only)
  demo_blank.csv              header only: id,text,date,label (the jury writes its own rows)
  timeline_examples.json      10 rows with the value timeline behind their verdict (for the timeline view)
  claim_reread_examples.csv   generated claims a text re-read would flag (only if an output.csv path is given)

Usage: .venv/Scripts/python eda/08_handoff.py [path/to/output.csv]
"""
import csv
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eda" / "out"
HAND = OUT / "handoff"
HAND.mkdir(parents=True, exist_ok=True)

# archetype -> (failure type shown to the jury, preferred labels in order)
DEMO = [
    ("stale_value", "outdated value (true earlier, moved on since)", ["false"]),
    ("change_point", "value ahead of its time / a real change", ["false", "true"]),
    ("contradicts_kb", "value contradicts what held on that date", ["false"]),
    ("unsupported_new_value", "unbacked value (nothing else confirms it)", ["false"]),
    ("hype_joke_tail", "hype / joke tail", ["unsure"]),
    ("lying_page", "statement on a hype page", ["false"]),
    ("speaker_quote", "quoted speaker", ["false", "true"]),
    ("unverifiable_single_value", "single-value pair: never false, maybe unverifiable", ["unsure", "true"]),
    ("value_outlier", "value far from the usual range", ["false", "true"]),
    ("hedged_or_bounded", "hedged or bounded value (over, about, up to)", ["false", "true"]),
    ("date_reasoning", "date written in the sentence differs from the page date", ["false", "true"]),
    ("multi_fact", "several facts in one sentence", ["false", "true"]),
    ("pronoun_subject", "subject only as a pronoun", ["false", "true"]),
    ("spec_fragment", "spec-sheet fragment without a verb", ["true"]),
    ("plain_kb_match", "plain fact that matches its date", ["true"]),
]
FAKE_STRIKE = ["p-1459", "p-1474", "p-108", "p-1217"]  # same sentence, true early, false after the value moved on


def demo_rows(A):
    picked, key = [], []
    for rid in FAKE_STRIKE:
        r = A.loc[A.id == rid].iloc[0]
        picked.append(r)
        key.append({"id": rid, "expected_label": r.label,
                    "failure_type": "same sentence, different date",
                    "why": f"on {r.date} the value {'still held' if r.label == 'true' else 'had already moved on'}"})
    for arch, shown, prefs in DEMO:
        sub = A[(A.archetype == arch) & ~A.id.isin([p.id for p in picked])]
        for lab in prefs:
            s = sub[sub.label == lab]
            if len(s):
                r = s.assign(n=s.text.str.len()).sort_values(["n", "id"]).iloc[len(s) // 4]  # short, but not the shortest
                picked.append(r)
                key.append({"id": r.id, "expected_label": r.label, "failure_type": shown, "why": r.reason})
                break
    demo = pd.DataFrame([{"id": r.id, "text": r.text, "date": r.date, "label": ""} for r in picked])
    return demo, pd.DataFrame(key)


def steps(g):
    """Consecutive runs of the same true value, by date: [{date, value, pages, trust}]."""
    out = []
    for r in g.sort_values(["date", "num"]).itertuples():
        v = json.loads(r.value)
        if out and out[-1]["value"] == v:
            out[-1]["_pages"].add(int(r.page))
            continue
        out.append({"date": r.date.strftime("%Y-%m-%d"), "value": v, "_pages": {int(r.page)}, "trust": None})
    for s in out:
        s["pages"] = len(s.pop("_pages"))
    return out


def timeline_examples(A, F):
    ids = list(FAKE_STRIKE)
    for arch, lab in [("stale_value", "false"), ("stale_value", "false"), ("change_point", "true"),
                      ("contradicts_kb", "false"), ("unverifiable_single_value", "unsure"), ("plain_kb_match", "true")]:
        s = A[(A.archetype == arch) & (A.label == lab) & ~A.id.isin(ids)]
        s = s[s.id.isin(F.id)].assign(n=s.text.str.len()).sort_values(["n", "id"])
        if len(s):
            ids.append(s.iloc[len(s) // 3].id)
    gold = F[F.label == "true"]
    out = []
    for rid in ids[:10]:
        a = A.loc[A.id == rid].iloc[0]
        f = F[F.id == rid].iloc[0]
        g = gold[(gold.subject == f.subject) & (gold.attribute == f.attribute) & (gold.id != rid)]
        out.append({"row_id": rid, "text": a.text, "date": a.date, "subject": f.subject, "attribute": f.attribute,
                    "kind": f.kind, "unit": f.unit, "sentence_value": json.loads(f.value), "held_values": steps(g),
                    "verdict": a.label, "reason": a.reason})
    return out


def claim_reread(path):
    lex = json.loads((ROOT / "data" / "lexicon.json").read_text(encoding="utf-8"))
    names = {s["name"] for s in lex["subjects"]} | {a for s in lex["subjects"] for a in s.get("aliases", [])}
    names |= {e if isinstance(e, str) else e["name"] for e in lex["events"]}
    names = sorted(names, key=len, reverse=True)
    num = re.compile(r"\d")
    # the refused time words of contracts/claim-forms.md, word for word
    refused = ["ago", "yesterday", "tomorrow", "once", "formerly", "previously", "originally", "initially", "recently",
               "lately", "soon", "eventually", "then", "before", "after", "since", "until", "till", "earlier", "later",
               "used to", "last week", "last month", "last quarter", "last year", "next week", "next month",
               "next quarter", "next year", "in the past", "in the future", "at the time", "at that time"]
    time_w = re.compile(r"\b(" + "|".join(map(re.escape, refused)) + r")\b", re.I)
    neg = re.compile(r"\b(not|no|never)\b|n't\b", re.I)
    flagged = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if not r.get("type"):
            continue
        t = r["text"]
        masked = t
        for n in names:
            masked = re.sub(re.escape(n), "<S>", masked, flags=re.I)
        checks = []
        if r["type"] == "compare" and num.search(masked):
            checks.append("digit outside subject names in a compare (e.g. inside attribute words)")
        if r["type"] in ("fact", "at_event", "relation") and time_w.search(masked):
            checks.append(f"refused time word '{time_w.search(masked).group(0)}' in a {r['type']}")
        if neg.search(masked):
            form = json.loads(r["form"])
            kind = (form.get("value") or form.get("to") or {}).get("kind")
            if kind != "boolean":
                checks.append("negation in a non-yes/no claim")
            else:
                checks.append("yes/no stated with a negation: check the attribute words are present word for word")
        for c in checks:
            flagged.append({"id": r["id"], "type": r["type"], "check": c, "text": t})
    return pd.DataFrame(flagged)


def main():
    A = pd.read_csv(OUT / "practice_annotated.csv", encoding="utf-8")
    F = pd.read_pickle(OUT / "facts.pkl")
    demo, key = demo_rows(A)
    demo.to_csv(HAND / "demo_failure_types.csv", index=False, encoding="utf-8")
    key.to_csv(HAND / "demo_failure_types_key.csv", index=False, encoding="utf-8")
    pd.DataFrame(columns=["id", "text", "date", "label"]).to_csv(HAND / "demo_blank.csv", index=False)
    tl = timeline_examples(A, F)
    (HAND / "timeline_examples.json").write_text(json.dumps(tl, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"demo rows: {len(demo)} | key labels: {key.expected_label.value_counts().to_dict()} | timelines: {len(tl)}")
    if len(sys.argv) > 1:
        fl = claim_reread(sys.argv[1])
        fl.to_csv(HAND / "claim_reread_examples.csv", index=False, encoding="utf-8")
        print(f"claims flagged: {fl.id.nunique()} ({fl.check.str.split(' \\(| in a | ').str[0].value_counts().to_dict() if len(fl) else {}})")


if __name__ == "__main__":
    main()
