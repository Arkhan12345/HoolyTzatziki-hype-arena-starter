"""Step 1: parse practice.csv + lexicon.json into a row table and a fact table.

Outputs (eda/out/):
  rows.pkl   one row per sentence, with page ids and lexicon flags
  facts.pkl  one row per fact from the gold `form`
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eda" / "out"
OUT.mkdir(parents=True, exist_ok=True)


def load_lexicon():
    lex = json.loads((ROOT / "data" / "lexicon.json").read_text(encoding="utf-8"))
    subjects = {}
    for s in lex["subjects"]:
        subjects[s["name"]] = {"aliases": s.get("aliases", []), "attributes": set(s.get("attributes", []))}
    for e in lex.get("events", []):
        name = e["name"] if isinstance(e, dict) else e
        attrs = set(e.get("attributes", [])) if isinstance(e, dict) else set()
        aliases = e.get("aliases", []) if isinstance(e, dict) else []
        subjects.setdefault(name, {"aliases": aliases, "attributes": attrs, "event": True})
    return lex, subjects


def main():
    rows = pd.read_csv(ROOT / "data" / "practice.csv", encoding="utf-8")
    rows["num"] = rows["id"].str.slice(2).astype(int)
    rows = rows.sort_values("num").reset_index(drop=True)
    rows["date"] = pd.to_datetime(rows["date"])

    # page = contiguous block of ids that share a date
    rows["page"] = (rows["date"] != rows["date"].shift()).cumsum()
    rows["page_size"] = rows.groupby("page")["id"].transform("size")
    rows["pos_in_page"] = rows.groupby("page").cumcount()
    rows["pos_rel"] = rows["pos_in_page"] / (rows["page_size"] - 1).clip(lower=1)

    lex, subjects = load_lexicon()
    names_by_subject = {k: [k] + v["aliases"] for k, v in subjects.items()}

    facts = []
    for r in rows.itertuples():
        form = json.loads(r.form) if isinstance(r.form, str) and r.form else []
        for j, f in enumerate(form):
            v = f["value"]
            subj = f["subject"]
            info = subjects.get(subj)
            mentioned = any(n.lower() in r.text.lower() for n in names_by_subject.get(subj, [subj]))
            # short name e.g. "Gemini" for "Hooly Gemini"
            short = subj.replace("Hooly ", "")
            mentioned_short = short.lower() in r.text.lower()
            facts.append({
                "id": r.id, "num": r.num, "date": r.date, "page": r.page, "label": r.label,
                "fact_idx": j, "subject": subj, "attribute": f["attribute"],
                "kind": v.get("kind"), "value": json.dumps(v.get("value")), "value_raw": v.get("value"),
                "unit": v.get("unit"), "op": v.get("op"), "currency": v.get("currency"), "per": v.get("per"),
                "subject_in_lexicon": info is not None,
                "attr_in_lexicon": bool(info) and f["attribute"] in info["attributes"],
                "subject_mentioned": mentioned or mentioned_short,
            })
    facts = pd.DataFrame(facts)
    nform = rows["form"].map(lambda s: len(json.loads(s)) if isinstance(s, str) and s else 0)
    rows["n_facts"] = nform
    assert len(facts) == nform.sum(), "fact count mismatch"

    agg = facts.groupby("id").agg(
        all_subj_in_lex=("subject_in_lexicon", "all"),
        all_attr_in_lex=("attr_in_lexicon", "all"),
        any_pronoun_subject=("subject_mentioned", lambda s: (~s).any()),
        kinds=("kind", lambda s: "|".join(sorted(set(s)))),
        subjects=("subject", lambda s: "|".join(sorted(set(s)))),
        attributes=("attribute", lambda s: "|".join(sorted(set(s)))),
    )
    rows = rows.merge(agg, left_on="id", right_index=True, how="left")

    rows.to_pickle(OUT / "rows.pkl")
    facts.to_pickle(OUT / "facts.pkl")
    print(f"rows={len(rows)} facts={len(facts)} pages={rows.page.nunique()} dates={rows.date.nunique()}")
    print("labels:", rows.label.value_counts().to_dict())
    print("rows with 0 facts:", (rows.n_facts == 0).sum())
    print("facts subject not in lexicon:", (~facts.subject_in_lexicon).sum(),
          "attr not in lexicon:", (~facts.attr_in_lexicon).sum(),
          "pronoun/implicit subject:", (~facts.subject_mentioned).sum())


if __name__ == "__main__":
    main()
