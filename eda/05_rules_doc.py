"""Step 5: put every rule into data/RULES.md.

The hand-written sections above the AUTO marker stay as they are; everything between the markers is
regenerated from eda/out: all tested rules, the decision-tree combination rules as IF/THEN paths,
the archetype -> option rules, and the relations (attributes, subject co-occurrence, templates, speakers).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eda" / "out"
DOC = ROOT / "data" / "RULES.md"
START, END = "<!-- AUTO:START (eda/05_rules_doc.py) -->", "<!-- AUTO:END -->"


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v).replace("|", "/") for v in r) + " |")
    return "\n".join(lines)


def tree_rules(X, labels, target):
    """Fit the same depth-4 tree as 02_rules.py and list its leaves as IF/THEN with raw counts."""
    y = (labels == target).astype(int).values
    clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=8, class_weight="balanced", random_state=0).fit(X, y)
    t = clf.tree_
    names = list(X.columns)
    leaf_of = clf.apply(X)
    out = []

    def walk(node, conds):
        if t.children_left[node] == -1:
            m = leaf_of == node
            n = int(m.sum())
            lab = labels[m]
            out.append({"IF": " AND ".join(conds) or "(any row)",
                        "rows": n,
                        "true": int((lab == "true").sum()), "false": int((lab == "false").sum()),
                        "unsure": int((lab == "unsure").sum()),
                        f"P({target})": round(float((lab == target).mean()), 3) if n else np.nan})
            return
        f = names[t.feature[node]]
        walk(t.children_left[node], conds + [f"not {f}"])
        walk(t.children_right[node], conds + [f])

    walk(0, [])
    df = pd.DataFrame(out).sort_values(f"P({target})", ascending=False)
    df.insert(1, "THEN", np.where(df[f"P({target})"] >= 0.5, f"**{target}**",
                                  np.where(df[f"P({target})"] >= 2 * (labels == target).mean(),
                                           f"leans {target}", f"not {target}")))
    return df


def main():
    rules = pd.read_csv(OUT / "rules.csv")
    flags = pd.read_pickle(OUT / "flags.pkl")
    labels = flags.pop("label")
    flags = flags.drop(columns="id")
    arch = pd.read_csv(OUT / "archetypes.csv")
    attrs = pd.read_csv(OUT / "attributes.csv")
    co = pd.read_csv(OUT / "subject_cooccurrence.csv")
    tpl = pd.read_csv(OUT / "templates.csv")
    spk = pd.read_csv(OUT / "speakers.csv")
    from importlib import import_module
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    A = import_module("03_recommend").ARCHETYPES

    parts = [START, "",
             "## 7. All tested rules (every rule from `eda/02_rules.py`)",
             "`confirmed` = at least 5 rows and non-true labels on at least 2 pages. "
             "Lift = P(false | rule) / 0.146. Sorted by family, then false rate.", ""]
    r = rules.sort_values(["family", "p_false"], ascending=[True, False])
    parts.append(md_table(r[["family", "rule", "support", "pages", "true", "false", "unsure", "p_false",
                             "p_unsure", "false_ci95", "lift_false", "confirmed"]]))

    parts += ["", "## 8. Combination rules (decision-tree leaves as IF/THEN)",
              "Depth-4 trees on the binary flags (balanced classes). Every leaf is one rule; the counts are raw rows. "
              "`not X` means the flag is off. Flags: `rel_*` = the fact's relation to the value in force on its date, "
              "`never_repeated` = no other row has this value, `pair_single_value` = the pair has one value in the "
              "whole corpus, `disagrees_prev_mention` = differs from the corpus' previous mention.", "",
              "### 8a. What makes a row **unsure**", "", md_table(tree_rules(flags, labels, "unsure")), "",
              "### 8b. What makes a row **false**", "", md_table(tree_rules(flags, labels, "false"))]

    parts += ["", "## 9. Archetype rules → model option (first match wins, top to bottom)", ""]
    a = pd.DataFrame([{"order": i + 1, "archetype": x[0], "IF": x[5], "option": x[1], "fallback": x[2],
                       "algorithms": x[3]} for i, x in enumerate(A)])
    a = a.merge(arch[["archetype", "n", "p_false", "p_unsure"]], on="archetype", how="left")
    parts.append(md_table(a[["order", "archetype", "IF", "n", "p_false", "p_unsure", "option", "fallback",
                             "algorithms"]]))

    parts += ["", "## 10. Relations in the data", "",
              "### 10a. Attribute → value kind (top 40 by rows)",
              "Each attribute always takes the same value kind; `n_values` = distinct values over the year "
              "(a high count means it changes often, so the date matters).", "",
              md_table(attrs.head(40)), "",
              "### 10b. Subjects that appear together on the same pages (top 30)", "",
              md_table(co.head(30)), "",
              "### 10c. Repeated sentence templates (Faker-style, seen ≥ 3 times)",
              "`<S>` subject, `<N>` number, `<M>` month, `<C>` other capitalised name.", "",
              md_table(tpl[tpl.n >= 3]), "",
              "### 10d. Speakers / labels in front of a colon (≥ 3 rows)",
              "Includes spec labels such as `Recruiter:` as well as real people.", "",
              md_table(spk[spk.n >= 3]), "", END]

    auto = "\n".join(parts)
    doc = DOC.read_text(encoding="utf-8")
    if START in doc:
        doc = doc[:doc.index(START)] + auto + doc[doc.index(END) + len(END):]
    else:
        doc = doc.rstrip() + "\n\n" + auto + "\n"
    DOC.write_text(doc, encoding="utf-8")
    print(f"wrote {DOC} ({len(doc.splitlines())} lines)")


if __name__ == "__main__":
    main()
