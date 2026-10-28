"""Step 10: external validation on data/hooly_statements_300.csv (run after every model change).

Train on all 1,897 practice rows, test on the 300 generated statements (100 true / 100 false / 100 unsure, all
dated 2026-10-28). Their labels are never used for features or fitting. Each statement is its own single-row
page, after every practice row. Scored twice: gold `form` facts (oracle) and facts extracted from the text by
eda/extract_simple.py (the arena's input has no `form`). Also scores the saved model of eda/09_word_values.py.

The 300 rows need their own parsed tables, so the script builds a workspace (eda/out/h300_work/: a copy of eda/
and app/ with data/practice.csv = practice + the 300) and runs 01_parse + 02_rules there. 02_rules takes ~10 min
on 591 pages, so the tables are cached by a hash of the inputs.

Run:  .venv/bin/python eda/10_h300_validation.py        (eda/09_word_values.py calls it after training)
Out:  eda/out/h300_validation.csv, eda/out/h300_predictions.csv
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "eda" / "out" / "h300_work"
H300 = ROOT / "data" / "hooly_statements_300.csv"
OFFSET = 100000  # h300 ids become p-100000.. so they sort after every practice row
VARIANTS = ["words_only", "L0_base", "L0+words", "L4_rules", "L4+words"]


def inputs_hash():
    h = hashlib.sha256()
    for p in [ROOT / "data" / "practice.csv", H300, ROOT / "data" / "lexicon.json", ROOT / "eda" / "01_parse.py",
              ROOT / "eda" / "02_rules.py", ROOT / "eda" / "features.py"]:
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def copy_code(work):
    """A workspace = a copy of eda/ and app/ whose data/practice.csv the caller replaces."""
    (work / "eda").mkdir(parents=True, exist_ok=True)
    (work / "data").mkdir(exist_ok=True)
    for p in (ROOT / "eda").glob("*.py"):
        shutil.copy(p, work / "eda" / p.name)
    for d in ["eda/improve", "app"]:
        shutil.copytree(ROOT / d, work / d, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(ROOT / "data" / "lexicon.json", work / "data" / "lexicon.json")
    shutil.copy(ROOT / "eda" / "splits.csv", work / "eda" / "splits.csv")


def prepare():
    """Copy code into the workspace; rebuild the parsed tables only when the inputs changed."""
    copy_code(WORK)
    p = pd.read_csv(ROOT / "data" / "practice.csv")
    h = pd.read_csv(H300)
    h["orig_id"] = h["id"]
    h["id"] = [f"p-{OFFSET + i}" for i in range(len(h))]
    h[["id", "orig_id"]].to_csv(WORK / "h300_idmap.csv", index=False)
    pd.concat([p, h.drop(columns="orig_id")]).to_csv(WORK / "data" / "practice.csv", index=False)
    # every h300 statement is its own page
    f = WORK / "eda" / "01_parse.py"
    s = f.read_text(encoding="utf-8")
    old = 'rows["page"] = (rows["date"] != rows["date"].shift()).cumsum()'
    assert old in s, "01_parse.py page rule changed: update eda/10_h300_validation.py"
    f.write_text(s.replace(old, f'rows["page"] = ((rows["date"] != rows["date"].shift()) | (rows["num"] >= {OFFSET})).cumsum()'),
                 encoding="utf-8")
    stamp = WORK / "tables.hash"
    hsh = inputs_hash()
    need = ["rows.pkl", "facts.pkl", "features.pkl"]
    if stamp.exists() and stamp.read_text() == hsh and all((WORK / "eda" / "out" / n).exists() for n in need):
        print("h300 tables cached", flush=True)
        return
    print("building h300 tables (01_parse + 02_rules, ~10 min)...", flush=True)
    for step in ["01_parse.py", "02_rules.py"]:
        subprocess.run([sys.executable, f"eda/{step}"], cwd=WORK, check=True, stdout=subprocess.DEVNULL)
    stamp.write_text(hsh)


def extract_h300(W, R, train, test):
    """Facts read from the 300 statements' text; the attribute picker learns from the practice rows' gold facts."""
    import extract_simple as ex
    lex = json.loads(Path("data/lexicon.json").read_text(encoding="utf-8"))
    gaz = ex.Gazetteer(lex)
    P = pd.read_csv("data/practice.csv").merge(R[["id", "page"]], on="id")
    Ptr, Pte = P[P.id.isin(train)], P[P.id.isin(test)]
    picker = ex.AttributePicker(gaz).fit(ex.training_docs(Ptr, ex.gold_facts(Ptr), gaz))
    E = pd.DataFrame(ex.extract_rows(Pte.sort_values("id", key=lambda s: s.str[2:].astype(int)), gaz, picker))
    _, ne, ng, pr, rc, f1 = ex.score(E, ex.gold_facts(Pte))
    print(f"h300 extraction: gold {ng} extracted {ne} P {pr:.3f} R {rc:.3f} F1 {f1:.3f}", flush=True)
    E = E.drop(columns=[c for c in ["page"] if c in E.columns]).merge(
        R.set_index("id")[["num", "date", "page", "label"]], left_on="id", right_index=True)
    E["fact_idx"] = E.groupby("id").cumcount()
    E["value_raw"] = E["value"].map(lambda v: v.get("value"))
    for c in ["unit", "op", "currency", "per"]:
        E[c] = E["value"].map(lambda v, c=c: v.get(c))
    E["kind"] = E["value"].map(lambda v: v["kind"])
    E["value"] = E["value"].map(lambda v: json.dumps(v.get("value")))
    E["vkey"] = [W.fact_key(r) for r in E.itertuples()]

    return E


def worker(model_file, out_csv, pred_csv):
    """Runs inside the workspace (cwd = WORK)."""
    import importlib.util

    import joblib
    import numpy as np
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import GroupKFold

    sys.path.insert(0, str(Path.cwd() / "eda"))
    spec = importlib.util.spec_from_file_location("wv", Path.cwd() / "eda" / "09_word_values.py")
    W = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(W)
    I7, R, Y = W.I7, W.R, W.Y
    num = R.set_index("id")["num"]
    train, test = set(num.index[num < OFFSET]), set(num.index[num >= OFFSET])
    trl = sorted(train)
    inner = [[trl[j] for j in a] for a, _ in GroupKFold(5).split(trl, groups=[I7.PAGE_OF[i] for i in trl])]
    idmap = pd.read_csv("h300_idmap.csv").set_index("id")["orig_id"]

    E = extract_h300(W, R, train, test)

    saved = joblib.load(model_file) if Path(model_file).exists() else None
    rows, preds = [], []

    def record(facts_name, v, te, P):
        y = Y[te].to_numpy()
        pred = P.argmax(1)
        m = W.metrics(y, P)
        rec = {c: float((pred[y == i] == i).mean()) if (y == i).any() else float("nan")
               for i, c in enumerate(W.CLASSES)}
        rows.append({"facts": facts_name, "variant": v, "n": len(te), "acc": round(float(m["acc"]), 4),
                     "macro_f1_tf": round(float(m["macro_f1_tf"]), 4), "recall_false": round(rec["false"], 4),
                     "recall_true": round(rec["true"], 4), "recall_unsure": round(rec["unsure"], 4),
                     "pred_false": m["pred_false"], "pred_unsure": m["pred_unsure"]})
        preds.append(pd.DataFrame({"facts": facts_name, "variant": v, "id": [idmap[i] for i in te],
                                   "gold": W.CLASSES[y], "pred": W.CLASSES[pred]}))
        return confusion_matrix(y, pred, labels=[0, 1, 2])

    for facts_name, facts in [("gold", I7.F_ALL), ("extracted", E)]:
        X, te_set, _, _ = W.build(train, test, inner, facts)
        te = sorted(te_set)
        for v in VARIANTS:
            P, _, _ = W.score_variant(v, X, trl, te)
            cm = record(facts_name, v, te, P)
            if v == "L4_rules":
                print(f"{facts_name} L4_rules confusion (rows gold false/true/unsure, cols pred):\n{cm}", flush=True)
        if saved is not None:
            Xs = X.copy()
            Xs.loc[te, W.WV] = W.predict_words(saved["vectorizer"], saved["word_model"], te)
            P = I7.proba3(saved["lgbm"], Xs.loc[te, saved["feature_columns"]])
            if saved["variant"].startswith("L4"):
                P = I7.hard_rules(P, Xs.loc[te])
            record(facts_name, f"saved:{saved['variant']}", te, P)
        y = Y[te].to_numpy()
        rows.append({"facts": facts_name, "variant": "all_true", "n": len(te), "acc": round(float((y == 1).mean()), 4)})
    D = pd.DataFrame(rows)
    D.to_csv(out_csv, index=False)
    pd.concat(preds).to_csv(pred_csv, index=False)
    pd.set_option("display.width", 220)
    print(D.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--model", default=str(ROOT / "eda" / "out" / "models" / "word_values.joblib"))
    ap.add_argument("--out", default=str(ROOT / "eda" / "out" / "h300_validation.csv"))
    ap.add_argument("--pred", default=str(ROOT / "eda" / "out" / "h300_predictions.csv"))
    a = ap.parse_args()
    if a.worker:
        worker(a.model, a.out, a.pred)
        return
    prepare()
    subprocess.run([sys.executable, "eda/10_h300_validation.py", "--worker", "--model", a.model, "--out", a.out,
                    "--pred", a.pred], cwd=WORK, check=True)


if __name__ == "__main__":
    main()
