"""Step 11: chains of facts, and how they shift the true / false / unsure ratio.

Four chains, all label-free (built from facts and dates only, so they can be model features):
  value chain     (subject, attribute) mentions ordered by date: how many distinct values the pair goes through,
                  where this fact's value sits in that chain, whether a later mention moves on to another value
                  (superseded), whether the value comes back after a change (flip-back A -> B -> A)
  entity chain    a name value that is itself a lexicon subject links two subjects (Hooly Cloud.reliability_lead
                  = Holdan -> Holdan's facts); in-links of the subject; size of its linked component
  coreference     the subject is not named in the sentence (pronoun), so it chains to an earlier sentence
  row chain       several facts in one sentence (one wrong fact makes the row false)

Part 1 (descriptive, practice labels): label shares per chain bucket with Wilson 95% CI, in
  eda/out/fact_chains.csv.
Part 2 (model): the chain features added to L0 / L4 LightGBM, fold_group CV (train/validation), holdout once
  (test), then the 300 statements (eda/10_h300_validation.py workspace). Test rows use extracted facts
  (realistic), training rows gold facts; results in eda/out/fact_chains_model.csv.

Run:  .venv/bin/python eda/11_fact_chains.py
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eda"))
CH = ["ch_chain_len", "ch_pos_rel", "ch_is_first", "ch_is_last", "ch_superseded", "ch_flip_back",
      "ch_same_before", "ch_links_subject", "ch_in_links", "ch_component", "ch_pronoun", "ch_n_facts"]


def chain_features(F, R, lexicon):
    """Per-row chain features from a fact table (gold or extracted); label-free."""
    F = F.copy()
    F["date"] = pd.to_datetime(F["date"])
    subjects = {s["name"] for s in lexicon["subjects"]}
    out = []
    for (_s, _a), g in F.sort_values(["date", "num"]).groupby(["subject", "attribute"], sort=False):
        vals = list(g["vkey"])
        runs = [v for i, v in enumerate(vals) if i == 0 or v != vals[i - 1]]
        order = list(dict.fromkeys(vals))  # distinct values in first-seen order
        dates = g["date"].to_numpy()
        for j, r in enumerate(g.itertuples()):
            later = vals[j + 1:]
            later_dates = dates[j + 1:]
            k = order.index(r.vkey)
            out.append({"id": r.id, "ch_chain_len": len(order), "ch_pos_rel": k / max(len(order) - 1, 1),
                        "ch_is_first": float(k == 0), "ch_is_last": float(r.vkey == vals[-1]),
                        "ch_superseded": float(any(v != r.vkey for v, d in zip(later, later_dates) if d > r.date)),
                        "ch_flip_back": float(runs.count(r.vkey) > 1),
                        "ch_same_before": float(sum(v == r.vkey for v in vals[:j]))})
    C = pd.DataFrame(out)
    # entity chain: name values that are subjects
    F["target"] = [json.loads(v) if k == "name" else None for v, k in zip(F["value"], F["kind"])]
    links = F[F["target"].isin(subjects) & (F["kind"] == "name")]
    in_links = links.groupby("target").size()
    parent = {s: s for s in set(F["subject"]) | set(links["target"])}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in zip(links["subject"], links["target"]):
        parent[find(a)] = find(b)
    comp = pd.Series([find(s) for s in parent], index=list(parent)).value_counts()
    root = {s: find(s) for s in parent}
    E = pd.DataFrame({"id": F["id"],
                      "ch_links_subject": F["target"].isin(subjects).astype(float),
                      "ch_in_links": F["subject"].map(in_links).fillna(0).to_numpy(),
                      "ch_component": F["subject"].map(lambda s: comp[root[s]]).to_numpy(),
                      "ch_pronoun": (~F["subject_mentioned"].astype(bool)).astype(float).to_numpy()
                      if "subject_mentioned" in F else 0.0})
    agg = pd.concat([C.groupby("id").agg({"ch_chain_len": "max", "ch_pos_rel": "min", "ch_is_first": "min",
                                          "ch_is_last": "min", "ch_superseded": "max", "ch_flip_back": "max",
                                          "ch_same_before": "min"}),
                     E.groupby("id").agg({"ch_links_subject": "max", "ch_in_links": "max", "ch_component": "max",
                                          "ch_pronoun": "max"})], axis=1)
    agg["ch_n_facts"] = F.groupby("id").size()
    return agg.reindex(R["id"])


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def describe(Ch, labels):
    buckets = {
        "values in the pair's chain": pd.cut(Ch.ch_chain_len, [0, 1, 2, 3, 99], labels=["1", "2", "3", "4+"]),
        "position in chain": np.select([Ch.ch_is_last == 1, Ch.ch_is_first == 1], ["latest value", "first value"],
                                       "middle value"),
        "superseded by a later mention": Ch.ch_superseded.map({0: "no", 1: "yes"}),
        "value comes back (A->B->A)": Ch.ch_flip_back.map({0: "no", 1: "yes"}),
        "same value said before": pd.cut(Ch.ch_same_before, [-1, 0, 2, 999], labels=["0", "1-2", "3+"]),
        "fact points to another subject": Ch.ch_links_subject.map({0: "no", 1: "yes"}),
        "subject in-links": pd.cut(Ch.ch_in_links, [-1, 0, 3, 999], labels=["0", "1-3", "4+"]),
        "pronoun subject (coref chain)": Ch.ch_pronoun.map({0: "no", 1: "yes"}),
        "facts in the sentence": pd.cut(Ch.ch_n_facts, [0, 1, 2, 99], labels=["1", "2", "3+"]),
    }
    base = labels.value_counts(normalize=True)
    rows = []
    for name, b in buckets.items():
        b = pd.Series(np.asarray(b, dtype=object), index=labels.index)
        for v in sorted(b.dropna().unique(), key=str):
            m = b == v
            n = int(m.sum())
            rec = {"chain": name, "bucket": v, "n": n}
            for c in ["true", "false", "unsure"]:
                k = int((labels[m] == c).sum())
                lo, hi = wilson(k, n)
                rec[f"{c}"] = round(k / n, 3)
                rec[f"{c}_ci"] = f"[{lo:.2f},{hi:.2f}]"
                rec[f"{c}_lift"] = round(k / n / base[c], 2)
            rows.append(rec)
    return pd.DataFrame(rows)


def model_eval(I7, W, R, Y, splits, chains_for, lexicon):
    """L0/L4 with and without chain features; returns result rows."""
    from sklearn.model_selection import GroupKFold
    rows = []

    def run(name, train, test, inner, F_test):
        F = pd.concat([I7.F_ALL[I7.F_ALL.id.isin(train)], F_test[F_test.id.isin(test)]], ignore_index=True)
        test = test & set(F.id)
        X, _ = I7.fold_table(F, train, test, inner)
        X = pd.concat([X, chains_for(F).reindex(X.index)], axis=1)
        tr, te = sorted(train), sorted(test)
        res = {}
        for v in ["L0_base", "L0+chains", "L4_rules", "L4+chains"]:
            c = I7.cols_for("L0_base" if v.startswith("L0") else "L4_rules", X)
            c = c + CH if v.endswith("+chains") else c
            m = W.lgbm().fit(X.loc[tr, c], Y[tr].to_numpy())
            P = I7.proba3(m, X.loc[te, c])
            if v.startswith("L4"):
                P = I7.hard_rules(P, X.loc[te])
            res[v] = (te, P)
        return res

    return run


def main_practice():
    spec = importlib.util.spec_from_file_location("wv", ROOT / "eda" / "09_word_values.py")
    W = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(W)
    I7, R, Y = W.I7, W.R, W.Y
    lexicon = json.loads((ROOT / "data" / "lexicon.json").read_text(encoding="utf-8"))
    labels = R.set_index("id")["label"]

    Ch = chain_features(I7.F_ALL, R, lexicon)
    D = describe(Ch, labels)
    D.to_csv(W.OUT / "fact_chains.csv", index=False)
    pd.set_option("display.width", 250)
    print(D[["chain", "bucket", "n", "true", "false", "false_ci", "false_lift", "unsure", "unsure_lift"]]
          .to_string(index=False))

    run = model_eval(I7, W, R, Y, W.SPLITS, lambda F: chain_features(F, R, lexicon), lexicon)
    E = W.extracted_facts()
    cv_ids = sorted(W.SPLITS.index[W.SPLITS.holdout == 0])
    hold = set(W.SPLITS.index[W.SPLITS.holdout == 1])
    rows = []
    store = {}
    for train, test, inner in I7.scheme_folds("fold_group", cv_ids):
        for v, (te, P) in run("cv", train, test, inner, E).items():
            store.setdefault(v, ([], []))
            store[v][0].extend(te)
            store[v][1].append(P)
    for v, (ids, Ps) in store.items():
        met = W.metrics(Y[ids].to_numpy(), np.vstack(Ps))
        rows.append({"split": "fold_group_cv (extracted)", "variant": v, "n": len(ids),
                     **{k: round(float(x), 4) for k, x in met.items()}})
    rows.append({"split": "fold_group_cv (extracted)", "variant": "all_true", "n": len(store["L0_base"][0]),
                 "acc": round(float((Y[store["L0_base"][0]] == 1).mean()), 4)})
    from sklearn.model_selection import GroupKFold
    trl = sorted(cv_ids)
    inner = [[trl[j] for j in a] for a, _ in GroupKFold(5).split(trl, groups=[I7.PAGE_OF[i] for i in trl])]
    for v, (te, P) in run("holdout", set(cv_ids), hold, inner, E).items():
        met = W.metrics(Y[te].to_numpy(), P)
        rows.append({"split": "holdout_once (extracted)", "variant": v, "n": len(te),
                     **{k: round(float(x), 4) for k, x in met.items()}})
    rows.append({"split": "holdout_once (extracted)", "variant": "all_true", "n": len(te),
                 "acc": round(float((Y[te] == 1).mean()), 4)})
    return pd.DataFrame(rows)


def main_h300():
    """Runs inside eda/out/h300_work (cwd): train on practice, test on the 300 statements."""
    spec = importlib.util.spec_from_file_location("wv", Path.cwd() / "eda" / "09_word_values.py")
    W = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(W)
    spec10 = importlib.util.spec_from_file_location("h3", Path.cwd() / "eda" / "10_h300_validation.py")
    H = importlib.util.module_from_spec(spec10)
    spec10.loader.exec_module(H)
    from sklearn.model_selection import GroupKFold
    I7, R, Y = W.I7, W.R, W.Y
    lexicon = json.loads(Path("data/lexicon.json").read_text(encoding="utf-8"))
    num = R.set_index("id")["num"]
    train, test = set(num.index[num < H.OFFSET]), set(num.index[num >= H.OFFSET])
    trl = sorted(train)
    inner = [[trl[j] for j in a] for a, _ in GroupKFold(5).split(trl, groups=[I7.PAGE_OF[i] for i in trl])]
    run = model_eval(I7, W, R, Y, None, lambda F: chain_features(F, R, lexicon), lexicon)
    rows = []
    for facts_name, facts in [("gold", I7.F_ALL), ("extracted", H.extract_h300(W, R, train, test))]:
        for v, (te, P) in run("h300", train, test, inner, facts).items():
            y = Y[te].to_numpy()
            pred = P.argmax(1)
            rows.append({"split": f"h300 ({facts_name})", "variant": v, "n": len(te),
                         "acc": round(float((pred == y).mean()), 4),
                         "recall_false": round(float((pred[y == 0] == 0).mean()), 4),
                         "recall_true": round(float((pred[y == 1] == 1).mean()), 4),
                         "recall_unsure": round(float((pred[y == 2] == 2).mean()), 4)})
    pd.DataFrame(rows).to_csv(sys.argv[2], index=False)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--h300-worker":
        main_h300()
        return
    res = main_practice()
    spec10 = importlib.util.spec_from_file_location("h3", ROOT / "eda" / "10_h300_validation.py")
    H = importlib.util.module_from_spec(spec10)
    spec10.loader.exec_module(H)
    H.prepare()
    tmp = H.WORK / "chains_h300.csv"
    subprocess.run([sys.executable, "eda/11_fact_chains.py", "--h300-worker", str(tmp)], cwd=H.WORK, check=True)
    res = pd.concat([res, pd.read_csv(tmp)], ignore_index=True)
    res.to_csv(ROOT / "eda" / "out" / "fact_chains_model.csv", index=False)
    cols = ["split", "variant", "n", "acc", "macro_f1_tf", "false_precision", "pred_false", "recall_false",
            "recall_true", "recall_unsure"]
    print("\n" + res[[c for c in cols if c in res.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
