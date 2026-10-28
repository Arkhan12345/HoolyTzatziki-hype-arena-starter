"""Step 12: retrain LightGBM with a synthetic labelled dataset (Ioan's ~10k rows), evaluated on real rows only.

Input: any CSV in the practice shape (id, text, date, label, form). Pages = contiguous rows that share a date.

Two workspaces keep the universes apart (synthetic values never enter the real KB / timelines and vice versa):
  A  eda/out/h300_work          practice + hooly_statements_300 (from eda/10_h300_validation.py, cached)
  B  eda/out/synth_work/<hash>  the synthetic rows alone
In B, the synthetic pages are split 80/20 (train / synthetic validation); the train part gets every
label-derived feature out-of-fold (inner page split), exactly like real training rows. In A, each real fold's
LightGBM trains on the real training fold + the synthetic train rows with sample weight w_synth, picked on the
real inner validation folds. The extractor's attribute picker learns from real + synthetic `form` (train-time
only: the arena input has no `form`); real test rows always use extracted facts.

Reported every run (eda/out/synth_results.csv), each with all-true and the train accuracy:
  real fold_group CV (train/validation), real holdout once (test), hooly_statements_300, synthetic validation
  (marked `synthetic`, not a real-world number).
Checks before training (eda/out/synth_checks.json): label shares, the chain table vs eda/out/fact_chains.csv,
value parser round-trip, copies of holdout / h300 sentences (a copy stops the run unless --smoke).

Saved (no retrain next time): eda/out/models/synth_lgbm.joblib + synth_lgbm_params.json; --load scores the
holdout and the 300 statements with it.

Run:  .venv/bin/python eda/12_train_synth.py --synth data/synth/<file>.csv
      .venv/bin/python eda/12_train_synth.py --synth data/hooly_statements_300.csv --smoke   (pipeline test only)
      .venv/bin/python eda/12_train_synth.py --load
"""
import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eda" / "out"
MODEL_FILE = OUT / "models" / "synth_lgbm.joblib"
PARAMS_FILE = OUT / "models" / "synth_lgbm_params.json"
W_GRID = [0.0, 0.25, 0.5, 1.0]
VAL_SHARE = 0.2
SEED = 0


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def file_hash(*paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------ workspace B (synthetic universe)
def fast_features(work):
    """features.pkl with only what improve/base.label_free_table reads (02_rules' slow descriptive KB skipped)."""
    sys.path.insert(0, str(work / "eda"))
    R2 = _load("rules02", work / "eda" / "02_rules.py")
    from features import text_flags
    R = pd.read_pickle(work / "eda" / "out" / "rows.pkl")
    R = pd.concat([R, text_flags(R)], axis=1)
    R["speaker"] = R.text.map(R2.speaker_of)
    genre = R.groupby("page").apply(R2.page_genre, include_groups=False).rename("page_genre")
    R = R.merge(genre, left_on="page", right_index=True)
    R["page_first_person_share"] = R.groupby("page")["first_person"].transform("mean")
    R["page_hype_share"] = R.groupby("page")["hype_score"].transform(lambda s: (s > 0).mean())
    R.to_pickle(work / "eda" / "out" / "features.pkl")


def prepare_synth(synth_csv):
    H = _load("h300", ROOT / "eda" / "10_h300_validation.py")
    hsh = file_hash(synth_csv, ROOT / "eda" / "01_parse.py", ROOT / "eda" / "features.py")
    work = OUT / "synth_work" / hsh
    H.copy_code(work)
    S = pd.read_csv(synth_csv)
    need = {"id", "text", "date", "label", "form"}
    assert need <= set(S.columns), f"synthetic CSV lacks {need - set(S.columns)}"
    S["orig_id"] = S["id"]
    S["id"] = [f"p-{i + 1}" for i in range(len(S))]
    S[["id", "orig_id"]].to_csv(work / "idmap.csv", index=False)
    S.drop(columns="orig_id")[["id", "text", "date", "label", "form"]].to_csv(work / "data" / "practice.csv",
                                                                             index=False)
    if not (work / "eda" / "out" / "features.pkl").exists():
        subprocess.run([sys.executable, "eda/01_parse.py"], cwd=work, check=True, stdout=subprocess.DEVNULL)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--fast-features", str(work)], check=True)
    return work


def synth_worker(out_pkl):
    """cwd = workspace B. Synthetic feature tables (train out-of-fold, validation) + data checks."""
    from sklearn.model_selection import GroupKFold
    sys.path.insert(0, str(Path.cwd() / "eda"))
    W = _load("wv", Path.cwd() / "eda" / "09_word_values.py")
    C = _load("chains", Path.cwd() / "eda" / "11_fact_chains.py")
    import extract_simple as ex
    from app.kb.normalize import key as vkey
    I7, R, Y = W.I7, W.R, W.Y
    pages = np.array(sorted(R.page.unique()))
    # split by page; a file with < 10 pages (e.g. one date for every row) is split by row instead
    group = R.set_index("id")["page"] if len(pages) >= 10 else pd.Series(range(len(R)), index=R["id"])
    groups = np.array(sorted(group.unique()))
    rng = np.random.default_rng(SEED)
    val_groups = set(rng.choice(groups, max(1, int(len(groups) * VAL_SHARE)), replace=False))
    tr = sorted(group.index[~group.isin(val_groups)])
    va = sorted(group.index[group.isin(val_groups)])
    inner = [[tr[j] for j in a] for a, _ in GroupKFold(5).split(tr, groups=[group[i] for i in tr])]
    X, _ = I7.fold_table(I7.F_ALL, set(tr), set(va), inner)

    # checks
    lexicon = json.loads(Path("data/lexicon.json").read_text(encoding="utf-8"))
    labels = R.set_index("id")["label"]
    chains = C.describe(C.chain_features(I7.F_ALL, R, lexicon), labels)
    ref = pd.read_csv(Path(sys.argv[3]))
    cmp = chains.merge(ref, on=["chain", "bucket"], suffixes=("_synth", "_practice"))
    cmp["false_in_practice_ci"] = [
        float(lo) <= fs <= float(hi) for fs, (lo, hi) in
        zip(cmp.false_synth, cmp.false_ci_practice.str.strip("[]").str.split(","))]
    parsed, total = 0, 0
    for r in R.itertuples():
        keys = set()
        for _, _, v in ex.parse_values(r.text):
            try:
                keys.add(vkey(v))
            except Exception:
                pass
        for f in json.loads(r.form):
            if f["value"]["kind"] in ("date", "money", "quantity", "percent", "duration", "version"):
                total += 1
                try:
                    parsed += vkey(f["value"]) in keys
                except Exception:
                    pass
    checks = {"rows": len(R), "pages": int(len(pages)), "label_share": labels.value_counts(normalize=True)
              .round(4).to_dict(), "parser_roundtrip": round(parsed / max(total, 1), 4),
              "chain_false_share_in_practice_ci": round(float(cmp.false_in_practice_ci.mean()), 3),
              "chain_table": cmp[["chain", "bucket", "n_synth", "false_synth", "false_practice", "false_ci_practice",
                                  "false_in_practice_ci"]].to_dict(orient="records")}
    S = pd.read_csv("data/practice.csv")
    pd.to_pickle({"X_train": X.loc[tr], "y_train": Y[tr].to_numpy(), "X_val": X.loc[va], "y_val": Y[va].to_numpy(),
                  "rows": S, "gold": ex.gold_facts(S), "checks": checks}, out_pkl)


# ------------------------------------------------------------------ copies of evaluation sentences
def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).split()


def overlap_check(synth_csv):
    S = pd.read_csv(synth_csv)
    P = pd.read_csv(ROOT / "data" / "practice.csv")
    sp = pd.read_csv(ROOT / "eda" / "splits.csv").set_index("id")
    hold = P[P.id.map(sp["holdout"]) == 1]
    H = pd.read_csv(ROOT / "data" / "hooly_statements_300.csv")
    grams = lambda w, n=6: {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}  # noqa: E731
    res = {}
    for name, ref in [("holdout", hold.text), ("h300", H.text)]:
        exact = set(" ".join(norm(t)) for t in ref)
        refg = set().union(*[grams(norm(t)) for t in ref])
        copies = sum(" ".join(norm(t)) in exact for t in S.text)
        near = sum(bool(g) and len(g & refg) / len(g) >= 0.5 for g in (grams(norm(t)) for t in S.text))
        res[name] = {"exact_copies": int(copies), "near_copies_6gram50": int(near)}
    return res


# ------------------------------------------------------------------ workspace A (real rows): train + evaluate
def facts_from_extraction(E, R, W):
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


def real_worker(synth_pkl, results_csv, smoke):
    import joblib
    from sklearn.model_selection import GroupKFold
    sys.path.insert(0, str(Path.cwd() / "eda"))
    W = _load("wv", Path.cwd() / "eda" / "09_word_values.py")
    import extract_simple as ex
    I7, R, Y = W.I7, W.R, W.Y
    SY = pd.read_pickle(synth_pkl)
    lex = json.loads(Path("data/lexicon.json").read_text(encoding="utf-8"))
    gaz = ex.Gazetteer(lex)
    P = pd.read_csv("data/practice.csv").merge(R[["id", "page"]], on="id")
    num = R.set_index("id")["num"]
    h300 = set(num.index[num >= 100000])
    cv_ids = sorted(W.SPLITS.index[W.SPLITS.holdout == 0])
    hold = set(W.SPLITS.index[W.SPLITS.holdout == 1])

    def picker_for(train_ids, with_synth):
        Rtr = P[P.id.isin(train_ids)]
        docs = [ex.training_docs(Rtr, ex.gold_facts(Rtr), gaz)]
        if with_synth:
            docs.append(ex.training_docs(SY["rows"], SY["gold"], gaz))
        return ex.AttributePicker(gaz).fit(pd.concat(docs, ignore_index=True))

    def extract(picker, test_ids):
        Rte = P[P.id.isin(test_ids)].sort_values("id", key=lambda s: s.str[2:].astype(int))
        E = pd.DataFrame(ex.extract_rows(Rte, gaz, picker))
        f1 = ex.score(E, ex.gold_facts(Rte))[5]
        return facts_from_extraction(E, R, W), f1

    def synth_block(cols_real):
        Xs = SY["X_train"].reindex(columns=cols_real)
        dummies = [c for c in cols_real if c.startswith(("genre_", "kind_", "rel_"))]
        Xs[dummies] = Xs[dummies].fillna(0)
        return Xs, SY["y_train"]

    def fit(variant, X, tr, w, cols):
        Xt, yt, sw = X.loc[tr, cols], Y[tr].to_numpy(), np.ones(len(tr))
        if w > 0:
            Xs, ys = synth_block(cols)
            Xt, yt = pd.concat([Xt, Xs]), np.concatenate([yt, ys])
            sw = np.concatenate([sw, np.full(len(ys), w)])
        return W.lgbm().fit(Xt, yt, sample_weight=sw)

    def predict(variant, m, X, ids, cols):
        Z = X.loc[ids] if isinstance(ids, list) else X
        P_ = I7.proba3(m, Z[cols])
        return I7.hard_rules(P_, Z) if variant.startswith("L4") else P_

    def tune_w(variant, X, tr, inner, cols):
        best, best_w = -1, 0.0
        for w in W_GRID:
            accs = []
            for a in inner:
                a = sorted(a)
                b = sorted(set(tr) - set(a))
                m = fit(variant, X, a, w, cols)
                accs.append((predict(variant, m, X, b, cols).argmax(1) == Y[b].to_numpy()).mean())
            if np.mean(accs) > best + 1e-9:
                best, best_w = np.mean(accs), w
        return best_w

    rows, store, fold_rows = [], {}, []
    variants = ["L0_base", "L0+synth", "L4_rules", "L4+synth"]
    f1s = {"real": [], "real+synth": []}
    for k, (train, test, inner) in enumerate(I7.scheme_folds("fold_group", cv_ids)):
        for pk in ["real", "real+synth"]:
            E, f1 = extract(picker_for(train, pk == "real+synth"), test)
            f1s[pk].append(f1)
        F = pd.concat([I7.F_ALL[I7.F_ALL.id.isin(train)], E], ignore_index=True)  # E: real+synth picker
        te = sorted(test & set(F.id))
        X, _ = I7.fold_table(F, train, set(te), inner)
        tr = sorted(train)
        for v in variants:
            cols = I7.cols_for("L0_base" if v.startswith("L0") else "L4_rules", X)
            w = tune_w(v, X, tr, inner, cols) if v.endswith("+synth") else 0.0
            m = fit(v, X, tr, w, cols)
            Pte = predict(v, m, X, te, cols)
            acc_tr = (predict(v, m, X, tr, cols).argmax(1) == Y[tr].to_numpy()).mean()
            store.setdefault(v, ([], []))
            store[v][0].extend(te)
            store[v][1].append(Pte)
            fold_rows.append({"fold": k, "variant": v, "w_synth": w, "train_acc": acc_tr,
                              "val_acc": (Pte.argmax(1) == Y[te].to_numpy()).mean()})
            if v.endswith("+synth"):
                Xsv = SY["X_val"].reindex(columns=cols)
                acc_sv = (predict(v, m, Xsv, None, cols).argmax(1) == SY["y_val"]).mean()
                fold_rows[-1]["synthetic_val_acc"] = acc_sv
        print(f"  fold {k} done", flush=True)
    FR = pd.DataFrame(fold_rows)
    for v, (ids, Ps) in store.items():
        met = W.metrics(Y[ids].to_numpy(), np.vstack(Ps))
        fr = FR[FR.variant == v]
        rows.append({"split": "real CV fold_group (extracted)", "variant": v, "n": len(ids),
                     "acc": round(float(met["acc"]), 4), "macro_f1_tf": round(float(met["macro_f1_tf"]), 4),
                     "train_acc": round(fr.train_acc.mean(), 4), "val_acc_std": round(fr.val_acc.std(), 4),
                     "w_synth": round(fr.w_synth.mean(), 3)})
        if "synthetic_val_acc" in fr and fr.synthetic_val_acc.notna().any():
            rows.append({"split": "synthetic validation (not real)", "variant": v, "n": len(SY["y_val"]),
                         "acc": round(fr.synthetic_val_acc.mean(), 4)})
    ids = store["L0_base"][0]
    rows.append({"split": "real CV fold_group (extracted)", "variant": "all_true", "n": len(ids),
                 "acc": round(float((Y[ids] == 1).mean()), 4)})
    rows.append({"split": "synthetic validation (not real)", "variant": "all_true", "n": len(SY["y_val"]),
                 "acc": round(float((SY["y_val"] == 1).mean()), 4)})
    cvres = pd.DataFrame(rows)
    best = cvres[cvres.split.str.startswith("real CV") & (cvres.variant != "all_true")] \
        .sort_values(["acc", "macro_f1_tf"], ascending=False).variant.iloc[0]
    print(f"best on real CV: {best}", flush=True)

    # holdout (once) + the 300 statements: trained on every non-holdout practice row (+ synthetic)
    train = set(cv_ids)
    trl = sorted(train)
    inner = [[trl[j] for j in a] for a, _ in GroupKFold(5).split(trl, groups=[I7.PAGE_OF[i] for i in trl])]
    picker = picker_for(train, True)
    E, f1_hold = extract(picker, hold | h300)
    F = pd.concat([I7.F_ALL[I7.F_ALL.id.isin(train)], E], ignore_index=True)
    test = (hold | h300) & set(F.id)
    X, _ = I7.fold_table(F, train, test, inner)
    saved = None
    for v in sorted({"L0_base", best}):
        cols = I7.cols_for("L0_base" if v.startswith("L0") else "L4_rules", X)
        w = tune_w(v, X, trl, inner, cols) if v.endswith("+synth") else 0.0
        m = fit(v, X, trl, w, cols)
        acc_tr = (predict(v, m, X, trl, cols).argmax(1) == Y[trl].to_numpy()).mean()
        for name, ids in [("real holdout once (extracted)", sorted(hold & test)),
                          ("hooly_statements_300 (extracted)", sorted(h300 & test))]:
            Pr = predict(v, m, X, ids, cols)
            y = Y[ids].to_numpy()
            pred = Pr.argmax(1)
            rows.append({"split": name, "variant": v, "n": len(ids), "acc": round(float((pred == y).mean()), 4),
                         "macro_f1_tf": round(float(W.metrics(y, Pr)["macro_f1_tf"]), 4),
                         "train_acc": round(float(acc_tr), 4), "w_synth": w,
                         "recall_false": round(float((pred[y == 0] == 0).mean()), 4) if (y == 0).any() else None,
                         "recall_unsure": round(float((pred[y == 2] == 2).mean()), 4) if (y == 2).any() else None})
        if v == best:
            saved = {"lgbm": m, "variant": v, "w_synth": w, "feature_columns": cols, "picker": picker,
                     "classes": list(W.CLASSES)}
    for name, ids in [("real holdout once (extracted)", sorted(hold & test)),
                      ("hooly_statements_300 (extracted)", sorted(h300 & test))]:
        rows.append({"split": name, "variant": "all_true", "n": len(ids), "acc": round(float((Y[ids] == 1).mean()), 4)})
    res = pd.DataFrame(rows)
    res.insert(0, "smoke_test", smoke)
    res.to_csv(results_csv, index=False)
    extra = {"extraction_f1_cv": {k: round(float(np.mean(v)), 4) for k, v in f1s.items()},
             "extraction_f1_holdout_h300": round(float(f1_hold), 4), "best": best,
             "per_fold": FR.round(4).to_dict(orient="records")}
    joblib.dump(saved, Path(results_csv).with_suffix(".joblib"), compress=3)
    Path(results_csv).with_suffix(".json").write_text(json.dumps(extra, indent=1, default=str), encoding="utf-8")


def load_worker(model_file):
    """cwd = workspace A. Score the holdout and the 300 statements with the saved model, no fitting."""
    import joblib
    from sklearn.model_selection import GroupKFold
    sys.path.insert(0, str(Path.cwd() / "eda"))
    W = _load("wv", Path.cwd() / "eda" / "09_word_values.py")
    import extract_simple as ex
    I7, R, Y = W.I7, W.R, W.Y
    saved = joblib.load(model_file)
    P = pd.read_csv("data/practice.csv").merge(R[["id", "page"]], on="id")
    num = R.set_index("id")["num"]
    h300 = set(num.index[num >= 100000])
    cv_ids = sorted(W.SPLITS.index[W.SPLITS.holdout == 0])
    hold = set(W.SPLITS.index[W.SPLITS.holdout == 1])
    Rte = P[P.id.isin(hold | h300)].sort_values("id", key=lambda s: s.str[2:].astype(int))
    gaz = ex.Gazetteer(json.loads(Path("data/lexicon.json").read_text(encoding="utf-8")))
    E = facts_from_extraction(pd.DataFrame(ex.extract_rows(Rte, gaz, saved["picker"])), R, W)
    F = pd.concat([I7.F_ALL[I7.F_ALL.id.isin(cv_ids)], E], ignore_index=True)
    test = (hold | h300) & set(F.id)
    inner = [[cv_ids[j] for j in a] for a, _ in GroupKFold(5).split(cv_ids, groups=[I7.PAGE_OF[i] for i in cv_ids])]
    X, _ = I7.fold_table(F, set(cv_ids), test, inner)
    for name, ids in [("holdout", sorted(hold & test)), ("hooly_statements_300", sorted(h300 & test))]:
        Pr = I7.proba3(saved["lgbm"], X.loc[ids, saved["feature_columns"]])
        if saved["variant"].startswith("L4"):
            Pr = I7.hard_rules(Pr, X.loc[ids])
        print(f"loaded {saved['variant']} (w_synth {saved['w_synth']}): {name} acc "
              f"{(Pr.argmax(1) == Y[ids].to_numpy()).mean():.4f}, no retraining")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", help="synthetic CSV (id,text,date,label,form)")
    ap.add_argument("--smoke", action="store_true", help="pipeline test: copies allowed, results marked smoke")
    ap.add_argument("--load", action="store_true", help="score holdout + 300 statements with the saved model")
    ap.add_argument("--fast-features", help=argparse.SUPPRESS)
    ap.add_argument("--synth-worker", nargs=2, help=argparse.SUPPRESS)
    ap.add_argument("--real-worker", nargs=2, help=argparse.SUPPRESS)
    ap.add_argument("--load-worker", help=argparse.SUPPRESS)
    a = ap.parse_args()
    me = str(Path(__file__).resolve())
    if a.fast_features:
        return fast_features(Path(a.fast_features))
    if a.synth_worker:
        return synth_worker(a.synth_worker[0])
    if a.real_worker:
        return real_worker(a.real_worker[0], a.real_worker[1], a.smoke)
    if a.load_worker:
        return load_worker(a.load_worker)
    H = _load("h300", ROOT / "eda" / "10_h300_validation.py")
    if a.load:
        H.prepare()
        subprocess.run([sys.executable, me, "--load-worker", str(MODEL_FILE)], cwd=H.WORK, check=True)
        return
    assert a.synth, "--synth PATH is required"
    synth = Path(a.synth).resolve()
    ov = overlap_check(synth)
    print("copies of evaluation sentences:", ov, flush=True)
    if not a.smoke and any(v["exact_copies"] for v in ov.values()):
        sys.exit("synthetic data copies holdout / hooly_statements_300 sentences: remove them (or --smoke)")
    work = prepare_synth(synth)
    pkl = work / "synth_features.pkl"
    subprocess.run([sys.executable, me, "--synth-worker", str(pkl), str(OUT / "fact_chains.csv")], cwd=work,
                   check=True)
    checks = pd.read_pickle(pkl)["checks"]
    checks["overlap"] = ov
    tag = "_smoke" if a.smoke else ""
    (OUT / f"synth_checks{tag}.json").write_text(json.dumps(checks, indent=1, default=str), encoding="utf-8")
    print(f"synthetic: {checks['rows']} rows, {checks['pages']} pages, labels {checks['label_share']}, parser "
          f"round-trip {checks['parser_roundtrip']}, chain buckets inside practice CI "
          f"{checks['chain_false_share_in_practice_ci']}", flush=True)
    H.prepare()
    res_csv = OUT / f"synth_results{tag}.csv"
    cmd = [sys.executable, me, "--real-worker", str(pkl), str(res_csv)] + (["--smoke"] if a.smoke else [])
    subprocess.run(cmd, cwd=H.WORK, check=True)
    res = pd.read_csv(res_csv)
    extra = json.loads(res_csv.with_suffix(".json").read_text(encoding="utf-8"))
    if not a.smoke:  # a smoke test never replaces the saved model
        MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        res_csv.with_suffix(".joblib").replace(MODEL_FILE)
        PARAMS_FILE.write_text(json.dumps({"best": extra["best"], "synth_csv": str(synth.relative_to(ROOT))
                                           if synth.is_relative_to(ROOT) else str(synth),
                                           "synth_hash": file_hash(synth),
                                           "practice_hash": file_hash(ROOT / "data" / "practice.csv"),
                                           "w_grid": W_GRID, "lgbm": "eda/improve/base.lgbm (frozen params)",
                                           "extraction": extra, "checks": checks,
                                           "results": res.to_dict(orient="records")},
                                          indent=1, default=str), encoding="utf-8")
        print(f"saved {MODEL_FILE.relative_to(ROOT)} and {PARAMS_FILE.relative_to(ROOT)}")
    pd.set_option("display.width", 250)
    print(("\n*** SMOKE TEST: pipeline check only, not results ***\n" if a.smoke else "\n") + res.to_string(index=False))
    print("extraction F1 (real CV):", extra["extraction_f1_cv"])


if __name__ == "__main__":
    main()
