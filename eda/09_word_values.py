"""Step 9: learned word values as extra LightGBM features ("every word gets a value"), leak-free.

Word values: TF-IDF over the row text (word 1-2-grams + char 2-5-grams) -> L2 logistic regression, so every
n-gram gets a learned weight per class. Its probabilities become 3 features (wv_false, wv_true, wv_unsure):
  training rows  out-of-fold through an inner GroupKFold by page (validation),
  test rows      from the model fitted on the whole training fold.
Splits (eda/splits.csv): fold_group = 5-fold train/validation CV by page; holdout = test, scored once.
Facts: gold `form` for the labelled (training) rows only; test rows are scored twice, with gold facts (oracle,
upper bound) and with eda/out/facts_extracted.pkl (what the arena allows: the input has no `form`).

The model scored on the holdout (trained on every non-holdout row) is saved, so a later run does not retrain:
  eda/out/models/word_values.joblib       vectorizer, word model, LightGBM, feature columns, variant
  eda/out/models/word_values_params.json  params, scores, data hash, git commit

Run:   .venv/bin/python eda/09_word_values.py            (trains unless the saved model matches the data)
       .venv/bin/python eda/09_word_values.py --retrain  (forces training)
       .venv/bin/python eda/09_word_values.py --load     (loads the saved model and scores the holdout, no fit)
Out:   eda/out/improve_word_values.csv; then runs eda/10_h300_validation.py (the 300 statements) every time it trains
"""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eda"))
warnings.filterwarnings("ignore")
_spec = importlib.util.spec_from_file_location("improve07", ROOT / "eda" / "07_improve.py")
I7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(I7)
from improve.base import CLASSES, OUT, fact_key, lgbm, metrics  # noqa: E402

MODEL_DIR = OUT / "models"
MODEL_FILE = MODEL_DIR / "word_values.joblib"
PARAMS_FILE = MODEL_DIR / "word_values_params.json"
WV = ["wv_false", "wv_true", "wv_unsure"]
WORD_C = 2.0
VARIANTS = ["words_only", "L0_base", "L0+words", "L4_rules", "L4+words"]
R, Y, SPLITS = I7.R, I7.Y, I7.SPLITS
TEXT = R.set_index("id")["text"]


def word_model():
    vec = make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                     TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True))
    return vec, LogisticRegression(C=WORD_C, max_iter=3000)


def fit_words(ids):
    vec, lr = word_model()
    lr.fit(vec.fit_transform(TEXT[ids]), Y[ids].to_numpy())
    return vec, lr


def predict_words(vec, lr, ids):
    p = np.zeros((len(ids), 3))
    p[:, lr.classes_] = lr.predict_proba(vec.transform(TEXT[ids]))
    return p


def word_features(train, test, inner):
    """wv_* for the training rows out-of-fold (inner page split) and for the test rows."""
    tr, te = sorted(train), sorted(test)
    out = pd.DataFrame(index=tr + te, columns=WV, dtype=float)
    for inner_train in inner:
        inner_test = sorted(train - set(inner_train))
        assert not set(inner_test) & set(inner_train)
        vec, lr = fit_words(sorted(inner_train))
        out.loc[inner_test, WV] = predict_words(vec, lr, inner_test)
    vec, lr = fit_words(tr)
    out.loc[te, WV] = predict_words(vec, lr, te)
    return out, vec, lr


def cols(variant, X):
    base = "L0_base" if variant.startswith("L0") else "L4_rules"
    c = I7.cols_for(base, X)
    return c + WV if variant.endswith("+words") else c


def score_variant(variant, X, tr, te, model=None):
    """Returns (proba on te, train accuracy, fitted model)."""
    if variant == "words_only":
        P = X.loc[te, WV].to_numpy(dtype=float)
        return P, float("nan"), None
    c = cols(variant, X)
    m = model or lgbm().fit(X.loc[tr, c], Y[tr].to_numpy())
    P = I7.proba3(m, X.loc[te, c])
    Ptr = I7.proba3(m, X.loc[tr, c])
    if variant.startswith("L4"):
        P, Ptr = I7.hard_rules(P, X.loc[te]), I7.hard_rules(Ptr, X.loc[tr])
    return P, float((Ptr.argmax(1) == Y[tr].to_numpy()).mean()), m


def extracted_facts():
    E = pd.read_pickle(OUT / "facts_extracted.pkl")
    if "vkey" not in E.columns:
        E["vkey"] = [fact_key(r) for r in E.itertuples()]
    return E


def build(train, test, inner, facts):
    """Feature table: gold facts for training rows; test rows' facts from `facts` (gold or extracted)."""
    assert not train & test, "leak: a test row is in the training set"
    F = pd.concat([I7.F_ALL[I7.F_ALL.id.isin(train)], facts[facts.id.isin(test)]], ignore_index=True)
    test = test & set(F.id)
    X, _ = I7.fold_table(F, train, test, inner)
    W, vec, lr = word_features(train, test, inner)
    return pd.concat([X, W.reindex(X.index)], axis=1), test, vec, lr


def data_hash():
    h = hashlib.sha256()
    for p in [ROOT / "data" / "practice.csv", ROOT / "eda" / "splits.csv"]:
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def evaluate():
    cv_ids = sorted(SPLITS.index[SPLITS.holdout == 0])
    hold = set(SPLITS.index[SPLITS.holdout == 1])
    E = extracted_facts()
    rows, per_fold = [], []
    for facts_name, facts in [("gold", I7.F_ALL), ("extracted", E)]:
        store = {v: ([], []) for v in VARIANTS}
        for k, (train, test, inner) in enumerate(I7.scheme_folds("fold_group", cv_ids)):
            X, te_set, _, _ = build(train, test, inner, facts)
            tr, te = sorted(train), sorted(te_set)
            yte = Y[te].to_numpy()
            for v in VARIANTS:
                P, acc_tr, _ = score_variant(v, X, tr, te)
                store[v][0].extend(te)
                store[v][1].append(P)
                per_fold.append({"facts": facts_name, "fold": k, "variant": v, "train_acc": acc_tr,
                                 "val_acc": float((P.argmax(1) == yte).mean()),
                                 "all_true": float((yte == 1).mean())})
            print(f"  {facts_name} fold {k} done", flush=True)
        for v, (ids, Ps) in store.items():
            met = metrics(Y[ids].to_numpy(), np.vstack(Ps))
            rows.append({"facts": facts_name, "split": "fold_group_cv", "variant": v, "n": len(ids),
                         **{a: round(float(b), 4) for a, b in met.items()}})
        rows.append({"facts": facts_name, "split": "fold_group_cv", "variant": "all_true", "n": len(cv_ids),
                     "acc": round(float((Y[cv_ids] == 1).mean()), 4)})
    PF = pd.DataFrame(per_fold)

    # pick the variant on CV with extracted facts (the realistic setting); holdout scored once afterwards
    cvx = pd.DataFrame(rows).query("facts == 'extracted' and variant != 'all_true' and variant != 'words_only'")
    best = cvx.sort_values(["acc", "macro_f1_tf"], ascending=False).variant.iloc[0]
    print(f"best on CV (extracted facts): {best}", flush=True)

    train = set(cv_ids)
    trl = sorted(train)
    from sklearn.model_selection import GroupKFold
    inner = [[trl[j] for j in a] for a, _ in GroupKFold(5).split(trl, groups=[I7.PAGE_OF[i] for i in trl])]
    saved = None
    for facts_name, facts in [("gold", I7.F_ALL), ("extracted", E)]:
        X, te_set, vec, lr = build(train, hold, inner, facts)
        te = sorted(te_set)
        for v in ["L0_base", best]:
            P, acc_tr, m = score_variant(v, X, trl, te)
            met = metrics(Y[te].to_numpy(), P)
            rows.append({"facts": facts_name, "split": "holdout_once", "variant": v, "n": len(te),
                         "train_acc": round(acc_tr, 4), **{a: round(float(b), 4) for a, b in met.items()}})
            if facts_name == "extracted" and v == best:
                saved = {"vectorizer": vec, "word_model": lr, "lgbm": m, "variant": best,
                         "feature_columns": cols(best, X), "classes": list(CLASSES), "word_C": WORD_C,
                         "holdout_acc_extracted": float(met["acc"])}
        rows.append({"facts": facts_name, "split": "holdout_once", "variant": "all_true", "n": len(te),
                     "acc": round(float((Y[te] == 1).mean()), 4)})
    return pd.DataFrame(rows), PF, best, saved


def save(saved, results):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(saved, MODEL_FILE, compress=3)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True).stdout.strip()
    except OSError:
        commit = ""
    params = {"variant": saved["variant"], "lgbm_params": saved["lgbm"].get_params(),
              "word_model": {"tfidf": "word 1-2gram min_df=2 + char_wb 2-5gram min_df=3, sublinear_tf",
                             "logreg_C": WORD_C},
              "feature_columns": saved["feature_columns"], "trained_on": "all non-holdout practice rows",
              "data_hash": data_hash(), "git_commit": commit,
              "scores": results.to_dict(orient="records")}
    PARAMS_FILE.write_text(json.dumps(params, indent=1, default=str), encoding="utf-8")
    print(f"saved {MODEL_FILE.relative_to(ROOT)} and {PARAMS_FILE.relative_to(ROOT)}")


def load_and_score():
    """Reuse the saved model: rebuild the (label lookup) features, predict the holdout, no fitting."""
    saved = joblib.load(MODEL_FILE)
    params = json.loads(PARAMS_FILE.read_text(encoding="utf-8"))
    if params["data_hash"] != data_hash():
        print("warning: data changed since the model was saved; run with --retrain")
    cv_ids = sorted(SPLITS.index[SPLITS.holdout == 0])
    hold = set(SPLITS.index[SPLITS.holdout == 1])
    train = set(cv_ids)
    trl = sorted(train)
    from sklearn.model_selection import GroupKFold
    inner = [[trl[j] for j in a] for a, _ in GroupKFold(5).split(trl, groups=[I7.PAGE_OF[i] for i in trl])]
    E = extracted_facts()
    F = pd.concat([I7.F_ALL[I7.F_ALL.id.isin(train)], E[E.id.isin(hold)]], ignore_index=True)
    te = sorted(hold & set(F.id))
    X, _ = I7.fold_table(F, train, set(te), inner)
    X.loc[te, WV] = predict_words(saved["vectorizer"], saved["word_model"], te)
    P = I7.proba3(saved["lgbm"], X.loc[te, saved["feature_columns"]])
    if saved["variant"].startswith("L4"):
        P = I7.hard_rules(P, X.loc[te])
    acc = float((P.argmax(1) == Y[te].to_numpy()).mean())
    print(f"loaded {saved['variant']}: holdout acc (extracted facts) {acc:.4f} "
          f"(saved run: {saved['holdout_acc_extracted']:.4f}), no retraining")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", action="store_true", help="score the holdout with the saved model, no training")
    ap.add_argument("--retrain", action="store_true", help="train even if a saved model matches the data")
    a = ap.parse_args()
    if a.load or (not a.retrain and MODEL_FILE.exists() and PARAMS_FILE.exists()
                  and json.loads(PARAMS_FILE.read_text(encoding="utf-8"))["data_hash"] == data_hash()):
        load_and_score()
        return
    t0 = time.time()
    results, PF, best, saved = evaluate()
    results.to_csv(OUT / "improve_word_values.csv", index=False)
    save(saved, results)
    pd.set_option("display.width", 220)
    show = ["facts", "split", "variant", "n", "acc", "macro_f1_tf", "false_precision", "pred_false", "train_acc"]
    print(results[[c for c in show if c in results.columns]].to_string(index=False))
    g = PF.groupby(["facts", "variant"])[["train_acc", "val_acc", "all_true"]].agg(["mean", "std"]).round(4)
    print("\nper-fold train vs validation (fold_group):\n", g.to_string())
    print(f"\ntotal {time.time() - t0:.0f}s")
    print("\nexternal validation on data/hooly_statements_300.csv (eda/10_h300_validation.py):", flush=True)
    subprocess.run([sys.executable, str(ROOT / "eda" / "10_h300_validation.py")], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
