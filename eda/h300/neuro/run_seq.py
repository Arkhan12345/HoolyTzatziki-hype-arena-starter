"""Liquid / recurrent sequence models over each row's timeline of EXTRACTED facts, plus an honest tree reference.

Per row: the first extracted fact's (subject, attribute); the sequence = the corpus's earlier mentions of that pair
(day gap, agrees with the row's value, same page, gap between mentions, 1) - label-free, as in the bake-off.
Tabular side (label-free): text flags, has_fact, n_facts, extraction confidence, chain features from
eda/h300/out/chains_inferred.csv (chain_degree = links touching the subject; chain_match = +1 the row's value equals
a value a chain derives for that pair, -1 it contradicts one = broken link, 0 none).
Facts: practice from eda/out/facts_extracted.pkl (out-of-fold), h300 and the synthetic validation rows from the
practice-fitted extractor (eda/h300/lib/extractor.py). Gold forms are never used for scored rows.

Models per practice fold (R1): CfC, LTC (hidden 32, log time gaps) and a LightGBM on the same features (tree reference,
strong regularisation). R2 (inside h300): the same three trained on the other h300 folds (+ train_only rows).
Outputs: out/seq_probs.pkl.

    .venv-neuro/Scripts/python eda/h300/neuro/run_seq.py [--seeds 0] [--perm]
"""
import argparse
import ast
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_h as K  # noqa: E402
from common_h import C  # noqa: E402

sys.path.insert(0, str(K.ROOT / "eda" / "neuro"))
sys.path.insert(0, str(K.ROOT / "eda"))
sys.path.insert(0, str(K.H))
from features import text_flags  # noqa: E402
from nets import RowNet, encode_dt  # noqa: E402
from lib.extractor import FactExtractor, Lexicon  # noqa: E402

SEQ_LEN = 16
CACHE = K.OUT / "seq_inputs.pkl"


def vkey(v):
    return json.dumps(v.get("value") if isinstance(v, dict) else v, sort_keys=True, default=str)


def extract_rows(ex, rows, page_base):
    """rows: DataFrame id,text,date -> facts frame (id, date, page, subject, attribute, value, conf, fact_idx)."""
    recs = [{"id": r.id, "text": r.text, "date": r.date, "label": ""} for r in rows.itertuples()]
    out = []
    for n, (r, facts) in enumerate(zip(recs, ex.extract(recs))):
        for i, f in enumerate(facts):
            out.append({"id": r["id"], "date": pd.Timestamp(r["date"]), "page": page_base + n, "subject": f.subject,
                        "attribute": f.attribute, "value": vkey(f.value), "conf": f.conf, "fact_idx": i})
    return pd.DataFrame(out, columns=["id", "date", "page", "subject", "attribute", "value", "conf", "fact_idx"])


def chain_index():
    p = K.H / "out" / "chains_inferred.csv"
    if not p.exists():
        return {}, {}
    ch = pd.read_csv(p)
    degree = pd.concat([ch["subject"], ch["target"]]).value_counts().to_dict()
    vals = {}
    for r in ch.itertuples():
        try:
            v = ast.literal_eval(r.value)
        except (ValueError, SyntaxError):
            continue
        vals.setdefault((r.target, r.attribute), set()).add(vkey(v))
    return degree, vals


def build(rows, F, corpus):
    """Sequences + label-free tab features for rows (DataFrame id,text,date) given their facts F and the corpus."""
    by_pair = {k: g.sort_values("date") for k, g in corpus.groupby(["subject", "attribute"])}
    first = F.sort_values(["id", "fact_idx"]).groupby("id").first()
    nf = F.groupby("id").size()
    degree, cvals = chain_index()
    seq = np.zeros((len(rows), SEQ_LEN, 5), np.float32)
    tab = []
    for n, r in enumerate(rows.itertuples()):
        t = {"has_fact": 0.0, "n_facts": float(nf.get(r.id, 0)), "conf": 0.0, "chain_degree": 0.0,
             "chain_match": 0.0, "n_prior": 0.0, "agree_last": 0.0}
        if r.id in first.index:
            f = first.loc[r.id]
            t.update(has_fact=1.0, conf=float(f.conf), chain_degree=float(degree.get(f.subject, 0)))
            cv = cvals.get((f.subject, f.attribute))
            if cv:
                t["chain_match"] = 1.0 if f.value in cv else -1.0
            g = by_pair.get((f.subject, f.attribute))
            if g is not None:
                g = g[(g.date <= f.date) & (g.id != r.id)].tail(SEQ_LEN)
                if len(g):
                    dd = (f.date - g.date).dt.days.values / 365.0
                    agree = (g.value.values == f.value).astype(np.float32)
                    same_page = (g.page.values == f.page).astype(np.float32)
                    gap = np.diff(np.concatenate([[dd[0]], dd])) * -1
                    seq[n, SEQ_LEN - len(g):] = np.stack([dd, agree, same_page, gap, np.ones(len(g))], 1)
                    t.update(n_prior=float(len(g)), agree_last=float(agree[-1]))
        tab.append(t)
    T = text_flags(rows.reset_index(drop=True).assign(date=pd.to_datetime(rows["date"].values)))
    X = pd.concat([pd.DataFrame(tab), T.reset_index(drop=True)], axis=1).fillna(0.0).astype(np.float32)
    return seq, X


def inputs():
    if CACHE.exists():
        return pickle.load(open(CACHE, "rb"))
    t0 = time.time()
    P, Hh = K.practice(), K.h300()
    Fp = pd.read_pickle(K.ROOT / "eda" / "out" / "facts_extracted.pkl")
    Fp = Fp.assign(conf=Fp["sim"].clip(0, 3) / 3.0)[["id", "date", "page", "subject", "attribute", "value", "conf",
                                                     "fact_idx"]]
    Fp["value"] = Fp["value"].map(lambda s: json.dumps(json.loads(s), sort_keys=True, default=str))
    lex = Lexicon.load(K.ROOT / "data" / "lexicon.json")
    prac = pd.read_csv(K.ROOT / "data" / "practice.csv")
    cv_rows = prac[prac.id.isin(P.id[P.fold >= 0])].to_dict("records")
    ex = FactExtractor(lex).fit(cv_rows)  # practice gold forms train the extractor; h300 labels never used
    Fh = extract_rows(ex, Hh, page_base=100000)
    synth_va = {}
    for mode in ("practice", "closed_world"):
        _, va, src = K.synth(mode, None)
        if va is not None:
            synth_va[mode] = (va, extract_rows(ex, va, page_base=200000 + (0 if mode == "practice" else 50000)), src)
    corpus = pd.concat([Fp, Fh], ignore_index=True)  # label-free mentions of the universe (practice + h300 text)
    data = {"P": P, "H": Hh}
    data["seqP"], data["XP"] = build(P, Fp, corpus)
    data["seqH"], data["XH"] = build(Hh, Fh, corpus)
    data["synth"] = {}
    for mode, (va, Fs, src) in synth_va.items():
        cs = pd.concat([corpus, Fs], ignore_index=True)
        sq, X = build(va, Fs, cs)
        data["synth"][mode] = {"rows": va, "seq": sq, "X": X, "source": src}
    data["extract_cov"] = {"practice": float(P.id.isin(Fp.id).mean()), "h300": float(Hh.id.isin(Fh.id).mean())}
    pickle.dump(data, open(CACHE, "wb"))
    print(f"inputs built ({time.time() - t0:.0f}s), coverage {data['extract_cov']}", flush=True)
    return data


def lgbm(seed):
    return LGBMClassifier(num_leaves=15, n_estimators=200, learning_rate=0.05, min_child_samples=20,
                          reg_lambda=10.0, min_sum_hessian_in_leaf=20, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=2, deterministic=True,
                          force_row_wise=True)


def seq_arrays(seq, X, cols):
    s, ts = encode_dt(seq, "log")
    return [s, ts, X[cols].values.astype(np.float32)]


def fit_predict(kind, seed, tr_in, ytr, te_ins, perm_seed=None):
    if perm_seed is not None:
        ytr = np.random.default_rng(perm_seed).permutation(ytr)
    if kind == "lgbm":
        m = lgbm(seed)
        Xtr = np.concatenate([tr_in[2], tr_in[0][..., 1].mean(1, keepdims=True)], 1)
        m.fit(Xtr, ytr)
        out = []
        for ins in te_ins:
            Xte = np.concatenate([ins[2], ins[0][..., 1].mean(1, keepdims=True)], 1)
            p = np.zeros((len(Xte), 3))
            p[:, m.classes_] = m.predict_proba(Xte)
            out.append(p)
        return out, 0.0, m, Xtr
    net = RowNet(kind, tr_in[0].shape[2], tr_in[2].shape[1], hidden=32)
    net, fs = C.train_dev(net, tr_in, ytr, epochs=40, seed=seed, weight=False, budget_s=40)
    return [C.predict_dev(net, ins) for ins in te_ins], fs, net, None


def sub(ins, m):
    return [a[m] for a in ins]


def run(args):
    D = inputs()
    P, Hh = D["P"], D["H"]
    cols = list(D["XP"].columns)
    cv = (P.fold >= 0).values
    Pi = sub(seq_arrays(D["seqP"], D["XP"], cols), cv)
    Pc = P[cv].reset_index(drop=True)
    Hi = seq_arrays(D["seqH"], D["XH"], cols)
    h_cv = (Hh.part == "cv").values
    Si = {m: (d["rows"], seq_arrays(d["seq"], d["X"], cols)) for m, d in D["synth"].items()}
    res = {"meta": {"extract_cov": D["extract_cov"], "device": K.DEVICE.type, "seeds": args.seeds, "perm": args.perm,
                    "features": cols}, "R1": {}, "R2": {}}
    t0 = time.time()
    for kind in ("lgbm", "cfc", "ltc"):
        for seed in args.seeds:
            oof = np.zeros((len(Pc), 3))
            hp, sp, fits, trf = [], {m: [] for m in Si}, [], []
            for k in range(5):
                trm, vam = Pc.fold.values != k, Pc.fold.values == k
                outs, fs, model, _ = fit_predict(kind, seed, sub(Pi, trm), Pc.y.values[trm],
                                                 [sub(Pi, vam), Hi, sub(Pi, trm)] + [s[1] for s in Si.values()],
                                                 perm_seed=(300 + k) if args.perm else None)
                oof[vam] = outs[0]
                hp.append(outs[1])
                trf.append(K.scores(Pc.y.values[trm], outs[2])["f1_tf"])
                for (m, _), o in zip(Si.items(), outs[3:]):
                    sp[m].append(o)
                fits.append(fs)
                if k == 0 and seed == args.seeds[0] and not args.perm and kind != "lgbm":
                    res.setdefault("cpu2_s", {})[kind] = C.cpu2_seconds(model, [a[:512] for a in Hi])
            Ph = np.mean(hp, 0)
            out = {"practice": K.scores(Pc.y.values, oof),
                   "practice_fold": {k: K.scores(Pc.y.values[Pc.fold.values == k], oof[Pc.fold.values == k])
                                     for k in range(5)},
                   "h300": K.scores(Hh.y.values[h_cv], Ph[h_cv]), "train_f1_tf_mean": float(np.mean(trf)),
                   "fit_s": float(np.sum(fits)), "oof": oof, "h300_probs": Ph, "synth_val": {}}
            for m, (rows, _) in Si.items():
                PS = np.mean(sp[m], 0)
                out["synth_val"][m] = {"scores": K.scores(rows["y"].values, PS), "per_case": K.per_case(rows, PS)}
            res["R1"][(kind, seed)] = out
            print(f"R1 {kind} seed={seed}{' PERM' if args.perm else ''}: practice f1_tf {out['practice']['f1_tf']:.3f} "
                  f"acc {out['practice']['acc']:.3f} | h300 f1_3 {out['h300']['f1_3']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    # R2: inside h300 (subject-grouped folds; train_only rows always in training)
    Hc = Hh[h_cv].reset_index(drop=True)
    Hci, Hti = sub(Hi, h_cv), sub(Hi, (Hh.part == "train_only").values)
    y_to = Hh.y.values[(Hh.part == "train_only").values]
    for kind in ("lgbm", "cfc", "ltc"):
        for seed in args.seeds:
            oof = np.zeros((len(Hc), 3))
            pp, trf = [], []
            for j in range(5):
                trm, vam = Hc.fold.values != j, Hc.fold.values == j
                tr_in = [np.concatenate([a, b]) for a, b in zip(sub(Hci, trm), Hti)]
                ytr = np.concatenate([Hc.y.values[trm], y_to])
                outs, _, _, _ = fit_predict(kind, seed, tr_in, ytr, [sub(Hci, vam), Pi, sub(Hci, trm)],
                                            perm_seed=(400 + j) if args.perm else None)
                oof[vam] = outs[0]
                pp.append(outs[1])
                trf.append(K.scores(Hc.y.values[trm], outs[2])["f1_3"])
            out = {"h300": K.scores(Hc.y.values, oof),
                   "h300_fold": {j: K.scores(Hc.y.values[Hc.fold.values == j], oof[Hc.fold.values == j])
                                 for j in range(5)},
                   "practice": K.scores(Pc.y.values, np.mean(pp, 0)), "train_f1_3_mean": float(np.mean(trf)),
                   "oof": oof}
            res["R2"][(kind, seed)] = out
            print(f"R2 {kind} seed={seed}{' PERM' if args.perm else ''}: h300 f1_3 {out['h300']['f1_3']:.3f} | "
                  f"practice acc {out['practice']['acc']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    res["ids"] = {"practice": list(Pc.id), "h300": list(Hh.id), "h300_cv": list(Hc.id)}
    name = "seq_perm.pkl" if args.perm else f"seq_probs_s{'-'.join(map(str, args.seeds))}.pkl"
    pickle.dump(res, open(K.OUT / name, "wb"))
    print(f"saved {name} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--perm", action="store_true")
    a = ap.parse_args()
    a.seeds = [int(s) for s in a.seeds.split(",")]
    run(a)
