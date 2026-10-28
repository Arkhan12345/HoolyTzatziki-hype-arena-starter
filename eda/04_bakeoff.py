"""Step 5: mini bake-off. Classic zoo vs liquid (CfC) vs spiking MLP on the same folds.

Leak control: label-free features (text, corpus support, page shape) are computed once; the gold-timeline
KB features are rebuilt per outer fold from training labels only. Training rows get their KB features
out-of-fold (inner split of the training set), so train and test rows see the KB the same way.

CV schemes: group  = GroupKFold(5) by page
            row5   = every 5th row hidden (like scripts/check.sh), 5 rotations
            time   = train < 2026-07-01, test >= 2026-07-01

Outputs (eda/out/): bakeoff.csv, ablation.csv, bakeoff_per_archetype.csv (if practice_annotated.csv exists),
bakeoff_oof_group.pkl, figs/bakeoff.png
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from scipy.sparse import hstack
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).parent))
from features import KB_REL_ORDER, facts_to_rows, kb_features, support_features, text_flags  # noqa: E402

warnings.filterwarnings("ignore")
torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eda" / "out"
(OUT / "figs").mkdir(parents=True, exist_ok=True)
CLASSES = np.array(["false", "true", "unsure"])
SEED = 0
SEQ_LEN = 16

FAMILIES = {
    "temporal_kb": [f"rel_{k}" for k in KB_REL_ORDER] + ["days_since_last_change_min", "days_since_last_change_max",
                                                        "n_gold_values_for_pair_min", "n_gold_values_for_pair_max"],
    "support": [f"{c}_{a}" for c in ["same_value_rows", "same_value_pages", "value_share", "pair_n_values",
                                     "pair_rows", "agree_prev_mention", "agree_next_mention", "n_values_before"]
                for a in ["min", "max"]],
    "wording": ["first_person", "we_our", "quote", "hearsay", "hedge", "superlative", "now_today", "still", "future",
                "negation", "change_verb", "money_sym", "number_words", "joke_tail", "no_period", "kv_line",
                "n_chars", "n_words", "n_numbers", "n_commas", "starts_pronoun", "hype_score"],
    "date": ["text_date_delta_days", "has_text_date", "text_date_in_past", "text_date_in_future",
             "today_date_mismatch"],
    "page_genre": ["page_size", "pos_in_page", "pos_rel", "page_first_person_share", "page_hype_share",
                   "n_facts", "any_pronoun_subject"],  # + genre_* one-hots added below
    "value_shape": [f"{c}_{a}" for c in ["log_dev_from_pair_median", "round_number", "op_present"]
                    for a in ["min", "max"]],  # + kind_* one-hots added below
}


# ---------------------------------------------------------------- features
def label_free_table():
    R = pd.read_pickle(OUT / "rows.pkl")
    F = pd.read_pickle(OUT / "facts.pkl")
    T = text_flags(R)
    FS = support_features(F)
    sup_cols = ["same_value_rows", "same_value_pages", "value_share", "pair_n_values", "pair_rows",
                "log_dev_from_pair_median", "round_number", "op_present", "agree_prev_mention",
                "agree_next_mention", "n_values_before"]
    agg = facts_to_rows(FS, sup_cols)
    X = pd.concat([R.set_index("id")[["page_size", "pos_in_page", "pos_rel", "n_facts", "any_pronoun_subject"]],
                   T.set_index(R["id"]), agg], axis=1)
    X["any_pronoun_subject"] = X["any_pronoun_subject"].astype(float)
    # page shape and genre are label-free (computed from text only) - take them from 02_rules output
    P = pd.read_pickle(OUT / "features.pkl").set_index("id")
    X["page_first_person_share"] = P["page_first_person_share"]
    X["page_hype_share"] = P["page_hype_share"]
    genre = pd.get_dummies(P["page_genre"], prefix="genre").astype(float)
    kinds = R.set_index("id")["kinds"].str.get_dummies(sep="|").add_prefix("kind_").astype(float)
    X = pd.concat([X, genre, kinds], axis=1)
    FAMILIES["page_genre"] += list(genre.columns)
    FAMILIES["value_shape"] += list(kinds.columns)
    X = X.reindex(R["id"])
    return R, F, FS, X


def kb_table(F, ids_to_score, train_ids, inner_splits):
    """KB features for ids_to_score from train_ids' gold; train rows get theirs out-of-fold via inner_splits."""
    num = ["days_since_last_change", "n_gold_values_for_pair"]
    parts = []
    # test rows: KB from all training labels
    test_ids = [i for i in ids_to_score if i not in train_ids]
    if test_ids:
        sub = F[F.id.isin(set(test_ids) | train_ids)]
        kb = kb_features(sub, train_ids)
        kb = pd.concat([sub[["id"]], kb], axis=1)
        parts.append(facts_to_rows(kb[kb.id.isin(test_ids)], num, rel_col="kb_relation"))
    # training rows: inner OOF
    for inner_train in inner_splits:
        inner_train = set(inner_train)
        inner_test = train_ids - inner_train
        sub = F[F.id.isin(train_ids)]
        kb = kb_features(sub, inner_train)
        kb = pd.concat([sub[["id"]], kb], axis=1)
        parts.append(facts_to_rows(kb[kb.id.isin(inner_test)], num, rel_col="kb_relation"))
    out = pd.concat(parts)
    return out.drop(columns=["kb_relation"])


def sequences(R, FS):
    """Label-free per-row timeline of prior corpus mentions of the row's first fact's (subject, attribute)."""
    first = FS.sort_values(["id", "fact_idx"]).groupby("id").first()
    by_pair = {k: g.sort_values("date") for k, g in FS.groupby(["subject", "attribute"])}
    seq = np.zeros((len(R), SEQ_LEN, 5), dtype=np.float32)
    for n, rid in enumerate(R["id"]):
        f = first.loc[rid]
        g = by_pair[(f.subject, f.attribute)]
        g = g[(g.date <= f.date) & (g.id != rid)].tail(SEQ_LEN)
        if len(g) == 0:
            continue
        dd = (f.date - g.date).dt.days.values / 365.0
        agree = (g.value.values == f.value).astype(np.float32)
        same_page = (g.page.values == f.page).astype(np.float32)
        gap = np.diff(np.concatenate([[dd[0]], dd])) * -1
        rows = np.stack([dd, agree, same_page, gap, np.ones(len(g))], axis=1)
        seq[n, SEQ_LEN - len(g):] = rows  # left-pad so the last step is the most recent mention
    return seq


# ---------------------------------------------------------------- neural models
class CfCNet(torch.nn.Module):
    def __init__(self, n_seq, n_tab, units=24):
        super().__init__()
        from ncps.torch import CfC
        self.rnn = CfC(n_seq, units, batch_first=True)
        self.head = torch.nn.Sequential(torch.nn.Linear(units + n_tab, 48), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                                        torch.nn.Linear(48, 3))

    def forward(self, s, x):
        _, h = self.rnn(s)
        return self.head(torch.cat([h, x], dim=1))


class SpikingMLP(torch.nn.Module):
    def __init__(self, n_in, hidden=64, steps=25):
        super().__init__()
        import snntorch as snn
        from snntorch import surrogate
        sg = surrogate.fast_sigmoid(slope=25)
        self.steps = steps
        self.fc1, self.lif1 = torch.nn.Linear(n_in, hidden), snn.Leaky(beta=0.9, spike_grad=sg)
        self.fc2, self.lif2 = torch.nn.Linear(hidden, hidden), snn.Leaky(beta=0.9, spike_grad=sg)
        self.fc3, self.lif3 = torch.nn.Linear(hidden, 3), snn.Leaky(beta=0.9, spike_grad=sg, output=True)

    def forward(self, x):
        from snntorch import spikegen
        spk_in = spikegen.rate(x, num_steps=self.steps)  # rate encoding: P(spike) = feature in [0, 1]
        m1, m2, m3 = self.lif1.init_leaky(), self.lif2.init_leaky(), self.lif3.init_leaky()
        mem_sum = 0
        for t in range(self.steps):
            s1, m1 = self.lif1(self.fc1(spk_in[t]), m1)
            s2, m2 = self.lif2(self.fc2(s1), m2)
            _, m3 = self.lif3(self.fc3(s2), m3)
            mem_sum = mem_sum + m3
        return mem_sum / self.steps


def train_torch(model, inputs, y, epochs, lr=3e-3, budget_s=55):
    torch.manual_seed(SEED)
    counts = np.bincount(y, minlength=3).astype(float)
    w = torch.tensor(len(y) / (3 * np.maximum(counts, 1)), dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    yt = torch.tensor(y)
    n = len(y)
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for b in range(0, n, 128):
            idx = perm[b:b + 128]
            opt.zero_grad()
            loss = loss_fn(model(*[i[idx] for i in inputs]), yt[idx])
            loss.backward()
            opt.step()
        if time.time() - t0 > budget_s:
            break
    return model


def predict_torch(model, inputs):
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(*inputs), dim=1).numpy()


# ---------------------------------------------------------------- classic models
def imp_scale(scaler=StandardScaler):
    return [SimpleImputer(strategy="median", add_indicator=False), scaler()]


def classic_models():
    bal = "balanced"
    return {
        "logreg_l2": lambda: make_pipeline(*imp_scale(), LogisticRegression(C=0.5, max_iter=2000, class_weight=bal)),
        "lightgbm": lambda: LGBMClassifier(num_leaves=15, n_estimators=200, learning_rate=0.05, min_child_samples=10,
                                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                           class_weight=bal, random_state=SEED, verbose=-1),
        "lightgbm_unweighted": lambda: LGBMClassifier(num_leaves=15, n_estimators=200, learning_rate=0.05,
                                                      min_child_samples=10, subsample=0.8, subsample_freq=1,
                                                      colsample_bytree=0.8, random_state=SEED, verbose=-1),
        "svm_rbf": lambda: make_pipeline(*imp_scale(), SVC(C=2.0, gamma="scale", probability=True,
                                                           class_weight=bal, random_state=SEED)),
        "knn_15": lambda: make_pipeline(*imp_scale(), KNeighborsClassifier(n_neighbors=15, weights="distance")),
        "extratrees": lambda: make_pipeline(SimpleImputer(strategy="median"),
                                            ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                                                                 class_weight=bal, n_jobs=2, random_state=SEED)),
    }


def fit_predict_stack(Xtr, ytr, Xte):
    base = classic_models()
    names = ["lightgbm", "logreg_l2"]
    oof = np.zeros((len(ytr), 3 * len(names)))
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for tr, va in skf.split(Xtr, ytr):
        for j, n in enumerate(names):
            m = base[n]().fit(Xtr.iloc[tr], ytr[tr])
            oof[va, 3 * j:3 * j + 3] = m.predict_proba(Xtr.iloc[va])
    te = np.hstack([base[n]().fit(Xtr, ytr).predict_proba(Xte) for n in names])
    meta = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(oof, ytr)
    return meta.predict_proba(te)


# ---------------------------------------------------------------- metrics
def ece(proba, y, bins=10):
    conf = proba.max(1)
    correct = (proba.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e


def metrics(y, proba):
    pred = CLASSES[proba.argmax(1)]
    ylab = CLASSES[y]
    return {"macro_f1_tf": f1_score(ylab, pred, labels=["true", "false"], average="macro", zero_division=0),
            "acc": accuracy_score(ylab, pred), "ece": ece(proba, y),
            "unsure_recall": ((pred == "unsure") & (ylab == "unsure")).sum() / max((ylab == "unsure").sum(), 1),
            "pred_unsure": int((pred == "unsure").sum())}


# ---------------------------------------------------------------- CV
def splits(R, scheme):
    idx = np.arange(len(R))
    if scheme == "group":
        return list(GroupKFold(5).split(idx, groups=R["page"].values))
    if scheme == "row5":
        return [(idx[idx % 5 != k], idx[idx % 5 == k]) for k in range(5)]
    if scheme == "time":
        m = (R["date"] < "2026-07-01").values
        return [(idx[m], idx[~m])]
    raise ValueError(scheme)


def inner_id_splits(R, tr, scheme):
    sub = R.iloc[tr]
    ids = sub["id"].values
    if scheme == "row5":
        it = KFold(5, shuffle=True, random_state=SEED).split(ids)
    else:
        it = GroupKFold(5).split(ids, groups=sub["page"].values)
    return [ids[a] for a, _ in it]


def run():
    t_all = time.time()
    R, F, FS, X0 = label_free_table()
    y_all = pd.Series(R["label"].values).map({c: i for i, c in enumerate(CLASSES)}).values
    print("label-free features:", X0.shape, f"({time.time() - t_all:.1f}s)")
    SEQ = sequences(R, FS)
    print("sequences:", SEQ.shape, f"({time.time() - t_all:.1f}s)")
    text_all = R["text"].fillna("").values

    results, ablation = [], []
    oof_group = {}
    for scheme in ["group", "row5", "time"]:
        sp = splits(R, scheme)
        fold_probs = {}
        for fi, (tr, te) in enumerate(sp):
            tf0 = time.time()
            train_ids = set(R["id"].values[tr])
            KB = kb_table(F, list(R["id"].values), train_ids, inner_id_splits(R, tr, scheme))
            X = pd.concat([X0, KB.reindex(X0.index)], axis=1)
            for c in [f"rel_{k}" for k in KB_REL_ORDER]:
                X[c] = X[c].fillna(0)
            X.columns = [str(c) for c in X.columns]
            Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y_all[tr], y_all[te]

            def record(name, proba, fit_s, pred_s):
                fold_probs.setdefault(name, []).append((te, proba, fit_s, pred_s))

            # majority baseline
            p = np.zeros((len(te), 3)); p[:, 1] = 1.0
            record("majority_all_true", p, 0.0, 0.0)
            # text only
            t0 = time.time()
            vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
            vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)
            Ttr = hstack([vw.fit_transform(text_all[tr]), vc.fit_transform(text_all[tr])]).tocsr()
            m = LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced").fit(Ttr, ytr)
            fit_s = time.time() - t0; t0 = time.time()
            Tte = hstack([vw.transform(text_all[te]), vc.transform(text_all[te])]).tocsr()
            record("tfidf_logreg_text_only", m.predict_proba(Tte), fit_s, time.time() - t0)
            # classic zoo
            for name, mk in classic_models().items():
                t0 = time.time(); m = mk().fit(Xtr, ytr); fit_s = time.time() - t0
                t0 = time.time(); record(name, m.predict_proba(Xte), fit_s, time.time() - t0)
            t0 = time.time()
            record("stack_lgbm_logreg", fit_predict_stack(Xtr, ytr, Xte), time.time() - t0, 0.0)
            # neural: shared preprocessing (impute + scale, fit on train)
            imp = SimpleImputer(strategy="median").fit(Xtr)
            std = StandardScaler().fit(imp.transform(Xtr))
            mm = MinMaxScaler(clip=True).fit(imp.transform(Xtr))
            xtr_s = torch.tensor(std.transform(imp.transform(Xtr)), dtype=torch.float32)
            xte_s = torch.tensor(std.transform(imp.transform(Xte)), dtype=torch.float32)
            s_tr, s_te = torch.tensor(SEQ[tr]), torch.tensor(SEQ[te])
            t0 = time.time()
            cfc = train_torch(CfCNet(SEQ.shape[2], xtr_s.shape[1]), [s_tr, xtr_s], ytr, epochs=40)
            fit_s = time.time() - t0; t0 = time.time()
            record("cfc_liquid_timeline", predict_torch(cfc, [s_te, xte_s]), fit_s, time.time() - t0)
            xtr_m = torch.tensor(mm.transform(imp.transform(Xtr)), dtype=torch.float32)
            xte_m = torch.tensor(mm.transform(imp.transform(Xte)), dtype=torch.float32)
            t0 = time.time()
            snn = train_torch(SpikingMLP(xtr_m.shape[1]), [xtr_m], ytr, epochs=40)
            fit_s = time.time() - t0; t0 = time.time()
            record("spiking_mlp_lif", predict_torch(snn, [xte_m]), fit_s, time.time() - t0)

            # ablation: LightGBM minus each family (group scheme only)
            if scheme == "group":
                for fam in ["none"] + list(FAMILIES):
                    drop = [c for c in FAMILIES.get(fam, []) if c in X.columns]
                    m = classic_models()["lightgbm"]().fit(Xtr.drop(columns=drop), ytr)
                    pr = m.predict_proba(Xte.drop(columns=drop))
                    ablation.append({"fold": fi, "minus_family": fam, "n_features": X.shape[1] - len(drop),
                                     "y": yte, "proba": pr})
            print(f"  {scheme} fold {fi}: {time.time() - tf0:.1f}s (total {time.time() - t_all:.0f}s)", flush=True)

        # aggregate pooled OOF per model
        for name, parts in fold_probs.items():
            te_all = np.concatenate([p[0] for p in parts])
            proba = np.vstack([p[1] for p in parts])
            met = metrics(y_all[te_all], proba)
            results.append({"model": name, "cv": scheme, **{k: round(v, 4) if isinstance(v, float) else v
                                                             for k, v in met.items()},
                            "fit_s": round(sum(p[2] for p in parts), 2),
                            "predict_s": round(sum(p[3] for p in parts), 3)})
            if scheme == "group":
                full = np.zeros((len(R), 3)); full[te_all] = proba
                oof_group[name] = full

    res = pd.DataFrame(results)
    res.to_csv(OUT / "bakeoff.csv", index=False)
    pd.to_pickle({"ids": R["id"].values, "y": y_all, "proba": oof_group}, OUT / "bakeoff_oof_group.pkl")

    ab = pd.DataFrame(ablation)
    rows = []
    for fam, g in ab.groupby("minus_family", sort=False):
        met = metrics(np.concatenate(g.y.values), np.vstack(g.proba.values))
        rows.append({"minus_family": fam, "n_features": int(g.n_features.iloc[0]),
                     "macro_f1_tf": round(met["macro_f1_tf"], 4), "acc": round(met["acc"], 4),
                     "ece": round(met["ece"], 4)})
    ab = pd.DataFrame(rows)
    ab["delta_vs_all"] = (ab.macro_f1_tf - ab.loc[ab.minus_family == "none", "macro_f1_tf"].iloc[0]).round(4)
    ab.to_csv(OUT / "ablation.csv", index=False)

    # per archetype
    ann = OUT / "practice_annotated.csv"
    per_arch = None
    if ann.exists():
        A = pd.read_csv(ann).set_index("id").reindex(R["id"].values)
        if "archetype" in A.columns:
            top = (res[(res.cv == "group") & (res.model != "majority_all_true")]
                   .sort_values("macro_f1_tf", ascending=False).model.head(4).tolist())
            top += [m for m in ["cfc_liquid_timeline", "spiking_mlp_lif"] if m not in top]
            out = []
            for arch, idx in pd.Series(np.arange(len(R))).groupby(A["archetype"].values):
                idx = idx.values
                for mname in top:
                    met = metrics(y_all[idx], oof_group[mname][idx])
                    out.append({"archetype": arch, "n": len(idx),
                                "n_false": int((y_all[idx] == 0).sum()), "n_unsure": int((y_all[idx] == 2).sum()),
                                "model": mname, "macro_f1_tf": round(met["macro_f1_tf"], 4),
                                "acc": round(met["acc"], 4)})
            per_arch = pd.DataFrame(out)
            per_arch.to_csv(OUT / "bakeoff_per_archetype.csv", index=False)

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = res[res.cv == "group"].sort_values("macro_f1_tf")
    base = g.loc[g.model == "majority_all_true", "macro_f1_tf"].iloc[0]
    g = g[g.model != "majority_all_true"]
    colors = ["#c0792b" if "cfc" in m else "#7d5ba6" if "spiking" in m else "#3b6ea5" for m in g.model]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh(g.model, g.macro_f1_tf, color=colors)
    ax.axvline(base, color="#c0392b", ls="--", lw=1, label=f"all-true baseline {base:.2f}")
    for yv, v in enumerate(g.macro_f1_tf):
        ax.text(v + 0.005, yv, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("macro-F1 (true/false), GroupKFold by page")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figs" / "bakeoff.png", dpi=110)
    plt.close(fig)

    pd.set_option("display.width", 220)
    for scheme in ["group", "row5", "time"]:
        print(f"\n== {scheme}")
        print(res[res.cv == scheme].sort_values("macro_f1_tf", ascending=False).drop(columns="cv").to_string(index=False))
    print("\n== ablation (LightGBM, group)\n", ab.to_string(index=False))
    if per_arch is not None:
        print("\n== per archetype (group OOF)\n",
              per_arch.pivot_table(index=["archetype", "n", "n_false", "n_unsure"], columns="model",
                                   values="macro_f1_tf").round(3).to_string())
    else:
        print("\nper-archetype: skipped (eda/out/practice_annotated.csv not found)")
    print(f"\ntotal {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    run()
