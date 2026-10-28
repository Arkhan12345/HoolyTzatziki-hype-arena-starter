"""Combination knobs for the loop's combination round (`loop.py --combos`).

The fold pipeline is the final model's (the L7 branch of eda/07_improve.run_fold: LightGBM, inner out-of-fold
probabilities, evidence-sum calibrator, hard rules, temperature), but it also keeps the calibrated inner
out-of-fold probabilities of the training rows, so anything stacked on top is tuned without touching test labels.

Knobs (all off by default; an all-off run reproduces the final model):
  et_vote       0 = off; w in (0, 1): soft vote  w * LightGBM + (1 - w) * ExtraTrees (median-imputed, 200 trees)
  lgbm_seeds    1 = off; n: LightGBM averaged over n random seeds
  bigru_blend   1.0 = off; w or "auto" (0.85 / 0.7 picked on inner log-loss): p = w * model + (1 - w) * char BiGRU
  neural_stack  False / True: logistic-regression stack of the model's probabilities + CfC and spiking unsure heads
  unsure_head   "none" / "snn" / "cfc": unsure where the head is sure and the model is not (thresholds tuned on the
                training rows' inner out-of-fold predictions)
Neural members come from eda/out/neuro_combo_oof.pkl (eda/neuro/combo_oof.py, GPU).
"""
import pickle
import time

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

DEFAULTS = {"et_vote": 0.0, "lgbm_seeds": 1, "bigru_blend": 1.0, "neural_stack": False, "unsure_head": "none"}
NEEDS_TORCH = "needs torch / WITH_NEURO"
_NEURO = None


def full(cfg):
    return {**DEFAULTS, **cfg}


def neural_members(cfg):
    c = full(cfg)
    m = []
    if c["bigru_blend"] != 1.0:
        m.append("bigru")
    if c["neural_stack"]:
        m += ["cfc", "snn"]
    if c["unsure_head"] != "none":
        m.append(c["unsure_head"])
    return sorted(set(m))


def neuro():
    global _NEURO
    if _NEURO is None:
        from pathlib import Path
        p = Path(__file__).resolve().parents[1] / "out" / "neuro_combo_oof.pkl"
        with open(p, "rb") as f:
            _NEURO = pickle.load(f)
    return _NEURO


def extra_cpu_s(cfg):
    """CPU seconds (2 threads, 20k rows) the neural members add at prediction time."""
    mem = neural_members(cfg)
    if not mem:
        return 0.0
    t = neuro()["cpu2_s_20k"]
    return float(sum(t[m] for m in mem))


# ------------------------------------------------------------------ model wrappers (combos 4 and 5)
class SeedBag:
    def __init__(self, params, n):
        self.params, self.n = params, n

    def fit(self, X, y):
        seed = self.params.get("random_state", 0)
        self.models = [LGBMClassifier(**{**self.params, "random_state": seed + s}).fit(X, y) for s in range(self.n)]
        self.classes_ = self.models[0].classes_
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)


class ETVote:
    def __init__(self, make_lgbm, w, seed):
        self.make_lgbm, self.w, self.seed = make_lgbm, w, seed

    def fit(self, X, y):
        self.gb = self.make_lgbm().fit(X, y)
        self.imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(X)
        self.et = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=2,
                                       random_state=self.seed).fit(self.imp.transform(X), y)
        assert list(self.gb.classes_) == list(self.et.classes_)
        self.classes_ = self.gb.classes_
        return self

    def predict_proba(self, X):
        return self.w * self.gb.predict_proba(X) + (1 - self.w) * self.et.predict_proba(self.imp.transform(X))


def model_factory(params, cfg):
    """Replacement for M.lgbm: plain LightGBM, a seed bag, and/or an ExtraTrees soft vote."""
    c = full(cfg)
    n = int(c["lgbm_seeds"])

    def make_gb():
        return SeedBag(params, n) if n > 1 else LGBMClassifier(**params)
    if float(c["et_vote"]) > 0:
        return lambda n_jobs=-1: ETVote(make_gb, float(c["et_vote"]), params.get("random_state", 0))
    return lambda n_jobs=-1: make_gb()


# ------------------------------------------------------------------ the final model's fold, keeping inner OOF
def final_fold(M, X, train_ids, test_ids, inner_splits):
    """eda/07_improve.run_fold, L7_runtime_retrain branch, returning (Pin, P) after calibrator/rules/temperature."""
    tr, te = sorted(train_ids), sorted(test_ids)
    ytr, yte = M.Y[tr].to_numpy(), M.Y[te].to_numpy()
    Ztr, Zte = X.loc[tr], X.loc[te]
    inner_masks = [(np.isin(tr, list(a)), ~np.isin(tr, list(a))) for a in inner_splits]
    cols = M.cols_for("L7_runtime_retrain", X)
    m = M.lgbm(2).fit(Ztr[cols], ytr)
    t0 = time.time()
    P = M.proba3(m, Zte[cols])
    pred_s = (time.time() - t0) * M.N_RUN / max(len(te), 1)
    Pin = np.zeros((len(tr), 3))
    for a, b in inner_masks:
        mi = M.lgbm(2).fit(Ztr.loc[a, cols], ytr[a])
        Pin[b] = M.proba3(mi, Ztr.loc[b, cols])
    items_tr, items_te = M.evidence_items(Ztr), M.evidence_items(Zte)
    lo_in = np.zeros(len(tr))
    for a, b in inner_masks:
        lo_in[b] = M.EvidenceJudge().fit(items_tr[a], ytr[a] == 0).log_odds(items_tr[b])
    judge = M.EvidenceJudge().fit(items_tr, ytr == 0)
    lo_te = judge.log_odds(items_te)
    P_ins, lo_ins = M.proba3(m, Ztr[cols]), judge.log_odds(items_tr)     # in-sample: for the train-OOF gap gate
    yb = (ytr == 0).astype(float)
    best_w, best_ll = 1.0, np.inf
    for w in np.linspace(0, 1, 11):
        ll = M.logloss_bin(1 / (1 + np.exp(-(w * M.logit(Pin[:, 0]) + (1 - w) * lo_in))), yb)
        if ll < best_ll:
            best_w, best_ll = w, ll
    cal = LogisticRegression(C=1.0).fit(np.column_stack([M.logit(Pin[:, 0]), lo_in]), yb)
    ll_cal = M.logloss_bin(cal.predict_proba(np.column_stack([M.logit(Pin[:, 0]), lo_in]))[:, 1], yb)
    if ll_cal < best_ll:
        pf_in = cal.predict_proba(np.column_stack([M.logit(Pin[:, 0]), lo_in]))[:, 1]
        pf_te = cal.predict_proba(np.column_stack([M.logit(P[:, 0]), lo_te]))[:, 1]
        pf_ins = cal.predict_proba(np.column_stack([M.logit(P_ins[:, 0]), lo_ins]))[:, 1]
    else:
        pf_in = 1 / (1 + np.exp(-(best_w * M.logit(Pin[:, 0]) + (1 - best_w) * lo_in)))
        pf_te = 1 / (1 + np.exp(-(best_w * M.logit(P[:, 0]) + (1 - best_w) * lo_te)))
        pf_ins = 1 / (1 + np.exp(-(best_w * M.logit(P_ins[:, 0]) + (1 - best_w) * lo_ins)))
    Pin, P = M.with_false_prob(Pin, pf_in), M.with_false_prob(P, pf_te)
    P_ins = M.with_false_prob(P_ins, pf_ins)
    Pin, P, P_ins = M.hard_rules(Pin, Ztr), M.hard_rules(P, Zte), M.hard_rules(P_ins, Ztr)
    T = M.fit_temperature(Pin, ytr)
    Pin, P, P_ins = M.apply_temperature(Pin, T), M.apply_temperature(P, T), M.apply_temperature(P_ins, T)
    return dict(tr=tr, te=te, ytr=ytr, yte=yte, Ztr=Ztr, Zte=Zte, Pin=Pin, P=P, P_ins=P_ins, masks=inner_masks,
                info={"predict_s_cpu2": pred_s})


def _logloss3(P, y):
    return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1)).mean())


def _norm(P):
    P = np.clip(P, 1e-9, None)
    return P / P.sum(1, keepdims=True)


def _stack_X(P, cfc, snn):
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
    return np.column_stack([np.log(np.clip(P, 1e-6, 1)), lg(cfc), lg(snn)])


def _lr3(Xs, y):
    return LogisticRegression(C=1.0, max_iter=3000).fit(Xs, y)


def _proba_lr(model, Xs):
    p = np.zeros((len(Xs), 3))
    p[:, model.classes_] = model.predict_proba(Xs)
    return p


def run_one_combo(M, X, train, test, inner, cfg, key):
    """One fold of the final model + the configured combinations. Returns (P, pred, info, yte, te)."""
    t0 = time.time()
    c = full(cfg)
    d = final_fold(M, X, train, test, inner)
    Pin, P, Pis, ytr, Ztr, Zte = d["Pin"], d["P"], d["P_ins"], d["ytr"], d["Ztr"], d["Zte"]
    if neural_members(c):
        parts = neuro()["parts"]
        Ntr = parts[(key, "train")].reindex(d["tr"])
        Nte = parts[(key, "test")].reindex(d["te"])
        assert not Ntr.isna().any().any() and not Nte.isna().any().any(), f"neural members missing for {key}"
    if c["bigru_blend"] != 1.0:
        Bin, Bte = _norm(Ntr[["bigru_p0", "bigru_p1", "bigru_p2"]].to_numpy()), \
            _norm(Nte[["bigru_p0", "bigru_p1", "bigru_p2"]].to_numpy())
        if c["bigru_blend"] == "auto":
            w = min((0.85, 0.7), key=lambda w: _logloss3(_norm(w * Pin + (1 - w) * Bin), ytr))
        else:
            w = float(c["bigru_blend"])
        Pin, P, Pis = _norm(w * Pin + (1 - w) * Bin), _norm(w * P + (1 - w) * Bte), _norm(w * Pis + (1 - w) * Bin)
        Pin, P, Pis = M.hard_rules(Pin, Ztr), M.hard_rules(P, Zte), M.hard_rules(Pis, Ztr)
        d["info"]["bigru_w"] = w
    if c["neural_stack"]:
        Xin = _stack_X(Pin, Ntr["cfc_uns"].to_numpy(), Ntr["snn_uns"].to_numpy())
        Xte = _stack_X(P, Nte["cfc_uns"].to_numpy(), Nte["snn_uns"].to_numpy())
        Sin = np.zeros_like(Pin)
        for a, b in d["masks"]:
            Sin[b] = _proba_lr(_lr3(Xin[a], ytr[a]), Xin[b])        # the stacker's own inner out-of-fold
        stk = _lr3(Xin, ytr)
        Ste = _proba_lr(stk, Xte)
        Sis = _proba_lr(stk, _stack_X(Pis, Ntr["cfc_uns"].to_numpy(), Ntr["snn_uns"].to_numpy()))
        Pin, P, Pis = M.hard_rules(_norm(Sin), Ztr), M.hard_rules(_norm(Ste), Zte), M.hard_rules(_norm(Sis), Ztr)
    cand_tr, cand_te = M.unsure_candidates(Ztr), M.unsure_candidates(Zte)
    tau = M.tune_tau(Pin, ytr, cand_tr)
    pred_in, pred, pred_is = M.decide(Pin, cand_tr, tau), M.decide(P, cand_te, tau), M.decide(Pis, cand_tr, tau)
    d["info"]["tau"] = tau
    if c["unsure_head"] != "none":
        col = f"{c['unsure_head']}_uns"
        h_in, h_te = Ntr[col].to_numpy(), Nte[col].to_numpy()
        best = ((pred_in == ytr).mean(), None)
        for th in np.arange(0.3, 0.951, 0.05):
            for tc in np.arange(0.55, 0.951, 0.05):
                pr = np.where((h_in >= th) & (Pin[:, :2].max(1) < tc), 2, pred_in)
                acc = (pr == ytr).mean()
                if acc > best[0] + 1e-9:
                    best = (acc, (th, tc))
        if best[1] is not None:
            th, tc = best[1]
            pred = np.where((h_te >= th) & (P[:, :2].max(1) < tc), 2, pred)
            pred_is = np.where((h_in >= th) & (Pis[:, :2].max(1) < tc), 2, pred_is)
        d["info"]["unsure_head_t"] = best[1]
    d["info"]["train_f1"] = M.metrics(ytr, Pis, pred_is)["macro_f1_tf"]
    d["info"]["retrain_s"] = time.time() - t0
    d["info"]["predict_s_cpu2"] += extra_cpu_s(c)
    return P, pred, d["info"], d["yte"], d["te"]
