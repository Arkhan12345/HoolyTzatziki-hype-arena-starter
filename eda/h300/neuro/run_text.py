"""Character-level text classifiers (BiGRU, CNN), 3-class, trained with synthetic augmentation weight w.

R1 (practice mode):  train = practice training folds + practice-rule synthetic rows (weight w)
                     validate = practice fold (out of fold), h300 cv rows (pure validation, no h300 label used),
                     the held-out 20% of the synthetic rows per case_type.
R2 (closed-world):   train = h300 cv rows of the other subject-grouped folds + h300 train_only rows
                     + closed-world synthetic rows (weight w); validate = the held-out h300 fold (subjects unseen).
                     Practice rows are scored too (trade-off), never trained on.
w is never chosen on the fold it is scored on (nested choice in report.py). Outputs: out/text_probs.pkl.

    .venv-neuro/Scripts/python eda/h300/neuro/run_text.py [--toy] [--seeds 0] [--archs bigru,cnn] [--perm]
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_h as K  # noqa: E402
from common_h import C  # noqa: E402

sys.path.insert(0, str(K.ROOT / "eda" / "neuro"))
from nets import CharBiGRU, CharCNN  # noqa: E402

WEIGHTS = [0.0, 0.3, 1.0]
MAX_LEN = 160


def encoder(texts):
    chars = sorted({c for s in texts for c in s})
    vocab = {c: i + 2 for i, c in enumerate(chars)}

    def enc(xs):
        arr = np.zeros((len(xs), MAX_LEN), np.int64)
        for n, s in enumerate(xs):
            ids = [vocab.get(c, 1) for c in str(s)[:MAX_LEN]]
            arr[n, :len(ids)] = ids
        return arr
    return enc, len(vocab) + 2


def fit(arch, vocab, X, y, sw, seed, epochs=14, budget_s=60):
    """Sample-weighted cross-entropy with sqrt-balanced class weights (full balancing over-predicts unsure)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = (CharBiGRU(vocab) if arch == "bigru" else CharCNN(vocab)).to(K.DEVICE)
    counts = np.bincount(y, minlength=3).astype(float)
    cw = torch.tensor(np.sqrt(len(y) / (3 * np.maximum(counts, 1))), dtype=torch.float32, device=K.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    Xt, yt = torch.as_tensor(X, device=K.DEVICE), torch.as_tensor(y, device=K.DEVICE)
    wt = torch.as_tensor(sw, dtype=torch.float32, device=K.DEVICE)
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(y), generator=g).to(K.DEVICE)
        for b in range(0, len(y), 256):
            i = perm[b:b + 256]
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(Xt[i]), yt[i], weight=cw, reduction="none")
            (loss * wt[i]).sum().div(wt[i].sum()).backward()
            opt.step()
        if time.time() - t0 > budget_s:
            break
    if K.DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return model, time.time() - t0


def predict(model, X):
    return C.predict_dev(model, [X])


def toy_synth(train, mode, seed):
    """Code-path test only: practice training rows relabelled as 'synthetic' (never used for results)."""
    rng = np.random.default_rng(seed)
    S = train.sample(min(300, len(train)), random_state=seed).copy()
    S["case_type"] = "toy"
    hold = rng.random(len(S)) < 0.2
    return S[~hold].reset_index(drop=True), S[hold].reset_index(drop=True), "toy"


def run(args):
    P, Hh = K.practice(), K.h300()
    P_cv = P[P.fold >= 0].reset_index(drop=True)
    form_by_id = pd.read_csv(K.ROOT / "data" / "practice.csv").set_index("id")["form"]
    probe_s = {}
    for mode in ("practice", "closed_world"):
        tr, va, src = K.synth(mode, None) if not args.toy else (None, None, None)
        probe_s[mode] = (tr, va, src)
    texts = list(P["text"]) + list(Hh["text"])
    for mode, (tr, va, _) in probe_s.items():
        if tr is not None:
            texts += list(tr["text"]) + list(va["text"])
    enc, vocab = encoder(texts)
    res = {"meta": {"toy": args.toy, "weights": WEIGHTS, "archs": args.archs, "seeds": args.seeds,
                    "perm": args.perm, "device": K.DEVICE.type}, "R1": {}, "R2": {}}
    XP, XH = enc(P_cv["text"]), enc(Hh["text"])
    h_cv = (Hh["part"] == "cv").values
    t_all = time.time()

    # ------------------------------------------------------------------ R1: practice-trained
    for arch in args.archs:
        for seed in args.seeds:
            for w in WEIGHTS if not args.perm else [args.perm_w]:
                key = (arch, w, seed)
                oof = np.zeros((len(P_cv), 3))
                h_probs, s_rows, fits, gaps = [], [], [], []
                for k in range(5):
                    trm, vam = P_cv.fold.values != k, P_cv.fold.values == k
                    val_pairs = K.gold_pairs(pd.DataFrame({"form": form_by_id.reindex(P_cv.id[vam]).values}))
                    if args.toy:
                        s_tr, s_va, src = toy_synth(P_cv[trm], "practice", seed + k)
                    else:
                        s_tr, s_va, src = K.synth("practice", k, val_pairs, seed=k)
                    y_tr = P_cv.y.values[trm].copy()
                    X_tr, sw = XP[trm], np.ones(trm.sum())
                    if w > 0 and s_tr is not None:
                        X_tr = np.concatenate([X_tr, enc(s_tr["text"])])
                        y_tr = np.concatenate([y_tr, s_tr["y"].values])
                        sw = np.concatenate([sw, np.full(len(s_tr), w)])
                    if args.perm:
                        y_tr = np.random.default_rng(100 + k).permutation(y_tr)
                    m, fs = fit(arch, vocab, X_tr, y_tr, sw, seed)
                    fits.append(fs)
                    oof[vam] = predict(m, XP[vam])
                    tr_s = K.scores(P_cv.y.values[trm], predict(m, XP[trm]))
                    gaps.append(tr_s["f1_tf"])
                    h_probs.append(predict(m, XH))
                    if s_va is not None and not args.toy:
                        s_rows.append((s_va, predict(m, enc(s_va["text"]))))
                    if k == 0 and seed == args.seeds[0] and not args.perm:
                        res.setdefault("cpu2_s", {})[arch] = C.cpu2_seconds(m, [XH[:256]])
                Ph = np.mean(h_probs, 0)
                out = {"practice": K.scores(P_cv.y.values, oof), "practice_fold": {},
                       "h300": K.scores(Hh.y.values[h_cv], Ph[h_cv]), "train_f1_tf_mean": float(np.mean(gaps)),
                       "fit_s": float(np.sum(fits)), "oof": oof, "h300_probs": Ph}
                for k in range(5):
                    vm = P_cv.fold.values == k
                    out["practice_fold"][k] = K.scores(P_cv.y.values[vm], oof[vm])
                if s_rows:
                    S = pd.concat([s for s, _ in s_rows], ignore_index=True)
                    PS = np.concatenate([p for _, p in s_rows])
                    out["synth_val"] = {"scores": K.scores(S["y"].values, PS), "per_case": K.per_case(S, PS),
                                        "source": src}
                res["R1"][key] = out
                print(f"R1 {arch} w={w} seed={seed}{' PERM' if args.perm else ''}: practice f1_tf "
                      f"{out['practice']['f1_tf']:.3f} acc {out['practice']['acc']:.3f} | h300 f1_3 "
                      f"{out['h300']['f1_3']:.3f} | fit {out['fit_s']:.0f}s ({time.time() - t_all:.0f}s)", flush=True)

    # ------------------------------------------------------------------ R2: inside h300 (subject-grouped)
    H_cv = Hh[h_cv].reset_index(drop=True)
    H_to = Hh[(Hh.part == "train_only").values].reset_index(drop=True)
    XHc, XHt = enc(H_cv["text"]), enc(H_to["text"])
    if args.toy:
        sc_tr, sc_va, src = toy_synth(P_cv, "closed_world", 7)
    else:
        sc_tr, sc_va, src = K.synth("closed_world", None)
    for arch in args.archs:
        for seed in args.seeds:
            for w in WEIGHTS if not args.perm else [args.perm_w]:
                key = (arch, w, seed)
                oof = np.zeros((len(H_cv), 3))
                p_probs, s_preds, fits, gaps = [], [], [], []
                for j in range(5):
                    trm, vam = H_cv.fold.values != j, H_cv.fold.values == j
                    X_tr = np.concatenate([XHc[trm], XHt])
                    y_tr = np.concatenate([H_cv.y.values[trm], H_to.y.values])
                    sw = np.ones(len(y_tr))
                    if w > 0 and sc_tr is not None:
                        X_tr = np.concatenate([X_tr, enc(sc_tr["text"])])
                        y_tr = np.concatenate([y_tr, sc_tr["y"].values])
                        sw = np.concatenate([sw, np.full(len(sc_tr), w)])
                    if args.perm:
                        y_tr = np.random.default_rng(200 + j).permutation(y_tr)
                    m, fs = fit(arch, vocab, X_tr, y_tr, sw, seed, epochs=25)
                    fits.append(fs)
                    oof[vam] = predict(m, XHc[vam])
                    gaps.append(K.scores(H_cv.y.values[trm], predict(m, XHc[trm]))["f1_3"])
                    p_probs.append(predict(m, XP))
                    if sc_va is not None and not args.toy:
                        s_preds.append(predict(m, enc(sc_va["text"])))
                out = {"h300": K.scores(H_cv.y.values, oof), "h300_fold": {},
                       "practice": K.scores(P_cv.y.values, np.mean(p_probs, 0)),
                       "train_f1_3_mean": float(np.mean(gaps)), "fit_s": float(np.sum(fits)), "oof": oof}
                for j in range(5):
                    vm = H_cv.fold.values == j
                    out["h300_fold"][j] = K.scores(H_cv.y.values[vm], oof[vm])
                if s_preds:
                    PS = np.mean(s_preds, 0)
                    out["synth_val"] = {"scores": K.scores(sc_va["y"].values, PS), "per_case": K.per_case(sc_va, PS),
                                        "source": src}
                res["R2"][key] = out
                print(f"R2 {arch} w={w} seed={seed}{' PERM' if args.perm else ''}: h300 f1_3 "
                      f"{out['h300']['f1_3']:.3f} | practice acc {out['practice']['acc']:.3f} | fit "
                      f"{out['fit_s']:.0f}s ({time.time() - t_all:.0f}s)", flush=True)
    res["ids"] = {"practice": list(P_cv.id), "h300": list(Hh.id), "h300_cv": list(H_cv.id)}
    name = "text_probs_toy.pkl" if args.toy else ("text_perm.pkl" if args.perm else
                                                  f"text_probs_s{'-'.join(map(str, args.seeds))}.pkl")
    pickle.dump(res, open(K.OUT / name, "wb"))
    print(f"saved {name} ({time.time() - t_all:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--archs", default="bigru,cnn")
    ap.add_argument("--perm", action="store_true", help="label-permutation run of one weight")
    ap.add_argument("--perm-w", type=float, default=0.0)
    a = ap.parse_args()
    a.seeds = [int(s) for s in a.seeds.split(",")]
    a.archs = a.archs.split(",")
    run(a)
