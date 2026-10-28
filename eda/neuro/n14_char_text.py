"""Item 14: character-level BiGRU and CNN on the raw sentence, trained from scratch (no pretrained weights),
alone and as ensemble members with LightGBM (probability average, weights 0.7/0.3 and 0.85/0.15)."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402
from nets import CharBiGRU, CharCNN, char_encode  # noqa: E402


def main():
    t0 = time.time()
    D = C.load()
    R, y, folds, Xf = D["R"], D["y"], D["folds"], D["Xf"]
    T, vocab = char_encode(R["text"].fillna("").tolist())
    names = ["n14_lgbm_reference", "n14_char_bigru", "n14_char_cnn", "n14_lgbm70_bigru30", "n14_lgbm85_bigru15",
             "n14_lgbm70_cnn30", "n14_lgbm85_cnn15"]
    probs = {n: {} for n in names}
    gpu = {"bigru": 0.0, "cnn": 0.0}
    cpu2 = {}
    for k, (tr, te) in folds.items():
        X = Xf[k].values.astype(np.float32)
        P_l = C.lgbm().fit(X[tr], y[tr]).predict_proba(X[te])
        probs["n14_lgbm_reference"][k] = P_l
        for tag, net in [("bigru", CharBiGRU(vocab)), ("cnn", CharCNN(vocab))]:
            m, fs = C.train_dev(net, [T[tr]], y[tr], epochs=25, lr=2e-3, weight=False, seed=0, budget_s=60)
            gpu[tag] += fs
            P = C.predict_dev(m, [T[te]])
            probs[f"n14_char_{tag}"][k] = P
            for wl in (70, 85):
                probs[f"n14_lgbm{wl}_{tag}{100 - wl}"][k] = (wl * P_l + (100 - wl) * P) / 100
            if tag not in cpu2:
                cpu2[tag] = C.cpu2_seconds(m, [T[te][:1000]])
        print(f"  fold {k} done ({time.time() - t0:.0f}s)", flush=True)
    rows = []
    for name in names:
        tag = "bigru" if "bigru" in name else "cnn" if "cnn" in name else None
        extra = {"predict_s_cpu2": round(cpu2[tag], 2), "gpu_fit_s": round(gpu[tag], 1)} if tag else {}
        row, _, _ = C.oof_eval(name, y, folds, probs[name], extra)
        rows.append(row)
        print(name, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in row.items()}, flush=True)
    C.save_rows(rows, "n14_")


if __name__ == "__main__":
    main()
