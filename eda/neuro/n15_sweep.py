"""Item 15: sweep of the timeline classifier: cell {CfC, LTC, GRU} x hidden {16, 32, 64} x day-gap encoding
{raw, log, none}, 3 seeds x the 5 group folds (mean +- std over seeds). Liquid cells get the real gaps as timespans.
Usage: python eda/neuro/n15_sweep.py [--seeds 3] [--cells cfc,ltc,gru]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402
from nets import RowNet, encode_dt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cells", default="cfc,ltc,gru")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    t0 = time.time()
    D = C.load()
    y, folds, Xf = D["y"], D["folds"], D["Xf"]
    scaled = {k: C.tab_scaled(Xf[k].values[tr], Xf[k].values[te]) for k, (tr, te) in folds.items()}
    rows, per_seed = [], []
    for cell in a.cells.split(","):
        for hidden in (16, 32, 64):
            for dt in ("raw", "log", "none"):
                SEQ, TS = encode_dt(D["SEQ"], dt)
                name = f"n15_{cell}_h{hidden}_{dt}"
                seed_rows, fit_total, cpu2 = [], 0.0, None
                for seed in range(a.seeds):
                    probs = {}
                    for k, (tr, te) in folds.items():
                        xa, xb = scaled[k]
                        m, fs = C.train_dev(RowNet(cell, SEQ.shape[2], xa.shape[1], hidden), [SEQ[tr], TS[tr], xa],
                                            y[tr], epochs=a.epochs, batch=a.batch, weight=False, seed=seed, budget_s=30)
                        fit_total += fs
                        probs[k] = C.predict_dev(m, [SEQ[te], TS[te], xb])
                        if cpu2 is None:
                            cpu2 = C.cpu2_seconds(m, [SEQ[te], TS[te], xb])
                    r, _, _ = C.oof_eval(name, y, folds, probs)
                    r["seed"] = seed
                    seed_rows.append(r)
                    per_seed.append(r)
                S = pd.DataFrame(seed_rows)
                row = {"variant": name, "cv": f"group5_splits_x{a.seeds}seeds", "device": C.DEVICE.type,
                       "predict_s_cpu2": round(cpu2, 2), "gpu_fit_s": round(fit_total, 1)}
                for c in ["acc", "macro_f1_tf", "ece", "unsure_recall", "false_precision"]:
                    row[c] = S[c].mean()
                    row[f"{c}_std"] = S[c].std()
                rows.append(row)
                print(f"{name}: macro-F1 {row['macro_f1_tf']:.4f}+-{row['macro_f1_tf_std']:.4f} acc {row['acc']:.4f}"
                      f" cpu2 {cpu2:.1f}s ({time.time() - t0:.0f}s)", flush=True)
                C.save_rows(rows, "n15_")
    pd.DataFrame(per_seed).round(4).to_csv(C.OUT / "neuro_sweep_seeds.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.DataFrame(rows).sort_values("macro_f1_tf")
    fig, ax = plt.subplots(figsize=(7, max(4, 0.25 * len(df))))
    ax.barh(df["variant"].str.replace("n15_", ""), df["macro_f1_tf"], xerr=df["macro_f1_tf_std"], color="#3b6ea5")
    ax.axvline(0.762, color="#c0392b", ls="--", lw=1, label="LightGBM (bake-off) 0.762")
    ax.set_xlabel("macro-F1 (true/false), 5 group folds, mean +- std over seeds")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(C.OUT / "figs" / "neuro_sweep.png", dpi=110)


if __name__ == "__main__":
    main()
