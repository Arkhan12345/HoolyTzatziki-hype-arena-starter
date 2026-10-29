"""Make every saved model loadable from the repo root and its manifest complete; drop models that overfit.

- Re-pickles each model so its classes live under h300.lib.* (importable with eda/ on sys.path), then
  predict_entry = "h300.lib.model:predict_rows" (called as predict_rows(model, rows, lexicon)).
- Adds the training ids file, seeds, the learned-part permutation maximum, the train - validation gap and the h300
  lockbox look counter; recomputes sha256.
- Removes a ladder model whose train - validation gap is above 0.10 (it is not a good model by our own gate);
  the frozen FINAL / FALLBACK are kept and their checks are reported as they are.
"""
import importlib
import json
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "eda"))
h300lib = importlib.import_module("h300.lib")
for sub in ("values", "extractor", "closed_world", "feats", "model"):
    m = importlib.import_module(f"h300.lib.{sub}")
    sys.modules[f"lib.{sub}"] = m
sys.modules["lib"] = h300lib
sys.path.insert(1, str(HERE))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import run  # noqa: E402

MODELS = ROOT / "eda" / "models"
IDS = MODELS / "training_ids"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    IDS.mkdir(parents=True, exist_ok=True)
    D = run.load("script")
    H, P, outer, S10 = D
    hc = H[H.part.isin(["cv", "train_only"])].id.tolist()
    sp = pd.read_csv(ROOT / "eda" / "splits.csv")
    all_practice = sp.id.tolist()
    ids_for = {"R1": P.id.tolist(), "P": P.id.tolist(), "R2": P.id.tolist() + hc, "R3": hc}
    removed, done = [], []
    for mp in sorted(MODELS.glob("*.manifest.json")):
        m = json.loads(mp.read_text(encoding="utf-8"))
        jf = ROOT / m["file"]
        regime = m["regime"]
        cfg = {k: (tuple(v) if k == "synth" and isinstance(v, list) else v) for k, v in m["hyperparams"].items()}
        dreg = "P" if regime in ("R1", "P") else regime
        km = "ftf" if dreg == "P" else "f3"
        if m.get("synth_version", "script") == "claude":
            run.SYNTH_VERSION = "claude"
            Dg = run.load("claude")
        else:
            run.SYNTH_VERSION = "script"
            Dg = D
        _, pooled, ins = run.eval_regime(dreg, cfg, Dg, in_sample=True)
        gap = float(np.mean(ins) - pooled[km])
        if gap > 0.10:
            jf.unlink(missing_ok=True)
            mp.unlink()
            removed.append((m["name"], round(gap, 4)))
            continue
        model = joblib.load(jf)
        joblib.dump(model, jf, compress=3)
        idf = IDS / f"{m['name']}.csv"
        pd.DataFrame({"id": ids_for[regime]}).to_csv(idf, index=False)
        g = m.get("gates", {})
        looks = g.pop("lockbox_looks", 0)
        m.update(gates=g, lockbox_looks=looks, seeds=list(run.SEEDS), training_ids_file=str(idf.relative_to(ROOT)),
                 permutation_learned_max=max(m["scores"]["permutation"]), train_val_gap=round(gap, 4),
                 train_val_gap_domain="practice CV (true/false macro-F1)" if dreg == "P" else "h300 CV (3-class)",
                 predict_entry="h300.lib.model:predict_rows", sha256=sha(jf), file=str(jf.relative_to(ROOT)))
        mp.write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
        done.append((m["name"], round(gap, 4)))
    for name in ("FINAL", "FALLBACK"):
        d = MODELS / name
        if not (d / "manifest.json").exists():
            continue
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        model = joblib.load(d / "model.joblib")
        joblib.dump(model, d / "model.joblib", compress=3)
        idf = IDS / f"{name}.csv"
        pd.DataFrame({"id": all_practice}).to_csv(idf, index=False)
        gc = m["generalisation_checks"]
        m["gates"] = {f"{k}_pass": bool(v.get("pass", True)) for k, v in gc.items() if isinstance(v, dict)}
        m.update(lockbox_looks=0, practice_holdout_looks=2, seeds=[0, 1, 2],
                 training_ids_file=str(idf.relative_to(ROOT)),
                 training_note="refitted once on every labelled practice row incl. the holdout pages, after their "
                               "look (by design of the freeze step)",
                 permutation_learned_max=max(gc["permutation"]["values"]), train_val_gap=gc["train_validation_difference"]["gap"],
                 predict_entry="h300.lib.model:predict_rows", sha256=sha(d / "model.joblib"),
                 file=str((d / "model.joblib").relative_to(ROOT)))
        (d / "manifest.json").write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
        done.append((name, gc["train_validation_difference"]["gap"]))
    print("kept", done)
    print("removed (train-val gap > 0.10)", removed)
    json.dump({"kept": done, "removed": removed}, open(HERE / "out" / "saved_models.json", "w"), indent=1)


if __name__ == "__main__":
    main()
