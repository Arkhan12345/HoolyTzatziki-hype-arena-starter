"""Step 10: one table of everything tested so far -> docs/RESULTS.md

Reads every results file in eda/out/ (and eda/selfimprove/trials.csv when it exists) and writes a single markdown
table: stream | model/variant | CV scheme | acc | macro-F1 | ECE | unsure recall | CPU predict 20k | verdict | source.
Rerun after any experiment: .venv/Scripts/python eda/10_results_table.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eda" / "out"
DOC = ROOT / "docs" / "RESULTS.md"
NOISE = 0.01  # macro-F1 band treated as "no change" (fold std of the reference is ~0.004-0.006)

STREAMS = [
    ("1 · bake-off", "bakeoff.csv"),
    ("2 · label ladder", "improve_labels.csv"),
    ("3 · weak labels", "improve_weak.csv"),
    ("4 · neural (GPU)", "improve_neuro.csv"),
]


def fmt(x, pct=False, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:.1%}" if pct else f"{x:.{nd}f}"


def cpu20k(row, n_rows):
    """predict_s_cpu2 is per scored partition in most files; scale to 20,000 rows when the partition size is known."""
    s = row.get("predict_s_cpu2")
    if s is None or (isinstance(s, float) and np.isnan(s)):
        s = row.get("predict_s")
        if s is None or (isinstance(s, float) and np.isnan(s)):
            return "–"
    return f"{float(s):.2f} s" if not n_rows else f"{float(s) * 20000 / n_rows:.1f} s"


def load(stream, fname):
    p = OUT / fname
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p)
    if "model" in d.columns and "variant" not in d.columns:
        d = d.rename(columns={"model": "variant"})
    d["stream"] = stream
    d["source"] = f"`eda/out/{fname}`"
    return d


def verdict(stream, r, ref):
    v, cv, f1 = r["variant"], str(r["cv"]), r.get("macro_f1_tf")
    if "all_true" in v or v == "majority_all_true":
        return "baseline"
    if stream.startswith("2") and v.startswith("L6") and cv == "fold_group":
        return "**current best** (calibrated)"
    if cv == "holdout_once":
        return "holdout, scored once (43 pages: ±0.03 noise)"
    if "extracted" in v:
        return "real extraction: −0.05 vs gold facts"
    if ref is None or f1 is None or (isinstance(f1, float) and np.isnan(f1)):
        return ""
    d = f1 - ref
    if abs(d) < 1e-9:
        return "reference"
    if d > NOISE:
        return f"better (+{d:.3f})"
    if d < -NOISE:
        return f"worse ({d:.3f})"
    return f"no change ({d:+.3f})"


def reference(stream, d, cv):
    """Reference row per stream and CV scheme: the plain LightGBM / L0 on the same folds."""
    sub = d[d["cv"] == cv]
    for name in ["lightgbm_unweighted", "L0_base", "lgbm_plain", "n14_lgbm_reference", "n13_lgbm_reference",
                 "n12_lgbm_reference"]:
        hit = sub[sub["variant"] == name]
        if len(hit):
            return float(hit["macro_f1_tf"].iloc[0])
    if cv.startswith("group5_splits") and cv != "group5_splits":   # seed sweeps: same folds as group5_splits
        return reference(stream, d, "group5_splits")
    return None


def main():
    rows = []
    for stream, fname in STREAMS:
        d = load(stream, fname)
        if d.empty:
            continue
        for cv, g in d.groupby("cv", sort=False):
            ref = reference(stream, d, cv)
            for _, r in g.iterrows():
                n = 1593 if cv in ("fold_group", "fold_row5", "group5_splits") else (304 if cv == "holdout_once" else 0)
                rows.append({
                    "stream": stream, "model / variant": f"`{r['variant']}`", "CV scheme": cv,
                    "acc": r.get("acc"), "macro-F1": r.get("macro_f1_tf"), "ECE": r.get("ece"),
                    "unsure recall": r.get("unsure_recall"),
                    "CPU predict 20k": cpu20k(r, 0),
                    "verdict": verdict(stream, r, ref), "source": r["source"]})

    # extraction: fact-level F1 against the gold forms (not a label classifier)
    ex = OUT / "extraction.csv"
    if ex.exists():
        e = pd.read_csv(ex)
        for _, r in e[(e["kind"] == "ALL") & e["fold"].isin(["ALL_FOLDS", "holdout"])].iterrows():
            rows.append({"stream": "5 · extraction", "model / variant": "`simple extractor` (fact F1)",
                         "CV scheme": "fold_group OOF" if r["fold"] == "ALL_FOLDS" else "holdout",
                         "acc": None, "macro-F1": None, "ECE": None, "unsure recall": None, "CPU predict 20k": "–",
                         "verdict": f"fact P {r['precision']:.3f} · R {r['recall']:.3f} · **F1 {r['f1']:.3f}**",
                         "source": "`eda/out/extraction.csv`"})

    # self-improvement loop trials (written by eda/selfimprove/loop.py)
    tr = ROOT / "eda" / "selfimprove" / "trials.csv"
    if tr.exists():
        t = pd.read_csv(tr)
        name = next((c for c in ["change", "trial", "variant", "knob"] if c in t.columns), t.columns[0])
        f1c = next((c for c in ["macro_f1_tf", "mean_macro_f1", "macro_f1"] if c in t.columns), None)
        for _, r in t.iterrows():
            kept = r.get("accepted", r.get("kept", ""))
            rows.append({"stream": "6 · self-improvement", "model / variant": f"`{r[name]}`",
                         "CV scheme": "fold_group", "acc": r.get("acc"), "macro-F1": r.get(f1c) if f1c else None,
                         "ECE": r.get("ece"), "unsure recall": r.get("unsure_recall"), "CPU predict 20k": "–",
                         "verdict": "kept" if str(kept).lower() in ("true", "1", "kept", "yes") else "dropped",
                         "source": "`eda/selfimprove/trials.csv`"})

    T = pd.DataFrame(rows)
    T["_s"] = T["macro-F1"].astype(float).fillna(-1)
    T = T.sort_values(["stream", "CV scheme", "_s"], ascending=[True, True, False]).drop(columns="_s")
    for c in ["acc", "unsure recall"]:
        T[c] = T[c].map(lambda x: fmt(x, pct=True))
    for c in ["macro-F1", "ECE"]:
        T[c] = T[c].map(fmt)

    cols = list(T.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in T.itertuples(index=False)]

    doc = f"""# Results: everything tested so far

*Generated by `eda/10_results_table.py` from `eda/out/*.csv`; do not edit by hand. {len(T)} rows.*

## Read this first
- **Metric:** macro-F1 over the true/false classes (unsure counts as wrong for both); accuracy over all three labels;
  ECE = calibration error of the predicted label's probability (lower is better).
- **Baselines (all true):** **83.3%** accuracy on all 1,897 practice rows; **85.3%** on the 380 rows `scripts/check.sh`
  hides (`fold_row5 == 0`). Macro-F1 of all-true is 0.455.
- **Current best: L6** (staircase timeline + page trust + failure statuses + hard rules + evidence-sum calibrator):
  macro-F1 **0.801**, ECE **0.050** on `fold_group`, vs 0.735 for the plain LightGBM on the same folds. The gain of the
  ladder (L4 vs L0) is +0.071, page-bootstrap 95% CI [+0.042, +0.103]. On the 43 holdout pages the result is
  inconclusive (−0.011, CI [−0.074, +0.057]).
- **Neural (liquid / spiking / character models):** no gain on true/false over LightGBM; useful for calibration
  (stack ECE 0.030) and unsure recall (CfC head 25%).
- **Facts:** every row except the extraction rows uses the gold `form` (perfect extraction). With our simple
  extractor, L4 falls to 0.858 / 0.752.

## CV schemes (compare rows only within the same scheme)
| scheme | rows scored | what it is |
|---|---|---|
| `group` / `groupkfold_all` | 1,897 | GroupKFold(5) by page over all rows (the first bake-off) |
| `fold_group` / `group5_splits` | 1,593 | `eda/splits.csv` StratifiedGroupKFold by page, holdout pages excluded (main CV) |
| `fold_row5` / `row5` | 1,593 / 1,897 | every 5th row hidden, like `scripts/check.sh` (pages shared between train and test) |
| `time` | 744 | train before 2026-07-01, test after (pessimistic) |
| `holdout_once` | 304 | 43 frozen pages, scored once per final candidate |
| `*_x3seeds` / `*_x5lgbm_seeds` | 1,593 | same folds, mean over 3 or 5 seeds |

"CPU predict 20k" = prediction time on 2 CPU threads as each file reports it: the ladder and neural rows are already
estimates for 20,000 rows; the bake-off and weak-label rows are the time for one test fold (~320 rows).
Verdicts compare macro-F1 with the plain LightGBM (or L0) on the same scheme; ±{NOISE} counts as no change.

## The table
""" + "\n".join(lines) + "\n"
    DOC.write_text(doc, encoding="utf-8")
    print(f"wrote {DOC} ({len(T)} rows)")


if __name__ == "__main__":
    main()
