"""Write eda/h300/REPORT.md from the run outputs (ladder, summaries for every synthetic version, freeze, synth stats)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
LAB = ["true", "false", "unsure"]


def score_baseline(name, path, hs):
    p = pd.read_csv(path).merge(hs, on="id", suffixes=("", "_gold"))
    p = p[p.part == "cv"]
    y, q = p.label_gold.values if "label_gold" in p else p.label.values, p.label.values
    f = []
    for c in LAB:
        tp = ((q == c) & (y == c)).sum()
        f.append(2 * tp / max(2 * tp + ((q == c) & (y != c)).sum() + ((q != c) & (y == c)).sum(), 1))
    return f"| {name} | {np.mean(q == y):.3f} | {np.mean(f):.3f} | {f[2]:.3f} |"


def ladder_table(df, regime):
    d = df[df.regime == regime]
    if d.empty:
        return ["(not run)"]
    km = "ftf" if regime == "P" else "f3"
    lines = [f"| step | {'true/false macro-F1' if regime == 'P' else '3-class macro-F1'} | acc | unsure F1 | ECE | "
             f"gain | folds better | gates a/b/c/d | verdict |", "|---|---|---|---|---|---|---|---|---|"]
    for r in d.itertuples():
        gates = "/".join("✓" if getattr(r, f"gate_{g}", None) is True else ("✗" if getattr(r, f"gate_{g}", None)
                                                                              is False else "–")
                         for g in ("a_nested", "b_gap", "c_seeds", "d_subjects")) if hasattr(r, "gate_a_nested") else "–"
        lines.append(f"| {r.step} | {getattr(r, km):.3f} | {r.acc:.3f} | {r.f_unsure:.3f} | {r.ece:.3f} | "
                     f"{'' if pd.isna(getattr(r, 'gain', np.nan)) else f'{r.gain:+.3f}'} | "
                     f"{'' if pd.isna(getattr(r, 'folds_better', np.nan)) else int(r.folds_better)} | {gates} | "
                     f"{r.kept} |")
    return lines


def main():
    hs = pd.read_csv(OUT / "h300_splits.csv")
    L = ["# h300 experiments: honest training, synthetic validation, frozen models", "",
         "All numbers below are on **extracted facts** (every scored row is read from its text; practice gold forms only "
         "build training-fold KB anchors). Every fold is refitted from scratch without its validation pages.", "",
         "## Protocol", "",
         "- **Practice:** 1,593 CV rows in 5 page-grouped folds (`eda/splits.csv`); 304 holdout rows kept out until the "
         "single look at freeze time. Per outer fold a reader (extractor, attribute model, KB) is fitted on the 4 "
         "training folds; training-row features come from 4 inner refits (out of fold), so train and validation rows "
         "are read the same way.",
         f"- **h300:** {len(hs)} statements; {int(hs.dedupe.sum())} near-copies of practice text (ratio ≥ 90, "
         "`out/h300_dedupe.csv`) are train-only; the rest is split by subject (no subject on both sides) and label into "
         f"a lockbox of {int((hs.part == 'lockbox').sum())} rows (scored once per regime and synthetic version) and "
         f"5 CV folds of {int((hs.part == 'cv').sum())} rows; a second subject grouping backs the held-out-subjects gate.",
         "- **Regimes:** R1 practice only → h300 folds · R2 practice + other h300 folds → held-out fold · R3 h300 only → "
         "held-out fold · P practice CV (the arena's label rule, true/false macro-F1).",
         "- **Accept rule + gates** (every kept piece): gain ≥ 0.01 (P: 0.005), better on ≥ 4/5 folds, ECE not worse by "
         "> 0.02; a nested selection, b train−validation gap ≤ +0.03, c sign over 3 seeds, d second subject grouping.",
         ""]
    L += ["## Baselines on the h300 CV rows", "", "| model | acc | 3-class macro-F1 | unsure F1 |", "|---|---|---|---|"]
    gold = pd.read_csv(HERE.parents[1] / "data" / "hooly_statements_300.csv")[["id", "label"]].rename(
        columns={"label": "label_gold"})
    hs2 = hs.merge(gold, on="id")
    all_true = pd.DataFrame({"id": hs.id, "label": "true"})
    all_true.to_csv(OUT / "baseline_all_true.csv", index=False)
    for nm, f in (("all true", "baseline_all_true.csv"), ("shipped option A stack", "baseline_shipped.csv"),
                  ("closed-world labeller (tuned on these statements' dev split)", "baseline_closed.csv")):
        if (OUT / f).exists():
            L.append(score_baseline(nm, OUT / f, hs2[["id", "part", "label_gold"]]))
    L.append("")
    for ver in ("script", "claude", "script_strong"):
        d = OUT if ver == "script" else OUT / f"synth_{ver}"
        if not (d / "ladder.csv").exists():
            continue
        df = pd.read_csv(d / "ladder.csv")
        s = json.load(open(d / "summary.json")) if (d / "summary.json").exists() else {}
        L += [f"## Ladder with synthetic version `{ver}`", ""]
        for regime in ("R1", "R2", "R3", "P"):
            L += [f"### {regime}", ""] + ladder_table(df, regime) + [""]
            e = s.get("regimes", {}).get(regime)
            if e:
                L += [f"- kept: {', '.join(e['kept_steps']) or 'base only'}; permutation {e['permutation']} "
                      f"({'pass' if e['permutation_pass'] else 'FAIL'}); lockbox {e.get('lockbox')}; "
                      f"saved: {e.get('saved', 'no')}", ""]
        if s.get("deflation"):
            L += [f"Deflation: {s['deflation']}", ""]
    sm = OUT / "saved_models.json"
    if sm.exists():
        d = json.load(open(sm))
        L += ["## Saved models (eda/models/)", "",
              "Kept only when the train - validation gap is <= 0.10 (our own overfitting gate); every kept model is "
              "pickled under `h300.lib.*` and loads with `predict_entry = h300.lib.model:predict_rows`.", "",
              f"- kept: {d['kept']}", f"- removed (gap > 0.10): {d['removed']}", ""]
    lb = OUT / "lockbox_log.json"
    if lb.exists():
        L += ["## Every h300 lockbox look (same 57 rows)", "", "| seq | regime | version | acc | 3-class F1 |",
              "|---|---|---|---|---|"]
        for e in json.load(open(lb)):
            L.append(f"| {e.get('log_sequence', '')} | {e['regime']} | {e.get('synth')} | {e['acc']} | {e['f3']} |")
        L += ["", "Each (regime, version) looked once; across versions the same rows were seen up to 3 times per "
              "regime, so the lockbox is weaker than a single look and its numbers are reported, not used to choose.", ""]
    st = OUT / "synth10k_stats.json"
    if st.exists():
        L += ["## Synthetic data (script version)", "", "```json", st.read_text(), "```", ""]
    fr = OUT / "freeze_report.json"
    if fr.exists():
        L += ["## Frozen models", "", "```json", fr.read_text(encoding="utf-8")[:6000], "```", ""]
    (HERE / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote", HERE / "REPORT.md")


if __name__ == "__main__":
    main()
