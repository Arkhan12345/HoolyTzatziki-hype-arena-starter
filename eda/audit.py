"""Research audit: leakage, reproducibility, determinism, dependency pins, runtime gate, multiple testing, claims.

Writes docs/AUDIT.md (PASS / FAIL / INFO table + details). Read-only on the committed outputs: the
reproducibility check reruns the EDA scripts in a temporary copy of the repo and compares what they write.

Run:  .venv/Scripts/python eda/audit.py            (~2 min)
"""
import hashlib
import importlib
import importlib.metadata as md
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
EDA, OUT = ROOT / "eda", ROOT / "eda" / "out"
sys.path.insert(0, str(EDA))
sys.path.insert(0, str(ROOT))

GATES = {"retrain_s": 10.0, "predict_20k_s": 5.0, "neural_predict_20k_s": 120.0}
RESULTS = []  # (check, status, summary, details)


def record(check, ok, summary, details=""):
    status = ok if isinstance(ok, str) else ("PASS" if ok else "FAIL")
    RESULTS.append((check, status, summary, details))
    print(f"[{status}] {check}: {summary}", flush=True)


# ------------------------------------------------------------------ 1. leakage
def leakage(n_folds=2):
    imp = importlib.import_module("07_improve")
    cv_ids = list(imp.SPLITS.index[imp.SPLITS["holdout"] == 0])
    folds = imp.scheme_folds("fold_group", cv_ids)[:n_folds]
    flip = {"true": "false", "false": "true", "unsure": "true"}
    keep = (imp.LABELS, imp.R, imp.F_ALL)
    lines, ok_all, cached = [], True, []
    try:
        for k, (train, test, inner) in enumerate(folds):
            X1, _ = imp.fold_table(imp.F_ALL, train, test, inner)
            labels2 = dict(keep[0])
            for i in test:
                labels2[i] = flip[labels2[i]]
            R2 = keep[1].copy()
            R2.loc[R2["id"].isin(test), "label"] = R2.loc[R2["id"].isin(test), "label"].map(flip)
            F2 = keep[2].copy()
            F2.loc[F2["id"].isin(test), "label"] = F2.loc[F2["id"].isin(test), "label"].map(flip)
            imp.LABELS, imp.R, imp.F_ALL = labels2, R2, F2
            X2, _ = imp.fold_table(F2, train, test, inner)
            imp.LABELS, imp.R, imp.F_ALL = keep
            same = X1.sort_index().equals(X2.sort_index())
            if not same:
                diff = (X1.sort_index().fillna(-999) != X2.sort_index().fillna(-999))
                cols = list(diff.columns[diff.any()])[:8]
                lines.append(f"fold {k}: features CHANGED when test labels were flipped: {cols}")
            else:
                lines.append(f"fold {k}: {len(test)} test rows flipped, {X1.shape[1]} features on "
                             f"{len(X1)} rows identical")
            ok_all &= same
            cached.append((X1, train, test, inner))
    finally:
        imp.LABELS, imp.R, imp.F_ALL = keep
    record("Leakage: test labels never reach features", ok_all,
           f"{len(folds)} fold_group folds, test labels flipped, every feature unchanged" if ok_all
           else "a feature depends on test labels", "\n".join(lines))
    return imp, cached


# ------------------------------------------------------------------ 2. determinism
def determinism(imp, cached):
    X, train, test, inner = cached[0]
    runs = [imp.run_fold(X, train, test, inner, ["L0_base", "L6_judge"])[0] for _ in range(2)]
    diffs = {v: float(np.abs(runs[0][v][0] - runs[1][v][0]).max()) for v in runs[0]}
    same_pred = all((runs[0][v][1] == runs[1][v][1]).all() for v in runs[0])
    ok = same_pred and max(diffs.values()) < 1e-9
    record("Determinism: same seed, same result", ok,
           "L0_base and L6_judge refit twice on fold 0: identical predictions" if ok
           else f"max probability difference {max(diffs.values()):.2e}",
           "; ".join(f"{v}: max |dP| = {d:.1e}" for v, d in diffs.items()))


# ------------------------------------------------------------------ 3. reproducibility
def reproducibility():
    files = ["rules.csv", "archetypes.csv", "practice_annotated.csv", "templates.csv", "speakers.csv",
             "genres.csv", "attributes.csv"]
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        (t / "eda" / "out").mkdir(parents=True)
        shutil.copytree(ROOT / "data", t / "data")
        shutil.copytree(ROOT / "app", t / "app", ignore=shutil.ignore_patterns("__pycache__"))
        for s in ("01_parse.py", "02_rules.py", "03_recommend.py", "06_splits.py", "features.py"):
            shutil.copy(EDA / s, t / "eda" / s)
        log = []
        for s in ("01_parse.py", "02_rules.py", "03_recommend.py", "06_splits.py"):
            r = subprocess.run([sys.executable, str(t / "eda" / s)], cwd=t, capture_output=True, text=True)
            log.append(f"{s}: exit {r.returncode}")
            if r.returncode:
                record("Reproducibility: EDA outputs rebuild identically", False, f"{s} failed",
                       r.stderr[-1500:])
                return
        diff = []
        pairs = [(OUT / f, t / "eda" / "out" / f) for f in files] + [(EDA / "splits.csv", t / "eda" / "splits.csv")]
        for a, b in pairs:
            ha, hb = hashlib.sha256(a.read_bytes()).hexdigest(), hashlib.sha256(b.read_bytes()).hexdigest()
            if ha != hb:
                same = pd.read_csv(a).equals(pd.read_csv(b))
                diff.append(f"{a.name}: bytes differ, {'table equal' if same else 'TABLE DIFFERS'}")
                if not same:
                    log.append(f"{a.name}: differs")
            else:
                log.append(f"{a.name}: sha256 {ha[:12]} identical")
    ok = not any("TABLE DIFFERS" in d for d in diff)
    record("Reproducibility: EDA outputs rebuild identically", ok,
           f"01, 02, 03, 06 rerun in a temp copy: {len(pairs)} outputs compared, "
           f"{len(pairs) - len(diff)} byte-identical" + (f", {len(diff)} equal as tables" if diff and ok else ""),
           "\n".join(log + diff))


# ------------------------------------------------------------------ 4. dependency pins
def pins():
    want = {}
    for line in (ROOT / "constraints.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)==([^\s;#]+)", line)
        if m:
            want[m.group(1).lower().replace("_", "-")] = m.group(2)
    bad, checked = [], 0
    for name, ver in want.items():
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            continue
        checked += 1
        if have != ver:
            bad.append(f"{name}: installed {have}, constraints {ver}")
    record("Dependency pins: .venv matches constraints.txt", not bad,
           f"{checked} image packages checked" + (f", {len(bad)} mismatched" if bad else ", all identical"),
           "\n".join(bad) or "dev-only extras (pytest-bdd, matplotlib) are outside the image and not compared")


# ------------------------------------------------------------------ 5. runtime gate
def runtime_gate():
    L = pd.read_csv(OUT / "improve_labels.csv")
    lad = L[L["cv"].isin(["fold_group", "fold_row5"]) & L["fit_s"].notna()]
    worst_fit, worst_pred = lad["fit_s"].max(), lad["predict_s_cpu2"].max()
    lines = [f"ladder: max retrain {worst_fit:.2f} s (gate {GATES['retrain_s']} s), "
             f"max predict for 20k rows {worst_pred:.2f} s (gate {GATES['predict_20k_s']} s)"]
    ok = worst_fit <= GATES["retrain_s"] and worst_pred <= GATES["predict_20k_s"]
    p = OUT / "improve_neuro.csv"
    if p.exists():
        N = pd.read_csv(p)
        if "predict_s_cpu2" in N:
            wn = N["predict_s_cpu2"].max()
            lines.append(f"neural: max CPU-2-thread predict for 20k rows {wn:.1f} s "
                         f"(gate {GATES['neural_predict_20k_s']} s)")
            ok &= wn <= GATES["neural_predict_20k_s"]
    record("Runtime gate (2 CPU threads)", ok, lines[0].split(", max predict")[0] + "; predict within gate"
           if ok else "a model breaks the runtime gate", "\n".join(lines))


# ------------------------------------------------------------------ 6. multiple testing
def multiple_testing():
    sources = {"bakeoff.csv": OUT / "bakeoff.csv", "ablation.csv": OUT / "ablation.csv",
               "improve_labels.csv": OUT / "improve_labels.csv", "improve_weak.csv": OUT / "improve_weak.csv",
               "improve_neuro.csv": OUT / "improve_neuro.csv", "neuro_sweep_seeds.csv": OUT / "neuro_sweep_seeds.csv",
               "selfimprove/trials.csv": EDA / "selfimprove" / "trials.csv"}
    def distinct(p):
        """Distinct model configurations in a results file (not rows: folds and seeds repeat a config)."""
        df = pd.read_csv(p)
        keys = [c for c in ("model", "variant", "minus_family", "config", "cell", "hidden", "dt", "dt_encoding",
                            "change", "knob", "value") if c in df.columns]
        return len(df.drop_duplicates(subset=keys)) if keys else len(df)

    counts = {k: distinct(p) for k, p in sources.items() if p.exists()}
    n = sum(counts.values())
    # page bootstrap of the chosen variant against the base, judged at a Bonferroni level for n variants
    with open(OUT / "improve_oof.pkl", "rb") as f:
        oof = pickle.load(f)["fold_group"]
    S = pd.read_csv(EDA / "splits.csv").set_index("id")
    a, b = oof["L0_base"], oof["L6_judge"]
    d = pd.DataFrame({"y": a["y"], "pa": a["pred"]}, index=a["ids"]).join(
        pd.DataFrame({"pb": b["pred"]}, index=b["ids"]), how="inner")
    page = S.loc[d.index, "page"].to_numpy()
    pages = np.unique(page)
    idx = {p: np.flatnonzero(page == p) for p in pages}
    y, pa, pb = d["y"].to_numpy(), d["pa"].to_numpy(), d["pb"].to_numpy()
    mf1 = lambda yy, pp: f1_score(yy, pp, labels=[1, 0], average="macro", zero_division=0)  # noqa: E731
    rng = np.random.default_rng(0)
    B = 2000
    diffs = np.empty(B)
    for i in range(B):
        ii = np.concatenate([idx[p] for p in rng.choice(pages, len(pages))])
        diffs[i] = mf1(y[ii], pb[ii]) - mf1(y[ii], pa[ii])
    from scipy.stats import norm
    alpha = 0.05 / max(n, 1)
    gain = mf1(y, pb) - mf1(y, pa)
    z = diffs.mean() / diffs.std(ddof=1)
    p_norm = float(norm.sf(z))  # one-sided, normal approximation of the bootstrap distribution
    share_le0 = float(np.mean(diffs <= 0))
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    ok = p_norm <= alpha
    record("Multiple testing: chosen gain survives Bonferroni", ok,
           f"{n} distinct variants tried; L6 vs L0 gain {gain:+.3f} macro-F1, z = {z:.1f}, "
           f"p = {p_norm:.1e} vs Bonferroni alpha {alpha:.1e}",
           "distinct variants per file: " + ", ".join(f"{k} {v}" for k, v in counts.items())
           + f"\nbootstrap: {B} page resamples over {len(pages)} pages; {share_le0:.2%} of resamples show no gain"
             f"\np from a normal approximation of the bootstrap distribution (a raw resample count cannot go "
             f"below 1/{B + 1}); Bonferroni-level interval [{lo:+.3f}, {hi:+.3f}]")


# ------------------------------------------------------------------ 7. claim re-read
def claims():
    p = OUT / "handoff" / "claim_reread_examples.csv"
    if not p.exists():
        record("Claim re-read findings", "INFO", "no claim re-read file yet")
        return
    C = pd.read_csv(p)
    by = C["check"].value_counts().to_dict()
    rules = (ROOT / "app" / "validate" / "rules.py").read_text(encoding="utf-8")
    has_digit_check = bool(re.search(r"soc2|isdigit|\\d.*compare|compare.*\\d", rules, re.I))
    record("Claim re-read findings", "INFO",
           f"{len(C)} generated claims flagged ({', '.join(f'{v} {k.split(':')[0]}' for k, v in by.items())}); "
           f"open for P2 (app/validate)" + ("; a digit check is visible in app/validate/rules.py" if has_digit_check
                                             else "; no digit-in-compare check found in app/validate/rules.py yet"),
           "\n".join(f"{r.id} ({r.type}): {r.text}" for r in C.itertuples()))


def write_report(t0):
    order = {"FAIL": 0, "PASS": 1, "INFO": 2}
    lines = ["# Research audit", "",
             f"*Generated by `eda/audit.py` on {datetime.now():%Y-%m-%d %H:%M} in {time.time() - t0:.0f} s. "
             "Rerun: `.venv/Scripts/python eda/audit.py`. Dev-only: nothing here runs in the image.*", "",
             "| Check | Result | Summary |", "|---|---|---|"]
    for c, s, summ, _ in sorted(RESULTS, key=lambda r: order.get(r[1], 3)):
        lines.append(f"| {c} | **{s}** | {summ} |")
    lines += ["", "## Details", ""]
    for c, s, _, det in RESULTS:
        lines += [f"### {c} — {s}", "", "```text", det.strip() or "(none)", "```", ""]
    lines += ["## What each check means", "",
              "- **Leakage:** the per-fold feature builder of `eda/07_improve.py` is run twice on the same fold, "
              "once with the test rows' labels flipped. Every feature of every row must stay identical, so no "
              "test label can reach a feature (KB, timeline anchors, page-trust priors).",
              "- **Determinism:** refitting the base and the chosen variant on the same fold gives the same "
              "predictions.",
              "- **Reproducibility:** `01_parse`, `02_rules`, `03_recommend` and `06_splits` are rerun in a "
              "temporary copy; their tables must equal the committed ones.",
              "- **Dependency pins:** every image package installed in `.venv` has the version in "
              "`constraints.txt`.",
              "- **Runtime gate:** retrain and prediction times measured on 2 CPU threads stay inside the gates "
              f"({GATES['retrain_s']} s retrain, {GATES['predict_20k_s']} s per 20k rows for the classifier, "
              f"{GATES['neural_predict_20k_s']} s for a neural model).",
              "- **Multiple testing:** every variant tried in the research is counted; the chosen gain must stay "
              "significant at a Bonferroni level for that count (page-level paired bootstrap).",
              "- **Claim re-read:** generated claims that the text checks in "
              "`eda/out/handoff/claim_reread_spec.md` flag, and whether `app/validate` already catches them."]
    (ROOT / "docs" / "AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {ROOT / 'docs' / 'AUDIT.md'}")


def main():
    t0 = time.time()
    imp, cached = leakage()
    determinism(imp, cached)
    reproducibility()
    pins()
    runtime_gate()
    multiple_testing()
    claims()
    write_report(t0)
    return 0 if all(s != "FAIL" for _, s, _, _ in RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
