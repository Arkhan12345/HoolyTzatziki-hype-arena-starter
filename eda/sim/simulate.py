"""Simulate the whole system on the outside set, like the arena run, with configurations fixed in advance.

    .venv/Scripts/python eda/sim/simulate.py            # one run; refuses a second (--force overrides, and is logged)

Setup
  training universe  practice rows NOT in eda/sim/out/outside.csv (labels + forms)
  universe.csv       every practice row in file order, outside rows with a blank label, + the renamed copies (blank),
                     no `form` column; lexicon.json copied unchanged (the invented names are not in it)
Systems
  runtime   `app.run.main` as shipped, with its gold file pointed at the training universe only (otherwise it would
            read the outside rows' labels and forms from data/practice.csv); output.csv checked by scripts/check_output.py
  FINAL     the frozen FINAL configuration (eda/models/FINAL/manifest.json), retrained from scratch: training rows read
            out of fold (5 page-grouped folds, a Reader fitted without the fold), synthetic practice-rule statements
            regenerated from the training universe only (weight from the manifest), outside rows read by a Reader
            fitted on the whole training universe
  FALLBACK  the frozen FALLBACK configuration, same protocol, no synthetic
  all-true  baseline
Learning check: FINAL's learned layer trained on 50% vs 100% of the training pages (same readings).
Writes eda/sim/out/results.csv, per_slice.csv, run_log.json, runtime/ (universe, output, check).
"""
import argparse
import hashlib
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
SIM = Path(__file__).resolve().parent
OUT = SIM / "out"
H300 = ROOT / "eda" / "h300"
sys.path.insert(0, str(H300))
from lib.extractor import Lexicon  # noqa: E402
from lib.feats import Reader  # noqa: E402
from lib.model import LAB2I, Learned  # noqa: E402
import run as h300run  # noqa: E402
import synth  # noqa: E402

SEED = 20261029
CLASSES = ["false", "true", "unsure"]


def blind(rows):
    return [{"id": r["id"], "text": r["text"], "date": r["date"], "label": ""} for r in rows]


def feats(reader, rows):
    F = pd.DataFrame(reader.read(blind(rows)))
    F.insert(0, "id", [r["id"] for r in rows])
    return F


def manifest_cfg(name):
    m = json.loads((ROOT / "eda" / "models" / name / "manifest.json").read_text(encoding="utf-8"))
    cfg = dict(m["config"])
    return cfg, m.get("synth_weight", 0)


def score(y, pred, conf_or_P):
    P = conf_or_P
    if P.ndim == 1:
        P = np.full((len(pred), 3), 0.0)
        P[np.arange(len(pred)), pred] = conf_or_P
    return h300run.metrics(y, pred, P)


def run_runtime(train_rows, outside, practice_cols):
    """The shipped runtime on the simulated universe; its gold file holds the training universe only."""
    rt = OUT / "runtime"
    if rt.exists():
        shutil.rmtree(rt)
    (rt / "universe").mkdir(parents=True)
    (rt / "output").mkdir()
    prac = pd.read_csv(ROOT / "data" / "practice.csv", dtype=str, keep_default_na=False)
    out_ids = set(outside.loc[outside.renamed == 0, "orig_id"])
    uni = prac[["id", "text", "date", "label"]].copy()
    uni.loc[uni.id.isin(out_ids), "label"] = ""
    ren = outside.loc[outside.renamed == 1, ["id", "text", "date"]].assign(label="")
    pd.concat([uni, ren], ignore_index=True).to_csv(rt / "universe" / "universe.csv", index=False)
    shutil.copy(ROOT / "data" / "lexicon.json", rt / "universe" / "lexicon.json")
    pd.DataFrame(train_rows)[practice_cols].to_csv(rt / "practice_train.csv", index=False)
    code = ("import sys, time; sys.path.insert(0, r'%s'); import app.run as r; r.PRACTICE = r'%s'; "
            "r.main(r'%s', r'%s', time.time() + 1500)") % (ROOT, rt / "practice_train.csv", rt / "universe", rt / "output")
    env = dict(os.environ, TZ_AUX_DIR=str(rt / "aux"))
    t0 = time.time()
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=1700)
    secs = time.time() - t0
    (rt / "run.log").write_text(p.stdout[-20000:] + "\n--- stderr ---\n" + p.stderr[-20000:], encoding="utf-8")
    chk = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_output.py"), str(rt / "output"),
                          str(ROOT / "data" / "practice.csv"), str(rt / "universe" / "universe.csv")],
                         capture_output=True, text=True)
    (rt / "check_output.txt").write_text(chk.stdout + chk.stderr, encoding="utf-8")
    table = pd.read_csv(rt / "output" / "output.csv", dtype=str, keep_default_na=False)
    labels = table[table["type"] == ""].set_index("id")
    claims = table[table["type"] != ""]
    return labels, {"exit": p.returncode, "seconds": round(secs, 1), "check_exit": chk.returncode,
                    "check": chk.stdout.strip()[-1500:], "claims": int(len(claims)),
                    "claims_by_type": claims["type"].value_counts().to_dict()}


def main_v2(force: bool):
    """One run of the v2 configuration (eda/h300/out/ladder_v2_choice.json, chosen on practice CV only) on the same
    frozen outside set, with the same training universe and protocol as the FINAL v1 run. Logged as label "v2"."""
    import build_v2
    import ladder_v2
    log_p = OUT / "run_log.json"
    log = json.loads(log_p.read_text()) if log_p.exists() else {"runs": []}
    if any(r.get("label") == "v2" for r in log["runs"]) and not force:
        sys.exit("one v2 run only: eda/sim/out/run_log.json already records it")
    sha_frozen = (OUT / "outside.sha256").read_text().strip()
    sha_now = hashlib.sha256((OUT / "outside.csv").read_bytes()).hexdigest()
    assert sha_now == sha_frozen, "the outside set changed after it was frozen"
    choice = json.loads((H300 / "out" / "ladder_v2_choice.json").read_text())
    cfg = choice["config"]
    t0 = time.time()
    outside = pd.read_csv(OUT / "outside.csv", dtype=str, keep_default_na=False)
    for c in ("renamed",):
        outside[c] = outside[c].astype(int)
    prac = pd.read_csv(ROOT / "data" / "practice.csv", dtype=str, keep_default_na=False)
    d = pd.to_datetime(prac.date)
    prac["page"] = (d != d.shift()).cumsum()
    train = prac[~prac.id.isin(set(outside.loc[outside.renamed == 0, "orig_id"]))].copy()
    train_rows = train.to_dict("records")
    lexicon = Lexicon.load(ROOT / "data" / "lexicon.json")
    lex_raw = json.loads((ROOT / "data" / "lexicon.json").read_text(encoding="utf-8"))
    mapping, aliases = build_v2.rename_map(lex_raw)
    lab = dict(zip(train.id, train.label))
    full = Reader.fit(train_rows, lexicon)
    parts, ren = [], []
    for tr, te in GroupKFold(5).split(train, groups=train.page.values):
        rd = Reader.fit([train_rows[i] for i in tr], lexicon, margin=full.margin)
        rows_te = [train_rows[i] for i in te]
        parts.append(build_v2.read(rd, rows_te))
        if cfg.get("ren_w"):
            ren.append(build_v2.read(rd, [build_v2.rename_row(r, mapping, aliases) for r in rows_te]))
    TF = pd.concat(parts, ignore_index=True)
    TF["y"] = TF.id.map(lab).map(LAB2I).values
    fold = {"train": TF, "ren_train": pd.concat(ren, ignore_index=True) if ren else TF.iloc[:0], "synth": {}}
    if len(fold["ren_train"]):
        fold["ren_train"]["y"] = fold["ren_train"].id.str.replace("^rn-", "", regex=True).map(lab).map(LAB2I).values
    held = list(outside.text)
    for mode, w in (("practice", cfg.get("synth_w")), ("closed_world", cfg.get("cw_w"))):
        if not w:
            fold["synth"][mode] = TF.iloc[:0]
            continue
        srows, _ = synth.generate(train_rows, lex_raw, held, mode, 10000 if mode == "practice" else 2000,
                                  seed=SEED, prefix=f"sim{mode[:2]}", fold="sim")
        S = pd.DataFrame(srows, columns=synth.COLUMNS)
        S = S[["id", "label", "case_type"]].merge(build_v2.read(full, S.to_dict("records")), on="id")
        S["y"] = S.label.map(LAB2I).values
        fold["synth"][mode] = S
    fold = ladder_v2.view(cfg, {**fold, "val": TF.iloc[:0], "ren_val": TF.iloc[:0]})
    F, y, w = ladder_v2.training_set(cfg, fold)
    model = ladder_v2.Model(cfg, 0).fit(F, y, w)
    OF = build_v2.read(full, outside[["id", "text", "date"]].to_dict("records")).merge(outside, on="id")
    if not cfg.get("unk_read"):
        OF = ladder_v2.as_v1_reading(OF)
    OF["y"] = OF.label.map(LAB2I).values
    pred, P = model.predict(OF)
    yv = OF.y.values
    slices = {"all original outside rows": OF.renamed.values == 0,
              "unseen subjects": (OF.renamed.values == 0) & (OF.slice.values == "subject"),
              "last 6 weeks": (OF.renamed.values == 0) & (OF.slice.values == "time"),
              "random pages": (OF.renamed.values == 0) & (OF.slice.values == "page"),
              "renamed subjects (invented names)": OF.renamed.values == 1}
    rows = [{"system": "FINAL v2 (retrained)", "slice": s, "rows": int(m.sum()),
             **{k: round(v, 4) for k, v in h300run.metrics(yv[m], pred[m], P[m]).items()}} for s, m in slices.items()]
    R = pd.DataFrame(rows)
    old = pd.read_csv(OUT / "per_slice.csv")
    pd.concat([old[old.system != "FINAL v2 (retrained)"], R], ignore_index=True).to_csv(OUT / "per_slice.csv",
                                                                                        index=False)
    log["runs"].append({"label": "v2", "when": datetime.now().isoformat(timespec="seconds"), "forced": force,
                        "outside_sha256": sha_now, "config_fixed_before_run": cfg, "training_rows": int(len(train)),
                        "seconds_total": round(time.time() - t0, 1),
                        "training_ids_sha256": hashlib.sha256("\n".join(sorted(train.id)).encode()).hexdigest()})
    log_p.write_text(json.dumps(log, indent=1, default=str), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(R.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", choices=["v1", "v2"], default="v1")
    a = ap.parse_args()
    if a.config == "v2":
        return main_v2(a.force)
    log_p = OUT / "run_log.json"
    log = json.loads(log_p.read_text()) if log_p.exists() else {"runs": []}
    if log["runs"] and not a.force:
        sys.exit("one run only: eda/sim/out/run_log.json already records a run (use --force; it is logged)")
    sha_frozen = (OUT / "outside.sha256").read_text().strip()
    sha_now = hashlib.sha256((OUT / "outside.csv").read_bytes()).hexdigest()
    assert sha_now == sha_frozen, "the outside set changed after it was frozen"
    t_all = time.time()

    outside = pd.read_csv(OUT / "outside.csv", dtype=str, keep_default_na=False)
    for c in ("renamed", "slice_subject", "slice_time", "slice_page"):
        outside[c] = outside[c].astype(int)
    prac = pd.read_csv(ROOT / "data" / "practice.csv", dtype=str, keep_default_na=False)
    pcols = list(prac.columns)
    d = pd.to_datetime(prac.date)
    prac["page"] = (d != d.shift()).cumsum()
    out_ids = set(outside.loc[outside.renamed == 0, "orig_id"])
    train = prac[~prac.id.isin(out_ids)].copy()
    train_rows = train.to_dict("records")
    outside_rows = outside[["id", "text", "date"]].to_dict("records")
    assert not (set(train.id) & set(outside.orig_id)), "an outside id is in the training universe"
    lexicon = Lexicon.load(ROOT / "data" / "lexicon.json")
    lex_raw = json.loads((ROOT / "data" / "lexicon.json").read_text(encoding="utf-8"))
    cfg_final, w_final = manifest_cfg("FINAL")
    cfg_fb, _ = manifest_cfg("FALLBACK")
    print(f"training universe {len(train)} rows; outside {len(outside)} rows (incl. renamed); "
          f"FINAL {cfg_final} w={w_final}; FALLBACK {cfg_fb}", flush=True)

    # ---------------------------------------------- (a) the shipped runtime
    rt_labels, rt_info = run_runtime(train_rows, outside, pcols)
    print(f"runtime: {rt_info['seconds']} s, exit {rt_info['exit']}, claims {rt_info['claims']}, "
          f"check exit {rt_info['check_exit']}", flush=True)

    # ---------------------------------------------- (b) FINAL / FALLBACK retrained on the training universe only
    t0 = time.time()
    full = Reader.fit(train_rows, lexicon)
    print(f"full reader fitted ({time.time() - t0:.0f}s)", flush=True)
    folds = list(GroupKFold(5).split(train, groups=train.page.values))
    oof = []
    for k, (tr, te) in enumerate(folds):
        rd = Reader.fit([train_rows[i] for i in tr], lexicon, margin=full.margin)
        oof.append(feats(rd, [train_rows[i] for i in te]))
        print(f"  oof fold {k} ({time.time() - t0:.0f}s)", flush=True)
    TF = pd.concat(oof, ignore_index=True)
    TF["y"] = TF.id.map(dict(zip(train.id, train.label))).map(LAB2I).values
    TF["page"] = TF.id.map(dict(zip(train.id, train.page))).values
    OF = feats(full, outside_rows).merge(outside, on="id")
    OF["y"] = OF.label.map(LAB2I).values
    held_texts = list(outside.text)
    srows, sstats = synth.generate(train_rows, lex_raw, held_texts, "practice", 10000, seed=SEED, prefix="sim",
                                   fold="sim")
    SF = pd.DataFrame(srows, columns=synth.COLUMNS)
    SF = SF.merge(feats(full, SF.to_dict("records")), on="id")
    SF["y"] = SF.label.map(LAB2I).values
    print(f"synthetic {len(SF)} rows from the training universe, rule agreement {sstats.get('rule_agreement')} "
          f"({time.time() - t0:.0f}s)", flush=True)

    def fit_pred(cfg, w_syn, keep=None):
        F, y = (TF, TF.y.values) if keep is None else (TF[keep], TF.y.values[keep])
        w = np.ones(len(F))
        if w_syn:
            F = pd.concat([F, SF], ignore_index=True)
            y = np.concatenate([y, SF.y.values])
            w = np.concatenate([w, np.full(len(SF), w_syn)])
        c = {k: v for k, v in cfg.items() if k not in ("synth", "selftrain")}
        m = Learned(c, h300run.cols_of(c), 0).fit(F, y.astype(int), w)
        return m.predict(OF)

    preds = {"FINAL (retrained)": fit_pred(cfg_final, w_final), "FALLBACK (retrained)": fit_pred(cfg_fb, 0)}
    pages = TF.page.unique()
    half = set(random.Random(SEED).sample(list(pages), len(pages) // 2))
    preds["FINAL, 50% of training pages"] = fit_pred(cfg_final, w_final, keep=TF.page.isin(half).values)

    # ---------------------------------------------- scoring per slice
    y = OF.y.values
    rt = OF.id.map(rt_labels["label"]).fillna("true")
    rt_pred = rt.map(LAB2I).values.astype(int)
    rt_conf = pd.to_numeric(OF.id.map(rt_labels["confidence"]), errors="coerce").fillna(0.83).values
    systems = {"all-true": (np.full(len(y), LAB2I["true"]), np.full(len(y), 0.83)),
               "runtime as shipped": (rt_pred, rt_conf)}
    systems.update({k: (v[0], v[1]) for k, v in preds.items()})
    slices = {"all original outside rows": OF.renamed.values == 0,
              "unseen subjects": (OF.renamed.values == 0) & (OF.slice.values == "subject"),
              "last 6 weeks": (OF.renamed.values == 0) & (OF.slice.values == "time"),
              "random pages": (OF.renamed.values == 0) & (OF.slice.values == "page"),
              "renamed subjects (invented names)": OF.renamed.values == 1}
    rows = []
    for sname, (pred, P) in systems.items():
        for slname, m in slices.items():
            r = score(y[m], pred[m], P[m] if P.ndim == 1 else P[m])
            rows.append({"system": sname, "slice": slname, "rows": int(m.sum()), **{k: round(v, 4) for k, v in r.items()}})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "per_slice.csv", index=False)
    run_entry = {"when": datetime.now().isoformat(timespec="seconds"), "forced": a.force,
                 "outside_sha256": sha_now, "configs_fixed_before_run": {"FINAL": cfg_final, "FINAL_synth_weight":
                                                                       w_final, "FALLBACK": cfg_fb},
                 "training_rows": int(len(train)), "synthetic_rows": int(len(SF)), "synthetic_stats": sstats,
                 "runtime": rt_info, "seconds_total": round(time.time() - t_all, 1),
                 "training_ids_sha256": hashlib.sha256("\n".join(sorted(train.id)).encode()).hexdigest()}
    (OUT / "training_ids.txt").write_text("\n".join(sorted(train.id)), encoding="utf-8")
    log["runs"].append(run_entry)
    log_p.write_text(json.dumps(log, indent=1, default=str), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(R.to_string(index=False))
    print(json.dumps(rt_info, indent=1)[:1500])


if __name__ == "__main__":
    main()
