"""Strict checks for the synthetic 10k data and the training protocol, shared by test_synth_data.py,
test_protocol.py and the step definitions of synth.feature / protocol.feature.

Every check returns (ok: bool, detail: str) so the tests can show *why* something fails. Loaders call
pytest.skip with a clear reason while the artefacts don't exist yet; once they exist, everything runs for real.
"""
import hashlib
import importlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import EDA, OUT, ROOT

H300 = EDA / "h300"
H300_OUT = H300 / "out"
MODELS = EDA / "models"
SYNTH_FILES = {"practice": H300_OUT / "synth10k_practice_rule.csv",
               "closed_world": H300_OUT / "synth10k_closed_world.csv"}
SYNTH_COLUMNS = ["id", "text", "date", "label", "form", "case_type", "chain_length", "broken_link", "fold", "mode"]
LABELS = {"true", "false", "unsure"}
MIX = {"practice": {"true": 0.83, "false": 0.15, "unsure": 0.02},
       "closed_world": {"true": 1 / 3, "false": 1 / 3, "unsure": 1 / 3}}
MIX_TOL = 0.05
MIN_PER_CASE = 100
MIN_CHAIN_SHARE = 0.15
NEAR_COPY = 90          # rapidfuzz ratio at or above this is a copy
MAX_DUP_SHARE = 0.005
MIN_RULE_AGREEMENT = 0.95
MIN_PARSE_BACK = 0.90
SAMPLE = 1000
JOKE_TAIL = re.compile(r"(curve is vertical|not even tired|more than anyone expected|nobody believed|coffee|"
                       r"I promise|personal best|before lunch|flawless)", re.I)


# ----------------------------------------------------------------------------- loaders

def practice():
    return pd.read_csv(ROOT / "data" / "practice.csv", encoding="utf-8-sig")


def splits():
    return pd.read_csv(EDA / "splits.csv")


def lexicon():
    lex = json.loads((ROOT / "data" / "lexicon.json").read_text(encoding="utf-8"))
    subj = {}
    for s in lex["subjects"]:
        attrs = set(s.get("attributes", []))
        for n in [s["name"]] + s.get("aliases", []):
            subj[n.lower()] = (s["name"], attrs)
    for e in lex.get("events", []):
        name = e["name"] if isinstance(e, dict) else e
        attrs = set(e.get("attributes", [])) if isinstance(e, dict) else set()
        subj.setdefault(name.lower(), (name, attrs))
    return subj


def universe_range():
    d = pd.to_datetime(practice()["date"])
    return d.min(), d.max()


def load_synth(mode):
    p = SYNTH_FILES[mode]
    if not p.exists():
        pytest.skip(f"{p.relative_to(ROOT)} not generated yet (eda/h300/synth.py writes it)")
    return pd.read_csv(p, dtype={"id": str, "text": str, "label": str, "form": str, "case_type": str},
                       keep_default_na=False)


def fold_files():
    files = sorted(H300_OUT.glob("synth_fold*_*.csv"))
    if not files:
        pytest.skip("no per-fold synthetic files in eda/h300/out yet (synth_fold<k>_<mode>.csv)")
    return files


def fold_of(path):
    m = re.search(r"synth_fold(\d+)_", path.name)
    return int(m.group(1)) if m else None


def h300_splits():
    p = H300_OUT / "h300_splits.csv"
    if not p.exists():
        pytest.skip("eda/h300/out/h300_splits.csv not built yet")
    return pd.read_csv(p)


def manifests():
    if not MODELS.is_dir():
        pytest.skip("eda/models/ does not exist yet (good models are saved there)")
    found = sorted(set(MODELS.glob("*manifest*.json")) | set(MODELS.glob("*/manifest.json")))
    if not found:
        pytest.skip("no saved model manifests in eda/models/ yet")
    return found


def read_manifest(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- held-out texts

def heldout_texts():
    """Every text no synthetic row may copy: practice holdout pages, check.sh hidden rows, h300 validation + lockbox."""
    P = practice()
    S = splits()
    ids = set(S.loc[(S.holdout == 1) | (S.fold_row5 == 0), "id"])
    texts = P.loc[P["id"].isin(ids), "text"].tolist()
    hp = ROOT / "data" / "hooly_statements_300.csv"
    if hp.exists():
        H = pd.read_csv(hp)
        hs_p = H300_OUT / "h300_splits.csv"
        if hs_p.exists():
            hs = pd.read_csv(hs_p)
            keep = set(hs.loc[hs["part"].isin(["cv", "lockbox"]), "id"])
            texts += H.loc[H["id"].isin(keep), "text"].tolist()
        else:
            texts += H["text"].tolist()
    return [t for t in texts if isinstance(t, str) and t.strip()]


def fold_validation_texts(k):
    P = practice()
    S = splits()
    ids = set(S.loc[S.fold_group == k, "id"])
    return P.loc[P["id"].isin(ids), "text"].tolist()


def max_similarity(texts, against):
    """Highest rapidfuzz ratio of each text against a list (vectorised, all cores)."""
    from rapidfuzz import fuzz, process
    if not against or not texts:
        return np.zeros(len(texts))
    m = process.cdist(list(texts), list(against), scorer=fuzz.ratio, dtype=np.uint8, workers=-1)
    return m.max(axis=1)


# ----------------------------------------------------------------------------- forms + values

def parse_form(s):
    try:
        f = json.loads(s)
    except (TypeError, ValueError):
        return None
    if isinstance(f, dict):
        f = [f]
    return f if isinstance(f, list) else None


def schema_validator():
    import jsonschema
    schema = json.loads((ROOT / "contracts" / "claimed_value.schema.json").read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator(schema)


def value_key(v):
    from app.kb.normalize import key
    try:
        return key(v)
    except Exception:  # noqa: BLE001 - malformed values just never match
        return json.dumps(v, sort_keys=True)


# ----------------------------------------------------------------------------- independent label recomputation

def training_true_timeline(fold):
    """{(subject, attribute): [(date, value_key)] sorted} from the gold forms of TRUE training rows of a fold.

    fold >= 0: practice rows outside that fold and outside the holdout; otherwise all non-holdout rows.
    Built here from data/practice.csv directly, independently of the generator."""
    P = practice()
    S = splits().set_index("id")
    P = P[P["id"].map(S["holdout"]).fillna(1) == 0]
    if fold is not None and fold >= 0:
        P = P[P["id"].map(S["fold_group"]) != fold]
    P = P[P["label"] == "true"]
    tl = {}
    for r in P.itertuples():
        for f in parse_form(r.form) or []:
            tl.setdefault((f["subject"], f["attribute"]), []).append((pd.Timestamp(r.date), value_key(f["value"])))
    for k in tl:
        tl[k].sort()
    return tl


def expected_fact_label(tl, subject, attribute, value, date, mode):
    steps = tl.get((subject, attribute))
    if not steps:
        return "true" if mode == "practice" else "unsure"   # a pair the training data never states
    vk = value_key(value)
    before = [v for d, v in steps if d <= date]
    if before and before[-1] == vk:
        return "true"                                        # the value in force (incl. an update on its first day)
    return "false"                                           # stale, premature, perturbed or swapped


def expected_row_label(tl, row, mode):
    forms = parse_form(row["form"]) or []
    if not forms:
        return None
    if row.get("case_type") == "joke" or JOKE_TAIL.search(str(row["text"])):
        return "unsure"
    if str(row.get("broken_link", "0")) in ("1", "True", "true"):
        return "false"
    date = pd.Timestamp(str(row["date"])[:10])
    labs = [expected_fact_label(tl, f.get("subject"), f.get("attribute"), f.get("value", {}), date, mode)
            for f in forms]
    if "false" in labs:
        return "false"
    if "unsure" in labs:
        return "unsure"
    return "true"


# ----------------------------------------------------------------------------- synthetic-data checks

def check_schema(df):
    missing = [c for c in SYNTH_COLUMNS if c not in df.columns]
    if missing:
        return False, f"missing columns {missing}"
    bad_label = df[~df["label"].isin(LABELS)]
    if len(bad_label):
        return False, f"{len(bad_label)} rows with a label outside {sorted(LABELS)}: {bad_label['label'].unique()[:5]}"
    if df["id"].duplicated().any():
        return False, f"{df['id'].duplicated().sum()} duplicated ids"
    if (df["text"].str.strip() == "").any():
        return False, "empty texts"
    cl = pd.to_numeric(df["chain_length"], errors="coerce")
    if cl.isna().any():
        return False, "chain_length is not numeric everywhere"
    bl = df["broken_link"].astype(str).str.lower()
    if not bl.isin(["0", "1", "true", "false"]).all():
        return False, f"broken_link holds non-boolean values {bl[~bl.isin(['0', '1', 'true', 'false'])].unique()[:5]}"
    return True, f"{len(df)} rows, schema ok"


def check_forms(df):
    v = schema_validator()
    lex = lexicon()
    bad, outside = [], []
    for r in df.itertuples():
        forms = parse_form(r.form)
        if not forms:
            bad.append((r.id, "form is not a JSON list"))
            continue
        for f in forms:
            if not isinstance(f, dict) or not {"subject", "attribute", "value"} <= set(f):
                bad.append((r.id, "fact lacks subject/attribute/value"))
                continue
            errs = list(v.iter_errors(f["value"]))
            if errs:
                bad.append((r.id, errs[0].message[:80]))
            s = lex.get(str(f["subject"]).lower())
            if s is None or (s[1] and f["attribute"] not in s[1]):
                outside.append((r.id, f["subject"], f["attribute"]))
    if bad:
        return False, f"{len(bad)} invalid forms, e.g. {bad[:3]}"
    if outside:
        return False, f"{len(outside)} facts outside the lexicon, e.g. {outside[:3]}"
    return True, "every form is a valid claimed value on a lexicon subject/attribute"


def check_dates(df):
    lo, hi = universe_range()
    d = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    if d.isna().any():
        return False, f"{d.isna().sum()} dates are not ISO YYYY-MM-DD, e.g. {df.loc[d.isna(), 'date'].head(3).tolist()}"
    out = df[(d < lo) | (d > hi)]
    if len(out):
        return False, f"{len(out)} dates outside the universe {lo.date()}..{hi.date()}, e.g. {out['date'].head(3).tolist()}"
    return True, f"all dates ISO and within {lo.date()}..{hi.date()}"


def check_label_mix(df, mode):
    share = df["label"].value_counts(normalize=True)
    off = {k: round(share.get(k, 0.0), 3) for k, want in MIX[mode].items() if abs(share.get(k, 0.0) - want) > MIX_TOL}
    if off:
        return False, f"{mode} mix off by > {MIX_TOL:.0%}: got {off}, want {MIX[mode]}"
    return True, f"{mode} mix ok: {share.round(3).to_dict()}"


def check_case_types(df):
    counts = df["case_type"].value_counts()
    small = counts[counts < MIN_PER_CASE]
    if len(counts) < 2:
        return False, f"only {len(counts)} case types: {counts.to_dict()}"
    if len(small):
        return False, f"case types with < {MIN_PER_CASE} rows: {small.to_dict()}"
    return True, f"{len(counts)} case types, each >= {MIN_PER_CASE}: {counts.to_dict()}"


def check_chains(df, mode):
    # 0 = not a chain; chain rows carry their hop count
    cl = pd.to_numeric(df["chain_length"], errors="coerce").fillna(0).astype(int)
    if not cl.isin([0, 1, 2, 3]).all():
        return False, f"chain_length outside {{0,1,2,3}}: {sorted(cl[~cl.isin([0, 1, 2, 3])].unique())[:5]}"
    share = (cl >= 1).mean()
    if share < MIN_CHAIN_SHARE:
        return False, f"chain rows {share:.1%} < {MIN_CHAIN_SHARE:.0%}"
    broken = df[df["broken_link"].astype(str).str.lower().isin(["1", "true"])]
    if len(broken) == 0:
        return False, "no broken-link rows: the chains never test a wrong inference"
    # closed-world mode may call an unknown hop unsure instead of false
    allowed = {"false", "unsure"} if "closed" in str(mode) else {"false"}
    wrong = broken[~broken["label"].isin(allowed)]
    if len(wrong):
        return False, f"{len(wrong)}/{len(broken)} broken-link rows are not labelled false in {mode} mode"
    return True, f"chain rows {share:.1%}; {len(broken)} broken links, all false"


def check_no_near_copies(df, against, what):
    sims = max_similarity(df["text"].tolist(), against)
    bad = int((sims >= NEAR_COPY).sum())
    if bad:
        i = int(np.argmax(sims))
        return False, f"{bad} rows are near-copies (ratio >= {NEAR_COPY}) of {what}, e.g. {df['text'].iloc[i]!r}"
    return True, f"max similarity to {what}: {int(sims.max()) if len(sims) else 0}"


def check_exact_dups(df):
    share = df["text"].str.strip().str.lower().duplicated().mean()
    return share <= MAX_DUP_SHARE, f"exact duplicate texts {share:.2%} (max {MAX_DUP_SHARE:.1%})"


def check_rule_agreement(df, mode, n=SAMPLE, seed=0):
    sample = df.sample(min(n, len(df)), random_state=seed)
    cache, hits, total, wrong = {}, 0, 0, []
    for _, r in sample.iterrows():
        f = pd.to_numeric(r.get("fold"), errors="coerce")
        f = int(f) if pd.notna(f) else -1
        if f not in cache:
            cache[f] = training_true_timeline(f)
        exp = expected_row_label(cache[f], r, mode)
        if exp is None:
            continue
        total += 1
        if exp == r["label"]:
            hits += 1
        elif len(wrong) < 5:
            wrong.append((r["id"], r["case_type"], r["label"], exp))
    if total == 0:
        return False, "no sampled row had a parsable form"
    rate = hits / total
    return rate >= MIN_RULE_AGREEMENT, f"label agreement with the training-fold timeline {rate:.1%} on {total} rows " \
                                       f"(min {MIN_RULE_AGREEMENT:.0%}); disagreements e.g. {wrong}"


def check_parse_back(df, n=SAMPLE, seed=0):
    import extract_simple as ex
    sample = df.sample(min(n, len(df)), random_state=seed)
    ok, total, miss = 0, 0, []
    for r in sample.itertuples():
        forms = parse_form(r.form) or []
        if not forms:
            continue
        total += 1
        found = [value_key(v) for _, _, v in ex.parse_values(r.text)]
        b = ex.parse_boolean(r.text)
        good = True
        for f in forms:
            v = f.get("value", {})
            if v.get("kind") == "name":
                good &= str(v.get("value", "")).lower() in r.text.lower()
            elif v.get("kind") == "boolean":
                good &= b is not None and bool(b) == bool(v.get("value"))
            else:
                good &= value_key(v) in found
        ok += good
        if not good and len(miss) < 5:
            miss.append((r.id, r.text[:70]))
    rate = ok / max(total, 1)
    return rate >= MIN_PARSE_BACK, f"{rate:.1%} of {total} texts parse back to their own form (min " \
                                   f"{MIN_PARSE_BACK:.0%}); misses e.g. {miss}"


# ----------------------------------------------------------------------------- protocol checks

def check_pages_disjoint():
    S = splits()
    cv = S[S.holdout == 0]
    folds = sorted(f for f in cv.fold_group.unique() if f >= 0)
    for k in folds:
        val = set(cv.loc[cv.fold_group == k, "page"])
        train = set(cv.loc[cv.fold_group != k, "page"])
        if val & train:
            return False, f"fold {k}: {len(val & train)} pages in both train and validation"
    hold = set(S.loc[S.holdout == 1, "page"])
    if hold & set(cv["page"]):
        return False, "holdout pages also appear in CV"
    return True, f"{len(folds)} folds: no page in both train and validation; holdout pages never in CV"


def check_h300_subjects_disjoint():
    hs = h300_splits()
    cv = hs[hs["part"] == "cv"]
    for k in sorted(cv["fold"].unique()):
        a = set(cv.loc[cv["fold"] == k, "subject"])
        b = set(cv.loc[cv["fold"] != k, "subject"])
        if a & b:
            return False, f"h300 fold {k}: subjects {sorted(a & b)[:3]} on both sides"
    lock = set(hs.loc[hs["part"] == "lockbox", "subject"])
    if lock & set(cv["subject"]):
        return False, f"lockbox subjects also in CV: {sorted(lock & set(cv['subject']))[:3]}"
    return True, "h300: no subject across folds or between lockbox and CV"


def _walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        yield path, obj
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list)):
                yield from _walk(v, f"{path}[{i}]")
    else:
        yield path, obj


def manifest_fields(m, pattern):
    rx = re.compile(pattern, re.I)
    return [(p, v) for p, v in _walk(m) if rx.search(p)]


def manifest_training_ids(m, mpath):
    ids = set()
    for p, v in manifest_fields(m, r"train"):
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            ids |= set(v)
        elif isinstance(v, str) and v.endswith((".csv", ".txt", ".json")):
            f = (Path(mpath).parent / v) if not Path(v).is_absolute() else Path(v)
            if not f.exists():
                f = ROOT / v
            if f.exists() and f.suffix == ".csv":
                df = pd.read_csv(f)
                if "id" in df:
                    ids |= set(df["id"].astype(str))
            elif f.exists():
                ids |= set(re.findall(r"[A-Za-z]+-\d+", f.read_text(encoding="utf-8")))
    return ids


def check_no_holdout_in_training(mpath):
    m = read_manifest(mpath)
    ids = manifest_training_ids(m, mpath)
    if not ids:
        return False, f"{Path(mpath).name}: the manifest records no training ids (list or file) — provenance is unverifiable"
    S = splits()
    hold = set(S.loc[S.holdout == 1, "id"])
    lock = set()
    hs_p = H300_OUT / "h300_splits.csv"
    if hs_p.exists():
        hs = pd.read_csv(hs_p)
        lock = set(hs.loc[hs["part"] == "lockbox", "id"])
    leak = ids & (hold | lock)
    if leak:
        return False, f"{Path(mpath).name}: {len(leak)} holdout/lockbox ids in training, e.g. {sorted(leak)[:5]}"
    return True, f"{Path(mpath).name}: {len(ids)} training ids, none from the holdout pages or the h300 lockbox"


def check_lockbox_once(mpath):
    m = read_manifest(mpath)
    hits = manifest_fields(m, r"lockbox")
    counts = [v for p, v in hits if re.search(r"(look|use|count|times)", p, re.I) and isinstance(v, (int, float))]
    flags = [v for p, v in hits if isinstance(v, bool)]
    if counts:
        ok = all(c <= 1 for c in counts)
        return ok, f"{Path(mpath).name}: lockbox looks {counts}"
    if flags:
        return all(flags), f"{Path(mpath).name}: lockbox flags {flags}"
    return False, f"{Path(mpath).name}: no lockbox look counter/flag recorded"


def check_gates_pass(mpath):
    m = read_manifest(mpath)
    gates = manifest_fields(m, r"gate")
    vals = [(p, v) for p, v in gates if not isinstance(v, (dict, list))]
    if not vals:
        return False, f"{Path(mpath).name}: no overfit gate results recorded"
    bad = [(p, v) for p, v in vals if not (v is True or str(v).strip().upper() in {"PASS", "PASSED", "TRUE", "OK"})]
    return not bad, f"{Path(mpath).name}: gates {len(vals)} recorded; failing {bad[:5]}"


def check_provenance_and_seeds(mpath):
    m = read_manifest(mpath)
    prov = manifest_fields(m, r"(training|train_data|provenance|data)")
    seeds = manifest_fields(m, r"seed")
    missing = [n for n, x in (("training data provenance", prov), ("seeds", seeds)) if not x]
    return not missing, f"{Path(mpath).name}: missing {missing}" if missing else f"{Path(mpath).name}: provenance + seeds"


def model_file(mpath):
    m = read_manifest(mpath)
    for p, v in manifest_fields(m, r"(path|file|joblib|model)"):
        if isinstance(v, str) and v.endswith(".joblib"):
            for cand in ((Path(mpath).parent / v), (ROOT / v), Path(v)):
                if cand.exists():
                    return cand
    mp = Path(mpath)
    if mp.name == "manifest.json":                       # eda/models/<name>/manifest.json + one .joblib beside it
        found = sorted(mp.parent.glob("*.joblib"))
        return found[0] if len(found) == 1 else None
    stem = re.sub(r"[._-]?manifest$", "", mp.stem)       # eda/models/<name>.manifest.json -> <name>.joblib
    cand = mp.parent / f"{stem}.joblib"
    return cand if cand.exists() else None


def check_sha256(mpath):
    m = read_manifest(mpath)
    f = model_file(mpath)
    if f is None:
        return False, f"{Path(mpath).name}: no .joblib file found for this manifest"
    recorded = [v for p, v in manifest_fields(m, r"sha256") if isinstance(v, str)]
    if not recorded:
        return False, f"{Path(mpath).name}: no sha256 recorded"
    actual = hashlib.sha256(f.read_bytes()).hexdigest()
    return actual in recorded, f"{Path(mpath).name}: sha256 {'matches' if actual in recorded else 'MISMATCH'} {f.name}"


def check_permutation(mpath):
    m = read_manifest(mpath)
    vals = [v for p, v in manifest_fields(m, r"permutation") if isinstance(v, (int, float))
            and re.search(r"learn", p, re.I)]
    if not vals:
        vals = [v for p, v in manifest_fields(m, r"permutation") if isinstance(v, (int, float))
                and not re.search(r"(rule|floor|full|seed|n_|repeat)", p, re.I)]
    if not vals:
        return False, f"{Path(mpath).name}: no label-permutation result for the learned part recorded"
    return max(vals) <= 0.50, f"{Path(mpath).name}: permuted learned-part score {max(vals):.3f} (max 0.50)"


def check_gap(mpath, limit=0.10):
    m = read_manifest(mpath)
    vals = [v for p, v in manifest_fields(m, r"gap") if isinstance(v, (int, float))]
    if not vals:
        return False, f"{Path(mpath).name}: no train−validation gap recorded"
    return max(vals) <= limit, f"{Path(mpath).name}: train−val gap {max(vals):.3f} (max {limit})"


def robustness_rows(n=20, seed=0):
    """20 practice + 20 h300 rows, renamed ids, slash dates, every label blank (zero labelled rows)."""
    P = practice().sample(n, random_state=seed)
    rows = P[["text", "date"]].copy()
    hp = ROOT / "data" / "hooly_statements_300.csv"
    if hp.exists():
        rows = pd.concat([rows, pd.read_csv(hp).sample(n, random_state=seed)[["text", "date"]]], ignore_index=True)
    rows = rows.reset_index(drop=True)
    rows.insert(0, "id", [f"zz-{i + 1}" for i in range(len(rows))])
    rows["date"] = pd.to_datetime(rows["date"]).dt.strftime("%Y/%m/%d")
    rows["label"] = ""
    return rows.to_dict("records")


def predict_with(mpath, rows):
    """Load a saved model and predict through its declared entry point. Returns a list of labels."""
    import joblib
    m = read_manifest(mpath)
    f = model_file(mpath)
    if f is None:
        raise AssertionError(f"{Path(mpath).name}: no .joblib to load")
    model = joblib.load(f)
    entry = next((v for p, v in manifest_fields(m, r"(predict_entry|entry_point|predict_fn)") if isinstance(v, str)),
                 None)
    lex = json.loads((ROOT / "data" / "lexicon.json").read_text(encoding="utf-8"))
    if entry:
        mod, _, fn = entry.partition(":")
        out = getattr(importlib.import_module(mod), fn)(model, rows, lex)
    elif hasattr(model, "predict_rows"):
        out = model.predict_rows(rows, lex)
    elif isinstance(model, dict) and callable(model.get("predict_rows")):
        out = model["predict_rows"](rows, lex)
    else:
        raise AssertionError(f"{Path(mpath).name}: model exposes no predict_rows(rows, lexicon) and the manifest "
                             f"declares no predict_entry 'module:function'")
    labels = []
    for o in out:
        lab = o.get("label") if isinstance(o, dict) else getattr(o, "label", o)
        conf = o.get("confidence") if isinstance(o, dict) else getattr(o, "confidence", None)
        if lab not in LABELS:
            raise AssertionError(f"prediction label {lab!r} not in {sorted(LABELS)}")
        if lab == "unsure":
            if conf not in (None, "") and not (isinstance(conf, float) and np.isnan(conf)):
                raise AssertionError(f"unsure must carry no confidence, got {conf}")
        elif conf is None or not 0.5 <= float(conf) <= 1.0:
            raise AssertionError(f"confidence {conf} outside [0.5, 1]")
        labels.append(lab)
    return labels
