"""Row features for the option-A truth classifier.

Four groups:
  text / page   wording flags and page shape (text + date only)
  extraction    how many facts the extractor read, their kinds, ops
  support       how the rest of the corpus talks about the same (subject,
                attribute): label-free, so it works on unlabelled rows
  kb            relation to the gold timeline built from LABELLED rows only
                (`known_labels`), e.g. "repeats an older value" (rule T1)

Support features avoid raw counts on purpose: the practice set has ~1.9k rows
and the real universe ~20k, so counts would shift at run time. Shares and
flags do not.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Sequence

import numpy as np
import pandas as pd

from app.extract.extractor import Fact, page_ids

# ---------------------------------------------------------------- text flags
PATTERNS = {
    "first_person": (r"\b(I|I'm|I've|me|my|mine)\b", 0),
    "we_our": (r"\b(we|our|us)\b", re.I),
    "quote": (r"[\"“”]|\bsaid\b|\bsays\b|\btold\b", re.I),
    "hearsay": (r"\b(I have been told|told me|reportedly|apparently|rumou?r|I hear|they say|word is)\b", re.I),
    "hedge": (r"\b(about|around|nearly|almost|roughly|over|more than|up to|under|less than|at least|at most|some)\b", re.I),
    "superlative": (r"\b(\w+est|most|best|first|only|record|ever|any competitor|more than any|all humans|nobody|everyone|flawless|bargain|vertical)\b", re.I),
    "now_today": (r"\b(now|today|from today|this week|as of)\b", re.I),
    "still": (r"\bstill\b", re.I),
    "future": (r"\b(will|is due|are due|plans? to|next|soon|coming|upcoming)\b", re.I),
    "negation": (r"\b(not|no|never|none|without)\b|n't\b", re.I),
    "change_verb": (r"\b(grows?|rises?|drops?|falls?|cuts?|raises?|increases?|doubles?|moves?|switch(?:es)?|replaces?|up from|down from|was)\b", re.I),
}
JOKE_TAIL = re.compile(
    r"(,| and | while | which | but ).*\b(I|me|my|nobody|everyone|all humans|curve|tired|lunch|coffee|"
    r"personal best|poetry|flawless|learning|impossible|higher|vertical|bargain)\b", re.I)

# Which facts of labelled-true rows build the timeline: "all", "anchored"
# (subject named in the sentence) or "conf0.6" (extractor confidence >= 0.6).
# "anchored" helps with the wide extractor (+1 pt); "all" is best with classic.
KB_SOURCE = "all"

KB_RELATIONS = ["match_latest", "future_value", "before_first_known", "unseen_pair",
                "contradicts", "stale_older"]  # mild -> severe
KINDS = ["quantity", "money", "duration", "name", "percent", "boolean", "date", "version"]


def text_page_features(rows: Sequence[dict]) -> pd.DataFrame:
    texts = pd.Series([str(r["text"]) for r in rows])
    X = pd.DataFrame(index=range(len(rows)))
    for name, (pattern, flags) in PATTERNS.items():
        X[name] = texts.str.contains(pattern, flags=flags, regex=True).astype(float)
    X["joke_tail"] = texts.map(lambda s: float(bool(JOKE_TAIL.search(s))))
    X["no_period"] = (~texts.str.strip().str.endswith((".", "!", "?", '"', "”"))).astype(float)
    X["kv_line"] = texts.str.contains(r"^[^.]{2,40}: ", regex=True).astype(float)
    X["n_words"] = texts.str.split().str.len().astype(float)
    X["n_commas"] = texts.str.count(",").astype(float)
    X["starts_pronoun"] = texts.str.contains(
        r"^(?:He|She|It|Its|They|Their|His|Her|This|That|The \w+)\b", regex=True).astype(float)
    X["hype_score"] = X[["first_person", "superlative", "joke_tail", "hearsay"]].sum(axis=1)

    pages = pd.Series(page_ids(rows))
    X["page_size"] = pages.map(pages.value_counts()).astype(float)
    X["pos_in_page"] = pages.groupby(pages).cumcount().astype(float)
    X["pos_rel"] = X["pos_in_page"] / (X["page_size"] - 1).clip(lower=1)
    X["is_headline"] = (X["pos_in_page"] == 0).astype(float)
    X["page_first_person_share"] = X.groupby(pages)["first_person"].transform("mean")
    X["page_hype_share"] = (X["hype_score"] > 0).groupby(pages).transform("mean").astype(float)
    return X


# ---------------------------------------------------------------- fact table
def _numeric(key: tuple) -> float:
    kind = key[0]
    if kind in ("quantity", "duration"):
        return float(key[2])
    if kind in ("money", "percent"):
        return float(key[1])
    return math.nan


def fact_table(rows: Sequence[dict], facts: Sequence[Sequence[Fact]]) -> pd.DataFrame:
    pages = page_ids(rows)
    records = []
    for i, (row, row_facts) in enumerate(zip(rows, facts)):
        date = pd.Timestamp(str(row["date"]).strip())
        for f in row_facts:
            key = f.key[2]
            op = key[-1] if key[0] != "boolean" else "eq"
            vkey = repr(key[:-1] if key[0] != "boolean" else key)
            records.append({
                "row": i, "date": date, "page": pages[i], "pair": f"{f.subject}|{f.attribute}",
                "kind": f.value.get("kind"), "vkey": vkey, "vnum": _numeric(key), "op": op,
                "conf": f.conf, "anchored": float(getattr(f, "anchored", True)),
            })
    cols = ["row", "date", "page", "pair", "kind", "vkey", "vnum", "op", "conf", "anchored"]
    return pd.DataFrame.from_records(records, columns=cols)


def _op_consistent(op: str, v: float, ref: float) -> float:
    if op == "eq" or math.isnan(v) or math.isnan(ref):
        return math.nan
    return float({"gt": ref > v, "ge": ref >= v, "lt": ref < v, "le": ref <= v}.get(op, math.nan))


def _log_ratio(v: float, ref: float) -> float:
    if math.isnan(v) or math.isnan(ref) or v <= 0 or ref <= 0:
        return math.nan
    return math.log10(v / ref)


# ---------------------------------------------------------------- support (label-free)
def support_features(F: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=F.index, dtype=float)
    if F.empty:
        return out
    cols = defaultdict(dict)
    for _, g in F.groupby("pair", sort=False):
        g = g.sort_values(["date", "row"])
        idx, dates, pages_ = g.index.values, g["date"].values, g["page"].values
        vkeys, vnums = g["vkey"].values, g["vnum"].values
        counts = g["vkey"].value_counts()
        mode = counts.index[0]
        n_vals = len(counts)
        med = np.nanmedian(vnums) if np.isfinite(vnums).any() else math.nan
        for j, ix in enumerate(idx):
            other = pages_ != pages_[j]
            same = vkeys == vkeys[j]
            before = other & (dates < dates[j])
            after = other & (dates > dates[j])
            near = other & (np.abs((dates - dates[j]).astype("timedelta64[D]").astype(float)) <= 60)
            c = cols[ix]
            c["value_share"] = (same & other).sum() / max(other.sum(), 1) if other.any() else math.nan
            c["repeated_other_page"] = float((same & other).any())
            c["is_pair_mode"] = float(vkeys[j] == mode and counts.iloc[0] > (counts.iloc[1] if n_vals > 1 else 0))
            c["single_value_pair"] = float(n_vals == 1)
            c["pair_value_diversity"] = n_vals / len(g)
            c["pair_seen_elsewhere"] = float(other.any())
            if before.any():
                last = np.flatnonzero(before)[-1]
                c["agree_prev_mention"] = float(vkeys[last] == vkeys[j])
                c["days_since_prev_mention"] = math.log1p(
                    (dates[j] - dates[last]).astype("timedelta64[D]").astype(float))
                c["ratio_to_prev_mention"] = _log_ratio(vnums[j], vnums[last])
                c["seen_before_anywhere"] = float((same & before).any())
            if after.any():
                first = np.flatnonzero(after)[0]
                c["agree_next_mention"] = float(vkeys[first] == vkeys[j])
                c["seen_after_anywhere"] = float((same & after).any())
            if near.any():
                near_vals = pd.Series(vkeys[near]).value_counts()
                c["near_share"] = near_vals.get(vkeys[j], 0) / near_vals.sum()
                c["is_near_mode"] = float(near_vals.index[0] == vkeys[j])
            r = _log_ratio(vnums[j], med)
            c["log_dev_from_pair_median"] = r
            c["abs_log_dev_from_pair_median"] = abs(r) if not math.isnan(r) else math.nan
    return pd.DataFrame.from_dict(cols, orient="index").reindex(F.index)


# ---------------------------------------------------------------- kb (label-based)
def kb_features(F: pd.DataFrame, rows: Sequence[dict], known_labels: dict[int, str]) -> pd.DataFrame:
    """Relation of each fact to the timeline of TRUE facts from rows in `known_labels`.

    `known_labels` maps row index -> label. A row is never compared with itself
    or with rows of its own page.
    """
    out = pd.DataFrame(index=F.index, dtype=float)
    if F.empty:
        return out
    true_rows = {i for i, lab in known_labels.items() if lab == "true"}
    false_rows = {i for i, lab in known_labels.items() if lab == "false"}
    is_true = F["row"].isin(true_rows).values
    is_false = F["row"].isin(false_rows).values

    gold_mask = is_true
    if KB_SOURCE == "anchored" and "anchored" in F:
        gold_mask = is_true & (F["anchored"].to_numpy() > 0)
    elif KB_SOURCE.startswith("conf") and "conf" in F:
        gold_mask = is_true & (F["conf"].to_numpy() >= float(KB_SOURCE[4:]))
    gold = F[gold_mask].sort_values(["date", "row"])
    gold_by_pair = {p: g for p, g in gold.groupby("pair", sort=False)}
    false_keys = defaultdict(set)
    for p, v in zip(F.loc[is_false, "pair"], F.loc[is_false, "vkey"]):
        false_keys[p].add(v)

    recs = {}
    for ix, f in F.iterrows():
        c = {}
        g = gold_by_pair.get(f["pair"])
        if g is not None:
            g = g[g["page"] != f["page"]]
        true_vals = set() if g is None else set(g["vkey"])
        c["kb_value_true_ever"] = float(f["vkey"] in true_vals)
        c["kb_value_false_only"] = float(f["vkey"] in false_keys[f["pair"]] and f["vkey"] not in true_vals)
        if g is None or g.empty:
            c["kb_rel"] = "unseen_pair"
            c["kb_n_true_values"] = 0.0
            recs[ix] = c
            continue
        c["kb_n_true_values"] = float(g["vkey"].nunique())
        prev = g[g["date"] <= f["date"]]
        fut = g[g["date"] > f["date"]]
        if not fut.empty:
            c["kb_next_matches"] = float(fut["vkey"].iloc[0] == f["vkey"])
            c["kb_days_to_next"] = math.log1p((fut["date"].iloc[0] - f["date"]).days)
        if prev.empty:
            c["kb_rel"] = "future_value" if f["vkey"] in set(fut["vkey"]) else "before_first_known"
            first = g.iloc[0]
            c["kb_ratio_to_latest"] = _log_ratio(f["vnum"], first["vnum"])
            c["kb_op_consistent"] = _op_consistent(f["op"], f["vnum"], first["vnum"])
            recs[ix] = c
            continue
        latest = prev.iloc[-1]
        changes = prev[prev["vkey"] != prev["vkey"].shift()]
        c["kb_days_since_change"] = math.log1p((f["date"] - changes["date"].iloc[-1]).days)
        c["kb_days_since_latest"] = math.log1p((f["date"] - latest["date"]).days)
        c["kb_ratio_to_latest"] = _log_ratio(f["vnum"], latest["vnum"])
        c["kb_abs_ratio_to_latest"] = abs(c["kb_ratio_to_latest"]) if not math.isnan(c["kb_ratio_to_latest"]) else math.nan
        c["kb_op_consistent"] = _op_consistent(f["op"], f["vnum"], latest["vnum"])
        if f["op"] != "eq" and c["kb_op_consistent"] == 1.0:
            c["kb_rel"] = "match_latest"
        elif f["vkey"] == latest["vkey"]:
            c["kb_rel"] = "match_latest"
        elif f["vkey"] in set(prev["vkey"]):
            c["kb_rel"] = "stale_older"
        elif f["vkey"] in set(fut["vkey"]):
            c["kb_rel"] = "future_value"
        else:
            c["kb_rel"] = "contradicts"
        recs[ix] = c
    return pd.DataFrame.from_dict(recs, orient="index").reindex(F.index)


# ---------------------------------------------------------------- row table
def _aggregate(F: pd.DataFrame, feats: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    out = pd.DataFrame(index=range(n_rows))
    if F.empty:
        return out
    num = feats.select_dtypes(include=[np.number])
    grouped = num.groupby(F["row"])
    agg = grouped.agg(["min", "max"])
    agg.columns = [f"{c}_{a}" for c, a in agg.columns]
    out = out.join(agg)
    if "kb_rel" in feats:
        rank = feats["kb_rel"].map({k: i for i, k in enumerate(KB_RELATIONS)})
        worst = rank.groupby(F["row"]).max()
        out["kb_worst_rank"] = worst
        for k in KB_RELATIONS:
            out[f"rel_{k}"] = (feats["kb_rel"] == k).groupby(F["row"]).any().astype(float)
            out[f"rel_{k}"] = out[f"rel_{k}"].fillna(0.0)
        # The same relations over facts whose subject the sentence names: those
        # are more reliable than facts whose subject was guessed from the page.
        anchored = F["anchored"] > 0 if "anchored" in F else pd.Series(True, index=F.index)
        rank_a = rank.where(anchored)
        out["kb_worst_rank_anchored"] = rank_a.groupby(F["row"]).max()
        for k in ("contradicts", "stale_older", "match_latest"):
            hit = ((feats["kb_rel"] == k) & anchored).groupby(F["row"]).any().astype(float)
            out[f"rel_{k}_anchored"] = hit
            out[f"rel_{k}_anchored"] = out[f"rel_{k}_anchored"].fillna(0.0)
        out["anchored_share"] = anchored.astype(float).groupby(F["row"]).mean()
    return out


def row_features(
    rows: Sequence[dict],
    facts: Sequence[Sequence[Fact]],
    known_labels: dict[int, str],
    *,
    F: pd.DataFrame | None = None,
    label_free: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Full feature table, one row per input row (same order)."""
    if F is None:
        F = fact_table(rows, facts)
    if label_free is None:
        label_free = label_free_features(rows, facts, F)
    kb = kb_features(F, rows, known_labels)
    return pd.concat([label_free, _aggregate(F, kb, len(rows))], axis=1)


def label_free_features(rows: Sequence[dict], facts: Sequence[Sequence[Fact]],
                        F: pd.DataFrame | None = None, *, truth_discovery: bool = False,
                        td_kwargs: dict | None = None) -> pd.DataFrame:
    """Features that need no labels. `truth_discovery` adds option B's EM features."""
    if F is None:
        F = fact_table(rows, facts)
    X = text_page_features(rows)
    X["n_facts"] = [float(len(f)) for f in facts]
    X["min_fact_conf"] = [min((x.conf for x in f), default=math.nan) for f in facts]
    for k in KINDS:
        X[f"kind_{k}"] = [float(any(x.value.get("kind") == k for x in f)) for f in facts]
    X["any_op"] = [float(any(x.value.get("op") for x in f)) for f in facts]
    X["n_subjects"] = [float(len({x.subject for x in f})) for f in facts]
    sup = support_features(F)
    parts = [X, _aggregate(F, sup, len(rows))]
    if truth_discovery:
        from app.classify.truth_discovery import truth_discovery_features
        parts.append(truth_discovery_features(F, len(rows), page_ids(rows), **(td_kwargs or {})))
    return pd.concat(parts, axis=1)
