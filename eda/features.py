"""Shared feature builders for the EDA scripts.

Three groups, by what they may see:
  text_flags(rows)            text + date only (label-free, safe everywhere)
  support_features(facts)     all rows' facts, no labels (label-free "universe" support)
  kb_features(facts, train)   gold timeline built ONLY from rows in `train` (fold-safe)
"""
import re

import numpy as np
import pandas as pd

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], 1)}
NUM_WORDS = r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|hundred)\b"

PATTERNS = {
    "first_person": r"\b(I|I'm|I've|me|my|mine)\b",
    "we_our": r"\b(we|our|us)\b",
    "quote": r"[\"“”]|\bsaid\b|\bsays\b|\btold\b",
    "hearsay": r"\b(I have been told|told me|reportedly|apparently|rumou?r|I hear|they say|word is)\b",
    "hedge": r"\b(about|around|nearly|almost|roughly|over|more than|up to|under|less than|at least|at most|some)\b",
    "superlative": r"\b(\w+est|most|best|first|only|record|ever|any competitor|more than any|all humans|nobody|everyone|flawless|bargain|vertical)\b",
    "now_today": r"\b(now|today|from today|this week|as of)\b",
    "still": r"\bstill\b",
    "future": r"\b(will|is due|are due|plans? to|next|soon|coming|upcoming|from \d)\b",
    "negation": r"\b(not|no|never|none|without|n't)\b",
    "change_verb": r"\b(grows?|rises?|drops?|falls?|cuts?|raises?|increases?|doubles?|moves?|switch(?:es)?|replaces?|up from|down from|was)\b",
    "money_sym": r"[€$£]|\bEUR\b|\bUSD\b",
    "number_words": NUM_WORDS,
}

JOKE_TAIL = re.compile(
    r"(,| and | while | which | but ).*\b(I|me|my|nobody|everyone|all humans|curve|tired|lunch|coffee|"
    r"personal best|poetry|flawless|learning|impossible|higher|vertical|bargain)\b", re.I)


def parse_text_dates(text):
    """Return list of (year, month, day|None) for dates written in the text."""
    out = []
    t = text.lower()
    for d, m, y in re.findall(r"\b(\d{1,2}) (" + "|".join(MONTHS) + r") (\d{4})", t):
        out.append((int(y), MONTHS[m], int(d)))
    for m, d, y in re.findall(r"\b(" + "|".join(MONTHS) + r") (\d{1,2}), (\d{4})", t):
        out.append((int(y), MONTHS[m], int(d)))
    if not out:
        for m, y in re.findall(r"\b(" + "|".join(MONTHS) + r") (\d{4})", t):
            out.append((int(y), MONTHS[m], None))
    return out


def text_flags(rows):
    X = pd.DataFrame(index=rows.index)
    txt = rows["text"].fillna("")
    for k, p in PATTERNS.items():
        flags = 0 if k in ("first_person",) else re.I
        X[k] = txt.str.contains(p, flags=flags, regex=True).astype(int)
    X["joke_tail"] = txt.map(lambda s: int(bool(JOKE_TAIL.search(s)))).astype(int)
    X["no_period"] = (~txt.str.strip().str.endswith((".", "!", "?", '"', "”"))).astype(int)
    X["kv_line"] = txt.str.contains(r"^[^.]{2,40}: ", regex=True).astype(int)
    X["n_chars"] = txt.str.len()
    X["n_words"] = txt.str.split().str.len()
    X["n_numbers"] = txt.str.count(r"\d[\d,.]*")
    X["n_commas"] = txt.str.count(",")
    X["starts_pronoun"] = txt.str.contains(r"^(He|She|It|Its|They|Their|His|Her|This|That)\b", regex=True).astype(int)
    X["hype_score"] = X[["first_person", "superlative", "joke_tail", "hearsay"]].sum(axis=1)

    # date written in text vs the row's (page) date
    deltas, today_mismatch = [], []
    for s, d in zip(txt, rows["date"]):
        ds = parse_text_dates(s)
        if not ds:
            deltas.append(np.nan)
            today_mismatch.append(0)
            continue
        y, m, dd = ds[0]
        td = pd.Timestamp(y, m, dd or 1)
        delta = (td - d).days if dd else (td.to_period("M") - d.to_period("M")).n * 30
        deltas.append(delta)
        today_mismatch.append(int(bool(re.search(r"\btoday\b", s, re.I)) and dd is not None and delta != 0))
    X["text_date_delta_days"] = deltas
    X["has_text_date"] = X["text_date_delta_days"].notna().astype(int)
    X["text_date_in_past"] = (X["text_date_delta_days"] < -1).astype(int)
    X["text_date_in_future"] = (X["text_date_delta_days"] > 1).astype(int)
    X["today_date_mismatch"] = today_mismatch
    return X


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def support_features(facts):
    """Label-free: how the rest of the corpus talks about the same (subject, attribute)."""
    F = facts.copy()
    F["vnum"] = F["value_raw"].map(_num)
    key = ["subject", "attribute"]
    F["pair_rows"] = F.groupby(key)["id"].transform("nunique")
    F["pair_n_values"] = F.groupby(key)["value"].transform("nunique")
    F["same_value_rows"] = F.groupby(key + ["value"])["id"].transform("nunique") - 1
    F["value_share"] = (F["same_value_rows"] + 1) / F["pair_rows"]
    F["same_value_pages"] = F.groupby(key + ["value"])["page"].transform("nunique") - 1
    med = F.groupby(key)["vnum"].transform("median")
    F["log_dev_from_pair_median"] = np.log10((F["vnum"].abs() + 1e-9) / (med.abs() + 1e-9)).where(F["vnum"].notna())
    F["round_number"] = F["vnum"].map(lambda v: int(v != 0 and v == round(v) and (v % 10 == 0)) if pd.notna(v) else 0)
    F["op_present"] = F["op"].notna().astype(int)

    # label-free temporal context: the value the corpus mentioned most recently before this date
    F = F.sort_values(["date", "num"])
    agree_prev, agree_next, n_vals_before = [], [], []
    for _, g in F.groupby(key, sort=False):
        dates, vals = g["date"].values, g["value"].values
        for i in range(len(g)):
            before = vals[dates < dates[i]]
            after = vals[dates > dates[i]]
            agree_prev.append(np.nan if len(before) == 0 else float(before[-1] == vals[i]))
            agree_next.append(np.nan if len(after) == 0 else float(after[0] == vals[i]))
            n_vals_before.append(len(set(before)))
    idx = np.concatenate([g.index.values for _, g in F.groupby(key, sort=False)])
    F.loc[idx, "agree_prev_mention"] = agree_prev
    F.loc[idx, "agree_next_mention"] = agree_next
    F.loc[idx, "n_values_before"] = n_vals_before
    return F.sort_index()


def kb_features(facts, train_ids):
    """Gold timeline from TRUE facts of rows in train_ids only. Relation of each fact to it."""
    F = facts.sort_values(["date", "num"])
    gold = F[F["id"].isin(train_ids) & (F["label"] == "true")]
    gold_by_pair = {k: g for k, g in gold.groupby(["subject", "attribute"])}
    rel, days_since, gold_vals = [], [], []
    for r in F.itertuples():
        g = gold_by_pair.get((r.subject, r.attribute))
        if g is not None:
            g = g[g["id"] != r.id]
        if g is None or len(g) == 0:
            rel.append("unseen_pair"); days_since.append(np.nan); gold_vals.append(0)
            continue
        gold_vals.append(g["value"].nunique())
        prev = g[g["date"] <= r.date]
        fut = g[g["date"] > r.date]
        if len(prev) == 0:
            rel.append("future_value" if r.value in set(fut["value"]) else "before_first_known")
            days_since.append(np.nan)
            continue
        latest = prev.iloc[-1]
        # date of the last change before this row
        changes = prev[prev["value"] != prev["value"].shift()]
        days_since.append((r.date - changes["date"].iloc[-1]).days)
        if r.value == latest["value"]:
            rel.append("match_latest")
        elif r.value in set(prev["value"]):
            rel.append("stale_older")
        elif r.value in set(fut["value"]):
            rel.append("future_value")
        else:
            rel.append("contradicts")
    out = pd.DataFrame({"kb_relation": rel, "days_since_last_change": days_since,
                        "n_gold_values_for_pair": gold_vals}, index=F.index)
    return out.reindex(facts.index)


KB_REL_ORDER = ["match_latest", "future_value", "before_first_known", "unseen_pair", "contradicts", "stale_older"]


def facts_to_rows(F, cols_num, rel_col=None):
    """Aggregate fact-level features to row level (worst fact wins for relations)."""
    agg = F.groupby("id")[cols_num].agg(["min", "max"])
    agg.columns = [f"{c}_{a}" for c, a in agg.columns]
    if rel_col is not None:
        rank = F[rel_col].map({k: i for i, k in enumerate(KB_REL_ORDER)})
        worst = rank.groupby(F["id"]).max().map(dict(enumerate(KB_REL_ORDER)))
        agg[rel_col] = worst
        for k in KB_REL_ORDER:
            agg[f"rel_{k}"] = (F[rel_col] == k).groupby(F["id"]).any().astype(int)
    return agg
