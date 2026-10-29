"""Fact -> row features: failure statuses, misread risk, three-valued sentence logic.

Fact status (from the staircase timeline):
  holds          the value is the one held on the row's date
  outdated       the value was held earlier and has since moved on
  ahead_of_time  the value is only held later
  unbacked       the value is never held
Three-valued fact truth: TRUE (holds and another page or an anchor backs it), FALSE (doesn't hold and another
page or an anchor backs a different value), UNKNOWN otherwise. A sentence is FALSE if any fact is FALSE, TRUE
if all are TRUE, otherwise it has an evidence gap.
Misread risk (sentence level): a range, more numbers than facts, a negation, a hedge or a bound.
"""
import re

import numpy as np
import pandas as pd

RANGE = re.compile(r"\d\s*(?:–|—|-|to)\s*\d|\bbetween\b", re.I)
STATUS_ORDER = ["holds", "ahead_of_time", "unbacked", "outdated"]


def fact_statuses(tl):
    agree = tl["tl_agree"] == 1
    out = np.where(agree, "holds",
                   np.where(tl["tl_was_held_before"] == 1, "outdated",
                            np.where(tl["tl_held_later"] == 1, "ahead_of_time", "unbacked")))
    backed_for = (tl["n_pages_for"] >= 1) | (tl["anchored_for"] == 1)
    backed_against = (tl["n_pages_against"] >= 1) | (tl["forbidden"] == 1)
    tv = np.where(agree & backed_for, "T", np.where(~agree & backed_against, "F", "U"))
    return out, tv


def row_features(fact_ids, tl, trust_by_page=None, fact_pages=None, prefix="tl"):
    """Aggregate timeline fact features to rows. fact_ids aligns with tl rows."""
    num = ["tl_agree", "support_margin", "support_for", "support_against", "n_pages_for", "n_pages_against",
           "days_to_next_stair", "days_since_stair", "n_stairs", "anchored_for", "forbidden",
           "tl_was_held_before", "tl_held_later", "pair_mentions"]
    d = tl[num].copy()
    d["id"] = fact_ids
    g = d.groupby("id")
    agg = pd.concat([g.min().add_suffix("_min"), g.max().add_suffix("_max")], axis=1)
    agg.columns = [f"{prefix}__{c}" for c in agg.columns]
    if trust_by_page is not None:
        t = pd.Series([trust_by_page.get(p, np.nan) for p in fact_pages], index=range(len(fact_ids)))
        agg[f"{prefix}__page_trust"] = t.groupby(np.asarray(fact_ids)).first()
    return agg


def status_features(fact_ids, tl, prefix="fs"):
    status, tv = fact_statuses(tl)
    d = pd.DataFrame({"id": fact_ids, "status": status, "tv": tv})
    out = pd.DataFrame(index=pd.Index(sorted(set(fact_ids)), name="id"))
    for s in STATUS_ORDER:
        out[f"{prefix}__any_{s}"] = (d["status"] == s).groupby(d["id"]).any().astype(float)
    out[f"{prefix}__n_bad"] = (d["status"] != "holds").groupby(d["id"]).sum().astype(float)
    rank = d["status"].map({s: i for i, s in enumerate(STATUS_ORDER)})
    out[f"{prefix}__worst"] = rank.groupby(d["id"]).max().astype(float)
    tvf = (d["tv"] == "F").groupby(d["id"]).any()
    tvt = (d["tv"] == "T").groupby(d["id"]).all()
    out["tv__false"] = tvf.astype(float)
    out["tv__true"] = (~tvf & tvt).astype(float)
    out["tv__gap"] = (~tvf & ~tvt).astype(float)
    return out


def misread_features(rows, X0):
    """Sentence-level misread risk, label-free."""
    txt = rows.set_index("id")["text"].reindex(X0.index).fillna("")
    rng = txt.map(lambda s: float(bool(RANGE.search(s))))
    extra_numbers = (X0["n_numbers"] > X0["n_facts"]).astype(float)
    parts = pd.concat([rng, extra_numbers, X0["negation"].astype(float), X0["hedge"].astype(float),
                       X0["op_present_max"].fillna(0).astype(float)], axis=1)
    out = pd.DataFrame(index=X0.index)
    out["mr__range"] = rng
    out["mr__extra_numbers"] = extra_numbers
    out["mr__count"] = parts.sum(axis=1)
    out["mr__any"] = (out["mr__count"] > 0).astype(float)
    return out
