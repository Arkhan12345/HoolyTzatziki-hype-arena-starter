"""Option B: unsupervised truth discovery (TruthFinder / Dawid-Skene style EM).

Sources are pages; claims are extracted facts. Two quantities are estimated
in turn until they settle:

  value belief   p(fact) = (support + lam * r(own page)) / (weight + lam)
                 support = sum over facts of the SAME (subject, attribute) on
                 OTHER pages that state the same value, weighted by their
                 page's reliability and by time closeness
                 exp(-|days apart| / tau); weight = the same sum over every
                 value. With no neighbours, p = r(own page).
  page reliability  r(page) = (sum of p over the page's facts + beta * r0)
                            / (n facts + beta)

Time decay matters because values legitimately change: a mention a year away
says little about what was true today.

Uses no labels, so it is fold-safe and runs on the whole universe.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

PRIOR_RELIABILITY = 0.8


def run_truth_discovery(
    F: pd.DataFrame,
    *,
    tau_days: float = 90.0,
    lam: float = 1.0,
    beta: float = 2.0,
    iterations: int = 15,
) -> tuple[pd.Series, pd.Series]:
    """Return (belief per fact, indexed like F; reliability per page)."""
    if F.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    pages = F["page"].to_numpy()
    page_codes, page_index = np.unique(pages, return_inverse=True)
    n_pages = len(page_codes)

    # Per (subject, attribute): time-decay weights between facts on different
    # pages, and which pairs of facts state the same value.
    blocks = []
    for _, g in F.groupby("pair", sort=False):
        pos = F.index.get_indexer(g.index)
        days = g["date"].to_numpy().astype("datetime64[D]").astype(float)
        w = np.exp(-np.abs(days[:, None] - days[None, :]) / tau_days)
        pg = page_index[pos]
        w[pg[:, None] == pg[None, :]] = 0.0
        same = g["vkey"].to_numpy()[:, None] == g["vkey"].to_numpy()[None, :]
        blocks.append((pos, w, w * same))

    rel = np.full(n_pages, PRIOR_RELIABILITY)
    belief = np.full(len(F), PRIOR_RELIABILITY)
    for _ in range(iterations):
        r_fact = rel[page_index]
        for pos, w, w_same in blocks:
            r = r_fact[pos]
            support = w_same @ r
            weight = w @ r
            belief[pos] = (support + lam * r) / (weight + lam)
        sums = np.bincount(page_index, weights=belief, minlength=n_pages)
        counts = np.bincount(page_index, minlength=n_pages)
        new_rel = (sums + beta * PRIOR_RELIABILITY) / (counts + beta)
        if np.max(np.abs(new_rel - rel)) < 1e-5:
            rel = new_rel
            break
        rel = new_rel

    return (pd.Series(belief, index=F.index, name="td_belief"),
            pd.Series(rel, index=page_codes, name="td_page_reliability"))


def truth_discovery_features(F: pd.DataFrame, n_rows: int, page_of_row: list[int],
                             **kwargs) -> pd.DataFrame:
    """Row features: belief in the row's facts and its page's reliability.

    Page reliability is also given leave-this-row-out, so a row's own facts do
    not vouch for themselves.
    """
    out = pd.DataFrame(index=range(n_rows), dtype=float)
    if F.empty:
        return out
    belief, rel = run_truth_discovery(F, **kwargs)
    by_row = belief.groupby(F["row"])
    out["td_belief_min"] = by_row.min()
    out["td_belief_mean"] = by_row.mean()
    out["td_belief_max"] = by_row.max()

    page_sum = belief.groupby(F["page"]).sum()
    page_n = F.groupby("page").size()
    row_sum = by_row.sum()
    row_n = by_row.size()
    pages = pd.Series(page_of_row)
    out["td_page_reliability"] = pages.map(rel).astype(float)
    loo_sum = pages.map(page_sum).fillna(0) - row_sum.reindex(range(n_rows)).fillna(0)
    loo_n = pages.map(page_n).fillna(0) - row_n.reindex(range(n_rows)).fillna(0)
    out["td_page_reliability_loo"] = (loo_sum + 2.0 * PRIOR_RELIABILITY) / (loo_n + 2.0)
    out["td_page_n_facts_log"] = np.log1p(pages.map(page_n).fillna(0).astype(float))
    out["td_belief_vs_page"] = out["td_belief_min"] - out["td_page_reliability_loo"]
    return out
