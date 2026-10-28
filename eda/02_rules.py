"""Step 2: rule discovery. Tests each candidate rule with support, false/unsure rate and a Wilson CI,
and keeps a rule as 'confirmed' only if it holds on >= 2 pages.

Outputs (eda/out/): features.pkl, rules.csv, unsure_tree.txt, false_tree.txt, templates.csv,
speakers.csv, genres.csv, attributes.csv, subject_cooccurrence.csv, same_text_diff_date.csv, facts_features.pkl,
figs/*.png
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, str(Path(__file__).parent))
from features import facts_to_rows, kb_features, support_features, text_flags  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eda" / "out"
FIGS = OUT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
BASE_FALSE = None


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def rule_row(name, family, mask, R):
    s = R[mask]
    n = len(s)
    kf, ku, kt = (s.label == "false").sum(), (s.label == "unsure").sum(), (s.label == "true").sum()
    lo, hi = wilson(kf, n)
    pages_hit = s.loc[s.label != "true", "page"].nunique()
    return {"rule": name, "family": family, "support": n, "pages": s.page.nunique(),
            "true": kt, "false": kf, "unsure": ku,
            "p_false": round(kf / n, 3) if n else np.nan,
            "p_unsure": round(ku / n, 3) if n else np.nan,
            "false_ci95": f"[{lo:.2f}, {hi:.2f}]" if n else "",
            "lift_false": round((kf / n) / BASE_FALSE, 2) if n else np.nan,
            "non_true_pages": pages_hit,
            "confirmed": bool(n >= 5 and pages_hit >= 2)}


def speaker_of(text):
    m = re.search(r"said ([A-Z][\w'-]+(?: [A-Z][\w'-]+)?)", text) or \
        re.search(r"([A-Z][\w'-]+(?: [A-Z][\w'-]+)?),? (?:said|says|told|recalls)", text) or \
        re.search(r"^([A-Z][\w'-]+(?: [A-Z][\w'-]+)?): ", text)
    return m.group(1) if m else None


def template_of(text, names):
    t = text
    for n in names:
        t = t.replace(n, "<S>")
    t = re.sub(r"[€$£]?\d[\d,.]*\s?%?", "<N>", t)
    t = re.sub(r"\b(" + "|".join(["January", "February", "March", "April", "May", "June", "July", "August",
                                    "September", "October", "November", "December"]) + r")\b", "<M>", t)
    t = re.sub(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b", "<C>", t)  # remaining capitalised names
    return re.sub(r"\s+", " ", t).strip().lower()


def page_genre(g):
    txt = " ".join(g.text).lower()
    fp = g.first_person.mean()
    if fp >= 0.3:
        return "first_person_hype"
    if g.quote.mean() >= 0.3:
        return "quote_interview"
    if g.no_period.mean() >= 0.5:
        return "spec_sheet_or_slides"
    if re.search(r"attendees|keynote|sessions|venue|sponsors|registrations", txt):
        return "event_recap"
    if re.search(r"release notes|ships today|is out|is here|\bv?\d+\.\d+\b", txt):
        return "release_notes"
    if re.search(r"joined|started at|answers to|hired|appointed|ceo|cto|head of", txt):
        return "people_announcement"
    if re.search(r"[€$£]|a month|per month|price|plan", txt):
        return "pricing_page"
    return "other_page"


def main():
    global BASE_FALSE
    R = pd.read_pickle(OUT / "rows.pkl")
    F = pd.read_pickle(OUT / "facts.pkl")
    BASE_FALSE = (R.label == "false").mean()

    # ---- features -------------------------------------------------------------------------------
    T = text_flags(R)
    R = pd.concat([R, T], axis=1)
    FS = support_features(F)
    # descriptive KB relation: leave-one-PAGE-out gold timeline (no row sees its own page's labels)
    kb_parts = []
    for p, idx in F.groupby("page").groups.items():
        train_ids = set(R.loc[R.page != p, "id"])
        sub = kb_features(F[F.page.isin([p]) | F.id.isin(train_ids)], train_ids)
        kb_parts.append(sub.loc[idx])
    KB = pd.concat(kb_parts).reindex(F.index)
    FS = pd.concat([FS, KB], axis=1)
    num_cols = ["same_value_rows", "same_value_pages", "value_share", "pair_n_values", "pair_rows",
                "log_dev_from_pair_median", "round_number", "op_present", "agree_prev_mention",
                "agree_next_mention", "n_values_before", "days_since_last_change", "n_gold_values_for_pair"]
    agg = facts_to_rows(FS, num_cols, rel_col="kb_relation")
    R = R.merge(agg, left_on="id", right_index=True, how="left")

    R["speaker"] = R.text.map(speaker_of)
    lex_names = sorted(set(F.subject), key=len, reverse=True)
    R["template"] = R.text.map(lambda s: template_of(s, lex_names))
    R["template_count"] = R.groupby("template")["id"].transform("size")
    genre = R.groupby("page").apply(page_genre, include_groups=False).rename("page_genre")
    R = R.merge(genre, left_on="page", right_index=True)
    R["page_first_person_share"] = R.groupby("page")["first_person"].transform("mean")
    R["page_hype_share"] = R.groupby("page")["hype_score"].transform(lambda s: (s > 0).mean())
    R.to_pickle(OUT / "features.pkl")
    FS.to_pickle(OUT / "facts_features.pkl")

    # ---- rules ----------------------------------------------------------------------------------
    rules = []
    add = lambda name, fam, m: rules.append(rule_row(name, fam, m.fillna(False).astype(bool), R))
    add("ALL ROWS (baseline)", "base", pd.Series(True, index=R.index))
    for k in ["match_latest", "future_value", "before_first_known", "unseen_pair", "contradicts", "stale_older"]:
        add(f"worst fact relation = {k}", "temporal", R.kb_relation == k)
    add("value repeated by >=1 other page (same pair)", "support", R.same_value_pages_min >= 1)
    add("value never repeated anywhere else", "support", R.same_value_rows_min == 0)
    add("value never repeated AND pair has >=3 values", "support", (R.same_value_rows_min == 0) & (R.pair_n_values_max >= 3))
    add("pair has a single value in the corpus", "support", R.pair_n_values_max == 1)
    add("agrees with previous mention (label-free)", "support", R.agree_prev_mention_min == 1)
    add("disagrees with previous mention (label-free)", "support", R.agree_prev_mention_min == 0)
    add("value >= 2x pair median", "value", R.log_dev_from_pair_median_max >= np.log10(2))
    add("value <= 0.5x pair median", "value", R.log_dev_from_pair_median_min <= -np.log10(2))
    add("round number (x10)", "value", R.round_number_max == 1)
    add("comparison op present (over/under)", "value", R.op_present_max == 1)
    for k in ["first_person", "we_our", "quote", "hearsay", "hedge", "superlative", "now_today", "still", "future",
              "negation", "change_verb", "joke_tail", "no_period", "kv_line", "starts_pronoun", "number_words"]:
        add(f"text: {k}", "wording", R[k] == 1)
    add("hype_score >= 2", "wording", R.hype_score >= 2)
    add("first person AND joke tail", "combo", (R.first_person == 1) & (R.joke_tail == 1))
    add("first person AND number", "combo", (R.first_person == 1) & (R.n_numbers > 0))
    add("hedge AND value never repeated", "combo", (R.hedge == 1) & (R.same_value_rows_min == 0))
    add("headline/spec line AND match_latest", "combo", (R.no_period == 1) & (R.kb_relation == "match_latest"))
    add("text has date", "date", R.has_text_date == 1)
    add("text date in the past (> 1 day)", "date", R.text_date_in_past == 1)
    add("text date in the future (> 1 day)", "date", R.text_date_in_future == 1)
    add("'today' but text date != page date", "date", R.today_date_mismatch == 1)
    add("subject only by pronoun/implicit", "extraction", R.any_pronoun_subject == True)  # noqa: E712
    add("n_facts >= 2", "structure", R.n_facts >= 2)
    add("n_facts >= 3", "structure", R.n_facts >= 3)
    add("first row of page (headline)", "structure", R.pos_in_page == 0)
    add("single-row page", "structure", R.page_size == 1)
    add("page first-person share >= 0.3", "page", R.page_first_person_share >= 0.3)
    add("template seen >= 3 times", "template", R.template_count >= 3)
    for g in sorted(R.page_genre.unique()):
        add(f"page genre = {g}", "genre", R.page_genre == g)
    for k in sorted(R.kinds.dropna().unique()):
        if (R.kinds == k).sum() >= 15:
            add(f"value kinds = {k}", "kind", R.kinds == k)
    rules = pd.DataFrame(rules)
    rules.to_csv(OUT / "rules.csv", index=False)

    # ---- speakers, genres, templates ----------------------------------------------------------
    sp = R[R.speaker.notna()].groupby("speaker").label.value_counts().unstack(fill_value=0)
    sp["n"] = sp.sum(axis=1)
    sp.sort_values("n", ascending=False).to_csv(OUT / "speakers.csv")
    gen = R.groupby("page_genre").label.value_counts().unstack(fill_value=0)
    gen["pages"] = R.groupby("page_genre").page.nunique()
    gen["p_false"] = (gen.get("false", 0) / gen[["false", "true", "unsure"]].sum(axis=1)).round(3)
    gen.to_csv(OUT / "genres.csv")
    tp = R.groupby("template").agg(n=("id", "size"), pages=("page", "nunique"),
                                   p_false=("label", lambda s: round((s == "false").mean(), 3)),
                                   p_unsure=("label", lambda s: round((s == "unsure").mean(), 3)),
                                   example=("text", "first"))
    tp.sort_values("n", ascending=False).to_csv(OUT / "templates.csv")

    # same text on different dates: the "fake strike"
    dup = R[R.duplicated("text", keep=False)].sort_values(["text", "date"])
    dup[["id", "date", "label", "kb_relation", "text"]].to_csv(OUT / "same_text_diff_date.csv", index=False)

    # relations: attribute -> kinds, subject co-occurrence on a page
    rel = F.groupby("attribute").agg(kinds=("kind", lambda s: "|".join(sorted(set(s)))),
                                     subjects=("subject", "nunique"), n=("id", "size"),
                                     n_values=("value", "nunique"),
                                     p_false=("label", lambda s: round((s == "false").mean(), 3)))
    rel.sort_values("n", ascending=False).to_csv(OUT / "attributes.csv")
    co = F[["page", "subject"]].drop_duplicates().merge(F[["page", "subject"]].drop_duplicates(), on="page")
    co = co[co.subject_x < co.subject_y].groupby(["subject_x", "subject_y"]).size().rename("pages_together")
    co.sort_values(ascending=False).head(60).to_csv(OUT / "subject_cooccurrence.csv")

    # ---- combination trees ------------------------------------------------------------------------
    flag_cols = ["first_person", "we_our", "quote", "hearsay", "hedge", "superlative", "now_today", "still",
                 "future", "negation", "joke_tail", "no_period", "kv_line", "starts_pronoun", "number_words",
                 "has_text_date", "text_date_in_past", "text_date_in_future", "today_date_mismatch",
                 "rel_match_latest", "rel_future_value", "rel_unseen_pair", "rel_before_first_known",
                 "rel_contradicts", "rel_stale_older", "op_present_max", "round_number_max"]
    X = R[flag_cols].fillna(0).astype(int).copy()
    X["never_repeated"] = (R.same_value_rows_min == 0).astype(int)
    X["pair_single_value"] = (R.pair_n_values_max == 1).astype(int)
    X["disagrees_prev_mention"] = (R.agree_prev_mention_min == 0).astype(int)
    X["page_first_person"] = (R.page_first_person_share >= 0.3).astype(int)
    X["multi_fact"] = (R.n_facts >= 2).astype(int)
    for tgt, fname in [("unsure", "unsure_tree.txt"), ("false", "false_tree.txt")]:
        y = (R.label == tgt).astype(int)
        clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=8, class_weight="balanced", random_state=0).fit(X, y)
        txt = export_text(clf, feature_names=list(X.columns), show_weights=True)
        imp = pd.Series(clf.feature_importances_, X.columns).sort_values(ascending=False).head(10)
        (OUT / fname).write_text(f"target = {tgt}  (weights shown as [not {tgt}, {tgt}] after class balancing)\n\n"
                                 + txt + "\n\nimportance:\n" + imp.round(3).to_string(), encoding="utf-8")
    X.assign(label=R.label.values, id=R.id.values).to_pickle(OUT / "flags.pkl")

    # ---- figures --------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top = rules[(rules.family != "base") & (rules.support >= 10)].sort_values("p_false")
    fig, ax = plt.subplots(figsize=(8, max(6, 0.22 * len(top))))
    ax.barh(top.rule, top.p_false, color="#3b6ea5")
    ax.axvline(BASE_FALSE, color="#c0392b", ls="--", lw=1, label=f"base false rate {BASE_FALSE:.2f}")
    ax.set_xlabel("P(false | rule)")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "rules_false_rate.png", dpi=110)
    plt.close(fig)

    m = R.groupby(R.date.dt.to_period("M")).label.value_counts(normalize=True).unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(m.index.astype(str), m.get("false", 0), marker="o", label="false", color="#c0392b")
    ax.plot(m.index.astype(str), m.get("unsure", 0), marker="o", label="unsure", color="#7f8c8d")
    ax.set_ylabel("share of rows")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "label_share_by_month.png", dpi=110)
    plt.close(fig)

    pd.set_option("display.width", 220)
    print(rules[["rule", "support", "pages", "false", "unsure", "p_false", "p_unsure", "false_ci95", "lift_false",
                 "confirmed"]].to_string(index=False))
    print("\ngenres:\n", gen.to_string())
    print("\nspeakers:\n", sp.sort_values("n", ascending=False).head(15).to_string())
    print("\ntemplates with >=3 rows:", (tp.n >= 3).sum(), "covering", tp.loc[tp.n >= 3, "n"].sum(), "rows")


if __name__ == "__main__":
    main()
