"""Step 4: per-row archetype + recommended architecture option (A-F) + concrete ML algorithms.

The first archetype whose condition matches wins (ordered from most to least decisive).
Outputs: eda/out/practice_annotated.csv, eda/out/archetypes.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eda" / "out"

# (archetype, option, secondary, algorithms, condition(R) -> bool Series, reason)
ARCHETYPES = [
    ("stale_value", "D", "A",
     "CfC/LTC over the pair's irregular timeline (GRU control) -> P(value still current); hard rule stale->false; LightGBM",
     lambda R: R.rel_stale_older == 1,
     "repeats an older value of the pair after it changed: the 'fake strike' (84% false)"),
    ("contradicts_kb", "A", "B",
     "temporal KB lookup feature -> LightGBM + L2 LogReg stack, isotonic calibration",
     lambda R: R.rel_contradicts == 1,
     "value differs from the value in force on its date and never appears as true elsewhere (45% false)"),
    ("hype_joke_tail", "E", "C",
     "spiking MLP (LIF, rate coding) / LSM reservoir on cue flags + TF-IDF ComplementNB cue detector; abstain->unsure",
     lambda R: (R.joke_tail == 1) | (R.hype_score >= 2) | (R.hearsay == 1),
     "first-person hype or joke tail ('and the curve is vertical'): the main source of unsure"),
    ("lying_page", "B", "E",
     "EM truth discovery (Dawid-Skene / TruthFinder) -> page reliability; Beta-Binomial smoothed page trust",
     lambda R: R.page_first_person_share >= 0.3,
     "on a first-person blog/hype page (40% false): page trust decides"),
    ("speaker_quote", "B", "A",
     "speaker reliability (Beta-Binomial / Dawid-Skene annotator model) + TruthFinder; LogReg on speaker x subject",
     lambda R: (R.quote == 1) | R.speaker.notna(),
     "quoted or attributed statement: trust depends on who says it"),
    ("unverifiable_single_value", "C", "A",
     "one-sided rule (never false) + conformal prediction / ensemble disagreement (KNN, NB, ExtraTrees) for unsure",
     lambda R: R.pair_n_values_max == 1,
     "pair has one value in the whole corpus: never false (0/280); unsure when nothing corroborates"),
    ("unsupported_new_value", "B", "C",
     "truth discovery support votes; IsolationForest on value deviation; LightGBM",
     lambda R: (R.rel_before_first_known == 1) & (R.same_value_rows_min == 0),
     "value appears before anything is known for the pair and no other row repeats it"),
    ("change_point", "D", "A",
     "CfC/LTC 'value still current' + survival/hazard features (days since last change, Weibull); GRU control",
     lambda R: R.rel_future_value == 1,
     "introduces the value the pair later holds: a change point, mostly true (10% false)"),
    ("value_outlier", "C", "A",
     "robust z-score / IsolationForest on log(value / pair median) + SVM-RBF; ExtraTrees",
     lambda R: R.log_dev_from_pair_median_max.abs().fillna(0) >= np.log10(2),
     "value is >= 2x or <= 0.5x the pair's median: perturbed number (54% false when inflated)"),
    ("hedged_or_bounded", "C", "A",
     "interval-aware comparison features (op gt/lt/le) + ExtraTrees / SVM soft vote",
     lambda R: (R.hedge == 1) | (R.op_present_max == 1),
     "hedge words or over/under bounds: needs interval logic, 31% false"),
    ("date_reasoning", "D", "A",
     "date-delta features (text date vs page date) + CfC with delta-t; rule: 'today' + mismatched date -> unsure",
     lambda R: (R.today_date_mismatch == 1) | (R.text_date_in_past == 1) | (R.text_date_in_future == 1),
     "the text names a date different from the page date"),
    ("multi_fact", "C", "A",
     "multi-instance aggregation (noisy-OR / worst-fact min) over per-fact scores + LightGBM",
     lambda R: R.n_facts >= 2,
     "several facts in one sentence: one false fact makes the row false"),
    ("pronoun_subject", "F", "A",
     "coreference: nearest-compatible-subject rule -> LogReg antecedent ranker; CRF tagger; MiniLM only if gated",
     lambda R: R.any_pronoun_subject.fillna(False).astype(bool),
     "subject is a pronoun or implicit: extraction (coreference) is the bottleneck"),
    ("spec_fragment", "F", "A",
     "regex value tagger + CRF; template matching (rapidfuzz) against known fragments",
     lambda R: (R.no_period == 1) | (R.kv_line == 1),
     "headline / spec-sheet fragment without a verb ('Captions in 11 languages')"),
    ("plain_kb_match", "A", "-",
     "temporal KB lookup -> LightGBM + L2 LogReg stack, isotonic calibration",
     lambda R: R.rel_match_latest == 1,
     "plain fact that matches the value in force on its date (3% false)"),
    ("default", "A", "-",
     "LightGBM + L2 LogReg stack on all features, isotonic calibration",
     lambda R: pd.Series(True, index=R.index),
     "no stronger signal"),
]


def main():
    R = pd.read_pickle(OUT / "features.pkl")
    R["archetype"] = None
    matches = pd.DataFrame(index=R.index)
    for name, opt, sec, algos, cond, reason in ARCHETYPES:
        hit = cond(R).fillna(False).astype(bool)
        matches[name] = hit
        m = hit & R.archetype.isna()
        R.loc[m, ["archetype", "recommended_option", "secondary_option", "recommended_algorithms", "reason"]] = \
            [name, opt, sec, algos, reason]
    # every archetype the row matches, for transparency
    names = matches.columns[:-1]
    R["all_archetypes"] = matches[names].apply(lambda r: "|".join(names[r.values]), axis=1)

    feature_cols = [
        "kb_relation", "days_since_last_change_min", "n_gold_values_for_pair_max", "pair_n_values_max",
        "same_value_rows_min", "same_value_pages_min", "agree_prev_mention_min", "agree_next_mention_min",
        "log_dev_from_pair_median_max", "round_number_max", "op_present_max",
        "speaker", "first_person", "we_our", "quote", "hearsay", "hedge", "superlative", "joke_tail", "hype_score",
        "now_today", "future", "negation", "no_period", "kv_line", "starts_pronoun", "any_pronoun_subject",
        "text_date_delta_days", "today_date_mismatch", "page", "page_size", "pos_in_page", "page_genre",
        "page_first_person_share", "template_count", "n_facts", "kinds", "subjects", "attributes"]
    cols = ["id", "text", "date", "label"] + feature_cols + [
        "archetype", "recommended_option", "secondary_option", "recommended_algorithms", "reason", "all_archetypes"]
    A = R[cols].copy()
    A["date"] = A["date"].dt.strftime("%Y-%m-%d")
    A.to_csv(OUT / "practice_annotated.csv", index=False, encoding="utf-8")
    assert len(A) == 1897 and A.recommended_option.isin(list("ABCDEF")).all()

    s = R.groupby(["archetype", "recommended_option"]).label.value_counts().unstack(fill_value=0)
    s["n"] = s.sum(axis=1)
    s["p_false"] = (s.get("false", 0) / s.n).round(3)
    s["p_unsure"] = (s.get("unsure", 0) / s.n).round(3)
    order = [a[0] for a in ARCHETYPES if a[0] in set(R.archetype)]
    s = s.reset_index().set_index("archetype").loc[order]
    s.to_csv(OUT / "archetypes.csv")
    print(s.to_string())
    print("\nrows per option:", R.recommended_option.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
