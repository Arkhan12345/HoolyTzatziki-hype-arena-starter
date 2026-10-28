# EDA v2: data rules, new parameters, per-row model advice, bake-off (Team Tzatziki, X)

> Follows [`EDA.md`](EDA.md).
> - Short rule sheet: [`data/RULES.md`](../data/RULES.md).
> - Scripts: `eda/01_parse.py` → `02_rules.py` → `03_recommend.py` → `04_bakeoff.py`, run in `.venv-eda`. Versions are pinned to the runtime image: Python 3.12, numpy 2.1.3, pandas 2.2.3, scikit-learn 1.5.2, lightgbm 4.5.0, plus torch-cpu, ncps and snntorch for dev only.
> - Outputs go to `eda/out/`.

## 1. Why
We have ~1,900 labelled rows to label ~18,000 rows. A model that memorises words will overfit. This universe is generated, so we look for **the generator's rules** and feed those to small models.

## 2. Structure of the data
- **291 documents (pages)** over 159 dates. A page is a contiguous block of ids with one date, and the first row is often a headline.
- **Ids are grouped by document type, not by date.** p-1001…1100 is the hype-blog block: 46% false, and 12 of the 39 unsure rows.
- **2,402 facts.** Every subject and attribute is in the lexicon, so "not in the lexicon" is never a signal.
- **733 facts name their subject only by a pronoun** or not at all.
- **A fixed name pool and templates.** 16 masked sentence templates repeat ≥ 3 times, and the same people keep their roles across pages. This is Faker-like generation.

## 3. Rules found
Full numbers are in `data/RULES.md` and `eda/out/rules.csv`.

| Family | Key rule | Effect |
|---|---|---|
| Time | repeats an older value after it changed ("fake strike") | **84% false** (68 rows) |
| Time | contradicts the value in force on its date | 45% false (218) |
| Time | matches the latest earlier value | 3% false (878) |
| Time | pair has a single value in the whole corpus | **0 false**: true or unsure (280) |
| Value | ≥ 2× the pair's median | 54% false |
| Value | hedge word + value never repeated | 69% false |
| Source | first-person hype page | 40% false |
| Wording | joke tail ("…and the curve is vertical") | 19% unsure (9× base) |
| Wording | superlatives, "now", "still", negation | ≈ base rate: no signal |

How false values are built: a stale value, or the true value multiplied (×2, ×0.5, ×2.5, ×5, ×10).

How unsure rows are built (from the decision tree):
- a single-fact sentence with a joke/hype tail, or
- a value nothing else in the corpus confirms or contradicts.

## 4. Proposed new parameters
All of these are columns in `eda/out/practice_annotated.csv`. Rate features are computed out of fold.

- **Time:**
  - `kb_relation` (match_latest / future_value / before_first_known / unseen_pair / contradicts / stale_older).
  - `days_since_last_change`, `n_gold_values_for_pair`.
- **Label-free support** (usable on unlabelled rows):
  - `pair_n_values`, `same_value_rows`, `same_value_pages`.
  - `agree_prev_mention`, `agree_next_mention`.
- **Value:** `log_dev_from_pair_median`, `round_number`, `op_present`.
- **Source:** `speaker`, `first_person`, `we_our`, `quote`, `hearsay`, `page_first_person_share`, `page_genre`.
- **Wording:** `hedge`, `superlative`, `joke_tail`, `hype_score`, `now_today`, `future`, `negation`, `no_period`, `kv_line`, `starts_pronoun`.
- **Date:** `text_date_delta_days`, `today_date_mismatch`.
- **Structure:** `n_facts`, `pos_in_page`, `page_size`, `template_count`, `any_pronoun_subject`.

## 5. Bake-off (`eda/04_bakeoff.py`, macro-F1 over true/false)
CV schemes:
- **group:** GroupKFold(5) by page.
- **row5:** every 5th row hidden, like `check.sh`.
- **time:** train before 2026-07-01, test after.

The KB features are rebuilt per fold from the training labels only.

| Model | group | row5 | time | acc | ECE |
|---|---|---|---|---|---|
| **LightGBM (unweighted)** | **0.762** | **0.765** | 0.621 | 0.879 | 0.046 |
| LightGBM (balanced) | 0.747 | 0.760 | **0.637** | 0.843 | 0.045 |
| ExtraTrees | 0.721 | 0.750 | 0.625 | 0.845 | 0.064 |
| CfC liquid (timeline) | 0.696 | 0.709 | 0.612 | 0.789 | 0.097 |
| Stack LGBM + LogReg | 0.677 | 0.689 | 0.586 | 0.707 | 0.206 |
| Spiking MLP (LIF) | 0.656 | 0.684 | 0.539 | 0.746 | 0.163 |
| SVM-RBF | 0.652 | 0.671 | 0.634 | 0.849 | 0.017 |
| LogReg L2 | 0.636 | 0.650 | 0.555 | 0.687 | 0.114 |
| KNN-15 | 0.631 | 0.608 | 0.585 | 0.860 | 0.044 |
| TF-IDF + LogReg (text only) | 0.560 | 0.575 | 0.514 | 0.729 | 0.010 |
| All true | 0.455 | 0.455 | 0.457 | 0.833 | 0.167 |

**Ablation** (LightGBM, group CV): removing the temporal KB family costs **−0.075**, page/genre −0.015 and value shape −0.006. Date, wording and label-free support add nothing on top of the KB.

What this means:
1. **The date-aware knowledge base is the model.** Words alone (TF-IDF) score 0.56, and the KB features raise that to 0.76. Improving extraction (to feed the KB) is worth more than more models.
2. **Unweighted LightGBM wins.** Class weighting makes the stack and LogReg over-predict unsure: 200+ predictions against 39 true unsure rows. Use unweighted models plus a tuned unsure threshold.
3. **The neural models are behind on accuracy but ahead on unsure.** CfC and spiking reach 0.28–0.40 unsure recall, against ~0 for trees, but calibrate worse. Use them as an extra feature or ensemble member (options D/E), not as the main classifier.
4. **Time-forward is the pessimistic bound** (0.62 vs 0.76). Later facts have no timeline when only earlier labels are known. The real run sees all practice labels, so expect something between the two.
5. **Per archetype** (noisy, few falses each):
   - CfC wins `lying_page` and `stale_value`.
   - LightGBM wins `contradicts_kb` and `unsupported_new_value`.
   - ExtraTrees wins `value_outlier`.
   - `unverifiable_single_value` (0 false, 13 unsure) is unsolved by every model. Handle it with a rule ("never false") plus abstention.

## 6. Per-row recommendation (A–F)
`eda/out/practice_annotated.csv` holds, for every row:
- `archetype`, `recommended_option`, `secondary_option`, `recommended_algorithms`, `reason`;
- `all_archetypes`, every archetype the row matches.

Summary: `eda/out/archetypes.csv`.

| Archetype | Rows | False | Unsure | Option | Algorithms |
|---|---|---|---|---|---|
| stale_value | 68 | 84% | 0% | D | CfC/LTC timeline, rule stale→false |
| contradicts_kb | 218 | 45% | 1% | A | KB lookup → LightGBM + LogReg, isotonic |
| hype_joke_tail | 45 | 7% | 22% | E | spiking MLP / LSM + cue NB, abstain→unsure |
| lying_page | 26 | 19% | 0% | B | Dawid-Skene / TruthFinder page trust |
| speaker_quote | 136 | 8% | 1% | B | speaker reliability (Beta-Binomial) |
| unverifiable_single_value | 232 | 0% | 6% | C | one-sided rule + conformal abstention |
| unsupported_new_value | 106 | 31% | 3% | B | support votes, IsolationForest |
| change_point | 361 | 12% | 1% | D | CfC "still current", hazard features |
| value_outlier | 51 | 4% | 4% | C | robust z / IsolationForest, SVM-RBF, ExtraTrees |
| hedged_or_bounded | 29 | 10% | 0% | C | interval features, ExtraTrees/SVM vote |
| date_reasoning | 31 | 3% | 0% | D | date-delta features, CfC with Δt |
| multi_fact | 134 | 3% | 0% | C | noisy-OR / worst-fact aggregation |
| pronoun_subject | 146 | 5% | 0% | F | coreference ranker, CRF |
| spec_fragment | 71 | 3% | 1% | F | regex tagger + CRF, template matching |
| plain_kb_match | 227 | 4% | 0% | A | KB lookup → LightGBM stack |
| default | 16 | 0% | 6% | A | LightGBM stack |

Rows per option: A 461 · B 268 · C 446 · D 460 · E 45 · F 217.
`speaker_quote` also catches "Label: value" lines ("Recruiter: …"), so it is broader than real quotes.

## 7. Recommended mix
- **Ship:** option **A**, with unweighted LightGBM on the temporal-KB features plus a tuned unsure threshold. Add these hard rules:
  - a stale value → false;
  - a single-value pair → never false.
- **Try next:** option **B**, page trust from truth discovery, for the hype pages. Add the CfC "still current" probability as one feature (option **D**), since it wins on stale values and lying pages.
- **Pitch only:** option **E** (spiking), for its unsure recall and energy story.
- **Extraction:** option **F**. Its gate still applies: it waits for QA DNA's answer on pretrained encoders.
