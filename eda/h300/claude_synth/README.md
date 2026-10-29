# Synthetic set v2: hand-written sentences, script-guaranteed labels

Second version of the synthetic training set, next to the scripted one (`eda/h300/synth.py`). Here Claude writes every
sentence; a script decides every label. The writer never sees or sets a label.

Outputs (`eda/h300/out/`):

| file | rows | mix | chain share |
|---|---|---|---|
| `synth10k_claude_practice_rule.csv` | 3,157 | true 82.1 / false 16.2 / unsure 1.7 | 16.4% |
| `synth10k_claude_closed_world.csv` | 3,072 | true 37.8 / unsure 31.6 / false 30.6 | 17.0% |

Columns: `id,text,date,label,form,case_type,chain_length,broken_link,fold,mode,source` (`fold` = `all`,
`source` = `claude`). **Both files fall short of the 10k target, at about 3.1k rows each.** The limit is the training
KB, not writing speed. See "Known limits".

## Method

1. **Specs (`plan.py`)**. Built by script from the CV training rows only: `holdout == 0` and `fold_row5 != 0`.
   The h300 validation and lockbox rows are never read. Plain cases come from `synth.FoldKB` / `synth.make_case`:
   true_current, update, stale, perturbed, premature, swapped, unseen_pair and joke. Chain cases come from
   `chains.build_link_graph` / `chains.chain_cases`, with 1–3 hops, valid / broken_known / broken_unknown /
   unknown_end, and list or nested style.
   - Every spec fixes the facts, the date and the case type.
   - Every spec carries both labels: `synth.LABEL` / `chains.LABELS`, for the practice rule and the closed-world rule.
   - `--top-up` and `--top-up-chains` append extra specs (batches 019–022), deduplicated against the existing pool.
   - `specs/` is the frozen record. A fresh `plan.py` run is not byte-identical, because the KB walk depends on
     set and hash order.
2. **Writing (`written/batch_NNN.jsonl`)**. One sentence per spec, written from `briefs/`. The writer rotates through
   these styles:
   - `headline`, `spec_line`, `release_note`, `press_release`, `interview_quote`, `faq`, `pronoun_followup`;
   - `blog_hype`: a joke case, with a first-person hype tail and no numbers;
   - `chain_list`, `chain_nested`.
3. **Acceptance (`assemble.py`)**. A sentence is kept only if:
   - it names the subject;
   - every non-boolean value parses back (compared with `app.kb.normalize.key`);
   - name values appear verbatim, except the intermediate link entities of nested chains;
   - no extra value remains once lexicon names are masked;
   - every fact is a lexicon subject/attribute;
   - it is not a near-copy of a held-out text (practice + h300; rapidfuzz ratio rounded ≥ 90);
   - it is not an exact duplicate.
4. **Per-mode sampling**. Label quotas match each mode's mix exactly. Inside a label, case types follow the
   `synth` weights until a pool runs dry, and the remainder moves to the other cases of that label.
   - `update` is capped at 202 rows by the training KB.
   - The chain share is drawn on top of the plain rows.
   - `chain_unknown_end` is left out of the practice file. It is labelled unsure but carries `broken_link = 1`, and
     the practice harsh check requires every broken link to be false.

## Acceptance

- Written: 4,340. Accepted: 3,859 (88.9%).
- Rejections:
  - duplicate: 212 (same facts written alike);
  - near-copy: 131;
  - value not parsed back: 56 (mostly inches: the parser returns `in`);
  - extra value: 47 (billion-sized counts, "N devices", "one"/"every" in joke tails);
  - fact outside the lexicon: 35 (chain hops on unlisted attributes).

Per style, accepted / written, from the first pool (the full table is in `assemble_stats.json`):

| style | accepted / written |
|---|---|
| interview_quote | 402/411 |
| pronoun_followup | 398/411 |
| faq | 392/411 |
| press_release | 391/411 |
| release_note | 385/411 |
| headline | 352/411 |
| spec_line | 347/411 |
| blog_hype | 237/249 |
| chain_list | 307/350 |
| chain_nested | 174/231 |

Short styles (headline, spec_line) collide most often with held-out texts and with each other.

## Harsh checks (`eda/tests/harsh_lib.py`, run directly; results in `harsh_results.json`)

| check | practice | closed-world |
|---|---|---|
| schema, forms, dates | pass | pass |
| label mix (±5%) | pass | pass |
| chains (≥15%, broken links false) | pass, 16.4% | pass, 17.0% |
| near-copies of held-out texts | pass (max 89) | pass (max 89) |
| exact duplicates | pass (0%) | pass (0%) |
| parse-back (≥90%) | pass, 98.0% | pass, 97.2% |
| case types ≥ 100 rows | **fail**: broken_known 88, premature 53, joke 53, broken_unknown 28 | **fail**: broken_known 87, broken_unknown 27, unknown_end 26 |
| rule agreement (≥95%) | **fail**: 82.6% | **fail**: 80.3% |

**Rule agreement.** The harsh check rebuilds the timeline from *all* non-holdout rows, which include the
`fold_row5 == 0` rows that check.sh hides. The specs deliberately exclude those rows. As a result, 525 unseen_pair
rows sit on pairs that only the hidden rows state, and the check expects false there. Against the spec KB
(`rule_agreement_full.json`, all rows), agreement is 99.75% (practice) and 98.05% (closed-world).
- In closed-world, the remaining gap is the 53 broken_unknown / unknown_end chains. `expected_row_label` calls every
  broken link false, while `chains.LABELS` and the harsh chain check both allow unsure in closed-world mode.
- No values leak. unseen_pair values are borrowed from other subjects in the training KB.

## Examples

| mode | case | label | date | text |
|---|---|---|---|---|
| practice | true_current | true | 2026-05-21 | The Hooly Pixel 10 offers choice; it comes in 4 colours. |
| practice | update | true | 2026-09-02 | The Hooly Pixel Watch is affordable; it costs €299. |
| practice | stale | false | 2026-04-04 | Hooly Gemini integrations: 9 integrations |
| practice | perturbed | false | 2026-04-25 | Hooly Workspace is generous; its free trial lasts 28 days. |
| practice | swapped | false | 2026-04-02 | When did Hooly Drive launch? On 9 October 2025. |
| practice | joke | unsure | 2026-04-25 | The Hooly Jules beta had 500 testers, and I was the most enthusiastic of them, easily. |
| practice | chain_valid | true | 2026-10-23 | Bertram Gilfoil is the CTO of Hooly and previously worked at Kestrel Bank. |
| closed_world | unseen_pair | unsure | 2026-10-14 | Hooly Meet went big on streaming; its livestream reached 18,000 viewers. |
| closed_world | premature | false | 2026-06-28 | Who is the CTO of Hooly? Bertram Gilfoil. |
| closed_world | chain_broken_unknown | unsure | 2026-07-15 | Hooly Maps acts as executive sponsor of the Hooly Hiring Programme, and Richard Hendrix owns Hooly Search. |

## Known limits

- **Size.** About 3.1k rows per mode instead of 10k. The training KB allows only 220 distinct `update` specs and
  a few hundred chains. Scaling further means repeating facts in new wording, which the duplicate and near-copy
  gates reject or the sampler cannot use.
- **Small case types.** Several case types have fewer than 100 rows (see the table above).
- **Practice case mix.** `update` makes up 6% of the practice file, against a target of 20%; `true_current` fills
  the gap.
- **Phrasing gaps.** Inch sizes, billion-sized counts and "N devices" do not parse back, so those facts are
  under-represented.
- **Joke cues.** Joke tails avoid number words, so the jokes are signalled by first-person hype, not by
  vague quantities.
- **Hidden-row pairs.** unseen_pair rows overlap the pairs that check.sh hides. That is correct under the
  training-KB labels, but a model that reads these rows learns "unstated means true" on exactly those pairs.
