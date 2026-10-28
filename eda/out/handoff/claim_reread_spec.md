# Claim re-read: checks to run on every claim's text before it ships (for P2, `app/validate`)

**Why:** the referee reads the claim's *text*, not our form. The validator already checks the schema, the lexicon names and the main text rules. This step re-reads each selected text with the same value parser the extractor uses. If the parser reads anything other than exactly the form's value(s), the claim is dropped and the next candidate takes its place. The pool is ~10,000 valid candidates for 1,000 slots, so dropping costs nothing.

**Measured on today's run** (practice input with every fifth label hidden; 1,000 claims selected, 519 compare / 386 change / 95 fact): **8 claims flagged**, listed in `claim_reread_examples.csv` (regenerate with `.venv/Scripts/python eda/08_handoff.py <output.csv>`).

| # | Check (on the text, after masking lexicon names) | Applies to | Protects this rule (contracts) | Today |
|---|---|---|---|---|
| 1 | **Stray number:** any digit, or a number word (`one`…`twelve`, `a dozen`, `hundred`), in a compare sentence | compare | claim-forms.md, Compare: "no value … and no number at all, in digits or in words". Only digits inside a compared subject's *name* are exempt | **5 flagged.** The attribute words `soc2 audit start` carry a digit. Drop them: attribute words must stay word for word, so the digit can't be removed |
| 2 | **Second value of the same kind:** the parser finds two values of the claim's kind (two prices in the same currency and period, two counts of the same unit, two dates) | fact, at_event, relation | claim-forms.md, General rules: "One value, one date" | 0 |
| 3 | **Value as read ≠ form value:** the parser's reading of the value (number, unit, currency, period, date precision) must equal the form exactly; no `over`, `about` or `at least` | fact, at_event, relation, change (both sides) | claim-forms.md: "Values … exactly … no comparison words … no approximation"; value-forms.md | 0 (the templates render via `app/generate/render.py`) |
| 4 | **Negation:** `not`, `never`, `no`, any `…n't` | all except yes/no | claim-forms.md: "Negations" | 0 in non-yes/no claims |
| 5 | **Yes/no wording:** a yes/no claim must hold the attribute's words word for word, and the polarity must be read from the sentence | fact (boolean) | claim-forms.md: "Yes or no … the sentence also holds the fact's attribute in words" | **3 to review by eye** ("The generally available flag of Hooly Vault is not set."). Keep them if the attribute words match `generally_available` exactly; otherwise drop |
| 6 | **Refused time word** (the exact list in claim-forms.md: `ago`, `since`, `until`, `last quarter`, `next year`, …) | fact, at_event, relation, compare (except the compare phrase itself) | claim-forms.md: "No time word" | 0 (the validator already drops them: 274 compare candidates rejected for this today). Note: bare `next`, `latest` and `first` inside attribute words are **not** on the list, so `The next meeting of Hooly Board went from …` is fine |
| 7 | **Name outside the lexicon / another entity named:** no capitalised name left after masking the claim's own subjects | all | contracts/README.md: "A subject no entity has … makes the entry invalid"; claim-forms.md "Names" | 0 (the validator drops these: 14+58+98 today) |
| 8 | **Unit mismatch:** the unit word in the text converts to the form's unit (`GB` vs `MB`, `min` vs `h`) and is singular/plural as value-forms.md writes it | fact, change, at_event | value-forms.md | 0 |

**Implementation sketch** (P2's folder, P2's call):
- In `app/validate/rules.py`, add `reread(claim, lexicon) -> list[str]`:
  - mask the lexicon names (longest first);
  - run the shared value parser (P3's `app/extract` value regexes, once merged; until then, checks 1, 4 and 6 are plain regexes);
  - compare what it reads with `claim.form`.
- Call it after the existing checks in `validate_claim`; a non-empty list means rejected, which counts toward the `rejected` reasons in metrics.
- Test: the 8 ids in `claim_reread_examples.csv` become regression cases. The 5 "soc2" compares must be rejected; the 3 yes/no claims pass only if the attribute words match.
