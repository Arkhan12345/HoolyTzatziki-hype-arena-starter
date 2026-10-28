# Hype or Truth — Team Tzatziki (pitch deck)

Rendered live at `/pitch`. One `## Slide N — title` per slide; `> notes:` lines are speaker notes (press N).
Every number comes from `pitch/STORY.md`; `{{…}}` values are filled from `/metrics` and `pitch/metrics.json`.

## Slide 1 — Hype or Truth

### Team Tzatziki

**The date decides truth.**

A system that reads a year of a company's pages, says which sentences were true *on their day*,
and writes 1,000 new claims that are.

No LLM inside. Offline. Under 5 seconds on 2 CPUs.

> notes: Hi, we're Team Tzatziki. In one line, what we learned is that the date decides truth.
> Our system reads a year of a fictional company's pages, labels every sentence true, false or unsure for the day
> it was written, and writes 1,000 new true claims. There's no LLM inside, and it runs offline in seconds. (STORY R1)

## Slide 2 — The problem

**Hooly**: a year of pages, docs, posts, decks and press releases.

- Prices change. Products ship new versions. People change jobs.
- **Some pages lie**: hype blogs, stage quotes, jokes.
- We get **1,897 labelled sentences** and must label **~18,000 more**: true, false or unsure, with a confidence.
- Then write **exactly 1,000 new claims**, every one true, in a strict format a referee checks word by word.
- The rule: an LLM may help us *build* the system, but may not *be* it.

> notes: Hooly is a made-up company. Its world is a year of pages where facts keep moving and some pages lie.
> We get about 1,900 labelled sentences and have to label about 18,000 more. Then we write 1,000 brand-new claims,
> every one true, phrased so that a strict, deterministic referee accepts them. And the system itself can't be an LLM.
> (STORY D1, D2, D5)

## Slide 3 — What the data taught us

**Same sentence, different date, different truth.**

- *"Hooly Gemini replies in 2.1 s at the median."*
  - 14 Nov 2025: **true**
  - 24 Mar 2026: Gemini 2.0 ships, the median drops to 1.4 s
  - 20 Aug 2026: the same sentence, **false**
- Repeating an old value after it changed is **false 84%** of the time: the "fake strike".
- A fact that only ever had **one value** is **never false** (0 of 280).
- First-person hype pages: **40% false**. Joke tails (*"…and the curve is vertical"*) make rows unsure.
- False numbers are the true ones multiplied by **×2, ×0.5, ×10**: the universe is generated, and it has rules.

> notes: This is the key slide. The exact same sentence about Gemini's reply time is true in November
> and false in August, because Gemini 2.0 brought the median down to 1.4 seconds in March. Repeating a value after it has
> changed is false 84% of the time. We call that the fake strike. If a fact only ever had one value, it's never false.
> Hype pages lie 40% of the time. And the fake numbers aren't random: they're the real number doubled, halved or times ten.
> (STORY T6, T1, T5, S1, S3, G1)

## Slide 4 — Architecture

**Text → facts → timeline → verdict.** The timeline also writes the claims.

1. **Extractor**: names and aliases from the lexicon, value patterns from the contract, an attribute classifier,
   and pronoun resolution ("He", "Its" → the right subject).
2. **Temporal knowledge base**: one timeline per subject and attribute. *What was true on this date?*
3. **Truth classifier**: LightGBM on the timeline features, plus two hard rules
   (stale value → false; single-value fact → never false), then calibrated confidence.
4. **Claim generator + validator**: five claim types, built only from facts the timeline firmly supports,
   checked against the referee's rules, exactly 1,000.
5. **Writer**: a safe table on disk within a second, the final one at the end. Never no output.

> notes: Here's how it runs, and this diagram is the real system, not a mock-up. First we read each sentence into facts:
> subject, attribute and value, including pronouns, since 733 facts name their subject only by "he" or "its". The facts go
> onto a timeline. The classifier asks one question: does this sentence agree with the value in force on its date?
> The same timeline feeds the claim generator, and a validator checks every claim against the referee's rules before it
> counts. (STORY D4; diagram: frontend /architecture)

## Slide 5 — Why no LLM inside, and why that's better

- **It's the rule**: the system may not be an LLM, and the universe's truth isn't the web's.
- **An LLM would bring the wrong facts**: Hooly uses real product names with invented histories.
  A model that "knows" Gemini would be confidently wrong.
- **Explainable**: every label points to the fact and the timeline value it was judged against.
  Every claim points to its source sentences.
- **Offline and tiny**: {{runtime.image.uncompressed_mb}} MB image, {{runtime.docker_synthetic_20k.rss_mb}} MB of memory,
  {{runtime.synthetic_20k.total_s}} s for ~19,000 rows.
- **Auditable**: same input, same output. No sampling, no prompt drift.

> notes: We're not allowed an LLM inside, but we'd have avoided one anyway. Hooly borrows real product names with fake
> histories, so a model that already knows Gemini brings exactly the wrong facts. Our system can explain every label and
> every claim, it runs offline in a few seconds on two CPUs with a 689 MB image, and it gives the same answer every time.
> (STORY R1, R2, R3, C5)

## Slide 6 — Live demo

**The 20 sample rows, live.**

- Upload `sample/sample.csv` on the run page.
- Watch each stage light up on the diagram.
- Read each verdict: label, confidence, and the facts and timeline value behind it.
- Open the generated claims: each one with its type, its form, and the sentences it came from.

> notes: Switch to the frontend. Run page, "Use sample.csv". Point at the diagram while the stages light up.
> Open one true row and one false row and read the explanation aloud. Then the claims tab: open one compare and one change
> and show their sources. Keep it under 45 seconds. If the API is down, the video has the same run.

## Slide 7 — Results

**Labels**
- Shipped classifier, offline evaluation: accuracy **{{offline.data.model.accuracy|pct}}**, macro-F1 **{{offline.data.model.macro_f1|pct}}**.
- Research cross-validation (grouped by page): LightGBM **0.76** macro-F1 against **0.455** for "everything is true",
  and **0.56** for words alone.

**Claims**
- **{{runtime.practice_run.claims_valid}} / 1,000** validator-clean, every one traceable to its source sentences.
- Measured precision: we hide pages from the generator and judge its claims on all of them.
  **{{runtime.claim_holdout.after_same_attribute_compares.wrong_rate.hide_10pct|pct}}** wrong with 10% hidden,
  **{{runtime.claim_holdout.after_same_attribute_compares.wrong_rate.hide_50pct|pct}}** with half hidden.

**Runtime**: {{runtime.synthetic_20k.total_s}} s for ~19,000 rows; `check.sh` green in Docker (offline, user 10001, read-only).

> notes: If the model line shows a dash, the classifier hasn't merged yet. Then say: "Our research model scores 0.76 macro-F1 in
> grouped cross-validation, against 0.455 for the baseline", and move on. For claims, don't just say they're true; say how
> we measured it. We hide pages from the generator, let it write claims, and check them against everything. With half the
> pages gone it's wrong under 2% of the time, and the real run sees all of them. (STORY L1, L2, M1, M5, M6, C1, C3, R1, R4)

## Slide 8 — Confidence is part of the answer

- Every label carries a confidence from 0.5 to 1, and we want it to **mean** something:
  when we say 90%, we should be right about 9 times in 10.
- Our classifier's calibration error (ECE) in cross-validation is **0.046**: its confidence matches how often it's right.
- **Unsure** is rare (2%) and is not "we don't know the subject". We predict it only when the model is truly undecided,
  not as a hiding place.
- Claim confidence is measured the same way: by type and by how recent its evidence is.

> notes: The confidence is part of the score, so we treat it as a measurement, not a decoration. When we say 90%,
> we mean right 9 times out of 10, and in cross-validation our calibration error is under 5 points. Unsure is only 2% of the
> data, so we don't hide in it. (STORY M8, D2; claims: pitch/metrics.json runtime.claim_holdout.confidence_by_type)

## Slide 9 — Model leaderboard: classic vs liquid vs spiking

**Macro-F1 (true/false), cross-validation grouped by page:**

- LightGBM: **0.76** (ships)
- ExtraTrees: **0.72**
- CfC liquid network over each fact's timeline: **0.70**
- Spiking network (LIF neurons): **0.66**
- Words only (TF-IDF): **0.56** · Everything true: **0.455**

**Honest reading:**
- With ~1,900 labels, **trees win**.
- The **knowledge base is the model**: without it LightGBM loses **0.075**.
- **Liquid and spiking nets catch unsure rows** (28–40% recall, against ~0 for trees) but calibrate worse.
  They're candidates for a second opinion, and the energy story for neuromorphic hardware.

> notes: We tried a whole zoo, classic, liquid and spiking, on the same folds. Trees win, which is what you'd expect
> with 1,900 labels. The real lesson is the ablation: take away the date-aware knowledge base and the best model drops by
> 7.5 points. The neural nets are worse overall, but they're the only ones that find unsure rows. That's an honest result,
> and it's where we'd go next. (STORY M1–M4, M5, M6, M7, M9)

## Slide 10 — Team + next steps

**Team Tzatziki**: five people, five Claude Code sessions in parallel, one repo.
- X: research + architecture · Ioan: runtime, knowledge base, claims, API · Arckan: extractor + classifier
- Vald: frontend · Matei: pitch + story

**Next**
- Page trust from truth discovery, for the pages that lie.
- The liquid net's "is this value still current?" as one more feature.
- Use the neural nets' unsure signal without losing calibration.
- Better extraction: the timeline is only as good as the facts we read into it.

**The date decides truth.**

> notes: We're five people who each ran their own Claude Code agent, all in one repo. The LLM helped us build the
> system, and none of it is inside. Next we'd add page trust for the lying pages, the liquid net's "still current"
> signal, and better extraction, since the timeline is only as good as the facts that feed it. Thank you.
