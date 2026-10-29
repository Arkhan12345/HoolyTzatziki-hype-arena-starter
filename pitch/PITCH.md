# Hype or Truth — Team Tzatziki (pitch deck)

Rendered live at `/pitch`. One `## Slide N — title` per slide; `> notes:` lines are speaker notes (press N).
Every number comes from `pitch/STORY.md`; `{{…}}` values are filled from `/metrics` and `pitch/metrics.json`.
The deck is told in chapters: 1 the problem · 2 how we planned · 3 what the data taught us · 4 the system ·
5 how we built the label engine · 6 how we guard against overfitting · 7 how the system improves itself · 8 results · 9 team.

## Slide 1 — Hype or Truth

### Team Tzatziki

**The date decides truth.**

A system that reads a year of a company's pages, says which sentences were true *on their day*,
and writes 1,000 new claims that are.

No LLM inside. Offline. Under 5 seconds on 2 CPUs.

> notes: Hi, we're Team Tzatziki. In one line, what we learned is that the date decides truth.
> Our system reads a year of a fictional company's pages, labels every sentence true, false or unsure for the day
> it was written, and writes 1,000 new true claims. There's no LLM inside, and it runs offline in seconds. (STORY R1)

## Slide 2 — Chapter 1 · The problem

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

## Slide 3 — Chapter 2 · How we planned

**Six architectures on paper before a line of model code — then one fair race.**

- **A** classic core: facts → date-aware knowledge base → LightGBM, calibrated.
- **B** + truth discovery: how reliable is each page?
- **C** + a classic model zoo · **D** + a liquid network's view of each fact's timeline.
- **E** + spiking and liquid ensemble · **F** a small text encoder, only if extraction is the bottleneck.
- Every option is scored on **the same folds, grouped by page**, with the same metrics: macro-F1, calibration, runtime.
- **Ship rule**: the best score that also passes the 2-CPU runtime gate. The fallback is always option A.

> notes: Before writing model code we planned six architectures, each building on the one before, from a classic core to a
> neuromorphic ensemble. We agreed one rule up front: everything races on the same folds, grouped by page so no page
> leaks into its own test, and the winner has to fit the runtime limits. That plan is what kept five people and five agents
> pulling in one direction. (README architecture options; docs/MODELS.md)

## Slide 4 — Chapter 3 · What the data taught us

**Same sentence, different date, different truth.**

- *"Hooly Gemini replies in 2.1 s at the median."*
  - 14 Nov 2025: **true**
  - 24 Mar 2026: Gemini 2.0 ships, the median drops to 1.4 s
  - 20 Aug 2026: the same sentence, **false**
- Repeating an old value after it changed is **false 84%** of the time: the "fake strike".
- A fact that only ever had **one value** is **never false** (0 of 280).
- First-person hype pages: **40% false**. Joke tails (*"…and the curve is vertical"*) make rows unsure.

> notes: This is the key slide. The exact same sentence about Gemini's reply time is true in November
> and false in August, because Gemini 2.0 brought the median down to 1.4 seconds in March. Repeating a value after it has
> changed is false 84% of the time. We call that the fake strike. If a fact only ever had one value, it's never false.
> Hype pages lie 40% of the time. (STORY T6, T1, T5, S1, S3)

## Slide 5 — Chapter 3 · The universe has rules

**We read the generator, not just the sentences.**

- **Pages are documents**: 291 of them over 159 dates; the first line is often a headline.
- **A fixed cast and fixed templates**: the same names keep their roles; sentences are filled-in patterns.
- **False numbers are made, not random**: the true value **×2, ×0.5, ×2.5, ×5, ×10**.
- **Unsure has a recipe**: a joke or hype tail, or a value nothing else in the year can confirm.
- Every rule is written down with its evidence and tested on ≥ 2 pages, so one odd page can't create a rule.

> notes: With only 1,900 labels, memorising words would overfit. So we looked for the rules the universe was built with.
> Pages behave like documents, the cast and templates repeat, and the fake numbers are the real ones multiplied. Each
> rule carries its evidence and must hold on at least two pages before we trust it. (STORY G1, D3, docs/RULES.md)

## Slide 6 — Chapter 4 · The system

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

## Slide 7 — Chapter 5 · Building the label engine, one piece at a time

**Each piece is added on its own, so we can see what it buys** (macro-F1, true/false, pages held out, **with perfect facts**):

- Plain LightGBM on the knowledge base: **0.735**
- + **Staircase timeline** (the value held on each date, from every mention; a new value needs two pages): **0.789**
- + **Page trust** (reliable pages vote louder): **0.792**
- + **Named failure types** (outdated · ahead of time · unbacked) and a misread-risk flag: **0.799**
- + **Hard rules** from the data: **0.806**
- + **Evidence-sum calibrator**: **0.801**, with calibration error **0.050** (was 0.060)
- The timeline alone brings **+0.054**; the whole ladder **+0.071**, 95% interval [+0.042, +0.103] over 248 pages.
- **But** with the facts our extractor really reads, the ladder loses to plain trees (85.8% vs 86.7%): the next slide is how we found out.

> notes: We never threw everything in at once. Each piece goes on top of the last, on the same folds, so its gain is
> visible. The staircase timeline is the big one: it rebuilds what was true on every date from all the mentions, and a
> single page can't move it. Trust, failure types and rules add smaller steps, and the calibrator makes the confidence
> honest. With perfect facts the gain holds when we resample whole pages, but with real extraction it doesn't, which is why
> we test everything the next way. (STORY I1, I2, I3, I4, O7)

## Slide 8 — Chapter 6 · How we guard against overfitting

**1,900 labels is small. So every result has to survive these checks:**

- **Pages and subjects never split** between training and validation; every choice is made on inner folds.
- **Nested selection**: a change is decided on four folds and must still win on the fifth, unseen one.
- **Label shuffle**: with shuffled labels the model falls to **0.455**, the everything-true level: nothing leaks.
- **A frozen outside set** (662 rows never seen: unseen subjects, the last six weeks, unseen pages), scored once:
  **84.6%** vs **81.9%** for everything-true.
- **It caught us twice**: rules that won on the folds that chose them (**0 of 16** kept), and a model that gained
  **+0.093** in cross-validation but lost on every outside slice. Both were rejected.

> notes: With so few labels, the danger is fooling ourselves. So nothing is kept unless it survives the nested check, the shuffle
> test and a frozen outside set we score once. And it works: it threw out sixteen rules that only won where they were chosen,
> and a whole second model that looked better in cross-validation but lost on data it had never seen. What we ship is what
> survived. (STORY O1, O4, O5, O6)

## Slide 9 — Chapter 7 · The system improves itself, within rules it can't change

**A self-improvement loop with a goal file it is not allowed to edit.**

- The goal file fixes the metric, the acceptance rule and the time budget; the loop checks it is untouched.
- It tries **one change at a time**: timeline window, tree settings, features, calibrators, combinations, dataset rules.
- A change is **kept only if** it wins on at least 4 of 5 folds **and** passes every overfitting check.
- **Honest result**: with perfect facts the loop lifted macro-F1 from **0.735 to 0.819**; with the facts we really
  extract it did not beat plain trees. So the loop's real job became saying **no**: everything is logged, kept or dropped, and why.

> notes: Instead of hand-tuning, we let the system try changes itself inside strict rules. With perfect facts it found real gains,
> but when we fed it the facts our extractor actually reads, those gains disappeared, and the loop rejected them. We'd rather show
> you that than a number that wouldn't survive tomorrow's data. (STORY I1, O7)

## Slide 10 — Why no LLM inside, and why that's better

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

## Slide 11 — Live demo

**The 20 sample rows, live.**

- Upload `sample/sample.csv` on the run page.
- Watch each stage light up on the diagram.
- Read each verdict: label, confidence, and the facts and timeline value behind it.
- Open the generated claims: each one with its type, its form, and the sentences it came from.

> notes: Switch to the frontend. Run page, "Use sample.csv". Point at the diagram while the stages light up.
> Open one true row and one false row and read the explanation aloud. Then the claims tab: open one compare and one change
> and show their sources. Keep it under 45 seconds. If the API is down, the video has the same run.

## Slide 12 — Chapter 8 · Results

**Labels**
- Shipped classifier, offline evaluation: accuracy **{{offline.data.model.accuracy|pct}}**, macro-F1 **{{offline.data.model.macro_f1|pct}}**.
- Honest test, a frozen outside set the model never saw: **84.6%** accuracy vs **81.9%** for "everything is true"
  (macro-F1 **0.654** vs 0.455); unseen pages **86.5%**, unseen subjects **83.7%**.
- Baselines, always with their rows: everything-true is **83.3%** on all 1,897 rows and **85.3%** on the 380 rows check.sh hides.

**Claims**
- **{{runtime.practice_run.claims_valid}} / 1,000** validator-clean, every one traceable to its source sentences.
- Measured precision: we hide pages from the generator and judge its claims on all of them.
  **{{runtime.claim_holdout.after_same_attribute_compares.wrong_rate.hide_10pct|pct}}** wrong with 10% hidden,
  **{{runtime.claim_holdout.after_same_attribute_compares.wrong_rate.hide_50pct|pct}}** with half hidden.

**Runtime**: {{runtime.synthetic_20k.total_s}} s for ~19,000 rows; `check.sh` green in Docker (offline, user 10001, read-only).

> notes: If the model line shows a dash, the classifier hasn't merged yet; then quote the research numbers and move on.
> Always say which rows a baseline is on. For claims, don't just say they're true; say how we measured it: hide pages,
> let the generator write, check against everything. (STORY L1, L2, I1, M5, M6, B1, C1, C3, R1, R4)

## Slide 13 — Confidence is part of the answer

- Every label carries a confidence from 0.5 to 1, and we want it to **mean** something.
- In cross-validation the label engine is right **66% / 88% / 94% / 99%** of the time in its
  0.50–0.70 / 0.70–0.85 / 0.85–0.95 / 0.95–1.00 confidence bands: at least as often as it claims.
- Calibration error **0.050** (plain LightGBM: 0.060).
- **Unsure** is rare (2%) and is not "we don't know the subject". We predict it only when the evidence really runs out.

> notes: The confidence is part of the score, so we treat it as a measurement. When we say 90%, we're right at least 9
> times in 10. Unsure is only 2% of the data, so we don't hide in it. (STORY I4, I5, D2)

## Slide 14 — Model leaderboard: classic vs liquid vs spiking

**Macro-F1 (true/false), cross-validation grouped by page:**

- LightGBM: **0.76** (ships, inside the label engine)
- ExtraTrees: **0.72**
- CfC liquid network over each fact's timeline: **0.70**
- Spiking network (LIF neurons): **0.66**
- Words only (TF-IDF): **0.56** · Everything true: **0.455**

**Honest reading:**
- With ~1,900 labels, **trees win**; no liquid or spiking net beat them on true/false.
- The **knowledge base is the model**: without it LightGBM loses **0.075**.
- **Liquid and spiking nets catch unsure rows** (28–40% recall, against ~0 for trees) and, stacked, **halve calibration error**.

> notes: We tried a whole zoo, classic, liquid and spiking, on the same folds. Trees win, which is what you'd expect
> with 1,900 labels. Take away the date-aware knowledge base and the best model drops by 7.5 points. The neural nets are
> worse overall but find unsure rows and smooth the confidence. That's an honest result. (STORY M1–M10, I9)

## Slide 15 — Chapter 9 · Team + next steps

**Team Tzatziki**: five Claude Code sessions in parallel, one repo, one plan.
- X: research, architecture, pitch · Ioan: runtime, knowledge base, claims, API · Arckan: extractor + classifier
- Vald: frontend · Matei: story and first deck

**Next**
- Port the label engine into the shipped classifier and measure it on our own extracted facts.
- Use the neural nets' unsure signal without losing accuracy.
- Better extraction: the timeline is only as good as the facts we read into it.

**The date decides truth.**

> notes: Five people, each with their own Claude Code agent, one plan. The LLM helped us build the system, and none of
> it is inside. Next: ship the label engine, measure it on our own extraction, and use the neural unsure signal.
> Thank you.
