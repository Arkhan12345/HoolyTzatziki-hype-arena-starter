# Jury Q&A (P5, Matei)

Two or three lines per answer. Every number comes from `pitch/STORY.md` (row ids in brackets).

**1. Why not just use an LLM?**
The rules forbid an LLM inside the system. It would also be wrong here: Hooly uses real product names with invented
histories, so a model that "knows" Gemini brings facts the universe contradicts. Ours judges only against the universe's own timeline. (`README.md`, `docs/REQUIREMENTS.md` R-01/R-02)

**2. So where did an LLM help?**
To build the system: Claude Code agents wrote the code, the rules and the tests, one agent per team member. Nothing
it produced runs as a model at run time. The image has no LLM, no network and no pretrained weights. (`DEPENDENCIES.md` rules 2–3)

**3. Do you use a pretrained encoder (MiniLM, BERT)?**
No. We asked QA DNA in issue #2 whether a small fine-tuned encoder is allowed and have no answer yet, so we ship
without one. It would only help extraction, never supply facts. (`docs/MODELS.md` §1, `README.md` option F)

**4. How do you handle pages that lie?**
Three ways. A value that contradicts the timeline on its date is a strong false signal (45% false; a stale value 84%).
First-person hype pages are a feature (40% false). Page trust from truth discovery is our next step. [T1, T2, S1] (`docs/RULES.md`)

**5. What does "the date decides truth" mean in practice?**
For each fact we look up the value in force on the sentence's date. The same Gemini sentence is true in Nov 2025 and
false in Aug 2026 because the value changed in March. Without these date features our model loses 0.075 macro-F1. [T6, M7] (`eda/out/ablation.csv`)

**6. How is confidence calibrated?**
The classifier's probabilities are checked against how often it's actually right, out of fold. Its calibration
error is 0.046 in grouped cross-validation. Claim confidences come from a measured holdout: we hide pages and count wrong claims per type. [M8, C3] (`eda/out/bakeoff.csv`, `pitch/metrics.json`)

**7. You have only ~1,900 labels. How do you avoid overfitting?**
Cross-validation is grouped by page, so sentences from one page never sit on both sides of a split. We also use a
time-forward split, and a holdout scored once at the end. We prefer the data's rules (stale value, single-value fact) over memorised words: words alone score 0.56. [M1, M5, M10] (`docs/EDA_v2.md` §5, `eda/splits.csv`)

**8. Why does the time-forward score drop to 0.62?**
When we train only on earlier dates, later facts have no timeline yet. That's the pessimistic bound. The real run sees
all practice labels plus 18,000 more sentences on the same timelines, so we expect it to land between 0.62 and 0.76. [M10] (`docs/EDA_v2.md` §5)

**9. What if a sentence names a subject or value you've never seen?**
Every subject and attribute is in the lexicon, so the subject is always known. An unseen *value* is a signal: a value
nothing else repeats is 29% false. A fact with only one value in the corpus is never false (0 of 280), so we never call it false. [T5] (`docs/RULES.md` §1, T5)

**10. How do you know the 1,000 claims are true?**
We measured it. We hide a share of the practice pages from the generator, let it write claims, and judge them on all
of practice: 0.17% wrong with 10% hidden, 1.9% with half hidden. Each claim is built only from gold-backed facts and re-checked independently. [C3] (`pitch/metrics.json` runtime.claim_holdout)

**11. How do you make sure the referee can read each claim?**
Our validator turns every example table of `claim-forms.md` and `value-forms.md` into a test: no approximations, no
negations, no time words, no second value, the subject named in the same sentence. 1,000 of 1,000 pass, and the generator has ~5,650 distinct valid candidates to choose from. [C1, C4] (`tests/runtime/`, `pitch/metrics.json`)

**12. Runtime and scaling to 20k rows or more?**
A full run on a synthetic 18,970-row universe takes 4 s and 57 MB of memory in Docker. The limits are 30 min and 6 GiB.
The image is 689 MB of the 3 GiB allowed. [R1, R2, R3] (`pitch/metrics.json` runtime)

**13. What happens if something crashes during the arena run?**
A valid table is on disk within a second of start. Every stage is wrapped and falls back. A watchdog rewrites the best
table so far 90 s before the deadline, and a SIGTERM handler writes again. The final table is written under a temp name and renamed. [R4] (`docs/ARCHITECTURE.md` failure strategy)

**14. Why did you try neuromorphic models at all?**
Two reasons. Each fact's history is an irregular time series, which is what liquid (CfC) nets are built for. And spiking
nets are the path to very low-energy hardware. Honestly, trees still win on accuracy (0.76 vs 0.70 and 0.66), but the neural nets are the only ones that catch unsure rows (28–40% recall). [M1, M3, M4, M9] (`eda/out/bakeoff.csv`)

**15. What are the failure modes?**
Extraction: the timeline is only as good as the facts we read, and 733 facts name their subject only by a pronoun.
Rare unsure rows: 2% of the data, hard for every model. And pages whose lies agree with each other. [D4, D2] (`docs/EDA_v2.md`)

**16. How did five people work in parallel overnight?**
Each of us ran a Claude Code agent with its own folders, a shared interface file and a status board. We merged small PRs
every hour or two. The runtime always worked: `check.sh` stayed green while the model was built behind a fallback. (`team/README.md`, `STATUS.md`)
