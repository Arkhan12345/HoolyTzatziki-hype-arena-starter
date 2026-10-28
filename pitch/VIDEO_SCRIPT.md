# Video script: ≤ 3 min, recorded 29 Oct after 09:00 (P5, Matei)

**What the rules require:** the frontend running our system **live** on the 20 rows of `sample/sample.csv`, **beside its
architecture diagram**. Hand-in by 11:00. Numbers: `pitch/STORY.md` only.

## Before recording (08:45–09:15)

1. `git pull upstream main` on the demo branch, so `sample/sample.csv` exists (X does this; check `STATUS.md`).
2. Start everything: `frontend/scripts/start.sh` (or `start.ps1`). Check `http://localhost:8000/health` shows
   `"sample_available": true`, and that the UI has no "API offline" banner.
3. Do one warm-up run (the first run takes ~8 s while it builds the claims; later runs are cached). Then reload the page.
4. Browser at 1920×1080, zoom 100%, dark theme, notifications off, other tabs closed. Screen recorder at 30 fps, mic tested.
5. Keep a fallback: if the live API fails during the take, stop and fix it. **Never** record the mock data; the "API offline" banner must not appear.

## Shot list

| Time | Screen | Click / do | Say (voice-over) |
|---|---|---|---|
| 0:00–0:15 | `/pitch` slide 1 | — | "We're Team Tzatziki. Our system reads a year of a company's pages and decides which sentences were true on their day. Our one-line lesson: the date decides truth." |
| 0:15–0:40 | `/pitch` slide 3 | — | "The same sentence, *Hooly Gemini replies in 2.1 seconds*, is true in November 2025 and false in August 2026, because Gemini 2.0 brought it down to 1.4 seconds. Repeating an old value after it changed is false 84% of the time." |
| 0:40–1:05 | `/architecture` | click **Extractor**, then **Temporal KB** | "This is the real system. Sentences become facts, facts go on a timeline, and every new sentence is judged against the value in force on its date. The same timeline writes the claims, and a validator checks each one against the referee's rules." |
| 1:05–1:35 | `/` (live run), diagram visible beside the results | click **Use sample.csv** | "Now live, on the 20 sample rows we were given this morning. Watch the stages light up: extract, knowledge base, classify, generate, validate, write." |
| 1:35–2:05 | `/` results table | open one TRUE row, then one FALSE row | "Each row gets a label and a confidence. Here's why: these are the facts we read and the value the timeline held on that date." *(read the row's explanation)* |
| 2:05–2:30 | `/` → **Generated claims** tab | open one *change* and one *compare* claim; expand the form JSON | "And the 1,000 new claims, each a true sentence the referee can read, with its structured form and the source sentences it came from." |
| 2:30–2:50 | `/metrics` | scroll: claims by type → runtime | "The whole run takes seconds on two CPUs, offline, with no LLM inside. Every claim passed our validator." *(say the model score only if `/metrics` shows one)* |
| 2:50–3:00 | `/pitch` slide 10 | — | "The date decides truth. Thank you." |

## Rules for the take

- The sample has **no labels**: never say "we got X right" about it. Show predictions and explanations, not accuracy.
- Say a model score only if the live `/metrics` shows one. Otherwise say nothing about accuracy in the video.
- Keep the diagram on screen during the run. Use the split view on `/`, or **Demo mode** (top right), which walks the diagram slowly.
- One continuous take if possible; cut only between shots. Total ≤ 3:00. Export MP4 1080p.

## Rehearsal (before 09:00, on practice rows)

- [ ] Run the whole shot list once with `{"use":"sample"}` before the file exists (it runs 20 random practice rows).
- [ ] Time each shot; trim the voice-over where it runs long.
- [ ] Check the run page shows the explanation and facts for a row (needs Arckan's extractor; if `facts` is empty, show the label and confidence only).
- [ ] Check slides 1, 3 and 10 render at `/pitch` with no `{{…}}` left over.
