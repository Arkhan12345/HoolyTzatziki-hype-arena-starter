# Frontend: Hype Arena (Team Tzatziki · Hype or Truth)

The page the jury sees: our pipeline as a live diagram, a live run on rows we give it (`sample/sample.csv`), single-claim verification, an explorer, the 1,000-claim generator, metrics, and the pitch deck.
Owner: P4 (Vald). Stack: Vite + React 19 + TypeScript + Tailwind + framer-motion, React Flow (`@xyflow/react`) for the diagram, Recharts for charts. The visual system is described in [DESIGN.md](DESIGN.md).

## Start everything with one command

From the repo root:

```sh
frontend/scripts/start.sh        # macOS / Linux / Git Bash
```

```powershell
frontend\scripts\start.ps1       # Windows PowerShell
```

This installs the UI's dependencies, starts the demo API on `:8000` if `app/api` exists and Python is available, and starts the UI on <http://localhost:5173>.

## Run the UI only

```sh
cd frontend
npm install
npm run dev      # talks to VITE_API_URL (default http://localhost:8000)
npm run mock     # demo data only, never calls the API (VITE_MOCK=1); every page says so
npm run build    # type-check + production build; run it before every push
```

| Env var | Default | Meaning |
| --- | --- | --- |
| `VITE_API_URL` | `http://localhost:8000` | Base URL of P2's FastAPI demo API |
| `VITE_MOCK` | unset | `1` = use `src/mocks/*.json` and never call the API |

`scripts/sync-assets.mjs` runs before `dev` and `build`. It copies `pitch/PITCH.md`, `data/lexicon.json` and `data/practice.csv` into `public/`. These copies are git-ignored.

## Pages

| Route | What |
| --- | --- |
| `/` | **Dashboard**: `/health` (extractor, classifier, sample availability), stage status from `stage_details`, then the run's `metrics`: rows, claims, accuracy on practice rows (`classification.evaluation`), verdict counts, confidence histogram, runtime. Also the live data flow, the pipeline log, and `predictions` with their `explanation` and `facts`. A `partial` run shows a warning naming `errors[].stage`. |
| `/verify` | **Verify a claim**: text + date go to `POST /run {"rows":[…]}`. Shows the verdict, a confidence gauge, the API's `explanation`, the extracted facts, the KB `evidence` (supports or contradicts) when the extractor is in, and the value history from the practice gold rows. A practice sentence keeps its own id, so the API holds its gold out of the KB and returns it for comparison. |
| `/explorer` | Search, filter (verdict, min confidence, value kind, date range), sort and page through the last run's `predictions` or the 1,897 practice rows. Expand a row for its facts or form, its explanation and its provenance. |
| `/generate` | Runs the generator, then pages through all claims with `GET /runs/{id}/claims` (type filter on the server). Shows `metrics.claims`: candidates, rejected by reason, semantic duplicates, selected by type and confidence. Each claim's form and source sentences can be expanded. The download is `GET /runs/{id}/output.csv`. |
| `/architecture` | React Flow diagram from `GET /architecture`, refetched after each run. Nodes are lit by `status` (ok green, fallback amber, not_available grey, skipped muted, failed red) and labelled with `seconds` (null shows no number). Click a node for its description, timing and status. |
| `/metrics` | `GET /metrics`. When `model` is null, it says "not available" instead of showing accuracy, macro-F1, confusion and calibration; a `model.label_source` of "offline evaluation (pitch/metrics.json)" is labelled so. Also the claim funnel, rejections by reason, claims by type, and runtime per stage with its status. A null shows as "—", never 0. |
| `/pitch` | Full-screen slides from `pitch/PITCH.md`: `## Slide N: Title` starts a slide, `> notes:` lines are speaker notes. Use ← → to move, N for notes, F for fullscreen, Esc to exit. `{{model.accuracy|pct}}` pulls a live number from `/metrics`. |

**Demo run** (sidebar) runs `sample.csv` through the real API. If the API is down, pages show an error with the start command. Demo data loads only when you click "Demo data", and it is labelled as such.

## API the UI uses (P2's API v1.1: `docs/API_CONTRACT.md`, handoff in `docs/VALD_HANDOFF.md`)

Types are in `src/api/types.ts` and calls in `src/api/client.ts`. The UI never computes labels, claims or metrics.

| Call | Used for |
| --- | --- |
| `POST /run?mode=async` (sample, multipart `file`, or `{rows}`), then poll `GET /runs/{id}` every 500 ms | every run; the API stays responsive while it works (60 s request timeout, 5 min overall) |
| `GET /runs/{id}/claims?offset&limit&type` | the 1,000 claims, 50 per page |
| `GET /runs/{id}/output.csv` | the download on /generate |
| `GET /metrics`, `GET /architecture`, `GET /health` | metrics page, diagram, status and boot checks |

Errors are explained in plain words. A 400 shows the API's `detail`. A 413 means more than 2,000 rows. A 404 on `/runs/...` means the API restarted, so start a new run.
