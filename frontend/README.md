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
| `/` | **Dashboard**: system status from the last run's `stage_details`, rows labelled, claims generated, agreement with practice gold, verdict distribution, confidence histogram, the live data flow, the pipeline log, and recent classifications. |
| `/verify` | **Verify a claim**: text + date go to `POST /run {"rows":[…]}`. Shows the verdict, a confidence gauge, the evidence extracted, what decided the verdict, a plain explanation, the extracted form, an evidence map, and the value history from the practice gold rows. A practice sentence keeps its own id, so the API holds its gold out of the KB and returns it for comparison. |
| `/explorer` | Search, filter (verdict, min confidence, value kind, date range), sort and page through the last run or the 1,897 practice rows. Expand a row for its facts or form and its provenance. |
| `/generate` | Runs `POST /run?claims_limit=1000`. Shows accepted claims out of 1,000, claims by type, format checks against the output contract (run in the browser, including novelty against the practice rows), a preview, and CSV export in the exact `output.csv` columns. |
| `/architecture` | React Flow diagram from `GET /architecture`. While a request runs, every stage shows "running"; afterwards the measured stage timings replay. Fallback or baseline stages turn amber. Click a node for its description, timing and API status. |
| `/metrics` | `GET /metrics` (pitch/metrics.json). Shows model scores when P3 adds `model`, and runtime and contract checks from P2. Missing sections say so; nothing is estimated. |
| `/pitch` | Full-screen slides from `pitch/PITCH.md`: `## Slide N: Title` starts a slide, `> notes:` lines are speaker notes. Use ← → to move, N for notes, F for fullscreen, Esc to exit. `{{model.accuracy|pct}}` pulls a live number from `/metrics`. |

**Demo run** (sidebar) runs `sample.csv` through the real API. If the API is down, pages show an error with the start command. Demo data loads only when you click "Demo data", and it is labelled as such.

## API shapes the UI reads (P2's `app/api/main.py`)

`POST /run` returns `{source, labels[{id,label,confidence,gold}], claims, claims_total, claims_by_type, facts, rows, timings, stages, stage_details[{id,seconds,status}], runtime}`.
`GET /architecture` returns `{nodes, edges}`. `GET /metrics` returns pitch/metrics.json. `GET /health` returns `{ok}`.
Optional and read when present: `report` on `/run` (generator candidates, valid, rejected and semantic duplicates).
