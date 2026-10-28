# Frontend: Hype or Truth (Team Tzatziki)

The page the jury sees: our pipeline as a live diagram, a live run on rows we give it (`sample/sample.csv`), metrics, and the pitch deck.
Owner: P4 (Vald). Stack: Vite + React 19 + TypeScript + Tailwind + shadcn-style components, React Flow (`@xyflow/react`) for the diagram, Recharts for charts.

## Start everything with one command

From the repo root:

```sh
frontend/scripts/start.sh        # macOS / Linux / Git Bash
```

```powershell
frontend\scripts\start.ps1       # Windows PowerShell
```

This installs the UI's dependencies, starts the demo API on `:8000` if `app/api` exists and Python is available, and starts the UI on <http://localhost:5173>.
If the API is not up, the UI still works: it shows **mock** results and an "API offline" banner.

## Run the UI only

```sh
cd frontend
npm install
npm run dev      # talks to VITE_API_URL (default http://localhost:8000), falls back to mocks
npm run mock     # mocks only, never calls the API (VITE_MOCK=1)
npm run build    # type-check + production build; run it before every push
```

| Env var | Default | Meaning |
| --- | --- | --- |
| `VITE_API_URL` | `http://localhost:8000` | Base URL of P2's FastAPI demo API |
| `VITE_MOCK` | unset | `1` = use `src/mocks/*.json`, never call the API |

## Pages

| Route | What |
| --- | --- |
| `/` | **Live run**: "Use sample.csv" (`POST /run {"use":"sample"}`) or upload/drop a CSV (`POST /run`, multipart field `file`). The pipeline diagram lights up stage by stage. Then: TRUE/FALSE/UNSURE stamps, confidence bars, extracted facts as chips, and a tab of generated claims with each form's JSON. |
| `/architecture` | Full React Flow diagram from `GET /architecture`. Click a node for what it does and its latest timing. It lights up during a run too. |
| `/metrics` | `GET /metrics` (pitch/metrics.json): accuracy and macro-F1 vs the all-true baseline, confusion matrix, calibration, claims by type, runtime per stage. Shows a Classic vs Liquid (CfC) vs Spiking card when the key `neuro` exists. |
| `/pitch` | Full-screen slides from `pitch/PITCH.md`. |

**Demo mode** (top right) runs sample.csv and walks the diagram slowly, for the video.

## API shapes the UI expects (from P2's `app/api`)

```jsonc
// POST /run
{ "labels":  [{ "id": "p-1", "label": "true", "confidence": 0.93 }],          // confidence null for unsure
  "claims":  [{ "id": "c-1", "type": "fact", "text": "…", "form": {…}, "confidence": 0.9 }],
  "facts":   [{ "row_id": "p-1", "subject": "Holdan", "attribute": "joined", "value": {"kind":"date","value":"2026-01-12"}, "span": "…", "conf": 0.9 }],
  "timings": { "load": 0.08, "extract": 0.42, "kb": 0.11, "classify": 0.27, "generate": 0.36, "validate": 0.14, "write": 0.03 },  // seconds
  "stages":  ["load", "extract", "kb", "classify", "generate", "validate", "write"],
  "rows":    [{ "id": "p-1", "text": "…", "date": "2026-01-12" }]               // optional, requested: lets "Use sample.csv" show row text
}
// GET /architecture
{ "nodes": [{ "id": "kb", "label": "Temporal KB", "kind": "stage|io|optional", "stage": "kb", "owner": "P2",
              "description": "…", "details": ["…"], "position": {"x": 520, "y": 160} }],   // position optional (auto-layout)
  "edges": [{ "source": "kb", "target": "classifier", "label": "…", "dashed": false, "loop": false }] }
// GET /metrics: see src/mocks/metrics.json for every key the page reads; missing keys are skipped.
```

A node lights up when its `stage` matches the stage being run. The mocks in `src/mocks/` are the reference shapes. `run.json` holds the first 20 practice rows with their gold forms as facts.

## Pitch deck (`/pitch`)

`scripts/copy-pitch.mjs` copies `../pitch/PITCH.md` to `public/pitch.md` before every `dev` and `build`. It is a plain copy, not a symlink, so it works on Windows.

- Every `## Slide N: Title` heading starts a slide. The body is Markdown.
- Lines starting `> notes:` (and the `> ` lines after them) are speaker notes.
- Live numbers: write `{{model.accuracy|pct}}`, `{{model.macro_f1|pct}}` or `{{runtime.total_s}}` in PITCH.md, and the value comes from `/metrics`.
- Keys: ← → / Space to move, **N** for presenter notes, **F** for fullscreen, **Esc** to leave. `#s=3` in the URL opens slide 3.
