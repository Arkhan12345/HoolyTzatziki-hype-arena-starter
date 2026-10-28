# Dependency contract (all agents, all branches)

Human-readable rules. The executable truth is: `Dockerfile`, `requirements.txt` (direct runtime deps),
`constraints.txt` (every package in the image, exact), `requirements-dev.txt` (tests only) and
`frontend/package-lock.json`. If this file and those disagree, those win: fix this file in the same commit.
Owner of the runtime files: P2 (Ioan). Last verified against a real build: 2026-10-28.

## Runtime environment (what QA DNA runs)

| Item | Value | Source |
| --- | --- | --- |
| Base image | `python:3.12.16-slim-trixie` (Debian 13.7), `linux/amd64` | `Dockerfile` |
| Python | 3.12.16 | image |
| Extra OS package | `libgomp1` (LightGBM's OpenMP runtime) | `Dockerfile` |
| User | `10001` (last line of the Dockerfile; root is refused) | contract |
| Limits | 2 CPUs, 6 GiB RAM, ≤ 256 processes/threads, 30 min (SIGTERM, kill 10 s later) | `contracts/README.md` |
| Network | none at run time (the build has network) | contract |
| Read-only | the whole root, and `/universe` (input) | contract |
| Writable | `/tmp` (128 MB, counts in RAM) and `/output` (256 MB) only; no home dir | contract |
| Env set in the image | `HOME=/tmp XDG_CACHE_HOME=/tmp HF_HOME=/tmp MPLCONFIGDIR=/tmp OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1` | `Dockerfile` |
| In the image | `app/ models/ data/ contracts/` (frontend, tests, docs, pitch excluded by `.dockerignore`) | `Dockerfile` |
| Image size | 689 MB unpacked (limit 3 GiB) | `docker image inspect` |
| Measured run | practice input: 8.3 s, 52 MB RSS, no OOM | `scripts/check.sh` |

## Python: direct runtime dependencies (`requirements.txt`)

| Package | Version | Used by | Why | In the image |
| --- | --- | --- | --- | --- |
| jsonschema | 4.23.0 | `app/validate/rules.py`, tests | validates every claim against `contracts/entry.schema.json` | yes |
| numpy | 2.1.3 | reserved for P3 (`app/extract`, `app/classify`) | numeric features | yes |
| pandas | 2.2.3 | reserved for P3 / training | tabular work; P2 code reads CSV with the `csv` module | yes |
| scikit-learn | 1.5.2 | reserved for P3 | TF-IDF, logistic regression, isotonic calibration | yes |
| lightgbm | 4.5.0 | reserved for P3 | main truth classifier (docs/MODELS.md) | yes |
| joblib | 1.4.2 | reserved for P3 | loading `models/*.joblib` | yes |
| fastapi | 0.115.6 | `app/api/main.py` | demo API | yes (cheap; the arena run does not import it) |
| uvicorn | 0.32.1 | `scripts/dev_api.*` | serves the demo API | yes |
| python-multipart | 0.0.20 | `app/api/main.py` | CSV upload on `POST /run` | yes |

"Reserved for P3": in the image now so Arckan's modules run offline the moment they merge; no P2 code imports them.
Transitive packages (scipy 1.18.1, pydantic 2.14.1, starlette 0.41.3, referencing 0.37.0, …) are pinned in
`constraints.txt`; the Dockerfile installs with `pip install -r requirements.txt -c constraints.txt`.

## Python: development only (`requirements-dev.txt`, never in the image)

| Package | Version | Used by |
| --- | --- | --- |
| pytest | 8.3.4 | `tests/` |
| httpx | 0.28.1 | FastAPI `TestClient` in `tests/runtime/test_api.py` |

Training-only packages (Arckan): put them in `requirements-train.txt` (P3's file), not in `requirements.txt`, unless
the shipped model needs them at run time. A neuro model that ships needs torch-CPU + ncps: that goes behind a Docker
build arg `WITH_NEURO=1` with `requirements-neuro.txt` (not implemented yet; ask P2).

## Frontend (`frontend/`, Vald) — build-time only, never in the Docker image

Node: 24.12 on the team laptops; Vite 6 needs Node 18, 20 or 22+. Versions below are the ones locked in
`frontend/package-lock.json` (`package.json` holds `^` ranges; `npm ci` installs the locked ones).

| Package | Locked | Used for | Kind |
| --- | --- | --- | --- |
| react / react-dom | 19.3.1 | UI | runtime (browser) |
| @xyflow/react | 12.3.6 | architecture diagram (React Flow) | runtime (browser) |
| react-router-dom | 6.28.0 | pages | runtime (browser) |
| recharts | 2.15.0 | metrics charts | runtime (browser) |
| framer-motion | 11.15.0 | animations | runtime (browser) |
| react-markdown | 9.0.1 | pitch slides | runtime (browser) |
| lucide-react | 0.468.0 | icons | runtime (browser) |
| clsx / class-variance-authority / tailwind-merge | 2.1.1 / 0.7.1 / 2.5.5 | styling helpers | runtime (browser) |
| vite | 6.4.5 | dev server, build | dev |
| typescript | 5.8.3 | type-check | dev |
| tailwindcss / postcss / autoprefixer | 3.4.17 / 8.4.49 / 10.4.20 | CSS | dev |
| @vitejs/plugin-react | 4.7.0 | build | dev |
| eslint | 9.39.5 | lint | dev |

## Rules

1. **Everything offline is in the image.** No `pip install`, no download, no model fetch at run time. Models go in
   `models/` (copied into the image); caches point at `/tmp`.
2. **No network calls during evaluation**, not even optional ones (telemetry, update checks, NLTK/HF downloads).
3. **No LLM inside the system**, and no pretrained model whose knowledge stands in for the universe (docs/MODELS.md).
4. **Pin exactly.** `==` in `requirements*.txt`, the lockfile for npm. Never `>=` or unpinned.
5. **One version per package across the team.** Before adding or upgrading: read this file, check `requirements.txt`,
   `constraints.txt` and `frontend/package-lock.json`, and pick the version already there.
6. **Changing a dependency = one commit that updates the manifest, `constraints.txt` (rebuild and `pip freeze` the
   image), and this file**, plus a line in `pitch/JOURNAL.md`. Runtime manifests belong to P2: if you are not P2,
   add a Request in `STATUS.md` instead of editing them.
7. **Keep it light.** Image ≤ 2.5 GiB (hard limit 3 GiB), run ≤ 10 min on 20,000 rows on 2 CPUs, RSS ≤ 3 GB.
   Heavy additions (torch, transformers) need a measured win and P2's build arg.
8. **Dev and training packages stay out of the image** (`requirements-dev.txt`, `requirements-train.txt`).
9. **Verify before merging** any dependency change: `sh scripts/check.sh` must pass (on Windows through WSL, see
   `.claude/skills/arena-runtime/SKILL.md`).
