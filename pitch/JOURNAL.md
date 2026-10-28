# Journal — one line per meaningful merge: `HH:MM | P# | what was built | why it matters / number`

11:40 | P2 | Runtime v1: Dockerfile (USER 10001, caches in /tmp), run.py with fallback table, watchdog at deadline-90 s and SIGTERM write | a run always leaves a valid output.csv; 19k rows in ~4 s without the model
11:40 | P2 | Temporal KB + claim generator + validator from practice gold | 1,000 claims, 100% validator-clean, 5 types; ~3,900 valid candidates to choose from
11:40 | P2 | Demo API (POST /run, GET /architecture, GET /metrics) in the frontend's shapes | Vald's UI can switch from mocks to the live pipeline
12:00 | P2 | scripts/check.sh green in real Docker (WSL): exit 0, no OOM, 1,000 claims, user 10001, read-only root, no network | the image is contest-ready; 19k rows in 4 s at 57 MB RSS, image 689 MB of the 3 GiB allowed
12:10 | P2 | Generator v2: reads only a FactInventory from the KB, no type quotas, confidence-first selection, both orientations of every compare, an independent truth re-check of every form | 10,190 valid new claims, 5,796 distinct; 3,981 distinct compares alone could fill the 1,000; selected all at confidence >= 0.93 with zero semantic duplicates
12:10 | P2 | KB adapter for P3's extractor output + scripts/claim_capacity.py | real extraction plugs in without touching the generator (tested with a mock extractor, dataclasses and dicts)
12:25 | P2 | Validator: a name value must resolve in the lexicon (the referee resolves names through it); claim-forms tables for at_event, relation and date compares are now tests | 0 of the 1,000 claims rely on a name the referee might not read; 114 tests green, check.sh green in Docker
12:50 | P2 | Structured metrics from the real pipeline (classification, claims, runtime; evaluation only where labels are known) + API v1.1 for the frontend | Vald's UI gets real numbers or explicit nulls, never invented ones; contract in docs/API_CONTRACT.md, handoff in docs/VALD_HANDOFF.md
12:50 | P2 | DEPENDENCIES.md + CLAUDE.md; every image package pinned (constraints.txt, python:3.12.16-slim-trixie) | reproducible offline image, 689 MB; check.sh green in Docker, a valid table on disk within a second of start
