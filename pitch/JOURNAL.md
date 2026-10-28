# Journal — one line per meaningful merge: `HH:MM | P# | what was built | why it matters / number`

11:40 | P2 | Runtime v1: Dockerfile (USER 10001, caches in /tmp), run.py with fallback table, watchdog at deadline-90 s and SIGTERM write | a run always leaves a valid output.csv; 19k rows in ~4 s without the model
11:40 | P2 | Temporal KB + claim generator + validator from practice gold | 1,000 claims, 100% validator-clean, 5 types; ~3,900 valid candidates to choose from
11:40 | P2 | Demo API (POST /run, GET /architecture, GET /metrics) in the frontend's shapes | Vald's UI can switch from mocks to the live pipeline
