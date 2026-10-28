# Team Tzatziki · Hype Arena — notes for every Claude Code agent

Read `team/README.md` (who owns what) and the official `contracts/` first; the contracts win over team docs.

- **Claims (official rule from the mentor, on top of contracts/):** one claim per piece of information. Paraphrases,
  aliases and inverse compares ("A costs more than B" / "B costs less than A") count as the same claim, and a claim
  must not restate an input sentence (one that names its subjects and states its values; tying a value to a subject
  the sentence leaves unnamed is new). Enforced by `app/validate/claims.py`.
- **Dependencies:** before adding or upgrading any dependency, read `DEPENDENCIES.md`, inspect the actual
  lockfiles/manifests (`requirements.txt`, `constraints.txt`, `requirements-dev.txt`, `frontend/package-lock.json`),
  and update the shared dependency contract when a change is required.
- **Ownership:** edit only your own folders (`team/README.md`); anything else is a Request line in `STATUS.md`.
- **Runtime:** the arena runs `python -m app.run` in the Docker image, offline, as user 10001, read-only root.
  No LLM and no network at run time. `/output/output.csv` is the only official artifact.
- **Backend contracts:** API and metrics shapes in `docs/API_CONTRACT.md` (schemas in `app/metrics/schemas.py`);
  frontend integration in `docs/VALD_HANDOFF.md`; P3 plugs in through `app.extract.extract(rows, lexicon)` and
  `app.classify.classify(rows, facts, kb)` (see `app/kb/adapter.py` for the fact shape the KB accepts).
- **Before pushing backend changes:** `.venv/Scripts/python -m pytest -q tests/runtime`; before a milestone,
  `sh scripts/check.sh` (WSL on Windows). P2's runtime skill: `.claude/skills/arena-runtime/SKILL.md`.
