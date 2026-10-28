"""check.sh without Docker: same input, same checker, plus our own claim validator.

    python scripts/local_check.py                 # practice, every 5th label hidden
    python scripts/local_check.py --minutes 5     # shorter deadline
    python scripts/local_check.py --universe DIR  # any dir holding universe.csv + lexicon.json

It runs `python -m app.run` in a subprocess with UNIVERSE_DIR / OUTPUT_DIR / ARENA_DEADLINE,
then scripts/check_output.py (upstream), then re-validates every claim with app.validate.
It cannot check the read-only root, user 10001 or the network: only scripts/check.sh can.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--universe", help="dir with universe.csv + lexicon.json (default: practice, 1 in 5 hidden)")
    ap.add_argument("--keep", action="store_true", help="keep the work dir and print its path")
    args = ap.parse_args()

    work = tempfile.mkdtemp(prefix="arena-local-")
    universe, output = os.path.join(work, "universe"), os.path.join(work, "output")
    os.makedirs(universe)
    os.makedirs(output)
    if args.universe:
        for f in ("universe.csv", "lexicon.json"):
            shutil.copy(os.path.join(args.universe, f), universe)
    else:
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "input.py"),
                        os.path.join(ROOT, "data", "practice.csv"), os.path.join(universe, "universe.csv")], check=True)
        shutil.copy(os.path.join(ROOT, "data", "lexicon.json"), universe)

    deadline = int(time.time() + args.minutes * 60)
    env = dict(os.environ, UNIVERSE_DIR=universe, OUTPUT_DIR=output, ARENA_DEADLINE=str(deadline),
               PYTHONPATH=ROOT, PYTHONIOENCODING="utf-8")
    t = time.time()
    try:
        proc = subprocess.run([sys.executable, "-m", "app.run"], cwd=ROOT, env=env,
                              timeout=args.minutes * 60 + 10, capture_output=True, text=True, encoding="utf-8")
        tail = (proc.stderr or "").strip().splitlines()[-12:]
        code = proc.returncode
    except subprocess.TimeoutExpired:
        tail, code = ["killed at the deadline"], "timeout"
    print(f"run: exit {code} in {time.time() - t:.1f}s")
    for line in tail:
        print("  | " + line[:300])

    practice = os.path.join(ROOT, "data", "practice.csv")
    rc = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "check_output.py"), output,
                         practice, os.path.join(universe, "universe.csv")]).returncode

    bad = _revalidate(os.path.join(output, "output.csv"), os.path.join(universe, "lexicon.json"))
    if args.keep:
        print(f"work dir: {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)
    return 1 if rc or bad else 0


def _revalidate(table: str, lexicon_path: str) -> int:
    if not os.path.isfile(table):
        return 1
    from app.io.types import Claim
    from app.validate.rules import validate_claim

    with open(lexicon_path, encoding="utf-8") as f:
        lexicon = json.load(f)
    with open(table, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["type"]]
    bad = 0
    for r in rows:
        c = Claim(id=r["id"], type=r["type"], text=r["text"], form=json.loads(r["form"]),
                  confidence=float(r["confidence"]))
        problems = validate_claim(c, lexicon, None)
        if problems:
            bad += 1
            if bad <= 10:
                print(f"  validator: {r['id']}: {problems[:2]} | {r['text'][:100]}")
    by_type: dict[str, int] = {}
    for r in rows:
        by_type[r["type"]] = by_type.get(r["type"], 0) + 1
    print(f"validator: {len(rows) - bad} of {len(rows)} claims clean; by type {by_type}")
    return bad


if __name__ == "__main__":
    sys.exit(main())
