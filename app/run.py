"""The run QA DNA starts: /universe in, /output/output.csv out, always.

Order: read -> gold KB -> claims -> FALLBACK TABLE -> extract (P3) -> KB + extracted ->
classify (P3) -> claims again (now with staleness vetoes) -> final table.
Every stage falls back on failure; a watchdog and a SIGTERM handler write the best table so far.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
import traceback

from app.io.reader import read_lexicon, read_practice, read_universe
from app.io.types import Label
from app.io.writer import write_table
from app.kb.store import KB
from app.validate.rules import norm_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRACTICE = os.path.join(ROOT, "data", "practice.csv")
BASELINE_TRUE_CONF = 0.83  # share of true rows in practice: an honest confidence for "true"
WATCHDOG_MARGIN = 90


def log(msg: str) -> None:
    print(f"[run {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def rss_mb() -> float:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:  # noqa: BLE001 - not on Linux
        return -1.0


class State:
    """Best table so far; written by the main flow, the watchdog and the SIGTERM handler."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.labels: list = []
        self.claims: list = []
        self.lock = threading.Lock()
        self.written = False

    def set(self, labels=None, claims=None) -> None:
        with self.lock:
            if labels is not None:
                self.labels = labels
            if claims is not None:
                self.claims = claims

    def write(self, why: str) -> None:
        with self.lock:
            labels, claims = list(self.labels), list(self.claims)
        try:
            write_table(self.output_dir, labels, claims)
            self.written = True
            log(f"table written ({why}): {len(labels)} labels, {len(claims)} claims")
        except Exception:  # noqa: BLE001
            log(f"table write failed ({why}):\n{traceback.format_exc()}")


def baseline_labels(rows: list[dict]) -> list:
    return [Label(id=r["id"], label="true", confidence=BASELINE_TRUE_CONF) for r in rows if not r["label"]]


def _load_p3():
    extract = classify = None
    try:
        from app.extract import extract  # type: ignore
    except Exception as e:  # noqa: BLE001
        log(f"extractor not available ({e!r}); running without extracted facts")
    try:
        from app.classify import classify  # type: ignore
    except Exception as e:  # noqa: BLE001
        log(f"classifier not available ({e!r}); keeping baseline labels")
    return extract, classify


def _clean_labels(labels, rows) -> list:
    """One label per unlabelled row, in input order; anything missing stays baseline."""
    blank = [r["id"] for r in rows if not r["label"]]
    got = {}
    for lb in labels or []:
        if lb.id in got:
            continue
        label = lb.label if lb.label in ("true", "false", "unsure") else "unsure"
        conf = None if label == "unsure" else min(1.0, max(0.5, float(lb.confidence or 0.5)))
        got[lb.id] = Label(id=lb.id, label=label, confidence=conf)
    return [got.get(i) or Label(id=i, label="true", confidence=BASELINE_TRUE_CONF) for i in blank]


def main(universe_dir: str | None = None, output_dir: str | None = None, deadline: float | None = None) -> dict:
    t0 = time.time()
    universe_dir = universe_dir or os.environ.get("UNIVERSE_DIR", "/universe")
    output_dir = output_dir or os.environ.get("OUTPUT_DIR", "/output")
    if deadline is None:
        deadline = float(os.environ.get("ARENA_DEADLINE") or (t0 + 30 * 60))
    timings: dict[str, float] = {}
    state = State(output_dir)
    report: dict = {"timings": timings}

    def stage(name, fn, fallback=None):
        s = time.time()
        try:
            out = fn()
        except Exception:  # noqa: BLE001
            log(f"stage {name} failed, falling back:\n{traceback.format_exc()}")
            out = fallback() if callable(fallback) else fallback
        timings[name] = round(time.time() - s, 3)
        log(f"stage {name}: {timings[name]}s, rss {rss_mb():.0f} MB")
        return out

    # Watchdog: whatever happens, a table exists before the kill.
    def watchdog():
        while True:
            left = deadline - time.time()
            if left <= WATCHDOG_MARGIN:
                state.write("watchdog")
                return
            time.sleep(min(5, max(0.5, left - WATCHDOG_MARGIN)))

    threading.Thread(target=watchdog, daemon=True).start()
    if threading.current_thread() is threading.main_thread():
        def on_term(signum, frame):
            state.write("SIGTERM")
            os._exit(0)

        signal.signal(signal.SIGTERM, on_term)

    rows, lexicon = stage("read", lambda: read_universe(universe_dir),
                          lambda: ([], read_lexicon(os.path.join(ROOT, "data", "lexicon.json"))))
    input_texts = {norm_text(r["text"]) for r in rows}
    gold = stage("gold", lambda: read_practice(PRACTICE) if os.path.exists(PRACTICE) else [], [])
    input_texts |= {norm_text(r["text"]) for r in gold}
    state.set(labels=baseline_labels(rows))

    from app.generate import generate

    kb = stage("kb_gold", lambda: KB(lexicon).build([], gold, rows), lambda: KB(lexicon))
    gen_report: dict = {}
    claims = stage("claims_gold", lambda: generate(kb, lexicon, input_texts, 1000, gen_report), [])
    state.set(claims=claims)
    state.write("fallback")

    extract, classify = _load_p3()
    facts = []
    if extract:
        facts = stage("extract", lambda: extract(rows, lexicon), []) or []
        if facts:
            kb2 = stage("kb_full", lambda: KB(lexicon).build(facts, gold, rows), None)
            if kb2 is not None:
                kb = kb2
    if classify:
        labels = stage("classify", lambda: _clean_labels(classify(rows, facts, kb), rows), None)
        if labels:
            state.set(labels=labels)
    if facts and deadline - time.time() > 5 * 60:
        gen_report = {}
        claims2 = stage("claims_full", lambda: generate(kb, lexicon, input_texts, 1000, gen_report), None)
        if claims2 and len(claims2) >= len(claims):
            state.set(claims=claims2)
    state.write("final")

    report.update({"rows": len(rows), "blank": sum(1 for r in rows if not r["label"]), "facts": len(facts),
                   "kb": kb.stats(), "claims": gen_report, "rss_mb": rss_mb(),
                   "total_s": round(time.time() - t0, 3)})
    log("report " + json.dumps(report, default=str))
    return report


if __name__ == "__main__":
    main()
