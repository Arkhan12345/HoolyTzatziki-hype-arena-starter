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
from app.metrics import collector
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

    def __init__(self, output_dir: str, stages=None):
        self.output_dir = output_dir
        self.stages = stages
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
        t = time.time()
        try:
            write_table(self.output_dir, labels, claims)
            self.written = True
            if self.stages is not None:
                self.stages.add("write", time.time() - t, "ok")
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


def aux_dir(output_dir: str) -> str:
    """Where metrics.json / architecture.json go. Never the arena's /output (it holds one table)."""
    if os.environ.get("TZ_AUX_DIR"):
        return os.environ["TZ_AUX_DIR"]
    if os.path.normpath(output_dir).replace("\\", "/") == "/output":
        return "/tmp/tzatziki"
    return output_dir


def main(universe_dir: str | None = None, output_dir: str | None = None, deadline: float | None = None) -> dict:
    t0 = time.time()
    universe_dir = universe_dir or os.environ.get("UNIVERSE_DIR", "/universe")
    output_dir = output_dir or os.environ.get("OUTPUT_DIR", "/output")
    if deadline is None:
        deadline = float(os.environ.get("ARENA_DEADLINE") or (t0 + 30 * 60))
    run_id = collector.new_run_id()
    stages = collector.Stages(log)
    state = State(output_dir, stages)

    def watchdog():  # whatever happens, a table exists before the kill
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

    rows, lexicon = stages.run("load", lambda: read_universe(universe_dir),
                               lambda: ([], read_lexicon(os.path.join(ROOT, "data", "lexicon.json"))))
    gold = stages.run("load", lambda: read_practice(PRACTICE) if os.path.exists(PRACTICE) else [], [])
    input_texts = {norm_text(r["text"]) for r in rows} | {norm_text(r["text"]) for r in gold}
    state.set(labels=baseline_labels(rows))
    state.write("initial")  # within a second: labels only, so even an early kill leaves a valid table

    from app.generate import generate

    kb = stages.run("kb", lambda: KB(lexicon).build([], gold, rows), lambda: KB(lexicon))
    gen_report: dict = {}
    claims = stages.run("generate", lambda: generate(kb, lexicon, input_texts, 1000, gen_report), [])
    stages.move("generate", "validate", gen_report.get("validate_s", 0) + gen_report.get("truth_s", 0))
    if gen_report.get("shortfall"):
        # only distinct, PASS claims are written (mentor's rule): fewer than 1,000 rather than duplicates
        log(f"WARNING claims: {gen_report['shortfall']} short of 1,000 distinct PASS claims")
        stages.errors.append({"stage": "generate", "message": f"{gen_report['shortfall']} claims short of 1,000"})
    state.set(claims=claims)
    state.write("fallback")

    extract, classify = _load_p3()
    facts: list = []
    if extract:
        facts = stages.run("extract", lambda: extract(rows, lexicon), []) or []
        if facts:
            kb2 = stages.run("kb", lambda: KB(lexicon).build(facts, gold, rows), None)
            if kb2 is not None:
                kb = kb2
    else:
        stages.status["extract"] = "not_available"
    classifier = "baseline"
    if classify:
        labels = stages.run("classify", lambda: _clean_labels(classify(rows, facts, kb), rows), None)
        if labels:
            state.set(labels=labels)
            classifier = "p3"
            # the classification's structured output (confident true rows + their facts) feeds the KB
            stages.run("kb", lambda: kb.add_classified(labels, facts, rows), 0)
    else:
        stages.status["classify"] = "not_available"
    if facts and deadline - time.time() > 5 * 60:
        rep2: dict = {}
        claims2 = stages.run("generate", lambda: generate(kb, lexicon, input_texts, 1000, rep2), None)
        stages.move("generate", "validate", rep2.get("validate_s", 0) + rep2.get("truth_s", 0))
        if claims2 and len(claims2) >= len(claims):
            state.set(claims=claims2)
            claims, gen_report = claims2, rep2
    state.write("final")

    # Auxiliary artifacts: after the official table, never able to break it.
    blank_ids = {r["id"] for r in rows if not r["label"]}
    hidden_gold = {r["id"]: r["label"] for r in gold if r["id"] in blank_ids}
    try:
        metrics = collector.build(
            run_id=run_id, kind="arena", status=stages.overall(), errors=stages.errors,
            classification=collector.classification_metrics(
                rows, state.labels, classifier, hidden_gold, "data/practice.csv",
                "rows whose label was hidden from the input but which data/practice.csv in the image holds: "
                "optimistic, the KB was built with these labels"),
            claims=collector.claim_metrics(gen_report, claims),
            runtime=collector.runtime_metrics(stages.timings, stages.status, time.time() - t0, rss_mb()))
        out = aux_dir(output_dir)
        collector.write_json(out, "metrics.json", metrics)
        collector.write_json(out, "architecture.json",
                             collector.architecture(metrics["runtime"]["stage_status"], stages.timings, run_id))
        collector.write_claims_debug(out, claims)
        if gen_report.get("_validation_report"):
            collector.write_json(out, "validation_report.json", gen_report["_validation_report"])
        log(f"auxiliary metrics.json, architecture.json, claims_debug.jsonl, validation_report.json in {out}")
    except Exception:  # noqa: BLE001
        log(f"auxiliary artifacts skipped:\n{traceback.format_exc()}")
        metrics = None

    report = {"run_id": run_id, "timings": {k: round(v, 3) for k, v in stages.timings.items()},
              "stage_status": stages.status, "errors": stages.errors, "rows": len(rows), "blank": len(blank_ids),
              "facts": len(facts), "kb": kb.stats(),
              "claims": {k: v for k, v in gen_report.items() if not k.startswith("_")}, "rss_mb": rss_mb(),
              "total_s": round(time.time() - t0, 3), "metrics": metrics}
    log("report " + json.dumps({k: v for k, v in report.items() if k != "metrics"}, default=str))
    return report


if __name__ == "__main__":
    main()
