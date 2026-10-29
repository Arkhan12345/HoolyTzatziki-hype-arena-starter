"""Build run metrics (schemas.METRICS) from what the pipeline actually produced.

Evaluation numbers (accuracy, macro-F1, confusion, calibration) are computed only over rows whose
true label is known from data/practice.csv; with no such rows they are null.
"""
from __future__ import annotations

import json
import os
import statistics
import time
import uuid

from app.metrics.schemas import SCHEMA_VERSION, STAGES

LABELS = ["true", "false", "unsure"]
HERE = os.path.dirname(os.path.abspath(__file__))
ARCHITECTURE_FILE = os.path.join(os.path.dirname(HERE), "api", "architecture.json")


class Stages:
    """Time, status and errors per canonical stage (schemas.STAGES); several calls add up."""

    def __init__(self, log=None):
        self.timings: dict[str, float] = {}
        self.status: dict[str, str] = {}
        self.errors: list[dict] = []
        self.log = log or (lambda msg: None)

    def run(self, stage: str, fn, fallback=None, status_if_ok: str = "ok"):
        t = time.time()
        try:
            out = fn()
            st = status_if_ok
        except Exception as e:  # noqa: BLE001 - every stage falls back, the run goes on
            import traceback

            self.log(f"stage {stage} failed, falling back:\n{traceback.format_exc()}")
            self.errors.append({"stage": stage, "message": repr(e)[:300]})
            out = fallback() if callable(fallback) else fallback
            st = "fallback"
        self.add(stage, time.time() - t, st)
        return out

    def add(self, stage: str, seconds: float, status: str | None = None) -> None:
        self.timings[stage] = self.timings.get(stage, 0.0) + max(0.0, seconds)
        if status and not (self.status.get(stage) in ("fallback", "failed") and status == "ok"):
            self.status[stage] = status

    def move(self, src: str, dst: str, seconds: float) -> None:
        """Re-attribute time measured inside one stage to another (validation inside generation)."""
        seconds = min(max(0.0, seconds), self.timings.get(src, 0.0))
        self.timings[src] = self.timings.get(src, 0.0) - seconds
        self.add(dst, seconds, self.status.get(src, "ok"))

    def overall(self) -> str:
        return "partial" if self.errors or any(v == "fallback" for v in self.status.values()) else "ok"


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def confidence_stats(values, bins: int = 10) -> dict | None:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    lo, width = 0.5, 0.5 / bins
    hist = [{"lo": round(lo + i * width, 3), "hi": round(lo + (i + 1) * width, 3), "count": 0} for i in range(bins)]
    for v in vals:
        hist[min(bins - 1, max(0, int((v - lo) / width)))]["count"] += 1
    return {"n": len(vals), "min": round(min(vals), 4), "max": round(max(vals), 4),
            "mean": round(statistics.fmean(vals), 4), "median": round(statistics.median(vals), 4), "histogram": hist}


def _macro_f1(pairs, labels) -> float | None:
    f1s = []
    for lb in labels:
        tp = sum(1 for p, g in pairs if p == lb and g == lb)
        fp = sum(1 for p, g in pairs if p == lb and g != lb)
        fn = sum(1 for p, g in pairs if p != lb and g == lb)
        if tp + fp + fn == 0:
            continue
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return round(sum(f1s) / len(f1s), 4) if f1s else None


def evaluation(labels, gold_by_id: dict[str, str], label_source: str, note: str) -> dict | None:
    pairs = [(lb.label, gold_by_id[lb.id]) for lb in labels if gold_by_id.get(lb.id) in LABELS]
    if not pairs:
        return None
    present = [lb for lb in LABELS if any(g == lb for _, g in pairs) or any(p == lb for p, _ in pairs)]
    matrix = [[sum(1 for p, g in pairs if g == gl and p == pl) for pl in LABELS] for gl in LABELS]
    # calibration: confidence vs. how often a confident true/false call was right
    scored = [(float(lb.confidence), lb.label == gold_by_id[lb.id]) for lb in labels
              if lb.label in ("true", "false") and lb.confidence is not None and gold_by_id.get(lb.id) in LABELS]
    calib, ece = [], None
    if scored:
        bins = [(0.5 + i * 0.1, 0.6 + i * 0.1) for i in range(5)]
        err = 0.0
        for lo, hi in bins:
            b = [(c, ok) for c, ok in scored if lo <= c < hi or (hi >= 1.0 and c == 1.0)]
            if b:
                pred, obs = statistics.fmean(c for c, _ in b), sum(ok for _, ok in b) / len(b)
                calib.append({"predicted": round(pred, 4), "observed": round(obs, 4), "count": len(b)})
                err += len(b) * abs(pred - obs)
        ece = round(err / len(scored), 4)
    base_pairs = [("true", g) for _, g in pairs]
    return {
        "label_source": label_source, "n": len(pairs), "note": note,
        "accuracy": round(sum(p == g for p, g in pairs) / len(pairs), 4),
        "macro_f1": _macro_f1(pairs, present),
        "baseline": {"name": "all true", "accuracy": round(sum(g == "true" for _, g in pairs) / len(pairs), 4),
                     "macro_f1": _macro_f1(base_pairs, present)},
        "confusion": {"labels": LABELS, "matrix": matrix},
        "calibration": calib, "ece": ece,
    }


def classification_metrics(rows, labels, classifier: str, gold_by_id: dict[str, str], label_source: str,
                           note: str) -> dict:
    counts = {k: 0 for k in LABELS}
    for lb in labels:
        counts[lb.label if lb.label in counts else "unsure"] += 1
    return {
        "rows_total": len(rows),
        "rows_labelled_in_input": sum(1 for r in rows if r.get("label")),
        "rows_predicted": len(labels),
        "predicted": counts,
        "confidence": confidence_stats(lb.confidence for lb in labels if lb.label != "unsure"),
        "classifier": classifier,
        "evaluation": evaluation(labels, gold_by_id, label_source, note) if gold_by_id else None,
    }


def _flat_rejections(rejected) -> dict:
    return {t: dict(v) for t, v in (rejected or {}).items()}


def claim_metrics(report: dict | None, claims) -> dict:
    r = report or {}
    return {
        "candidates_total": r.get("candidates"),
        "candidates_by_type": r.get("candidates_by_type", {}),
        "rejected_by_reason": _flat_rejections(r.get("rejected")),
        "valid": r.get("valid"),
        "unique_text": r.get("valid"),  # screen() keeps one candidate per normalized text
        "unique_semantic": r.get("distinct_semantic"),
        "semantic_duplicates": r.get("semantic_duplicates"),
        "selected": len(claims) if claims is not None else None,
        "selected_by_type": r.get("by_type", {}),
        "selected_semantic_duplicates": r.get("selected_semantic_duplicates"),
        "confidence": confidence_stats(c.confidence for c in claims or []),
        "gold_only_supported": r.get("selected_gold_only"),
        "kb_supported": r.get("selected_kb_supported"),
        "inventory": r.get("inventory"),
        # independent validator (app/validate/claims.py): candidates' verdicts and the final set's
        "validation": {
            "candidates_verdicts_by_type": r.get("verdicts_by_type"),
            "inverse_compares_detected": r.get("semantic_duplicates_inverse_compare"),
            "paraphrases_detected": r.get("semantic_duplicates_paraphrase"),
            "selected": r.get("validation"),
        } if r.get("validation") else None,
    }


def runtime_metrics(timings: dict, stage_status: dict, total_s: float | None, rss_mb: float | None) -> dict:
    return {"total_s": None if total_s is None else round(total_s, 3),
            "stages": {s: (round(timings[s], 4) if timings.get(s) is not None
                           and stage_status.get(s) not in ("not_available", "skipped") else None) for s in STAGES},
            "stage_status": {s: stage_status.get(s, "idle") for s in STAGES},
            "rss_mb": None if rss_mb is None or rss_mb < 0 else round(rss_mb, 1)}


def build(*, run_id, kind, status, errors, classification, claims, runtime) -> dict:
    ev = classification.get("evaluation")
    return {
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "kind": kind, "created_at": now_iso(),
        "status": status, "errors": errors, "classification": classification, "claims": claims, "runtime": runtime,
        # frontend /metrics page reads these two keys directly
        "model": None if ev is None else {k: ev[k] for k in ("accuracy", "macro_f1", "baseline", "confusion",
                                                             "calibration", "n", "label_source", "note")},
        "claims_by_type": claims.get("selected_by_type") or {},
    }


def empty() -> dict:
    """Metrics before any run: every number null."""
    return build(run_id=None, kind="none", status="none", errors=[],
                 classification={"rows_total": 0, "rows_labelled_in_input": 0, "rows_predicted": 0,
                                 "predicted": {k: 0 for k in LABELS}, "confidence": None, "classifier": "baseline",
                                 "evaluation": None},
                 claims=claim_metrics(None, None), runtime=runtime_metrics({}, {}, None, None))


# The classifier's steps, in order, as the diagram shows them; their seconds come from classify()'s report.
CLASSIFY_STEPS = [("items", "generate_items"), ("lkb", "kb_em"), ("train", "fit"), ("classify", "predict")]
SUBSTAGE_OF = {s: "classify" for s, _ in CLASSIFY_STEPS}


def expand_classify(stages: list, timings: dict, details: list, seconds: dict | None, mode: str | None = None):
    """Replace the "classify" stage by its steps (items -> lkb -> train -> classify) for the demo diagram."""
    if not seconds or "classify" not in stages:
        return list(stages), dict(timings), list(details)
    parent = next((d for d in details if d["id"] == "classify"), {"status": "idle"})
    out_stages, out_details, out_t = [], [], dict(timings)
    for s in stages:
        if s != "classify":
            out_stages.append(s)
            out_details += [d for d in details if d["id"] == s]
            continue
        for sub, key in CLASSIFY_STEPS:
            out_stages.append(sub)
            out_t[sub] = round(float(seconds.get(key) or 0.0), 4)
            note = {"items": "true, flipped false, unstated unsure", "lkb": "input labels pin, the rest vote; EM page trust",
                    "train": f"{mode or 'prod'}: " + ("half the items held out" if mode == "debug" else "all items"),
                    "classify": "labels + trust + sources"}[sub]
            out_details.append({"id": sub, "seconds": out_t[sub], "status": parent.get("status", "idle"), "note": note})
    return out_stages, out_t, out_details


def architecture(stage_status: dict | None = None, timings: dict | None = None, run_id: str | None = None) -> dict:
    with open(ARCHITECTURE_FILE, encoding="utf-8") as f:
        arch = json.load(f)
    stage_status, timings = stage_status or {}, timings or {}
    for n in arch["nodes"]:
        st = n.get("stage", "")
        optional_off = n.get("kind") == "optional" and os.environ.get("USE_NEURO") != "1"
        n["status"] = "not_available" if optional_off else stage_status.get(st, stage_status.get(SUBSTAGE_OF.get(st, ""), "idle"))
        n["seconds"] = None if optional_off else timings.get(st)
    arch.update({"stage_status": {s: stage_status.get(s, "idle") for s in STAGES},
                 "timings": {s: timings.get(s) for s in STAGES}, "run_id": run_id})
    return arch


def claim_debug_record(claim) -> dict:
    meta = getattr(claim, "meta", None) or {}
    return {"id": claim.id, "type": claim.type, "text": claim.text, "form": claim.form,
            "confidence": claim.confidence, "sources": meta.get("sources", []),
            "semantic_key": meta.get("semantic_key")}


def write_claims_debug(directory: str, claims) -> str:
    """One JSON line per selected claim with the sentences it was built from (internal debug)."""
    os.makedirs(directory, exist_ok=True)
    final = os.path.join(directory, "claims_debug.jsonl")
    tmp = final + f".{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for c in claims:
            f.write(json.dumps(claim_debug_record(c), ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, final)
    return final


def write_json(directory: str, name: str, data: dict) -> str | None:
    """Auxiliary artifact, atomic; failures are logged by the caller and never raise into the run."""
    os.makedirs(directory, exist_ok=True)
    final = os.path.join(directory, name)
    tmp = final + f".{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, final)
    return final
