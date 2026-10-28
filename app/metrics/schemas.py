"""JSON Schemas of what the backend hands out: run metrics, the architecture graph, /run results.

These are the contract docs/API_CONTRACT.md describes; tests validate every response against them.
A number that cannot be computed is null, never a guess.
"""
from __future__ import annotations

SCHEMA_VERSION = "1.0"
STAGES = ["load", "extract", "kb", "classify", "generate", "validate", "write"]
STAGE_STATUS = ["idle", "ok", "fallback", "not_available", "failed", "skipped"]

_num = {"type": ["number", "null"]}
_int = {"type": ["integer", "null"]}
_counts = {"type": "object", "additionalProperties": {"type": "integer"}}

CONFIDENCE_STATS = {
    "type": ["object", "null"],
    "required": ["n", "min", "max", "mean", "median", "histogram"],
    "properties": {
        "n": {"type": "integer"}, "min": _num, "max": _num, "mean": _num, "median": _num,
        "histogram": {"type": "array", "items": {
            "type": "object", "required": ["lo", "hi", "count"],
            "properties": {"lo": {"type": "number"}, "hi": {"type": "number"}, "count": {"type": "integer"}}}},
    },
}

EVALUATION = {
    "type": ["object", "null"],
    "required": ["label_source", "n", "accuracy", "macro_f1", "baseline", "confusion", "calibration", "ece", "note"],
    "properties": {
        "label_source": {"type": "string"}, "n": {"type": "integer"},
        "accuracy": _num, "macro_f1": _num, "ece": _num, "note": {"type": "string"},
        "baseline": {"type": "object", "required": ["name", "accuracy", "macro_f1"],
                     "properties": {"name": {"type": "string"}, "accuracy": _num, "macro_f1": _num}},
        "confusion": {"type": "object", "required": ["labels", "matrix"],
                      "properties": {"labels": {"type": "array", "items": {"type": "string"}},
                                     "matrix": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}}}},
        "calibration": {"type": "array", "items": {
            "type": "object", "required": ["predicted", "observed", "count"],
            "properties": {"predicted": {"type": "number"}, "observed": {"type": "number"}, "count": {"type": "integer"}}}},
    },
}

METRICS = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Tzatziki run metrics",
    "type": "object",
    "required": ["schema_version", "run_id", "kind", "created_at", "status", "errors", "classification", "claims",
                 "runtime", "model", "claims_by_type"],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "run_id": {"type": ["string", "null"]},
        "kind": {"enum": ["arena", "demo", "none"]},
        "created_at": {"type": ["string", "null"]},
        "status": {"enum": ["ok", "partial", "failed", "running", "none"]},
        "errors": {"type": "array", "items": {"type": "object", "required": ["stage", "message"],
                                              "properties": {"stage": {"type": "string"}, "message": {"type": "string"}}}},
        "classification": {
            "type": "object",
            "required": ["rows_total", "rows_labelled_in_input", "rows_predicted", "predicted", "confidence",
                         "classifier", "evaluation"],
            "properties": {
                "rows_total": {"type": "integer"}, "rows_labelled_in_input": {"type": "integer"},
                "rows_predicted": {"type": "integer"},
                "predicted": {"type": "object", "required": ["true", "false", "unsure"],
                              "properties": {k: {"type": "integer"} for k in ("true", "false", "unsure")}},
                "confidence": CONFIDENCE_STATS,
                "classifier": {"enum": ["p3", "baseline"]},
                "evaluation": EVALUATION,
            },
        },
        "claims": {
            "type": "object",
            "required": ["candidates_total", "candidates_by_type", "rejected_by_reason", "valid", "unique_text",
                         "unique_semantic", "semantic_duplicates", "selected", "selected_by_type",
                         "selected_semantic_duplicates", "confidence", "gold_only_supported", "kb_supported",
                         "inventory"],
            "properties": {
                "candidates_total": _int, "candidates_by_type": _counts,
                "rejected_by_reason": {"type": "object", "additionalProperties": _counts},
                "valid": _int, "unique_text": _int, "unique_semantic": _int, "semantic_duplicates": _int,
                "selected": _int, "selected_by_type": _counts, "selected_semantic_duplicates": _int,
                "confidence": CONFIDENCE_STATS, "gold_only_supported": _int, "kb_supported": _int,
                "inventory": {"type": ["object", "null"]},
            },
        },
        "runtime": {
            "type": "object", "required": ["total_s", "stages", "stage_status", "rss_mb"],
            "properties": {
                "total_s": _num, "rss_mb": _num,
                "stages": {"type": "object", "required": STAGES, "properties": {s: _num for s in STAGES}},
                "stage_status": {"type": "object", "required": STAGES,
                                 "properties": {s: {"enum": STAGE_STATUS} for s in STAGES}},
            },
        },
        # Kept for the frontend's /metrics page (frontend/src/api/types.ts `Metrics`):
        "model": {"type": ["object", "null"]},
        "claims_by_type": _counts,
    },
}

ARCHITECTURE = {
    "type": "object",
    "required": ["nodes", "edges", "stage_status", "timings", "run_id"],
    "properties": {
        "nodes": {"type": "array", "minItems": 1, "items": {
            "type": "object", "required": ["id", "label", "kind", "stage", "owner", "description", "status"],
            "properties": {"id": {"type": "string"}, "label": {"type": "string"}, "kind": {"type": "string"},
                           "stage": {"type": "string"}, "owner": {"type": "string"}, "description": {"type": "string"},
                           "status": {"enum": STAGE_STATUS}, "seconds": _num,
                           "position": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}}}}}},
        "edges": {"type": "array", "items": {
            "type": "object", "required": ["source", "target"],
            "properties": {"source": {"type": "string"}, "target": {"type": "string"}, "label": {"type": "string"},
                           "dashed": {"type": "boolean"}, "loop": {"type": "boolean"}}}},
        "stage_status": {"type": "object"}, "timings": {"type": "object"}, "run_id": {"type": ["string", "null"]},
    },
}

_LABEL = {"type": "object", "required": ["id", "label", "confidence"],
          "properties": {"id": {"type": "string"}, "label": {"enum": ["true", "false", "unsure"]},
                         "confidence": _num, "gold": {"type": ["string", "null"]}}}
_CLAIM = {"type": "object", "required": ["id", "type", "text", "form", "confidence"],
          "properties": {"id": {"type": "string"}, "type": {"enum": ["fact", "compare", "change", "at_event", "relation"]},
                         "text": {"type": "string"}, "form": {"type": "object"}, "confidence": {"type": "number"}}}

RUN_RESULT = {
    "type": "object",
    "required": ["run_id", "status", "source", "created_at", "rows", "labels", "predictions", "facts", "claims",
                 "claims_total", "claims_by_type", "timings", "stages", "stage_details", "metrics", "errors"],
    "properties": {
        "run_id": {"type": "string"}, "status": {"enum": ["running", "ok", "partial", "failed"]},
        "source": {"type": "string"}, "created_at": {"type": "string"},
        "rows": {"type": "array", "items": {"type": "object", "required": ["id", "text", "date"]}},
        "labels": {"type": "array", "items": _LABEL},
        "predictions": {"type": "array", "items": {
            "type": "object", "required": ["id", "text", "date", "label", "confidence", "gold", "facts", "evidence", "explanation"]}},
        "facts": {"type": "array"},
        "claims": {"type": "array", "items": _CLAIM},
        "claims_total": {"type": "integer"}, "claims_by_type": _counts,
        "timings": {"type": "object", "required": STAGES, "additionalProperties": {"type": ["number", "null"]}}, "stages": {"type": "array", "items": {"enum": STAGES}},
        "stage_details": {"type": "array"}, "metrics": METRICS, "errors": {"type": "array"},
    },
}
