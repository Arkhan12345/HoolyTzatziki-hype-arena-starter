"""Write the output table atomically: temp file in the output dir, then os.replace."""
from __future__ import annotations

import csv
import json
import os
import threading

COLUMNS = ["id", "type", "text", "form", "label", "confidence"]
_lock = threading.Lock()


def _conf(c) -> str:
    if c is None or c == "":
        return ""
    return f"{min(1.0, max(0.5, float(c))):.3f}"


def label_rows(labels) -> list[list[str]]:
    out = []
    for lb in labels:
        label = lb.label if lb.label in ("true", "false", "unsure") else "unsure"
        out.append([lb.id, "", "", "", label, "" if label == "unsure" else _conf(lb.confidence or 0.5)])
    return out


def claim_rows(claims) -> list[list[str]]:
    return [
        [c.id, c.type, c.text, json.dumps(c.form, ensure_ascii=False, separators=(", ", ": ")), "true", _conf(c.confidence)]
        for c in claims
    ]


def write_table(output_dir: str, labels, claims) -> str:
    """Write labels then claims to output_dir/output.csv; returns the path."""
    os.makedirs(output_dir, exist_ok=True)
    final = os.path.join(output_dir, "output.csv")
    rows = label_rows(labels) + claim_rows(claims)
    with _lock:
        tmp = os.path.join(output_dir, f".output.{os.getpid()}.{threading.get_ident()}.tmp")
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            w.writerow(COLUMNS)
            w.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, final)
    return final
