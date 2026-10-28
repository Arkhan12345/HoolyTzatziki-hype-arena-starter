"""Read the universe: rows of id,text,date,label and the lexicon."""
from __future__ import annotations

import csv
import json
import os
import sys

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def read_rows(path: str) -> list[dict]:
    """Read a CSV of rows, comma or semicolon separated, BOM tolerant.

    Column names are lower-cased and stripped; missing cells become "".
    """
    with open(path, encoding="utf-8-sig", newline="") as f:
        head = f.readline()
        f.seek(0)
        delimiter = ";" if head.count(";") > head.count(",") else ","
        rows = []
        for r in csv.DictReader(f, delimiter=delimiter):
            row = {(k or "").strip().lower(): (v or "") for k, v in r.items() if k is not None}
            row.setdefault("id", "")
            row.setdefault("text", "")
            row.setdefault("date", "")
            row["label"] = (row.get("label") or "").strip().lower()
            row["id"] = row["id"].strip()
            row["date"] = row["date"].strip()
            if row["id"]:
                rows.append(row)
    return rows


def read_lexicon(path: str) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        lex = json.load(f)
    lex.setdefault("subjects", [])
    lex.setdefault("events", [])
    return lex


def read_universe(universe_dir: str) -> tuple[list[dict], dict]:
    rows = read_rows(os.path.join(universe_dir, "universe.csv"))
    lexicon = read_lexicon(os.path.join(universe_dir, "lexicon.json"))
    return rows, lexicon


def read_practice(path: str) -> list[dict]:
    """Practice rows with their gold `form` parsed (empty list if unparsable)."""
    rows = read_rows(path)
    for r in rows:
        try:
            r["form"] = json.loads(r.get("form") or "[]")
        except ValueError:
            r["form"] = []
    return rows
