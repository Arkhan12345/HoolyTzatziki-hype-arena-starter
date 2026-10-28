"""Test doubles for teammates' modules. Never imported by production code.

gold_extractor: a stand-in for P3's extract(rows, lexicon) that returns the gold forms of practice
rows as agreed `Fact` objects (a perfect extractor), so the KB adapter and the full-KB generation
path can be exercised before the real extractor lands.
"""
from app.io.types import Fact


def gold_extractor(practice_rows, as_dicts=False):
    by_id = {r["id"]: r for r in practice_rows}

    def extract(rows, lexicon):
        out = []
        for r in rows:
            g = by_id.get(r["id"])
            for f in (g or {}).get("form", []):
                rec = dict(row_id=r["id"], subject=f["subject"], attribute=f["attribute"], value=f["value"],
                           span=r["text"], conf=0.9)
                out.append(rec if as_dicts else Fact(**rec))
        return out

    return extract
