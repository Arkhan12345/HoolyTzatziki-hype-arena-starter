"""P3 extract entry point used by app/run.py: extract(rows, lexicon) -> list[Fact].

The label model (app.classify, frozen FINAL) reads facts from each sentence itself, so no extracted facts are fed
to the knowledge base yet: extract returns an empty list and the claims keep using practice gold only. Feeding
extracted facts to the KB (staleness vetoes, more claim evidence) is a separate decision for the KB and claim owners;
the option A extractor in extractor.py is the candidate for it.
"""
from __future__ import annotations


def extract(rows, lexicon):
    return []
