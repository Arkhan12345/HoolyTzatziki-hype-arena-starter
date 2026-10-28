"""One sentence per claim type, in the forms of contracts/claim-forms.md.

Every template keeps the sentence minimal: subject name, attribute words, the value(s) and the
type's phrase, nothing else the referee could read as another value or date. Natural phrasings
come first; each returns None when it does not fit, and the plain "The <attribute> of <S>" form
is the fallback the validator has always accepted.
"""
from __future__ import annotations

from app.generate.render import value_text

ACRONYMS = {"ai": "AI", "api": "API", "cli": "CLI", "eu": "EU", "sdks": "SDKs", "ides": "IDEs", "ar": "AR",
            "p99": "p99", "soc2": "SOC 2", "cto": "CTO", "ceo": "CEO"}
# date attributes that read as a verb: fact "<S> joined on <date>", compare "<L> joined before <R>"
DATE_VERBS = {"joined": "joined", "founded": "was founded", "launched": "launched", "opened": "opened",
              "released": "was released", "announced": "was announced", "published": "was published"}
DATE_NOUNS = {"joined": "start date", "founded": "founding date", "launched": "launch date", "opened": "opening date",
              "released": "release date", "announced": "announcement date", "published": "publication date",
              "chair_since": "chair start date", "lease_end": "lease end date", "next_round": "next round date",
              "next_meeting": "next meeting date", "soc2_audit_start": "SOC 2 audit start date",
              "planned_general_availability": "planned general availability date",
              "stable_release": "stable release date", "training_data_cutoff": "training data cutoff"}
YES_IS = {"fully remote", "anonymous", "carbon neutral", "open source", "generally available", "discontinued",
          "eu hosted", "livestreamed", "externally audited", "soc2 certified"}
VERB_START = ("is ", "has ", "had ", "raised ", "chairs ")
AMOUNT_GT = {"duration": "longer", "money": "higher", "percent": "higher", "version": "higher", "quantity": "higher"}
AMOUNT_LT = {"duration": "shorter", "money": "lower", "percent": "lower", "version": "lower", "quantity": "lower"}
DATE_PHRASE = {"gt": "after", "lt": "before", "eq": "on the same day as"}


def attr_words(attribute: str) -> str:
    """Attribute in words, acronyms upper-cased (the referee reads them in any case)."""
    return " ".join(ACRONYMS.get(w, w) for w in attribute.split("_"))


def noun(attribute: str) -> str:
    return DATE_NOUNS.get(attribute) or attr_words(attribute)


def _date_prep(v: dict) -> str:
    return "on" if len(str(v.get("value", ""))) == 10 else "in"


# ---- fact ------------------------------------------------------------------------------------------
def _yes_no(subject: str, attribute: str, yes: bool) -> str | None:
    aw = attr_words(attribute)
    low = aw.lower()
    if low.startswith("is "):  # "is engineering hub": a negation would split the attribute's words
        return f"{subject} {aw}." if yes else None
    if low.startswith(("had ", "raised ")):  # past events: "Hooly never had major outage."
        return f"{subject} {aw}." if yes else f"{subject} never {aw}."
    if low.startswith(VERB_START):  # "has api", "chairs …": a negation would split or overstate them
        return f"{subject} {aw}." if yes else None
    if low in YES_IS or low.endswith(("ed", "able", "ible")):
        return f"{subject} is {aw}." if yes else f"{subject} is not {aw}."
    return f"{subject} has {aw}." if yes else f"{subject} has no {aw}."


def fact_text(subject: str, attribute: str, value: dict) -> str | None:
    if value.get("kind") == "boolean":
        return _yes_no(subject, attribute, bool(value["value"])) or fact_text_plain(subject, attribute, value)
    vt = value_text(value)
    if vt is None:
        return None
    if value.get("kind") == "date" and attribute in DATE_VERBS:
        return f"{subject} {DATE_VERBS[attribute]} {_date_prep(value)} {vt}."
    if value.get("kind") == "date" and attribute == "date":
        return f"{subject} took place {_date_prep(value)} {vt}."  # events: "The May 2026 outage took place on 19 May 2026."
    if attribute.endswith("_for"):
        return f"{subject} is {attr_words(attribute)} {vt}."
    counted = _counted(attribute, value, vt)
    if counted:
        return f"{subject} has {counted}."  # "Hooly has 74 employees.", "… has 362,000,000 monthly active users."
    if value.get("kind") == "version":
        if attribute == "version":
            return f"{subject} is at {vt}."
        if attribute.endswith("_version"):
            return f"{subject} is on {attr_words(attribute)[: -len(' version')]} {vt}."  # is on stable version 69.0
    if value.get("kind") == "name":
        return f"The {attr_words(attribute)} of {subject} is {vt}."
    return f"The {noun(attribute)} of {subject} is {vt}."


def _counted(attribute: str, value: dict, vt: str) -> str | None:
    """'362,000,000 monthly active users' when the attribute ends with the counted noun (<= 2 words before it)."""
    if value.get("kind") != "quantity" or " " not in vt:
        return None
    num, noun_ = vt.split(" ", 1)
    aw = attr_words(attribute)
    if aw.lower() == noun_.lower():
        return vt
    if aw.lower().endswith(" " + noun_.lower()) and len(aw.split()) - len(noun_.split()) <= 2:
        return f"{num} {aw}"
    return None


def fact_text_plain(subject: str, attribute: str, value: dict) -> str | None:
    aw = attr_words(attribute)
    if value.get("kind") == "boolean":
        return f"The {aw} flag of {subject} is {'set' if value['value'] else 'not set'}."
    vt = value_text(value)
    return None if vt is None else f"The {aw} of {subject} is {vt}."


def fact_text_alt(subject: str, attribute: str, value: dict) -> str | None:
    aw = noun(attribute)
    if value.get("kind") == "boolean":
        return fact_text_plain(subject, attribute, value)
    vt = value_text(value)
    return None if vt is None else f"{subject} has a {aw} of {vt}."


# ---- change ----------------------------------------------------------------------------------------
def change_text(subject: str, attribute: str, frm: dict, to: dict) -> str | None:
    a, b = value_text(frm), value_text(to)
    if a is None or b is None or frm.get("kind") == "boolean":
        return None
    ca, cb = _counted(attribute, frm, a), _counted(attribute, to, b)
    if ca and cb:
        return f"{subject} went from {ca} to {cb}."  # Hooly Chrome went from 341,000,000 monthly active users to …
    if attribute == "version":
        return f"{subject} went from {a} to {b}."
    return f"The {noun(attribute)} of {subject} went from {a} to {b}."


# ---- at_event --------------------------------------------------------------------------------------
def at_event_text(subject: str, attribute: str, event: str, value: dict) -> str | None:
    aw = noun(attribute)
    if value.get("kind") == "boolean":
        return f"The {attr_words(attribute)} flag of {subject} was {'set' if value['value'] else 'not set'} at {event}."
    vt = value_text(value)
    return None if vt is None else f"The {aw} of {subject} was {vt} at {event}."


# ---- relation --------------------------------------------------------------------------------------
def relation_text(subject: str, link: str, attribute: str, value: dict) -> str | None:
    lw = attr_words(link)
    if value.get("kind") == "boolean":
        return f"The {attr_words(attribute)} flag of the {lw} of {subject} is {'set' if value['value'] else 'not set'}."
    vt = value_text(value)
    if vt is None:
        return None
    if value.get("kind") == "date" and attribute in DATE_VERBS:
        return f"The {lw} of {subject} {DATE_VERBS[attribute]} {_date_prep(value)} {vt}."  # The CTO of Hooly joined on …
    return f"The {noun(attribute)} of the {lw} of {subject} is {vt}."


# ---- compare ---------------------------------------------------------------------------------------
def compare_text(left: tuple[str, str], right: tuple[str, str], op: str, between: str, kind: str | None = None) -> str:
    (ls, la), (rs, ra) = left, right
    if la == ra:
        natural = _compare_same(ls, rs, la, op, between, kind)
        if natural:
            return natural
    if between == "date":
        return f"The {noun(la)} of {ls} is {DATE_PHRASE[op]} the {noun(ra)} of {rs}."
    phrase = {"gt": "higher than", "lt": "lower than", "eq": "the same as"}[op]
    return f"The {noun(la)} of {ls} is {phrase} the {noun(ra)} of {rs}."


def compare_text_plain(left, right, op, between) -> str:
    (ls, la), (rs, ra) = left, right
    if between == "date":
        return f"The {noun(la)} of {ls} is {DATE_PHRASE[op]} the {noun(ra)} of {rs}."
    phrase = {"gt": "higher than", "lt": "lower than", "eq": "the same as"}[op]
    return f"The {noun(la)} of {ls} is {phrase} the {noun(ra)} of {rs}."


def _compare_same(ls: str, rs: str, attribute: str, op: str, between: str, kind: str | None) -> str | None:
    if between == "date":
        if attribute in DATE_VERBS:
            if op == "eq":
                return f"{ls} {DATE_VERBS[attribute]} on the same day as {rs}."
            return f"{ls} {DATE_VERBS[attribute]} {DATE_PHRASE[op]} {rs}."
        if attribute == "date":
            return f"{ls} took place {DATE_PHRASE[op]} {rs}." if op != "eq" else f"{ls} took place on the same day as {rs}."
        return None
    aw = noun(attribute)
    if len(aw.split()) > 3:  # "… " in a phrase stands for at most three words
        return None
    if attribute == "price" and op != "eq":
        return f"{ls} costs {'more' if op == 'gt' else 'less'} than {rs}."
    if attribute == "price":
        return f"{ls} costs the same as {rs}."
    if (kind or "") == "money":
        return None  # "has a higher cash/revenue" reads wrong: "The revenue of A is higher than the revenue of B"
    if op == "eq":
        if (kind or "quantity") == "quantity" and aw.endswith("s") and not aw.endswith("ss"):
            return f"{ls} has as many {aw} as {rs}."  # equal counts, not the same people
        return f"{ls} has the same {aw} as {rs}."
    k = kind or "quantity"
    if k == "quantity" and aw.endswith("s") and not aw.endswith("ss"):
        return f"{ls} has {'more' if op == 'gt' else 'fewer'} {aw} than {rs}."  # more employees than
    word = (AMOUNT_GT if op == "gt" else AMOUNT_LT).get(k, "higher" if op == "gt" else "lower")
    return f"{ls} has a {word} {aw} than {rs}."
