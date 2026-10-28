"""One sentence per claim type, in the forms of contracts/claim-forms.md.

Every template keeps the sentence minimal: subject name, attribute words, the value(s) and
the type's phrase, nothing else the referee could read as another value or date.
"""
from __future__ import annotations

from app.generate.render import attr_words, value_text

AMOUNT_PHRASE = {"gt": "higher than", "lt": "lower than", "eq": "the same as"}
DATE_PHRASE = {"gt": "after", "lt": "before", "eq": "on the same day as"}


def fact_text(subject: str, attribute: str, value: dict) -> str | None:
    aw = attr_words(attribute)
    if value.get("kind") == "boolean":
        return f"The {aw} flag of {subject} is {'set' if value['value'] else 'not set'}."
    vt = value_text(value)
    return None if vt is None else f"The {aw} of {subject} is {vt}."


def fact_text_alt(subject: str, attribute: str, value: dict) -> str | None:
    aw = attr_words(attribute)
    if value.get("kind") == "boolean":
        return f"{subject} has the {aw} flag {'set' if value['value'] else 'not set'}."
    vt = value_text(value)
    return None if vt is None else f"{subject} has a {aw} of {vt}."


def change_text(subject: str, attribute: str, frm: dict, to: dict) -> str | None:
    a, b = value_text(frm), value_text(to)
    if a is None or b is None or frm.get("kind") == "boolean":
        return None
    return f"The {attr_words(attribute)} of {subject} went from {a} to {b}."


def at_event_text(subject: str, attribute: str, event: str, value: dict) -> str | None:
    aw = attr_words(attribute)
    if value.get("kind") == "boolean":
        return f"The {aw} flag of {subject} was {'set' if value['value'] else 'not set'} at {event}."
    vt = value_text(value)
    return None if vt is None else f"The {aw} of {subject} was {vt} at {event}."


def relation_text(subject: str, link: str, attribute: str, value: dict) -> str | None:
    aw, lw = attr_words(attribute), attr_words(link)
    if value.get("kind") == "boolean":
        return f"The {aw} flag of the {lw} of {subject} is {'set' if value['value'] else 'not set'}."
    vt = value_text(value)
    return None if vt is None else f"The {aw} of the {lw} of {subject} is {vt}."


def compare_text(left: tuple[str, str], right: tuple[str, str], op: str, between: str) -> str:
    phrase = (DATE_PHRASE if between == "date" else AMOUNT_PHRASE)[op]
    (ls, la), (rs, ra) = left, right
    return f"The {attr_words(la)} of {ls} is {phrase} the {attr_words(ra)} of {rs}."
