"""Every example table in contracts/value-forms.md as a test of eda/extract_simple.parse_values.

Read examples pass when a parsed value has the same comparable key (app/kb/normalize.key: units convert,
op included) as the expected one; "Not read" examples pass when no parsed value has that key.
Run: .venv/Scripts/python eda/test_values_simple.py   (prints pass count and the failures)
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eda"))
from app.kb.normalize import key  # noqa: E402
from extract_simple import is_approx_or_negated, parse_boolean, parse_values  # noqa: E402

ROW = re.compile(r"^\|\s*`(?P<w>[^`]+)`\s*\|\s*`(?P<v>\{[^`]+\})`\s*\|")


def examples():
    section, out = None, []
    for line in (ROOT / "contracts" / "value-forms.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
        m = ROW.match(line)
        if m:
            out.append((section, m.group("w"), json.loads(m.group("v"))))
    return out


def reads(written, expected):
    k = expected["kind"]
    if k == "name":
        at = written.replace(" ", "").lower().find(expected["value"].replace(" ", "").lower())
        if at < 0:
            return False
        start = written.lower().find(expected["value"].split()[0].lower())
        return not is_approx_or_negated(written, start, start + len(expected["value"]))
    if k == "boolean":
        return parse_boolean(written) == expected
    return any(key(v) == key(expected) for _, _, v in parse_values(written))


def main():
    ex = examples()
    passed, fails = 0, []
    for section, written, expected in ex:
        got = reads(written, expected)
        ok = (not got) if section == "Not read" else got
        passed += ok
        if not ok:
            fails.append((section, written, expected, [v for _, _, v in parse_values(written)]))
    print(f"value-forms examples: {passed}/{len(ex)} pass")
    for f in fails:
        print("  FAIL", f[0], "|", f[1], "| expected", f[2], "| got", f[3])
    return passed, len(ex)


if __name__ == "__main__":
    main()
