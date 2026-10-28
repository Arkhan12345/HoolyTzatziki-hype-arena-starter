"""Steps for features/docs.feature: every relative markdown link resolves."""
import re
from pathlib import Path

from pytest_bdd import given, scenarios, then, when

from conftest import ROOT

scenarios("docs.feature")
LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
FENCE = re.compile(r"(```|~~~).*?\1", re.S)
INLINE = re.compile(r"`[^`\n]*`")


def markdown_files():
    files = [ROOT / "README.md"]
    for pattern in ("docs/*.md", "team/*.md", "pitch/*.md", "eda/**/*.md"):
        files += sorted(ROOT.glob(pattern))
    return [f for f in files if f.is_file() and ".venv" not in f.parts]


def relative_links(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    text = INLINE.sub("", FENCE.sub("", text))
    for target in LINK.findall(text):
        if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) or target.startswith(("#", "{{")):
            continue
        yield target.split("#")[0].split("?")[0]


@given("the markdown files of the repo's docs", target_fixture="ctx")
def md_files():
    return {"files": markdown_files()}


@when("I collect their relative links")
def collect(ctx):
    ctx["links"] = [(f, t) for f in ctx["files"] for t in relative_links(f) if t]


@then("every link points to an existing file or folder")
def all_resolve(ctx):
    broken = [f"{f.relative_to(ROOT)} -> {t}" for f, t in ctx["links"]
              if not (f.parent / t).exists() and not (ROOT / t.lstrip("/")).exists()]
    assert not broken, "broken links:\n" + "\n".join(broken)
