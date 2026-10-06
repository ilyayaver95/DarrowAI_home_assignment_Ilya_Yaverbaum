"""Deterministic name normalization. Pure functions, no I/O.

The same code path is used for model output (`name_raw` -> canonical fields) and
for the reviewer labels in dev.jsonl, so that a prediction and a label are compared
in one canonical space. The rules come from reading the dev labels:

  * labels are `name, designator`, lower case, legal form spelled out
  * trade names (d/b/a), former names (f/k/a) and "successor to" clauses are dropped
  * punctuation is inconsistent ("u.s." vs "us", curly vs straight apostrophes),
    so matching uses a key with all non-alphanumerics removed
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Canonical legal form -> surface variants (regex fragments, matched on lower-cased text).
# Order is irrelevant: the suffix regex is end-anchored and re.search returns the
# leftmost start, i.e. the longest designator that reaches the end of the name.
_DESIGNATOR_VARIANTS: list[tuple[str, list[str]]] = [
    ("limited liability limited partnership", [r"limited liability limited partnership", r"l\.?l\.?l\.?p\.?"]),
    ("professional limited liability company", [r"professional limited liability company", r"p\.?l\.?l\.?c\.?"]),
    ("limited liability partnership", [r"limited liability partnership", r"l\.?l\.?p\.?"]),
    ("limited liability company", [r"limited liability company", r"limited liability co\.?", r"l\.?\s?l\.?\s?c\.?", r"l\.c\.?"]),
    ("limited partnership", [r"limited partnership", r"l\.?\s?p\.?"]),
    ("professional corporation", [r"professional corporation", r"p\.?c\.?"]),
    ("professional association", [r"professional association", r"p\.a\.?"]),
    ("national association", [r"national association", r"n\.\s?a\.?"]),
    ("incorporated", [r"incorporated", r"inc\.?"]),
    ("corporation", [r"corporation", r"corp\.?"]),
    ("private limited", [r"private limited", r"(?:pte|pty|pvt)\.?,?\s?(?:ltd\.?|limited)"]),
    ("limited", [r"limited", r"ltd\.?"]),
    ("public limited company", [r"public limited company", r"p\.?l\.?c\.?"]),
    ("company", [r"company", r"co\.?"]),
    ("gmbh", [r"gmbh(?:\s?&\s?co\.?\s?kg)?", r"g\.m\.b\.h\.?"]),
    ("ag", [r"ag", r"a\.g\.?"]),
    ("s.a. de c.v.", [r"s\.?\s?a\.?\s?de\s?c\.?\s?v\.?"]),
    ("s.a.r.l.", [r"s\.?a\.?r\.?l\.?"]),
    ("s.p.a.", [r"s\.p\.a\.?"]),
    ("s.a.", [r"s\.?a\.?"]),
    ("b.v.", [r"b\.?v\.?"]),
    ("n.v.", [r"n\.?v\.?"]),
]

_SUFFIX_RE = re.compile(
    r"(?:^|[\s,])(?:"
    + "|".join(
        f"(?P<d{i}>{'|'.join(variants)})" for i, (_, variants) in enumerate(_DESIGNATOR_VARIANTS)
    )
    + r")[\s.,;]*$"
)

# The reviewers use four designators only; "Corp." is recorded as "incorporated" and
# Company / PLC / GmbH / S.A. are dropped. Used for the strict scoring tier.
_LABEL_CLASS = {
    "incorporated": "incorporated",
    "corporation": "incorporated",
    "limited liability company": "limited liability company",
    "limited partnership": "limited partnership",
    "limited": "limited",
    "private limited": "limited",
}

# Everything from one of these markers onward is an alias or a role clause, not the name.
_ALIAS_RE = re.compile(
    r"""[\s,(]+(?:
        d\s*/\s*b\s*/\s*a | d\.b\.a\.? | dba | doing\s+business\s+as
      | f\s*/\s*k\s*/\s*a | f\.k\.a\.? | fka | formerly\s+known\s+as | formerly
      | a\s*/\s*k\s*/\s*a | a\.k\.a\.? | aka | also\s+known\s+as
      | n\s*/\s*k\s*/\s*a | now\s+known\s+as
      | t\s*/\s*a | trading\s+as
      | as\s+successor | successor[\s-]+(?:in[\s-]+interest|by\s+merger)
      | c\s*/\s*o | et\s+al
    )(?![a-z0-9]).*$""",
    re.X | re.S,
)
# ", a Delaware corporation" / ", an Oregon Professional Corporation": the name ends at the comma
_DESCRIPTION_GUARD = re.compile(r",\s+an?\s+[a-z]")
# ("Defendant"), ("MEMA"), (hereinafter ...), (collectively ...)
_DEFINED_TERM_RE = re.compile(r"\s*\((?:[^()]*\"[^()]*|\s*(?:hereinafter|collectively)[^()]*)\)")

_TRANSLATE = str.maketrans(
    {
        "‘": "'", "’": "'", "‚": "'", "`": "'", "´": "'",
        "“": '"', "”": '"', "„": '"',
        "–": "-", "—": "-", "‐": "-", "‑": "-", "−": "-",
        " ": " ",
    }
)


@dataclass(frozen=True)
class ParsedName:
    name_normalized: str  # lower case, designator and aliases removed, punctuation kept
    designator: str | None  # expanded legal form, or None
    key: str  # comparison key: alphanumerics only


def clean(text: str) -> str:
    """Unicode-normalize, unify quotes and dashes, lower-case, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).translate(_TRANSLATE)
    text = re.sub(r"\s+([,;.])", r"\1", text)  # OCR: "GEO GROUP , Inc."
    return re.sub(r"\s+", " ", text).strip().lower()


def strip_aliases(text: str) -> str:
    """Keep the primary legal name: drop defined terms, alias clauses and descriptions."""
    text = _DEFINED_TERM_RE.sub("", text)
    text = _ALIAS_RE.sub("", text)
    m = _DESCRIPTION_GUARD.search(text)
    if m:
        text = text[: m.start()]
    return _drop_unbalanced_parens(text)


def _drop_unbalanced_parens(text: str) -> str:
    """Remove stray brackets from column separators or a cut-off "(f/k/a ..."."""
    out, depth = [], 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                continue
            depth -= 1
        out.append(ch)
    text = "".join(out)
    while depth:  # unmatched "(" : drop the last one
        i = text.rfind("(")
        text, depth = text[:i] + text[i + 1 :], depth - 1
    return re.sub(r"\s+", " ", text).strip()


def _strip_trailing(text: str) -> str:
    """Remove dangling punctuation/connectors left after cutting a suffix."""
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"[\s,;:(\-]+$", "", text)
        text = re.sub(r"\s(?:and|&)$", "", text)
        # keep the period of a dotted abbreviation ("archroma u.s."), drop a sentence period
        if text.endswith(".") and not re.search(r"(?:^|[\s(])(?:[a-z]\.\s?){2,}$", text):
            text = text[:-1]
    return text


def split_designator(text: str) -> tuple[str, str | None]:
    """Split one trailing legal designator off a cleaned name."""
    m = _SUFFIX_RE.search(text)
    if not m:
        return _strip_trailing(text), None
    core = _strip_trailing(text[: m.start()])
    if not core:  # the whole name is a designator-like word; leave it alone
        return _strip_trailing(text), None
    idx = next(i for i in range(len(_DESIGNATOR_VARIANTS)) if m.group(f"d{i}") is not None)
    return core, _DESIGNATOR_VARIANTS[idx][0]


def match_key(name_normalized: str) -> str:
    """Comparison key: ASCII alphanumerics only, no leading "the", no trailing "trust".

    Removing every separator makes the key robust to OCR spacing ("mid- america"),
    punctuation differences ("u.s." / "us") and apostrophe variants.
    """
    text = unicodedata.normalize("NFKD", name_normalized.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("&", " and ")
    text = re.sub(r"^\s*the\s+", "", text)
    text = re.sub(r"\s+and\s*$", "", text)
    stripped = re.sub(r"[\s,]+trust\s*$", "", text)  # "camden property trust" -> label "camden property"
    key = re.sub(r"[^a-z0-9]", "", stripped) or re.sub(r"[^a-z0-9]", "", text)
    return key or name_normalized.strip().lower()


def designator_class(designator: str | None) -> str | None:
    """Collapse a legal form to the reviewers' four-value vocabulary (strict tier)."""
    return _LABEL_CLASS.get(designator) if designator else None


def parse_name(text: str) -> ParsedName:
    """Raw name (from the text or from a label) -> canonical fields."""
    cleaned = strip_aliases(clean(text)).replace('"', "")  # quotes around fictitious names
    core, designator = split_designator(cleaned.strip())
    if not core:  # nothing left after stripping: fall back to the cleaned input
        core, designator = _strip_trailing(clean(text)), None
    return ParsedName(name_normalized=core, designator=designator, key=match_key(core))
