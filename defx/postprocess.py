"""Deterministic post-processing: model output -> prediction record. Pure functions.

Everything that can be decided without a model is decided here, so it can be unit
tested and so a prompt change cannot silently change the output contract:
canonical name fields, de-duplication, grounding against the source text, and the
scope policy (which sued parties go into `defendants`).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .normalize import parse_name
from .schema import Defendant, ExcludedParty, Extraction, Prediction

_PLACEHOLDER_RE = re.compile(
    r"""^\W*does\b(?:\s+(?:\d|[ivx]+\b|one\b|through\b|defendants?\b)|\W*$)   # DOES 1-10, "Does"
      | ^\W*(?:john|jane|richard|mary|baby)\s+(?:does?|roes?)\b           # John Doe, Jane Roe 1, John Doe Corporation
      | \b(?:does?|roes?)\s+(?:\d|[ivx]+\b|one\b)                        # "and Does 1 through 50", Roes I-X
      | ^\W*(?:doe|roe)\s+(?:defendants?|corporations?|compan(?:y|ies)|entit(?:y|ies))\b
      | \bunknown\s+(?:named\s+)?(?:defendants?|agents?|officers?|persons?|entit)
      | ^\W*(?:abc|xyz)\s+(?:corp|corporation|company|inc|llc|entit)\w*\W*$   # ABC Corporation, not ABC Supply Co.
      | \bfictitious\b""",
    re.I | re.X,
)
# A real company can start with "Doe" (Doe Run Resources Corporation), so a bare surname is not enough.

_US_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA", "colorado": "CO",
    "connecticut": "CT", "delaware": "DE", "district of columbia": "DC", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD", "massachusetts": "MA",
    "michigan": "MI", "minnesota": "MN", "mississippi": "MS", "missouri": "MO", "montana": "MT",
    "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
    "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY", "puerto rico": "PR",
}
_STATE_BY_ABBREVIATION = {abbr: name for name, abbr in _US_STATES.items()}


# Short-form (MDL) complaints print every possible defendant next to a check box.
_MENU_HEADER_RE = re.compile(
    r"check\s+all\s+(?:defendants?\s+)?applicable|check\s+(?:the\s+)?defendants?\s+against\s+whom", re.I
)
_CHECK_MARK_RE = re.compile(r"[☑☒✓✔🗹]|\[\s*[xX✓]\s*\]|\(\s*[xX]\s*\)")


def text_before_unmarked_menu(text: str) -> str | None:
    """If the document has a defendant check-list with no visible selection, return the
    text that precedes the list (caption and preamble); otherwise None.

    Hand-written check marks are usually lost in OCR. An unmarked menu says nothing
    about who is sued, so only names that also appear before it can be trusted.
    """
    header = _MENU_HEADER_RE.search(text)
    if not header or _CHECK_MARK_RE.search(text, header.start()):
        return None
    return text[: header.start()]


@dataclass(frozen=True)
class Policy:
    """Scope of `defendants`. The default follows the reviewers' convention in dev.jsonl:
    organizations only, no placeholder parties. `raw` keeps everything the model returns."""

    organizations_only: bool = True
    drop_placeholders: bool = True

    @classmethod
    def named(cls, name: str) -> "Policy":
        if name == "raw":
            return cls(organizations_only=False, drop_placeholders=False)
        if name == "labels":
            return cls()
        raise ValueError(f"unknown policy: {name}")


def squeeze(text: str) -> str:
    """Lower-case ASCII alphanumerics only: comparison form robust to OCR spacing."""
    text = unicodedata.normalize("NFKD", text.lower())
    return re.sub(r"[^a-z0-9]", "", "".join(c for c in text if not unicodedata.combining(c)))


def tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    return re.findall(r"[a-z0-9]+", "".join(c for c in text if not unicodedata.combining(c)))


class Grounder:
    """Checks that an extracted name is really in the document.

    ok           the name appears contiguously (ignoring spacing and punctuation)
    ocr_suspect  its words are all present but not adjacent: the name was reassembled
                 across a broken layout, or the model repaired an OCR error
    ungrounded   words of the name are missing from the document: possible hallucination
    """

    def __init__(self, text: str) -> None:
        self._squeezed = squeeze(text)
        self._tokens = set(tokens(text))

    def quality(self, name: str) -> str:
        key = squeeze(name)
        if key and key in self._squeezed:
            return "ok"
        words = [w for w in tokens(name) if len(w) > 1]
        if not words:
            return "ungrounded"
        found = sum(1 for w in words if w in self._tokens or w in self._squeezed)
        if found == len(words):
            return "ocr_suspect"
        return "ocr_suspect" if len(words) >= 3 and found / len(words) >= 0.75 else "ungrounded"


def is_placeholder(name: str) -> bool:
    return bool(_PLACEHOLDER_RE.search(name))


def normalize_state(value: str | None) -> str | None:
    """Full US state name, or None for anything else (foreign jurisdictions included)."""
    if not value:
        return None
    text = re.sub(r"^(?:the\s+)?(?:state|commonwealth)\s+of\s+", "", value.strip().lower()).strip(" .")
    if text in _US_STATES:
        return text.title().replace(" Of ", " of ")
    abbreviation = text.replace(".", "").upper()
    if abbreviation in _STATE_BY_ABBREVIATION:
        return _STATE_BY_ABBREVIATION[abbreviation].title().replace(" Of ", " of ")
    return None


_LEGAL_NOUN = r"(?:corporation|company|partnership|llc|l\.l\.c|entity|association|trust|bank|municipal\w*|non-?profit)"
_LEGAL_ADJECTIVE = r"(?:limited|liability|professional|municipal|public|domestic|business|stock|general|for-profit|chartered|state)"


def state_is_stated(text: str, state: str) -> bool:
    """True if the document states `state` as a place of incorporation or organization.

    The model also infers a state from a street address ("175 Berkeley Street, Boston,
    Massachusetts"). An address is not a registration, so the claim must be backed by wording
    such as "a Delaware corporation" or "organized under the laws of Delaware". Short-form
    complaints carry a "State of Incorporation" table instead of sentences and are accepted.
    """
    flat = re.sub(r"\s+", " ", text.lower())
    if re.search(r"state of (?:principal place )?incorporation", flat):
        return True
    name = re.escape(state.lower())
    return bool(
        re.search(
            rf"\b{name},? (?:{_LEGAL_ADJECTIVE} )*{_LEGAL_NOUN}"
            rf"|laws of (?:the )?(?:state of |commonwealth of )?{name}\b"
            rf"|(?:incorporated|organized|formed|chartered|registered|domiciled)(?: and existing)? (?:in|under)\b[^.]{{0,40}}\b{name}\b"
            rf"|(?:corporation|company|entity) of (?:the )?state of {name}\b"
            rf"|\bstate of {name}\b[^.]{{0,30}}(?:corporation|company)",
            flat,
        )
    )


def tidy(name: str) -> str:
    """Whitespace only: `name_raw` stays as the model read it."""
    return re.sub(r"\s+", " ", name).strip()


def build_prediction(
    doc_id: str,
    primary_text: str,
    supplemental_text: str,
    extraction: Extraction,
    policy: Policy = Policy(),
    input_repairs: list[str] | None = None,
) -> Prediction:
    text = primary_text + "\n" + supplemental_text
    grounder = Grounder(text)
    caption_grounder = Grounder(primary_text)
    before_menu = text_before_unmarked_menu(text) if policy.drop_placeholders else None
    menu_grounder = Grounder(before_menu) if before_menu is not None else None

    defendants: list[Defendant] = []
    excluded: list[ExcludedParty] = []
    seen: set[tuple[str, str | None]] = set()
    seen_excluded: set[str] = set()

    def exclude(name_raw: str, is_organization: bool, reason: str) -> None:
        if squeeze(name_raw) not in seen_excluded:
            seen_excluded.add(squeeze(name_raw))
            excluded.append(ExcludedParty(name_raw=name_raw, is_organization=is_organization, reason=reason))

    for party in extraction.defendants:
        name_raw = tidy(party.name_raw)
        if not name_raw:
            continue
        aliases = [tidy(a) for a in party.aliases if tidy(a)]
        placeholder = party.is_placeholder or is_placeholder(name_raw)
        # A fictitious name with a real trade name ("ABC Corp." d/b/a Joe's Diner) is an
        # identifiable business and stays; an anonymous "Does 1-10" does not.
        identifiable = placeholder and party.is_organization and bool(aliases)
        if placeholder and policy.drop_placeholders and not identifiable:
            exclude(name_raw, party.is_organization, "placeholder")
            continue
        if not party.is_organization and policy.organizations_only:
            exclude(name_raw, False, "individual")
            continue

        parsed = parse_name(name_raw)
        state = normalize_state(party.state_of_registration)
        if menu_grounder is not None and menu_grounder.quality(parsed.name_normalized) == "ungrounded":
            exclude(name_raw, party.is_organization, "unselected_option")
            continue
        identity = (parsed.key, parsed.designator)
        if identity in seen:  # same party listed twice (caption and body)
            continue
        seen.add(identity)
        in_caption = caption_grounder.quality(parsed.name_normalized) != "ungrounded"
        defendants.append(
            Defendant(
                name_raw=name_raw,
                name_normalized=parsed.name_normalized,
                designator=parsed.designator,
                is_organization=party.is_organization,
                us_state_of_registration=state if state and state_is_stated(text, state) else None,
                name_quality="placeholder" if placeholder else grounder.quality(parsed.name_normalized),
                # computed, not asked: the model's own caption/body answer was wrong in 9 of 15 dev documents
                source="caption" if in_caption else "body",
                aliases=aliases,
            )
        )
    return Prediction(doc_id=doc_id, defendants=defendants, excluded_parties=excluded, input_repairs=input_repairs or [])
