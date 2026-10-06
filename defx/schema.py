"""Schemas: what the model must return, and what run.py writes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---- model output (sent to the API as a strict JSON schema) -----------------------------


class ExtractedParty(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # evidence comes first so the model has to locate the party before describing it
    evidence: str = Field(description="Short verbatim quote from the document showing this party is a defendant.")
    name_raw: str = Field(description="The party's name as written in the document.")
    aliases: list[str] = Field(description="Other names given for the same party (d/b/a, f/k/a, a/k/a).")
    is_organization: bool = Field(description="True for companies, government bodies and other entities; false for natural persons.")
    is_placeholder: bool = Field(description="True for unnamed or fictitious parties such as 'Does 1-10'.")
    source: Literal["caption", "body"] = Field(description="Where the document identifies this party as a defendant.")
    state_of_registration: str | None = Field(description="State of incorporation or organization if the document states it, else null.")


class Extraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    defendants: list[ExtractedParty]


# ---- predictions file --------------------------------------------------------------------

NameQuality = Literal["ok", "ocr_suspect", "ungrounded", "placeholder"]


class Defendant(BaseModel):
    name_raw: str
    name_normalized: str
    designator: str | None
    is_organization: bool
    us_state_of_registration: str | None = None
    name_quality: NameQuality = "ok"
    source: Literal["caption", "body"] = "caption"
    aliases: list[str] = []


class ExcludedParty(BaseModel):
    """A party the complaint sues but the reviewers do not record (see REPORT.md)."""

    name_raw: str
    is_organization: bool
    reason: Literal["individual", "placeholder", "unselected_option"]


class Prediction(BaseModel):
    doc_id: str
    defendants: list[Defendant]
    excluded_parties: list[ExcludedParty] = []
    input_repairs: list[str] = []  # deterministic repairs applied to the text, e.g. "font_shift"
    error: str | None = None
