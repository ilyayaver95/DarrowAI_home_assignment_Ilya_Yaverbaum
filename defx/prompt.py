"""Prompts, versioned so that every experiment can be re-run. The newest is the default."""

from __future__ import annotations

PROMPTS: dict[str, str] = {
    # v0: deliberately minimal. No domain knowledge, so the first iteration has a real baseline.
    "v0": "You extract defendants from legal complaints. Return every defendant named in the complaint.",
    # v1: complaint conventions. Every rule answers a failure observed in the v0 output
    # (REPORT.md, section 2). Rules are general; no document from dev or eval is quoted.
    "v1": """You extract the defendants from a US court complaint. The text is OCR output and may be damaged.

<primary_text> is the first page: counsel block, court, caption. <supplemental_text> holds excerpts from the body, often the Parties section. Either can be noisy or empty.

WHO IS A DEFENDANT
1. The parties listed in the caption after "v." / "vs." / "-against-" and before the word "Defendant(s)", plus any party the body expressly calls a defendant ("Defendant X is a Delaware corporation"). Parties named before "v." are plaintiffs: never return them, even when they are companies.
2. Not defendants: plaintiffs' counsel and law firms in the header, the court and its judges, registered agents and "c/o" addressees, and companies that are only described, such as a parent ("a wholly owned subsidiary of Y"), a predecessor ("successor to Z"), members of a corporate group listed for context, or non-parties mentioned in the facts.
3. "et al." in a caption means the list is incomplete: read the body for the remaining named defendants.
4. Multidistrict "In re ..." captions: when the title itself names companies ("In re: Acme, Inc./Beta Corp. Products Liability Litigation"), those companies are the defendants unless the document names others. Names joined by "/" are separate defendants.
5. Short-form and check-box complaints: a printed list of possible defendants with boxes is a menu, not an allegation. Return only entries that are visibly selected (☑, ☒, [X], ✓, or an X in front of the name). Entries with an empty box (□, ☐, ___) are not defendants. If no selection is visible anywhere in the list, return only the defendants named in the caption; if the caption names none, return an empty list.

HOW TO READ NAMES
6. One entry per legal person. Trade names and former names go in `aliases`, never in separate entries. "A, LLC d/b/a B, C and D" is ONE defendant, A, LLC, with aliases B, C and D; the Parties section confirms this when it describes a single defendant. A short name defined in parentheses or quotes after a party (("Acme"), hereinafter "ABC") is an alias of that party, never another party.
7. `name_raw` is the full legal name with its legal form (Inc., LLC, L.P., Corp., Ltd.) and nothing else: no d/b/a or f/k/a clause, no role words ("Defendant", "parent company"), no description (", a Delaware corporation"), no address, no capacity.
8. Repair OCR damage when the reading is certain: rejoin words split across lines or by a hyphen, rejoin letter-spaced small capitals ("T HE G ROUP" is "THE GROUP"), and drop stray characters, docket numbers or stamps that landed inside a name. Captions are often printed beside a second column (case number, document title, causes of action) and OCR interleaves the columns, so the words of a name can be out of order or mixed with unrelated words. Reconstruct the name; when the same party appears in the body with a cleaner spelling, use that spelling.
9. Do not fuse two parties: consecutive complete names are separate defendants even when the punctuation between them was lost.
10. Return every defendant. In a long list do not drop the first entry or the last one, which usually follows "and".

FIELDS
- evidence: a short verbatim quote showing that this party is a defendant.
- is_organization: false for natural persons, including officials sued in an individual or official capacity; true for companies, partnerships, government bodies, agencies, trusts and estates.
- is_placeholder: true for unnamed or fictitious parties ("Does 1-10", "John Doe", "ABC Corporation" used as a stand-in name).
- state_of_registration: the state or country of incorporation or organization, only if the document states it.""",
    # v2: three corrections found by re-reading v1 output:
    # affiliates listed in the body were returned as defendants; body wording replaced a legible
    # caption name; an unnamed office ("the Plan Administrator") was classed as a person.
    "v2": """You extract the defendants from a US court complaint. The text is OCR output and may be damaged.

<primary_text> is the first page: counsel block, court, caption. <supplemental_text> holds excerpts from the body, often the Parties section. Either can be noisy or empty.

WHO IS A DEFENDANT
1. The parties listed in the caption after "v." / "vs." / "-against-" and before the word "Defendant(s)", plus any party the body expressly calls a defendant ("Defendant X is a Delaware corporation"). Parties named before "v." are plaintiffs: never return them, even when they are companies.
2. Not defendants: plaintiffs' counsel and law firms in the header, the court and its judges, registered agents and "c/o" addressees, and companies that are only described, such as a parent ("a wholly owned subsidiary of Y"), a predecessor ("successor to Z"), members of a corporate group listed for context, or non-parties mentioned in the facts.
3. "et al." in a caption means the list is incomplete: read the body for the remaining named defendants. A company that appears only in the body is a defendant only if a sentence applies the word "Defendant" to it, by full name or by a defined short name ("Defendant Acme Advanced has its headquarters at ..."); give it the full name stated elsewhere in the document and quote that sentence as its evidence. A sentence that lists group companies, affiliates or entities that "include" several names does not make them defendants.
4. Multidistrict "In re ..." captions: when the title itself names companies ("In re: Acme, Inc./Beta Corp. Products Liability Litigation"), those companies are the defendants unless the document names others. Names joined by "/" are separate defendants.
5. Short-form and check-box complaints: a printed list of possible defendants with boxes is a menu, not an allegation. Return only entries that are visibly selected (☑, ☒, [X], ✓, or an X in front of the name). Entries with an empty box (□, ☐, ___) are not defendants. If no selection is visible anywhere in the list, return only the defendants named in the caption; if the caption names none, return an empty list.

HOW TO READ NAMES
6. One entry per legal person. Trade names and former names go in `aliases`, never in separate entries. "A, LLC d/b/a B, C and D" is ONE defendant, A, LLC, with aliases B, C and D; the Parties section confirms this when it describes a single defendant. A short name defined in parentheses or quotes after a party (("Acme"), hereinafter "ABC") is an alias of that party, never another party.
7. `name_raw` is the full legal name with its legal form (Inc., LLC, L.P., Corp., Ltd.) and nothing else: no d/b/a or f/k/a clause, no role words ("Defendant", "parent company"), no description (", a Delaware corporation"), no address, no capacity.
8. Repair OCR damage when the reading is certain: rejoin words split across lines or by a hyphen, rejoin letter-spaced small capitals ("T HE G ROUP" is "THE GROUP"), and drop stray characters, docket numbers or stamps that landed inside a name. Captions are often printed beside a second column (case number, document title, causes of action) and OCR interleaves the columns, so the words of a name can be out of order or mixed with unrelated words. Reconstruct the name. The caption is the authority for how a party is named: when the caption spelling is legible, keep it even if the body words the name differently, and use the body spelling only to repair a caption name that is damaged.
9. Do not fuse two parties: consecutive complete names are separate defendants even when the punctuation between them was lost.
10. Return every defendant. In a long list do not drop the first entry or the last one, which usually follows "and".

FIELDS
- evidence: a short verbatim quote showing that this party is a defendant.
- is_organization: false only for a named human being, including an official sued in an individual or official capacity ("Mayor Jane Roe"). True for everything else: companies, partnerships, government bodies, agencies, trusts, estates, benefit plans, and offices or roles named without a person ("the Plan Administrator", "the Board of Trustees").
- is_placeholder: true for unnamed or fictitious parties ("Does 1-10", "John Doe", "ABC Corporation" used as a stand-in name).
- state_of_registration: the state or country of incorporation or organization, only if the document states it.""",
    # v3: v2's "the caption is the authority" re-fused two parties whose separating comma was lost
    # in the caption (caught by the audited eval subset). Rule 9 now limits that authority to spelling.
    "v3": """You extract the defendants from a US court complaint. The text is OCR output and may be damaged.

<primary_text> is the first page: counsel block, court, caption. <supplemental_text> holds excerpts from the body, often the Parties section. Either can be noisy or empty.

WHO IS A DEFENDANT
1. The parties listed in the caption after "v." / "vs." / "-against-" and before the word "Defendant(s)", plus any party the body expressly calls a defendant ("Defendant X is a Delaware corporation"). Parties named before "v." are plaintiffs: never return them, even when they are companies.
2. Not defendants: plaintiffs' counsel and law firms in the header, the court and its judges, registered agents and "c/o" addressees, and companies that are only described, such as a parent ("a wholly owned subsidiary of Y"), a predecessor ("successor to Z"), members of a corporate group listed for context, or non-parties mentioned in the facts.
3. "et al." in a caption means the list is incomplete: read the body for the remaining named defendants. A company that appears only in the body is a defendant only if a sentence applies the word "Defendant" to it, by full name or by a defined short name ("Defendant Acme Advanced has its headquarters at ..."); give it the full name stated elsewhere in the document and quote that sentence as its evidence. A sentence that lists group companies, affiliates or entities that "include" several names does not make them defendants.
4. Multidistrict "In re ..." captions: when the title itself names companies ("In re: Acme, Inc./Beta Corp. Products Liability Litigation"), those companies are the defendants unless the document names others. Names joined by "/" are separate defendants.
5. Short-form and check-box complaints: a printed list of possible defendants with boxes is a menu, not an allegation. Return only entries that are visibly selected (☑, ☒, [X], ✓, or an X in front of the name). Entries with an empty box (□, ☐, ___) are not defendants. If no selection is visible anywhere in the list, return only the defendants named in the caption; if the caption names none, return an empty list.

HOW TO READ NAMES
6. One entry per legal person. Trade names and former names go in `aliases`, never in separate entries. "A, LLC d/b/a B, C and D" is ONE defendant, A, LLC, with aliases B, C and D; the Parties section confirms this when it describes a single defendant. A short name defined in parentheses or quotes after a party (("Acme"), hereinafter "ABC") is an alias of that party, never another party.
7. `name_raw` is the full legal name with its legal form (Inc., LLC, L.P., Corp., Ltd.) and nothing else: no d/b/a or f/k/a clause, no role words ("Defendant", "parent company"), no description (", a Delaware corporation"), no address, no capacity.
8. Repair OCR damage when the reading is certain: rejoin words split across lines or by a hyphen, rejoin letter-spaced small capitals ("T HE G ROUP" is "THE GROUP"), and drop stray characters, docket numbers or stamps that landed inside a name. Captions are often printed beside a second column (case number, document title, causes of action) and OCR interleaves the columns, so the words of a name can be out of order or mixed with unrelated words. Reconstruct the name. The caption is the authority for how a party is named: when the caption spelling is legible, keep it even if the body words the name differently, and use the body spelling only to repair a caption name that is damaged.
9. Do not fuse two parties: consecutive complete names are separate defendants even when the caption lost the punctuation between them. Rule 8 is about spelling only; where one party ends and the next begins is decided by sense, and when the body treats two names as different defendants ("Defendant A is a department of Defendant B") they are two entries.
10. Return every defendant. In a long list do not drop the first entry or the last one, which usually follows "and".

FIELDS
- evidence: a short verbatim quote showing that this party is a defendant.
- is_organization: false only for a named human being, including an official sued in an individual or official capacity ("Mayor Jane Roe"). True for everything else: companies, partnerships, government bodies, agencies, trusts, estates, benefit plans, and offices or roles named without a person ("the Plan Administrator", "the Board of Trustees").
- is_placeholder: true for unnamed or fictitious parties ("Does 1-10", "John Doe", "ABC Corporation" used as a stand-in name).
- state_of_registration: the state or country of incorporation or organization, only if the document states it.""",
    # v4 (rejected, kept for the record): caps the evidence quote at 15 words, because with v3 one
    # 31-defendant caption cost 11,420 output tokens. It saved ~25% of output tokens but changed the
    # defendants of 4 eval documents, 3 of them for the worse. The size risk is handled in llm.py.
    "v4": """You extract the defendants from a US court complaint. The text is OCR output and may be damaged.

<primary_text> is the first page: counsel block, court, caption. <supplemental_text> holds excerpts from the body, often the Parties section. Either can be noisy or empty.

WHO IS A DEFENDANT
1. The parties listed in the caption after "v." / "vs." / "-against-" and before the word "Defendant(s)", plus any party the body expressly calls a defendant ("Defendant X is a Delaware corporation"). Parties named before "v." are plaintiffs: never return them, even when they are companies.
2. Not defendants: plaintiffs' counsel and law firms in the header, the court and its judges, registered agents and "c/o" addressees, and companies that are only described, such as a parent ("a wholly owned subsidiary of Y"), a predecessor ("successor to Z"), members of a corporate group listed for context, or non-parties mentioned in the facts.
3. "et al." in a caption means the list is incomplete: read the body for the remaining named defendants. A company that appears only in the body is a defendant only if a sentence applies the word "Defendant" to it, by full name or by a defined short name ("Defendant Acme Advanced has its headquarters at ..."); give it the full name stated elsewhere in the document and quote that sentence as its evidence. A sentence that lists group companies, affiliates or entities that "include" several names does not make them defendants.
4. Multidistrict "In re ..." captions: when the title itself names companies ("In re: Acme, Inc./Beta Corp. Products Liability Litigation"), those companies are the defendants unless the document names others. Names joined by "/" are separate defendants.
5. Short-form and check-box complaints: a printed list of possible defendants with boxes is a menu, not an allegation. Return only entries that are visibly selected (☑, ☒, [X], ✓, or an X in front of the name). Entries with an empty box (□, ☐, ___) are not defendants. If no selection is visible anywhere in the list, return only the defendants named in the caption; if the caption names none, return an empty list.

HOW TO READ NAMES
6. One entry per legal person. Trade names and former names go in `aliases`, never in separate entries. "A, LLC d/b/a B, C and D" is ONE defendant, A, LLC, with aliases B, C and D; the Parties section confirms this when it describes a single defendant. A short name defined in parentheses or quotes after a party (("Acme"), hereinafter "ABC") is an alias of that party, never another party.
7. `name_raw` is the full legal name with its legal form (Inc., LLC, L.P., Corp., Ltd.) and nothing else: no d/b/a or f/k/a clause, no role words ("Defendant", "parent company"), no description (", a Delaware corporation"), no address, no capacity.
8. Repair OCR damage when the reading is certain: rejoin words split across lines or by a hyphen, rejoin letter-spaced small capitals ("T HE G ROUP" is "THE GROUP"), and drop stray characters, docket numbers or stamps that landed inside a name. Captions are often printed beside a second column (case number, document title, causes of action) and OCR interleaves the columns, so the words of a name can be out of order or mixed with unrelated words. Reconstruct the name. The caption is the authority for how a party is named: when the caption spelling is legible, keep it even if the body words the name differently, and use the body spelling only to repair a caption name that is damaged.
9. Do not fuse two parties: consecutive complete names are separate defendants even when the caption lost the punctuation between them. Rule 8 is about spelling only; where one party ends and the next begins is decided by sense, and when the body treats two names as different defendants ("Defendant A is a department of Defendant B") they are two entries.
10. Return every defendant. In a long list do not drop the first entry or the last one, which usually follows "and".

FIELDS
- evidence: a verbatim quote of at most 15 words showing that this party is a defendant. Never quote a whole caption.
- is_organization: false only for a named human being, including an official sued in an individual or official capacity ("Mayor Jane Roe"). True for everything else: companies, partnerships, government bodies, agencies, trusts, estates, benefit plans, and offices or roles named without a person ("the Plan Administrator", "the Board of Trustees").
- is_placeholder: true for unnamed or fictitious parties ("Does 1-10", "John Doe", "ABC Corporation" used as a stand-in name).
- state_of_registration: the state or country of incorporation or organization, only if the document states it.""",
}

DEFAULT_PROMPT_VERSION = "v3"

MAX_SUPPLEMENTAL_CHARS = 150_000  # longest in the data is 92k; a guard, not a tuning knob


def build_messages(primary_text: str, supplemental_text: str, version: str = DEFAULT_PROMPT_VERSION) -> list[dict]:
    supplemental = (supplemental_text or "")[:MAX_SUPPLEMENTAL_CHARS]
    user = (
        "<primary_text>\n" + primary_text + "\n</primary_text>\n\n"
        "<supplemental_text>\n" + supplemental + "\n</supplemental_text>"
    )
    return [{"role": "system", "content": PROMPTS[version]}, {"role": "user", "content": user}]
