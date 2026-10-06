"""Deterministic text repair applied before the model sees a document. Pure functions.

Font-shift repair. Some PDFs embed a font whose character codes are offset by 29, and
the OCR pipeline emits the raw codes: "IN THE UNITED STATES" arrives as
",1\\x037+(\\x0381,7('\\x0367$7(6" (space becomes \\x03, "I" becomes ","). Two of the 40
eval documents are affected. The offset is constant, so the text is recoverable
exactly; doing it here means a small model can read the page and extracted names can
be checked against the text.
"""

from __future__ import annotations

import re

_OFFSET = 29
_MIN_SHIFTED_SPACES = 10  # \x03 is a shifted space; real text has none
_EXTRA = {"³": '"', "´": '"', "¶": "'"}  # quote glyphs of the same font
_NUMBER_LIKE = re.compile(r"^[0-9/:.,#()\-]+$")
_ONLY_SHIFTED_CAPITALS = re.compile(r"^[$%&'()*+,\-./0-9:;<=]{2,}$")  # A-Z shifted down by 29


def _is_control(ch: str) -> bool:
    return 0x01 <= ord(ch) <= 0x1F and ch not in "\n\r\t"


def _decode(token: str) -> str:
    return "".join(_EXTRA.get(c) or (chr(ord(c) + _OFFSET) if 0x03 <= ord(c) <= 0x61 else c) for c in token)


def repair_font_shift(text: str) -> tuple[str, int]:
    """Decode offset-29 tokens. Returns (text, number of tokens decoded).

    A token is decoded when it contains shifted control characters (certain), or when
    it could be shifted (no character above 'a') and sits next to a certain token, or
    when it consists only of the symbols that shifted capitals map to. Tokens with
    ordinary lower-case letters and number-like tokens are left alone, so pages that
    mix a normal header or stamp with shifted body text are handled per token.
    """
    if text.count("\x03") < _MIN_SHIFTED_SPACES:
        return text, 0
    decoded_count = 0
    lines = []
    for line in text.split("\n"):
        tokens = line.split(" ")
        certain = [any(_is_control(c) for c in t) for t in tokens]
        out = []
        for i, token in enumerate(tokens):
            decode = certain[i]
            if not decode and token and not any(ord(c) > 0x61 for c in token) and not _NUMBER_LIKE.match(token):
                neighbour = (i > 0 and certain[i - 1]) or (i + 1 < len(tokens) and certain[i + 1])
                decode = neighbour or bool(_ONLY_SHIFTED_CAPITALS.match(token))
            if decode:
                decoded_count += 1
                token = _decode(token)
            out.append(token)
        lines.append(" ".join(out))
    return "\n".join(lines), decoded_count


def prepare_text(text: str) -> tuple[str, list[str]]:
    """All repairs for one text field. Returns (text, names of the repairs applied)."""
    repairs = []
    text, decoded = repair_font_shift(text or "")
    if decoded:
        repairs.append("font_shift")
    cleaned = "".join(c for c in text if not _is_control(c))  # leftover control characters carry no content
    return cleaned, repairs
