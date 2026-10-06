import pytest

from defx.llm import backoff_seconds, cost_usd
from defx.postprocess import Grounder, Policy, build_prediction, is_placeholder, normalize_state, state_is_stated, text_before_unmarked_menu
from defx.preprocess import prepare_text, repair_font_shift
from defx.schema import Extraction, Prediction


def party(name, org=True, placeholder=False, source="caption", state=None, aliases=()):
    return {"evidence": "", "name_raw": name, "aliases": list(aliases), "is_organization": org,
            "is_placeholder": placeholder, "source": source, "state_of_registration": state}


def predict(parties, text="", policy=Policy(), supplemental=""):
    return build_prediction("d1", text, supplemental, Extraction.model_validate({"defendants": parties}), policy)


def test_canonical_fields_follow_the_brief_example():
    p = predict([party("Wal-Mart Stores, Inc.", state="Delaware")], "WAL-MART STORES, INC., a Delaware corporation")
    d = p.defendants[0]
    assert (d.name_raw, d.name_normalized, d.designator, d.is_organization) == (
        "Wal-Mart Stores, Inc.", "wal-mart stores", "incorporated", True)
    assert d.us_state_of_registration == "Delaware" and d.name_quality == "ok"


def test_individuals_and_placeholders_are_excluded_not_lost():
    p = predict([party("Acme LLC"), party("John Smith", org=False), party("DOES 1-50", placeholder=True)])
    assert [d.name_normalized for d in p.defendants] == ["acme"]
    assert [(e.name_raw, e.reason) for e in p.excluded_parties] == [("John Smith", "individual"), ("DOES 1-50", "placeholder")]


def test_placeholder_is_caught_even_if_the_model_misses_the_flag():
    p = predict([party("Does 1 through 10, inclusive", org=True, placeholder=False)])
    assert not p.defendants and p.excluded_parties[0].reason == "placeholder"


def test_raw_policy_keeps_everything():
    p = predict([party("Acme LLC"), party("John Smith", org=False), party("DOES 1-50", placeholder=True)],
                policy=Policy.named("raw"))
    assert len(p.defendants) == 3 and not p.excluded_parties


def test_same_party_in_caption_and_body_is_emitted_once():
    p = predict([party("TIKTOK INC."), party("TikTok Inc.", source="body")])
    assert len(p.defendants) == 1


def test_same_name_with_different_legal_form_is_two_parties():
    p = predict([party("APOTEX CORP."), party("APOTEX INC."), party("TikTok Ltd."), party("TikTok Pte. Ltd.")])
    assert [(d.name_normalized, d.designator) for d in p.defendants] == [
        ("apotex", "corporation"), ("apotex", "incorporated"), ("tiktok", "limited"), ("tiktok", "private limited")]


def test_alias_is_kept_out_of_the_canonical_name():
    p = predict([party("PurFoods, LLC d/b/a Mom's Meals", aliases=["Mom's Meals"])])
    assert p.defendants[0].name_normalized == "purfoods" and p.defendants[0].aliases == ["Mom's Meals"]


def test_empty_names_are_dropped_and_whitespace_is_tidied():
    p = predict([party("  "), party("HAWAIIAN ELECTRIC\n  COMPANY, INC.")])
    assert [d.name_raw for d in p.defendants] == ["HAWAIIAN ELECTRIC COMPANY, INC."]


@pytest.mark.parametrize("name,expected", [
    ("DOES 1-10", True), ("Does 1 through 50, inclusive", True), ("John Doe", True), ("JANE ROE 1", True),
    ("John Doe Corporation", True), ("ABC Corporation", True), ("Unknown Named Agents", True),
    ("Doe Run Resources Corporation", False), ("Dow Chemical Company", False), ("Roe & Associates LLC", False),
    ("ABC Supply Co., Inc.", False), ("Acme Corp.", False),
])
def test_placeholder_detection(name, expected):
    assert is_placeholder(name) is expected


def test_grounding_levels():
    g = Grounder("T HE GEO G ROUP , Inc., a Florida\ncorporation; SYSINFORMATION HEALTHCARE\n\nCase No.: 1\n\nSERVICES, LLC")
    assert g.quality("the geo group") == "ok"  # OCR letter-spacing is ignored
    assert g.quality("sysinformation healthcare services") == "ocr_suspect"  # split by a column break
    assert g.quality("globex holdings") == "ungrounded"


@pytest.mark.parametrize("raw,expected", [
    ("Delaware", "Delaware"), ("the State of New York", "New York"), ("CA", "California"), ("N.Y.", "New York"),
    ("District of Columbia", "District of Columbia"), ("Cayman Islands", None), ("Singapore", None), (None, None), ("", None),
])
def test_state_normalization(raw, expected):
    assert normalize_state(raw) == expected


def test_prediction_round_trips_through_the_schema():
    p = predict([party("Acme LLC")])
    assert Prediction.model_validate_json(p.model_dump_json()) == p


def test_cost_estimate_handles_dated_and_prefixed_model_names():
    assert cost_usd("gpt-4o-mini-2024-07-18", 1_000_000, 0) == pytest.approx(0.15)
    assert cost_usd("openai/gpt-4o-mini", 0, 1_000_000) == pytest.approx(0.60)
    assert cost_usd("gpt-4.1", 1_000_000, 1_000_000) == pytest.approx(10.0)
    assert cost_usd("some-unknown-model", 10, 10) == 0.0


def test_backoff_is_bounded():
    assert all(0 <= backoff_seconds(a) <= 30 for a in range(12))


def test_source_is_computed_from_where_the_name_appears():
    p = predict([party("Acme LLC", source="body"), party("Beta Advanced Insurance Company", source="caption")],
                text="JOHN ROE v. ACME LLC, ET AL., Defendants.",
                supplemental="Defendant Beta Advanced Insurance Company has its headquarters in Ohio.")
    assert [d.source for d in p.defendants] == ["caption", "body"]  # the model's answer is ignored


def test_fictitious_entity_with_a_trade_name_is_kept_and_flagged():
    p = predict([party("ABC CORPORATION", placeholder=True, aliases=["NDN LANDSCAPING"]),
                 party("DOES 1-20", placeholder=True)])
    assert [(d.name_normalized, d.name_quality) for d in p.defendants] == [("abc", "placeholder")]
    assert [e.reason for e in p.excluded_parties] == ["placeholder"]


MENU = """IN RE: WIDGET PRODUCTS LIABILITY LITIGATION
Jane Roe v. Luster Products, Inc.
4. Plaintiff is suing the following Defendants:
Check All
Applicable
Defendants
{a} Avlon Industries
{b} Luster Products, Inc.
"""


def test_unmarked_menu_keeps_only_names_from_the_caption():
    text = MENU.format(a="□", b="□")
    assert text_before_unmarked_menu(text) is not None
    p = predict([party("Avlon Industries"), party("Luster Products, Inc.")], text=text)
    assert [d.name_normalized for d in p.defendants] == ["luster products"]
    assert [(e.name_raw, e.reason) for e in p.excluded_parties] == [("Avlon Industries", "unselected_option")]


def test_menu_with_visible_marks_is_left_to_the_model():
    text = MENU.format(a="☑", b="□")
    assert text_before_unmarked_menu(text) is None
    assert len(predict([party("Avlon Industries")], text=text).defendants) == 1


def test_ordinary_complaints_have_no_menu():
    assert text_before_unmarked_menu("ACME v. BETA INC. (please check the appropriate box)") is None


def shift(text: str) -> str:
    return "".join(chr(ord(c) - 29) for c in text)


def test_font_shift_is_decoded_and_normal_text_on_the_same_page_is_kept():
    page = "\n".join([
        shift("IN THE UNITED STATES DISTRICT COURT FOR"),
        shift("THE PPG PLAN,") + " " + shift("and") + " " + shift("PPG INDUSTRIES, INC."),
        "ELECTRONICALLY FILED 5:18-CV-114 (Bailey)",
        shift("COMPLAINT"),
    ] * 3)
    text, repairs = prepare_text(page)
    assert repairs == ["font_shift"]
    assert "IN THE UNITED STATES DISTRICT COURT FOR" in text
    assert "THE PPG PLAN, and PPG INDUSTRIES, INC." in text
    assert "ELECTRONICALLY FILED 5:18-CV-114 (Bailey)" in text and "COMPLAINT" in text


def test_ordinary_text_is_not_touched_by_the_font_repair():
    page = "ACME, INC. v. BETA & CO. (No. 1:23-cv-00818) $1,000 12/04/18"
    assert repair_font_shift(page) == (page, 0)
    assert prepare_text(page) == (page, [])


@pytest.mark.parametrize("text,state,expected", [
    ("Defendant Acme, Inc. is a Delaware corporation with its principal place of business in Texas.", "Delaware", True),
    ("Acme is a limited liability company organized under the laws of the State of Wyoming.", "Wyoming", True),
    ("Acme is a New Jersey limited\nliability company located at 415 Jack Martin Blvd.", "New Jersey", True),
    ("Venture 475 is a corporation of the State of New York, which is not licensed in Illinois.", "New York", True),
    ("Acme is incorporated in Delaware.", "Delaware", True),
    ("LIBERTY MUTUAL INSURANCE COMPANY, 175 Berkeley Street, Boston, Massachusetts 02116", "Massachusetts", False),
    ("Acme, Inc. is a Delaware corporation with its principal place of business in Texas.", "Texas", False),
    ("State of Incorporation or Formation ... Luster Products, Inc. Illinois Illinois", "Illinois", True),
])
def test_state_must_be_stated_as_a_registration(text, state, expected):
    assert state_is_stated(text, state) is expected


def test_state_inferred_from_an_address_is_dropped():
    p = predict([party("Liberty Mutual Insurance Company", state="Massachusetts")],
                text="LIBERTY MUTUAL INSURANCE COMPANY 175 Berkeley Street, Boston, Massachusetts 02116")
    assert p.defendants[0].us_state_of_registration is None


def test_output_token_ceiling_is_read_from_the_api_error():
    from defx.llm import _TOKEN_LIMIT_RE
    message = "max_tokens is too large: 32000. This model supports at most 16384 completion tokens, whereas you provided 32000."
    assert int(_TOKEN_LIMIT_RE.search(message).group(1)) == 16384
