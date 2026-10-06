"""Normalizer cases taken from real captions and the labels reviewers wrote for them."""

import pytest

from defx.normalize import designator_class, match_key, parse_name

# (name as it appears in the complaint, reviewer label for it)
CAPTION_TO_LABEL = [
    ("3M COMPANY (f/k/a Minnesota Mining and Manufacturing Company)", "3m"),
    ("AMEREX CORPORATION", "amerex, incorporated"),
    ("ARCHROMA U.S. INC.", "archroma u.s., incorporated"),
    ("BUCKEYE FIRE EQUIPMENT COMPANY", "buckeye fire equipment"),
    ("E.I. DU PONT DE NEMOURS AND ) COMPANY", "e.i. du pont de nemours and"),
    ("DU PONT DE NEMOURS INC. (f/k/a DOWDUPONT INC.)", "du pont de nemours, incorporated"),
    ("KIDDE PLC", "kidde"),
    ("THE CHEMOURS COMPANY", "the chemours"),
    ("TYCO  FIRE PRODUCTS LP, as successor-in-interest to The Ansul Company", "tyco fire products, limited partnership"),
    ("BOEHRINGER INGELHEIM INTERNATIONAL GMBH", "boehringer ingelheim international"),
    ("BOEHRINGER INGELHEIM PROMECO, S.A. DE C.V.", "boehringer ingelheim promeco"),
    ("GLAXOSMITHKLINE HOLDINGS (AMERICAS) INC.", "glaxosmithkline holdings (americas), incorporated"),
    ("SANOFI SA", "sanofi"),
    ("SANOFI-AVENTIS U.S. LLC", "sanofi-aventis u.s., limited liability company"),
    ("L. PERRIGO CO", "l. perrigo"),
    ("PERRIGO RESEARCH & DEVELOPMENT COMPANY", "perrigo research & development"),
    ("DR. REDDY'S LABORATORIES, LTD.", "dr. reddy's laboratories, limited"),
    ("B/T WASHINGTON, LLC d/b/a BLANTON TURNER", "b/t washington, limited liability company"),
    ("CAMDEN PROPERTY TRUST", "camden property"),
    ("LINCOLN PROPERTY CO.", "lincoln property"),
    ("ESSEX PROPERTY TRUST, INC.", "essex property trust, incorporated"),
    ("SYSINFORMATION HEALTHCARE SERVICES, LLC d/b/a EQUALIZERCM AND 1ST CREDENTIALING",
     "sysinformation healthcare services, limited liability company"),
    ("PurFoods, LLC d/b/a Mom’s Meals (“PurFoods” or “Defendant”)", "purfoods, limited liability company"),
    ("TIKTOK U.S. DATA SECURITY, INC.", "tiktok us data security, incorporated"),
    ("Juvia’s Place LLC", "juvia’s place, limited liability company"),
    ("The GEO Group, Inc., a Florida corporation", "the geo group, incorporated"),
    ("the CITY OF ADELANTO, a municipal entity", "the city of adelanto"),
    ("CITY OF CINCINNATI", "city of cincinnati"),
    ("PROGRESSIVE CASUALTY INSURANCE COMPANY, ET AL.", "progressive casualty insurance company"),
    ("REMARKABLE HEALTHCARE OF FORT WORTH, LP", "remarkable healthcare of fort worth, limited partnership"),
    ("1ST SOURCE BANK", "1st source bank"),
]


@pytest.mark.parametrize("raw,label", CAPTION_TO_LABEL)
def test_caption_and_label_share_a_key(raw, label):
    assert parse_name(raw).key == parse_name(label).key


@pytest.mark.parametrize(
    "raw,name,designator",
    [
        ("Wal-Mart Stores, Inc.", "wal-mart stores", "incorporated"),  # the example in the brief
        ("Walmart Real Estate Business Trust", "walmart real estate business trust", None),
        ("ACME, LLC", "acme", "limited liability company"),
        ("Acme L.L.C.", "acme", "limited liability company"),
        ("Acme Corp.", "acme", "corporation"),
        ("Acme Holdings LP", "acme holdings", "limited partnership"),
        ("Acme L.P.", "acme", "limited partnership"),
        ("Smith & Jones LLP", "smith & jones", "limited liability partnership"),
        ("TikTok Pte. Ltd.", "tiktok", "private limited"),
        ("Wells Fargo Bank, N.A.", "wells fargo bank", "national association"),
        ("GOLDMAN SACHS & CO. LLC", "goldman sachs & co", "limited liability company"),
        ("UTC Fire & Security Americas Corporation, Inc. (f/k/a GE Interlogix, Inc.)",
         "utc fire & security americas corporation", "incorporated"),
        ("BRANDSNESS, BRANDSNESS & RUDD, P.C., an Oregon Professional Corporation",
         "brandsness, brandsness & rudd", "professional corporation"),
        ("“ABC CORPORATION” d/b/a NDN LANDSCAPING", "abc", "corporation"),
        ("United States of America", "united states of america", None),
        ("TOYOTA MOTOR SALES, U.S.A., INC.", "toyota motor sales, u.s.a.", "incorporated"),
    ],
)
def test_canonical_fields(raw, name, designator):
    parsed = parse_name(raw)
    assert (parsed.name_normalized, parsed.designator) == (name, designator)


def test_only_one_designator_layer_is_removed():
    # "The Chemours Company" and "Chemours Company FC, LLC" are different defendants in c1ddb6
    assert parse_name("The Chemours Company").key != parse_name("Chemours Company FC, LLC").key


def test_distinct_entities_keep_distinct_keys():
    keys = {parse_name(n).key for n in [
        "Remarkable Healthcare, LLC", "Remarkable Healthcare of Carrollton, LP",
        "Remarkable Healthcare of Fort Worth, LP", "Wockhardt USA LLC", "Wockhardt Ltd.",
    ]}
    assert len(keys) == 5


def test_a_bare_designator_word_is_not_emptied():
    assert parse_name("Limited").name_normalized == "limited"


def test_usa_is_not_read_as_a_designator():
    assert parse_name("Teva Pharmaceuticals USA").designator is None


@pytest.mark.parametrize("name", ["wal-mart stores", "the geo group", "l'oréal usa", "dr. reddy's laboratories"])
def test_match_key_is_idempotent_on_normalized_names(name):
    assert match_key(name) == parse_name(name).key


def test_key_ignores_accents_spacing_and_apostrophes():
    assert match_key("L’ORÉAL USA") == match_key("l'oreal usa")
    assert match_key("mid- america apartment communities") == match_key("Mid-America Apartment Communities")


def test_label_designator_vocabulary():
    assert designator_class("corporation") == "incorporated"
    assert designator_class("company") is None
    assert designator_class("public limited company") is None
    assert designator_class(None) is None


def test_spaced_initials_keep_their_last_period():
    assert parse_name("K. G. P. INC.").name_normalized == "k. g. p."
