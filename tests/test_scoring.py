import pytest

from defx.normalize import parse_name
from defx.scoring import aggregate, bootstrap_ci, gold_item, match_document, pred_item, prf, score, similarity


def pred(raw: str) -> dict:
    parsed = parse_name(raw)
    return {"name_raw": raw, "name_normalized": parsed.name_normalized, "designator": parsed.designator,
            "is_organization": True}


def run(labels, raws):
    return match_document("d", [gold_item(x) for x in labels], [pred_item(pred(r)) for r in raws])


def test_exact_match_all_tiers():
    r = run(["acme, limited liability company"], ["ACME, LLC"])
    assert r.tp == {"strict": 1, "core": 1, "fuzzy": 1}


def test_designator_disagreement_is_core_but_not_strict():
    r = run(["acme, limited liability company"], ["Acme, Inc."])
    assert r.tp == {"strict": 0, "core": 1, "fuzzy": 1}


def test_corporation_counts_as_incorporated_in_strict_tier():
    # reviewers wrote "amerex, incorporated" for "AMEREX CORPORATION"
    assert run(["amerex, incorporated"], ["AMEREX CORPORATION"]).tp["strict"] == 1


def test_duplicate_prediction_is_a_false_positive():
    r = run(["acme, incorporated"], ["Acme, Inc.", "ACME INC"])
    assert r.counts("core") == (1, 1, 0)


def test_two_labels_with_one_key_need_two_predictions():
    labels = ["apotex", "apotex, incorporated"]
    assert run(labels, ["APOTEX CORP.", "APOTEX INC."]).counts("core") == (2, 0, 0)
    assert run(labels, ["APOTEX INC."]).counts("core") == (1, 0, 1)


def test_strict_pairs_are_preferred_before_core_pairs():
    r = run(["apotex", "apotex, incorporated"], ["APOTEX INC.", "Apotex Company"])
    assert r.tp["strict"] == 2


def test_label_typo_is_matched_by_fuzzy_only():
    r = run(["bytendance, incorporated"], ["BYTEDANCE, INC."])
    assert r.tp == {"strict": 0, "core": 0, "fuzzy": 1}
    assert r.matches[0][2] == "fuzzy"


def test_fuzzy_does_not_merge_different_entities():
    r = run(["wockhardt usa, incorporated"], ["Wockhardt Ltd."])
    assert r.counts("fuzzy") == (0, 1, 1)
    assert similarity("fund1", "fund2") == 0.0


def test_fuzzy_assignment_takes_best_pair_first():
    r = run(["perrigo", "l. perrigo"], ["L. Perrigo Co", "Perrigo Company"])
    assert r.tp["core"] == 2 and not r.false_positives and not r.false_negatives


def test_empty_document_conventions():
    assert prf(0, 0, 0) == (1.0, 1.0, 1.0)  # nothing labeled, nothing predicted
    assert prf(0, 2, 0) == (0.0, 1.0, 0.0)  # predictions on a document with no label
    assert prf(0, 0, 3) == (1.0, 0.0, 0.0)  # nothing predicted


def test_micro_and_macro_differ_on_skewed_documents():
    big = match_document("big", [gold_item(f"co{i}") for i in range(10)], [pred_item(pred(f"co{i}")) for i in range(10)])
    small = match_document("small", [gold_item("acme")], [])
    agg = aggregate([big, small], "core")
    assert agg["micro"]["recall"] == pytest.approx(10 / 11)
    assert agg["macro"]["recall"] == pytest.approx(0.5)


def test_score_treats_missing_documents_as_empty_and_hand_computed_values():
    gold = {"a": ["acme, incorporated", "beta, limited"], "b": ["gamma"], "c": []}
    predictions = {"a": [pred("Acme Inc."), pred("Delta LLC")], "c": []}
    results = score(gold, predictions)
    agg = aggregate(results, "core")
    assert (agg["tp"], agg["fp"], agg["fn"]) == (1, 1, 2)
    assert agg["micro"]["precision"] == pytest.approx(0.5)
    assert agg["micro"]["recall"] == pytest.approx(1 / 3)
    assert agg["micro"]["f1"] == pytest.approx(0.4)
    # per-document F1: a = 0.5, b = 0.0, c = 1.0
    assert agg["macro"]["f1"] == pytest.approx(0.5)


def test_bootstrap_interval_is_deterministic_and_ordered():
    gold = {"a": ["acme"], "b": ["beta"], "c": ["gamma"]}
    results = score(gold, {"a": [pred("Acme")], "b": [pred("Other")]})
    lo, hi = bootstrap_ci(results, "core", n_resamples=200)
    assert 0.0 <= lo <= hi <= 1.0
    assert (lo, hi) == bootstrap_ci(results, "core", n_resamples=200)


def test_prediction_without_canonical_fields_falls_back_to_name_raw():
    assert pred_item({"name_raw": "ACME, LLC"}).key == "acme"
