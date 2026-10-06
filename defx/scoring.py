"""Scoring: match predicted defendants to labels and compute precision / recall / F1.

Matching semantics (why: see REPORT.md, "What correct means"):

  * A label and a prediction are both reduced to (key, designator class) by
    defx.normalize, so formatting differences cannot cause a mismatch.
  * Matching is one-to-one inside a document. A duplicate prediction is a false
    positive; two labels that share a key ("apotex" / "apotex, incorporated") need
    two predictions.
  * Three nested tiers are reported:
      strict  key equal and designator class equal
      core    key equal                                (headline)
      fuzzy   key similarity >= threshold              (tolerates OCR / label typos)
  * Aggregation: micro (pooled counts) is the headline because documents range from
    0 to 31 defendants; macro (mean of per-document scores) is reported next to it
    so that the two large documents cannot hide failures on the small ones.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .normalize import designator_class, match_key, parse_name

TIERS = ("strict", "core", "fuzzy")
DEFAULT_FUZZY_THRESHOLD = 0.90


@dataclass(frozen=True)
class Item:
    text: str  # what to show in an error listing
    key: str
    des_class: str | None


@dataclass
class DocResult:
    doc_id: str
    n_gold: int
    n_pred: int
    tp: dict[str, int]  # per tier
    matches: list[tuple[Item, Item, str, float]] = field(default_factory=list)  # gold, pred, tier, score
    false_positives: list[Item] = field(default_factory=list)  # after fuzzy tier
    false_negatives: list[Item] = field(default_factory=list)

    def counts(self, tier: str) -> tuple[int, int, int]:
        tp = self.tp[tier]
        return tp, self.n_pred - tp, self.n_gold - tp


def gold_item(label: str) -> Item:
    parsed = parse_name(label)
    return Item(text=label, key=parsed.key, des_class=designator_class(parsed.designator))


def pred_item(defendant: dict) -> Item:
    """Score what the pipeline emits as its canonical form, not a re-derivation of it."""
    name_normalized = defendant.get("name_normalized")
    if name_normalized:
        key, designator = match_key(name_normalized), defendant.get("designator")
    else:  # foreign predictions file without canonical fields
        parsed = parse_name(defendant.get("name_raw") or "")
        key, designator = parsed.key, defendant.get("designator") or parsed.designator
    text = defendant.get("name_raw") or name_normalized or ""
    return Item(text=text, key=key, des_class=designator_class(designator))


def similarity(a: str, b: str) -> float:
    """Character similarity of two keys; numbers must agree exactly ("fund 1" != "fund 2")."""
    if a == b:
        return 1.0
    if re.findall(r"\d+", a) != re.findall(r"\d+", b):
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def match_document(
    doc_id: str, gold: list[Item], pred: list[Item], fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD
) -> DocResult:
    gold_left, pred_left = list(gold), list(pred)
    matches: list[tuple[Item, Item, str, float]] = []

    def take(tier: str, same) -> None:
        for g in list(gold_left):
            for p in pred_left:
                if same(g, p):
                    matches.append((g, p, tier, 1.0))
                    gold_left.remove(g)
                    pred_left.remove(p)
                    break

    # Equality is an equivalence relation, so greedy pairing is optimal for these two tiers.
    take("strict", lambda g, p: g.key == p.key and g.des_class == p.des_class)
    take("core", lambda g, p: g.key == p.key)

    candidates = sorted(
        ((similarity(g.key, p.key), gi, pi) for gi, g in enumerate(gold_left) for pi, p in enumerate(pred_left)),
        reverse=True,
    )
    used_g: set[int] = set()
    used_p: set[int] = set()
    for score, gi, pi in candidates:
        if score < fuzzy_threshold:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append((gold_left[gi], pred_left[pi], "fuzzy", score))

    by_tier = Counter(tier for _, _, tier, _ in matches)
    tp = {
        "strict": by_tier["strict"],
        "core": by_tier["strict"] + by_tier["core"],
        "fuzzy": len(matches),
    }
    return DocResult(
        doc_id=doc_id,
        n_gold=len(gold),
        n_pred=len(pred),
        tp=tp,
        matches=matches,
        false_positives=[p for i, p in enumerate(pred_left) if i not in used_p],
        false_negatives=[g for i, g in enumerate(gold_left) if i not in used_g],
    )


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall, F1. An empty side scores 1.0: predicting nothing for a
    document with no labeled defendant is correct, not undefined."""
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    return precision, recall, f1


def aggregate(results: list[DocResult], tier: str) -> dict:
    counts = [r.counts(tier) for r in results]
    tp, fp, fn = (sum(c[i] for c in counts) for i in range(3))
    micro = prf(tp, fp, fn)
    per_doc = [prf(*c) for c in counts]
    n = len(per_doc) or 1
    macro = tuple(sum(d[i] for d in per_doc) / n for i in range(3))
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "micro": dict(zip(("precision", "recall", "f1"), micro)),
        "macro": dict(zip(("precision", "recall", "f1"), macro)),
    }


def bootstrap_ci(
    results: list[DocResult], tier: str, n_resamples: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% interval for micro-F1, resampling documents (the unit of independence)."""
    if not results:
        return (0.0, 0.0)
    rng = random.Random(seed)
    counts = [r.counts(tier) for r in results]
    stats = []
    for _ in range(n_resamples):
        sample = rng.choices(counts, k=len(counts))
        stats.append(prf(*(sum(c[i] for c in sample) for i in range(3)))[2])
    stats.sort()
    return stats[int(0.025 * n_resamples)], stats[min(int(0.975 * n_resamples), n_resamples - 1)]


def score(
    gold: dict[str, list[str]],
    predictions: dict[str, list[dict]],
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> list[DocResult]:
    """Score every gold document. A document missing from predictions counts as empty."""
    return [
        match_document(
            doc_id,
            [gold_item(label) for label in labels],
            [pred_item(d) for d in predictions.get(doc_id, [])],
            fuzzy_threshold,
        )
        for doc_id, labels in gold.items()
    ]
