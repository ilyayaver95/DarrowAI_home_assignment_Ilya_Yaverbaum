#!/usr/bin/env python
"""Score a predictions file against labeled documents.

    python evaluate.py --predictions dev_predictions.jsonl --gold data/dev.jsonl

Prints per-document and aggregate precision / recall / F1. Matching semantics and
aggregation are defined and justified in defx/scoring.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from defx import monitor
from defx.scoring import DEFAULT_FUZZY_THRESHOLD, TIERS, aggregate, bootstrap_ci, prf, score


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                sys.exit(f"{path}:{line_no}: invalid JSON ({exc})")
    return rows


def load_gold(path: str, fixes_path: str | None, exclude: set[str]) -> dict[str, list[str]]:
    gold = {}
    for row in read_jsonl(path):
        if "labels" not in row:
            sys.exit(f"{path}: document {row.get('doc_id')} has no 'labels'; is this the labeled file?")
        if row["doc_id"] not in exclude:
            gold[row["doc_id"]] = list(row["labels"])
    if fixes_path:
        # Documented label corrections: {doc_id: {"remove": [...], "add": [...], "note": "..."}}
        fixes = json.loads(Path(fixes_path).read_text(encoding="utf-8"))
        for doc_id, fix in fixes.items():
            if doc_id not in gold:
                continue
            for label in fix.get("remove", []):
                if label not in gold[doc_id]:
                    sys.exit(f"{fixes_path}: '{label}' is not a label of {doc_id}")
                gold[doc_id].remove(label)
            gold[doc_id].extend(fix.get("add", []))
    return gold


def load_predictions(path: str, gold_ids: set[str]) -> dict[str, list[dict]]:
    predictions: dict[str, list[dict]] = {}
    for row in read_jsonl(path):
        doc_id = row.get("doc_id")
        if doc_id in predictions:
            print(f"warning: duplicate prediction for {doc_id}; keeping the first", file=sys.stderr)
            continue
        predictions[doc_id] = row.get("defendants") or []
    missing = sorted(gold_ids - predictions.keys())
    extra = sorted(predictions.keys() - gold_ids)
    if missing:
        print(f"warning: {len(missing)} gold document(s) without a prediction, scored as empty: {missing}", file=sys.stderr)
    if extra:
        print(f"warning: {len(extra)} predicted document(s) not in gold, ignored", file=sys.stderr)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True, help="predictions JSONL written by run.py")
    parser.add_argument("--gold", required=True, help="labeled JSONL (doc_id, labels)")
    parser.add_argument("--errors", action="store_true", help="list false positives / negatives per document")
    parser.add_argument("--label-fixes", help="JSON file of documented label corrections to apply to gold")
    parser.add_argument("--exclude-docs", default="", help="comma-separated doc_ids to leave out")
    parser.add_argument("--fuzzy-threshold", type=float, default=DEFAULT_FUZZY_THRESHOLD)
    parser.add_argument("--json", dest="json_out", help="also write the metrics to this file")
    parser.add_argument("--tag", help="append the metrics to experiments/metrics_history.jsonl under this name")
    args = parser.parse_args()

    exclude = {d for d in args.exclude_docs.split(",") if d}
    gold = load_gold(args.gold, args.label_fixes, exclude)
    predictions = load_predictions(args.predictions, set(gold))
    results = score(gold, predictions, args.fuzzy_threshold)

    print(f"Per document (core tier: name key equal, designator ignored)   gold={args.gold}")
    print(f"{'doc_id':<10}{'gold':>5}{'pred':>5}{'TP':>4}{'FP':>4}{'FN':>4}{'P':>7}{'R':>7}{'F1':>7}   strict/fuzzy TP")
    for r in results:
        tp, fp, fn = r.counts("core")
        p, rec, f1 = prf(tp, fp, fn)
        print(f"{r.doc_id:<10}{r.n_gold:>5}{r.n_pred:>5}{tp:>4}{fp:>4}{fn:>4}{p:>7.2f}{rec:>7.2f}{f1:>7.2f}"
              f"   {r.tp['strict']}/{r.tp['fuzzy']}")

    summary = {tier: aggregate(results, tier) for tier in TIERS}
    ci = bootstrap_ci(results, "core")
    print(f"\nAggregate over {len(results)} documents, {sum(r.n_gold for r in results)} labels, "
          f"{sum(r.n_pred for r in results)} predictions")
    print(f"{'tier':<8}{'TP':>4}{'FP':>4}{'FN':>4}  {'micro P':>8}{'micro R':>8}{'micro F1':>9}  "
          f"{'macro P':>8}{'macro R':>8}{'macro F1':>9}")
    for tier in TIERS:
        s = summary[tier]
        mark = "  <- headline" if tier == "core" else ""
        print(f"{tier:<8}{s['tp']:>4}{s['fp']:>4}{s['fn']:>4}  "
              f"{s['micro']['precision']:>8.3f}{s['micro']['recall']:>8.3f}{s['micro']['f1']:>9.3f}  "
              f"{s['macro']['precision']:>8.3f}{s['macro']['recall']:>8.3f}{s['macro']['f1']:>9.3f}{mark}")
    print(f"core micro-F1 95% bootstrap interval over documents: [{ci[0]:.3f}, {ci[1]:.3f}]")

    fuzzy_only = [(r.doc_id, g, p, s) for r in results for g, p, tier, s in r.matches if tier == "fuzzy"]
    if fuzzy_only:
        print(f"\nMatched only by the fuzzy tier (similarity >= {args.fuzzy_threshold}); review by eye:")
        for doc_id, g, p, s in fuzzy_only:
            print(f"  {doc_id}  {s:.2f}  label={g.text!r}  prediction={p.text!r}")

    if args.errors:
        print("\nErrors after the fuzzy tier")
        for r in results:
            if not (r.false_positives or r.false_negatives):
                continue
            print(f"  {r.doc_id}")
            for item in r.false_positives:
                print(f"    FP  {item.text!r}  (key={item.key})")
            for item in r.false_negatives:
                print(f"    FN  {item.text!r}  (key={item.key})")

    record = {
        "timestamp": monitor.now(),
        "tag": args.tag,
        "predictions": args.predictions,
        "gold": args.gold,
        "variant": "fixed" if args.label_fixes else "raw",
        "excluded_docs": sorted(exclude),
        "docs": len(results),
        "labels": sum(r.n_gold for r in results),
        "predicted": sum(r.n_pred for r in results),
        **summary,
        "core_micro_f1_ci": list(ci),
        "fuzzy_only_matches": len(fuzzy_only),
    }
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(record, indent=2), encoding="utf-8")
    if args.tag:
        monitor.append(monitor.EVAL_HISTORY, record)
        print(f"\nappended to {monitor.EVAL_HISTORY} as '{args.tag}'")


if __name__ == "__main__":
    main()
