"""KPI tracking without infrastructure: two append-only JSONL files and a report.

  experiments/metrics_history.jsonl   one record per tagged `evaluate.py` run (quality)
  experiments/runs.jsonl              one record per `run.py` run (operations and output health)

`python -m defx.monitor` prints both histories with the change against the previous
record and raises alerts when a KPI crosses a threshold. Output-health KPIs
(grounding rate, empty documents, defendants per document) need no labels, so they
are the only quality signal available for eval.jsonl and for production traffic.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

EVAL_HISTORY = Path("experiments/metrics_history.jsonl")
RUN_HISTORY = Path("experiments/runs.jsonl")

# Alert thresholds. A breach is printed and makes `--strict` exit non-zero.
MAX_F1_DROP = 0.02  # core micro-F1 against the previous record on the same gold file
MIN_GROUNDING_RATE = 0.95  # share of emitted names that can be found in the document text
MAX_FAILED_DOCS = 0
MAX_EMPTY_DOC_RATE = 0.15
MAX_COST_PER_DOC_USD = 0.02
MAX_LATENCY_P95_S = 60.0  # one model call, retries included


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def eval_alerts(records: list[dict]) -> list[str]:
    """Regression check: latest record against the previous one scored on the same gold."""
    if len(records) < 2:
        return []
    last = records[-1]
    previous = [r for r in records[:-1] if r.get("gold") == last.get("gold") and r.get("variant") == last.get("variant")]
    if not previous:
        return []
    drop = previous[-1]["core"]["micro"]["f1"] - last["core"]["micro"]["f1"]
    if drop > MAX_F1_DROP:
        return [f"core micro-F1 dropped {drop:.3f} from '{previous[-1]['tag']}' to '{last['tag']}'"]
    return []


def run_alerts(record: dict) -> list[str]:
    alerts = []
    docs = record.get("docs") or 0
    if record.get("failed_docs", 0) > MAX_FAILED_DOCS:
        alerts.append(f"{record['failed_docs']} document(s) failed and were written with empty output")
    if record.get("grounding_rate") is not None and record["grounding_rate"] < MIN_GROUNDING_RATE:
        alerts.append(f"grounding rate {record['grounding_rate']:.3f} < {MIN_GROUNDING_RATE}")
    if docs and record.get("empty_docs", 0) / docs > MAX_EMPTY_DOC_RATE:
        alerts.append(f"{record['empty_docs']}/{docs} documents have no defendant")
    if record.get("latency_p95_s", 0) > MAX_LATENCY_P95_S:
        alerts.append(f"p95 LLM latency {record['latency_p95_s']:.1f}s > {MAX_LATENCY_P95_S:.0f}s")
    if docs and record.get("llm_calls") and record.get("cost_usd", 0) / docs > MAX_COST_PER_DOC_USD:
        alerts.append(f"cost per document ${record['cost_usd'] / docs:.4f} > ${MAX_COST_PER_DOC_USD}")
    return alerts


def _table(header: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    lines = ["  ".join(str(h).ljust(w) for h, w in zip(header, widths))]
    lines += ["  ".join(str(c).ljust(w) for c, w in zip(row, widths)) for row in rows]
    return "\n".join(lines)


def _delta(current: float, previous: float | None) -> str:
    return "" if previous is None else f"{current - previous:+.3f}"


def report(last: int = 15) -> list[str]:
    alerts: list[str] = []

    evals = load(EVAL_HISTORY)
    print(f"== Quality KPIs ({EVAL_HISTORY}) ==")
    if evals:
        rows, prev_by_gold = [], {}
        for r in evals:
            group = (r.get("gold"), r.get("variant"))
            f1 = r["core"]["micro"]["f1"]
            rows.append([
                r["timestamp"], r["tag"], r.get("variant", ""), r["docs"],
                f"{r['core']['micro']['precision']:.3f}", f"{r['core']['micro']['recall']:.3f}", f"{f1:.3f}",
                _delta(f1, prev_by_gold.get(group)),
                f"{r['core']['macro']['f1']:.3f}", f"{r['strict']['micro']['f1']:.3f}", f"{r['fuzzy']['micro']['f1']:.3f}",
                f"[{r['core_micro_f1_ci'][0]:.2f}, {r['core_micro_f1_ci'][1]:.2f}]",
            ])
            prev_by_gold[group] = f1
        print(_table(
            ["timestamp", "tag", "labels", "docs", "P", "R", "F1", "dF1", "macroF1", "strictF1", "fuzzyF1", "95% CI"],
            rows[-last:],
        ))
        alerts += eval_alerts(evals)
    else:
        print("(no tagged evaluations yet: python evaluate.py ... --tag NAME)")

    runs = load(RUN_HISTORY)
    print(f"\n== Run KPIs ({RUN_HISTORY}) ==")
    if runs:
        rows = []
        for r in runs[-last:]:
            rows.append([
                r["timestamp"], Path(r.get("input", "")).name, r.get("prompt_version", ""), r.get("docs", 0),
                r.get("failed_docs", 0), r.get("llm_calls", 0), r.get("cache_hits", 0), r.get("retries", 0),
                "" if "llm_time_s" not in r else f"{r['llm_time_s']:.0f}/{r['llm_time_from_scratch_s']:.0f}",
                f"{r.get('latency_p50_s', 0):.1f}/{r.get('latency_p95_s', 0):.1f}",
                f"{r.get('prompt_tokens', 0)}/{r.get('completion_tokens', 0)}", f"${r.get('cost_usd', 0):.4f}",
                r.get("defendants", 0), r.get("empty_docs", 0), r.get("excluded_parties", 0),
                "" if r.get("grounding_rate") is None else f"{r['grounding_rate']:.3f}", r.get("ocr_suspect", 0),
            ])
        print(_table(
            ["timestamp", "input", "prompt", "docs", "failed", "calls", "cached", "retries", "LLM s now/scratch", "p50/p95 s",
             "tokens in/out", "cost", "defendants", "empty", "excluded", "grounded", "ocr_suspect"],
            rows,
        ))
        alerts += run_alerts(runs[-1])
    else:
        print("(no runs yet: python run.py --input ... --output ...)")

    print("\n== Alerts ==")
    print("\n".join(f"ALERT: {a}" for a in alerts) if alerts else "none")
    return alerts


def main() -> None:
    parser = argparse.ArgumentParser(description="Show tracked KPIs and threshold alerts.")
    parser.add_argument("--last", type=int, default=15, help="rows to show per table")
    parser.add_argument("--strict", action="store_true", help="exit 1 if any alert fires")
    args = parser.parse_args()
    alerts = report(args.last)
    if args.strict and alerts:
        sys.exit(1)


if __name__ == "__main__":
    main()
