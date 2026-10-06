#!/usr/bin/env python
"""Extract defendants from legal complaints.

    python run.py --input data/dev.jsonl --output dev_predictions.jsonl

Environment: PORTKEY_API_KEY (required), LLM_MODEL (default gpt-4.1),
OPENAI_BASE_URL (default https://api.portkey.ai/v1). A `.env` file is read if present.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from defx import monitor
from defx.pipeline import Config, run
from defx.prompt import DEFAULT_PROMPT_VERSION, PROMPTS


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env reader; variables already in the environment win."""
    env_file = Path(path)
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="JSONL with doc_id, primary_text, supplemental_text")
    parser.add_argument("--output", required=True, help="where to write predictions JSONL")
    parser.add_argument("--model", help="model identifier (default: $LLM_MODEL)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT_VERSION, choices=sorted(PROMPTS), help="prompt version")
    parser.add_argument("--policy", default="labels", choices=["labels", "raw"],
                        help="labels: organizations only, placeholders excluded (reviewer convention); raw: everything")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--no-cache", action="store_true", help="ignore and do not reuse cached model responses")
    parser.add_argument("--no-track", action="store_true", help="do not append run KPIs to experiments/runs.jsonl")
    args = parser.parse_args()

    load_dotenv()
    config = Config(prompt_version=args.prompt, policy=args.policy, concurrency=args.concurrency,
                    use_cache=not args.no_cache, track=not args.no_track)
    kpis = asyncio.run(run(args.input, args.output, config, args.model))

    print(f"wrote {kpis['docs']} documents to {args.output}  "
          f"(model={kpis['model']}, prompt={kpis['prompt_version']}, policy={kpis['policy']})", file=sys.stderr)
    print(f"  calls={kpis['llm_calls']} cached={kpis['cache_hits']} retries={kpis['retries']} "
          f"failed={kpis['failed_docs']}  wall={kpis['wall_s']}s",
          file=sys.stderr)
    print(f"  LLM time: {kpis['llm_time_s']}s this run ({kpis['llm_time_from_scratch_s']}s without the cache)  "
          f"per call mean/p50/p95/max={kpis['latency_mean_s']}/{kpis['latency_p50_s']}/{kpis['latency_p95_s']}/{kpis['latency_max_s']}s",
          file=sys.stderr)
    print(f"  tokens in/out={kpis['prompt_tokens']}/{kpis['completion_tokens']}  cost=${kpis['cost_usd']:.4f} "
          f"(spent now ${kpis['spent_usd']:.4f})", file=sys.stderr)
    print(f"  defendants={kpis['defendants']} (from body {kpis['from_body']})  empty docs={kpis['empty_docs']}  "
          f"excluded: {kpis['excluded_individuals']} individuals, {kpis['excluded_placeholders']} placeholders  "
          f"grounding={kpis['grounding_rate']}  ocr_suspect={kpis['ocr_suspect']}", file=sys.stderr)
    for alert in monitor.run_alerts(kpis):
        print(f"  ALERT: {alert}", file=sys.stderr)
    if kpis["failed_docs"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
