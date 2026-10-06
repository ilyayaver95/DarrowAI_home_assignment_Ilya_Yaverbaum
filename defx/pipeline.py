"""Orchestration: read documents, extract concurrently, post-process, write, record KPIs."""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import monitor
from .llm import CallResult, ConfigurationError, LLMClient, cost_usd
from .postprocess import Policy, build_prediction
from .preprocess import prepare_text
from .prompt import DEFAULT_PROMPT_VERSION, build_messages
from .schema import Prediction


@dataclass
class Config:
    prompt_version: str = DEFAULT_PROMPT_VERSION
    policy: str = "labels"
    concurrency: int = 8
    use_cache: bool = True
    track: bool = True


def read_documents(path: str) -> list[dict]:
    documents = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON ({exc.msg})") from None
            if not isinstance(row, dict) or row.get("doc_id") in (None, "") or "primary_text" not in row:
                raise SystemExit(f"{path}:{line_no}: expected an object with 'doc_id' and 'primary_text'")
            row["doc_id"] = str(row["doc_id"])  # the output schema requires a string; ids may arrive as numbers
            documents.append(row)
    return documents


async def process_document(
    doc: dict, client: LLMClient, config: Config, semaphore: asyncio.Semaphore
) -> tuple[Prediction, CallResult | None]:
    """One document end to end. Never raises: a failure becomes an empty prediction with `error`."""
    try:
        primary, repairs_p = prepare_text(doc.get("primary_text") or "")
        supplemental, repairs_s = prepare_text(doc.get("supplemental_text") or "")
        async with semaphore:
            result = await client.extract(build_messages(primary, supplemental, config.prompt_version))
        prediction = build_prediction(
            str(doc["doc_id"]), primary, supplemental, result.extraction, Policy.named(config.policy),
            sorted(set(repairs_p + repairs_s)),
        )
        return prediction, result
    except ConfigurationError:
        raise  # not a property of this document: abort the run
    except Exception as exc:  # isolate: one bad document must not lose the other 54
        doc_id = str(doc.get("doc_id"))  # never let the failure record itself fail validation
        print(f"  {doc_id}: FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return Prediction(doc_id=doc_id, defendants=[], error=f"{type(exc).__name__}: {exc}"), None


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def run_kpis(
    predictions: list[Prediction], calls: list[CallResult | None], client: LLMClient, config: Config,
    input_path: str, output_path: str, wall_s: float,
) -> dict:
    done = [c for c in calls if c is not None]
    fresh = [c for c in done if not c.cached]
    defendants = [d for p in predictions for d in p.defendants]
    quality = Counter(d.name_quality for d in defendants)
    excluded = Counter(e.reason for p in predictions for e in p.excluded_parties)
    per_doc = [len(p.defendants) for p in predictions]
    prompt_tokens = sum(c.prompt_tokens for c in done)
    completion_tokens = sum(c.completion_tokens for c in done)
    return {
        "timestamp": monitor.now(),
        "input": input_path,
        "output": output_path,
        "model": client.model,
        "prompt_version": config.prompt_version,
        "policy": config.policy,
        # operations
        "docs": len(predictions),
        "failed_docs": sum(1 for p in predictions if p.error),
        "llm_calls": len(fresh),
        "cache_hits": len(done) - len(fresh),
        "retries": sum(c.retries for c in fresh),
        "schema_repairs": sum(c.schema_repairs for c in fresh),
        # LLM run time. One call's latency covers its retries and backoff. `llm_time_s` is the time
        # spent in the model during this run; the `from_scratch` figure also counts cached calls at
        # the latency they originally had. Calls overlap, so both can exceed `wall_s`.
        "llm_time_s": round(sum(c.latency_s for c in fresh), 2),
        "llm_time_from_scratch_s": round(sum(c.latency_s for c in done), 2),
        "latency_mean_s": round(statistics.fmean([c.latency_s for c in done]), 2) if done else 0.0,
        "latency_p50_s": round(statistics.median([c.latency_s for c in done]), 2) if done else 0.0,
        "latency_p95_s": round(percentile([c.latency_s for c in done], 0.95), 2),
        "latency_max_s": round(max((c.latency_s for c in done), default=0.0), 2),
        "wall_s": round(wall_s, 1),
        # cost: tokens needed to produce this output from scratch; `spent_usd` excludes cache hits
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": round(cost_usd(client.model, prompt_tokens, completion_tokens), 4),
        "spent_usd": round(
            cost_usd(client.model, sum(c.prompt_tokens for c in fresh), sum(c.completion_tokens for c in fresh)), 4
        ),
        # output health: computable without labels
        "defendants": len(defendants),
        "defendants_per_doc_mean": round(sum(per_doc) / len(per_doc), 2) if per_doc else 0.0,
        "defendants_per_doc_max": max(per_doc, default=0),
        "empty_docs": sum(1 for n in per_doc if n == 0),
        "excluded_parties": sum(excluded.values()),
        "excluded_individuals": excluded["individual"],
        "excluded_placeholders": excluded["placeholder"],
        "excluded_unselected_options": excluded["unselected_option"],
        "docs_repaired": sum(1 for p in predictions if p.input_repairs),
        "grounding_rate": round(1 - quality["ungrounded"] / len(defendants), 4) if defendants else None,
        "ocr_suspect": quality["ocr_suspect"],
        "ungrounded": quality["ungrounded"],
        "from_body": sum(1 for d in defendants if d.source == "body"),
        "with_state": sum(1 for d in defendants if d.us_state_of_registration),
    }


def write_predictions(path: str, predictions: list[Prediction]) -> None:
    target = Path(path)
    if target.parent != Path(""):
        target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for prediction in predictions:
            optional = {name for name in ("error", "input_repairs") if not getattr(prediction, name)}
            fh.write(json.dumps(prediction.model_dump(exclude=optional), ensure_ascii=False) + "\n")
    tmp.replace(target)


async def run(input_path: str, output_path: str, config: Config, model: str | None = None) -> dict:
    documents = read_documents(input_path)
    client = LLMClient(model=model, use_cache=config.use_cache)
    semaphore = asyncio.Semaphore(config.concurrency)
    started = time.monotonic()
    try:
        outcomes = await asyncio.gather(*(process_document(d, client, config, semaphore) for d in documents))
    except ConfigurationError as exc:
        raise SystemExit(f"configuration error, nothing written: {exc}") from None
    finally:
        await client.close()
    predictions = [p for p, _ in outcomes]  # gather keeps input order
    write_predictions(output_path, predictions)

    kpis = run_kpis(predictions, [c for _, c in outcomes], client, config, input_path, output_path,
                    time.monotonic() - started)
    if config.track:
        monitor.append(monitor.RUN_HISTORY, kpis)
    return kpis
