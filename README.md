# Defendant extraction from legal complaints

Reads OCR'd complaints (JSONL), extracts the defendants with an LLM, and scores the result.
The reasoning, numbers, error analysis and known flaws are in [REPORT.md](REPORT.md).

## Setup (about one minute)

Python 3.10 or newer (developed on 3.14).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PORTKEY_API_KEY=<your Portkey key>
```

LLM access is **option A** from the brief: the OpenAI SDK with the Portkey gateway as base URL.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `PORTKEY_API_KEY` | yes | - | Portkey key (`OPENAI_API_KEY` is accepted as a fallback) |
| `LLM_MODEL` | no | `gpt-4.1` | model identifier; `--model` overrides it |
| `OPENAI_BASE_URL` | no | `https://api.portkey.ai/v1` | gateway URL |

A `.env` file in the repository root is read if present (`PORTKEY_API_KEY=...`, one variable per line). It is git-ignored, and the key is never written to disk by the code.

## Reproduce

```bash
python run.py --input data/dev.jsonl  --output dev_predictions.jsonl
python evaluate.py --predictions dev_predictions.jsonl --gold data/dev.jsonl
python run.py --input data/eval.jsonl --output predictions.jsonl
```

All 55 documents take one to two minutes (50 s and 95 s in two fresh-clone runs) and cost about $0.85 on `gpt-4.1`. `predictions.jsonl` in this repository is the output of the third command.

A fresh run can differ from the committed file on a few borderline documents: `temperature=0` and `seed=0` do not make the model deterministic (measured: 0 of 40 eval and 2 of 15 dev documents changed between fresh runs; the eval output was identical in four of four).

Model responses are cached in `.cache/`, keyed by model, prompt and schema, so an interrupted run resumes where it stopped and a repeated run makes no API calls. `--no-cache` forces fresh calls.

## More commands

```bash
pytest -q                                   # 149 tests: deterministic parts, plus the retry loop and failure isolation with a fake client
python evaluate.py --predictions dev_predictions.jsonl --gold data/dev.jsonl --errors
                                            # list false positives / negatives per document
python evaluate.py --predictions dev_predictions.jsonl --gold data/dev.jsonl \
       --label-fixes data/dev_label_fixes.json   # score against documented label corrections
python evaluate.py --predictions predictions.jsonl --gold data/eval_audit.jsonl
                                            # score against my hand-audited subset of eval
python -m defx.monitor                      # tracked KPIs and threshold alerts (see below)
```

Every `run.py` run appends one record to `experiments/runs.jsonl` (`--no-track` to skip), and every `evaluate.py --tag NAME` run appends one to `experiments/metrics_history.jsonl`. The folder is created on first use and is git-ignored:

| Group | KPIs |
|---|---|
| Quality (needs labels) | precision, recall, F1 at three tiers, micro and macro, bootstrap interval |
| LLM run time | `llm_time_s` (time in the model this run), `llm_time_from_scratch_s` (the same without the cache), per-call mean / p50 / p95 / max, `wall_s` |
| Operations | failed documents, model calls, cache hits, retries, output repairs |
| Cost | input and output tokens, `cost_usd`, `spent_usd` |
| Output health (no labels needed) | defendants per document, empty documents, exclusions by reason, grounding rate, `ocr_suspect` count |

Alerts fire on: F1 drop over 0.02, any failed document, grounding rate under 0.95, more than 15% empty documents, cost over $0.02 per document, p95 call latency over 60 s. The thresholds are defaults, not tuned.

`run.py` options: `--model`, `--prompt {v0..v4}` (default `v3`), `--policy {labels,raw}`, `--concurrency` (default 8), `--no-cache`, `--no-track`.

`run.py` exits 1 if any document failed; a failed document is still written, with `defendants: []` and an `error` field. A wrong key or an inaccessible model aborts the run before anything is written.

## Output

One JSON object per input document, in input order:

```json
{
  "doc_id": "c0412",
  "defendants": [
    {
      "name_raw": "Wal-Mart Stores, Inc.",
      "name_normalized": "wal-mart stores",
      "designator": "incorporated",
      "is_organization": true,
      "us_state_of_registration": "Delaware",
      "name_quality": "ok",
      "source": "caption",
      "aliases": []
    }
  ],
  "excluded_parties": [
    {"name_raw": "DOES 1-50", "is_organization": false, "reason": "placeholder"}
  ]
}
```

| Field | Meaning |
|---|---|
| `name_raw` | name as read from the text, OCR line breaks repaired |
| `name_normalized` | lower case, legal designator and d/b/a, f/k/a clauses removed |
| `designator` | legal form spelled out (`incorporated`, `limited liability company`, ...) or `null` |
| `us_state_of_registration` | US state, only when the text states it as the place of incorporation or organization |
| `name_quality` | `ok`: found verbatim in the text; `ocr_suspect`: reassembled from a broken layout; `ungrounded`: not found, possible hallucination; `placeholder`: fictitious name with a real trade name |
| `source` | `caption` if the name appears in `primary_text`, else `body` (computed, not asked of the model) |
| `excluded_parties` | parties the complaint sues that are left out of `defendants`: `individual`, `placeholder` (Does), `unselected_option` (entry of an unmarked check-list). `--policy raw` keeps individuals and placeholders in `defendants` |
| `input_repairs` | present when the text was repaired before extraction (`font_shift`) |

## Layout

```
run.py, evaluate.py     thin command-line entry points
defx/preprocess.py      deterministic text repair before the model (font-shift decoding)
defx/prompt.py          prompts, versioned v0-v4
defx/llm.py             the only network code: client, structured output, retries, cache
defx/postprocess.py     deterministic: canonical fields, de-duplication, grounding, scope policy
defx/normalize.py       name normalization shared by the pipeline and the scorer
defx/scoring.py         matching semantics and metrics
defx/monitor.py         KPI history and alerts
tests/                  unit tests: normalize, postprocess, preprocess, scoring, pipeline (fake model client)
data/dev_label_fixes.json   documented corrections to dev labels (opt-in)
data/eval_audit.jsonl       my own labels for 28 eval documents (opt-in)
```
