"""Tests for the parts that surround the model call: input reading, failure isolation,
the retry loop and the cache. The model itself is replaced by fakes; no network is used."""

import asyncio
import json

import pytest

from defx import llm, pipeline
from defx.llm import CallResult, ConfigurationError, ExtractionError, LLMClient
from defx.pipeline import Config, process_document, read_documents
from defx.schema import Extraction


def extraction(*names):
    return Extraction.model_validate({"defendants": [
        {"evidence": "", "name_raw": n, "aliases": [], "is_organization": True, "is_placeholder": False,
         "source": "caption", "state_of_registration": None} for n in names]})


def write_lines(path, *lines):
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------- reading the input

def test_read_documents_accepts_valid_lines_and_skips_blank_ones(tmp_path):
    path = write_lines(tmp_path / "in.jsonl", json.dumps({"doc_id": "a", "primary_text": "x"}), "",
                       json.dumps({"doc_id": "b", "primary_text": "y", "supplemental_text": None}))
    assert [d["doc_id"] for d in read_documents(path)] == ["a", "b"]


def test_numeric_doc_id_is_coerced_to_a_string(tmp_path):
    path = write_lines(tmp_path / "in.jsonl", json.dumps({"doc_id": 12345, "primary_text": "x"}))
    assert read_documents(path)[0]["doc_id"] == "12345"


@pytest.mark.parametrize("line,expected", [
    ('{"doc_id": "x1", "primary_text": "ACME v. BETA INC."', "invalid JSON"),
    ('{"doc_id": "x2", "supplemental_text": "..."}', "expected an object with 'doc_id' and 'primary_text'"),
    ('{"doc_id": null, "primary_text": "x"}', "expected an object"),
    ('["not", "an", "object"]', "expected an object"),
])
def test_malformed_lines_stop_with_file_and_line_number(tmp_path, line, expected):
    path = write_lines(tmp_path / "in.jsonl", json.dumps({"doc_id": "ok", "primary_text": "x"}), line)
    with pytest.raises(SystemExit) as stop:
        read_documents(path)
    assert f"{path}:2: " in str(stop.value) and expected in str(stop.value)


# ---------------------------------------------------------------- failure isolation

class FakeClient:
    """Stands in for LLMClient: scripted outcome per document text."""

    model = "fake-model"

    def __init__(self, fail_on=(), fatal_on=()):
        self.fail_on, self.fatal_on, self.calls = fail_on, fatal_on, 0

    async def extract(self, messages):
        self.calls += 1
        text = messages[1]["content"]
        if any(marker in text for marker in self.fatal_on):
            raise ConfigurationError("bad key")
        if any(marker in text for marker in self.fail_on):
            raise ExtractionError("model returned no valid extraction")
        return CallResult(extraction=extraction("Acme, Inc."), prompt_tokens=10, completion_tokens=5, latency_s=0.1)

    async def close(self):
        pass


def process(doc, client):
    return asyncio.run(process_document(doc, client, Config(track=False), asyncio.Semaphore(2)))


def test_successful_document_is_post_processed():
    prediction, call = process({"doc_id": "a", "primary_text": "ROE v. ACME, INC., Defendant."}, FakeClient())
    assert [d.name_normalized for d in prediction.defendants] == ["acme"] and prediction.error is None
    assert call.prompt_tokens == 10


def test_a_failing_document_becomes_an_empty_record_with_an_error():
    prediction, call = process({"doc_id": "a", "primary_text": "BROKEN"}, FakeClient(fail_on=("BROKEN",)))
    assert prediction.defendants == [] and "ExtractionError" in prediction.error and call is None


@pytest.mark.parametrize("doc", [
    {"doc_id": 12345, "primary_text": "BROKEN"},  # this used to crash the whole run
    {"doc_id": "a", "primary_text": 42},  # wrong type for the text
    {"doc_id": "a", "primary_text": "ok", "supplemental_text": ["a", "list"]},
])
def test_process_document_never_raises_on_bad_documents(doc):
    prediction, _ = process(doc, FakeClient(fail_on=("BROKEN",)))
    assert prediction.doc_id == str(doc["doc_id"]) and prediction.defendants == [] and prediction.error


def test_a_configuration_error_is_not_isolated():
    with pytest.raises(ConfigurationError):
        process({"doc_id": "a", "primary_text": "NOKEY"}, FakeClient(fatal_on=("NOKEY",)))


def test_run_writes_one_line_per_document_in_input_order(tmp_path, monkeypatch):
    path = write_lines(tmp_path / "in.jsonl", *(json.dumps({"doc_id": i, "primary_text": t})
                                                 for i, t in [(1, "ROE v. ACME, INC."), (2, "BROKEN"), (3, "ROE v. ACME, INC.")]))
    monkeypatch.setattr(pipeline, "LLMClient", lambda model=None, use_cache=True: FakeClient(fail_on=("BROKEN",)))
    out = tmp_path / "nested" / "out.jsonl"
    kpis = asyncio.run(pipeline.run(path, str(out), Config(track=False)))
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["doc_id"] for r in rows] == ["1", "2", "3"]
    assert "error" in rows[1] and rows[1]["defendants"] == [] and "error" not in rows[0]
    assert (kpis["docs"], kpis["failed_docs"], kpis["defendants"], kpis["empty_docs"]) == (3, 1, 2, 1)
    assert not list(out.parent.glob("*.tmp"))  # the temporary file was renamed away
    # LLM run time: two successful calls of 0.1 s each; the failed document contributes nothing
    assert kpis["llm_time_s"] == pytest.approx(0.2) and kpis["llm_time_from_scratch_s"] == pytest.approx(0.2)
    assert (kpis["latency_mean_s"], kpis["latency_p95_s"], kpis["latency_max_s"]) == (0.1, 0.1, 0.1)


def test_llm_time_separates_this_run_from_cached_calls():
    from defx.schema import Prediction
    predictions = [Prediction(doc_id=str(i), defendants=[]) for i in range(3)]
    calls = [CallResult(extraction=extraction(), latency_s=2.0),
             CallResult(extraction=extraction(), latency_s=4.0, cached=True),
             CallResult(extraction=extraction(), latency_s=9.0, cached=True)]
    kpis = pipeline.run_kpis(predictions, calls, FakeClient(), Config(), "in.jsonl", "out.jsonl", wall_s=2.1)
    assert (kpis["llm_calls"], kpis["cache_hits"]) == (1, 2)
    assert (kpis["llm_time_s"], kpis["llm_time_from_scratch_s"]) == (2.0, 15.0)
    assert (kpis["latency_mean_s"], kpis["latency_p50_s"], kpis["latency_max_s"]) == (5.0, 4.0, 9.0)


def test_slow_calls_raise_an_alert():
    from defx import monitor
    assert monitor.run_alerts({"docs": 1, "latency_p95_s": 61.0}) == ["p95 LLM latency 61.0s > 60s"]
    assert monitor.run_alerts({"docs": 1, "latency_p95_s": 12.0}) == []


def test_run_stops_without_output_on_a_configuration_error(tmp_path, monkeypatch):
    path = write_lines(tmp_path / "in.jsonl", json.dumps({"doc_id": "a", "primary_text": "NOKEY"}))
    monkeypatch.setattr(pipeline, "LLMClient", lambda model=None, use_cache=True: FakeClient(fatal_on=("NOKEY",)))
    out = tmp_path / "out.jsonl"
    with pytest.raises(SystemExit, match="configuration error"):
        asyncio.run(pipeline.run(path, str(out), Config(track=False)))
    assert not out.exists()


# ---------------------------------------------------------------- the retry loop and the cache

class Transient(Exception):
    pass


class Fatal(Exception):
    pass


class BadRequest(Exception):
    pass


MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
TOO_LARGE = "max_tokens is too large: 32000. This model supports at most 16384 completion tokens, whereas you provided 32000."


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A real LLMClient whose single API call (_request) is scripted by the test."""
    monkeypatch.setenv("PORTKEY_API_KEY", "test-key")
    monkeypatch.setattr(llm, "RETRYABLE", (Transient,))
    monkeypatch.setattr(llm, "FATAL", (Fatal,))
    monkeypatch.setattr(llm.openai, "BadRequestError", BadRequest)
    monkeypatch.setattr(llm, "backoff_seconds", lambda attempt: 0.0)
    c = LLMClient(model="fake-model", cache_dir=tmp_path / "cache")

    def script(*outcomes):
        queue, c.requests = list(outcomes), 0

        async def fake_request(messages):
            c.requests += 1
            outcome = queue.pop(0) if len(queue) > 1 else queue[0]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome, None

        monkeypatch.setattr(c, "_request", fake_request)

    c.script = script
    return c


def test_transient_errors_are_retried_and_counted(client):
    client.script(Transient(), Transient(), extraction("Acme"))
    result = asyncio.run(client.extract(MESSAGES))
    assert (result.retries, result.schema_repairs, client.requests) == (2, 0, 3)


def test_gives_up_after_max_attempts(client):
    client.script(Transient())
    with pytest.raises(ExtractionError, match="gave up after 5 attempts"):
        asyncio.run(client.extract(MESSAGES))
    assert client.requests == 5


def test_wrong_key_or_model_is_a_configuration_error_and_is_not_retried(client):
    client.script(Fatal("403"))
    with pytest.raises(ConfigurationError):
        asyncio.run(client.extract(MESSAGES))
    assert client.requests == 1


def test_invalid_output_is_asked_again_and_counted_separately(client):
    client.script(None, extraction("Acme"))  # None = refusal or JSON that failed validation
    result = asyncio.run(client.extract(MESSAGES))
    assert (result.retries, result.schema_repairs) == (0, 1)


def test_output_ceiling_is_lowered_from_the_error_and_the_call_retried(client):
    client.script(BadRequest(TOO_LARGE), extraction("Acme"))
    asyncio.run(client.extract(MESSAGES))
    assert client.max_output_tokens == 16384 and client.requests == 2


def test_every_concurrent_call_recovers_from_the_output_ceiling_error(client, monkeypatch):
    # Regression: the first fix retried only the call that lowered the limit, so of eight
    # concurrent calls seven failed. Each call here is rejected until the limit is lowered.
    async def fake_request(messages):
        sent = client.max_output_tokens
        await asyncio.sleep(0)  # let the other calls send their request with the old limit too
        if sent > 16384:
            raise BadRequest(TOO_LARGE)
        return extraction("Acme"), None

    monkeypatch.setattr(client, "_request", fake_request)

    async def eight():
        return await asyncio.gather(*(client.extract([MESSAGES[0], {"role": "user", "content": f"doc {i}"}])
                                      for i in range(8)))

    assert len(asyncio.run(eight())) == 8


def test_other_bad_requests_fail_the_document(client):
    client.script(BadRequest("context length exceeded"))
    with pytest.raises(ExtractionError, match="bad request"):
        asyncio.run(client.extract(MESSAGES))
    assert client.requests == 1


def test_second_call_is_served_from_the_cache(client):
    client.script(extraction("Acme"))
    first = asyncio.run(client.extract(MESSAGES))
    second = asyncio.run(client.extract(MESSAGES))
    assert (first.cached, second.cached, client.requests) == (False, True, 1)
    assert second.extraction == first.extraction


def test_a_different_request_does_not_hit_the_cache(client):
    client.script(extraction("Acme"))
    asyncio.run(client.extract(MESSAGES))
    asyncio.run(client.extract([MESSAGES[0], {"role": "user", "content": "another document"}]))
    assert client.requests == 2


def test_failures_are_not_cached_and_a_corrupt_entry_is_ignored(client):
    client.script(Transient())
    with pytest.raises(ExtractionError):
        asyncio.run(client.extract(MESSAGES))
    assert not list(client.cache_dir.glob("*.json"))
    client.script(extraction("Acme"))
    asyncio.run(client.extract(MESSAGES))
    entry = next(client.cache_dir.glob("*.json"))
    entry.write_text("{not json", encoding="utf-8")
    assert asyncio.run(client.extract(MESSAGES)).cached is False


def test_missing_key_stops_with_a_clear_message(monkeypatch):
    monkeypatch.delenv("PORTKEY_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="PORTKEY_API_KEY is not set"):
        LLMClient()
