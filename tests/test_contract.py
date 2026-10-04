import asyncio
import json
import time
import httpx
import pytest
import torch
from fastapi.testclient import TestClient
from myjev.schema import ScoreRequest
from myjev.prompt import encode
from myjev.server import create_app, WARMUP
from myjev.data import stratified_groups, validate_isolation
from myjev.metrics import thresholds_from_calibration, summarize


def test_duplicates_and_count():
    with pytest.raises(ValueError):
        ScoreRequest.model_validate({**WARMUP, "candidates": [WARMUP["candidates"][0]]*2})
    with pytest.raises(ValueError):
        ScoreRequest.model_validate({**WARMUP, "candidates": []})


def test_no_truncation():
    def tokenizer(*a, **kw):
        assert kw["truncation"] is False
        return {"input_ids": torch.ones(1, 10)}
    with pytest.raises(ValueError, match="no truncation"):
        encode(tokenizer, ScoreRequest.model_validate(WARMUP), ["A", "B"], 9)


def test_split_group_isolation():
    rows = [{"group": str(i), "label": str(i%2)} for i in range(100)]
    parts = stratified_groups(rows + [rows[0]])
    validate_isolation(parts)
    assert sum(map(len, parts.values())) == 101
    parts["test"] = [parts["train"][0]]
    with pytest.raises(ValueError):
        validate_isolation(parts)


class Fake:
    def score(self, request):
        return {"selected_id": "a", "confidence": .7}


def test_http_contract():
    with TestClient(create_app(loader=Fake)) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 200
        assert client.post("/score", json=WARMUP).json() == Fake().score(WARMUP)
        assert client.post("/score", json={}).status_code == 422


def test_timeout_keeps_capacity_until_worker_finishes():
    class Slow:
        def score(self, request):
            if isinstance(request, ScoreRequest):
                time.sleep(.15)
            return {"ok": True}
    with TestClient(create_app(loader=Slow, queue_size=1, timeout=.01)) as client:
        assert client.post("/score", json=WARMUP).status_code == 504
        assert client.post("/score", json=WARMUP).status_code == 429
        time.sleep(.2)
        assert client.post("/score", json=WARMUP).status_code == 504


def test_tied_confidence_no_false_coverage_claim():
    t = thresholds_from_calibration([1,0,1,0], [.5]*4)
    assert t["coverage_0.5"] == .5
    assert t["empirical_error_0.01"] > 1
    rows = [{"selected_id": "a", "label": "a" if i%2 else "b", "group": str(i),
             "confidence": .5, "selection_scores": {"a": .5, "b": .5}} for i in range(4)]
    result = summarize(rows,t)
    assert result["operating_points"]["coverage_0.5"]["coverage"] == 1
    assert result["operating_points"]["empirical_error_0.01"]["error"] is None


def test_chat_request_cannot_change_candidate_mapping():
    from myjev.prompt import render, PROMPT_VERSION
    request=ScoreRequest.model_validate(WARMUP)
    text=render(request,["A","B"])
    assert PROMPT_VERSION=="myjev-qwen-chat-v2"
    assert text.startswith("<|im_start|>system\n")
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    payload=text.split("<|im_start|>user\n")[1].split("<|im_end|>")[0]
    assert json.loads(payload)["candidates"]==[
        {"alias":"A","description":"First route"},{"alias":"B","description":"Second route"}]
