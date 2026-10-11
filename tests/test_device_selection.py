import logging

import pytest
from fastapi.testclient import TestClient

from myjev.inference import DecisionModel, resolve_device
from myjev.server import create_app


def test_auto_uses_visible_cuda_and_preserves_explicit_index(monkeypatch, caplog):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert resolve_device("auto", "bf16") == "cuda:0"
    assert resolve_device("cuda:2", "nf4") == "cuda:2"
    assert not caplog.records


def test_no_gpu_fallback_warns_and_explicit_cuda_does_not_fall_back(monkeypatch, caplog):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with caplog.at_level(logging.WARNING):
        assert resolve_device("auto", "bf16") == "cpu"
    assert "Falling back to CPU" in caplog.text
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_device("cuda:0", "bf16")


@pytest.mark.parametrize("device", ["auto", "cpu"])
def test_nf4_does_not_silently_change_precision_or_model_on_cpu(monkeypatch, device):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(ValueError, match="NF4 9B"):
        resolve_device(device, "nf4")


def test_explicit_cpu_is_honored_on_a_gpu_host(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert resolve_device("cpu", "bf16") == "cpu"


def test_server_default_and_override_reach_the_shared_loader(monkeypatch):
    calls = []
    class Loaded:
        def score(self, request):
            return {"selected_id": "a"}
    def load(cls, artifact, revision=None, device=None):
        calls.append((artifact, revision, device))
        return Loaded()
    monkeypatch.setattr(DecisionModel, "load", classmethod(load))
    monkeypatch.setenv("MYJEV_ARTIFACT", "fixture")
    monkeypatch.delenv("MYJEV_REVISION", raising=False)
    monkeypatch.delenv("MYJEV_DEVICE", raising=False)
    with TestClient(create_app()) as client:
        assert client.get("/readyz").status_code == 200
    monkeypatch.setenv("MYJEV_DEVICE", "cpu")
    with TestClient(create_app()) as client:
        assert client.get("/readyz").status_code == 200
    assert calls == [("fixture", None, "auto"), ("fixture", None, "cpu")]
