import hashlib
import json
import re
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoTokenizer
from .model import DecisionNetwork, load_backbone
from .prompt import PROMPT_VERSION, encode
from .schema import ScoreRequest


def _artifact_file(directory, filename, *, hub_revision=None):
    """Resolve a manifest member without allowing traversal or arbitrary links.

    Only SDK-downloaded, pinned Hub snapshots may link to their repository's
    sibling blobs directory or the SDK's shared sharded blob store. Local
    artifacts retain the within-directory rule.
    """
    if (not isinstance(filename, str) or not filename or "\\" in filename
            or ":" in filename or any(part in ("", ".", "..") for part in filename.split("/"))):
        raise ValueError(f"unsafe artifact path: {filename}")
    root = directory.resolve()
    target = (root / filename).resolve()
    if target.is_relative_to(root):
        return target
    if hub_revision and root.name == hub_revision and root.parent.name == "snapshots":
        # Do not resolve this allowed directory: a redirected blobs directory
        # must not grant access outside the SDK repository cache.
        blobs = root.parent.parent / "blobs"
        if target.parent == blobs:
            return target
        # Newer Hub SDKs deduplicate large files in cache/blobs/aa/<64hex>.
        # These are storage object addresses, not necessarily file SHA-256s;
        # the artifact manifest still verifies the actual file content below.
        shared_blobs = root.parent.parent.parent / "blobs"
        if target.is_relative_to(shared_blobs):
            relative = target.relative_to(shared_blobs)
            if (len(relative.parts) == 2
                    and re.fullmatch(r"[0-9a-f]{64}", relative.name)
                    and relative.parts[0] == relative.name[:2]):
                return target
    raise ValueError(f"unsafe artifact path: {filename}")


class DecisionModel:
    def __init__(self, network, tokenizer, manifest):
        self.network = network.eval()
        self.tokenizer = tokenizer
        self.manifest = manifest

    @classmethod
    def load(cls, artifact, revision=None, device="cuda:0", adapter_trainable=False):
        path = Path(artifact)
        hub_revision = None
        if not path.is_dir():
            if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ValueError("Hub artifacts require an immutable 40-character commit revision")
            path = Path(snapshot_download(artifact, revision=revision))
            hub_revision = revision
        m = json.loads(_artifact_file(path, "manifest.json", hub_revision=hub_revision).read_text())
        if m.get("format") == "myjev-scratch-v1":
            if adapter_trainable:
                raise ValueError("scratch artifacts do not contain LoRA adapters")
            from .scratch.model import ScratchDecisionModel
            return ScratchDecisionModel.load(path, device=device)
        if m["schema_version"] != 1 or m["prompt_version"] != PROMPT_VERSION:
            raise ValueError("unsupported manifest/prompt version")
        if not re.fullmatch(r"[0-9a-f]{40}", m["backbone_revision"]):
            raise ValueError("backbone revision must be pinned")
        for filename, digest in m["checksums"].items():
            p = _artifact_file(path, filename, hub_revision=hub_revision)
            if hashlib.sha256(p.read_bytes()).hexdigest() != digest:
                raise ValueError(f"artifact checksum mismatch: {filename}")
        tokenizer = AutoTokenizer.from_pretrained(m["backbone"], revision=m["tokenizer_revision"])
        if len(set(m["alias_ids"])) != len(m["aliases"]):
            raise ValueError("alias IDs must be distinct")
        for alias, token in zip(m["aliases"], m["alias_ids"], strict=True):
            if tokenizer.encode(alias, add_special_tokens=False) != [token]:
                raise ValueError("alias/tokenizer mismatch")
        backbone = load_backbone(m["backbone"], m["backbone_revision"], m["precision"], device)
        if m.get("adapter"):
            backbone = PeftModel.from_pretrained(backbone, path / "adapter", is_trainable=adapter_trainable)
        network = DecisionNetwork(backbone)
        network.heads.load_state_dict(load_file(str(path / "heads.safetensors")))
        return cls(network, tokenizer, m)

    def logits(self, request):
        r = request if isinstance(request, ScoreRequest) else ScoreRequest.model_validate(request)
        m = self.manifest
        if len(r.candidates) > m["max_candidates"]:
            raise ValueError("candidate count exceeds artifact limit")
        inputs = encode(self.tokenizer, r, m["aliases"], m["max_tokens"])
        device = next(self.network.heads.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        return r, self.network(inputs, m["alias_ids"][:len(r.candidates)])

    @torch.inference_mode()
    def score(self, request, diagnostics=False):
        r, (answers, policy, scalar) = self.logits(request)
        m = self.manifest
        scores = (answers[0].float() / m.get("temperature", 1.0)).softmax(-1)
        chosen = int(scores.argmax())
        mode = m["confidence_mode"]
        if mode == "policy":
            confidence = (policy[0, chosen].softmax(-1) * torch.linspace(0, 1, 21, device=policy.device)).sum()
        elif mode == "scalar":
            confidence = scalar[0, chosen].sigmoid()
        elif mode == "selection":
            confidence = scores[chosen]
        elif mode == "constant":
            confidence = torch.tensor(m["constant_confidence"])
        else:
            raise ValueError("unknown confidence mode")
        result = {"selected_id": r.candidates[chosen].id,
                "selection_scores": {c.id: float(s) for c, s in zip(r.candidates, scores)},
                "confidence": float(confidence), "confidence_mode": mode,
                "artifact_revision": m["artifact_revision"],
                "calibration_revision": m.get("calibration_revision", "uncalibrated")}
        if diagnostics:
            result["selection_logits"] = answers[0].float().cpu().tolist()
            grid = torch.linspace(0, 1, 21, device=policy.device)
            qp = policy[0].softmax(-1)
            result["confidence_moments"] = {"mean": (qp * grid).sum(-1).cpu().tolist(),
                                              "second": (qp * grid.square()).sum(-1).cpu().tolist()}
            result["confidence_controls"] = {
                "scalar": float(scalar[0, chosen].sigmoid()),
                "policy": float((policy[0, chosen].softmax(-1) * torch.linspace(0, 1, 21, device=policy.device)).sum()),
                "selection": float(scores[chosen]),
            }
        return result
