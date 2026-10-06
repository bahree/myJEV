"""Byte encoder and shared candidate scorer, without vocabulary answer aliases."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file, save_file
from myjev.schema import ScoreRequest


@dataclass(frozen=True)
class ScratchConfig:
    width: int = 64
    layers: int = 2
    heads: int = 4
    ff_multiplier: int = 4
    max_context_bytes: int = 256
    max_candidate_bytes: int = 64
    max_candidates: int = 32
    candidate_interaction: bool = True

    def __post_init__(self):
        if min(self.width, self.layers, self.heads, self.ff_multiplier,
               self.max_context_bytes, self.max_candidate_bytes) < 1:
            raise ValueError('positive architecture dimensions required')
        if self.width % self.heads or not 2 <= self.max_candidates <= 160:
            raise ValueError('invalid head divisibility or candidate count')


def encode(text, limit):
    # PAD=0, BOS=1, byte values occupy 2..257. No corpus or pretrained vocabulary.
    raw = text.encode('utf-8')
    if len(raw) > limit:
        raise ValueError(f'input exceeds tested byte limit {limit}')
    return [1] + [b+2 for b in raw]


def collate(requests, config, device='cpu'):
    requests = [r if isinstance(r, ScoreRequest) else ScoreRequest.model_validate(r) for r in requests]
    if not requests:
        raise ValueError('empty request batch')
    contexts, candidates = [], []
    for r in requests:
        if len(r.candidates) > config.max_candidates:
            raise ValueError('too many candidates for scratch artifact')
        contexts.append(encode(r.instructions+'\n'+r.context, config.max_context_bytes))
        candidates.append([encode(c.description, config.max_candidate_bytes) for c in r.candidates])
    b, k = len(requests), max(map(len, candidates))
    context = torch.zeros(b, max(map(len, contexts)), dtype=torch.long, device=device)
    options = torch.zeros(b, k, max(len(c) for row in candidates for c in row), dtype=torch.long, device=device)
    valid = torch.zeros(b, k, dtype=torch.bool, device=device)
    for i, (ctx, row) in enumerate(zip(contexts, candidates)):
        context[i, :len(ctx)] = torch.tensor(ctx, device=device)
        for j, c in enumerate(row):
            options[i, j, :len(c)] = torch.tensor(c, device=device)
            valid[i, j] = True
        # Dummy BOS keeps all-masked padded candidate encodings numerically finite.
        options[i, len(row):, 0] = 1
    return dict(context=context, options=options, valid=valid)


class ScratchNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token = nn.Embedding(258, config.width, padding_idx=0)
        self.position = nn.Embedding(max(config.max_context_bytes, config.max_candidate_bytes)+1, config.width)
        block = nn.TransformerEncoderLayer(config.width, config.heads,
                    config.width*config.ff_multiplier, dropout=0., batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, config.layers, enable_nested_tensor=False)
        self.cross = nn.MultiheadAttention(config.width, config.heads, dropout=0., batch_first=True)
        if config.candidate_interaction:
            self.interaction = nn.TransformerEncoderLayer(config.width, config.heads,
                config.width*config.ff_multiplier, dropout=0., batch_first=True, norm_first=True)
        else:
            self.interaction = None
        self.norm = nn.LayerNorm(config.width)
        self.selection = nn.Linear(config.width, 1)
        self.scalar = nn.Linear(config.width, 1)
        self.policy = nn.Linear(config.width, 21)

    def encode(self, tokens):
        mask = tokens.eq(0)
        x = self.token(tokens) + self.position(torch.arange(tokens.shape[-1], device=tokens.device))
        x = self.encoder(x, src_key_padding_mask=mask)
        pooled = (x * (~mask).unsqueeze(-1)).sum(1) / (~mask).sum(1, keepdim=True)
        return x, pooled, mask

    def forward(self, context, options, valid):
        b, k, length = options.shape
        ctx, _, mask = self.encode(context)
        _, opt, _ = self.encode(options.reshape(b*k, length))
        opt = opt.reshape(b, k, -1)
        evidence, _ = self.cross(opt, ctx, ctx, key_padding_mask=mask, need_weights=False)
        x = self.norm(opt + evidence)
        if self.interaction is not None:
            x = self.interaction(x, src_key_padding_mask=~valid)
        return dict(answer=self.selection(x).squeeze(-1).masked_fill(~valid, -torch.inf),
                    scalar=self.scalar(x).squeeze(-1), policy=self.policy(x))


class ScratchDecisionModel:
    def __init__(self, network, *, confidence_mode='untrained', temperature=1., revision='unsaved', calibration_revision='none'):
        if confidence_mode not in ('untrained', 'scalar', 'policy', 'temperature'):
            raise ValueError('unknown confidence mode')
        if not 0 < temperature < float('inf'):
            raise ValueError('invalid temperature')
        self.network = network.eval()
        self.config = network.config
        self.confidence_mode, self.temperature = confidence_mode, temperature
        self.revision, self.calibration_revision = revision, calibration_revision

    @torch.inference_mode()
    def score(self, request):
        r = request if isinstance(request, ScoreRequest) else ScoreRequest.model_validate(request)
        device = next(self.network.parameters()).device
        out = self.network(**collate([r], self.config, device))
        probabilities = (out['answer'][0]/self.temperature).softmax(-1)
        selected = int(probabilities.argmax())
        if self.confidence_mode == 'untrained':
            confidence = None
        elif self.confidence_mode == 'scalar':
            confidence = float(out['scalar'][0, selected].sigmoid())
        elif self.confidence_mode == 'policy':
            q = torch.linspace(0, 1, 21, device=device)
            confidence = float((out['policy'][0, selected].softmax(-1)*q).sum())
        else:
            confidence = float(probabilities[selected])
        return dict(selected_id=r.candidates[selected].id,
                    selection_scores={c.id: float(probabilities[i]) for i,c in enumerate(r.candidates)},
                    confidence=confidence, confidence_mode=self.confidence_mode,
                    artifact_revision=self.revision, calibration_revision=self.calibration_revision)

    def save(self, directory):
        path = Path(directory)
        if path.exists() and any(path.iterdir()):
            raise ValueError('refusing to overwrite nonempty artifact directory')
        path.mkdir(parents=True, exist_ok=True)
        save_file({k:v.detach().cpu().contiguous() for k,v in self.network.state_dict().items()}, str(path/'model.safetensors'))
        manifest = dict(format='myjev-scratch-v1', tokenizer='utf8-byte-v1', config=asdict(self.config),
                        confidence_mode=self.confidence_mode, temperature=self.temperature,
                        calibration_revision=self.calibration_revision,
                        weights_sha256=hashlib.sha256((path/'model.safetensors').read_bytes()).hexdigest())
        raw=json.dumps(manifest, sort_keys=True).encode()
        manifest['artifact_revision']=hashlib.sha256(raw).hexdigest()
        (path/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        self.revision=manifest['artifact_revision']

    @classmethod
    def load(cls, directory, device='cpu'):
        path=Path(directory);manifest=json.loads((path/'manifest.json').read_text())
        revision=manifest.pop('artifact_revision')
        if hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()!=revision:
            raise ValueError('scratch manifest checksum mismatch')
        if manifest['format']!='myjev-scratch-v1' or manifest['tokenizer']!='utf8-byte-v1':
            raise ValueError('unsupported scratch artifact')
        if hashlib.sha256((path/'model.safetensors').read_bytes()).hexdigest()!=manifest['weights_sha256']:
            raise ValueError('scratch weights checksum mismatch')
        network=ScratchNetwork(ScratchConfig(**manifest['config']))
        network.load_state_dict(load_file(str(path/'model.safetensors')))
        return cls(network.to(device), confidence_mode=manifest['confidence_mode'], temperature=manifest['temperature'],
                   revision=revision, calibration_revision=manifest['calibration_revision'])
