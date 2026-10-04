import torch
from torch import nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


class DecisionNetwork(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        config = backbone.config.get_text_config()
        hidden = config.hidden_size
        # Low-rank interaction conditions confidence on both input and candidate alias.
        self.state_proj = nn.Linear(hidden, 128)
        self.candidate_proj = nn.Linear(hidden, 128, bias=False)
        self.policy = nn.Linear(128, 21)
        self.scalar = nn.Linear(128, 1)
        self.heads.to(device=backbone.get_input_embeddings().weight.device)

    @property
    def heads(self):
        return nn.ModuleDict({name: getattr(self, name) for name in
                              ("state_proj", "candidate_proj", "policy", "scalar")})

    def forward(self, inputs, alias_ids):
        # Invoke the decoder directly: no autoregressive decode and no full-vocabulary
        # logits for every input position. PEFT's wrapper remains active on base layers.
        base = self.backbone.get_base_model() if hasattr(self.backbone, "peft_config") else self.backbone
        decoder = getattr(base, base.base_model_prefix)
        out = decoder(**inputs, use_cache=False, return_dict=True)
        hidden = out.last_hidden_state[:, -1].float()
        weights = base.get_output_embeddings().weight[alias_ids].float()
        answers = hidden @ weights.T
        z = torch.tanh(self.state_proj(hidden)[:, None, :] + self.candidate_proj(weights)[None, :, :])
        return answers, self.policy(z), self.scalar(z).squeeze(-1)


def load_backbone(model_id, revision, precision="bf16", device="cuda:0"):
    kwargs = dict(revision=revision, trust_remote_code=False,
                  dtype=torch.float32 if precision == "fp32" else torch.bfloat16)
    if precision == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kwargs["device_map"] = {"": device}
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if precision == "nf4":
        # Match PEFT k-bit preparation at inference and every continuation: embeddings,
        # layer norms and the unquantized output projection remain FP32.
        from peft import prepare_model_for_kbit_training
        return prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    return model.to(device)
