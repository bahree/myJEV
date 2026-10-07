import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
from .artifacts import save_artifact
from .data import read_jsonl, request_from_row, training_order
from .inference import DecisionModel
from .model import DecisionNetwork, load_backbone
from .objectives import exact_loss, joint_log_probs, sampled_loss, supervised_loss
from .prompt import PROMPT_VERSION, discover_aliases, encode
from .schema import ScoreRequest
from .tracking import start_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--initial")
    p.add_argument("--resume")
    p.add_argument("--method", choices=["sft", "sft_brier", "continued_sft", "exact", "sampled"], default="sft")
    p.add_argument("--reward", choices=["confidence", "correctness"], default="confidence")
    p.add_argument("--updates", type=int)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    cfg.setdefault("reinforce_samples", 8)
    cfg.setdefault("weight_decay", 0.01)
    if args.updates is not None:
        cfg["updates"] = args.updates
    if args.method in ("exact", "sampled", "continued_sft") and not args.initial:
        p.error("method requires a shared supervised --initial artifact")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    if Path(args.data).stem != "train":
        p.error("training input must be the frozen train.jsonl partition")
    dataset_manifest = Path(args.data).parent / "manifest.json"
    if dataset_manifest.exists() and json.loads(dataset_manifest.read_text()).get("purpose") == "transfer-only":
        p.error("transfer-only data cannot be used for training")
    rows = read_jsonl(args.data)
    if not rows:
        p.error("empty training data")
    digest = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    reference = None
    initial_revision = None
    if args.initial:
        initial = DecisionModel.load(args.initial, device=args.device, adapter_trainable=True)
        initial_revision = initial.manifest["artifact_revision"]
        net, tokenizer, m = initial.network, initial.tokenizer, dict(initial.manifest)
        for key, expected in (("backbone",cfg["backbone"]),("backbone_revision",cfg["revision"]),("precision",cfg["precision"])):
            if m[key] != expected:
                raise ValueError(f"initial artifact/config mismatch: {key}")
        if args.method in ("exact", "sampled"):
            # Both policies share the immutable backbone, with distinct adapter weights
            # and confidence heads. This avoids a second 9B backbone on a 24 GiB GPU.
            net.backbone.load_adapter(Path(args.initial) / "adapter", adapter_name="reference", is_trainable=False)
            reference = DecisionNetwork(net.backbone)
            reference.heads.load_state_dict(net.heads.state_dict())
            reference.heads.requires_grad_(False)
            net.backbone.set_adapter("default")

    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"], revision=cfg["revision"])
        backbone = load_backbone(cfg["backbone"], cfg["revision"], cfg["precision"], args.device)
        # load_backbone already prepares NF4 weights; checkpointing is enabled
        # once below after adapter attachment.
        backbone = get_peft_model(backbone, LoraConfig(r=cfg["lora_rank"], lora_alpha=2*cfg["lora_rank"],
            target_modules=cfg["lora_targets"], lora_dropout=0.0, task_type="CAUSAL_LM"))
        net = DecisionNetwork(backbone)
        aliases, ids = discover_aliases(tokenizer, cfg["max_candidates"])
        m = {"schema_version": 1, "backbone": cfg["backbone"], "backbone_revision": cfg["revision"],
             "tokenizer_revision": cfg["revision"], "precision": cfg["precision"],
             "nonquantized_dtype": cfg.get("nonquantized_dtype", "bf16"), "adapter": True,
             "aliases": aliases, "alias_ids": ids, "max_candidates": cfg["max_candidates"],
             "max_tokens": cfg["max_tokens"], "prompt_version": PROMPT_VERSION, "temperature": 1.0,
             "confidence_mode": "scalar", "calibration_revision": "uncalibrated"}
    net.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    net.backbone.enable_input_require_grads()
    net.train()
    optimizer = torch.optim.AdamW([v for v in net.parameters() if v.requires_grad], lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    start = 0
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    order, cursor = training_order(rows, rng, cfg.get("data_offset", 0))
    if args.resume:
        state = torch.load(args.resume, map_location=args.device, weights_only=False)
        # Historical checkpoints omitted these defaults; preserve resume parity.
        state["config"].setdefault("reinforce_samples", 8)
        state["config"].setdefault("weight_decay", 0.01)
        if state["data_sha256"] != digest or state["method"] != args.method or state["config"] != cfg or state.get("reward") != args.reward or state.get("initial") != args.initial:
            raise ValueError("resume dataset/method/config mismatch")
        trainable = dict(net.named_parameters())
        with torch.no_grad():
            for k, v in state["parameters"].items():
                trainable[k].copy_(v)
        optimizer.load_state_dict(state["optimizer"])
        start, order, cursor = state["step"], state["order"], state["cursor"]
        rng.setstate(state["rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda_rng"]])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(args.device)
        torch.cuda.synchronize(args.device)
    tracking = start_run(out, {**cfg, "seed": args.seed, "method": args.method,
                              "reward": args.reward, "data_sha256": digest,
                              "prompt_version": PROMPT_VERSION, "resumed_from_step": start})
    started = time.perf_counter()
    for step in range(start, cfg["updates"]):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(cfg["accumulation"]):
            if cursor == len(order):
                rng.shuffle(order)
                cursor = 0
            row = rows[order[cursor]]
            cursor += 1
            request, target = request_from_row(row, rng)
            inputs = encode(tokenizer, ScoreRequest.model_validate(request), m["aliases"], m["max_tokens"])
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            ids = m["alias_ids"][:len(request["candidates"])]
            if reference is not None:
                net.backbone.set_adapter("reference")
                with torch.no_grad():
                    ra, rp, _ = reference(inputs, ids)
                    ref = joint_log_probs(ra, rp)
                net.backbone.set_adapter("default")
            answers, policy, scalar = net(inputs, ids)
            targets = torch.tensor([target], device=args.device)
            if reference is None:
                loss = supervised_loss(answers, policy, scalar, targets,
                                       cfg["brier_weight"] if args.method == "sft_brier" else 0.0)
            else:
                correctness = torch.nn.functional.one_hot(targets, len(ids)).float()
                fn = exact_loss if args.method == "exact" else sampled_loss
                kwargs = {"samples": cfg["reinforce_samples"]} if args.method == "sampled" else {}
                loss = fn(answers, policy, correctness, ref, cfg["kl_beta"], args.reward == "confidence", **kwargs)
            (loss / cfg["accumulation"]).backward()
            total += loss.item() / cfg["accumulation"]
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize(args.device)
        log = {"step": step+1, "loss": total, "examples": (step+1)*cfg["accumulation"],
               "session_seconds": time.perf_counter()-started,
               "peak_vram_bytes": torch.cuda.max_memory_allocated(args.device) if torch.cuda.is_available() else 0}
        with (out / "training.jsonl").open("a") as f:
            f.write(json.dumps(log)+"\n")
        if tracking is not None:
            tracking.log(log, step=step+1)
        if (step+1) % 10 == 0:
            print(json.dumps(log), flush=True)
        if (step+1) % cfg.get("save_every", 100) == 0 or step+1 == cfg["updates"]:
            m.update(confidence_mode="policy" if reference is not None else "scalar",
                     training={"method": args.method, "reward": args.reward, "seed": args.seed,
                               "config": cfg, "data_sha256": digest, "steps": step+1,
                               "initial": str(args.initial), "initial_artifact_revision": initial_revision,
                               "examples": (step+1)*cfg["accumulation"]})
            saved_manifest = save_artifact(out / "artifact", net, m)
            from .server import WARMUP
            expected = DecisionModel(net, tokenizer, saved_manifest).score(WARMUP)
            (out / "reload-fixture.json").write_text(json.dumps({"request": WARMUP, "expected": expected}, indent=2))
            net.train()
            state = {"parameters": {k: v.detach().cpu() for k, v in net.named_parameters() if v.requires_grad},
                     "optimizer": optimizer.state_dict(), "step": step+1, "order": order, "cursor": cursor,
                     "rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                     "data_sha256": digest, "method": args.method, "reward": args.reward, "initial": args.initial, "config": cfg}
            torch.save(state, out / "resume.tmp.pt")
            (out / "resume.tmp.pt").replace(out / "resume.pt")
    if tracking is not None:
        tracking.finish()
    print(json.dumps({"artifact": str(out / "artifact"), "elapsed_seconds": time.perf_counter()-started}))


if __name__ == "__main__":
    main()
