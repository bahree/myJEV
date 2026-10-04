import hashlib
import json
from pathlib import Path
from safetensors.torch import save_file


def save_artifact(path, network, manifest):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if manifest.get("adapter"):
        network.backbone.save_pretrained(path / "adapter", selected_adapters=["default"])
    save_file({k: v.detach().cpu().contiguous() for k, v in network.heads.state_dict().items()},
              str(path / "heads.safetensors"))
    manifest = dict(manifest)
    manifest["checksums"] = {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(path.rglob("*")) if p.is_file()
                             and p.name != "manifest.json" and p.suffix != ".pt"}
    content = json.dumps(manifest, sort_keys=True).encode()
    manifest["artifact_revision"] = hashlib.sha256(content).hexdigest()
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
