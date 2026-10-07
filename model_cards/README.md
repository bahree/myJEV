# Hugging Face model cards

The six `myJEV*.md` files are editable, reader-facing cards. They explain which model to choose, measured quality and resource costs, the training data, confidence semantics, complete inference commands and licensing. They are published as model-repository `README.md` files.

`facts.json` retains the metric values, input hashes and verified runtime revisions used to build the first reader-facing edition. `scripts/build_hub_model_cards.py` regenerates the six cards from those facts and its authored introductory text. Regeneration overwrites local card edits, so review the diff before publishing. The publisher reads the reviewed Markdown files directly; it does not regenerate them.

```bash
python scripts/build_hub_model_cards.py
# Use fresh paths for the next edition. Without --apply, no upload occurs.
python scripts/publish_hub_model_cards.py \
  --manifest results/model-card-refresh-v2/publication-manifest.json \
  --packages artifacts/hub-reader-next \
  --output results/model-card-refresh-next/publication-manifest.json
# Review the six cards and prepared package, then publish the same snapshot.
python scripts/publish_hub_model_cards.py \
  --manifest results/model-card-refresh-v2/publication-manifest.json \
  --packages artifacts/hub-reader-next \
  --output results/model-card-refresh-next/publication-manifest.json --apply
```

Each edition needs a fresh `--packages` path and `--output` manifest. The current reader edition uses `artifacts/hub-reader-v2` and `results/model-card-refresh-v2/publication-manifest.json`; version 1 predates the additional fixed-taxonomy encoder comparison. Old packages, cards and receipts remain intact. The publisher changes only `README.md`; all weight, head, calibration, license and manifest hashes must match the prior release.

The code samples pin a verified earlier immutable artifact revision with identical runtime files. A card cannot embed its own not-yet-known Git commit hash, so a later card-only commit does not replace the examples' runtime pin. The source code pin includes support for both Hugging Face cache layouts.
