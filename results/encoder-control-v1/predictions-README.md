# Encoder prediction archives

The four `*-predictions.jsonl.gz` files preserve all recorded prediction fields without input text. `prediction-archive.json` records compressed and uncompressed SHA-256 values. Raw test files exceed the public exporter’s 10 MiB per-file cap, so this lossless encoding is used instead.

To restore one file for analysis:

```bash
gzip -dk results/encoder-control-v1/temperature-test-predictions.jsonl.gz
```

The labels originate from BANKING77, whose provenance and CC BY 4.0 terms are documented in the dataset card. These files contain no blog-archive data.
