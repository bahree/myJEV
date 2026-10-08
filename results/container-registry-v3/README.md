# Container 0.1.3 verification scope

`publication.json` identifies the immutable image, source revision and evidence hashes. `gpu-check.json`, `gpu-check.http.txt`, the CLI log and installed-source checks record the actual behavior verified on an A30 with a populated, read-only model cache and offline Hub loading.

`verify.log` is intentionally empty captured stdout. Its hash verifies the captured file only; it is not evidence that checks passed. Use the structured checks and request transcripts for that conclusion. The source-only patch reused the public 0.1.2 dependency image. The failed BuildKit attempt and successful legacy-builder logs are retained separately. No 0.1.3 empty-model-cache measurement is claimed.
