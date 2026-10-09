# Runtime choices for the readout comparison

The first model load failed in both arms, despite requesting BF16 without quantization. The import path required bitsandbytes. The console recorded:

```text
ModuleNotFoundError: No module named 'bitsandbytes'
ImportError: Unsloth: Please install bitsandbytes via `pip install bitsandbytes`
```

Installing bitsandbytes 0.50.2 resolved the import. It remains in `requirements.unsloth.lock`; this experiment does not use four-bit weights.

Clef's first compiled backward pass then failed before completing an optimizer update:

```text
AssertionError: expected size 4==4, stride 1910==1920 at dim=0
```

The generated assertion expected a stride of 1,920 for a tensor whose actual stride was 1,910. The alias arm was stopped while compiling; it had also completed no update. No validation or test results existed for these attempts.

Both arms subsequently set `UNSLOTH_COMPILE_DISABLE=1` and `TORCH_COMPILE_DISABLE=1` before the successful pilots and all tuning. They use the reference PyTorch causal-convolution implementation. The [frozen configuration](../../configs/head-comparison-v1.json) records those settings, and the [environment record](environment.json) records the pinned dependencies.

These choices allow a comparison within the same eager runtime. They leave the performance of optimized Unsloth kernels unmeasured. The [readout guide](../../docs/unsloth.md) describes the prompt and head differences that also affect work per request.
