# Qwen Image 2.1 context reuse on Metal

## What changed

The server adapts the native mflux 0.20.0 transformer during a generation.
Loading, weights, quantization, text encoding, the scheduler and the VAE remain
native. This is an inference optimization, not a replacement model backend.

Qwen 2.1 uses causal attention for text, and text receives timestep-zero
modulation. Consequently, changing the image latents or diffusion timestep does
not change the text's state at any layer. On the first step we process the joint
sequence and retain each layer's normalized, rotary-encoded text keys and its
text values. On subsequent steps we compute Q/K/V, attention queries, residuals
and feed-forward activations only for image tokens. Each image query still
attends to all text and image keys and values. No denoising steps are skipped.

The adapter also broadcasts image modulation from a single row, prepares shared
scale/gate terms once per step, and projects only the image tokens at the output.
Both paths are compiled with MLX. KV arrays enter and leave compiled functions
explicitly; mutable Python cache state stays outside tracing. Compiled functions
survive subsequent requests, while cached prompt tensors do not.
Cached and new K/V are concatenated in the native token-major storage layout,
then exposed as attention heads. This avoids an extra layout-conversion copy and
reduces the numerical drift observed when concatenating in head order.

## Boundaries

- Enabled by default for Qwen Image 2.1; `--no-qwen21-context-cache` restores the
  native transformer call. `/api/info` advertises `context_kv_cache`.
- Two independent branches cover positive and negative conditioning. Identity
  and image geometry checks prevent same-length prompts or different image
  layouts from sharing entries.
- Cache entries are bounded to 512 MiB combined and cleared on success or error.
  That cap covers retained KV tensors, not model weights or temporary workspace.
- Padding, unsupported batches, oversized contexts and unsupported transformer
  types retain native execution. Fallback invalidates the upstream geometry
  entry so a different padding mask cannot reuse a same-shape mask.
- No cross-request KV reuse, cache quantization, approximate layer skipping,
  reference-image editing extension, or changes to other model families.
- Mathematically equivalent computation need not be bit-identical in bf16:
  changing matrix sizes/layouts changes floating-point rounding. An opt-out and
  numerical comparison tools are included for this reason.

## Measurements

Machine: Apple M1 Ultra, 64 GiB unified memory. Python 3.12, mflux 0.20.0,
MLX 0.32.2. Real Qwen Image 2.1 transformer weights quantized to 8 bits. These
are synchronized denoising timings, **not end-to-end generation times**. Text
encoding, VAE decode, network and queue latency are excluded.

The synthetic-input benchmark warms each path, alternates measurement order,
clears the context before each run, and reports the median of two runs. Its
cached timing includes the first step/cache fill. Metal allocator cache is zero,
matching the existing server default. [Raw measurements](qwen21-metal-results.json)
include individual timings, memory peaks and numerical differences.

| Image | Steps | Text tokens | Native | Cached | Speedup |
|---|---:|---:|---:|---:|---:|
| 512×512 | 6 | 64 | 8.672 s | 8.364 s | 1.04× |
| 512×512 | 6 | 256 | 10.116 s | 8.489 s | 1.19× |
| 512×512 | 6 | 768 | 14.282 s | 9.542 s | 1.50× |

More image tokens dilute the benefit of text reuse: a short prompt at a larger
resolution has much less invariant work to eliminate. At 512×512, measured peak
MLX allocation rose from 8.44–8.58 GB to 8.92–9.39 GB across these cases (decimal
GB, transformer process only). The server's encoder and VAE require additional
memory. The 512×512 measurements were made with the live server left idle.

Synthetic embeddings are useful for controlled token-count comparisons, but are
not a quality evaluation. Comparison against an actual encoded prompt and
decoded pixels is also necessary. Relative final-latent RMSE was 1.28%, 11.29%
and 7.13% for the three synthetic cases above; the raw results retain these
errors alongside the timings.

For a real 101-token prompt describing a red teapot, 256×256, seed 42, 20 steps,
and a 512 MiB allocator cache, a single native/cached comparison measured
10.302 / 7.271 seconds of denoising (1.42×). Final-latent relative RMSE was 1.88%
and decoded-image PSNR was 37.82 dB. Visual inspection preserved composition,
objects and detail; small pixel differences remain. This is one example, not a
claim that all prompts retain the same pixel similarity or speedup. That
comparison includes first-call compilation; use the warmed benchmark for timing.

An allocator experiment with the initial cache implementation changed only the
allocator cache from zero to 512 MiB and improved cached timings by approximately
0.5–1.5% at 256×256. The server default stays zero;
the measured benefit did not justify changing its memory-sharing policy.
`--cache_limit 536870912` remains available for workload-specific experiments.
This allocator cache is distinct from the new context KV cache.

## Reproduce

Use an already-downloaded local snapshot; neither weight loader downloads model
weights. Keep other GPU workloads idle when measuring. Start with the small
synthetic model to check the harness, then use actual weights:

```sh
.venv/bin/python3.12 benchmarks/benchmark_qwen21_context.py
.venv/bin/python3.12 benchmarks/benchmark_qwen21_context.py \
  --model-path /path/to/Qwen-Image-2.1/snapshot \
  --size 512 --steps 6 --text-lengths 64 256 768 --repeats 2
HF_HUB_OFFLINE=1 .venv/bin/python3.12 benchmarks/verify_qwen21_context.py \
  --model-path /path/to/Qwen-Image-2.1/snapshot \
  --output-dir /tmp/qwen21-comparison
.venv/bin/python3.12 -m unittest discover -s tests -v
```

The image verifier writes `native.png`, `cached.png`, and `metrics.json`. It
loads encoder, transformer and VAE sequentially to reduce memory pressure.
The regression tests cover every-layer invariance, denoising trajectories at
unquantized/q4/q8 precision, CFG isolation, geometry changes, padding, memory
bounds, native restoration after exceptions, and repeated requests with different
prompts through the same compiled functions.

## Current primary guidance used

- [Qwen 2.1 transformer API](https://huggingface.co/docs/diffusers/main/api/models/qwenimage21_transformer2d):
  causal conditioning is the prerequisite for prefix extraction/reuse.
- [MLX compilation](https://ml-explore.github.io/mlx/build/html/usage/compile.html):
  pure compiled functions with explicit state avoid capturing stale arrays;
  shape-dependent code should not blindly use shapeless compilation.
- [MLX fast attention](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html):
  retain native SDPA and its hardware heuristics. Forcing a fused kernel is not
  necessarily faster, so this adapter does not force it or add a custom Metal
  attention kernel.
- [MLX allocation cache](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_cache_limit.html):
  zero disables allocator reuse; this is separate from retaining model context.
