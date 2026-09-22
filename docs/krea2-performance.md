# Krea 2 inference preparation

Implemented against mflux 0.20.0 / MLX 0.32.2, 22 September 2026.
Krea 2 Turbo uses 8 steps, guidance 1, native `er_sde` sampling and optional
quantization (none unless requested). Select it with `--model krea2`; the server
now defaults to the ungated FLUX.2 Klein 4B.

## What changed

`krea2_inference.py` prepares each positive/negative text branch through the
native text fusion and text MLP once per request. The denoising loop receives
those prepared arrays explicitly, alongside cached positional frequencies.
At eight steps this removes seven repeated text-fusion evaluations per branch.
Joint text/image hidden states are still computed at every step; Krea does not
have Qwen 2.1's invariant per-layer prefix.

The adapter also passes 12 K/V heads directly to MLX attention with 48 query
heads, instead of materializing four copies, and projects only image tokens
through the final output layer. It retains native weights, normalization,
RoPE, modulation, feed-forward layers, sampler, img2img and VAE behavior.
It does not skip steps, change guidance, lower precision or reuse changing
hidden states.

Compiled executables belong to the model; prepared tensors belong to the
request and are released in `finally`. The native `_predict` method is restored
after each call, including failures. Conditioning is passed into compiled
executables as array arguments so later prompts cannot reuse captured inputs.
Compilation follows upstream's helper: disabled on M1/M2 **except Max/Ultra**.
The M1 Ultra used here therefore compiles both preparation and denoising.

`inference_runtime.py` supplies reusable building blocks for later model
adapters: scoped method replacement, tensor-budgeted LRU caches and opt-in
synchronized stage timing. Krea prompt caching is limited to 16 entries and
256 MiB of logical tensor payload. Request geometry has a 64 MiB budget.
Backing buffers and runtime overhead are additional; these are not process
RAM limits. No synchronization for profiling is added to normal server requests.

Use `--no-krea2-optimizations` to compare the native prediction path. The bounded
prompt cache still applies. `/api/info` reports `prepared_text`, `cached_geometry`,
`grouped_attention` and `image_only_output` in `inference_optimizations` when
enabled for the current model. Its existing `context_kv_cache` remains false
for Krea.

## Validation performed

Tests use random native components without downloading weights:

- Eight-step native Euler and ER-SDE trajectories at BF16, Q8 and Q4, both
  compiled and uncompiled, with guidance 1 and 2.5 and unequal branch lengths.
- Individual optimizations, batches 1 and 2, odd latent dimensions/padding and
  different aspect ratios with the same token count.
- Prompt changes through reused compiled executables, preparation once per
  branch, model replacement, disabled mode and cleanup after failure.
- The actual native `generate_image` pipeline for text-to-image and img2img,
  with external text/VAE encoders stubbed, plus server defaults/API forwarding.
- LRU eviction by byte budget and entry count.

Relative RMS tolerances are 1e-5 for uncompiled comparisons and 1e-4 for
compiled comparisons. These tests establish component/pipeline compatibility;
they do not establish image quality with trained weights.

## Reduced-model performance measurement

Apple M1 Ultra, BF16, compilation enabled, Metal allocator cache zero, eight
ER-SDE steps, guidance 1, seven repeats with seeds 42–48 and alternating native/
optimized order. Each path is warmed with two steps. Preparation is included.
The random transformer has **4 layers, width 512, 8 query / 2 K/V heads,
text width 256 and 128 text tokens**. The real model has much larger/different
dimensions, so these percentages cannot be extrapolated to Krea Turbo images.
There is no text encoder or VAE in this synthetic measurement.

| Latent grid (image-size equivalent) | Native median | Optimized median | Less time |
|---|---:|---:|---:|
| 32×32 (256×256) | 73.30 ms | 56.03 ms | 23.6% |
| 64×64 (512×512) | 116.29 ms | 97.75 ms | 15.9% |
| 128×128 (1024×1024) | 277.24 ms | 233.82 ms | 15.7% |

Maximum final-latent relative RMS difference across these comparisons was
3.90e-5. Observed peak MLX memory dropped from approximately 225→167,
310→238 and 758→721 MiB respectively. These are this small benchmark's process
peaks, including resident parameters and comparison state, not real-model RAM
requirements. [Raw samples and stage timings](krea2-synthetic-results.json).

```sh
.venv/bin/python3.12 benchmarks/benchmark_krea2.py \
  --synthetic --sizes 256 512 1024 --repeats 7 \
  --output-dir /tmp/krea2-synthetic
```

For component comparisons, add
`--modes native prepared grouped output optimized`. Every adapter mode also
reuses geometry. The benefit of grouped attention depends on shapes/hardware;
the combination and complete generation must be measured, not just one kernel.

## Full-checkpoint benchmark still required

The official `krea/Krea-2-Turbo` snapshot download returned **HTTP 401** because
the repository is gated and no Hugging Face token is configured here. No real
Krea checkpoint generation or full-model speed claim was possible in this run.
The adapter is enabled based on the tests and reduced-model measurements above;
trained-checkpoint quality, latency and peak memory remain unverified.

After authenticating with an account granted access and obtaining the complete
snapshot, compare real generations without starting or changing the server:

```sh
.venv/bin/python3.12 benchmarks/benchmark_krea2.py \
  --model-path /path/to/Krea-2-Turbo/snapshot \
  --sizes 512 1024 --steps 8 --repeats 3 --output-dir /tmp/krea2-real
```

The script uses offline loading and reports prompt encoding, input latent
preparation, invariant preparation, denoising, VAE decoding, total time and
peak MLX bytes. It saves first-repeat images and compares final latents with
native results. Model loading, HTTP and output-file writes are excluded from
generation timings. Cold prompt-cache runs are the default; repeat with
`--prompt-cache warm` for repeated prompts. Repeat in separate processes with
`--quantize 8` or `--quantize 4` to determine the best Krea precision locally.

Also compare longer prompts, `--guidance 2.5` and
`--init-image /path/to/image.png`. Img2img retains the server's strength 0.4 and
the native scheduler's resulting subset of steps. Inspect generated images as
well as numerical differences before claiming full-model quality/performance.

The next adapters can reuse the same cache/lifetime/measurement utilities while
preparing their own invariant tensors: Z-Image caption refinement, then FIBO
caption projections. See the [cross-model audit](model-performance-audit.md).
