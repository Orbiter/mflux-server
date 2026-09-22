# Performance across model families

Architecture audit of this server and installed mflux 0.20.0 / MLX 0.32.2,
22 September 2026. Qwen 2.1 context reuse is implemented and benchmarked.
Krea 2 now implements prepared conditioning, geometry reuse, grouped attention
and image-only output projection; see the [implementation and validation report](krea2-performance.md).
FLUX.2 Klein 4B reuses native predictors and bounded prompt/reference encodings;
see its [real-model performance report](flux2-performance.md). Its native joint
denoising graph remains intact.
The remaining families are proposals requiring their own correctness and
end-to-end performance validation.

## What transfers from Qwen 2.1

Reuse work whose inputs do not change. A fixed prompt alone does not guarantee
fixed per-layer attention keys and values: in most image transformers, text
tokens attend to changing image tokens and receive timestep-dependent modulation.
Those text states change throughout sampling even though the input prompt does not.

Qwen 2.1's prefix uses timestep zero and cannot attend to the generated image.
That makes its full per-layer prefix reusable. Elsewhere, the safe reuse boundary
may end at a text projection or a text-only refinement stage, before joint attention.

There are three distinct caches to manage:

1. **Request conditioning:** prepared text, geometry, masks and image encodings;
   per-layer KV only for architectures that make it invariant.
2. **Repeated-request inputs:** bounded prompt/reference embedding caches, keyed
   by model state and the full preprocessing inputs.
3. **Execution resources:** compiled functions and Metal allocator buffers.
   These affect execution overhead, not the mathematical validity of context reuse.

## Model-by-model findings

| Family supported by this server | Exact reuse boundary / opportunity | What must still change each step |
|---|---|---|
| Qwen Image 2.1 | Existing text/reference prefix KV reuse; further reuse of reference vision/VAE encodings across requests is a candidate | Target image states and their attention to the prefix |
| Krea 2 Turbo | `txtfusion` and `txtmlp`, positional embeddings, image-only final output projection; native grouped-query attention without repeating K/V heads | All joint denoising-block text/image states: both timestep modulation and image attention affect them |
| Z-Image Turbo | Caption embedding and the two text-only `context_refiner` blocks, caption/image geometry and masks | Noise refinement and all joint transformer layers |
| FIBO | Initial context embedding and up to 46 layer-specific `caption_projection` results, positional embeddings and masks | Joint/single-stream hidden states; the static injected caption features are only part of the context |
| ERNIE Image / Turbo | Initial `text_proj`; reuse existing positional/mask cache rather than duplicating it | Joint hidden states and timestep modulation |
| FLUX.1, including Krea-dev | Initial context projection, constant portion of pooled-text conditioning, positional embeddings | Joint text/image states and their timestep-dependent modulation |
| Qwen Image 1.x | Initial text normalization/projection and positional data | Both streams are updated through joint attention and timestep-dependent gates |
| FLUX.2 Klein 4B / 9B currently registered | 4B now caches repeated prompt/reference encodings and native predictors; static projection reuse remains experimental | Standard checkpoint joint attention states are not an exact reusable prefix |
| Ideogram 4 | Static conditioning normalization/projection, indicator embeddings, rotary embeddings and segment masks; profile repeated FP8 weight expansion | Joint attention states and timestep modulation; retain existing preset/CFG semantics |

### Krea 2: first implementation beyond Qwen

The native transformer currently performs this sequence on each denoising call:

```text
fixed Qwen3-VL layer stack
  -> text fusion (two layerwise blocks, projection, two refiner blocks)
  -> text MLP
  -> concatenate with changing image tokens
  -> timestep-modulated joint denoising blocks
  -> final projection
```

Everything through the text MLP depends only on conditioning and model weights.
Prepare and materialize it once per positive/negative branch. At the default
eight steps, this removes seven repeated evaluations of that stage. The total
speedup depends on how much generation time that stage occupies.

The final normalization/modulation/projection is token-local. Applying it to
image tokens after slicing avoids computing text outputs that are immediately
discarded. This also needs a numerical comparison against the native path.

Krea's main attention has 48 query heads and 12 K/V heads, but currently repeats
K/V four times before SDPA. MLX explicitly supports grouped-query attention
without that expansion. The potential saving concerns temporary attention buffers
and attention execution; it does not reduce all transformer work by four times.
[MLX attention documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html)

Small local probes, using **random weights/synthetic inputs**, found:

- Reusing prepared text fusion + text MLP in a reduced native Krea transformer
  produced zero maximum absolute difference for two different image/timestep pairs.
- Attention only, 1,152 tokens: expanded K/V median 4.82 ms; grouped K/V 4.10 ms.
- Attention only, 4,224 tokens: expanded K/V median 43.11 ms; grouped K/V 40.77 ms.
- Both attention comparisons had zero maximum absolute output difference in this
  run. Logical K/V storage at 4,224 tokens fell from 99 MiB to 24.75 MiB.

These are feasibility probes, **not measured Krea image-generation speedups**.
There is no Krea 2 checkpoint in the inspected local Hugging Face cache, and no
weights were downloaded for this audit. The subsequent implementation enables
the adapter after component/trajectory tests and reduced-model benchmarks, with
`--no-krea2-optimizations` available for comparison. Full-checkpoint validation
is still outstanding: the attempted download returned HTTP 401 for the gated
repository. Do not treat reduced-model gains as real-checkpoint results.

Run `.venv/bin/python3.12 benchmarks/probe_krea2_reuse.py` to reproduce the probes.
[Raw probe results](krea2-reuse-probes.json) retain all six alternating attention
timing samples, including their variability.

### FLUX.2 has a separate checkpoint designed for cached editing

mflux 0.20.0 already supports `flux2-klein-9b-kv`, a model variant designed for
reference-image KV reuse. This server currently registers the regular 4B and
9B variants, not that KV variant. Adding it as a distinct model would reuse
upstream functionality. It would require its own weights and validation.

Do not force `use_kv_cache=True` on the regular checkpoint and call that an exact
optimization: extraction changes reference-token attention and timestep behavior.
[Upstream FLUX.2 documentation](https://github.com/mflux-community/mflux/blob/v.0.20.0/src/mflux/models/flux2/README.md#kv-cache-editing-flux2-klein-9b-kv)

## Shared performance design

Keep a common measurement and resource-management layer, with small adapters
that know each architecture's valid preparation boundary. Reuse native mflux
components rather than maintaining independent copies of entire models.

```text
request + model identity + execution policy
  -> adapter prepares invariant conditioning
  -> native sampling loop consumes prepared conditioning and changing latents
  -> native VAE decode
  -> output encoding
  -> release request resources; retain only explicitly bounded shared caches
```

The shared layer should provide:

- **Stage measurements:** queue wait, load, prompt/reference encoding, first
  denoising step, later steps, decode, output encoding and peak memory. Native
  lazy arrays must be evaluated at profiling boundaries for honest attribution;
  avoid adding per-step synchronization overhead to normal serving merely for metrics.
- **Cache accounting:** byte-bounded prompt/prepared-conditioning caches, hit/miss
  counters, and cleanup on errors, model changes and memory pressure. Several
  current native prompt caches are plain dictionaries. `mx.clear_cache()` does
  not evict live embedding tensors held by those dictionaries.
- **Complete identities:** checkpoint/model instance, quantization, adapters/LoRAs
  if present, dtype, prompt/template/mask, branch, reference contents and order,
  resize/normalization settings and geometry where relevant. Keep compiled
  functions independent of a specific request's arrays by passing those arrays
  explicitly; do not accidentally keep old prompts alive in compiled closures.
- **Hardware-specific policies:** choose quantization, selective compilation,
  allocator budget and decode tiling from measurements for each family and chip.
  Qwen's bf16 advantage on this M1 Ultra does not establish the best precision
  for Krea, Ideogram or another Mac.
- **Native fallback and isolation:** preserve the single GPU worker; adapters must
  restore original modules and release request state on every exit path. One
  family's optimization should not monkey-patch classes globally for other models.

### Shared candidates that need measurement

1. **Move invariant computation out of sampling.** Prioritize Krea text fusion,
   Z-Image caption refinement and FIBO caption projections. Then measure smaller
   static projection/geometry opportunities in the remaining families.
2. **Improve repeated-request reuse.** Most families already cache prompt encoder
   outputs; extend their useful preparation boundary and bound memory rather than
   layering duplicate caches. Z-Image's current path re-encodes prompts. Reference
   vision/VAE encodings can help repeated edits, with preprocessing-aware keys.
3. **Benchmark allocator retention.** This server defaults to allocator cache zero
   and clears it after each task. Compare bounded reuse against that policy while
   respecting the server's purpose of sharing memory with other applications.
   Prior Qwen allocator experiments found only a small gain, not a universal win.
4. **Compile selected repeated work.** Upstream explicitly bypasses whole-predictor
   compilation on M1/M2 variants **other than Max/Ultra** for Krea 2, Z-Image,
   ERNIE, FLUX.2 and Ideogram 4. The helper explicitly exempts Max/Ultra; this
   M1 Ultra therefore uses compilation. Preserve that policy while testing
   smaller compiled units and amortized compilation.
   Compilation can fuse operations, but is not an unconditional speed switch.
   [MLX compilation documentation](https://ml-explore.github.io/mlx/build/html/usage/compile.html)
5. **Profile decoding and memory conversions.** Eight- or four-step models can
   spend a larger share outside denoising. Ideogram's `Fp8Linear` expands/scales
   stored FP8 weights on each call: carefully bounded reuse is worth profiling,
   but expanding all weights persistently can exceed the memory budget. Tiling
   usually trades memory against compute and can change normalization behavior
   for some VAEs; do not enable it universally as a speed optimization.
6. **Optimize attention data movement.** Investigate native GQA in Krea, avoid
   materialized all-valid/quadratic masks where equivalent, and prepare static
   masks once. Ideogram currently recreates its segment equality mask per layer.
   Keep padded/batched cases correct. Its explicit fp32 Q/K/V casts are a separate
   precision experiment, not an automatically safe removal.

Reducing steps, changing guidance, reusing changing hidden states (such as
TeaCache/FBCache-style approaches), changing attention precision, and switching
checkpoints are quality/behavior trade-offs. Keep those separately selectable
from exact-algorithm reuse. More simultaneous GPU requests or larger batches
also need independent latency/throughput/memory measurements.

## Implementation order and acceptance

1. Generalize the benchmark's stage/memory reporting and define prepared
   conditioning adapters with bounded lifetimes.
2. Implement and benchmark Krea 2 text preparation, output slicing and native GQA
   individually, then together, against the unchanged native model.
3. Repeat for Z-Image and FIBO, followed by the smaller opportunities in ERNIE,
   FLUX.1, Qwen 1.x, FLUX.2 and Ideogram.
4. Evaluate the separate FLUX.2 KV model and optional quality/performance modes.

For each change, test multiple prompts and sizes, padding, positive/negative
branches, img2img/reference paths where supported, changed model/quantization,
and cancellation/error cleanup. Measure cold and warm latency separately. Compare
native and optimized outputs through full denoising trajectories and decoded
images; kernel/layout changes may introduce normal floating-point differences.
Ship defaults only after meaningful end-to-end gains are demonstrated without
unacceptable output drift or memory growth.

## Source anchors

Inspected installed source paths below are relative to `mflux/models/` in the
pinned mflux 0.20.0 package. These are code findings; apart from the probes and
earlier Qwen results, their performance benefit has not been measured.

- Krea: `krea2/model/krea2_transformer/{transformer,text_fusion,transformer_block,attention,final_layer}.py`;
  `krea2/variants/txt2img/krea2.py::_predict`.
- Z-Image: `z_image/model/z_image_transformer/{transformer,context_block}.py`;
  `z_image/variants/z_image.py::_encode_prompts`.
- FIBO: `fibo/model/fibo_transformer/{transformer,joint_transformer_block}.py`.
- ERNIE: `ernie_image/model/ernie_transformer/transformer.py::get_pos_encoding`.
- FLUX.1: `flux/model/flux_transformer/transformer.py`; Qwen 1.x:
  `qwen/model/qwen_transformer/qwen_transformer_block.py`.
- FLUX.2: `common/config/model_config.py`,
  `flux2/variants/edit/flux2_klein_edit.py`,
  `flux2/model/flux2_transformer/{transformer,flux2_kv_cache}.py`.
- Ideogram: `ideogram4/model/ideogram4_transformer/{transformer,attention,fp8_linear}.py`;
  `ideogram4/variants/txt2img/ideogram4.py::_denoise_step` already skips unneeded
  negative passes at guidance 1. Preserve that existing optimization.
- Server: `server.py::process_image_task`, `generate_with_model`, `MODEL_REGISTRY`.
