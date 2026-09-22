# FLUX.2 Klein 4B performance

FLUX.2 Klein 4B is the default server model: four steps, guidance 1, no
quantization unless requested. Its official checkpoint downloads without
Hugging Face authentication. The server supports text-to-image and up to four
ordered reference images through the native `Flux2KleinEdit` generator.

## Implemented optimizations

- Model-owned prompt embeddings: a 128 MiB / 16-entry LRU.
- Model-owned deterministic VAE reference encodings: a separate 128 MiB /
  16-entry LRU, keyed by the preprocessed pixel tensor's content, shape and
  dtype. Filenames and attachment order are not cache keys; native reference
  preprocessing and order-dependent positional IDs are recomputed normally.
- Native prediction functions reused for at most four input signatures. All
  request arrays remain explicit arguments. Upstream decides compilation support;
  the M1 Ultra used here compiles, while M1/M2 variants other than Max/Ultra do not.
- An event wakes the serial worker when a generation or model-load request
  arrives, removing the former one-second polling delay for **all models**.

Tensor budgets describe logical payload, excluding backing buffers and runtime
overhead. Cache ownership follows the loaded model; switching models creates
new caches. Encoder results are materialized before retention. Hooks are local
to the model instance and restored on failure as well as success. Nothing is
patched globally in mflux. Native denoising, precision, scheduler and VAE decode
are preserved. There is no per-layer prefix KV reuse for the ordinary 4B model.

`--no-flux2-optimizations` disables the three model-specific caches for comparison.
`/api/info` lists `prompt_cache`, `reference_encode_cache` and `predictor_cache`
when enabled for either registered 4B alias. The 9B model retains its native path.
Both paths fix absent references by passing empty tensors to the native editing
predictor; mflux 0.20 otherwise attempts to concatenate `None` for text-only
requests. Tests compare that fix with the separate native text-to-image predictor.

## Real-checkpoint measurements

Apple M1 Ultra, mflux 0.20.0, MLX 0.32.2, BF16, four inference steps,
guidance 1 and Metal allocator cache zero. Checkpoint revision
`e7b7dc27f91deacad38e78976d1f2b499d76a294`. Reference cases use two 256×256
images. Each path is warmed for two steps; native/optimized order alternates
over three repeats with seeds 42–44. Cold runs clear encoder caches; warm runs
retain them. Timings include prompt/reference encoding, sampling, VAE decoding
and PIL conversion, with synchronized stage boundaries. Loading weights, HTTP,
queue delay and output-file writes are excluded.

An existing server with the same model loaded was present during measurement;
its queue was empty when checked. Other processes and system load can affect
timings. The additional confirmation run below addresses variability in the
512px editing group. These measurements do not establish minimum system RAM.

| Workload | Cold native | Cold optimized | Warm native | Warm optimized |
|---|---:|---:|---:|---:|
| Text-to-image, 512×512 | 4.045 s | 4.138 s | 3.990 s | 3.685 s |
| Two references, 512×512 | 5.377 s | 5.376 s | 5.370 s | 4.931 s |
| Text-to-image, 1024×1024 | 13.462 s | 13.544 s | 13.469 s | 13.083 s |
| Two references, 1024×1024 | 14.934 s | 15.017 s | 14.883 s | 14.429 s |

There is **no demonstrated cold-request speedup**: measured changes range from
approximately unchanged to 2.3% slower. The benefit comes from repeated prompts
and reference images. Warm text-only medians improved by 7.7% at 512px and 2.9%
at 1024px; the 1024px edit median improved by 3.1%. The warm 512px editing
values in the table come from a separate five-repeat confirmation with seeds
42–46: **8.2% less time**, native range 5.346–5.382 s, optimized 4.915–4.944 s.
The original three-repeat editing group was noisier; its apparent 22% gain
was not reproduced. [Confirmation samples](flux2-warm-confirmation-results.json).

All **29 comparison pairs** (24 initial pairs and five confirmation pairs)
produced bit-identical final latents and decoded images. Observed peak MLX memory was
about 17.59 GiB at 512px and 24.29 GiB at 1024px, with little change between paths.
[Raw samples and stage timings](flux2-performance-results.json).

## Reproduction and validation

The benchmark uses a complete local checkpoint, offline, without starting or
reconfiguring the server:

```sh
.venv/bin/python3.12 benchmarks/benchmark_flux2.py \
  --model-path /path/to/FLUX.2-klein-4B/snapshot \
  --output-dir /tmp/flux2-benchmark --sizes 512 1024 --references 0 2 --repeats 3
```

Use `--references 4` for four-image testing, `--reference-size 512` for larger
inputs, or `--cache-modes warm --repeats 5` for repeated-input measurements.
`--quantize 4` / `--quantize 8` support separate precision comparisons; the
timings above concern BF16 only. Do not infer a quantization speedup from them.

Component tests cover native four-step trajectories in BF16/Q4/Q8, compiled
and uncompiled, CFG, missing references, changed prompts/content/positions,
bounded cache eviction, model lifetime, exception cleanup and server defaults.
Worker tests cover wakeup on submission, draining model-load and generation
backlogs, and a request arriving as the worker becomes idle.

A separate live HTTP check started the server on port 4031 without a `--model`
argument, verified the default model and `/api/info`, and completed two text-only
and two four-reference generations through the actual queue/worker. Repeated
requests returned identical PNG hashes. Measured idle queue delays were
0.2–0.5 ms. The test server was stopped afterward; the user's existing server
was left running. [HTTP check results](flux2-http-results.json).

The demo page's heading and browser title are both
`MFLUX Server - Image Generation`. Restart an existing server process to load
the final backend changes; the HTML is read from disk when the page reloads.
