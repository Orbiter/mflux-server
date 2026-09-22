# Qwen Image 2.1 quantization on Metal

The benchmark in `benchmarks/benchmark_qwen21_quantization.py` compares bf16,
Q8 and Q4 using the actual server model adapter. Each precision runs in a fresh
process with the same local checkpoint. It neither starts nor changes the server.

## Results — 22 September 2026

**bf16 was fastest in every tested workload.** Q8 and Q4 reduced memory, but neither improved generation speed on this M1 Ultra. These are medians of three runs, in seconds; lower is better.

| Workload, 12 steps | bf16 | Q8 | Q4 |
|---|---:|---:|---:|
| 512×512 text-to-image | **15.50** | 18.22 (+17.5%) | 18.43 (+18.9%) |
| 1024×1024 text-to-image | **68.26** | 76.22 (+11.7%) | 77.41 (+13.4%) |
| 512×512, two-reference edit | **17.59** | 19.72 (+12.1%) | 20.24 (+15.1%) |

Peak allocated MLX memory, GiB (lower is better):

| Workload | bf16 | Q8 | Q4 |
|---|---:|---:|---:|
| 512×512 text-to-image | 33.69 | 27.48 | 24.17 |
| 1024×1024 text-to-image | 43.02 | 36.81 | 33.49 |
| 512×512, two-reference edit | 34.72 | 28.51 | 25.19 |

For the 1024×1024 case, the median stage times explain the overall result:

| Stage, seconds | bf16 | Q8 | Q4 |
|---|---:|---:|---:|
| Prompt encoding | 0.25 | 0.25 | 0.25 |
| Denoising, including context prefill | 61.63 | 72.38 | 73.40 |
| VAE decode and image conversion | 6.39 | 3.53 | 3.79 |

Quantized decoding was faster at 1024×1024, but denoising took longer and dominated the total. Independently calculated stage medians need not sum to the median total.

For this 64 GiB machine, keep quantization disabled when speed is the priority and memory is available. Q8 saves about 6.2 GiB of peak allocation; Q4 saves about 9.5 GiB. Q4 offered no speed advantage over Q8 in these cases. A workload that otherwise exceeds available RAM can have a different trade-off.

The server was stopped throughout. Precisions ran in order bf16, Q8, Q4; runs within a case were consecutive. No new system swap-outs were observed during the comparison, and macOS reported no recorded thermal/performance warnings. Small differences between Q8 and Q4 should not be generalized to every workload.

The three saved 512×512 text-to-image examples were visually checked: all produced coherent images, with composition changes across precisions. This is not a broad image-quality evaluation.

Raw timings, including all repetitions, warmups, stage times, memory peaks, checkpoint revision, software versions and one-off local model loading times: [measurement JSON](qwen21-quantization-results.json). Model loading was excluded from generation timing and used cached local files.

## Method

- Apple M1 Ultra, 64 GiB unified memory; Python 3.12, mflux 0.20.0, MLX 0.32.2.
- Context KV caching enabled for every precision, allocator cache zero (server default).
- Seed 42, guidance 1.0, no negative prompt, 12 sampling steps.
- 512×512 and 1024×1024 text-to-image, plus a 512×512 two-reference edit.
- A two-step warmup at each size, followed by three measured generations.
- The same prompt, seed, reference files and dimensions for every precision.
- Prompt cache cleared before each measurement. Reference inputs are re-encoded
  on every edit, matching server behavior. Text and vision encoders stay bf16.
- Metal synchronization at timing boundaries. The before-loop callback also
  materializes native lazy prompt embeddings to separate encoding from denoising.
- Generation time includes encoding, denoising, decoding and PIL conversion.
  Model/vision weight loading, HTTP, queueing and PNG file writes are excluded.
- Peak memory is MLX's peak allocated memory during a generation, including
  resident model tensors; it is not whole-process RSS or a minimum RAM requirement.

These are short performance workloads, not the model's recommended 40-step
quality setting. Quantization can change images even with the same seed. Twelve-step
times should not be multiplied directly to predict 40-step end-to-end latency:
encoding, cache prefill and VAE decoding have different costs from later steps.

## Reproduce

Keep other GPU workloads idle and use an already downloaded, complete checkpoint:

```sh
.venv/bin/python3.12 benchmarks/benchmark_qwen21_quantization.py \
  --model-path /path/to/Qwen-Image-2.1/snapshot \
  --output-dir /tmp/qwen21-quantization
```

The script sets Hugging Face offline mode. It writes per-precision and combined
JSON results, the input reference images, and the first generated image for each
case. `--precision bf16`, `--precision q8`, and `--precision q4` run individual
precisions; `--steps`, `--repeats`, `--sizes`, `--edit-size`, and `--allocator-mib`
allow further controlled comparisons. Defaults reproduce the method above.
