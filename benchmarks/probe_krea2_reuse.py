"""Small Krea 2 correctness/attention probes, not full-model speed benchmarks.

Uses random native mflux components and synthetic inputs; downloads no weights
and changes no server configuration. Prints JSON for the architecture audit.
"""
import gc
import json
import statistics
import time

import mlx.core as mx
from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer


def probe_conditioning():
    mx.random.seed(42)
    model = Krea2Transformer(features=128, tdim=32, txtdim=64, heads=4, kvheads=2,
                             layers=2, txtlayers=3, txtheads=2, txtkvheads=2)
    model.set_dtype(mx.bfloat16)
    context = mx.random.normal((1, 9, 3 * 64)).astype(mx.bfloat16)
    images = [mx.random.normal((1, 16, 8, 8)).astype(mx.bfloat16) for _ in range(2)]
    times = [mx.array([value], dtype=mx.bfloat16) for value in (1., .2)]
    mx.eval(model.parameters(), context, images)
    expected = [model(x, t, context) for x, t in zip(images, times)]
    mx.eval(expected)
    fusion, projection = model.txtfusion, model.txtmlp
    prepared = projection(fusion(model._unpack_context(context)))
    mx.eval(prepared)
    try:
        # Substitute only the invariant stages; all denoising blocks stay native.
        model.txtfusion = lambda x, mask=None: prepared
        model.txtmlp = lambda x: x
        actual = [model(x, t, context) for x, t in zip(images, times)]
        mx.eval(actual)
    finally:
        model.txtfusion, model.txtmlp = fusion, projection
    errors = [float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())
              for a, b in zip(expected, actual)]
    assert all(error == 0 for error in errors), errors
    return {'test': 'prepared_text_fusion', 'random_small_model': True, 'max_abs_errors': errors}


def probe_gqa(length):
    mx.random.seed(42)
    q = mx.random.normal((1, 48, length, 128)).astype(mx.bfloat16)
    k = mx.random.normal((1, 12, length, 128)).astype(mx.bfloat16)
    v = mx.random.normal((1, 12, length, 128)).astype(mx.bfloat16)
    mx.eval(q, k, v)

    def run(expand):
        keys = mx.repeat(k, 4, axis=1) if expand else k
        values = mx.repeat(v, 4, axis=1) if expand else v
        return mx.fast.scaled_dot_product_attention(q, keys, values, scale=128 ** -.5)

    for expand in (True, False):
        mx.eval(run(expand))
    samples, outputs = {True: [], False: []}, {}
    for repeat in range(6):
        for expand in ((True, False) if repeat % 2 == 0 else (False, True)):
            mx.synchronize()
            start = time.perf_counter()
            outputs[expand] = run(expand)
            mx.eval(outputs[expand])
            samples[expand].append(time.perf_counter() - start)
    error = float(mx.max(mx.abs(outputs[True].astype(mx.float32) - outputs[False].astype(mx.float32))).item())
    return {'test': 'sdpa_gqa', 'synthetic_inputs': True, 'tokens': length,
            'expanded_median_ms': statistics.median(samples[True]) * 1000,
            'grouped_median_ms': statistics.median(samples[False]) * 1000,
            'expanded_samples_ms': [s * 1000 for s in samples[True]],
            'grouped_samples_ms': [s * 1000 for s in samples[False]], 'max_abs_error': error,
            'unexpanded_kv_bytes': k.nbytes + v.nbytes, 'expanded_kv_bytes': (k.nbytes + v.nbytes) * 4}


if __name__ == '__main__':
    mx.set_cache_limit(0)
    results = [probe_conditioning()]
    gc.collect()
    mx.clear_cache()
    for length in (1152, 4224):
        results.append(probe_gqa(length))
    print(json.dumps({'device': mx.device_info()['device_name'], 'results': results}, indent=2))
