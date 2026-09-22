"""Synchronized Metal A/B benchmark; no downloads and no server changes.

Use --model-path pointing to a cached Qwen Image 2.1 snapshot to benchmark the
real transformer weights. Otherwise a small random transformer is used. Inputs
are synthetic in either case; these are denoising timings, not full image times.
"""
import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx
from mlx import nn
from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
from qwen21_context import Qwen21ContextCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--quantize', type=int, choices=[4, 8], default=8)
    parser.add_argument('--text-lengths', type=int, nargs='+', default=[64, 256, 1024])
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--allocator-mib', type=int, default=0)
    args = parser.parse_args()
    if args.size < 16 or args.size % 16 or args.steps < 2 or args.repeats < 1 or min(args.text_lengths) < 1:
        parser.error('Use a positive multiple of 16 for size, at least 2 steps, and positive lengths/repeats.')
    if args.allocator_mib < 0:
        parser.error('Allocator size cannot be negative.')
    mx.set_cache_limit(args.allocator_mib * 1024**2)
    mx.random.seed(42)
    if args.model_path:
        from mflux.models.common.weights.loading.weight_loader import WeightLoader
        from mflux.models.common.weights.loading.weight_applier import WeightApplier
        from mflux.models.qwen21.weights.qwen21_weight_definition import Qwen21WeightDefinition
        component = next(c for c in Qwen21WeightDefinition.get_components() if c.name == 'transformer')
        weights = WeightLoader.load_single_local(component, args.model_path)
        model = Qwen21Transformer()
        WeightApplier.apply_and_quantize_single(weights, model, component, args.quantize)
        del weights
        context_dim = 4096
    else:
        context_dim = 512
        model = Qwen21Transformer(num_layers=4, num_attention_heads=4, attention_head_dim=128,
                                  context_in_dim=context_dim)
        model.set_dtype(mx.bfloat16)
        nn.quantize(model, bits=args.quantize)
    mx.eval(model.parameters())
    gc.collect()
    mx.clear_cache()
    print(json.dumps({'device': mx.device_info()['device_name'], 'real_weights': bool(args.model_path),
                      'parameters': vars(args)}, default=str), flush=True)
    config = Config(ModelConfig.qwen_image_21(), num_inference_steps=args.steps,
                    height=args.size, width=args.size, guidance=1.)
    for length in args.text_lengths:
        text = mx.random.normal((1, length, context_dim)).astype(mx.bfloat16)
        initial = mx.random.normal((1, (args.size // 16)**2, 64)).astype(mx.bfloat16)
        mx.eval(text, initial)
        cached = Qwen21ContextCache(model)

        def run(call):
            latent = initial
            start = time.perf_counter()
            for step in range(args.steps):
                noise = call(step, config, latent, text)
                latent = config.scheduler.step(noise, step, latent)
                mx.eval(latent)
            return time.perf_counter() - start, latent

        # Exclude graph compilation from steady state; report cold runs separately.
        cold_native, _ = run(model)
        cold_cached, _ = run(cached)
        samples = {'native': [], 'cached': []}
        peaks = {}
        outputs = {}
        for repetition in range(args.repeats):
            order = [('native', model), ('cached', cached)]
            if repetition % 2:
                order.reverse()
            for name, call in order:
                cached.clear()
                mx.clear_cache()
                mx.reset_peak_memory()
                seconds, outputs[name] = run(call)
                samples[name].append(seconds)
                peaks[name] = max(peaks.get(name, 0), mx.get_peak_memory())
        a, b = (outputs[name].astype(mx.float32) for name in ('native', 'cached'))
        rmse = mx.sqrt(mx.mean((a-b)**2))
        native_s, cached_s = (statistics.median(samples[name]) for name in ('native', 'cached'))
        print(json.dumps({'text_tokens': length, 'image_tokens': initial.shape[1],
                          'native_s': native_s, 'cached_s': cached_s, 'speedup': native_s/cached_s,
                          'cold_s': {'native': cold_native, 'cached': cold_cached},
                          'samples_s': samples, 'peak_bytes': peaks,
                          'relative_rmse': (rmse / mx.maximum(mx.sqrt(mx.mean(a*a)), 1e-8)).item(),
                          'max_abs_error': mx.max(mx.abs(a-b)).item()}), flush=True)
        cached.clear()
        del cached, outputs
        gc.collect()


if __name__ == '__main__':
    main()
