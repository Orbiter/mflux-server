"""Offline native/cached FLUX.2 Klein 4B comparisons through the server adapter.

Measures real text-only and reference generation, cold and warm encoder caches,
complete four-step trajectories and decoded images. Does not start the server.
"""
import argparse
from contextlib import ExitStack
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sizes', type=int, nargs='+', default=[512, 1024])
    parser.add_argument('--references', type=int, nargs='+', default=[0, 2])
    parser.add_argument('--reference-size', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--quantize', type=int, choices=[4, 8])
    parser.add_argument('--cache-modes', nargs='+', choices=['cold', 'warm'], default=['cold', 'warm'])
    parser.add_argument('--allocator-mib', type=int, default=0)
    args = parser.parse_args()
    if not args.model_path.is_dir() or args.steps < 1 or args.repeats < 1 or args.allocator_mib < 0:
        parser.error('Supply a complete checkpoint, positive steps/repeats and a nonnegative allocator limit.')
    if any(s < 32 or s % 16 for s in [*args.sizes, args.reference_size]) or any(n < 0 or n > 4 for n in args.references):
        parser.error('Sizes must be multiples of 16 >=32; reference count must be 0–4.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    import mlx.core as mx
    import numpy as np
    from PIL import Image, ImageDraw
    from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit
    from inference_runtime import StageTimings, instance_override
    from flux2_inference import Flux2Runtime
    import server

    mx.set_cache_limit(args.allocator_mib * 1024**2)
    start = time.perf_counter()
    model = Flux2KleinEdit(model_path=str(args.model_path), quantize=args.quantize)
    mx.eval(model.parameters())
    mx.synchronize()
    runtime = Flux2Runtime(model)
    model._flux2_runtime = runtime
    results = dict(device=mx.device_info()['device_name'], python=platform.python_version(),
                   macos=platform.mac_ver()[0], mflux=importlib.metadata.version('mflux'),
                   mlx=importlib.metadata.version('mlx'), checkpoint=args.model_path.name,
                   quantize=args.quantize, steps=args.steps, guidance=1., compiled=runtime.compiled,
                   allocator_mib=args.allocator_mib, reference_size=args.reference_size,
                   load_s=time.perf_counter()-start, cases=[])

    class Loop:
        def call_before_loop(self, latents, **kwargs):
            mx.eval(latents)
            mx.synchronize()
            self.before = time.perf_counter()

        def call_after_loop(self, latents, **kwargs):
            mx.eval(latents)
            mx.synchronize()
            self.after = time.perf_counter()
            self.latents = latents

    loop = Loop()
    model.callbacks.register(loop)
    paths = []
    for i, color in enumerate(('red', 'blue', 'green', 'yellow')):
        image = Image.new('RGB', (args.reference_size, args.reference_size), 'white')
        draw = ImageDraw.Draw(image)
        margin = args.reference_size // 6
        draw.ellipse((margin, margin, args.reference_size-margin, args.reference_size-margin), fill=color)
        path = args.output_dir / f'reference-{i}.png'
        image.save(path)
        paths.append(path)

    def run(enabled, cache_mode, size, count, seed, steps):
        if cache_mode == 'cold':
            runtime.clear_inputs()
        loop.latents = None
        gc.collect()
        mx.clear_cache()
        mx.synchronize()
        mx.reset_peak_memory()
        timings = StageTimings()
        prompt = ('Arrange the colored circles from the reference images in a row on a clean white background.'
                  if count else 'A small ceramic teapot on a wooden table beside a window, morning light.')
        task = dict(seed=seed, prompt=prompt, width=size, height=size, steps=steps, guidance=1.)
        prompt_hits, reference_hits = runtime.prompt_hits, runtime.reference_hits
        started = time.perf_counter()
        with ExitStack() as stack:
            # These wrappers measure native encoder work; warm hits are reported
            # separately and any hash/lookup overhead remains in total/preloop.
            for owner, name, label in ((model, '_encode_prompt_pair', 'text_encoder_s'),
                                       (model.vae, 'encode', 'reference_vae_s'),
                                       (model.vae, 'decode_packed_latents', 'decode_s')):
                stack.enter_context(instance_override(owner, name, timings.wrap(label, getattr(owner, name))))
            stack.enter_context(instance_override(server, 'flux2_optimizations', enabled))
            generated = server.generate_with_model(model, 'flux2-klein-4b', task, paths[:count])
        mx.synchronize()
        sample = dict(total_s=time.perf_counter()-started, preloop_s=loop.before-started,
                      denoise_s=loop.after-loop.before,
                      **{k: timings.seconds.get(k, 0.) for k in ('text_encoder_s', 'reference_vae_s', 'decode_s')},
                      peak_mlx_bytes=mx.get_peak_memory(), seed=seed,
                      prompt_hits=runtime.prompt_hits-prompt_hits, reference_hits=runtime.reference_hits-reference_hits)
        return sample, np.asarray(loop.latents.astype(mx.float32)), generated.image

    for size in args.sizes:
        for count in args.references:
            case = dict(size=size, references=count, samples=[], warmups=[])
            for enabled in (False, True):
                sample, _, _ = run(enabled, 'cold', size, count, 41, min(args.steps, 2))
                case['warmups'].append(dict(optimized=enabled, **sample))
            for cache_mode in args.cache_modes:
                for repeat in range(args.repeats):
                    compared = {}
                    for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
                        sample, latent, image = run(enabled, cache_mode, size, count, 42+repeat, args.steps)
                        record = dict(optimized=enabled, cache=cache_mode, **sample)
                        case['samples'].append(record)
                        compared[enabled] = (latent, np.asarray(image).astype(np.float32), record)
                        if repeat == 0:
                            image.save(args.output_dir / f'{size}-ref{count}-{cache_mode}-{enabled}.png')
                        print(json.dumps(dict(size=size, references=count, **record)), flush=True)
                    base, actual = compared[False], compared[True]
                    actual[2]['latent_relative_rmse'] = float(np.sqrt(np.mean((base[0]-actual[0])**2)) /
                                                             max(np.sqrt(np.mean(base[0]**2)), 1e-8))
                    actual[2]['image_mean_abs_difference_255'] = float(np.mean(np.abs(base[1]-actual[1])))
                    actual[2]['image_max_abs_difference_255'] = float(np.max(np.abs(base[1]-actual[1])))
            case['medians'] = []
            for enabled in (False, True):
                for cache_mode in args.cache_modes:
                    samples = [s for s in case['samples'] if s['optimized'] == enabled and s['cache'] == cache_mode]
                    case['medians'].append(dict(optimized=enabled, cache=cache_mode,
                        **{k: statistics.median(s[k] for s in samples) for k in samples[0] if k.endswith('_s')}))
            results['cases'].append(case)
            (args.output_dir / 'results.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
