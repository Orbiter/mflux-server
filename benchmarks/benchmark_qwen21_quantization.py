"""Compare real Qwen 2.1 generation at bf16, q8 and q4 in isolated processes.

Requires a complete local checkpoint. Does not start or change the live server.
Includes prompt/reference encoding and VAE decode, excludes HTTP and file writes.
"""
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--precision', choices=['all', 'bf16', 'q8', 'q4'], default='all')
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--sizes', type=int, nargs='+', default=[512, 1024])
    parser.add_argument('--edit-size', type=int, default=512)
    parser.add_argument('--allocator-mib', type=int, default=0)
    args = parser.parse_args()
    if not args.model_path.is_dir() or args.steps < 2 or args.repeats < 1 or args.allocator_mib < 0:
        parser.error('Supply a local checkpoint, at least two steps and one repeat, and a nonnegative cache size.')
    if any(size < 32 or size % 32 for size in [*args.sizes, args.edit_size]):
        parser.error('Image sizes must be positive multiples of 32.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    if args.precision == 'all':
        results = []
        for precision in ('bf16', 'q8', 'q4'):
            subprocess.run([
                sys.executable, str(Path(__file__).resolve()), '--model-path', str(args.model_path),
                '--output-dir', str(args.output_dir), '--precision', precision, '--steps', str(args.steps),
                '--repeats', str(args.repeats), '--sizes', *map(str, args.sizes),
                '--edit-size', str(args.edit_size), '--allocator-mib', str(args.allocator_mib),
            ], check=True)
            results.append(json.loads((args.output_dir / f'{precision}.json').read_text()))
        (args.output_dir / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        return

    import mlx.core as mx
    from PIL import Image, ImageDraw
    from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21
    from qwen21_edit import ReferenceEncoder
    from benchmarks.verify_qwen21_context import DEFAULT_PROMPT
    import server

    class Timings:
        def call_before_loop(self, latents, **kwargs):
            # Native prompt encoding is lazy. Materialize it before attributing
            # time to the denoising stage instead of counting it as step one.
            mx.eval(latents, model.prompt_cache)
            mx.synchronize()
            self.before = time.perf_counter()

        def call_after_loop(self, latents, **kwargs):
            mx.eval(latents)
            mx.synchronize()
            self.after = time.perf_counter()

    mx.set_cache_limit(args.allocator_mib * 1024**2)
    bits = {'bf16': None, 'q8': 8, 'q4': 4}[args.precision]
    print(f'Loading {args.precision}', flush=True)
    start = time.perf_counter()
    model = QwenImage21(quantize=bits, model_path=str(args.model_path))
    mx.eval(model.parameters())
    mx.synchronize()
    load_seconds = time.perf_counter() - start
    timings = Timings()
    model.callbacks.register(timings)
    results = {
        'precision': args.precision, 'device': mx.device_info()['device_name'],
        'python': platform.python_version(), 'macos': platform.mac_ver()[0],
        'mflux': importlib.metadata.version('mflux'), 'mlx': importlib.metadata.version('mlx'),
        'checkpoint': args.model_path.name, 'steps': args.steps, 'seed': 42,
        'guidance': 1.0, 'context_cache': True, 'allocator_mib': args.allocator_mib,
        'repeats': args.repeats, 'load_s': load_seconds, 'cases': [],
    }
    reference_paths = []
    for label, color in [('circle', 'red'), ('square', 'blue')]:
        path = args.output_dir / f'reference-{label}.png'
        image = Image.new('RGB', (256, 256), 'white')
        draw = ImageDraw.Draw(image)
        if label == 'circle':
            draw.ellipse((40, 40, 216, 216), fill=color)
        else:
            draw.rectangle((40, 40, 216, 216), fill=color)
        image.save(path)
        reference_paths.append(path)
    scenarios = [(f'text-{size}', size, []) for size in args.sizes]
    scenarios.append((f'edit2-{args.edit_size}', args.edit_size, reference_paths))
    for name, size, paths in scenarios:
        if paths:
            start = time.perf_counter()
            model._qwen21_reference_encoder = ReferenceEncoder(str(args.model_path))
            mx.synchronize()
            results['vision_load_s'] = time.perf_counter() - start
        prompt = ('Place the red circle from image 1 to the left of the blue square from image 2, '
                  'on one clean white background, with both shapes fully visible.') if paths else DEFAULT_PROMPT
        task = dict(seed=42, prompt=prompt, steps=args.steps, width=size, height=size,
                    guidance=1.0, negative_prompt=None)

        def run(steps):
            # Measure a newly encoded prompt each time, as with distinct user prompts.
            model.prompt_cache.clear()
            gc.collect()
            mx.clear_cache()
            mx.synchronize()
            mx.reset_peak_memory()
            started = time.perf_counter()
            generated = server.generate_with_model(model, 'qwen-image-2.1', {**task, 'steps': steps}, paths)
            mx.synchronize()
            finished = time.perf_counter()
            return generated, {
                'total_s': finished - started,
                'encode_s': timings.before - started,
                'denoise_s': timings.after - timings.before,
                'decode_s': finished - timings.after,
                'peak_mlx_bytes': mx.get_peak_memory(),
            }

        print(f'{args.precision}: warmup {name}', flush=True)
        _, warmup = run(2)
        case = {'name': name, 'size': size, 'references': len(paths), 'prompt': prompt,
                'warmup_2steps': warmup, 'samples': []}
        for repeat in range(args.repeats):
            image, sample = run(args.steps)
            case['samples'].append(sample)
            if repeat == 0:
                image.image.save(args.output_dir / f'{args.precision}-{name}.png')
            print(json.dumps({'precision': args.precision, 'case': name, 'repeat': repeat + 1, **sample}), flush=True)
        case['median'] = {key: statistics.median(sample[key] for sample in case['samples'])
                          for key in ('total_s', 'encode_s', 'denoise_s', 'decode_s')}
        case['peak_mlx_bytes'] = max(sample['peak_mlx_bytes'] for sample in case['samples'])
        results['cases'].append(case)
        (args.output_dir / f'{args.precision}.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
