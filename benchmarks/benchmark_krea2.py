"""Compare native and prepared Krea inference without starting the server.

Use --model-path for real generations (offline), or --synthetic for a reduced
random transformer feasibility benchmark. Synthetic timings are NOT Krea Turbo
generation timings. All runs include per-request text preparation; model loading
and output-file writes are excluded. Native sampler and VAE remain unchanged.
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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--model-path', type=Path)
    source.add_argument('--synthetic', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--quantize', type=int, choices=[4, 8])
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--sizes', type=int, nargs='+', default=[512, 1024])
    parser.add_argument('--modes', nargs='+', choices=['native', 'prepared', 'grouped', 'output', 'optimized'],
                        default=['native', 'optimized'])
    parser.add_argument('--allocator-mib', type=int, default=0)
    parser.add_argument('--prompt-cache', choices=['cold', 'warm'], default='cold')
    parser.add_argument('--init-image', type=Path)
    parser.add_argument('--guidance', type=float, default=1.)
    parser.add_argument('--prompt', default='A small ceramic teapot on a wooden table beside a window, morning light.')
    parser.add_argument('--negative-prompt', default='blur, distorted shapes')
    args = parser.parse_args()
    if args.steps < 2 or args.repeats < 1 or args.allocator_mib < 0:
        parser.error('Use at least two steps, one repeat and a nonnegative allocator limit.')
    if any(size < 32 or size % 16 for size in args.sizes):
        parser.error('Sizes must be positive multiples of 16, at least 32.')
    if args.model_path and not args.model_path.is_dir():
        parser.error('The checkpoint directory does not exist.')
    if args.init_image and (args.synthetic or not args.init_image.is_file()):
        parser.error('--init-image requires a real checkpoint and an existing image.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    import mlx.core as mx
    from mlx import nn
    from mflux.models.krea2 import Krea2
    from mflux.models.krea2.model.krea2_sampler import Krea2Sampler
    from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer
    from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition
    from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
    from mflux.utils.apple_silicon import AppleSiliconUtil
    from inference_runtime import StageTimings, instance_override
    from krea2_inference import Krea2Options, optimized_krea2

    mx.set_cache_limit(args.allocator_mib * 1024**2)
    mx.eval(QwenVAE.LATENTS_MEAN, QwenVAE.LATENTS_STD)
    load_start = time.perf_counter()
    if args.synthetic:
        # Small model with a substantial static text-fusion stage; not scaled
        # to infer the proportion of time this takes in the real checkpoint.
        mx.random.seed(42)
        model = Krea2.__new__(Krea2)
        nn.Module.__init__(model)
        model.transformer = Krea2Transformer(features=512, tdim=64, txtdim=256, heads=8,
                                             kvheads=2, layers=4, txtlayers=12, txtheads=4, txtkvheads=4)
        model.transformer.set_dtype(mx.bfloat16)
        if args.quantize:
            nn.quantize(model.transformer, bits=args.quantize,
                        class_predicate=Krea2WeightDefinition.quantization_predicate)
        model.prompt_cache = {}
        text = mx.random.normal((1, 128, 12 * 256)).astype(mx.bfloat16)
        negative = -text if args.guidance != 1 else None
        mx.eval(text, negative if negative is not None else text)
    else:
        model = Krea2(model_path=str(args.model_path), quantize=args.quantize)
    mx.eval(model.parameters())
    mx.synchronize()
    modes = {'native': None, 'prepared': Krea2Options(True, False, False),
             'grouped': Krea2Options(False, True, False), 'output': Krea2Options(False, False, True),
             'optimized': Krea2Options()}

    class Loop:
        def call_before_loop(self, latents, **kwargs):
            mx.eval(latents)
            mx.synchronize()
            self.start = time.perf_counter()

        def call_after_loop(self, latents, **kwargs):
            mx.eval(latents)
            mx.synchronize()
            self.end = time.perf_counter()
            self.latents = latents

    loop = Loop()
    if not args.synthetic:
        model.callbacks.register(loop)
    results = dict(synthetic=args.synthetic, device=mx.device_info()['device_name'],
                   python=platform.python_version(), macos=platform.mac_ver()[0],
                   mflux=importlib.metadata.version('mflux'), mlx=importlib.metadata.version('mlx'),
                   checkpoint=args.model_path.name if args.model_path else None,
                   quantize=args.quantize, steps=args.steps, repeats=args.repeats,
                   allocator_mib=args.allocator_mib, prompt_cache=args.prompt_cache,
                   guidance=args.guidance, prompt=args.prompt, negative_prompt=args.negative_prompt,
                   init_image=str(args.init_image) if args.init_image else None,
                   compiled=not AppleSiliconUtil.is_m1_or_m2(),
                   excluded_stages=['model_loading', 'HTTP', 'file_writes'] +
                                   (['text_encoder', 'VAE'] if args.synthetic else []),
                   synthetic_transformer=dict(features=512, layers=4, heads=8, kvheads=2,
                                              txtdim=256, txtlayers=12, text_tokens=128) if args.synthetic else None,
                   load_s=time.perf_counter() - load_start, cases=[])
    runtimes = {}

    def run(mode, size, seed, steps):
        if args.prompt_cache == 'cold':
            model.prompt_cache.clear()
        loop.latents = None
        gc.collect()
        mx.clear_cache()
        mx.synchronize()
        mx.reset_peak_memory()
        timings = StageTimings()
        if mode in runtimes:
            model._krea2_runtime = runtimes[mode]
        started = time.perf_counter()
        with optimized_krea2(model, enabled=mode != 'native', options=modes[mode] or Krea2Options()) as runtime, ExitStack() as stack:
            if runtime is not None:
                runtimes[mode] = runtime
            names = [('prepare_s', '_predict')]
            if not args.synthetic:
                names += [('encode_s', '_encode_prompts'), ('latent_input_s', '_prepare_latents'),
                          ('decode_s', '_decode_latents')]
            for label, name in names:
                stack.enter_context(instance_override(model, name, timings.wrap(label, getattr(model, name))))
            if args.synthetic:
                latents = mx.random.normal((1, 16, size // 8, size // 8), key=mx.random.key(seed)).astype(mx.bfloat16)
                sigmas = mx.linspace(1., 0., steps + 1)
                sampler = Krea2Sampler.make_stepper('er_sde', sigmas, seed)
                loop.call_before_loop(latents)
                predict = model._predict(model.transformer, text, negative, args.guidance)
                for i in range(steps):
                    v = predict(latents, sigmas[i:i+1])
                    latents = sampler.step(i, latents, v, latents - sigmas[i] * v)
                    mx.eval(latents)
                loop.call_after_loop(latents)
                generated = None
            else:
                generated = model.generate_image(seed=seed, prompt=args.prompt, negative_prompt=args.negative_prompt,
                                                 num_inference_steps=steps, height=size, width=size,
                                                 guidance=args.guidance, image_path=args.init_image,
                                                 image_strength=.4 if args.init_image else None)
        mx.synchronize()
        sample = dict(total_s=time.perf_counter() - started, **timings.seconds,
                      denoise_s=loop.end - loop.start - timings.seconds['prepare_s'],
                      peak_mlx_bytes=mx.get_peak_memory(), seed=seed,
                      prompt_cache_bytes=model.prompt_cache.nbytes)
        return sample, loop.latents, generated

    for size in args.sizes:
        case = dict(size=size, samples={mode: [] for mode in args.modes}, warmup={})
        for mode in args.modes:
            case['warmup'][mode] = run(mode, size, 41, 2)[0]
        for repeat in range(args.repeats):
            outputs = {}
            # Alternate order to reduce systematic thermal/order bias.
            for mode in (args.modes if repeat % 2 == 0 else args.modes[::-1]):
                sample, latents, generated = run(mode, size, 42 + repeat, args.steps)
                outputs[mode] = latents
                case['samples'][mode].append(sample)
                if generated is not None and repeat == 0:
                    generated.image.save(args.output_dir / f'{mode}-{size}.png')
                print(json.dumps(dict(size=size, mode=mode, **sample)), flush=True)
            if 'native' in outputs:
                expected = outputs['native'].astype(mx.float32)
                for mode, output in outputs.items():
                    error = output.astype(mx.float32) - expected
                    sample = case['samples'][mode][-1]
                    sample['latent_max_abs_error'] = mx.max(mx.abs(error)).item()
                    sample['latent_relative_rmse'] = (mx.sqrt(mx.mean(error**2)) /
                                                      mx.maximum(mx.sqrt(mx.mean(expected**2)), 1e-8)).item()
        case['median'] = {mode: {key: statistics.median(sample[key] for sample in samples)
                                 for key in samples[0] if key.endswith('_s')}
                          for mode, samples in case['samples'].items()}
        results['cases'].append(case)
        (args.output_dir / 'results.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
