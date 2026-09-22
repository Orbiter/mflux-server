"""Compare real-prompt denoising and decoded images using local Qwen 2.1 weights.

Components are loaded sequentially to avoid keeping the text encoder, transformer,
and VAE resident together. This measures denoising only, not end-to-end latency.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx
import numpy as np
from PIL import Image
from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.qwen21.latent_creator.qwen21_latent_creator import Qwen21LatentCreator
from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_prompt_encoder import Qwen21PromptEncoder
from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_text_encoder import Qwen21TextEncoder
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
from mflux.models.qwen21.model.qwen21_vae.qwen21_vae import Qwen21VAE
from mflux.models.qwen21.weights.qwen21_weight_definition import Qwen21WeightDefinition
from qwen21_context import Qwen21ContextCache

DEFAULT_PROMPT = (
    'A red ceramic teapot on an old wooden table in a sunlit kitchen. A small green potted plant stands beside the teapot. '
    'Soft morning light enters from the left, casting natural shadows across the grain of the wood. '
    'A white linen cloth lies folded at the edge of the table, beside a blue cup. In the background are pale yellow walls '
    'and a window looking into a quiet garden. Realistic photography with fine ceramic detail and warm, gentle colors.'
)


def load_component(root, name, cls, bits):
    component = next(c for c in Qwen21WeightDefinition.get_components() if c.name == name)
    weights = WeightLoader.load_single_local(component, root)
    model = cls()
    WeightApplier.apply_and_quantize_single(weights, model, component, bits)
    mx.eval(model.parameters())
    del weights
    gc.collect()
    mx.clear_cache()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prompt', default=DEFAULT_PROMPT)
    parser.add_argument('--quantize', type=int, choices=[4, 8], default=8)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--allocator-mib', type=int, default=512)
    args = parser.parse_args()
    if args.size < 16 or args.size % 16 or args.steps < 2 or args.allocator_mib < 0:
        parser.error('Use a positive multiple of 16 for size, at least 2 steps, and a nonnegative allocator size.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mx.set_cache_limit(args.allocator_mib * 1024**2)
    print('Encoding prompt', flush=True)
    encoder = load_component(args.model_path, 'text_encoder', Qwen21TextEncoder, args.quantize)
    tokenizer = TokenizerLoader.load_all(Qwen21WeightDefinition.get_tokenizers(), str(args.model_path))['qwen21']
    text, mask = Qwen21PromptEncoder.encode_prompt(args.prompt, {}, tokenizer, encoder)
    mx.eval(text, mask)
    del encoder, tokenizer
    gc.collect()
    mx.clear_cache()

    print('Comparing denoising', flush=True)
    model = load_component(args.model_path, 'transformer', Qwen21Transformer, args.quantize)
    cached = Qwen21ContextCache(model)
    config = Config(ModelConfig.qwen_image_21(), num_inference_steps=args.steps,
                    height=args.size, width=args.size, guidance=1.)
    initial = Qwen21LatentCreator.create_noise(args.seed, args.size, args.size)
    mx.eval(initial)
    results, times = {}, {}
    for name, call in [('native', model), ('cached', cached)]:
        latents = initial
        start = time.perf_counter()
        for step in range(args.steps):
            noise = call(step, config, latents, text, mask)
            latents = config.scheduler.step(noise, step, latents)
            mx.eval(latents)
        results[name] = latents
        times[name] = time.perf_counter() - start
    a, b = (results[name].astype(mx.float32) for name in ('native', 'cached'))
    relative_rmse = (mx.sqrt(mx.mean((a-b)**2))/mx.maximum(mx.sqrt(mx.mean(a*a)), 1e-8)).item()
    cached.clear()
    del cached, model, call
    gc.collect()
    mx.clear_cache()

    print('Decoding comparison images', flush=True)
    vae = load_component(args.model_path, 'vae', Qwen21VAE, args.quantize)
    pixels = {}
    for name, latent in results.items():
        unpacked = Qwen21LatentCreator.unpack_latents(latent, args.size, args.size)
        decoded = VAEUtil.decode(vae, unpacked, tiling_config=None)
        values = np.array(mx.clip(decoded[0]/2+0.5, 0, 1).astype(mx.float32)).transpose(1, 2, 0)
        pixels[name] = values
        Image.fromarray((values * 255).round().astype(np.uint8)).save(args.output_dir / (name + '.png'))
    mse = float(np.mean((pixels['native'] - pixels['cached'])**2))
    metrics = {'device': mx.device_info()['device_name'], 'prompt': args.prompt, 'text_tokens': text.shape[1],
               'steps': args.steps, 'size': args.size, 'seed': args.seed, 'quantize': args.quantize,
               'allocator_mib': args.allocator_mib, 'denoise_s': times, 'relative_latent_rmse': relative_rmse,
               'pixel_psnr_db': float(-10 * np.log10(max(mse, 1e-12)))}
    (args.output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
