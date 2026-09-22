"""Reference-image editing using mflux 0.20.0's native Qwen 2.1 components.

The language model, vision tower, transformer, VAE and scheduler remain mflux
implementations. This module supplies the missing multimodal input plumbing.
"""
import inspect
import json
import struct
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps

from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.loading.safetensors_reader import SafetensorsReader, SafetensorsTensorInfo
from mflux.models.common_models.qwen3_vl.qwen3_vl_vision_model import Qwen3VLVisionModel
from mflux.models.qwen21.latent_creator.qwen21_latent_creator import Qwen21LatentCreator
from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_prompt_encoder import Qwen21PromptEncoder
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.image_util import ImageUtil

from qwen21_edit_transformer import ReferenceTransformer, build_prefix, image_runs

MAX_REFERENCE_IMAGES = 10
REFERENCE_PIXEL_BUDGET = 1024 ** 2
REFERENCE_CACHE_BYTES = 3 * 1024 ** 3


def reference_size(size, pixel_budget):
    """Keep aspect ratio, avoid enlargement, and align both encoders to 32px."""
    width, height = size
    scale = min(1.0, (pixel_budget / (width * height)) ** 0.5)
    width, height = max(32, int(width * scale) // 32 * 32), max(32, int(height * scale) // 32 * 32)
    # Extremely thin images must still fit the area budget after rounding up.
    width = min(width, max(32, pixel_budget // height // 32 * 32))
    height = min(height, max(32, pixel_budget // width // 32 * 32))
    return width, height


def vision_pixels(image):
    """RGB over white; spatial 2x2 groups with two repeated temporal frames."""
    rgba = image.convert('RGBA')
    rgb = Image.new('RGB', image.size, 'white')
    rgb.paste(rgba, mask=rgba.getchannel('A'))
    h, w = image.height // 16, image.width // 16
    pixels = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1
    patches = pixels.reshape(3, h // 2, 2, 16, w // 2, 2, 16).transpose(1, 4, 2, 5, 0, 3, 6)
    patches = patches.reshape(h * w, 3, 16, 16)
    patches = np.repeat(patches[:, :, None], 2, axis=2).reshape(h * w, -1)
    return mx.array(patches).astype(ModelConfig.precision), mx.array([[1, h, w]])


def language_positions(ids, image_token, shapes):
    """Three-axis Qwen3-VL positions before the system prefix is removed."""
    runs = image_runs(np.asarray(ids) == image_token)
    if len(runs) != len(shapes):
        raise ValueError("Prompt contains unexpected Qwen image placeholders.")
    positions = np.empty((3, len(ids)), dtype=np.int32)
    cursor = position = 0
    for (start, end), (h, w) in zip(runs, shapes):
        h, w = h // 2, w // 2
        if end - start != h * w:
            raise ValueError("Qwen image placeholder size does not match its reference.")
        positions[:, cursor:start] = np.arange(position, position + start - cursor)
        position += start - cursor
        positions[0, start:end] = position
        positions[1, start:end] = position + np.repeat(np.arange(h), w)
        positions[2, start:end] = position + np.tile(np.arange(w), h)
        position += max(h, w)
        cursor = end
    positions[:, cursor:] = np.arange(position, position + len(ids) - cursor)
    return mx.array(positions[:, None])


class ReferenceEncoder:
    """Lazy-loaded vision weights; no request images or embeddings are retained."""

    def __init__(self, source):
        def resolve(name):
            local = Path(source).expanduser()
            return local / name if local.is_dir() else Path(hf_hub_download(source, name))

        config = json.loads(resolve('text_encoder/config.json').read_text())['vision_config']
        accepted = inspect.signature(Qwen3VLVisionModel).parameters
        self.vision = Qwen3VLVisionModel(**{key: value for key, value in config.items() if key in accepted})
        index = json.loads(resolve('text_encoder/model.safetensors.index.json').read_text())['weight_map']
        shards = sorted({file for name, file in index.items() if name.startswith('model.visual.')})
        weights = []
        for shard in shards:
            path = resolve('text_encoder/' + shard)
            with path.open('rb') as stream:
                header_size = struct.unpack('<Q', stream.read(8))[0]
                header = json.loads(stream.read(header_size))
                for name, metadata in header.items():
                    if name.startswith('model.visual.'):
                        tensor = SafetensorsTensorInfo(name, metadata['dtype'], tuple(metadata['shape']),
                                                      tuple(metadata['data_offsets']))
                        value = SafetensorsReader._read_tensor(path, header_size + 8, tensor)
                        name = name.removeprefix('model.visual.')
                        if name == 'patch_embed.proj.weight':
                            value = value.transpose(0, 2, 3, 4, 1)
                        weights.append((name, value))
        if not weights:
            raise ValueError("Qwen checkpoint has no vision weights for reference editing.")
        # inv_freq is computed by mflux, not a checkpoint parameter.
        weights.append(('rotary_pos_emb.inv_freq', self.vision.rotary_pos_emb.inv_freq))
        self.vision.load_weights(weights, strict=True)
        self.vision.set_dtype(ModelConfig.precision)
        mx.eval(self.vision.parameters())

    def images(self, model, paths, width, height):
        references, shapes, embeddings, stacks = [], [], [], []
        budget = min(width * height, REFERENCE_PIXEL_BUDGET) // len(paths)
        for path in paths:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert('RGBA')
                image = image.resize(reference_size(image.size, budget), Image.Resampling.LANCZOS)
            pixels, grid = vision_pixels(image)
            visual, deep = self.vision(pixels, grid, return_deepstack=True)
            visual = visual.astype(ModelConfig.precision)
            deep = [item.astype(ModelConfig.precision) for item in deep]
            mx.eval(visual, deep)
            embeddings.append(visual)
            stacks.append(deep)
            vae_pixels = mx.array(np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None] / 127.5 - 1)
            encoded = VAEUtil.encode(model.vae, vae_pixels.astype(ModelConfig.precision), model.tiling_config)
            latent = Qwen21LatentCreator.pack_latents(encoded, image.height, image.width)
            mx.eval(latent)
            references.append(latent)
            shapes.append((image.height // 16, image.width // 16))
        return references, shapes, mx.concatenate(embeddings), [mx.concatenate(items) for items in zip(*stacks)]

    @staticmethod
    def prompt(model, prompt, shapes, visual, deep):
        tokenizer = model.tokenizers['qwen21'].tokenizer
        image_token = tokenizer.convert_tokens_to_ids('<|image_pad|>')
        placeholders = ' '.join(
            f'<image{i + 1}><|vision_start|>' + '<|image_pad|>' * (h * w // 4) + '<|vision_end|>'
            for i, (h, w) in enumerate(shapes)
        )
        template = Qwen21PromptEncoder.PROMPT_TEMPLATE_T2I.format(placeholders + (prompt or ' '))
        ids = tokenizer(template, add_special_tokens=False)['input_ids']
        positions = language_positions(ids, image_token, shapes)
        mask = np.asarray(ids) == image_token
        slots = mx.array(np.flatnonzero(mask).astype(np.int32))
        encoder = model.text_encoder
        hidden = encoder.embed_tokens(mx.array([ids]))
        hidden = hidden.at[0, slots].add(visual - hidden[0, slots])
        rotary = encoder.rotary_emb(hidden, positions)
        sequence = mx.arange(len(ids))
        causal = (sequence[:, None] >= sequence[None, :])[None, None]
        for i, layer in enumerate(encoder.layers):
            hidden, _ = layer(hidden, causal, rotary)
            if i < len(deep):
                hidden = hidden.at[0, slots].add(deep[i])
            mx.eval(hidden)
        # This checkpoint consumes the last layer BEFORE final RMSNorm.
        drop = Qwen21PromptEncoder._system_prefix_length(model.tokenizers['qwen21'])
        return hidden[:, drop:], mask[drop:]


def generate_qwen21_references(model, image_paths, *, seed, prompt, num_inference_steps,
                               height, width, guidance=1.0, negative_prompt=None, use_cache=True):
    if not 1 <= len(image_paths) <= MAX_REFERENCE_IMAGES:
        raise ValueError(f"Qwen Image 2.1 requires 1–{MAX_REFERENCE_IMAGES} reference images.")
    started = time.monotonic()
    runtime = getattr(model, '_qwen21_reference_encoder', None)
    if runtime is None:
        runtime = ReferenceEncoder(model.model_config.model_name)
        model._qwen21_reference_encoder = runtime
    references, shapes, visual, deep = runtime.images(model, image_paths, width, height)
    prompts = [prompt]
    if guidance > 1 and negative_prompt:
        prompts.append(negative_prompt)
    branches = []
    config = Config(model_config=model.model_config, width=width, height=height, guidance=guidance,
                    num_inference_steps=num_inference_steps, scheduler='linear')
    callbacks = model.callbacks.start(seed=seed, prompt=prompt, config=config)
    try:
        for text in prompts:
            embeds, mask = runtime.prompt(model, text, shapes, visual, deep)
            prefix, segments = build_prefix(model.transformer, embeds, mask, references, shapes)
            mx.eval(prefix)
            branches.append(ReferenceTransformer(model.transformer, prefix, segments, (height // 16, width // 16),
                                                 REFERENCE_CACHE_BYTES // len(prompts) if use_cache else 0))
        del visual, deep, references, embeds
        latents = Qwen21LatentCreator.create_noise(seed, height, width)
        callbacks.before_loop(latents)
        for step in config.time_steps:
            latents = config.scheduler.scale_model_input(latents, step)
            noise = branches[0](step, config, latents)
            if len(branches) == 2:
                negative = branches[1](step, config, latents)
                noise = negative + guidance * (noise - negative)
            latents = config.scheduler.step(noise=noise, timestep=step, latents=latents)
            callbacks.in_loop(step, latents)
            mx.eval(latents)
        callbacks.after_loop(latents)
    except KeyboardInterrupt:
        raise StopImageGenerationException("Qwen reference editing interrupted.") from None
    finally:
        for branch in branches:
            branch.clear()
    unpacked = Qwen21LatentCreator.unpack_latents(latents, height, width)
    decoded = VAEUtil.decode(model.vae, unpacked, model.tiling_config)
    return ImageUtil.to_image(decoded_latents=decoded, config=config, seed=seed, prompt=prompt,
                              quantization=model.bits, generation_time=time.monotonic() - started,
                              negative_prompt=negative_prompt)
