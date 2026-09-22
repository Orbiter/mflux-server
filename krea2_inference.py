"""Request-prepared Krea 2 inference using the native mflux 0.20 components.

Only text fusion and positional frequencies are invariant. Joint transformer
states remain fully recomputed at every timestep; this is not prefix KV reuse.
The native generator still owns sampling, img2img, callbacks and VAE decoding.
"""
from contextlib import contextmanager
from dataclasses import dataclass

import mlx.core as mx
from mflux.models.krea2.model.krea2_transformer.rope_embedder import Krea2RopeEmbedder
from mflux.models.krea2.model.krea2_transformer.timestep_embedder import Krea2TimestepMLP
from mflux.utils.apple_silicon import AppleSiliconUtil

from inference_runtime import BoundedArrayCache, instance_override

PROMPT_CACHE_BYTES = 256 * 1024**2


@dataclass(frozen=True)
class Krea2Options:
    prepare_text: bool = True
    grouped_attention: bool = True
    image_only_output: bool = True


def grouped_attention(attention, x, freqs, mask=None):
    """Keep K/V at their original head count for MLX's native GQA kernel."""
    batch, length, _ = x.shape

    def project(layer, heads):
        return layer(x).reshape(batch, length, heads, attention.head_dim).transpose(0, 2, 1, 3)

    q = project(attention.wq, attention.heads)
    k = project(attention.wk, attention.kvheads)
    v = project(attention.wv, attention.kvheads)
    q, k = attention.qknorm(q, k)
    q, k = Krea2RopeEmbedder.apply_rope(q, k, freqs)
    attended = mx.fast.scaled_dot_product_attention(
        q.astype(v.dtype), k.astype(v.dtype), v, scale=attention.scale, mask=mask,
    ).transpose(0, 2, 1, 3).reshape(batch, length, -1)
    return attention.wo(attended * mx.sigmoid(attention.gate(x)))


class Krea2Runtime:
    """Model-owned executable; request arrays are explicit inputs, never captured."""

    def __init__(self, transformer, options=Krea2Options(), compile_step=None):
        self.transformer = transformer
        self.options = options
        if compile_step is None:
            compile_step = not AppleSiliconUtil.is_m1_or_m2()
        self.compiled = compile_step
        self.prepare = mx.compile(self.prepare_text) if compile_step else self.prepare_text
        self.step = mx.compile(self._step) if compile_step else self._step

    def prepare_text(self, context):
        model = self.transformer
        return model.txtmlp(model.txtfusion(model._unpack_context(context), mask=None))

    def frequencies(self, batch, text_length, height, width):
        rows, columns = mx.meshgrid(mx.arange(height, dtype=mx.float32),
                                   mx.arange(width, dtype=mx.float32), indexing="ij")
        positions = mx.stack((mx.zeros_like(rows), rows, columns), axis=-1).reshape(1, -1, 3)
        positions = mx.broadcast_to(positions, (batch, height * width, 3))
        text_positions = mx.zeros((batch, text_length, 3), dtype=mx.float32)
        return self.transformer.pe_embedder(mx.concatenate((text_positions, positions), axis=1))

    def _step(self, latents, timestep, text, frequencies):
        model, options = self.transformer, self.options
        batch, channels, original_h, original_w = latents.shape
        patch = model.patch
        padded = model._pad_to_multiple(latents, patch)
        height, width = padded.shape[-2:]
        rows, columns = height // patch, width // patch
        patches = padded.reshape(batch, channels, rows, patch, columns, patch)
        image = model.first(patches.transpose(0, 2, 4, 1, 3, 5).reshape(batch, rows * columns, -1))
        time = model.tmlp(Krea2TimestepMLP.timestep_embedding(timestep, model.tdim)[:, None, :].astype(image.dtype))
        modulation = model.tproj(time)
        if not options.prepare_text:
            text = self.prepare_text(text)
        text_length = text.shape[1]
        hidden = mx.concatenate((text, image), axis=1)
        for block in model.blocks:
            if options.grouped_attention and block.attn.heads != block.attn.kvheads:
                scale, shift, gate, ff_scale, ff_shift, ff_gate = block.mod(modulation)
                attention_input = (1 + scale) * block.prenorm(hidden) + shift
                hidden = hidden + gate * grouped_attention(block.attn, attention_input, frequencies)
                hidden = hidden + ff_gate * block.mlp((1 + ff_scale) * block.postnorm(hidden) + ff_shift)
            else:
                hidden = block(hidden, modulation, frequencies, None)
        if options.image_only_output:
            output = model.last(hidden[:, text_length:, :], time)
        else:
            output = model.last(hidden, time)[:, text_length:, :]
        output = output.reshape(batch, rows, columns, model.channels, patch, patch)
        output = output.transpose(0, 3, 1, 4, 2, 5).reshape(batch, model.channels, height, width)
        return output[:, :, :original_h, :original_w]

    def predictor(self, transformer, embeds, neg_embeds, guidance):
        if transformer is not self.transformer:
            raise ValueError("Krea runtime belongs to a different transformer")
        return PreparedKrea2(self, embeds, neg_embeds, guidance)


class PreparedKrea2:
    """Positive/negative conditioning and bounded geometry scoped to one request."""

    def __init__(self, runtime, positive, negative, guidance):
        self.runtime = runtime
        self.guidance = guidance
        prepare = runtime.prepare if runtime.options.prepare_text else lambda x: x
        self.positive = prepare(positive)
        self.negative = prepare(negative) if negative is not None else None
        mx.eval(self.positive)
        if self.negative is not None:
            mx.eval(self.negative)
        self.geometry = BoundedArrayCache(64 * 1024**2)

    def __call__(self, latents, timestep):
        def branch(text):
            patch = self.runtime.transformer.patch
            height, width = ((size + patch - 1) // patch for size in latents.shape[-2:])
            key = (latents.shape[0], text.shape[1], height, width)
            if key not in self.geometry:
                frequencies = self.runtime.frequencies(*key)
                mx.eval(frequencies)
                self.geometry[key] = frequencies
            else:
                frequencies = self.geometry[key]
            return self.runtime.step(latents, timestep, text, frequencies)

        positive = branch(self.positive)
        if self.negative is None:
            return positive
        negative = branch(self.negative)
        return negative + self.guidance * (positive - negative)

    def clear(self):
        self.positive = self.negative = None
        self.geometry.clear()


def bound_prompt_cache(model):
    if not isinstance(model.prompt_cache, BoundedArrayCache):
        bounded = BoundedArrayCache(PROMPT_CACHE_BYTES)
        bounded.update(model.prompt_cache)
        model.prompt_cache = bounded


@contextmanager
def optimized_krea2(model, enabled=True, options=Krea2Options()):
    """Adapt one serialized generation; restore native methods even on errors."""
    bound_prompt_cache(model)
    if not enabled:
        yield None
        return
    runtime = getattr(model, "_krea2_runtime", None)
    if runtime is None or runtime.transformer is not model.transformer or runtime.options != options:
        runtime = Krea2Runtime(model.transformer, options)
        model._krea2_runtime = runtime
    prepared = []

    def predict(*args, **kwargs):
        request = runtime.predictor(*args, **kwargs)
        prepared.append(request)
        return request

    try:
        with instance_override(model, "_predict", predict):
            yield runtime
    finally:
        for request in prepared:
            request.clear()
