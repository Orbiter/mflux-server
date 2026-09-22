"""Request-local, exact-algorithm prefix reuse for mflux 0.20.0 Qwen Image 2.1.

The native model still owns loading, quantization, scheduling, text encoding and
VAE decoding. Only its transformer call is adapted during one generation. Text
has t=0 modulation and causal attention, so its layer K/V do not depend on the
sampled timestep or image latents. Never apply this to bidirectional models.
"""
from contextlib import contextmanager
from dataclasses import dataclass

import mlx.core as mx
from mlx import nn
from mflux.models.qwen21.model.qwen21_transformer.qwen21_attention import Qwen21Attention
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer


@dataclass
class _Prefix:
    embeddings: mx.array
    mask: mx.array | None
    geometry: tuple
    keys_values: tuple
    nbytes: int


class Qwen21ContextCache:
    """Single-worker inference adapter; conditioning lives for one request.

    Cache arrays are explicit compiled inputs/outputs, never tracing side effects.
    Positive and negative conditioning occupy separate entries. Padding and batch
    sizes other than one deliberately use the native implementation.
    """

    def __init__(self, transformer, max_bytes=512 * 1024**2):
        self.transformer = transformer
        self.max_bytes = max_bytes
        self.entries = []
        self.hits = 0
        self.misses = 0
        self.fallbacks = 0
        self._prefill = mx.compile(self._prefill_forward)
        self._decode = mx.compile(self._decode_forward)

    @property
    def cache_bytes(self):
        return sum(entry.nbytes for entry in self.entries)

    def clear(self):
        self.entries.clear()

    def __call__(self, t, config, hidden_states, encoder_hidden_states, encoder_hidden_states_mask=None):
        text = encoder_hidden_states
        mask = encoder_hidden_states_mask
        geometry_key = (config.height, config.width, hidden_states.shape, hidden_states.dtype)
        entry = next((entry for entry in self.entries if entry.embeddings is text
                      and entry.mask is mask and entry.geometry[0] == geometry_key), None)
        timestep = self.transformer._compute_timestep(t, config)
        times = mx.concatenate([timestep, mx.zeros((1,), dtype=timestep.dtype)])
        if entry is not None:
            self.hits += 1
            _, cos, sin = entry.geometry
            return self._decode(hidden_states, times, cos, sin, entry.keys_values)

        # Check once per conditioning branch; do not synchronize the GPU each step.
        size = 2 * len(self.transformer.transformer_blocks) * text.shape[1] * self.transformer.inner_dim * text.itemsize
        supported = (hidden_states.shape[0] == text.shape[0] == 1 and text.shape[1] > 0
                     and len(self.entries) < 2 and size + self.cache_bytes <= self.max_bytes)
        if supported and mask is not None:
            supported = bool(mx.all(mask).item())
        if not supported:
            self.fallbacks += 1
            # Native 0.20.0 keys geometry by lengths, without the padding mask.
            # A different mask with identical dimensions must not reuse that mask.
            self.transformer._geometry_cache.pop((text.shape[1], config.height // 16, config.width // 16), None)
            return self.transformer(t, config, hidden_states, text, mask)

        cos, sin = self.transformer.pos_embed(text.shape[1], config.height // 16, config.width // 16)
        output, keys_values = self._prefill(hidden_states, text, times, cos, sin)
        # Materialize once: caching a lazy graph would retain the first full step's
        # activations. Compact K/V also avoid retaining the image part of a view.
        mx.eval(output, keys_values)
        nbytes = sum(x.nbytes for pair in keys_values for x in pair)
        if self.cache_bytes + nbytes > self.max_bytes:
            self.fallbacks += 1
            return output  # A promoted dtype must not exceed the retention budget.
        self.entries.append(_Prefix(text, mask, (geometry_key, cos[text.shape[1]:], sin[text.shape[1]:]),
                                    keys_values, nbytes))
        self.misses += 1
        return output

    @staticmethod
    def _qkv(attention, x, cos, sin):
        shape = (*x.shape[:-1], attention.num_heads, attention.head_dim)
        q = attention.norm_q(attention.to_q(x).reshape(shape))
        k = attention.norm_k(attention.to_k(x).reshape(shape))
        q = Qwen21Attention._apply_rope(q, cos, sin).transpose(0, 2, 1, 3)
        k = Qwen21Attention._apply_rope(k, cos, sin).transpose(0, 2, 1, 3)
        v = attention.to_v(x).reshape(shape).transpose(0, 2, 1, 3)
        return q, k, v

    @staticmethod
    def _finish_block(block, x, attention_output, modulation):
        _, gate1, scale2, gate2 = modulation
        attended = attention_output.transpose(0, 2, 1, 3).reshape(x.shape)
        x = x + gate1 * block.attn.to_out[0](attended)
        return x + gate2 * block.img_mlp(block.img_norm2(x) * scale2)

    @staticmethod
    def _modulation(params):
        # Shared by all 32 layers: compute these elementwise terms once per step.
        scale1, gate1, scale2, gate2 = mx.split(params, 4, axis=-1)
        return 1 + scale1, nn.tanh(gate1), 1 + scale2, nn.tanh(gate2)

    def _conditioning(self, times):
        embedding = self.transformer.time_text_embed(times)
        return embedding, self.transformer.modulation(embedding)

    @staticmethod
    def _join_prefix(prefix, current):
        # Keep the native token-major backing layout for SDPA. Joining in head
        # order changes Metal's attention execution and increases bf16 drift.
        # Concatenate directly in token order, avoiding a second layout copy.
        return mx.concatenate([prefix.transpose(0, 2, 1, 3), current.transpose(0, 2, 1, 3)],
                              axis=1).transpose(0, 2, 1, 3)

    def _output(self, x, embedding):
        scale = self.transformer.norm_out.linear(nn.silu(embedding))[0:1, None, :]
        return self.transformer.proj_out(self.transformer.norm_out(x, scale))

    def _prefill_forward(self, image, text, times, cos, sin):
        model = self.transformer
        length = text.shape[1]
        embedding, modulation = self._conditioning(times)
        modulation = model._select_modulation_rows(modulation, length, image.shape[1])
        modulation = self._modulation(modulation)
        x = mx.concatenate([model.txt_in(text), model.img_in(image)], axis=1)
        cache = []
        for block in model.transformer_blocks:
            q, k, v = self._qkv(block.attn, block.img_norm1(x) * modulation[0], cos, sin)
            cache.append((mx.contiguous(k[:, :, :length]), mx.contiguous(v[:, :, :length])))
            scale = block.attn.head_dim ** -0.5
            prefix = mx.fast.scaled_dot_product_attention(q[:, :, :length], k[:, :, :length],
                                                         v[:, :, :length], scale=scale, mask='causal')
            target = mx.fast.scaled_dot_product_attention(q[:, :, length:], k, v, scale=scale)
            x = self._finish_block(block, x, mx.concatenate([prefix, target], axis=2), modulation)
        return self._output(x[:, length:], embedding), tuple(cache)

    def _decode_forward(self, image, times, cos, sin, cache):
        embedding, modulation = self._conditioning(times)
        # Broadcast one row instead of allocating modulation for every image token.
        modulation = self._modulation(modulation[0:1, None, :])
        x = self.transformer.img_in(image)
        for block, (prefix_k, prefix_v) in zip(self.transformer.transformer_blocks, cache):
            q, k, v = self._qkv(block.attn, block.img_norm1(x) * modulation[0], cos, sin)
            attended = mx.fast.scaled_dot_product_attention(
                q, self._join_prefix(prefix_k, k), self._join_prefix(prefix_v, v),
                scale=block.attn.head_dim ** -0.5,
            )
            x = self._finish_block(block, x, attended, modulation)
        return self._output(x, embedding)


@contextmanager
def cached_qwen21_context(model, enabled=True):
    """Restore the native transformer and release conditioning on every exit path."""
    original = getattr(model, 'transformer', None)
    if not enabled or type(original) is not Qwen21Transformer:
        yield None
        return
    adapter = getattr(model, '_qwen21_context_runtime', None)
    if not isinstance(adapter, Qwen21ContextCache) or adapter.transformer is not original:
        adapter = Qwen21ContextCache(original)
        model._qwen21_context_runtime = adapter
    adapter.clear()
    model.transformer = adapter
    try:
        yield adapter
    finally:
        model.transformer = original
        adapter.clear()
