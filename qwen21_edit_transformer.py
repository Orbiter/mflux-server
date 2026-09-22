"""Ordered multimodal conditioning for the native Qwen 2.1 transformer.

Text is causal; each reference image is internally bidirectional. All prefix
tokens use t=0, so their per-layer keys and values can be reused during sampling.
"""
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from qwen21_context import Qwen21ContextCache


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    shape: tuple[int, int] | None = None


def image_runs(mask):
    """Half-open runs of image slots in an unpadded, single-sample prompt."""
    edges = np.diff(np.pad(np.asarray(mask, dtype=np.int8), (1, 1)))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def build_prefix(model, text, image_mask, references, shapes):
    runs = image_runs(image_mask)
    if len(runs) != len(references) or len(shapes) != len(references):
        raise ValueError("Qwen reference images and prompt image slots do not match.")
    pieces, segments = [], []
    cursor = length = 0
    for (start, end), latent, shape in zip(runs, references, shapes):
        if (end - start) * 4 != shape[0] * shape[1] or latent.shape[1] != shape[0] * shape[1]:
            raise ValueError("Qwen vision and VAE image dimensions do not match.")
        if start > cursor:
            pieces.append(model.txt_in(text[:, cursor:start]))
            segments.append(Segment(length, length + start - cursor))
            length += start - cursor
        pieces.append(model.img_in(latent))
        segments.append(Segment(length, length + latent.shape[1], shape))
        length += latent.shape[1]
        cursor = end
    if cursor < text.shape[1]:
        pieces.append(model.txt_in(text[:, cursor:]))
        segments.append(Segment(length, length + text.shape[1] - cursor))
    return mx.concatenate(pieces, axis=1), tuple(segments)


def rotary_positions(rope, segments):
    axes, position = [], 0
    for segment in segments:
        count = segment.end - segment.start
        if segment.shape is None:
            axis = np.arange(position, position + count)
            axes.append(np.stack([axis, axis, axis], axis=1))
            position += count
        else:
            h, w = segment.shape
            rows = np.arange(-(h - h // 2), h // 2)
            cols = np.arange(-(w - w // 2), w // 2)
            axes.append(np.stack([np.full(count, position), np.repeat(rows, w), np.tile(cols, h)], axis=1))
            position += max(h, w)
    positions = np.concatenate(axes).astype(np.float32)
    # Computing from positions also avoids the native table's 8192-token bound.
    angles = [positions[:, i:i + 1] / rope.theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim)
              for i, dim in enumerate(rope.axes_dim)]
    angles = np.concatenate(angles, axis=1)
    return mx.array(np.cos(angles)), mx.array(np.sin(angles))


class ReferenceTransformer(Qwen21ContextCache):
    """One prompt branch of one request; never retain reference data on the model."""

    def __init__(self, transformer, prefix, segments, target_shape, max_bytes):
        super().__init__(transformer, max_bytes=max_bytes)
        self.prefix = prefix
        self.length = prefix.shape[1]
        self.segments = segments
        target = Segment(self.length, self.length + target_shape[0] * target_shape[1], target_shape)
        self.cos, self.sin = rotary_positions(transformer.pos_embed, (*segments, target))
        size = 2 * len(transformer.transformer_blocks) * self.length * transformer.inner_dim * prefix.itemsize
        self.retain = size <= max_bytes
        self.keys_values = None

    def clear(self):
        self.keys_values = None

    def __call__(self, t, config, image):
        timestep = self.transformer._compute_timestep(t, config)
        times = mx.concatenate([timestep, mx.zeros((1,), dtype=timestep.dtype)])
        if self.keys_values is not None:
            return self._decode(image, times, self.cos[self.length:], self.sin[self.length:], self.keys_values)
        embedding, params = self._conditioning(times)
        modulation = self._modulation(self.transformer._select_modulation_rows(params, self.length, image.shape[1]))
        x = mx.concatenate([self.prefix, self.transformer.img_in(image)], axis=1)
        cache = []
        for block in self.transformer.transformer_blocks:
            q, k, v = self._qkv(block.attn, block.img_norm1(x) * modulation[0], self.cos, self.sin)
            if self.retain:
                cache.append((mx.contiguous(k[:, :, :self.length]), mx.contiguous(v[:, :, :self.length])))
            attended = []
            for segment in self.segments:
                attended.append(mx.fast.scaled_dot_product_attention(
                    q[:, :, segment.start:segment.end], k[:, :, :segment.end], v[:, :, :segment.end],
                    scale=block.attn.head_dim ** -0.5, mask='causal' if segment.shape is None else None,
                ))
            attended.append(mx.fast.scaled_dot_product_attention(
                q[:, :, self.length:], k, v, scale=block.attn.head_dim ** -0.5,
            ))
            x = self._finish_block(block, x, mx.concatenate(attended, axis=2), modulation)
            # Bound temporary activations for long image prefixes on unified memory.
            mx.eval(x, cache)
        if self.retain:
            self.keys_values = tuple(cache)
        return self._output(x[:, self.length:], embedding)
