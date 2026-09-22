"""Exact-work reuse for ordinary FLUX.2 Klein 4B, without prefix KV caching.

Native mflux owns generation, reference preprocessing, sampling and decoding.
Only deterministic encoder results and compiled predictors are reused. Joint
text/reference states are updated at every denoising step by native mflux.
"""
from collections import OrderedDict
from contextlib import contextmanager, ExitStack
import hashlib
import weakref

import mlx.core as mx
import numpy as np
from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
from mflux.utils.apple_silicon import AppleSiliconUtil

from inference_runtime import BoundedArrayCache, instance_override

FLUX2_4B_NAMES = {'flux2-klein-4b', 'black-forest-labs/FLUX.2-klein-4B'}


def with_empty_references(predict):
    """mflux 0.20's edit predictor concatenates None for text-only requests."""
    def call(**kwargs):
        if kwargs.get('image_latents') is None:
            kwargs['image_latents'] = kwargs['latents'][:, :0]
            kwargs['image_latent_ids'] = kwargs['latent_ids'][:, :0]
        return predict(**kwargs)
    return call


class Flux2Runtime:
    """Loaded-model caches and compiled functions; no captured request tensors."""

    def __init__(self, model):
        self.transformer = model.transformer
        self.vae = model.vae
        self.text_encoder = model.text_encoder
        self.tokenizers = model.tokenizers
        self.prompts = BoundedArrayCache(128 * 1024**2)
        self.references = BoundedArrayCache(128 * 1024**2)
        self.prompt_hits = self.reference_hits = 0
        self.compiled = not AppleSiliconUtil.is_m1_or_m2()
        self._predict_factory = weakref.WeakMethod(model._predict)
        self._predictors = OrderedDict()

    def matches(self, model):
        return (self.transformer is model.transformer and self.vae is model.vae
                and self.text_encoder is model.text_encoder and self.tokenizers is model.tokenizers)

    def clear_inputs(self):
        self.prompts.clear()
        self.references.clear()

    def encode_prompt(self, native, *, prompt, negative_prompt, guidance):
        key = (prompt, negative_prompt, guidance)
        if key in self.prompts:
            self.prompt_hits += 1
            return self.prompts[key]
        result = native(prompt=prompt, negative_prompt=negative_prompt, guidance=guidance)
        mx.eval(result)
        self.prompts[key] = result
        return result

    def encode_reference(self, native, image):
        # Hash actual preprocessed pixels, not temporary filenames or image
        # indices. Native preprocessing and order-dependent position IDs stay
        # outside this cache. Tiled inputs are keyed by their actual tile data.
        pixels = np.asarray(image.astype(mx.float32))
        key = (image.shape, str(image.dtype), hashlib.sha256(pixels.tobytes()).digest())
        if key in self.references:
            self.reference_hits += 1
            return self.references[key]
        encoded = native(image)
        mx.eval(encoded)
        self.references[key] = encoded
        return encoded

    def predict(self, **kwargs):
        if kwargs.get('image_latents') is None:
            kwargs['image_latents'] = kwargs['latents'][:, :0]
            kwargs['image_latent_ids'] = kwargs['latent_ids'][:, :0]
        # Keep separate compiled executables for a bounded set of signatures.
        # All conditioning remains an explicit native predictor input.
        signature = tuple((name, (value.shape, str(value.dtype)) if isinstance(value, mx.array) else value)
                          for name, value in sorted(kwargs.items()))
        if signature not in self._predictors:
            while len(self._predictors) >= 4:
                self._predictors.popitem(last=False)
            factory = self._predict_factory()
            if factory is None:
                raise RuntimeError('The loaded FLUX model has been released')
            self._predictors[signature] = factory(self.transformer)
        self._predictors.move_to_end(signature)
        return self._predictors[signature](**kwargs)


@contextmanager
def optimized_flux2(model, enabled=True):
    """One serialized native generation with instance-local, reversible hooks."""
    if not isinstance(getattr(model, 'transformer', None), Flux2Transformer):
        yield None
        return
    native_predict = model._predict
    supported = (model.model_config.model_name == 'black-forest-labs/FLUX.2-klein-4B'
                 and not model.model_config.supports_kv_cache)
    if not enabled or not supported:
        with instance_override(model, '_predict', lambda transformer: with_empty_references(native_predict(transformer))):
            yield None
        return
    runtime = getattr(model, '_flux2_runtime', None)
    if runtime is None or not runtime.matches(model):
        runtime = Flux2Runtime(model)
        model._flux2_runtime = runtime
    native_encode, native_vae_encode = model._encode_prompt_pair, model.vae.encode

    def build_predict(transformer):
        if transformer is not runtime.transformer:
            raise ValueError('FLUX runtime belongs to a different transformer')
        return runtime.predict

    with ExitStack() as stack:
        stack.enter_context(instance_override(model, '_predict', build_predict))
        stack.enter_context(instance_override(model, '_encode_prompt_pair',
            lambda **kwargs: runtime.encode_prompt(native_encode, **kwargs)))
        stack.enter_context(instance_override(model.vae, 'encode',
            lambda image: runtime.encode_reference(native_vae_encode, image)))
        yield runtime
