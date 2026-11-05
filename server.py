# mflux-server
# Server for image generation with mflux (https://github.com/filipstrand/mflux)
# (C) 2024 by @orbiter Michael Peter Christen
# This code is licensed under the Apache License, Version 2.0

import os
import io
import gc
import json
import time
import base64
import hashlib
import argparse
import threading
import binascii
import math
from contextlib import contextmanager, ExitStack
from dataclasses import asdict
from importlib.metadata import version as package_version
from tempfile import TemporaryDirectory
import mlx.core as mx
from mlx.core import metal as metal_compat
from PIL import Image, ImageOps, PngImagePlugin
from tqdm import tqdm
from pathlib import Path
from flask import Flask, request, Response, jsonify
from flask_restx import Api, Resource, fields
from flask_cors import CORS
from flask import send_file, redirect
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError
from mflux.models.common.config import ModelConfig
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21
from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit
from mflux.models.qwen21.variants.controlnet.qwen_image_21_controlnet import QwenImage21Controlnet
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
from mflux.models.anima import Anima
from mflux.models.iris.variants import Iris
from mflux.models.ming_image import MingImage
from mflux.models.krea2.variants.controlnet.krea2_controlnet import Krea2Controlnet
from mflux.models.common.checking.model_checker import ModelChecker
from mflux.models.common.resolution.model_revisions import ModelRevisions
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.models.fibo.variants.txt2img.fibo import FIBO
from mflux.models.fibo2.variants import Fibo2, Fibo2Edit
from mflux.models.fibo2.variants.edit.fibo2_edit_images import Fibo2EditImages
from mflux.models.fibo2.model.fibo2_text_encoder.fibo2_caption_check import Fibo2CaptionCheck
from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit
from mflux.models.flux3.variants import Flux3Klein, Flux3KleinEdit
from mflux.models.flux3.latent_creator.flux3_reference_images import MAX_REFERENCES as FLUX3_MAX_REFERENCE_IMAGES
from mflux.models.ernie_image import ErnieImage
from mflux.models.krea2 import Krea2
from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition
from mflux.models.moss_image import MOSSImage
from mflux.models.moss_image.latent_creator import MOSSImageLatentCreator
from mflux.models.moss_image.model.moss_image_scheduler import MOSSImageScheduler
from mflux.models.ideogram4 import Ideogram4
from mflux.models.ideogram4.latent_creator import Ideogram4LatentCreator
from mflux.models.ideogram4.model.ideogram4_scheduler import Ideogram4Scheduler
from mflux.models.ideogram4.model.ideogram4_text_encoder import Ideogram4PromptEncoder
from mflux.models.z_image.variants.z_image import ZImage
from qwen21_context import cached_qwen21_context
from qwen21_edit import MAX_REFERENCE_IMAGES, generate_qwen21_references
from krea2_inference import optimized_krea2
from flux2_inference import FLUX2_4B_NAMES, optimized_flux2
from inference_runtime import instance_override
from model_capabilities import (extend_registry, capabilities, resolve_entry, validate_load_options,
                                validate_parameters, discover_saved_models, GENERATION_FIELDS,
                                IMAGE_FIELDS, LOAD_FIELDS)

# These class constants contain lazy reshape operations created during import.
# Materialize them on their owning thread before Qwen/Krea decode on the worker.
mx.eval(QwenVAE.LATENTS_MEAN, QwenVAE.LATENTS_STD)

import requests
try:
    from huggingface_hub.errors import GatedRepoError
except Exception:
    GatedRepoError = None

# monkey pathing the Session to ignore SSL verification
old_request = requests.Session.request
def new_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return old_request(self, *args, **kwargs)
requests.Session.request = new_request

def _is_gated_repo_error(exc: Exception) -> bool:
    if GatedRepoError is not None and isinstance(exc, GatedRepoError):
        return True
    message = str(exc).lower()
    return "gatedrepoerror" in message or "gated repo" in message or "access to model" in message and "restricted" in message

def _hf_cache_root() -> str:
    cache_dir = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if cache_dir:
        return cache_dir
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return os.path.join(hf_home, "hub")
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

def _hf_repo_cache_path(repo_id: str) -> str:
    return os.path.join(_hf_cache_root(), f"models--{repo_id.replace('/', '--')}")

def _hf_repo_cached(repo_id: str) -> bool:
    try:
        return os.path.isdir(_hf_repo_cache_path(repo_id))
    except Exception:
        return False

app = Flask(__name__)
api = Api(app, version='1.0', title='MFLUX API Server',
          description='An image generation server. Workflow: /generate -> /status -> /image',
          doc='/swagger',
          prefix='/api')

CORS(app, resources={r"/*": {"origins": "*"}})

apppath = os.path.dirname(__file__)
tasklist = []         # list which holds the image computation tasks
model_instance = None # the model object, initialized in main()
pixels = 1024 * 1024  # the number of pixels in all of the computed images (start value)
ctime = 120           # initial estimate: 120 seconds per 1024×1024 image
metal_cache_limit = 0 # the cache limit for the metal library
qwen21_context_cache = True
krea2_optimizations = True
flux2_optimizations = True
DEFAULT_MODEL = "flux-3-image-klein"
model = DEFAULT_MODEL
model_quantize = None # quantization level in use
model_options = {}
models_directory = os.environ.get('MFLUX_MODELS_DIR', '')
model_lock = threading.Lock()
model_load_requests = []
model_worker_thread = None
worker_wakeup = threading.Event()
IDEOGRAM4_DEFAULT_PRESET = "V4_DEFAULT_20"
IDEOGRAM4_PRESETS = {
    name: preset.num_steps for name, preset in Ideogram4Scheduler.PRESETS.items()
}
MODEL_REGISTRY = {
    "dev": {"loader": "flux", "steps": 25},
    "dhairyashil/FLUX.1-dev-mflux-4bit": {"loader": "flux", "steps": 25},
    "schnell": {"loader": "flux", "steps": 4},
    "dhairyashil/FLUX.1-schnell-mflux-v0.6.2-4bit": {"loader": "flux", "steps": 4},
    "krea-dev": {"loader": "flux", "steps": 25},
    "filipstrand/FLUX.1-Krea-dev-mflux-4bit": {"loader": "flux", "steps": 25},
    "qwen": {"loader": "qwen", "steps": 25},
    **{
        name: {"loader": "qwen21", "steps": 40, "guidance": 1.0}
        for name in ("qwen-image-2.1", "qwen-2.1", "qwen-image-21", "Qwen/Qwen-Image-2.1")
    },
    "filipstrand/Qwen-Image-mflux-6bit": {"loader": "qwen", "steps": 25, "quantize": 6},
    "fibo": {"loader": "fibo", "steps": 25},
    "briaai/Fibo-mlx-4bit": {"loader": "fibo", "steps": 25},
    "briaai/Fibo-mlx-8bit": {"loader": "fibo", "steps": 25},
    **{
        name: {"loader": "fibo2", "steps": 50, "guidance": 5.0}
        for name in ("fibo2", "fibo-2.0-dev", "briaai/Fibo-2.0-dev")
    },
    **{
        name: {"loader": "fibo2", "steps": 4, "guidance": 1.0}
        for name in ("fibo2-edit-turbo", "fibo-2-edit-turbo", "fibo-2.0-edit-turbo",
                     "briaai/Fibo-2.0-edit-turbo")
    },
    "z-image-turbo": {"loader": "z-image", "steps": 9},
    "filipstrand/Z-Image-Turbo-mflux-4bit": {"loader": "z-image", "steps": 9},
    "flux2-klein-9b": {"loader": "flux2", "steps": 4, "guidance": 1.0},
    "black-forest-labs/FLUX.2-klein-9B": {"loader": "flux2", "steps": 4, "guidance": 1.0},
    "flux2-klein-4b": {"loader": "flux2", "steps": 4, "guidance": 1.0},
    "black-forest-labs/FLUX.2-klein-4B": {"loader": "flux2", "steps": 4, "guidance": 1.0},
    **{
        name: {"loader": "flux3", "steps": 16, "guidance": 1.0}
        for name in ("flux3-image-klein", "flux3-klein", "flux3", "flux-3-image-klein",
                     "black-forest-labs/flux-3-image-klein")
    },
    **{
        name: {"loader": "flux3", "steps": 50, "guidance": 3.5}
        for name in ("flux3-image-klein-base", "flux3-klein-base", "flux3-base", "flux-3-image-klein-base",
                     "black-forest-labs/flux-3-image-klein-base")
    },
    "ernie-image-turbo": {"loader": "ernie", "steps": 8, "guidance": 1.0},
    "baidu/ERNIE-Image-Turbo": {"loader": "ernie", "steps": 8, "guidance": 1.0},
    "ernie-image": {"loader": "ernie", "steps": 50, "guidance": 4.0},
    "baidu/ERNIE-Image": {"loader": "ernie", "steps": 50, "guidance": 4.0},
    "krea2": {"loader": "krea2", "steps": 8, "guidance": 1.0},
    "krea-2": {"loader": "krea2", "steps": 8, "guidance": 1.0},
    "krea-2-turbo": {"loader": "krea2", "steps": 8, "guidance": 1.0},
    "krea/Krea-2-Turbo": {"loader": "krea2", "steps": 8, "guidance": 1.0},
    **{
        name: {"loader": "moss-image", "steps": 50, "guidance": 3.0}
        for name in ("moss-image-sft", "moss-image", "moss", "OpenMOSS-Team/MOSS-Image-SFT",
                     "moss-image-base", "moss-base", "OpenMOSS-Team/MOSS-Image-Base")
    },
    **{
        name: {
            "loader": "ideogram4",
            "steps": IDEOGRAM4_PRESETS[IDEOGRAM4_DEFAULT_PRESET],
            "preset": IDEOGRAM4_DEFAULT_PRESET,
            "presets": IDEOGRAM4_PRESETS,
        }
        for name in ("ideogram4", "ideogram-4-fp8", "ideogram-ai/ideogram-4-fp8")
    },
}
extend_registry(MODEL_REGISTRY)

FLUX2_NAME_MAP = {
    "black-forest-labs/flux.2-klein-9b": "flux2-klein-9b",
    "black-forest-labs/flux.2-klein-4b": "flux2-klein-4b"
}

ERNIE_CONFIG_MAP = {
    "ernie-image-turbo": ModelConfig.ernie_image_turbo,
    "baidu/ernie-image-turbo": ModelConfig.ernie_image_turbo,
    "ernie-image": ModelConfig.ernie_image,
    "baidu/ernie-image": ModelConfig.ernie_image
}

def _normalize_flux2_model_name(model_name: str) -> str:
    normalized = model_name.strip()
    mapped = FLUX2_NAME_MAP.get(normalized.lower())
    return mapped or normalized

def image_capabilities(model_name: str):
    """Input limits for the adapters implemented by this server."""
    info = capabilities(model_name, MODEL_REGISTRY[model_name])
    return {key: info[key] for key in ('edit', 'multi_image_edit', 'max_init_images')}


def validate_image_count(model_name: str, count: int):
    limit = image_capabilities(model_name)["max_init_images"]
    if count > limit:
        raise ValueError(f"{model_name} accepts at most {limit} input image(s); received {count}.")


def decode_input_images(args, model_name: str):
    """Validate the complete request before decoding images into queue-owned copies."""
    encoded_images = args.get("init_images", [])
    if not isinstance(encoded_images, list):
        raise ValueError("init_images must be an array of encoded images.")
    single = args.get("init_image")
    if single is not None and not isinstance(single, str):
        raise ValueError("init_image must be an encoded image string.")
    if encoded_images and single:
        raise ValueError("Specify either init_image or init_images.")
    entries = [(f"init_images[{i}]", value) for i, value in enumerate(encoded_images)]
    if single:
        entries = [("init_image", single)]
    validate_image_count(model_name, len(entries))
    return [decode_encoded_image(value, field) for field, value in entries]


def decode_encoded_image(value, field):
    try:
        if not isinstance(value, str) or not value.strip():
            raise ValueError('expected a nonempty string')
        value = value.strip()
        if value.lower().startswith('data:'):
            header, value = value.split(',', 1)
            if not header.lower().startswith('data:image/') or not header.lower().endswith(';base64'):
                raise ValueError('expected an image data URL')
        raw = base64.b64decode(''.join(value.split()), validate=True)
        with Image.open(io.BytesIO(raw)) as source:
            source.load()
            return ImageOps.exif_transpose(source).copy()
    except (ValueError, binascii.Error, OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f'{field}: invalid base64 image or image data URL.') from exc


def current_load_spec():
    info = MODEL_REGISTRY[model]
    return {'model': model, 'quantize': model_quantize, 'options': dict(model_options),
            **({'base_model': info['base_model']} if info.get('base_model') else {})}


def prepare_generation(payload, name, quantize, loaded_options, load_spec):
    """Prepare a queue-owned task and an image-free recipe for exact request replay."""
    if not isinstance(payload, dict):
        raise ValueError('The request must be an object.')
    args = dict(payload)
    replay = args.pop('recipe', None)
    if replay is not None:
        if not isinstance(replay, dict) or replay.get('schema_version') != 1:
            raise ValueError('recipe must use schema_version 1.')
        if replay.get('load') != load_spec:
            raise ValueError('Load the model and options in recipe.load before replaying it.')
        if replay.get('runtime') is not None and replay['runtime'] != runtime_settings():
            raise ValueError('Match the server cache/optimization flags recorded in recipe.runtime before replaying.')
        if replay.get('mflux_version', package_version('mflux')) != package_version('mflux'):
            raise ValueError('Recipe mflux_version differs from the running server.')
        generation = replay.get('generation')
        if not isinstance(generation, dict):
            raise ValueError('recipe.generation must be an object.')
        args = {**generation, **args}
    info = MODEL_REGISTRY[name]
    caps = capabilities(name, info)
    if loaded_options.get('qwen21_backend') == 'legacy':
        caps.update(default_format='JPEG', dimension_multiple=16, min_dimension=16, rgba=False, mask=False)
    prompt = args.get('prompt', 'A beautiful landscape')
    if not isinstance(prompt, str):
        raise ValueError('prompt must be a string.')
    seed = str(args.get('seed', int(time.time())))
    int(seed)
    for key, default in (('width', 1024), ('height', 1024), ('steps', caps['default_steps'])):
        value = args.get(key, default)
        if type(value) is not int:
            raise ValueError(f'{key} must be an integer.')
        args[key] = value
    guidance = args.get('guidance', caps['default_guidance'])
    if type(guidance) not in (int, float):
        raise ValueError('guidance must be a number.')
    args['guidance'] = float(guidance)
    raw_images = args.get('init_images', [])
    count = len(raw_images) if isinstance(raw_images, list) else 0
    if args.get('init_image'):
        count = 1
    if (info['loader'] == 'qwen21' and loaded_options.get('qwen21_backend') != 'legacy'
            and args.get('use_step_cache') and not args.get('use_kv_cache', qwen21_context_cache)):
        raise ValueError('use_step_cache requires use_kv_cache.')
    validate_parameters(name, info, args, count, loaded_options)
    if info['loader'] == 'moss-image':
        MOSSImageLatentCreator.check_size(height=args['height'], width=args['width'])
        MOSSImageScheduler.compute_sigmas(args['steps'])
        if 'uncond_skip_layers' in args:
            from mflux.models.moss_image.model.moss_image_transformer.moss_image_transformer import MOSSImageTransformer
            from mflux.models.moss_image.weights.moss_image_weight_mapping import MOSSImageWeightMapping
            MOSSImageTransformer.check_skip_layers(args['uncond_skip_layers'], MOSSImageWeightMapping.NUM_TRANSFORMER_LAYERS)
    if info['loader'] == 'fibo2':
        Fibo2CaptionCheck.check(prompt, edit=count > 0)
    if info['loader'] == 'qwen21':
        from mflux.models.qwen21.model.qwen21_scheduler import Qwen21TurboScheduler
        config = ModelConfig.from_name(info.get('config_name', name))
        Qwen21TurboScheduler.check_generate(config, args['steps'], args['guidance'], args.get('scheduler', 'linear'))
    preset = args.get('preset', IDEOGRAM4_DEFAULT_PRESET)
    if info['loader'] == 'ideogram4':
        Ideogram4LatentCreator.validate_dimensions(width=args['width'], height=args['height'])
        args['steps'] = Ideogram4Scheduler.get_preset(preset).num_steps
        Ideogram4PromptEncoder.resolve_prompt(prompt, strict_caption_validation=args.get('strict_caption_validation', False),
                                             warn_on_caption_issues=False)
    images = decode_input_images(args, name)
    aux_images = {}
    for field in IMAGE_FIELDS:
        if args.get(field):
            # Decode auxiliary inputs with the same strict base64/data-URL rules.
            aux_images[field] = decode_encoded_image(args[field], field)
    if aux_images.get('mask_image') and images and aux_images['mask_image'].size != images[0].size:
        raise ValueError('mask_image must have the same dimensions as the first source image.')
    output_format = args.get('format', caps['default_format'])
    if not isinstance(output_format, str) or output_format.upper() not in ('PNG', 'JPEG'):
        raise ValueError('format must be PNG or JPEG.')
    quality = args.get('quality', 85)
    if type(quality) is not int or not 1 <= quality <= 100:
        raise ValueError('quality must be an integer between 1 and 100.')
    if type(args.get('priority', False)) is not bool:
        raise ValueError('priority must be a boolean.')
    options = {key: args[key] for key in GENERATION_FIELDS if key in args}
    generation = dict(prompt=prompt, seed=seed, width=args['width'], height=args['height'],
                      steps=args['steps'], guidance=args['guidance'], format=output_format.upper(),
                      quality=quality, priority=args.get('priority', False), **options)
    hashes = {'init_images': [input_digest(image) for image in images],
              **{key: input_digest(image) for key, image in aux_images.items()}}
    if replay is not None and replay.get('inputs', {}) != hashes:
        raise ValueError('Replay requires the same source, mask and control images recorded in recipe.inputs.')
    config = ModelConfig.from_name(info.get('config_name', name))
    source = info.get('source') or (name if '/' in name and name != config.controlnet_model else config.model_name)
    revisions = {source: ModelRevisions.get(source)}
    if config.controlnet_model:
        control_source = loaded_options.get('controlnet_path', config.controlnet_model)
        revisions[control_source] = ModelRevisions.get(control_source)
    if loaded_options.get('small_text_encoder'):
        if info['loader'] == 'flux2':
            from mflux.models.flux2.model.flux2_text_encoder.small_text_encoder import SMALL_ENCODER_REPO, ADAPTER_REPO
            components = (SMALL_ENCODER_REPO, ADAPTER_REPO)
        else:
            from mflux.models.qwen21.model.qwen21_small_text_encoder.zen_image_edit_weights import ZenImageEditWeights
            components = (ZenImageEditWeights.REPO,)
        revisions.update({component: ModelRevisions.get(component) for component in components})
    if loaded_options.get('small_decoder'):
        from mflux.models.flux2.model.flux2_vae.small_decoder import SMALL_DECODER_REPO
        revisions[SMALL_DECODER_REPO] = ModelRevisions.get(SMALL_DECODER_REPO)
    revisions.update({path: ModelRevisions.get(path) for path in loaded_options.get('lora_paths', [])})
    recipe = {'schema_version': 1, 'load': load_spec, 'generation': generation,
              'inputs': hashes, 'revisions': revisions, 'runtime': runtime_settings(),
              'mflux_version': package_version('mflux')}
    return {**generation, **({'negative_prompt': args.get('negative_prompt')}
                            if 'negative_prompt' in caps['generation_options'] else {}),
            'generation_options': options, 'init_images': images,
            'aux_images': aux_images, 'init_image_count': len(images), 'recipe': recipe,
            **({'preset': preset} if info['loader'] == 'ideogram4' else {})}


def input_digest(image):
    # Include palette colors and transparency, rather than hashing palette indices.
    image = image.convert('RGBA')
    digest = hashlib.sha256()
    digest.update(f'{image.mode}:{image.size}'.encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def runtime_settings():
    return {'qwen21_context_cache': qwen21_context_cache, 'flux2_optimizations': flux2_optimizations,
            'krea2_optimizations': krea2_optimizations}


def load_model(model_name: str, quantize: int | None, options=None, entry=None):
    info = entry or MODEL_REGISTRY.get(model_name, {})
    loader = info.get("loader")
    options = validate_load_options(model_name, info, quantize, options or {})
    effective_quantize = quantize if quantize is not None else info.get("quantize")
    model_path = info.get('source', model_name if "/" in model_name else None)
    config_name = info.get('config_name', _normalize_flux2_model_name(model_name))
    extras = {key: value for key, value in options.items()
              if key not in ('vae_full_precision', 'qwen21_backend')}
    if 'compute_precision' in extras:
        extras['compute_precision'] = mx.float16 if extras['compute_precision'] == 'float16' else None
    if options.get('lora_paths') and 'lora_scales' not in extras:
        extras['lora_scales'] = [1.0] * len(options['lora_paths'])
    if loader == "flux" and not extras and not info.get('source'):
        return Flux1.from_name(quantize=effective_quantize, model_name=model_name)
    constructors = {
        'flux': Flux1, 'flux2': Flux2KleinEdit, 'flux3': Flux3KleinEdit,
        'qwen21': QwenImage21 if options.get('qwen21_backend') == 'legacy' else QwenImage21Edit,
        'qwen21-controlnet': QwenImage21Controlnet, 'qwen-edit': QwenImageEdit,
        'anima': Anima, 'iris': Iris, 'ming': MingImage, 'krea2-controlnet': Krea2Controlnet,
        'qwen': QwenImage, 'fibo': FIBO, 'fibo2': Fibo2Edit, 'moss-image': MOSSImage,
        'ideogram4': Ideogram4, 'krea2': Krea2, 'ernie': ErnieImage, 'z-image': ZImage,
    }
    if loader not in constructors:
        raise ValueError(f"Unknown model loader for '{model_name}'")
    kwargs = dict(quantize=effective_quantize, model_path=model_path, **extras)
    if loader not in ('qwen', 'fibo'):
        kwargs['model_config'] = ModelConfig.from_name(model_name=config_name)
        if not info.get('source') and model_path == kwargs['model_config'].controlnet_model:
            # A ControlNet repository contains the branch, not the base image model.
            kwargs['model_path'] = None
    if loader == 'krea2' and model_path is None:
        config = ModelConfig.krea2()
        kwargs['model_path'] = snapshot_download(
            repo_id=config.model_name, revision=ModelRevisions.get(config.model_name),
            allow_patterns=Krea2WeightDefinition.get_download_patterns())
    instance = constructors[loader](**kwargs)
    if options.get('vae_full_precision'):
        from mflux.models.common.vae.block_decoder import VAEBlockDecoder
        instance.vae.decode_mode = VAEBlockDecoder.FULL_PRECISION
    # Model-owned state travels with the actual instance, including direct benchmark calls.
    instance._server_load_options = options
    return instance

def generate_with_model(instance, model_name: str, task, init_image_path, aux_paths=None):
    info = MODEL_REGISTRY[model_name]
    loader = info['loader']
    paths = list(init_image_path) if isinstance(init_image_path, (list, tuple)) else (
        [init_image_path] if init_image_path is not None else [])
    validate_image_count(model_name, len(paths))
    aux = aux_paths or {}
    opts = dict(task.get('generation_options', {}))
    opts.pop('negative_prompt', None)
    loaded = getattr(instance, '_server_load_options', {})
    loaded = loaded if isinstance(loaded, dict) else {}
    guidance = task['guidance'] if task['guidance'] is not None else info.get('guidance', 3.5)
    common = dict(seed=int(task['seed']), prompt=task['prompt'],
                  num_inference_steps=task['steps'] or info.get('steps', 4),
                  height=task['height'], width=task['width'])
    negative = task.get('negative_prompt')
    if loader == 'ideogram4':
        common.pop('num_inference_steps')
        return instance.generate_image(**common, preset=task.get('preset', IDEOGRAM4_DEFAULT_PRESET),
            strict_caption_validation=task.get('strict_caption_validation', False),
            **{k: v for k, v in opts.items() if k not in ('preset', 'strict_caption_validation')})
    if loader == 'qwen21' and loaded.get('qwen21_backend') != 'legacy':
        opts.setdefault('use_kv_cache', qwen21_context_cache)
        if aux.get('mask_image'):
            opts['mask_image'] = aux['mask_image']
        return instance.generate_image(**common, image_paths=paths or None,
                                       guidance=guidance, negative_prompt=negative, **opts)
    if loader == 'qwen21-controlnet':
        return instance.generate_image(**common, guidance=guidance, negative_prompt=negative,
            controlnet_image_path=aux.get('control_image'), mask_image=aux.get('mask_image'),
            image_path=paths[0] if paths else None, **opts)
    if loader == 'krea2-controlnet':
        return instance.generate_image(**common, guidance=guidance, negative_prompt=negative,
            image_path=paths[0] if paths else None, depth_image_path=aux.get('depth_image'), **opts)
    if loader == 'qwen-edit':
        return instance.generate_image(**common, guidance=guidance, negative_prompt=negative,
            image_paths=[str(path) for path in paths], image_path=str(paths[0]) if paths else None, **opts)
    if loader in ('anima', 'iris', 'moss-image'):
        return instance.generate_image(**common, guidance=guidance, negative_prompt=negative, **opts)
    if loader == 'ming':
        return instance.generate_image(**common, guidance=guidance)
    if loader == 'fibo2':
        if paths:
            if aux.get('mask_image'):
                opts['mask_path'] = aux['mask_image']
            return instance.generate_image(**common, image_paths=paths, guidance=guidance,
                                           negative_prompt=negative, **opts)
        return Fibo2.generate_image(instance, **common, guidance=guidance, negative_prompt=negative, **opts)
    if loader == 'flux3':
        if paths:
            return instance.generate_image(**common, image_paths=paths, guidance=guidance, **opts)
        return Flux3Klein.generate_image(instance, **common, guidance=guidance, **opts)
    if loader == 'flux2':
        with optimized_flux2(instance, enabled=flux2_optimizations):
            return instance.generate_image(**common, image_paths=paths or None, guidance=guidance, **opts)
    common['image_path'] = paths[0] if paths else None
    common['image_strength'] = 0.4 if paths else None
    if loader == 'qwen21':
        if paths:
            common.pop('image_path'); common.pop('image_strength')
            return generate_qwen21_references(instance, paths, **common, guidance=guidance,
                negative_prompt=negative, use_cache=qwen21_context_cache)
        with cached_qwen21_context(instance, enabled=qwen21_context_cache):
            return instance.generate_image(**common, guidance=guidance, negative_prompt=negative)
    if loader == 'z-image':
        return instance.generate_image(**common, **opts)
    if loader == 'krea2':
        with optimized_krea2(instance, enabled=krea2_optimizations):
            return instance.generate_image(**common, guidance=guidance, negative_prompt=negative, **opts)
    if loader == 'fibo':
        try:
            json.loads(common['prompt'])
        except json.JSONDecodeError:
            common['prompt'] = json.dumps({'prompt': common['prompt']})
    return instance.generate_image(**common, guidance=guidance, **opts)

def _load_model_runtime_now(model_name: str, quantize: int | None, options=None, entry=None):
    global model_instance, model, model_quantize, model_options
    info = entry or resolve_entry(MODEL_REGISTRY, model_name)
    options = validate_load_options(model_name, info, quantize, options or {})
    if info.get('source') and Path(info['source']).is_dir():
        checked = ModelChecker.check(info['source'], base_model=info['config_name'])
        if checked.check != 'pass':
            raise ValueError('Model package validation failed: ' + '; '.join(checked.problems))
    effective_quantize = quantize if quantize is not None else info.get("quantize")
    repo_id = model_name if "/" in model_name else None
    cache_before = _hf_repo_cached(repo_id) if repo_id else None
    start_time = time.time()
    print(f"Model load started: '{model_name}'")
    try:
        if options or entry is not None:
            loaded_instance = load_model(model_name, effective_quantize, options, info)
        else:
            loaded_instance = load_model(model_name, effective_quantize)
    except Exception as exc:
        elapsed = time.time() - start_time
        exc_name = exc.__class__.__name__
        print(f"Model load failed: '{model_name}' in {elapsed:.2f}s ({exc_name}: {exc})")
        raise
    elapsed = time.time() - start_time
    if repo_id:
        if cache_before:
            source_note = "from cache"
        elif _hf_repo_cached(repo_id):
            source_note = "downloaded"
        else:
            source_note = "cache status unknown"
        print(f"Model load finished: '{model_name}' in {elapsed:.2f}s ({source_note})")
    else:
        print(f"Model load finished: '{model_name}' in {elapsed:.2f}s")
    with model_lock:
        MODEL_REGISTRY[model_name] = info
        model = model_name
        model_quantize = effective_quantize
        model_options = options
        model_instance = loaded_instance


def load_model_runtime(model_name: str, quantize: int | None, options=None, base_model=None):
    entry = resolve_entry(MODEL_REGISTRY, model_name, base_model)
    options = validate_load_options(model_name, entry, quantize, options or {})
    extended = bool(options or base_model is not None or model_name not in MODEL_REGISTRY)
    if threading.current_thread() is model_worker_thread:
        if extended:
            _load_model_runtime_now(model_name, quantize, options, entry)
        else:
            _load_model_runtime_now(model_name, quantize)
        return

    done = threading.Event()
    request = {
        "model": model_name,
        "quantize": quantize,
        "options": options if extended else None,
        "entry": entry if extended else None,
        "done": done,
        "error": None
    }
    with model_lock:
        model_load_requests.append(request)
    worker_wakeup.set()
    done.wait()
    if request["error"]:
        raise request["error"]


def _process_model_load_request(worker_stream):
    with model_lock:
        request = model_load_requests.pop(0) if model_load_requests else None
    if request is None:
        return False

    try:
        with mx.stream(worker_stream):
            if request.get('entry') is not None:
                _load_model_runtime_now(request["model"], request["quantize"], request['options'], request['entry'])
            else:
                _load_model_runtime_now(request["model"], request["quantize"])
    except Exception as exc:
        request["error"] = exc
    finally:
        request["done"].set()
    return True

# we implement image generation as asynchronous task
# this will be executed in a separate thread
def _set_mlx_cache_limit(limit: int) -> None:
    try:
        mx.set_cache_limit(limit)
    except AttributeError:
        metal_compat.set_cache_limit(limit)


def _clear_mlx_cache() -> None:
    try:
        mx.clear_cache()
    except AttributeError:
        metal_compat.clear_cache()


def _new_mlx_worker_stream():
    try:
        return mx.new_thread_local_stream(mx.gpu)
    except AttributeError:
        return mx.new_stream(mx.gpu)


def generation_progress(phase, step=0, total_steps=None, fraction=None, attempt=0):
    return dict(phase=phase, step=step, total_steps=total_steps, fraction=fraction, attempt=attempt,
                sampling_fraction=None, elapsed_seconds=None, remaining_seconds=None,
                iterations_per_second=None, seconds_per_step=None, sampling_stats=None)


class GenerationProgress:
    """Expose native log counters and timing without evaluating or retaining tensors."""

    def __init__(self, task):
        self.task = task
        self.step = self.attempt = 0
        self.total = None
        self.bar = None
        self.native_display = False
        self.display_hooks = ExitStack()
        self.watched_bars = set()
        self.metrics = {}
        self.started = None

    def publish(self, phase, fraction=None):
        # Replace the snapshot atomically for concurrent /status readers.
        self.task['progress'] = {**generation_progress(phase, self.step, self.total, fraction, self.attempt),
                                 **self.metrics}

    def capture(self, values):
        self.step = max(0, int(values.get('n', 0)))
        self.total = values.get('total', self.total)
        elapsed = values.get('elapsed')
        elapsed = float(elapsed) if isinstance(elapsed, (int, float)) and math.isfinite(elapsed) and elapsed >= 0 else None
        rate = values.get('rate')
        if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            completed = max(0, self.step - values.get('initial', 0))
            rate = completed / elapsed if elapsed and completed else None
        remaining = max(0, self.total - self.step) / rate if self.total and rate else None
        sampling_fraction = min(1, self.step / self.total) if self.total else None
        self.metrics = dict(sampling_fraction=sampling_fraction, elapsed_seconds=elapsed,
                            remaining_seconds=remaining, iterations_per_second=rate,
                            seconds_per_step=1 / rate if rate else None,
                            sampling_stats=tqdm.format_meter(**{
                                **values, 'elapsed': elapsed or 0, 'rate': rate, 'ncols': None, 'prefix': '',
                                'bar_format': '{percentage:.0f}% · {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
                            }))

    def watch(self, bar):
        if bar is self.bar:
            return
        self.bar = bar
        self.native_display = isinstance(bar, tqdm) and not bar.disable
        if not self.native_display:
            return
        original_display = bar.display

        def display(*args, **kwargs):
            result = original_display(*args, **kwargs)
            if bar is self.bar:
                self.capture(bar.format_dict)
                self.publish('sampling', self.work_fraction())
            return result

        if id(bar) not in self.watched_bars:
            self.display_hooks.enter_context(instance_override(bar, 'display', display))
            self.watched_bars.add(id(bar))
        self.capture(bar.format_dict)

    def work_fraction(self):
        return min(0.95, 0.95 * self.step / self.total) if self.total and self.attempt == 1 else None

    def call_before_loop(self, config, **kwargs):
        self.attempt += 1
        self.step = 0
        self.metrics = {}
        self.started = time.monotonic()
        self.bar = None
        self.total = getattr(config.time_steps, 'total', None)
        if self.total is None:
            self.total = max(0, config.num_inference_steps - getattr(config, 'init_time_step', 0))
        self.watch(config.time_steps)
        self.publish('sampling', 0 if self.attempt == 1 else None)

    def call_in_loop(self, time_steps=None, **kwargs):
        if time_steps is not None:
            self.watch(time_steps)
        if self.native_display:
            # Native display runs after the iterator resumes, including mx.eval on models
            # such as FLUX 3. The callback itself may run before the GPU finishes a step.
            return
        self.step += 1
        self.capture(dict(n=self.step, total=self.total, elapsed=time.monotonic() - self.started))
        self.publish('sampling', self.work_fraction())

    def call_after_loop(self, **kwargs):
        self.publish('finishing', 0.95 if self.attempt == 1 else None)


@contextmanager
def track_generation_progress(instance, task):
    registry = getattr(instance, 'callbacks', None)
    reporter = GenerationProgress(task)
    if isinstance(registry, CallbackRegistry):
        registry.register(reporter)
    try:
        yield
    finally:
        try:
            reporter.display_hooks.close()
        finally:
            if isinstance(registry, CallbackRegistry):
                for callbacks in (registry.before_loop, registry.in_loop, registry.after_loop):
                    callbacks[:] = [callback for callback in callbacks if callback is not reporter]


def process_image_task(task, instance, model_name, quantize, worker_stream):
    """Run one queued request and release every temporary reference on all exit paths."""
    global pixels, ctime
    task['compute_time'] = time.time()
    task['model_used'] = model_name
    task['quantize_used'] = quantize
    task['progress'] = generation_progress('preparing')
    try:
        inputs = task.get('init_images', [])
        aux_images = task.get('aux_images', {})
        if (inputs or aux_images or 'recipe' in task) and (
            task.get('model_at_submit', model_name) != model_name
            or task.get('quantize_at_submit', quantize) != quantize
            or ('recipe' in task and task['recipe']['load'] != current_load_spec())
        ):
            raise ValueError("The loaded model changed after this generation request was queued; submit it again.")
        validate_image_count(model_name, len(inputs))
        _set_mlx_cache_limit(metal_cache_limit)
        with TemporaryDirectory(prefix="mflux-inputs-") as directory:
            paths = []
            for index, input_image in enumerate(inputs):
                path = Path(directory) / f"reference-{index + 1}.png"
                input_image.save(path, format="PNG")
                paths.append(path)
            aux_paths = {}
            for field, input_image in aux_images.items():
                path = Path(directory) / f'{field}.png'
                input_image.save(path, format='PNG')
                aux_paths[field] = path
            with mx.stream(worker_stream), track_generation_progress(instance, task):
                result = (generate_with_model(instance, model_name, task, paths, aux_paths)
                          if aux_paths else generate_with_model(instance, model_name, task, paths))
            task['progress'] = {**task['progress'], 'phase': 'finishing'}
            if isinstance(getattr(result, 'prompt', None), str) and 'recipe' in task:
                task['recipe']['effective_prompt'] = result.prompt
            output = result.image
            output_format = task.get('format', 'JPEG').upper()
            if output_format not in ('PNG', 'JPEG'):
                output_format = 'JPEG'
            if output_format == 'JPEG':
                if 'A' in output.getbands() or 'transparency' in output.info:
                    rgba = output.convert('RGBA')
                    background = Image.new('RGBA', rgba.size, 'white')
                    output = Image.alpha_composite(background, rgba).convert('RGB')
                else:
                    output = output.convert('RGB')
            encoded = io.BytesIO()
            options = {'quality': task['quality']} if output_format == 'JPEG' else {}
            if output_format == 'PNG' and 'recipe' in task:
                metadata = PngImagePlugin.PngInfo()
                metadata.add_text('mflux_recipe', json.dumps(task['recipe'], ensure_ascii=False))
                options['pnginfo'] = metadata
            output.save(encoded, format=output_format, **options)
            encoded.seek(0)
        # Publish only once encoding and temporary-file cleanup have succeeded.
        ctime += time.time() - task['compute_time']
        pixels += output.width * output.height
        task['format'] = output_format
        task['image'] = encoded
        task['progress'] = {**task['progress'], 'phase': 'done', 'fraction': 1.0}
    except Exception as exc:
        task['error'] = f"{type(exc).__name__}: {exc}"
        task['progress'] = {**task['progress'], 'phase': 'error'}
        app.logger.exception("Image generation failed for task %s", task['task_id'])
    finally:
        task['end_time'] = time.time()
        task.pop('init_images', None)
        task.pop('aux_images', None)
        for cleanup in (_clear_mlx_cache, gc.collect):
            try:
                cleanup()
            except Exception:
                app.logger.exception("Cleanup failed for task %s", task['task_id'])


def compute_image_task():
    global model_instance, tasklist, pixels, ctime, model_worker_thread
    model_worker_thread = threading.current_thread()
    worker_stream = _new_mlx_worker_stream()
    while True:
        # Clear before inspecting queues so arrivals during inspection remain
        # signalled. Drain load requests before waiting, even when no model loaded.
        worker_wakeup.clear()
        if _process_model_load_request(worker_stream):
            continue
        with model_lock:
            current_model_instance = model_instance
            current_model_name = model
            current_model_quantize = model_quantize
        if current_model_instance == None or len(tasklist) == 0:
            worker_wakeup.wait()
            continue
        
        # Process each pending task once, including tasks that fail.
        processed_task = False
        for task in tasklist:
            if 'image' in task or 'error' in task: continue
            process_image_task(task, current_model_instance, current_model_name,
                               current_model_quantize, worker_stream)
            processed_task = True
            break
        
        if not processed_task:
            worker_wakeup.wait()

def str_to_bool(value):
    return value.lower() in ['true', '1', 't', 'y', 'yes']


# Legacy clients send either a string or null. Validate this union in
# decode_input_images; Flask-RESTX's Raw field otherwise requires an object.
class EncodedImageField(fields.Raw):
    __schema_type__ = None


# generate image endpoint

task_model = api.model('TaskInput', {
    'prompt': fields.String(description='Description or JSON caption/layout string; may be omitted when replaying a recipe.', default='A beautiful landscape', required=False),
    'init_image': EncodedImageField(description='One base64 image or image data URL; null or empty means no image. Cannot be combined with a nonempty init_images array.'),
    'init_images': fields.List(fields.String, description='Ordered base64 images or image data URLs for one output. Empty means no inputs. See /api/info for the model limit.'),
    'negative_prompt': fields.String(description='Qwen Image 2.1, Krea 2 or FIBO 2 negative conditioning above guidance 1; MOSS-Image negative conditioning when guidance differs from 1. Unsupported by FLUX 3.', required=False),
    'seed': fields.String(description='Entropy Seed', default=str(int(time.time())), required=False),
    'height': fields.Integer(description='Image height', default=1024, required=False),
    'width': fields.Integer(description='Image width', default=1024, required=False),
    'steps': fields.Integer(description='Inference Steps (ignored by Ideogram 4; use preset)', default=MODEL_REGISTRY.get(model, {}).get("steps", 4), required=False),
    'guidance': fields.Float(description='Guidance Scale (ignored by distilled FLUX 3 and Ideogram 4; use preset for Ideogram 4)', default=MODEL_REGISTRY.get(model, {}).get("guidance", 3.5), required=False),
    'preset': fields.String(description='Ideogram 4 sampler preset', enum=list(IDEOGRAM4_PRESETS), default=IDEOGRAM4_DEFAULT_PRESET, required=False),
    'strict_caption_validation': fields.Boolean(description='Reject Ideogram 4 caption warnings before queuing', default=False, required=False),
    'format': fields.String(description='JPEG or PNG; PNG is the default for Qwen 2.1, Qwen ControlNet and Ming.', required=False),
    'quality': fields.Integer(description='JPEG compression quality (1-100) if format is JPEG, default is 85', default=85, required=False),
    'priority': fields.Boolean(description='Set to true to put this task to the head of the queue', default=False, required=False)
})
for _field in IMAGE_FIELDS:
    task_model[_field] = EncodedImageField(description='Base64 image or data URL; see /api/info for support.')
task_model['recipe'] = fields.Raw(description='Recipe from /api/recipe; load recipe.load first and supply the original inputs.')
for _field in ('scheduler', 'auto_mask'):
    task_model[_field] = fields.String(description=f'Model-specific {_field}; see /api/info.')
for _field in ('step_cache_ratio', 'cfg_end', 'strength', 'controlnet_strength', 'step_cache_threshold'):
    task_model[_field] = fields.Float(description=f'Optional {_field}; see /api/info.')
for _field in ('use_kv_cache', 'use_step_cache', 'enhance_prompt', 'verify', 'render_text_images'):
    task_model[_field] = fields.Boolean(description=f'Optional {_field}; see /api/info.')
for _field in ('verify_retries', 'output_resolution'):
    task_model[_field] = fields.Integer(description=f'Optional {_field}; see /api/info.')
task_model['uncond_skip_layers'] = fields.List(fields.Integer, description='MOSS negative-branch layers; [] disables skipping.')

generate_response_model = api.model('GenerateResponse', {
    'task_id': fields.String(description='ID of the image generation task'),
    'task_length': fields.Integer(description='Length of the image generation task queue excluding this new one'),
    'expected_time_seconds': fields.Float(description='Expected time in seconds for the image generation task to complete'),
    'model': fields.String(description='Model in use when the task was queued'),
    'quantize': fields.Integer(description='Quantization level in use when the task was queued')
})

# function which counts number of pixels in images from the tasklist up to a certain index
def count_pixels(index):
    global tasklist
    pixels = 0
    for i in range(index):
        if i >= len(tasklist): break
        task = tasklist[i]
        if 'image' not in task and 'error' not in task:
            pixels += task['width'] * task['height']
    return pixels

@api.route('/ls')
class ListModels(Resource):
    @api.response(200, 'Success')
    def get(self):
        """
        The /ls endpoint provides a catalog of available models and defaults.
        """
        with model_lock:
            entries = list(MODEL_REGISTRY.items())
        return jsonify({name: {**settings, **capabilities(name, settings)} for name, settings in entries})

info_model = api.model('ModelInfo', {
    'model': fields.String(description='Currently selected model'),
    'quantize': fields.Integer(description='Current quantization, or null'),
    'edit': fields.Boolean(description='Accepts at least one input image (including ordinary img2img)'),
    'multi_image_edit': fields.Boolean(description='Combines multiple ordered references into one output'),
    'max_init_images': fields.Integer(description='Server input-image limit for this model'),
    'min_init_images': fields.Integer(description='Minimum number of source images required'),
    'default_steps': fields.Integer(description='Default inference steps'),
    'default_guidance': fields.Float(description='Default guidance; ignored by models without guidance'),
    'guidance_fixed': fields.Float(description='Required guidance, or null when configurable'),
    'context_kv_cache': fields.Boolean(description='Qwen 2.1 request-local prefix reuse enabled; unsupported inputs use native execution'),
    'inference_optimizations': fields.List(fields.String, description='Enabled model-specific inference optimizations'),
})
for _field in ('load_options', 'generation_options', 'quantization', 'schedulers'):
    info_model[_field] = fields.List(fields.Raw, description=f'Supported {_field}.')
for _field in ('dimension_multiple', 'min_dimension', 'max_dimension', 'min_steps', 'max_steps', 'fixed_steps'):
    info_model[_field] = fields.Integer(description=f'Model-specific {_field}; null means unspecified.')
for _field in ('mask', 'control_image', 'depth_image', 'rgba'):
    info_model[_field] = fields.Boolean(description=f'Model supports {_field}.')
info_model['options'] = fields.Raw(description='Currently loaded component and adapter options.')
info_model['load'] = fields.Raw(description='Model-load configuration for recipes.')
info_model['default_format'] = fields.String(description='PNG for RGBA models; JPEG otherwise.')

@api.route('/info')
class GetInfo(Resource):
    @api.response(200, 'Success', info_model)
    def get(self):
        with model_lock:
            name, quantize, options = model, model_quantize, dict(model_options)
            load_spec = current_load_spec()
        settings = MODEL_REGISTRY[name]
        caps = capabilities(name, settings)
        if options.get('small_text_encoder'):
            caps['generation_options'] = [key for key in caps['generation_options']
                                          if key not in ('auto_mask', 'enhance_prompt', 'verify', 'verify_retries')]
        if options.get('qwen21_backend') == 'legacy':
            caps.update(generation_options=['negative_prompt'], dimension_multiple=16, min_dimension=16,
                        rgba=False, default_format='JPEG', mask=False)
        native_qwen = settings['loader'] == 'qwen21' and options.get('qwen21_backend') != 'legacy'
        optimizations = (['native_prefix_kv_cache'] if native_qwen and qwen21_context_cache else
            ['prepared_text', 'cached_geometry', 'grouped_attention', 'image_only_output']
            if krea2_optimizations and settings['loader'] == 'krea2' else
            ['prompt_cache', 'reference_encode_cache', 'predictor_cache']
            if flux2_optimizations and settings.get('config_name', name) in FLUX2_4B_NAMES else [])
        return jsonify({'model': name, 'quantize': quantize, **caps,
            'options': options, 'load': load_spec, 'cache_limit': metal_cache_limit,
            'default_preset': settings.get('preset'), 'presets': settings.get('presets', {}),
            'context_kv_cache': qwen21_context_cache and settings['loader'] == 'qwen21',
            'inference_optimizations': optimizations})

@api.route('/ps')
class GetSettings(Resource):
    @api.response(200, 'Success')
    def get(self):
        """
        The /ps endpoint provides the current server settings and default model.
        """
        return GetInfo().get()

@api.route('/load')
class LoadModel(Resource):
    @api.response(200, 'Success')
    @api.response(400, 'Invalid model')
    @api.response(401, 'Hugging Face model access is gated')
    @api.response(503, 'Model repository or pinned revision is unavailable')
    def post(self):
        """
        The /load endpoint replaces the currently loaded model.
        """
        args = request.json or {}
        if not isinstance(args, dict) or not isinstance(args.get('model'), str) or not args['model'].strip():
            return {'error': 'model is required'}, 400
        try:
            unknown = set(args) - {'model', 'quantize', 'base_model', 'options'}
            if unknown:
                raise ValueError('Unknown load fields: ' + ', '.join(sorted(unknown)))
            if 'options' in args and not isinstance(args['options'], dict):
                raise ValueError('options must be an object.')
            requested_model, requested_quantize = args['model'], args.get('quantize')
            if isinstance(requested_quantize, str) and requested_quantize.isdigit():
                requested_quantize = int(requested_quantize)
            if args.get('options') or args.get('base_model'):
                load_model_runtime(requested_model, requested_quantize, args.get('options', {}), args.get('base_model'))
            else:
                load_model_runtime(requested_model, requested_quantize)
        except (ValueError, TypeError) as exc:
            return {'error': str(exc)}, 400
        except Exception as exc:
            if _is_gated_repo_error(exc):
                return {'error': 'Model access appears gated; login to Hugging Face is required.'}, 401
            if isinstance(exc, HfHubHTTPError):
                return {'error': f'Model repository or pinned revision is unavailable: {exc}'}, 503
            raise
        return GetInfo().get()


@api.route('/check')
class CheckModel(Resource):
    def post(self):
        args = request.json or {}
        try:
            if not isinstance(args, dict) or not isinstance(args.get('model'), str):
                raise ValueError('model is required.')
            if set(args) - {'model', 'base_model'}:
                raise ValueError('Only model and base_model are accepted; /api/check is read-only.')
            entry = resolve_entry(MODEL_REGISTRY, args['model'], args.get('base_model'))
            config = ModelConfig.from_name(entry.get('config_name', args['model']))
            reference = entry.get('source') or (args['model'] if '/' in args['model']
                and args['model'] != config.controlnet_model else config.model_name)
            result = ModelChecker.check(reference,
                                        base_model=entry.get('config_name', args['model']))
            return asdict(result), 200 if result.check == 'pass' else 400
        except (KeyError, ValueError, TypeError, OSError) as exc:
            return {'error': str(exc)}, 400


@api.route('/saved')
class SavedModels(Resource):
    def get(self):
        with model_lock:
            names = list(MODEL_REGISTRY)
        return {'directory': models_directory,
                'models': discover_saved_models(models_directory) if models_directory else [],
                'base_models': names}


@api.route('/recipe')
class GetRecipe(Resource):
    def get(self):
        task_id = request.args.get('task_id')
        for task in tasklist:
            if task['task_id'] == task_id and 'recipe' in task:
                return jsonify(task['recipe'])
        return {'error': 'Recipe not found; retrieve it before deleting the image task.'}, 404


@api.route('/generate')
class GenerateImage(Resource):
    @api.expect(task_model, validate=True)
    @api.response(200, 'Success', generate_response_model)
    @api.response(400, 'Invalid generation parameters')
    @api.response(404, 'Cannot append task')
    def post(self):
        """
        The /generate endpoint is used to generate an image as an asynchronous task.
        This will put the task in the queue and return the task ID.
        The task is either at the end of the queue or at the beginning if priority is set to true.
        To save memory, the image is not stored in it's raw form but in the form demanded by the client.
        Therefore the format has to be declared in the request at generation time in this endpoint.
        """
        global tasklist, pixels, ctime
        args = request.json or {}
        with model_lock:
            name, quantize, loaded_options = model, model_quantize, dict(model_options)
            load_spec = current_load_spec()
        try:
            prepared = prepare_generation(args, name, quantize, loaded_options, load_spec)
        except (ValueError, TypeError, OverflowError) as exc:
            return {'error': str(exc)}, 400
        start_time = time.time()
        task_id = hashlib.md5(f'{start_time}-{time.monotonic_ns()}'.encode()).hexdigest()[:8]
        task = {**prepared, 'task_id': task_id, 'start_time': start_time,
                'model_at_submit': name, 'quantize_at_submit': quantize,
                'progress': generation_progress('queued', fraction=0)}
        wait_for_pixels = task['width'] * task['height']
        if task['priority'] and len(tasklist) > 1:
            wait_for_pixels += count_pixels(1)
            tasklist.insert(1, task)
        else:
            wait_for_pixels += count_pixels(len(tasklist))
            tasklist.append(task)
        worker_wakeup.set()
        return {'task_id': task_id, 'task_length': len(tasklist) - 1,
                'expected_time_seconds': ctime * wait_for_pixels / pixels,
                'model': name, 'quantize': quantize}, 200

status_model = api.model('Status', {
    'status': fields.String(description='Status of the image generation task', enum=['waiting', 'done', 'error']),
    'error': fields.String(description='Failure description when status is error'),
    'pos': fields.Integer(description='Position in queue'),
    'progress': fields.Nested(api.model('GenerationProgress', {
        'phase': fields.String(enum=['queued', 'preparing', 'sampling', 'finishing', 'done', 'error']),
        'step': fields.Integer(description='Completed sampling iterations, matching the native log counter'),
        'total_steps': fields.Integer(description='Actual sampling-loop length, or null'),
        'fraction': fields.Float(description='0–1 progress; null for unknown work. Sampling covers up to 0.95.'),
        'attempt': fields.Integer(description='Sampling pass, starting at one; verification may retry'),
        'sampling_fraction': fields.Float(description='Native sampling-loop progress, 0–1 independently of image finishing'),
        'elapsed_seconds': fields.Float(description='Elapsed sampling time from the native progress display'),
        'remaining_seconds': fields.Float(description='Estimated remaining sampling time, using the log rate'),
        'iterations_per_second': fields.Float(description='Native smoothed sampling rate, or average when unavailable'),
        'seconds_per_step': fields.Float(description='Inverse sampling rate, or null until available'),
        'sampling_stats': fields.String(description='Native log formatter: percentage, iteration count, elapsed/remaining and rate')
    }))
})

@api.route('/status')
class GetStatus(Resource):
    @api.doc(params={'task_id': 'The ID of the image generation task'})
    @api.response(200, 'Success', status_model)
    @api.response(404, 'Task not found')
    def get(self):
        """
        The /status endpoint is used to check the image generation progress of a task.
        The returned status can be i.e. when the task is not ready, position 3 in the queue, estimated time remaining 43 seconds:
        { "status": "waiting", "pos": 3, "wait_remaining": 43}
        .. or when the task is done:
        { "status": "done"}
        When the status is "done", the image can be retrieved with the /image endpoint.
        Known task responses also include progress: phase, step, total_steps, fraction and attempt.
        A null fraction means unknown work; sampling contributes up to 0.95 and a ready image is 1.
        sampling_fraction (0–1), elapsed_seconds, remaining_seconds, iterations_per_second,
        seconds_per_step and sampling_stats mirror the native sampling log; remaining time is an estimate.
        These sampling metrics exclude image finishing and are null until available.
        A failed task returns { "status": "error", "error": "..." } and is not retried.
        If the task / the task_id is unknown, the endpoint returns a 404 status code.
        """
        task_id = request.args.get('task_id', default='')
        c = -1
        for i, task in enumerate(tasklist):
            if 'image' not in task and 'error' not in task: c += 1
            if task['task_id'] == task_id:
                phase = ('error' if 'error' in task else 'done' if 'image' in task else
                         'preparing' if 'compute_time' in task else 'queued')
                progress = dict(task.get('progress', generation_progress(phase)))
                if 'error' in task:
                    progress['phase'] = 'error'
                    return jsonify({'status': 'error', 'error': task['error'], 'progress': progress})
                elif 'image' in task:
                    progress.update(phase='done', fraction=1.0)
                    return jsonify({'status': 'done', 'progress': progress})
                else:
                    # compute the remaining time
                    wait_remaining = count_pixels(i + 1) * ctime / pixels
                    start_time = task.get('start_time', 0)
                    compute_time = task.get('compute_time', start_time)
                    wait_remaining = int(wait_remaining - (time.time() - compute_time))
                    if wait_remaining < 1: wait_remaining = 1
                    return jsonify({'status': 'waiting', 'pos': c, 'wait_remaining': wait_remaining,
                                    'progress': progress})
        return Response(status=404)

@api.route('/image')
class GetImage(Resource):
    @api.doc(params={
        'task_id': 'The ID of the image generation task',
        'base64': 'Set to true to return the image as base64 encoded string, default false',
        'delete': 'Set to true to delete the task after getting the image, default is true'
    })
    @api.response(200, 'Success')
    @api.response(404, 'Task not found')
    def get(self):
        """
        The /image endpoint is used to get the produced image after a task has completed.
        The image is already encoded in PNG or JPEG according to the formet given in the /generate endpoint.
        The image can be returned as base64 encoded string or as binary data.
        By default calling this endpoint will delete the task from the queue;
        this means the image can only be retrieved once. To keep the task in the queue set delete to false.
        If the image is not ready at the time of the request, the endpoint returns a 404 status code.
        """
        task_id = request.args.get('task_id', default='')
        for task in tasklist:
            if task['task_id'] == task_id:
                if 'image' in task:
                    image = task['image']
                    format = task['format']
                    base64p = str_to_bool(request.args.get('base64', default='false'))
                    deletep = str_to_bool(request.args.get('delete', default='true'))
                    if deletep: 
                        tasklist.remove(task)
                        gc.collect()
                    if base64p:
                        return Response(base64.b64encode(image.getvalue()), mimetype='text/plain; charset=utf-8')
                    else:
                        return Response(image.getvalue(), mimetype='image/png' if format == 'PNG' else 'image/jpeg')
        return Response(status=404)

@api.route('/cancel')
class CancelTask(Resource):
    @api.doc(params={'task_id': 'The ID of the image generation task'})
    @api.response(200, 'Success')
    @api.response(404, 'Task not found')
    def get(self):
        """
        The /cancel endpoint is used to cancel a task.
        """
        task_id = request.args.get('task_id', default='')
        for task in tasklist:
            if task['task_id'] == task_id:
                tasklist.remove(task)
                return Response(status=200)
        return Response(status=404)

task_output_model = api.model('TaskOutput', {
    **{key: field for key, field in task_model.items() if key not in ('init_image', 'init_images', *IMAGE_FIELDS)},
    'init_image_count': fields.Integer(description='Number of input images'),
    'task_id': fields.String(description='ID of the image generation task', default=None, required=False),
    'start_time': fields.String(description='Time when the image generation task was submitted', default=None, required=False),
    'compute_time': fields.String(description='Time when the image computation started', default=None, required=False),
    'end_time': fields.String(description='Time when the image generation task ended', default=None, required=False)
})
tasks_model = api.model('Tasks', {
    'tasks': fields.List(fields.Nested(task_output_model), description='List of tasks')
})

@api.route('/tasks')
class GetTasks(Resource):
    @api.response(200, 'Success', tasks_model)
    def get(self):
        """
        The /tasks endpoint is used to list all tasks.
        This can be used to implement a task manager.
        """
        tasklist0 = []
        for task in tasklist:
            task0 = task.copy()
            for field in ('image', 'init_images', 'init_image', 'aux_images', *IMAGE_FIELDS):
                task0.pop(field, None)
            tasklist0.append(task0)        
        return jsonify(tasklist0)

@api.route('/clear')
class ClearTasks(Resource):
    @api.response(200, 'Success')
    def get(self):
        tasklist.clear()
        return Response(status=200)

@app.route('/')
def redirect_to_index():
    return redirect('/index.html')

@app.route('/index.html')
def serve_index():
    return send_file(os.path.join(apppath, 'clients/web-ui/index.html'))

def build_argument_parser():
    parser = argparse.ArgumentParser(description='Start a server to generate images with mflux.')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help='Registered model, saved directory or HF repository.')
    parser.add_argument('--base-model', choices=MODEL_REGISTRY.keys(), help='Required model family for custom checkpoints.')
    parser.add_argument('--quantize', '-q', type=int, choices=[4, 5, 6, 8], default=None, help='Model-specific quantization; omit to keep stored precision.')
    parser.add_argument('--models-dir', default=models_directory, help='Directory of saved model packages (MFLUX_MODELS_DIR).')
    parser.add_argument('--lora', action='append', default=[], metavar='PATH', help='Repeat for multiple LoRAs.')
    parser.add_argument('--lora-scale', action='append', type=float, help='One scale per --lora, in the same order.')
    parser.add_argument('--bake-lora', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--compute-precision', choices=['bfloat16', 'float16'])
    parser.add_argument('--small-text-encoder', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--small-decoder', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--vae-full-precision', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--text-encoder-quantize', type=int, choices=[4, 5, 6, 8])
    parser.add_argument('--qwen21-backend', choices=['native', 'legacy'], default=None)
    parser.add_argument('--controlnet-path')
    parser.add_argument('--controlnet-strength', type=float)
    parser.add_argument('--host', type=str, default='127.0.0.1', help='The host to listen on')
    parser.add_argument('--port', type=int, default=4030, help='The port to listen on')
    parser.add_argument('--cache_limit', type=int, default=0, help='The metal cache limit in bytes')
    parser.add_argument('--qwen21-context-cache', action=argparse.BooleanOptionalAction, default=True,
                        help='Reuse invariant Qwen Image 2.1 per-layer context within each generation (default: enabled)')
    parser.add_argument('--krea2-optimizations', action=argparse.BooleanOptionalAction, default=True,
                        help='Prepare Krea 2 conditioning once and use native grouped attention (default: enabled)')
    parser.add_argument('--flux2-optimizations', action=argparse.BooleanOptionalAction, default=True,
                        help='Reuse FLUX.2 Klein 4B predictors and bounded encoder caches (default: enabled)')
    return parser

def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    global metal_cache_limit, qwen21_context_cache, krea2_optimizations, flux2_optimizations, models_directory
    metal_cache_limit = args.cache_limit
    qwen21_context_cache = args.qwen21_context_cache
    krea2_optimizations = args.krea2_optimizations
    flux2_optimizations = args.flux2_optimizations
    models_directory = args.models_dir
    options = {key: getattr(args, key) for key in LOAD_FIELDS
               if hasattr(args, key) and getattr(args, key) is not None}
    if args.lora_scale and not args.lora:
        parser.error('--lora-scale requires at least one --lora.')
    if args.lora:
        options['lora_paths'] = args.lora
        options['lora_scales'] = args.lora_scale or [1.0] * len(args.lora)
    threading.Thread(target=compute_image_task, daemon=True).start()
    if options or args.base_model:
        load_model_runtime(args.model, args.quantize, options, args.base_model)
    else:
        load_model_runtime(args.model, args.quantize)
    print(f"Server started, view swagger API documentation at http://{args.host}:{args.port}/swagger")
    app.run(host=args.host, port=args.port)

if __name__ == '__main__':
    main()
