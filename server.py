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
from tempfile import TemporaryDirectory
import mlx.core as mx
from mlx.core import metal as metal_compat
from PIL import Image
from pathlib import Path
from flask import Flask, request, Response, jsonify
from flask_restx import Api, Resource, fields
from flask_cors import CORS
from flask import send_file, redirect
from huggingface_hub import snapshot_download
from mflux.models.common.config import ModelConfig
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.models.fibo.variants.txt2img.fibo import FIBO
from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit
from mflux.models.ernie_image import ErnieImage
from mflux.models.krea2 import Krea2
from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition
from mflux.models.ideogram4 import Ideogram4
from mflux.models.ideogram4.latent_creator import Ideogram4LatentCreator
from mflux.models.ideogram4.model.ideogram4_scheduler import Ideogram4Scheduler
from mflux.models.ideogram4.model.ideogram4_text_encoder import Ideogram4PromptEncoder
from mflux.models.z_image.variants.z_image import ZImage
from qwen21_context import cached_qwen21_context
from qwen21_edit import MAX_REFERENCE_IMAGES, generate_qwen21_references
from krea2_inference import optimized_krea2
from flux2_inference import FLUX2_4B_NAMES, optimized_flux2

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
DEFAULT_MODEL = "flux2-klein-4b"
model = DEFAULT_MODEL
model_quantize = None # quantization level in use
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
    "z-image-turbo": {"loader": "z-image", "steps": 9},
    "filipstrand/Z-Image-Turbo-mflux-4bit": {"loader": "z-image", "steps": 9},
    "flux2-klein-9b": {"loader": "flux2", "steps": 4, "guidance": 1.0},
    "black-forest-labs/FLUX.2-klein-9B": {"loader": "flux2", "steps": 4, "guidance": 1.0},
    "flux2-klein-4b": {"loader": "flux2", "steps": 4, "guidance": 1.0},
    "black-forest-labs/FLUX.2-klein-4B": {"loader": "flux2", "steps": 4, "guidance": 1.0},
    "ernie-image-turbo": {"loader": "ernie", "steps": 8, "guidance": 1.0},
    "baidu/ERNIE-Image-Turbo": {"loader": "ernie", "steps": 8, "guidance": 1.0},
    "ernie-image": {"loader": "ernie", "steps": 50, "guidance": 4.0},
    "baidu/ERNIE-Image": {"loader": "ernie", "steps": 50, "guidance": 4.0},
    "krea2": {"loader": "krea2", "steps": 8, "guidance": 1.0},
    "krea-2": {"loader": "krea2", "steps": 8, "guidance": 1.0},
    "krea-2-turbo": {"loader": "krea2", "steps": 8, "guidance": 1.0},
    "krea/Krea-2-Turbo": {"loader": "krea2", "steps": 8, "guidance": 1.0},
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
    loader = MODEL_REGISTRY[model_name]["loader"]
    limit = {"ideogram4": 0, "flux2": 4, "qwen21": MAX_REFERENCE_IMAGES}.get(loader, 1)
    return {"edit": limit > 0, "multi_image_edit": limit > 1, "max_init_images": limit}


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
    decoded = []
    for field, value in entries:
        try:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("expected a nonempty string")
            value = value.strip()
            if value.lower().startswith("data:"):
                header, value = value.split(",", 1)
                if not header.lower().startswith("data:image/") or not header.lower().endswith(";base64"):
                    raise ValueError("expected an image data URL")
            raw = base64.b64decode("".join(value.split()), validate=True)
            with Image.open(io.BytesIO(raw)) as source:
                source.load()
                decoded.append(source.copy())
        except (ValueError, binascii.Error, OSError, Image.DecompressionBombError) as exc:
            raise ValueError(f"{field}: invalid base64 image or image data URL.") from exc
    return decoded


def load_model(model_name: str, quantize: int | None):
    info = MODEL_REGISTRY.get(model_name, {})
    loader = info.get("loader")
    effective_quantize = quantize if quantize is not None else info.get("quantize")
    model_path = model_name if "/" in model_name else None
    if loader == "flux":
        return Flux1.from_name(quantize=effective_quantize, model_name=model_name)
    if loader == "flux2":
        normalized_name = _normalize_flux2_model_name(model_name)
        return Flux2KleinEdit(
            model_config=ModelConfig.from_name(model_name=normalized_name),
            quantize=effective_quantize,
            model_path=model_path,
        )
    if loader == "qwen":
        return QwenImage(quantize=effective_quantize, model_path=model_path)
    if loader == "qwen21":
        return QwenImage21(quantize=effective_quantize, model_path=model_path)
    if loader == "ideogram4":
        return Ideogram4(
            model_config=ModelConfig.ideogram4_fp8(),
            quantize=effective_quantize,
            model_path=model_path,
        )
    if loader == "krea2":
        model_config = ModelConfig.krea2()
        # mflux 0.18.1's cache check overlooks the root turbo.safetensors file
        # when component subdirectories are present. Complete the snapshot first.
        model_path = snapshot_download(
            repo_id=model_config.model_name,
            allow_patterns=Krea2WeightDefinition.get_download_patterns(),
        )
        return Krea2(
            model_config=model_config,
            quantize=effective_quantize,
            model_path=model_path,
        )
    if loader == "fibo":
        return FIBO(quantize=effective_quantize, model_path=model_path)
    if loader == "ernie":
        model_config_factory = ERNIE_CONFIG_MAP.get(model_name.lower())
        if model_config_factory is None:
            raise ValueError(f"Unknown ERNIE model '{model_name}'")
        return ErnieImage(
            model_config=model_config_factory(),
            quantize=effective_quantize,
            model_path=model_path,
        )
    if loader == "z-image":
        return ZImage(
            model_config=ModelConfig.from_name(model_name="z-image-turbo"),
            quantize=effective_quantize,
            model_path=model_path,
        )
    raise ValueError(f"Unknown model loader for '{model_name}'")

def generate_with_model(instance, model_name: str, task, init_image_path):
    info = MODEL_REGISTRY.get(model_name, {})
    # Accept the former single-path call form as well as the worker's ordered list.
    paths = list(init_image_path) if isinstance(init_image_path, (list, tuple)) else (
        [init_image_path] if init_image_path is not None else []
    )
    validate_image_count(model_name, len(paths))
    init_image_path = paths[0] if paths else None
    if info.get("loader") == "ideogram4":
        if init_image_path is not None:
            raise ValueError("Ideogram 4 does not support init_image (image-to-image generation).")
        # Passing steps would replace the preset's per-step guidance schedule.
        return instance.generate_image(
            seed=int(task['seed']),
            prompt=task['prompt'],
            height=task['height'],
            width=task['width'],
            preset=task.get('preset', IDEOGRAM4_DEFAULT_PRESET),
            strict_caption_validation=task.get('strict_caption_validation', False),
        )
    steps = task['steps'] or MODEL_REGISTRY.get(model_name, {}).get("steps", 4)
    guidance = task['guidance'] or info.get("guidance", 3.5)
    prompt = task['prompt']
    if info.get("loader") == "fibo":
        try:
            json.loads(prompt)
        except json.JSONDecodeError:
            prompt = json.dumps({"prompt": prompt})
    common_kwargs = {
        "seed": int(task['seed']),
        "prompt": prompt,
        "num_inference_steps": steps,
        "height": task['height'],
        "width": task['width'],
        "image_path": init_image_path,
        "image_strength": 0.4 if init_image_path else None
    }
    if info.get("loader") == "flux2":
        common_kwargs.pop("image_path")
        common_kwargs.pop("image_strength")
        # Reference conditioning starts from fresh noise; no img2img strength applies.
        with optimized_flux2(instance, enabled=flux2_optimizations):
            return instance.generate_image(**common_kwargs, image_paths=paths or None, guidance=1.0)
    if info.get("loader") == "z-image":
        return instance.generate_image(**common_kwargs)
    if info.get("loader") == "qwen21":
        common_kwargs["negative_prompt"] = task.get("negative_prompt")
        if paths:
            common_kwargs.pop("image_path")
            common_kwargs.pop("image_strength")
            return generate_qwen21_references(instance, paths, **common_kwargs, guidance=guidance,
                                              use_cache=qwen21_context_cache)
        with cached_qwen21_context(instance, enabled=qwen21_context_cache):
            return instance.generate_image(**common_kwargs, guidance=guidance)
    if info.get("loader") == "krea2":
        common_kwargs["negative_prompt"] = task.get("negative_prompt")
        with optimized_krea2(instance, enabled=krea2_optimizations):
            return instance.generate_image(**common_kwargs, guidance=guidance)
    return instance.generate_image(**common_kwargs, guidance=guidance)

def _load_model_runtime_now(model_name: str, quantize: int | None):
    global model_instance, model, model_quantize
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'")
    info = MODEL_REGISTRY.get(model_name, {})
    effective_quantize = quantize if quantize is not None else info.get("quantize")
    repo_id = model_name if "/" in model_name else None
    cache_before = _hf_repo_cached(repo_id) if repo_id else None
    start_time = time.time()
    print(f"Model load started: '{model_name}'")
    try:
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
        model = model_name
        model_quantize = effective_quantize
        model_instance = loaded_instance


def load_model_runtime(model_name: str, quantize: int | None):
    if threading.current_thread() is model_worker_thread:
        _load_model_runtime_now(model_name, quantize)
        return

    done = threading.Event()
    request = {
        "model": model_name,
        "quantize": quantize,
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


def process_image_task(task, instance, model_name, quantize, worker_stream):
    """Run one queued request and release every temporary reference on all exit paths."""
    global pixels, ctime
    task['compute_time'] = time.time()
    task['model_used'] = model_name
    task['quantize_used'] = quantize
    try:
        inputs = task.get('init_images', [])
        if inputs and (
            task.get('model_at_submit', model_name) != model_name
            or task.get('quantize_at_submit', quantize) != quantize
        ):
            raise ValueError("The loaded model changed after this image-edit request was queued; submit it again.")
        validate_image_count(model_name, len(inputs))
        _set_mlx_cache_limit(metal_cache_limit)
        with TemporaryDirectory(prefix="mflux-inputs-") as directory:
            paths = []
            for index, input_image in enumerate(inputs):
                path = Path(directory) / f"reference-{index + 1}.png"
                input_image.save(path, format="PNG")
                paths.append(path)
            with mx.stream(worker_stream):
                result = generate_with_model(instance, model_name, task, paths)
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
            output.save(encoded, format=output_format, **options)
            encoded.seek(0)
        # Publish only once encoding and temporary-file cleanup have succeeded.
        ctime += time.time() - task['compute_time']
        pixels += output.width * output.height
        task['format'] = output_format
        task['image'] = encoded
    except Exception as exc:
        task['error'] = f"{type(exc).__name__}: {exc}"
        app.logger.exception("Image generation failed for task %s", task['task_id'])
    finally:
        task['end_time'] = time.time()
        task.pop('init_images', None)
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
    'prompt': fields.String(description='The textual description, or a JSON caption encoded as a string for Ideogram 4.', default='A beautiful landscape', required=True),
    'init_image': EncodedImageField(description='One base64 image or image data URL; null or empty means no image. Cannot be combined with a nonempty init_images array.'),
    'init_images': fields.List(fields.String, description='Ordered base64 images or image data URLs for one output. Empty means no inputs. See /api/info for the model limit.'),
    'negative_prompt': fields.String(description='Qwen Image 2.1 or Krea 2 negative conditioning; use guidance above 1.', required=False),
    'seed': fields.String(description='Entropy Seed', default=str(int(time.time())), required=False),
    'height': fields.Integer(description='Image height', default=1024, required=False),
    'width': fields.Integer(description='Image width', default=1024, required=False),
    'steps': fields.Integer(description='Inference Steps (ignored by Ideogram 4; use preset)', default=MODEL_REGISTRY.get(model, {}).get("steps", 4), required=False),
    'guidance': fields.Float(description='Guidance Scale (ignored by Ideogram 4; use preset)', default=MODEL_REGISTRY.get(model, {}).get("guidance", 3.5), required=False),
    'preset': fields.String(description='Ideogram 4 sampler preset', enum=list(IDEOGRAM4_PRESETS), default=IDEOGRAM4_DEFAULT_PRESET, required=False),
    'strict_caption_validation': fields.Boolean(description='Reject Ideogram 4 caption warnings before queuing', default=False, required=False),
    'format': fields.String(description='The image format (JPEG or PNG), default is JPEG', default="JPEG", required=False),
    'quality': fields.Integer(description='JPEG compression quality (1-100) if format is JPEG, default is 85', default=85, required=False),
    'priority': fields.Boolean(description='Set to true to put this task to the head of the queue', default=False, required=False)
})

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
        return jsonify({name: {**settings, **image_capabilities(name)} for name, settings in MODEL_REGISTRY.items()})

info_model = api.model('ModelInfo', {
    'model': fields.String(description='Currently selected model'),
    'quantize': fields.Integer(description='Current quantization, or null'),
    'edit': fields.Boolean(description='Accepts at least one input image (including ordinary img2img)'),
    'multi_image_edit': fields.Boolean(description='Combines multiple ordered references into one output'),
    'max_init_images': fields.Integer(description='Server input-image limit for this model'),
    'default_steps': fields.Integer(description='Default inference steps'),
    'default_guidance': fields.Float(description='Default guidance; ignored by models without guidance'),
    'context_kv_cache': fields.Boolean(description='Qwen 2.1 request-local prefix reuse enabled; unsupported inputs use native execution'),
    'inference_optimizations': fields.List(fields.String, description='Enabled model-specific inference optimizations'),
})

@api.route('/info')
class GetInfo(Resource):
    @api.response(200, 'Success', info_model)
    def get(self):
        with model_lock:
            name, quantize = model, model_quantize
        settings = MODEL_REGISTRY[name]
        return jsonify({
            'model': name, 'quantize': quantize,
            **image_capabilities(name),
            'default_steps': settings.get('steps', 4),
            'default_guidance': settings.get('guidance', 3.5),
            'context_kv_cache': qwen21_context_cache and settings.get('loader') == 'qwen21',
            'inference_optimizations': (
                ['prepared_text', 'cached_geometry', 'grouped_attention', 'image_only_output']
                if krea2_optimizations and settings.get('loader') == 'krea2' else
                ['prompt_cache', 'reference_encode_cache', 'predictor_cache']
                if flux2_optimizations and name in FLUX2_4B_NAMES else []
            ),
        })

@api.route('/ps')
class GetSettings(Resource):
    @api.response(200, 'Success')
    def get(self):
        """
        The /ps endpoint provides the current server settings and default model.
        """
        return jsonify({
            "model": model,
            **image_capabilities(model),
            "quantize": model_quantize,
            "cache_limit": metal_cache_limit,
            "default_steps": MODEL_REGISTRY.get(model, {}).get("steps", 4),
            "default_preset": MODEL_REGISTRY.get(model, {}).get("preset")
        })

@api.route('/load')
class LoadModel(Resource):
    @api.response(200, 'Success')
    @api.response(400, 'Invalid model')
    def post(self):
        """
        The /load endpoint replaces the currently loaded model.
        """
        args = request.json or {}
        requested_model = args.get('model')
        if not requested_model:
            return {"error": "model is required"}, 400
        requested_quantize = args.get('quantize', None)
        try:
            if requested_quantize is not None:
                requested_quantize = int(requested_quantize)
            load_model_runtime(requested_model, requested_quantize)
        except ValueError as exc:
            return {"error": str(exc)}, 400
        except Exception as exc:
            if _is_gated_repo_error(exc):
                return {"error": "Model access appears gated; login to Hugging Face is required."}, 401
            raise
        return {
            "model": model,
            **image_capabilities(model),
            "quantize": model_quantize,
            "default_steps": MODEL_REGISTRY.get(model, {}).get("steps", 4),
            "default_preset": MODEL_REGISTRY.get(model, {}).get("preset")
        }
    
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
        # Parse the JSON body into a dictionary
        args = request.json
        prompt = args.get('prompt', 'A beautiful landscape')
        seed = args.get('seed', str(int(time.time())))
        height = int(args.get('height', 1024))
        width = int(args.get('width', 1024))
        with model_lock:
            model_at_submit = model
            quantize_at_submit = model_quantize
        steps = int(args.get('steps', MODEL_REGISTRY[model_at_submit].get("steps", 4)))
        guidance = float(args.get('guidance', MODEL_REGISTRY[model_at_submit].get("guidance", 3.5)))
        format = args.get('format', 'JPEG').upper()
        quality = args.get('quality', 85)
        priority = args.get('priority', False)

        preset = args.get('preset', IDEOGRAM4_DEFAULT_PRESET)
        strict_caption_validation = args.get('strict_caption_validation', False)
        if MODEL_REGISTRY[model_at_submit].get("loader") == "qwen21":
            if width <= 0 or height <= 0 or width % 16 or height % 16:
                return {"error": "Qwen Image 2.1 requires positive width and height divisible by 16."}, 400
            if steps < 2:
                return {"error": "Qwen Image 2.1 requires at least 2 steps."}, 400
        if MODEL_REGISTRY[model_at_submit].get("loader") == "ideogram4":
            try:
                Ideogram4LatentCreator.validate_dimensions(width=width, height=height)
                steps = Ideogram4Scheduler.get_preset(preset).num_steps
                Ideogram4PromptEncoder.resolve_prompt(
                    prompt,
                    strict_caption_validation=strict_caption_validation,
                    warn_on_caption_issues=False,
                )
                int(seed)
            except ValueError as exc:
                return {"error": str(exc)}, 400
            guidance = None  # Guidance is a per-step schedule selected by the preset.

        try:
            init_images = decode_input_images(args, model_at_submit)
        except ValueError as exc:
            return {"error": str(exc)}, 400

        start_time = time.time()
        # taskid is a 8-digit hex hash to identify the image
        md5 = hashlib.md5()
        md5.update(str(start_time).encode())
        task_id = md5.hexdigest()[:8]

        task_metadata = {
            'task_id': task_id,
            'prompt': prompt,
            'seed': seed,
            'height': height,
            'width': width,
            'steps': steps,
            'guidance': guidance,
            'format': format,
            'quality': quality,
            'priority': priority,
            'start_time': start_time,
            'init_images': init_images,
            'init_image_count': len(init_images),
            'model_at_submit': model_at_submit,
            'quantize_at_submit': quantize_at_submit
        }
        if MODEL_REGISTRY[model_at_submit].get("loader") == "ideogram4":
            task_metadata['preset'] = preset
            task_metadata['strict_caption_validation'] = strict_caption_validation
        if MODEL_REGISTRY[model_at_submit].get("loader") in ("qwen21", "krea2"):
            task_metadata['negative_prompt'] = args.get('negative_prompt')
        
        # compute waiting time based on the number of pixels in the queue
        wait_for_pixels = width * height # include the current task
        if priority and len(tasklist) > 1:
            wait_for_pixels += count_pixels(1)
            tasklist.insert(1, task_metadata)
        else:
            wait_for_pixels += count_pixels(len(tasklist))
            tasklist.append(task_metadata)
        worker_wakeup.set()

        expected_time_seconds = ctime * wait_for_pixels / pixels
        return {
            'task_id': task_id,
            'task_length': len(tasklist) - 1,
            'expected_time_seconds': expected_time_seconds,
            'model': model_at_submit,
            'quantize': quantize_at_submit
        }, 200

status_model = api.model('Status', {
    'status': fields.String(description='Status of the image generation task', enum=['waiting', 'done', 'error']),
    'error': fields.String(description='Failure description when status is error'),
    'pos': fields.Integer(description='Position in queue')
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
        A failed task returns { "status": "error", "error": "..." } and is not retried.
        If the task / the task_id is unknown, the endpoint returns a 404 status code.
        """
        task_id = request.args.get('task_id', default='')
        c = -1
        for i, task in enumerate(tasklist):
            if 'image' not in task and 'error' not in task: c += 1
            if task['task_id'] == task_id:
                if 'error' in task:
                    return jsonify({'status': 'error', 'error': task['error']})
                elif 'image' in task:
                    return jsonify({'status': 'done'})
                else:
                    # compute the remaining time
                    wait_remaining = count_pixels(i + 1) * ctime / pixels
                    start_time = task.get('start_time', 0)
                    compute_time = task.get('compute_time', start_time)
                    wait_remaining = int(wait_remaining - (time.time() - compute_time))
                    if wait_remaining < 1: wait_remaining = 1
                    return jsonify({'status': 'waiting', 'pos': c, 'wait_remaining': wait_remaining})
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
    **{key: field for key, field in task_model.items() if key not in ('init_image', 'init_images')},
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
            for field in ('image', 'init_images', 'init_image'):
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
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, choices=MODEL_REGISTRY.keys(), help='The model to use (default: flux2-klein-4b).')
    parser.add_argument('--quantize',  "-q", type=int, choices=[4, 8], default=None, help='Quantize the model (4 or 8, Default is None)')
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

    global metal_cache_limit, qwen21_context_cache, krea2_optimizations, flux2_optimizations
    metal_cache_limit = args.cache_limit
    qwen21_context_cache = args.qwen21_context_cache
    krea2_optimizations = args.krea2_optimizations
    flux2_optimizations = args.flux2_optimizations
    threading.Thread(target=compute_image_task, daemon=True).start()
    load_model_runtime(args.model, args.quantize)
    print(f"Server started, view swagger API documentation at http://{args.host}:{args.port}/swagger")
    app.run(host=args.host, port=args.port)

if __name__ == '__main__':
    main()
