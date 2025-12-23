# mflux-server
# CUDA-backed server for image generation on Linux using diffusers/torch
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
from pathlib import Path

import torch
from PIL import Image
from flask import Flask, request, Response, jsonify
from flask_restx import Api, Resource, fields
from flask_cors import CORS
from flask import send_file, redirect

import requests

# monkey patching the Session to ignore SSL verification
old_request = requests.Session.request
def new_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return old_request(self, *args, **kwargs)
requests.Session.request = new_request

app = Flask(__name__)
api = Api(app, version='1.0', title='MFLUX CUDA API Server',
          description='An image generation server. Workflow: /generate -> /status -> /image',
          doc='/swagger',
          prefix='/api')

CORS(app, resources={r"/*": {"origins": "*"}})

apppath = os.path.dirname(__file__)
tasklist = []         # list which holds the image computation tasks
model_instance = None # the model object, initialized in main()
pixels = 1024 * 1024  # the number of pixels in all of the computed images (start value)
ctime = 80            # the total computation time for all images in seconds (start value)
cuda_cache_limit = 0  # approximate CUDA memory cap in bytes (0 = no cap)
model = "schnell"     # default model alias
model_quantize = None # quantization level in use (ignored for CUDA)
model_lock = threading.Lock()
tasklist_lock = threading.Lock()
load_lock = threading.Lock()
model_version = 0
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16
device_map = None
devices = []
low_vram = False
load_retries = 0
bnb_4bit = False
workers = 1

MODEL_REGISTRY = {
    # Known FLUX aliases mapped to HF IDs for CUDA/diffusers.
    "dev": {"hf_id": "black-forest-labs/FLUX.1-dev", "steps": 25},
    "schnell": {"hf_id": "black-forest-labs/FLUX.1-schnell", "steps": 4},
    "krea-dev": {"hf_id": "black-forest-labs/FLUX.1-Krea-dev", "steps": 25},
}

def _resolve_model(model_name: str):
    info = MODEL_REGISTRY.get(model_name)
    if info:
        return info["hf_id"], info.get("steps", 25)
    if "/" in model_name:
        return model_name, 25
    raise ValueError(f"Unknown model '{model_name}'")

def _default_steps_for_model(model_name: str) -> int:
    info = MODEL_REGISTRY.get(model_name)
    if info:
        return info.get("steps", 25)
    if "/" in model_name:
        return 25
    return 4

def _parse_seed(seed_value):
    try:
        return int(seed_value)
    except Exception:
        return int(hashlib.md5(str(seed_value).encode()).hexdigest(), 16) % (2**31 - 1)

def _set_cuda_memory_limit(limit: int, device_name: str) -> None:
    if not torch.cuda.is_available() or limit <= 0 or "cuda" not in device_name:
        return
    try:
        props = torch.cuda.get_device_properties(device_name)
        total_bytes = props.total_memory
        fraction = min(1.0, float(limit) / float(total_bytes))
        torch.cuda.set_per_process_memory_fraction(fraction, device=device_name)
    except Exception:
        pass

def _clear_cuda_cache(device_name: str) -> None:
    if torch.cuda.is_available() and "cuda" in device_name:
        torch.cuda.empty_cache()

def _load_pipelines():
    # Lazy import to avoid import-time hard failures when optional deps are missing.
    try:
        from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
        return AutoPipelineForText2Image, AutoPipelineForImage2Image, False
    except Exception:
        from diffusers import DiffusionPipeline
        return DiffusionPipeline, None, True

def load_model(model_name: str, quantize: int | None, device_name: str, device_map_override: str | None):
    # quantize is accepted for API compatibility but ignored here.
    AutoPipelineForText2Image, _, is_fallback = _load_pipelines()
    hf_id, _ = _resolve_model(model_name)
    if bnb_4bit:
        if device_map_override:
            effective_map = device_map_override
        elif device_name.startswith("cuda"):
            try:
                torch.cuda.set_device(device_name)
            except Exception:
                pass
            effective_map = "cuda"
        else:
            effective_map = None
        pipe = AutoPipelineForText2Image.from_pretrained(
            hf_id,
            torch_dtype=torch_dtype,
            device_map=effective_map,
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
    elif device_map_override:
        pipe = AutoPipelineForText2Image.from_pretrained(
            hf_id,
            torch_dtype=torch_dtype,
            device_map=device_map_override,
        )
    else:
        pipe = AutoPipelineForText2Image.from_pretrained(
            hf_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=False,
        )
        pipe = pipe.to(device_name)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass
    try:
        pipe.enable_vae_tiling()
    except Exception:
        pass
    if low_vram and device_name.startswith("cuda"):
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pass
    return {"txt2img": pipe, "fallback": is_fallback}

def _get_img2img_pipe(pipes):
    _, AutoPipelineForImage2Image, is_fallback = _load_pipelines()
    if is_fallback:
        return None
    if "img2img" in pipes:
        return pipes["img2img"]
    pipes["img2img"] = AutoPipelineForImage2Image.from_pipe(pipes["txt2img"])
    return pipes["img2img"]

def generate_with_model(pipes, model_name: str, task, init_image_path, device_name: str):
    _, default_steps = _resolve_model(model_name)
    steps = task['steps'] or default_steps
    guidance = task['guidance'] or 3.5
    prompt = task['prompt']
    generator = torch.Generator(device=device_name).manual_seed(_parse_seed(task['seed']))
    common_kwargs = {
        "prompt": prompt,
        "num_inference_steps": steps,
        "height": task['height'],
        "width": task['width'],
        "guidance_scale": guidance,
        "generator": generator,
    }
    if init_image_path:
        init_image = Image.open(init_image_path).convert("RGB")
        pipe = _get_img2img_pipe(pipes)
        if pipe is None:
            raise RuntimeError("img2img requires AutoPipeline support; upgrade diffusers/transformers.")
        result = pipe(image=init_image, strength=0.4, **common_kwargs)
    else:
        result = pipes["txt2img"](**common_kwargs)
    return result.images[0]

def load_model_runtime(model_name: str, quantize: int | None):
    global model_instance, model, model_quantize, model_version
    loaded_instance = None
    with model_lock:
        model = model_name
        model_quantize = quantize
        model_instance = loaded_instance
        model_version += 1

# we implement image generation as asynchronous task
# this will be executed in a separate thread
def compute_image_task(device_name: str):
    global model_instance, tasklist, pixels, ctime, model_version, load_retries
    local_model_version = -1
    local_model_name = None
    local_pipes = None
    # we loop forever and in every iteration we check if there is a task to process
    while True:
        with model_lock:
            current_model_name = model
            current_model_version = model_version
            current_device_map = device_map
        if local_model_version != current_model_version or local_pipes is None:
            try:
                with load_lock:
                    local_pipes = load_model(current_model_name, model_quantize, device_name, current_device_map)
                local_model_version = current_model_version
                local_model_name = current_model_name
                load_retries = 0
            except Exception as exc:
                print(f"Model load failed on {device_name}: {exc}")
                _clear_cuda_cache(device_name)
                gc.collect()
                load_retries += 1
                time.sleep(min(10, 1 + load_retries))
                continue

        with tasklist_lock:
            if len(tasklist) == 0:
                task = None
            else:
                task = None
                for candidate in tasklist:
                    if 'image' in candidate or candidate.get('in_progress'):
                        continue
                    candidate['in_progress'] = True
                    task = candidate
                    break
        if task is None:
            time.sleep(0.5)
            continue
        
        compute_time = time.time()
        task['compute_time'] = compute_time
        _set_cuda_memory_limit(cuda_cache_limit, device_name)
        init_image = task['init_image']

        if init_image:
            init_image_path = Path(f"/tmp/init_image_{task['task_id']}.png")
            init_image.save(str(init_image_path))
        else:
            init_image_path = None

        try:
            generated_image = generate_with_model(local_pipes, local_model_name, task, init_image_path, device_name)
            end_time = time.time()
            ctime += end_time - compute_time
            pixels += task['height'] * task['width']

            format = task.get('format', 'JPEG').upper()
            if format not in ['PNG', 'JPEG']:
                format = 'JPEG'
            if format == 'PNG':
                png_image = io.BytesIO()
                generated_image.save(png_image, format='PNG')
                png_image.seek(0)
                task['image'] = png_image
                del png_image
            else:
                quality = task['quality']
                jpeg_image = io.BytesIO()
                generated_image.save(jpeg_image, format='JPEG', quality=quality)
                jpeg_image.seek(0)
                task['image'] = jpeg_image
                del jpeg_image
        except Exception as exc:
            task['error'] = str(exc)
            end_time = time.time()
        finally:
            if init_image_path:
                os.remove(init_image_path)
            _clear_cuda_cache(device_name)
            gc.collect()
            task['end_time'] = end_time
            task.pop('in_progress', None)

def str_to_bool(value):
    return value.lower() in ['true', '1', 't', 'y', 'yes']

# generate image endpoint

task_model = api.model('TaskInput', {
    'prompt': fields.String(description='The textual description of the image to generate.', default='A beautiful landscape', required=True),
    'seed': fields.String(description='Entropy Seed', default=str(int(time.time())), required=False),
    'height': fields.Integer(description='Image height', default=1024, required=False),
    'width': fields.Integer(description='Image width', default=1024, required=False),
    'steps': fields.Integer(description='Inference Steps', default=MODEL_REGISTRY.get(model, {}).get("steps", 4), required=False),
    'guidance': fields.Float(description='Guidance Scale', default=3.5, required=False),
    'format': fields.String(description='The image format (JPEG or PNG), default is JPEG', default="JPEG", required=False),
    'quality': fields.Integer(description='JPEG compression quality (1-100) if format is JPEG, default is 85', required=False),
    'priority': fields.Boolean(description='Set to true to put this task to the head of the queue', default=False, required=False)
})

generate_response_model = api.model('GenerateResponse', {
    'task_id': fields.String(description='ID of the image generation task'),
    'task_length': fields.Integer(description='Length of the image generation task queue excluding this new one'),
    'expected_time_seconds': fields.Float(description='Expected time in seconds for the image generation task to complete')
})

# function which counts number of pixels in images from the tasklist up to a certain index
def count_pixels(index):
    global tasklist
    pixels = 0
    for i in range(index):
        if i >= len(tasklist):
            break
        task = tasklist[i]
        if not 'image' in task:
            pixels += task['width'] * task['height']
    return pixels

@api.route('/ls')
class ListModels(Resource):
    @api.response(200, 'Success')
    def get(self):
        """
        The /ls endpoint provides a catalog of available models and defaults.
        """
        return jsonify(MODEL_REGISTRY)

@api.route('/ps')
class GetSettings(Resource):
    @api.response(200, 'Success')
    def get(self):
        """
        The /ps endpoint provides the current server settings and default model.
        """
        return jsonify({
            "model": model,
            "quantize": model_quantize,
            "cache_limit": cuda_cache_limit,
            "default_steps": _default_steps_for_model(model),
            "device": device,
            "device_map": device_map,
            "torch_dtype": str(torch_dtype),
            "devices": devices,
            "low_vram": low_vram,
            "bnb_4bit": bnb_4bit,
            "workers": workers,
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
            return jsonify({"error": "model is required"}), 400
        requested_quantize = args.get('quantize', None)
        try:
            if requested_quantize is not None:
                requested_quantize = int(requested_quantize)
            load_model_runtime(requested_model, requested_quantize)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({
            "model": model,
            "quantize": model_quantize,
            "default_steps": _default_steps_for_model(model)
        })
    
@api.route('/generate')
class GenerateImage(Resource):
    @api.expect(task_model, validate=True)
    @api.response(200, 'Success', generate_response_model)
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
        steps = int(args.get('steps', _default_steps_for_model(model)))
        guidance = float(args.get('guidance', 3.5))
        format = args.get('format', 'JPEG').upper()
        quality = args.get('quality', 85)
        priority = args.get('priority', False)

        # Decode init_image if it is provided
        init_image = None
        if 'init_image' in args:
            try:
                init_image_data = base64.b64decode(args['init_image'])
                init_image = Image.open(io.BytesIO(init_image_data))
                # log properties of the init_image, width, height, mode
                print("init_image", init_image.size, init_image.mode)
            except Exception:
                pass # ignore errors
            
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
            'init_image': init_image
        }
        
        # compute waiting time based on the number of pixels in the queue
        wait_for_pixels = width * height # include the current task
        with tasklist_lock:
            if priority and len(tasklist) > 1:
                wait_for_pixels += count_pixels(1)
                tasklist.insert(1, task_metadata)
            else:
                wait_for_pixels += count_pixels(len(tasklist))
                tasklist.append(task_metadata)

        expected_time_seconds = ctime * wait_for_pixels / pixels
        return {
            'task_id': task_id,
            'task_length': len(tasklist) - 1,
            'expected_time_seconds': expected_time_seconds
        }, 200

status_model = api.model('Status', {
    'status': fields.String(description='Status of the image generation task'),
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
        If the task / the task_id is unknown, the endpoint returns a 404 status code.
        """
        task_id = request.args.get('task_id', default='')
        with tasklist_lock:
            c = -1
            for i, task in enumerate(tasklist):
                if not 'image' in task:
                    c += 1
                if task['task_id'] == task_id:
                    if 'image' in task:
                        return jsonify({'status': 'done'})
                    if 'error' in task:
                        return jsonify({'status': 'error', 'message': task['error']})
                    else:
                        # compute the remaining time
                        wait_remaining = count_pixels(i + 1) * ctime / pixels
                        start_time = task.get('start_time', 0)
                        compute_time = task.get('compute_time', start_time)
                        wait_remaining = int(wait_remaining - (time.time() - compute_time))
                        if wait_remaining < 1:
                            wait_remaining = 1
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
        with tasklist_lock:
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
        with tasklist_lock:
            for task in tasklist:
                if task['task_id'] == task_id:
                    tasklist.remove(task)
                    return Response(status=200)
        return Response(status=404)

task_output_model = api.inherit('TaskOutput', task_model, {
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
        with tasklist_lock:
            for task in tasklist:
                task0 = task.copy()
                if 'image' in task0:
                    del task0['image']
                tasklist0.append(task0)
        return jsonify(tasklist0)

@api.route('/clear')
class ClearTasks(Resource):
    @api.response(200, 'Success')
    def get(self):
        with tasklist_lock:
            tasklist.clear()
        return Response(status=200)

@app.route('/')
def redirect_to_index():
    return redirect('/index.html')

@app.route('/index.html')
def serve_index():
    return send_file(os.path.join(apppath, 'clients/web-ui/index.html'))

def main():
    parser = argparse.ArgumentParser(description='Start a CUDA server to generate images with diffusers.')
    global model_quantize
    global model_instance
    global cuda_cache_limit
    global device
    global device_map
    global torch_dtype
    global devices
    global low_vram
    global bnb_4bit
    global workers

    default_device = device
    parser.add_argument('--model', type=str, default=model, help='The model to use (i.e. "schnell" or a HF model id).')
    parser.add_argument('--quantize',  "-q", type=int, choices=[4, 8], default=None, help='Quantize the model (4 or 8, ignored for CUDA).')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='The host to listen on')
    parser.add_argument('--port', type=int, default=4030, help='The port to listen on')
    parser.add_argument('--cache_limit', type=int, default=0, help='Approximate CUDA memory cap in bytes (0 = no cap)')
    parser.add_argument('--device', type=str, default=default_device, help='Torch device, e.g. cuda, cuda:0, cuda:1, cpu')
    parser.add_argument('--device_map', type=str, default=None, help='Device map for multi-GPU (e.g. auto, balanced)')
    parser.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'bf16', 'fp32'], help='Torch dtype for model weights')
    parser.add_argument('--low_vram', action='store_true', help='Enable CPU offload and VAE slicing to reduce GPU memory use')
    parser.add_argument('--bnb4', action='store_true', help='Enable 4-bit quantization with bitsandbytes')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel model replicas')
    args = parser.parse_args()

    device = args.device
    if args.device_map:
        device_map = args.device_map
    else:
        device_map = None
    low_vram = args.low_vram
    bnb_4bit = args.bnb4
    workers = max(1, int(args.workers))

    if device_map:
        devices = [device]
    elif device.startswith("cuda") and device == "cuda":
        available = torch.cuda.device_count()
        count = min(workers, max(1, available))
        devices = [f"cuda:{i}" for i in range(count)]
    else:
        devices = [device]
    if args.dtype == 'bf16':
        torch_dtype = torch.bfloat16
    elif args.dtype == 'fp32':
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16

    load_model_runtime(args.model, args.quantize)
    cuda_cache_limit = args.cache_limit
    for worker_device in devices:
        threading.Thread(target=compute_image_task, args=(worker_device,), daemon=True).start()
    print(f"Server started, view swagger API documentation at http://{args.host}:{args.port}/swagger")
    app.run(host=args.host, port=args.port)

if __name__ == '__main__':
    main()
