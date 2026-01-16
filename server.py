import os
import time
import argparse
import threading
import uuid
import base64
import io
import json
import logging
import tempfile
from typing import Dict, Any, List, Optional
from flask import Flask, request, jsonify, abort, Response
from flask_cors import CORS
from server_adapters import ZImageTurboAdapter, FluxAdapter, QwenAdapter, FIBOAdapter, ModelAdapter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

base_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__,
            static_folder=os.path.join(base_dir, 'clients/web-ui'),
            static_url_path='')
CORS(app)

# Global state
model_adapter: Optional[ModelAdapter] = None
configured_model_path: Optional[str] = None
task_queue: List[Dict[str, Any]] = []  # List behaving as a queue
results: Dict[str, Any] = {}         # Store results by task_id
queue_lock = threading.Lock()
result_lock = threading.Lock()
shutdown_event = threading.Event()

# Constants
ResultCleanupTime = 300 # Seconds to keep results around if not picked up (though we use sync wait)

def worker_loop():
    """
    Background thread that processes tasks from the queue.
    """
    global model_adapter

    logger.info("Worker thread started")

    while not shutdown_event.is_set():
        task = None

        # safely pop a task
        with queue_lock:
            if task_queue:
                task = task_queue.pop(0)

        if task:
            task_id = task['id']
            files_to_cleanup = []
            try:
                logger.info(f"Processing task {task_id}")

                # Parse parameters
                prompt = task['params'].get('prompt')
                requested_model = task['params'].get('model')

                # Dynamic Model Loading
                if requested_model:
                    # Check if we need to load or switch models
                    if model_adapter is None or model_adapter.model_name != requested_model:
                        logger.info(f"Switching model to {requested_model}...")
                        try:
                            load_model(requested_model)
                        except Exception as e:
                            logger.error(f"Failed to load model {requested_model}: {e}")
                            raise RuntimeError(f"Failed to load model {requested_model}: {e}")
                elif model_adapter is None:
                     raise RuntimeError("No model loaded and no model specified in request.")

                # OpenAI uses 'size' like "1024x1024", we need to parse it or expect width/height
                size = task['params'].get('size', "1024x1024")
                if isinstance(size, str) and "x" in size:
                    try:
                        w, h = map(int, size.split('x'))
                        width, height = w, h
                    except:
                        width, height = 1024, 1024
                else:
                    width = task['params'].get('width', 1024)
                    height = task['params'].get('height', 1024)

                # Extract other known params and pass rest as kwargs
                kwargs = {
                    'width': width,
                    'height': height,
                    'num_inference_steps': task['params'].get('steps', 4), # Default 4 for turbo
                    'seed': task['params'].get('seed', int(time.time())),
                    'scheduler': task['params'].get('scheduler', 'linear')
                }

                # Add any other extra parameters
                for k, v in task['params'].items():
                    if k not in ['prompt', 'n', 'size', 'response_format', 'model', 'steps', 'seed', 'width', 'height', 'scheduler', 'init_image', 'init_image_path', 'mask_image_path']:
                        kwargs[k] = v

                # Handle init_image (Base64)
                if 'init_image' in task['params']:
                    try:
                        init_image_data = task['params']['init_image']
                        # Sometimes base64 strings come with data:image/png;base64, prefix
                        if ',' in init_image_data:
                            init_image_data = init_image_data.split(',')[1]

                        img_bytes = base64.b64decode(init_image_data)

                        # Create a temporary file for the init image
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                            tmp.write(img_bytes)
                            files_to_cleanup.append(tmp.name)

                        kwargs['init_image_path'] = files_to_cleanup[-1]
                        logger.info(f"Decoded init_image to {kwargs['init_image_path']}")
                    except Exception as e:
                        logger.error(f"Failed to decode init_image: {e}")
                        # Continue without init_image or fail? For now, we'll try to continue or it might fail later

                # Handle paths passed directly (from edits endpoint)
                if 'init_image_path' in task['params']:
                    kwargs['init_image_path'] = task['params']['init_image_path']
                    files_to_cleanup.append(task['params']['init_image_path'])

                if 'mask_image_path' in task['params']:
                    kwargs['mask_image_path'] = task['params']['mask_image_path']
                    files_to_cleanup.append(task['params']['mask_image_path'])

                # Generate
                start_time = time.time()

                if model_adapter is None:
                    raise RuntimeError("No model loaded. Please restart the server with a --model argument or ensure a model is loaded.")

                image = model_adapter.generate(prompt=prompt, **kwargs)
                generation_time = time.time() - start_time
                logger.info(f"Generation for {task_id} completed in {generation_time:.2f}s")

                # Store result
                with result_lock:
                    results[task_id] = {
                        'status': 'completed',
                        'image': image,
                        'created': int(time.time()),
                        'params': task['params']
                    }

            except Exception as e:
                logger.error(f"Error processing task {task_id}: {str(e)}")
                with result_lock:
                    results[task_id] = {
                        'status': 'failed',
                        'error': str(e)
                    }
            finally:
                # Clean up temporary files
                for path in files_to_cleanup:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                            logger.info(f"Cleaned up temporary file {path}")
                        except Exception as e:
                            logger.error(f"Error removing temporary file {path}: {e}")
        else:
            time.sleep(0.1)

@app.route('/v1/images/generations', methods=['POST'])
def generate_image():
    if not request.is_json:
        return jsonify({"error": {"message": "Request must be JSON", "type": "invalid_request_error", "code": "invalid_json"}}), 400

    data = request.json

    # Validate required fields
    if 'prompt' not in data:
         return jsonify({"error": {"message": "Missing required parameter 'prompt'", "type": "invalid_request_error", "code": "missing_required_parameter"}}), 400

    # Create task
    task_id = str(uuid.uuid4())
    task = {
        'id': task_id,
        'params': data,
        'created_at': time.time()
    }

    logger.info(f"Queuing task {task_id}")
    with queue_lock:
        task_queue.append(task)

    # Wait for result (Blocking the request)
    # Timeout after 60 seconds (or configurable)
    timeout = 600 # 10 minutes wait time max
    start_wait = time.time()

    while time.time() - start_wait < timeout:
        with result_lock:
            if task_id in results:
                result = results[task_id]
                # Clean up result immediately as we are returning it (unless we want to support async retrieval later)
                del results[task_id]

                if result['status'] == 'failed':
                    return jsonify({"error": {"message": result.get('error', 'Unknown error'), "type": "server_error", "code": "generation_failed"}}), 500

                # Format success response
                image = result['image']
                response_format = data.get('response_format', 'url') # OpenAI defaults to url, but we might default to b64_json as we don't host images yet

                # For this local server, let's default to b64_json if url isn't easy to implement without file hosting
                # If user asks for url, we could save to tmp and return local url, but b64_json is safer for pure API.
                # Let's support b64_json.

                resp_data = []

                # Handle 'n' (number of images) - simplistic loop if adapter didn't handle it
                # For now, we assume adapter generated one image.
                # To support n>1 properly, the worker should have generated n images.
                # MVP: n=1 support only for now.

                if response_format == 'b64_json':
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    resp_data.append({"b64_json": img_str})
                else:
                    # Fallback or 'url' request - since we don't have a static file server setup in this snippet yet
                    # We'll just return b64_json with a warning or just b64_json pretending it's what was asked
                    # Or better, let's implement a quick base64 return even for url for this MVP
                     buffered = io.BytesIO()
                     image.save(buffered, format="PNG")
                     img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                     resp_data.append({"b64_json": img_str, "msg": "returned b64_json as url hosting is not configured"})

                return jsonify({
                    "created": result['created'],
                    "data": resp_data
                })

        time.sleep(0.5)

    return jsonify({"error": {"message": "Request timed out", "type": "server_error", "code": "timeout"}}), 504


@app.route('/v1/images/edits', methods=['POST'])
def edit_image():
    # OpenAI compatible endpoint for image edits (inpainting/img2img)
    # Uses multipart/form-data

    if 'image' not in request.files:
         return jsonify({"error": {"message": "Missing required parameter 'image'", "type": "invalid_request_error", "code": "missing_required_parameter"}}), 400

    image_file = request.files['image']
    mask_file = request.files.get('mask')

    # Extract form parameters
    prompt = request.form.get('prompt')
    if not prompt:
         return jsonify({"error": {"message": "Missing required parameter 'prompt'", "type": "invalid_request_error", "code": "missing_required_parameter"}}), 400

    # Optional parameters
    n = int(request.form.get('n', 1))
    size = request.form.get('size', "1024x1024")
    response_format = request.form.get('response_format', 'url')
    # model = request.form.get('model', 'z-image-turbo') # Unused for now as we have a single loaded model

    # Parse generic params usually passed in JSON
    # We might receive 'steps', 'seed', 'scheduler' etc in form data
    params = {}
    for key, value in request.form.items():
        if key not in ['image', 'mask']:
            # Try to convert to int/float if possible for known numeric params
            if key in ['steps', 'num_inference_steps', 'seed', 'width', 'height', 'guidance']:
                try:
                    params[key] = int(value)
                except:
                    try:
                        params[key] = float(value)
                    except:
                        params[key] = value
            else:
                params[key] = value

    # Save uploaded files to temporary paths
    # These will be passed to the worker which is responsible for cleanup
    try:
        init_image_path = None
        mask_image_path = None

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            image_file.save(tmp.name)
            init_image_path = tmp.name
            params['init_image_path'] = init_image_path

        if mask_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                mask_file.save(tmp.name)
                mask_image_path = tmp.name
                params['mask_image_path'] = mask_image_path

    except Exception as e:
        logger.error(f"Failed to save uploaded files: {e}")
        # Clean up if partial failure
        if init_image_path and os.path.exists(init_image_path):
            os.remove(init_image_path)
        if mask_image_path and os.path.exists(mask_image_path):
            os.remove(mask_image_path)
        return jsonify({"error": {"message": f"Failed to process uploaded files: {str(e)}", "type": "server_error", "code": "file_upload_error"}}), 500

    # Create task
    task_id = str(uuid.uuid4())
    task = {
        'id': task_id,
        'params': params,
        'created_at': time.time()
    }

    logger.info(f"Queuing edit task {task_id}")
    with queue_lock:
        task_queue.append(task)

    # Wait for result (Blocking the request)
    timeout = 600
    start_wait = time.time()

    while time.time() - start_wait < timeout:
        with result_lock:
            if task_id in results:
                result = results[task_id]
                del results[task_id]

                if result['status'] == 'failed':
                    return jsonify({"error": {"message": result.get('error', 'Unknown error'), "type": "server_error", "code": "generation_failed"}}), 500

                image = result['image']

                resp_data = []
                if response_format == 'b64_json':
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    resp_data.append({"b64_json": img_str})
                else:
                     buffered = io.BytesIO()
                     image.save(buffered, format="PNG")
                     img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                     resp_data.append({"b64_json": img_str, "msg": "returned b64_json as url hosting is not configured"})

                return jsonify({
                    "created": result['created'],
                    "data": resp_data
                })

        time.sleep(0.5)

    return jsonify({"error": {"message": "Request timed out", "type": "server_error", "code": "timeout"}}), 504



@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model": model_adapter.model_name if model_adapter else "none"})

def load_model(model_name: str):
    """
    Loads or switches the model adapter based on the requested model name.
    """
    global model_adapter, configured_model_path

    logger.info(f"Attempting to load model: {model_name}")

    if model_name == 'z-image-turbo' or "z-image-turbo" in model_name:
        new_adapter = ZImageTurboAdapter()
        new_adapter.load(model_name, model_path=configured_model_path)
        model_adapter = new_adapter
    elif model_name in ['schnell', 'dev'] or 'flux' in model_name.lower():
        new_adapter = FluxAdapter()
        new_adapter.load(model_name, model_path=configured_model_path)
        model_adapter = new_adapter
    elif 'qwen' in model_name.lower():
        new_adapter = QwenAdapter()
        new_adapter.load(model_name, model_path=configured_model_path)
        model_adapter = new_adapter
    elif 'fibo' in model_name.lower():
        new_adapter = FIBOAdapter()
        new_adapter.load(model_name, model_path=configured_model_path)
        model_adapter = new_adapter
    else:
        # Fallback: try to guess based on standard names or fail
        # For now, we'll try Flux if it looks like a repo, otherwise error
        # Actually, let's error if we can't determine type, or assume Flux as generic
        if "flux" in model_name.lower():
             new_adapter = FluxAdapter()
             new_adapter.load(model_name, model_path=configured_model_path)
             model_adapter = new_adapter
        else:
            raise ValueError(f"Unknown model type for: {model_name}")

    logger.info(f"Successfully loaded model: {model_name}")

def scan_models(model_path: Optional[str]) -> List[Dict[str, Any]]:
    """
    Scans the model_path for available models.
    Expected structure: {model_path}/{org}/{repo}
    """
    models = []

    # Always include the currently loaded model if it exists
    if model_adapter and model_adapter.model_name:
        models.append({
            "id": model_adapter.model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "mflux"
        })

    if not model_path or not os.path.exists(model_path):
        return models

    # Scan the directory
    try:
        for org in os.listdir(model_path):
            if org.startswith('.'): continue
            org_path = os.path.join(model_path, org)
            if os.path.isdir(org_path):
                for repo in os.listdir(org_path):
                    if repo.startswith('.'): continue
                    repo_path = os.path.join(org_path, repo)
                    if os.path.isdir(repo_path):
                        model_id = f"{org}/{repo}"
                        # Check if this model ID is already in the list
                        if not any(m["id"] == model_id for m in models):
                            models.append({
                                "id": model_id,
                                "object": "model",
                                "created": int(os.path.getctime(repo_path)),
                                "owned_by": org
                            })
    except Exception as e:
        logger.error(f"Error scanning models in {model_path}: {e}")

    return models

@app.route('/', methods=['GET'])
def index():
    return app.send_static_file('index.html')

@app.route('/v1/models', methods=['GET'])
def list_models():
    models = scan_models(configured_model_path)
    return jsonify({
        "object": "list",
        "data": models
    })

def main():
    global model_adapter, configured_model_path

    parser = argparse.ArgumentParser(description='OpenAI-compatible Image Generation Server')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=4030, help='Port to listen on')
    parser.add_argument('--quantize', type=int, default=None, help='Quantization level')
    parser.add_argument('--model_path', type=str, default=None, help='Base path for pre-converted MLX models')
    parser.add_argument('--cache_limit', type=int, default=0, help='The metal cache limit in bytes')
    parser.add_argument('--low-ram', action='store_true', help='Enable Low-RAM mode')

    args = parser.parse_args()
    configured_model_path = args.model_path

    logger.info("No model loaded at startup. Use API to load a model.")

    # Start worker thread
    worker = threading.Thread(target=worker_loop, daemon=True)
    worker.start()

    logger.info(f"Server starting on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)

if __name__ == '__main__':
    main()
