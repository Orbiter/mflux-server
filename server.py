import os
import time
import argparse
import threading
import uuid
import base64
import io
import json
import logging
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
            try:
                logger.info(f"Processing task {task_id}")

                # Extract generation parameters
                prompt = task['params'].get('prompt')
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
                    if k not in ['prompt', 'n', 'size', 'response_format', 'model', 'steps', 'seed', 'width', 'height', 'scheduler']:
                        kwargs[k] = v

                # Generate
                start_time = time.time()
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


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model": model_adapter.model_name if model_adapter else "none"})

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
        # If no path, and no model loaded, return a default
        if not models:
            models.append({
                "id": "z-image-turbo",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mflux"
            })
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
    parser.add_argument('--model', type=str, default='z-image-turbo', help='Model to load (default: z-image-turbo)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=4030, help='Port to listen on')
    parser.add_argument('--quantize', type=int, default=None, help='Quantization level')
    parser.add_argument('--model_path', type=str, default=None, help='Base path for pre-converted MLX models')
    parser.add_argument('--cache_limit', type=int, default=0, help='The metal cache limit in bytes')
    parser.add_argument('--low-ram', action='store_true', help='Enable Low-RAM mode')

    args = parser.parse_args()
    configured_model_path = args.model_path

    # Initialize adapter
    if args.model == 'z-image-turbo':
        model_adapter = ZImageTurboAdapter()
        logger.info(f"Loading model {args.model}...")
        model_adapter.load(args.model, quantize=args.quantize, model_path=args.model_path)
    elif args.model in ['schnell', 'dev'] or 'flux' in args.model.lower():
        model_adapter = FluxAdapter()
        logger.info(f"Loading Flux model {args.model}...")
        model_adapter.load(args.model, quantize=args.quantize, model_path=args.model_path)
    elif 'qwen' in args.model.lower():
        model_adapter = QwenAdapter()
        logger.info(f"Loading Qwen model {args.model}...")
        model_adapter.load(args.model, quantize=args.quantize, model_path=args.model_path)
    elif 'fibo' in args.model.lower():
        model_adapter = FIBOAdapter()
        logger.info(f"Loading FIBO model {args.model}...")
        model_adapter.load(args.model, quantize=args.quantize, model_path=args.model_path)
    else:
        logger.error(f"Unknown model: {args.model}")
        return

    # Start worker thread
    worker = threading.Thread(target=worker_loop, daemon=True)
    worker.start()

    logger.info(f"Server starting on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)

if __name__ == '__main__':
    main()
