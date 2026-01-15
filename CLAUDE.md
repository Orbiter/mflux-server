# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mflux-server is an asynchronous image generation API server built on top of [mflux](https://github.com/filipstrand/mflux). It provides a Flask-based REST API for queuing and processing image generation tasks, designed for environments where GPU resources are shared across multiple users.

## Running the Server

### Quick Start (Recommended: using uv)
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# macOS (Apple Silicon / MLX backend)
./run.sh

# Linux/CUDA (diffusers/torch backend)
./run-cuda.sh
```

### Manual Setup with uv
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# macOS (Apple Silicon / MLX backend)
uv sync --extra mlx
uv run python server.py --quantize 8 --host 0.0.0.0

# Linux/CUDA (diffusers/torch backend)
uv sync --extra cuda
uv run python server_cuda.py --model schnell --host 0.0.0.0
```

### Legacy Setup (without uv)

#### macOS (Apple Silicon / MLX backend)
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip3.12 install -r requirements.txt
python3.12 server.py --quantize 8 --host 0.0.0.0
```

#### Linux/CUDA (diffusers/torch backend)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements_cuda.txt
python3.12 server_cuda.py --model schnell --host 0.0.0.0
```

### Server Options
- `--model`: Model alias (schnell, dev, krea-dev, qwen, fibo, z-image-turbo) or HF model ID
- `--quantize`: Quantization level (4 or 8, MLX only)
- `--model_path`: Base path for pre-converted MLX models (MLX only)
- `--host`: Host to bind to (default: 127.0.0.1)
- `--port`: Port to listen on (default: 4030)
- `--cache_limit`: Memory cache limit in bytes

Server runs on port 4030 by default with Swagger docs at `/swagger`.

### Custom Model Path

The server supports loading pre-converted MLX models from a custom filesystem path.

**Configuration:**
```bash
# Via CLI argument
uv run python server.py --model_path /Volumes/LLMS/image

# Via environment variable
export MFLUX_MODEL_PATH=/Volumes/LLMS/image
uv run python server.py

# Via API override
curl -X POST http://localhost:4030/api/load \
  -H "Content-Type: application/json" \
  -d '{"model": "schnell", "model_path": "/custom/path"}'
```

When configured, models are loaded from `{model_path}/{org}/{model_name}` instead of the default HuggingFace cache.

## Architecture

### Two Server Implementations

**server.py** - MLX-based (macOS/Apple Silicon)
- Uses `mflux` library with MLX backend
- Supports multiple MLX-native models via `MODEL_REGISTRY`
- Models loaded via mflux's Flux1, QwenImage, FIBO, ZImageTurbo classes

**server_cuda.py** - CUDA-based (Linux/NVIDIA)
- Uses `diffusers` library with torch backend
- Supports FLUX models via HuggingFace IDs
- Supports multi-GPU with `--workers` and `--device_map`
- Optional 4-bit quantization via bitsandbytes (`--bnb4`)

### Common Architecture Patterns

Both servers share the same Flask API structure and core design:

1. **Task Queue System**: Global `tasklist` holds pending/completed generation tasks
2. **Worker Thread Pattern**: `compute_image_task()` runs in background thread, processing tasks sequentially
3. **Model Loading**: Runtime model switching via `/api/load` endpoint
4. **Cache Management**: Platform-specific cache clearing after each generation

### Threading Model

- `model_lock`: Protects model instance changes
- `tasklist_lock`: Protects task queue access (CUDA only)
- Worker thread checks for tasks without assigned images and processes them

### API Workflow

1. `POST /api/generate` - Submit generation request, returns `task_id`
2. `GET /api/status?task_id=xxx` - Poll for completion status
3. `GET /api/image?task_id=xxx` - Retrieve generated image (deleted by default)

### Model Registry

Models are defined in `MODEL_REGISTRY` dict with format:
```python
"alias": {
    "loader": "flux|qwen|fibo|z-image",  # MLX only
    "hf_id": "huggingface/model-id",      # CUDA only
    "steps": default_steps,
    "quantize": default_quantize
}
```

### Key Files

- `server.py` / `server_cuda.py` - Main server implementations
- `pyproject.toml` - Project configuration and dependencies (uv-compatible)
- `run.sh` - Convenience script for MLX server with uv
- `run-cuda.sh` - Convenience script for CUDA server with uv
- `clients/python/mflux_client.py` - Python client example
- `clients/web-ui/` - Web frontend (served at `/index.html`)
- `clients/gradio-ui/` - Gradio interface
- `requirements.txt` / `requirements_cuda.txt` - Legacy dependency files

## HuggingFace Authentication

The server requires HF credentials for gated models:
```bash
pip install huggingface-hub
huggingface-cli login
```

## Development Notes

- The server monkey-patches `requests.Session.request` to disable SSL verification
- Images are converted to requested format (JPEG/PNG) immediately after generation to save memory
- Default inference steps vary by model (4 for schnell, 25 for dev)
- Priority tasks insert at position 1 in queue (after currently processing task)
