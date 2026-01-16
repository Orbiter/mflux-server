# mflux-server (MVP)

This is an MVP reimplementation of the mflux-server, focusing on using native model APIs (specifically ZImageTurbo) and providing an OpenAI-compatible API endpoint.

## Features

- **Native Model Integration**: Uses specific adapters for models (currently ZImageTurbo) to leverage all native parameters.
- **OpenAI-Compatible API**: Exposes a `/v1/images/generations` endpoint compatible with OpenAI clients.
- **Synchronous/Blocking API**: The API waits for generation to complete before returning (up to a timeout), simplifying client integration.
- **Background Worker**: Processes tasks sequentially to manage GPU resources.

## Quick Start

### Prerequisites
- macOS with Apple Silicon (MLX requirement)
- Python 3.12+
- `uv` (recommended for dependency management)

### Installation & Running

```bash
# Using the provided run script (handles dependencies and args)
./run.sh

# Or manually with uv
uv sync --extra mlx
uv run python server.py --model z-image-turbo
```

## API Usage

The server runs on `http://127.0.0.1:4030` by default.

### Generate Image (OpenAI Compatible)

**Endpoint:** `POST /v1/images/generations`

**Headers:**
- `Content-Type: application/json`

**Body:**
```json
{
  "prompt": "A cyberpunk city street at night",
  "size": "1024x1024",
  "steps": 4,
  "scheduler": "linear",
  "response_format": "b64_json"
}
```

**Response:**
```json
{
  "created": 1705391234,
  "data": [
    {
      "b64_json": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    }
  ]
}
```

### Health Check

**Endpoint:** `GET /health`

Returns `{"status": "healthy", "model": "z-image-turbo"}`.

## Configuration

Server arguments:
- `--model`: Model to load (default: `z-image-turbo`)
- `--host`: Host to bind to (default: `127.0.0.1`)
- `--port`: Port to listen on (default: `4030`)
- `--quantize`: Quantization level (e.g., 4 or 8)
- `--model_path`: Custom path for MLX models
- `--low-ram`: Enable low-RAM mode

## Legacy Server

The original generic server implementation has been moved to `server_generic.py`. See `README_legacy.md` for its documentation.

## Client Compatibility

**Important:** The existing clients in the `clients/` directory (`web-ui`, `gradio-ui`, `mflux_client.py`) are designed for the legacy generic server API and **will not work** with this new MVP server.

This MVP server uses an OpenAI-compatible API schema (`/v1/images/generations`). You can use standard OpenAI client libraries to interact with it.

### Example Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:4030/v1",
    api_key="not-needed"
)

response = client.images.generate(
    model="z-image-turbo",
    prompt="A futuristic city",
    size="1024x1024",
    quality="standard",
    n=1,
)

print(response.data[0].b64_json)
```
