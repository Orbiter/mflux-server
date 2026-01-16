# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mflux-server is an image generation API server built on top of [mflux](https://github.com/filipstrand/mflux).

**Current State (MVP):**
The project is undergoing a refactor to support native model APIs.
- `server.py`: New MVP implementation using `ZImageTurboAdapter` and an OpenAI-compatible API (`/v1/images/generations`).
- `server_generic.py`: The original generic MLX server implementation (renamed from old `server.py`).

## Running the Server

### Quick Start (Recommended: using uv)
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# macOS (Apple Silicon / MLX backend) - Runs new ZImageTurbo MVP
./run.sh
```

### Manual Setup with uv

#### New MVP Server (ZImageTurbo)
```bash
uv sync --extra mlx
uv run python server.py --model z-image-turbo
```

#### Legacy Generic Server
```bash
uv sync --extra mlx
uv run python server_generic.py --quantize 8 --host 0.0.0.0
```

## Architecture (MVP)

### Key Files
- `server.py`: Main entry point for the new MVP. Uses Flask and `server_adapters.py`.
- `server_adapters.py`: Contains `ModelAdapter` abstract base class and `ZImageTurboAdapter` implementation.
- `server_generic.py`: Legacy generic server supporting multiple models via `mflux` generic loaders.

### API (New MVP)
- `POST /v1/images/generations`: OpenAI-compatible endpoint. Blocking call (waits for generation).
- `GET /health`: Health check.

### API (Legacy/Generic)
- `POST /api/generate`: Async task submission.
- `GET /api/status`: Polling.
- `GET /api/image`: Retrieval.

## Development Notes

- **Testing**: Use `uv run python tests/test_server_mock.py` to verify the new server logic without loading heavy models.
- **Model Support**: Currently only `z-image-turbo` is fully implemented in the new adapter architecture.
