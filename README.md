# API Server for mflux

This API server is designed for asynchronous image generation tasks with [mflux](https://github.com/filipstrand/mflux). It is particularly optimized for environments where GPU resources need to be shared across multiple tasks, such as in generative AI chat programs. This server ensures that only one image generation task runs at a time to efficiently use GPU resources. We also add a default user interface to provide a multi-image generation front-end.

## Examples

Here are two different client applications that use the server API. The first one is the default web-frontend which is available at `http://localhost:4030`

![Screenshot of mflux Image Generator Web Front-end](clients/web-ui/screenshot.png)

The second screenshot shows the Gradio Front-End:

![Screenshot of mflux Image Generator Gradio Front-end](clients/gradio-ui/screenshot.png)

Code for both client applications is located in the `clients` subdirectory.

## Features

The API supports features such as:

- Queuing of image generation tasks.
- Forecasting computation time for better user experience in multi-user environments.
- Managing task statuses and retrieving generated images.
- Reporting failed tasks without stopping the worker or retrying them automatically.

Furthermore, the API exposes a swagger endpoint to self-document the server.

## Example usage

Use Python 3.12. The server dependencies in `requirements.txt` pin
`mflux==0.19.2` and require `mlx>=0.32.2,<0.33.0` and
`protobuf>=4.25.0,<8.0`.

Install the dependencies in a virtual environment:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python3.12 -m pip install -r requirements.txt
```

For gated Hugging Face models, obtain access to the model repository and log in
with `hf auth login`, or set `HF_TOKEN`. The Hugging Face CLI is installed with
the server dependencies. Authentication is separate from starting the server.

```sh
python3.12 server.py --quantize 8 --host 0.0.0.0
```

The default model is `ernie-image-turbo`, with 8 inference steps and guidance
1.0. Select another model with `--model`; `/api/ls` lists supported model names
and defaults, and `/api/ps` reports the current settings. Without `--host`, the
server listens on `127.0.0.1`.

Alternatively, `./run.sh --quantize 8` creates or reuses `.venv`, upgrades the
dependencies within the requirements constraints, and starts the server on
`0.0.0.0`. It requires Python 3.12 and does not perform Hugging Face login.

### Krea 2 Turbo

Krea 2 Turbo uses mflux's dedicated `Krea2` loader and defaults to 8 steps,
guidance 1.0, and mflux's `er_sde` sampler. Start it with `--model krea2`,
`--model krea-2`, `--model krea-2-turbo`, or `--model krea/Krea-2-Turbo`:

```sh
./run.sh --model krea2 --quantize 8
```

These names also work with `POST /api/load`, for example
`{"model": "krea2", "quantize": 8}`, and appear in the web frontend's model
selector. Generate through `/api/generate` with a plain text prompt; omitted
`steps` and `guidance` use the Turbo defaults. The existing base64 `init_image`
input enables image-to-image generation with the server's fixed strength of 0.4.
See the [MFLUX Krea 2 guide](https://github.com/mflux-community/mflux/blob/main/src/mflux/models/krea2/README.md)
for upstream model details.

The first load downloads the Krea weights (about 33 GB, including the text encoder
and VAE), even when quantization is enabled. Existing cached files are reused.
Before loading Krea, the server completes its Hugging Face snapshot, including
`turbo.safetensors` and tokenizer files that may be missing from an earlier
download with another loader. There is no need to delete the cache.

### Ideogram 4

Ideogram 4 is supported by the pinned `mflux==0.19.2` dependency. Start it with
`--model ideogram4`, `--model ideogram-4-fp8`, or
`--model ideogram-ai/ideogram-4-fp8`:

```sh
python3.12 server.py --model ideogram4 --quantize 4 --host 0.0.0.0
```

Before the first load, request access to
[ideogram-ai/ideogram-4-fp8](https://huggingface.co/ideogram-ai/ideogram-4-fp8),
wait for approval, and authenticate with `hf auth login` or set `HF_TOKEN`.
The initial FP8 checkpoint download is about 28 GB, even when using quantization.
Omit `--quantize` to use the original FP8 layout; `--quantize 4` and
`--quantize 8` enable MLX quantization.

You can also switch a running server with
`POST /api/load` and `{"model": "ideogram4", "quantize": 4}`. The model is
listed by `/api/ls` and in the web frontend's model selector.

Generation uses `preset` to select the sampler:

| Preset | Steps |
| --- | --- |
| `V4_DEFAULT_20` (default) | 20 |
| `V4_QUALITY_48` | 48 |
| `V4_TURBO_12` | 12 |

For this model, `steps` and `guidance` are ignored so that the preset's guidance
and noise schedules are preserved. Width and height must be multiples of 16
between 256 and 2048. Image-to-image input (`init_image`) is unsupported and
returns HTTP 400.

The API's `prompt` remains a string. Plain text works, but structured JSON
captions are recommended; encode the caption as a JSON string:

```python
import json
import requests

caption = {
    "high_level_description": "A white ceramic teapot on a simple studio table.",
    "compositional_deconstruction": {
        "background": "A neutral tabletop with a pale wall behind it.",
        "elements": [
            {"type": "obj", "bbox": [250, 320, 780, 690],
             "desc": "A glossy white ceramic teapot with a curved handle."}
        ],
    },
}
response = requests.post("http://localhost:4030/api/generate", json={
    "prompt": json.dumps(caption, ensure_ascii=False),
    "seed": "42",
    "width": 1024,
    "height": 1024,
    "preset": "V4_DEFAULT_20",
    "strict_caption_validation": True,
    "format": "PNG",
})
response.raise_for_status()
task_id = response.json()["task_id"]
```

Use the usual `/api/status` and `/api/image` workflow to retrieve the result.
`strict_caption_validation` defaults to false; enabling it rejects caption
warnings with HTTP 400 before queuing. See the
[MFLUX Ideogram 4 guide](https://github.com/mflux-community/mflux/blob/main/src/mflux/models/ideogram4/README.md)
for the caption format and more examples. In the web frontend, paste the JSON
caption into the prompt box; generation uses the default 20-step preset.

### API workflow

The server runs on port 4030 by default. Host and port can be configured with
`--host` and `--port`; run `python3.12 server.py --help` for all options.
To see the swagger documentation, open `http://localhost:4030/swagger`

To produce an image, the usual workflow is:

- `/api/generate` to initialize the generation, this returns a `task_id`
- `/api/status` to poll for `waiting`, `done`, or `error`; waiting tasks include an estimated remaining time
- `/api/image` to retrieve the produced image as soon as the status turns to "done"

In detail - here is a call to generate an image:

```sh
curl -X 'POST' \
  'http://localhost:4030/api/generate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "A beautiful landscape",
  "seed": "1725311496",
  "height": 1024,
  "width": 1024,
  "format": "JPEG",
  "quality": 85,
  "priority": false
}'
```

Omit `steps` and `guidance` to use the selected model's defaults. An example
response with the default model and `--quantize 8` is:

```json
{
  "task_id": "1fc9cc4f",
  "task_length": 1,
  "expected_time_seconds": 146.88675427253082,
  "model": "ernie-image-turbo",
  "quantize": 8
}
```

The `task_id` can then be used to check the image generation status:

```sh
curl -X 'GET' \
  'http://localhost:4030/api/status?task_id=1fc9cc4f' \
  -H 'accept: application/json'
```

An example response is:

```json
{
  "pos": 0,
  "status": "waiting",
  "wait_remaining": 15
}
```
The image is expected to be ready in 15 seconds. Position 0 means that no other
pending task precedes it. Completed and failed tasks do not count toward the
position or estimated waiting time.

If generation fails, `/api/status` returns HTTP 200 with
`{"status": "error", "error": "..."}`. Stop polling that task and display the
error. Failed tasks are not retried and do not delay subsequent tasks; they can
be removed with `GET /api/cancel?task_id=...` or `GET /api/clear` (all tasks).
Generation and image encoding exceptions mark the task as failed; cleanup
exceptions are logged without changing a successfully generated result.
In either case, the worker continues with the next pending task. The web,
Gradio, and Python clients stop polling on `error`. An unknown task ID returns
HTTP 404.

Finally, the image can be retrieved with:

```sh
curl -X 'GET' \
  'http://localhost:4030/api/image?task_id=1fc9cc4f' \
  -H 'accept: application/json'
```

This returns the jpeg binary and removes the image from the production queue.

There are more API endpoints to list the queue and delete entries from the queue, see swagger documentation for details.

## Python client (quick example)

Here are three functions which implement a client endpoint for the image generation process as shown above with curl:

```python
import time
from io import BytesIO

import requests
from PIL import Image


def mflux_generate_client(mfluxendpoint, prompt, width=1280, height=720, steps=None, seed=None, format="JPEG", quality=85, priority=False):
    data = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "format": format,
        "quality": quality,
        "priority": priority
    }
    if steps is not None:
        data["steps"] = steps
    if seed is not None:
        data["seed"] = str(seed)
    response = requests.post(mfluxendpoint + "/api/generate", json=data)
    response.raise_for_status()
    # parse the response and get the task_id
    json = response.json()
    task_id = json["task_id"]
    return task_id
    
def mflux_status_ready(mfluxendpoint, task_id):
    response = requests.get(mfluxendpoint + "/api/status?task_id=" + task_id)
    response.raise_for_status()
    result = response.json()
    if result["status"] == "done":
        return 0
    if result["status"] == "error":
        raise RuntimeError(result.get("error", "Image generation failed."))
    return max(result.get("wait_remaining", 1), 1)

def mflux_get_image(mfluxendpoint, task_id):
    response = requests.get(mfluxendpoint + "/api/image?task_id=" + task_id + "&base64=false&delete=true")
    response.raise_for_status()
    return response.content
```

The mfluxendpoint would be a string like `http://localhost:4030`. 
A single function which uses the client endpoints above to get an image can be i.e.:

```python
def generate_image(mfluxendpoint, prompt, width=1280, height=720, steps=None):
    startt = time.time()
    task_id = mflux_generate_client(mfluxendpoint, prompt, width=width, height=height, steps=steps)
    for i in range(10000):
        waiting_time = mflux_status_ready(mfluxendpoint, task_id)
        print("Waiting time: ", waiting_time, " seconds")
        if waiting_time == 0: break
        nextsleep = max(min(waiting_time / 4, 10), 1)
        time.sleep(nextsleep)
    else:
        raise TimeoutError("Image generation did not finish within the polling limit.")
    imageb = mflux_get_image(mfluxendpoint, task_id)    
    stopt = time.time()
    print("Time taken: ", stopt - startt, " seconds")
    image = Image.open(BytesIO(imageb))
    return image
```

## Development checks

Run the tests with the project virtual environment:

```sh
.venv/bin/python3.12 -m pip check
.venv/bin/python3.12 -m unittest discover -s tests -v
```

The tests mock model loading and inference and do not download model weights.
Importing `server.py` still initializes MLX, so the test environment needs an
accessible GPU backend. On macOS, a sandbox without Metal access cannot run
the full suite directly.

## License

The code is licensed under the Apache 2.0 license.

## Contribution and Contact

Pull requests to enhance the code are welcome!

If you want to share your experience with mflux-server on social media, please notify me under one of the following addresses:

- Mastodon: `@orbiterlab@sigmoid.social`
- X: `@orbiterlab`
