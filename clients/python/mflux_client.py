#!/usr/bin/env python3
import json
import argparse
import base64
import time
from pathlib import Path
import urllib.request
import urllib.parse


def post_json(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_bytes(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def generate_one_image(base_url, prompt, out_path="output.jpg", *, init_images=None, steps=None, seed=None):
    base_url = base_url.rstrip('/')
    payload = {
        "prompt": prompt,
        "height": 1024,
        "width": 1024,
        "format": "PNG" if Path(out_path).suffix.lower() == '.png' else "JPEG",
        "quality": 85,
        "priority": False,
    }
    if steps is not None:
        payload['steps'] = steps
    if seed is not None:
        payload['seed'] = str(seed)
    if init_images:
        capabilities = get_json(base_url + '/api/info')
        limit = capabilities.get('max_init_images', 0) if capabilities.get('edit') is True else 0
        if capabilities.get('multi_image_edit') is not True:
            limit = min(limit, 1)
        if len(init_images) > limit:
            raise ValueError(f"{capabilities['model']} accepts at most {limit} input image(s).")
        payload['init_images'] = [
            base64.b64encode(Path(filename).read_bytes()).decode('ascii')
            for filename in init_images
        ]
    resp = post_json(base_url + "/api/generate", payload)
    task_id = resp["task_id"]

    while True:
        status = get_json(base_url + "/api/status?task_id=" + urllib.parse.quote(task_id))
        if status.get("status") == "done":
            break
        if status.get("status") == "error":
            raise RuntimeError(status.get("error", "Image generation failed."))
        wait_seconds = status.get("wait_remaining", 1)
        if wait_seconds < 1:
            wait_seconds = 1
        time.sleep(min(wait_seconds, 5))

    image_bytes = get_bytes(
        base_url
        + "/api/image?task_id="
        + urllib.parse.quote(task_id)
        + "&base64=false&delete=true"
    )
    with open(out_path, "wb") as f:
        f.write(image_bytes)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate one image through the mflux server.')
    parser.add_argument('--server', default='http://localhost:4030')
    parser.add_argument('--prompt', default='A beautiful landscape')
    parser.add_argument('--output', default='output.jpg')
    parser.add_argument('--init-images', nargs='+', metavar='FILE', help='Reference files in prompt order')
    parser.add_argument('--steps', type=int, help='Omit to use the model default')
    parser.add_argument('--seed')
    args = parser.parse_args()
    path = generate_one_image(args.server, args.prompt, args.output,
                              init_images=args.init_images, steps=args.steps, seed=args.seed)
    print("Saved:", path)
