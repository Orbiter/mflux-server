#!/usr/bin/env python3
import json
import time
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


def generate_one_image(base_url, prompt, out_path="output.jpg"):
    payload = {
        "prompt": prompt,
        "height": 1024,
        "width": 1024,
        "steps": 4,
        "format": "JPEG",
        "quality": 85,
        "priority": False,
    }
    resp = post_json(base_url + "/api/generate", payload)
    task_id = resp["task_id"]

    while True:
        status = get_json(base_url + "/api/status?task_id=" + urllib.parse.quote(task_id))
        if status.get("status") == "done":
            break
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
    server = "http://localhost:4030"
    prompt = "A beautiful landscape"
    path = generate_one_image(server, prompt)
    print("Saved:", path)
