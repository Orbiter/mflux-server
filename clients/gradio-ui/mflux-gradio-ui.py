# mflux-gradio-ui
# Gradio UI Server for the mflux-server
# (C) 2024 by @orbiter Michael Peter Christen
# This code is licensed under the Apache License, Version 2.0

import time
import base64
from pathlib import Path
import requests
import argparse
import gradio as gr
from PIL import Image
from io import BytesIO

mfluxendpoint = "http://localhost:4030"


def model_info():
    response = requests.get(mfluxendpoint + '/api/info', timeout=10)
    response.raise_for_status()
    return response.json()


def image_limit(info):
    limit = info.get('max_init_images', 0)
    if info.get('edit') is not True or not isinstance(limit, int):
        return 0
    return max(0, limit if info.get('multi_image_edit') is True else min(limit, 1))


def attachment_controls(files, uploads=None):
    """Keep attachments visible and removable even when model support changes."""
    files = list(files or [])
    try:
        info = model_info()
        limit = image_limit(info)
        additions = [str(path) for path in (uploads or []) if str(path) not in files]
        if len(files) + len(additions) > limit and additions:
            gr.Warning(f"This model accepts at most {limit} image(s). Remove attachments or select fewer files.")
        else:
            files.extend(additions)
        invalid = len(files) > limit
        hint = (f"Remove attachments: this model allows {limit}." if invalid else
                f"{len(files)}/{limit} images · {info['model']}" if limit else
                "Images unavailable for this model")
        can_add = len(files) < limit
        can_generate = not invalid
    except (requests.RequestException, ValueError, KeyError):
        hint, can_add, can_generate = 'Image support unavailable', False, not files
    choices = [(Path(path).name, path) for path in files]
    return (gr.Dropdown(choices=choices, value=files, multiselect=True),
            gr.UploadButton(interactive=can_add), hint, gr.Button(interactive=can_generate))


def refresh_attachment_controls(files):
    # Background refresh must not overwrite files added/removed while /info is in flight.
    return attachment_controls(files)[1:]


def generate_image_gradio(prompt, width, height, steps, seed, format, quality, priority, init_images=None):
    # Send POST request to /generate
    data = {
        "prompt": prompt,
        "height": int(height),
        "width": int(width),
        "format": format,
        "quality": int(quality),
        "priority": priority
    }
    if seed is not None and seed != "":
        data["seed"] = str(seed)
    try:
        if steps is not None and str(steps).strip():
            data['steps'] = int(steps)
            if data['steps'] < 1:
                raise ValueError('Steps must be positive, or blank for the model default.')
        if init_images:
            info = model_info()
            limit = image_limit(info)
            if len(init_images) > limit:
                yield None, f"{info['model']} accepts at most {limit} image(s). Remove attachments first."
                return
            data['init_images'] = [base64.b64encode(Path(path).read_bytes()).decode('ascii')
                                   for path in init_images]
        response = requests.post(mfluxendpoint + "/api/generate", json=data, timeout=30)
    except (requests.RequestException, OSError, ValueError) as exc:
        yield None, f"Could not submit the request: {exc}"
        return
    if response.status_code != 200:
        yield None, f"Failed to start image generation: {response.text}"
        return
    json_resp = response.json()
    task_id = json_resp["task_id"]
    expected_time = json_resp.get("expected_time_seconds", None)
    status_text = "Task started." if expected_time is None else f"Task started. Expected time: {expected_time:.1f} seconds."
    yield None, status_text
    # Start polling for status
    for _ in range(10000):
        status_resp = requests.get(f"{mfluxendpoint}/api/status?task_id={task_id}")
        if status_resp.status_code != 200:
            yield None, "Failed to get status."
            return
        status_json = status_resp.json()
        status = status_json["status"]
        if status == "done":
            break
        elif status == "error":
            yield None, status_json.get("error", "Image generation failed.")
            return
        else:
            wait_remaining = status_json.get("wait_remaining", 1)
            pos = status_json.get("pos", 0)
            # Update progress/status
            status_text = f"Status: {status}, Position in queue: {pos}, Estimated wait time: {wait_remaining:.1f} seconds."
            yield None, status_text
            time.sleep(1)
    # Get the image
    image_resp = requests.get(f"{mfluxendpoint}/api/image?task_id={task_id}&base64=false&delete=true")
    if image_resp.status_code != 200:
        yield None, "Failed to retrieve image."
        return
    image_bytes = image_resp.content
    image = Image.open(BytesIO(image_bytes))
    yield image, "Image generation completed."

def build_interface():
    with gr.Blocks(title="MFLUX Image Generator") as interface:
        gr.Markdown(f"Generate images with mflux · [Swagger]({mfluxendpoint}/swagger)")
        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(label="Prompt", lines=3)
                with gr.Row():
                    add_images = gr.UploadButton("+ Images", file_count="multiple", file_types=["image"],
                                                 type="filepath", size="sm", scale=0, min_width=80,
                                                 interactive=False, elem_id="add-images")
                    attachments = gr.Dropdown(label="Attached images", choices=[], value=[], multiselect=True,
                                               show_label=False, interactive=True, min_width=160,
                                               container=False, filterable=False,
                                               elem_id="attachments")
                    refresh = gr.Button("Refresh", size="sm", scale=0, min_width=70)
                image_hint = gr.Markdown("Checking image support…")
                with gr.Row():
                    width_input = gr.Slider(minimum=64, maximum=2048, step=64, value=1024, label="Width")
                    height_input = gr.Slider(minimum=64, maximum=2048, step=64, value=1024, label="Height")
                with gr.Row():
                    steps_input = gr.Textbox(label="Steps (optional)", placeholder="Model default", value="")
                    seed_input = gr.Textbox(label="Seed (optional)", placeholder="Random if blank")
                with gr.Row():
                    format_input = gr.Radio(choices=["JPEG", "PNG"], value="JPEG", label="File format")
                    quality_input = gr.Slider(minimum=1, maximum=100, step=1, value=85, label="JPEG quality")
                priority_input = gr.Checkbox(label="Priority", value=False)
                generate = gr.Button("Generate", variant="primary")
            with gr.Column():
                image_output = gr.Image(label="Generated image")
                status_output = gr.Textbox(label="Status", lines=2)
        controls = [attachments, add_images, image_hint, generate]
        interface.load(refresh_attachment_controls, inputs=[attachments], outputs=controls[1:], queue=False)
        refresh.click(refresh_attachment_controls, inputs=[attachments], outputs=controls[1:], queue=False)
        add_images.upload(attachment_controls, inputs=[attachments, add_images], outputs=controls, queue=False)
        attachments.change(refresh_attachment_controls, inputs=[attachments], outputs=controls[1:], queue=False)
        # Another client may switch the server's model while this page is open.
        gr.Timer(10).tick(refresh_attachment_controls, inputs=[attachments], outputs=controls[1:],
                          queue=False, show_progress="hidden")
        generate.click(generate_image_gradio,
                       inputs=[prompt_input, width_input, height_input, steps_input, seed_input,
                               format_input, quality_input, priority_input, attachments],
                       outputs=[image_output, status_output])
    return interface


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Launch mflux Gradio front-end")
    parser.add_argument('--server', default=mfluxendpoint, help='mflux-server URL')
    args = parser.parse_args()
    mfluxendpoint = args.server.rstrip('/')
    build_interface().launch()
