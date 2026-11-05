# mflux-gradio-ui
# Gradio UI Server for the mflux-server
# (C) 2024 by @orbiter Michael Peter Christen
# This code is licensed under the Apache License, Version 2.0

import time
import json
import tempfile
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


def generate_image_gradio(prompt, width, height, steps, seed, format, quality, priority, init_images=None,
                           generation_options=None, mask_image=None, control_image=None, depth_image=None):
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
        if generation_options:
            data.update(generation_options)
        for field, path in (('mask_image', mask_image), ('control_image', control_image), ('depth_image', depth_image)):
            if path:
                data[field] = base64.b64encode(Path(path).read_bytes()).decode('ascii')
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
    # Retrieve the recipe before downloading with delete=true, including for JPEG.
    recipe_resp = requests.get(f"{mfluxendpoint}/api/recipe?task_id={task_id}", timeout=10)
    recipe = recipe_resp.json() if recipe_resp.status_code == 200 else None
    image_resp = requests.get(f"{mfluxendpoint}/api/image?task_id={task_id}&base64=false&delete=true")
    if image_resp.status_code != 200:
        yield None, "Failed to retrieve image."
        return
    image_bytes = image_resp.content
    image = Image.open(BytesIO(image_bytes))
    if recipe is not None:
        image.info['mflux_recipe'] = json.dumps(recipe, ensure_ascii=False, indent=2)
    yield image, "Image generation completed."

def load_model_gradio(name, source, quantize, lora_paths, scales, precision, small_encoder, small_decoder, bake,
                      full_precision=False, encoder_bits='default', backend='native', control_path='', control_strength=None):
    try:
        response = requests.get(mfluxendpoint + '/api/ls', timeout=10)
        response.raise_for_status()
        info = response.json()[name]
        supported = set(info.get('load_options', []))
        options = {}
        for field, value in (('compute_precision', precision), ('small_text_encoder', small_encoder),
                             ('small_decoder', small_decoder), ('vae_full_precision', full_precision),
                             ('qwen21_backend', backend), ('controlnet_path', control_path),
                             ('controlnet_strength', control_strength)):
            if field in supported and value not in (None, '', 'default'):
                options[field] = value
        if 'lora_paths' in supported and lora_paths.strip():
            options['lora_paths'] = [value.strip() for value in lora_paths.splitlines() if value.strip()]
            if scales.strip():
                options['lora_scales'] = [float(value) for value in scales.split(',')]
        if 'bake_lora' in supported and bake != 'default':
            options['bake_lora'] = bake == 'true'
        if 'text_encoder_quantize' in supported and encoder_bits != 'default':
            options['text_encoder_quantize'] = int(encoder_bits)
        payload = {'model': source.strip() or name, 'options': options,
                   'quantize': None if quantize == 'stored' else int(quantize)}
        if source.strip():
            payload['base_model'] = name
        response = requests.post(mfluxendpoint + '/api/load', json=payload, timeout=3600)
        response.raise_for_status()
        return 'Loaded ' + response.json()['model']
    except (requests.RequestException, ValueError, KeyError) as exc:
        return 'Model load failed: ' + str(exc)


def catalog_controls():
    try:
        response = requests.get(mfluxendpoint + '/api/ls', timeout=10)
        response.raise_for_status()
        info = model_info()
        return gr.Dropdown(choices=list(response.json()), value=info['model'])
    except (requests.RequestException, ValueError, KeyError):
        return gr.Dropdown(choices=[], value=None)


def advanced_generate(prompt, width, height, steps, seed, format, quality, priority, images,
                      negative, guidance, scheduler, ratio, cfg_end, strength, auto_mask, enhance,
                      verify, retries, use_kv_cache, reuse_steps, mask, control, depth,
                      cache_threshold=None, control_strength=None, resolution=None, render_text=True,
                      skip_layers='', preset='', strict_caption=False):
    try:
        info = model_info()
        supported = set(info.get('generation_options', []))
        options = {}
        values = {'negative_prompt': negative, 'scheduler': scheduler, 'step_cache_ratio': ratio,
                  'cfg_end': cfg_end, 'strength': strength, 'auto_mask': auto_mask,
                  'enhance_prompt': enhance, 'verify': verify, 'verify_retries': retries,
                  'use_kv_cache': use_kv_cache, 'use_step_cache': reuse_steps,
                  'step_cache_threshold': cache_threshold, 'controlnet_strength': control_strength,
                  'output_resolution': int(resolution) if resolution is not None else None,
                  'render_text_images': render_text, 'preset': preset, 'strict_caption_validation': strict_caption}
        if 'uncond_skip_layers' in supported and skip_layers.strip():
            values['uncond_skip_layers'] = [] if skip_layers.strip() == '[]' else [int(value) for value in skip_layers.split(',')]
        for field, value in values.items():
            if field in supported and value is not None and (value != '' or field == 'negative_prompt'):
                options[field] = value
        if guidance is not None:
            options['guidance'] = guidance
        if steps is not None and str(steps).strip():
            options_steps = int(steps)
            if info.get('fixed_steps') is not None and options_steps != info['fixed_steps']:
                raise ValueError(f"This model uses exactly {info['fixed_steps']} steps.")
        for field, path in (('mask', mask), ('control_image', control), ('depth_image', depth)):
            if path and not info.get(field):
                raise ValueError(f'This model does not accept {field}.')
        for image, status in generate_image_gradio(prompt, width, height, steps, seed, format, quality,
                priority, images, options, mask, control, depth):
            recipe_path = None
            if image is not None and image.info.get('mflux_recipe'):
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', prefix='mflux-recipe-', delete=False) as file:
                    file.write(image.info['mflux_recipe'])
                    recipe_path = file.name
            yield image, status, recipe_path
    except (ValueError, requests.RequestException) as exc:
        yield None, str(exc), None


def build_interface():
    with gr.Blocks(title="MFLUX Image Generator") as interface:
        gr.Markdown(f"Generate images with mflux · [Swagger]({mfluxendpoint}/swagger)")
        with gr.Row():
            with gr.Column():
                with gr.Accordion('Load a model and adapters', open=False):
                    model_select = gr.Dropdown(label='Model / family', choices=[])
                    source = gr.Textbox(label='Custom saved directory or HF repository (optional)')
                    quant = gr.Dropdown(label='Quantization', choices=['stored', '4', '5', '6', '8'], value='stored')
                    loras = gr.Textbox(label='LoRA paths or repositories (one per line)', lines=2)
                    scales = gr.Textbox(label='LoRA scales (comma separated, optional)')
                    precision = gr.Dropdown(label='Compute precision', choices=['default', 'bfloat16', 'float16'], value='default')
                    small_encoder = gr.Checkbox(label='Small text encoder (supported models only)')
                    small_decoder = gr.Checkbox(label='Small FLUX.2 decoder')
                    bake = gr.Dropdown(label='Merge adapters into weights', choices=['default', 'true', 'false'], value='default')
                    full_precision = gr.Checkbox(label='Full-image VAE decode (more memory)')
                    encoder_bits = gr.Dropdown(label='Text encoder quantization', choices=['default', '4', '5', '6', '8'], value='default')
                    backend = gr.Dropdown(label='Qwen backend', choices=['native', 'legacy'], value='native')
                    control_path = gr.Textbox(label='ControlNet path or repository (optional)')
                    load_control_strength = gr.Number(label='Krea ControlNet strength at load (optional)', value=None)
                    load = gr.Button('Load model')
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
                with gr.Accordion('Editing and optional acceleration', open=False):
                    negative = gr.Textbox(label='Negative prompt (optional)')
                    guidance = gr.Number(label='Guidance (blank uses model default)', value=None)
                    scheduler = gr.Dropdown(label='Sampler (optional)', choices=[], value=None)
                    ratio = gr.Number(label='Step reuse ratio (optional)', value=None)
                    cfg_end = gr.Number(label='Guidance end fraction (optional)', value=None)
                    strength = gr.Number(label='Edit strength (optional)', value=None)
                    auto_mask = gr.Textbox(label='Object to mask automatically (optional)')
                    enhance = gr.Checkbox(label='Enhance edit instruction')
                    verify = gr.Checkbox(label='Verify edit result')
                    retries = gr.Number(label='Verification retries', value=0, precision=0)
                    kv = gr.Checkbox(label='Reuse prompt attention', value=True)
                    reuse = gr.Checkbox(label='Reuse stable Qwen edit steps')
                    cache_threshold = gr.Number(label='Edit reuse threshold (optional)', value=None)
                    control_strength = gr.Number(label='Control strength (optional)', value=None)
                    resolution = gr.Number(label='Reference resolution budget (optional)', value=None, precision=0)
                    render_text = gr.Checkbox(label='Render caption text for conditioning', value=True)
                    skip_layers = gr.Textbox(label='Negative-branch skipped layers (comma separated; [] disables)')
                    preset = gr.Dropdown(label='Sampler preset', choices=[], value=None)
                    strict_caption = gr.Checkbox(label='Strict caption validation')
                    mask = gr.File(label='Mask (white regenerates)', type='filepath', file_types=['image'])
                    control = gr.File(label='Control image', type='filepath', file_types=['image'])
                    depth = gr.File(label='Prepared depth map', type='filepath', file_types=['image'])
                priority_input = gr.Checkbox(label="Priority", value=False)
                generate = gr.Button("Generate", variant="primary")
            with gr.Column():
                image_output = gr.Image(label="Generated image")
                recipe_output = gr.File(label='Generation recipe')
                status_output = gr.Textbox(label="Status", lines=2)
        interface.load(catalog_controls, outputs=[model_select], queue=False)
        load_fields = {'lora_paths': loras, 'lora_scales': scales, 'compute_precision': precision,
                       'small_text_encoder': small_encoder, 'small_decoder': small_decoder, 'bake_lora': bake,
                       'vae_full_precision': full_precision, 'text_encoder_quantize': encoder_bits,
                       'qwen21_backend': backend, 'controlnet_path': control_path,
                       'controlnet_strength': load_control_strength}
        generation_fields = {'negative_prompt': negative, 'scheduler': scheduler, 'step_cache_ratio': ratio,
            'cfg_end': cfg_end, 'strength': strength, 'auto_mask': auto_mask, 'enhance_prompt': enhance,
            'verify': verify, 'verify_retries': retries, 'use_kv_cache': kv, 'use_step_cache': reuse,
            'step_cache_threshold': cache_threshold, 'controlnet_strength': control_strength,
            'output_resolution': resolution, 'render_text_images': render_text, 'uncond_skip_layers': skip_layers,
            'preset': preset, 'strict_caption_validation': strict_caption}

        def family_controls(name):
            try:
                response = requests.get(mfluxendpoint + '/api/ls', timeout=10)
                response.raise_for_status()
                info = response.json()[name]
                return [gr.update(visible=key in info['load_options']) for key in load_fields] + [
                    gr.update(choices=['stored', *map(str, info['quantization'])], value='stored')]
            except (requests.RequestException, ValueError, KeyError):
                return [gr.skip() for _ in range(len(load_fields) + 1)]

        def current_controls():
            try:
                info = model_info()
                updates = []
                for key in generation_fields:
                    values = {'visible': key in info['generation_options']}
                    if key in ('scheduler', 'preset'):
                        choices = info['schedulers'] if key == 'scheduler' else list(info.get('presets', {}))
                        values.update(choices=choices, value=info.get('default_preset') if key == 'preset' else None)
                    updates.append(gr.update(**values))
                return updates + [gr.update(visible=info[key], **({} if info[key] else {'value': None}))
                    for key in ('mask', 'control_image', 'depth_image')] + [
                    gr.update(value=info['default_format']),
                    gr.update(placeholder=f"Default: {info['default_steps']}", value=str(info.get('fixed_steps') or ''),
                              interactive=info.get('fixed_steps') is None),
                    gr.update(value=info['default_guidance'], interactive=info.get('guidance_fixed') is None)]
            except (requests.RequestException, ValueError, KeyError):
                return [gr.skip() for _ in range(len(generation_fields) + 6)]

        model_select.change(family_controls, inputs=[model_select], outputs=[*load_fields.values(), quant], queue=False)
        current_outputs = [*generation_fields.values(), mask, control, depth, format_input, steps_input, guidance]
        interface.load(current_controls, outputs=current_outputs, queue=False)
        refresh.click(current_controls, outputs=current_outputs, queue=False)
        load.click(load_model_gradio, inputs=[model_select, source, quant, loras, scales, precision,
            small_encoder, small_decoder, bake, full_precision, encoder_bits, backend, control_path,
            load_control_strength], outputs=[status_output]).then(current_controls, outputs=current_outputs)
        controls = [attachments, add_images, image_hint, generate]
        interface.load(refresh_attachment_controls, inputs=[attachments], outputs=controls[1:], queue=False)
        refresh.click(refresh_attachment_controls, inputs=[attachments], outputs=controls[1:], queue=False)
        add_images.upload(attachment_controls, inputs=[attachments, add_images], outputs=controls, queue=False)
        attachments.change(refresh_attachment_controls, inputs=[attachments], outputs=controls[1:], queue=False)
        # Another client may switch the server's model while this page is open.
        gr.Timer(10).tick(refresh_attachment_controls, inputs=[attachments], outputs=controls[1:],
                          queue=False, show_progress="hidden")
        generate.click(advanced_generate,
                       inputs=[prompt_input, width_input, height_input, steps_input, seed_input,
                               format_input, quality_input, priority_input, attachments,
                               negative, guidance, scheduler, ratio, cfg_end, strength, auto_mask,
                               enhance, verify, retries, kv, reuse, mask, control, depth, cache_threshold,
                               control_strength, resolution, render_text, skip_layers, preset, strict_caption],
                       outputs=[image_output, status_output, recipe_output])
    return interface


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Launch mflux Gradio front-end")
    parser.add_argument('--server', default=mfluxendpoint, help='mflux-server URL')
    args = parser.parse_args()
    mfluxendpoint = args.server.rstrip('/')
    build_interface().launch()
