import time
import requests
import argparse
import gradio as gr
from PIL import Image
from io import BytesIO

# Set up argument parsing
parser = argparse.ArgumentParser(description="Launch mflux Gradio front-end")
parser.add_argument("--server", type=str, default="http://localhost:4030", 
                    help="Specify the mflux-server endpoint (default: http://localhost:4030)")
args = parser.parse_args()

# Set mflux endpoint based on command-line argument
mfluxendpoint = args.server

def generate_image_gradio(prompt, width, height, steps, seed, format, quality, priority):
    # Send POST request to /generate
    data = {
        "prompt": prompt,
        "height": int(height),
        "width": int(width),
        "steps": int(steps),
        "guidance": 3.5,
        "format": format,
        "quality": int(quality),
        "priority": priority
    }
    if seed is not None and seed != "":
        data["seed"] = str(seed)
    response = requests.post(mfluxendpoint + "/generate", json=data)
    if response.status_code != 200:
        yield None, "Failed to start image generation."
        return
    json_resp = response.json()
    task_id = json_resp["task_id"]
    expected_time = json_resp.get("expected_time_seconds", None)
    status_text = f"Task started. Expected time: {expected_time:.1f} seconds."
    yield None, status_text
    # Start polling for status
    for _ in range(10000):
        status_resp = requests.get(f"{mfluxendpoint}/status?task_id={task_id}")
        if status_resp.status_code != 200:
            yield None, "Failed to get status."
            return
        status_json = status_resp.json()
        status = status_json["status"]
        if status == "done":
            break
        else:
            wait_remaining = status_json.get("wait_remaining", 1)
            pos = status_json.get("pos", 0)
            # Update progress/status
            status_text = f"Status: {status}, Position in queue: {pos}, Estimated wait time: {wait_remaining:.1f} seconds."
            yield None, status_text
            time.sleep(1)
    # Get the image
    image_resp = requests.get(f"{mfluxendpoint}/image?task_id={task_id}&base64=false&delete=true")
    if image_resp.status_code != 200:
        yield None, "Failed to retrieve image."
        return
    image_bytes = image_resp.content
    image = Image.open(BytesIO(image_bytes))
    yield image, "Image generation completed."

# Define the inputs
prompt_input = gr.Textbox(label="Prompt", lines=3)
width_input = gr.Slider(minimum=64, maximum=2048, step=64, value=512, label="Width")
height_input = gr.Slider(minimum=64, maximum=2048, step=64, value=512, label="Height")
steps_input = gr.Slider(minimum=1, maximum=8, step=1, value=4, label="Number of Inference Steps")
seed_input = gr.Textbox(label="Seed (optional)", placeholder="Leave blank for random seed", lines=1)
#guidance_input = gr.Slider(minimum=0.0, maximum=20.0, step=0.1, value=3.5, label="Guidance Scale (only used for 'dev' model)")
format_input = gr.Radio(choices=["JPEG", "PNG"], value="JPEG", label="File Format")
quality_input = gr.Slider(minimum=1, maximum=100, step=1, value=85, label="JPEG Quality")
priority_input = gr.Checkbox(label="Priority (enqueue at begginning of queue, not end)", value=False)
inputs = [prompt_input, width_input, height_input, steps_input, seed_input, format_input, quality_input, priority_input]

# Define the outputs
image_output = gr.Image(label="Generated Image")
status_output = gr.Textbox(label="Status", lines=2)
outputs = [image_output, status_output]

swagger_url = f"{mfluxendpoint}/swagger"
 
# Create the interface
iface = gr.Interface(
    fn=generate_image_gradio,
    inputs=inputs,
    outputs=outputs,
    description=f"Generate images using the mflux API server. [Swagger]({swagger_url})",
    allow_flagging='never'
)

# Launch the app
iface.launch()

