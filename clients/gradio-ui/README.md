# Gradio Front-end for the mflux-server Image Generator

This project is a Python-based front-end for the `mflux-server` image generation server, built with Gradio. It provides a simple, interactive interface for generating images based on user-defined prompts and parameters.

![Screenshot of mflux Image Generator Gradio Front-end](screenshot.png)

## Features

- **Image Generation**: Generate an image with the mflux-server using configurable parameters (image dimensions, quality, seed, etc.).
- **Status Updates**: Real-time status updates on the progress of the image generation process.
- **Customizable Server Endpoint**: Specify a custom server endpoint when launching the application if `mflux-server` is hosted on a different server or port.
- **Compact Image Attachments**: Add files with **+ Images**. Filenames appear as small chips beside the button; use each chip's × to remove it, or clear the selection. Images are sent together in the displayed order in the `init_images` array for one output.
- **Model Capabilities**: `/api/info` enables uploads only when the current model supports them, up to its limit. Capabilities refresh on page load, every ten seconds, with Refresh, and before submission. Attachments remain removable if the model changes; excess inputs block generation. If capability lookup fails, adding images is disabled.

## Installation

1. Clone this repository and navigate to the project directory.
2. Create a virtual environment and activate it:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install the required dependencies:
   ```
   python3.12 -m pip install -r clients/gradio-ui/requirements.txt
   ```

## Running the Application

To run the Gradio front-end, ensure mflux-server is running and then execute the following command:

```bash
python3.12 clients/gradio-ui/mflux-gradio-ui.py
```

This command starts a web server for the Gradio interface, accessible locally at http://127.0.0.1:7860.

By default, the application connects to mflux-server at http://localhost:4030. If your mflux-server is hosted on a different server or port, use the --server parameter to specify the endpoint:

```bash
python3.12 clients/gradio-ui/mflux-gradio-ui.py --server http://custom-server:port
```

## Usage

Qwen Image 2.1 accepts up to ten ordered reference attachments; FLUX.2 Klein
accepts four. The limit and counter come from `/api/info`. Restart an older
server after updating, then refresh the model capabilities in the client.
- Open http://127.0.0.1:7860 in your browser.
- Enter a prompt and configure the parameters in the Gradio interface.
- Optionally add input images. Their displayed order corresponds to image 1, image 2, etc. in the prompt.
- Leave Steps blank to use the selected model's inference default.
- Click the "Generate" button to generate the image.
- The generated image will be displayed, along with a status message showing progress.

## Requirements
mflux-server running and accessible at the specified endpoint.

## License
This project is licensed under the Apache License 2.0.
