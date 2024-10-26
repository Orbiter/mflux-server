# Gradio Front-end for the mflux-server Image Generator

This project is a Python-based front-end for the `mflux-server` image generation server, built with Gradio. It provides a simple, interactive interface for generating images based on user-defined prompts and parameters.

![Screenshot of mflux Image Generator Gradio Front-end](screenshot.png)

## Features

- **Image Generation**: Generate an image with the mflux-server using configurable parameters (image dimensions, quality, seed, etc.).
- **Status Updates**: Real-time status updates on the progress of the image generation process.
- **Customizable Server Endpoint**: Specify a custom server endpoint when launching the application if `mflux-server` is hosted on a different server or port.

## Installation

1. Clone this repository and navigate to the project directory.
2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

To run the Gradio front-end, ensure mflux-server is running and then execute the following command:

```bash
python mflux-gradio-ui.py
```

This command starts a web server for the Gradio interface, accessible locally at http://127.0.0.1:7860.

By default, the application connects to mflux-server at http://localhost:4030. If your mflux-server is hosted on a different server or port, use the --server parameter to specify the endpoint:

```bash
python mflux-gradio-ui.py --server http://custom-server:port
```

## Usage
- Open http://127.0.0.1:7860 in your browser.
- Enter a prompt and configure the parameters in the Gradio interface.
- Click the "Submit" button to generate the image.
- The generated image will be displayed, along with a status message showing progress.

## Requirements
mflux-server running and accessible at the specified endpoint.

## License
This project is licensed under the Apache License 2.0.
