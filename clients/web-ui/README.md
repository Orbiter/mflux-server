# mflux Image Generator Front-end

This project is a web-based front-end for the `mflux-server` image generation server. It provides a simple and intuitive interface to generate multiple images based on user-defined prompts and parameters.

![Screenshot of mflux Image Generator Front-end](screenshot.png)

## Features

- **Multi-Image Generation**: Generate multiple images in a single request, allowing for efficient batch image creation.
- **Convenient Image Download**: Each generated image includes a dedicated download button, making it easy to save images to your local system.
- **Collision-Free Naming**: Images are saved with unique, timestamped filenames to prevent filename collisions, enabling smooth handling of large image sets.

## Installation

1. Ensure you have a running instance of `mflux-server`. The front-end requires this server to handle image generation requests.
2. No additional installation is needed for the front-end. Simply double-click `index.html` to open it in your browser.

## Usage

1. Enter a prompt and configure the parameters (image dimensions, quality, and other settings).
2. Click the "Generate Image(s)" button to start the image generation process.
3. Once generated, each image appears with a download button, allowing for quick saving.

## Requirements

- `mflux-server` running on a specified endpoint (default is `http://localhost:4030`).

## Getting Started

1. Ensure the `mflux-server` is running and accessible.
2. Open the `index.html` file in your browser.
3. Customize your settings and start generating images!

## Customization

You can modify default settings such as the `mflux-server` endpoint directly in the form. The server address is also saved locally for convenience.

## License

This project is licensed under the Apache License 2.0.
