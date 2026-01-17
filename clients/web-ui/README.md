# MFLUX Modern Web Client

A high-fidelity, modern web interface for the `mflux-server` image generation backend.

## Features

- **Modern UI/UX**: Sophisticated deep charcoal theme with a split-screen layout for an immersive creation experience.
- **Real-time Configuration**: Adjust model, dimensions, steps, and quality with custom controls.
- **Image-to-Image Support**: Drag and drop or upload initial images to guide generation.
- **Gallery View**: Generated images appear in a responsive grid with one-click download.
- **Auto-Persistence**: Server settings are saved automatically.

## Usage

1. **Start the Server**: Ensure your `mflux-server` is running.
   ```bash
   ./run.sh
   ```

2. **Open the Client**: Open `index.html` in any modern web browser. No build step required.

3. **Generate**:
   - Enter your prompt in the large text area.
   - Adjust settings in the sidebar.
   - Click **Generate** (or `Cmd/Ctrl + Enter`).

## Requirements

- `mflux-server` running (default: `http://localhost:4030`)
- A modern web browser (Chrome, Firefox, Safari, Edge)

## Customization

The interface uses modern CSS variables defined in `style.css`. You can easily adjust the color palette (`--bg-main`, `--accent-primary`, etc.) to match your preferences.
