# macOS service deployment

Install Python 3.12 and the dependencies as described in the root README.
Keep the repository at a stable location accessible to your user. The daemon
runs `run.sh` as the user invoking sudo, including its dependency installation
and model downloads. Authenticate to Hugging Face as that user if needed.

From the repository root:

```sh
sudo ./deploy/deploy.sh --model qwen-image-2.1 --quantize 8
```

All arguments are forwarded to `run.sh` and then `server.py`. Omitting them
uses the server defaults (FLUX.2 Klein 4B, 4 steps, no quantization). `run.sh` binds
to `0.0.0.0`; add `--host 127.0.0.1` to restrict access to this Mac.

The script renders `de.anomic.mflux-server.plist.template` into
`/Library/LaunchDaemons/de.anomic.mflux-server.plist`, sets root:wheel ownership,
and reloads the service. It starts at boot and restarts if it exits. Re-run the
deployment command with the desired arguments to change its model or options.
Qwen Image 2.1 is provided by the pinned `mflux==0.20.0` dependency installed
by `run.sh`.

Inspect service status and the combined stdout/stderr log:

```sh
sudo launchctl list | grep de.anomic.mflux-server
tail -f /tmp/de.anomic.mflux-server.log
```

The service may be loaded while its initial weight download is still running.
Check the log for the server startup message, then verify `/api/ps`:

```sh
curl http://localhost:4030/api/ps
```

For a reviewable plist without installing or starting a service:

```sh
.venv/bin/python3.12 deploy/render_plist.py \
  --username "$USER" --runscript "$PWD/run.sh" --output /tmp/mflux-server.plist \
  -- --model qwen-image-2.1 --quantize 8
plutil -lint /tmp/mflux-server.plist
```
