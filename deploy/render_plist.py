"""Render the launchd template with safely escaped server arguments."""

import argparse
import plistlib
from pathlib import Path


def render_plist(username, runscript, server_args):
    template = Path(__file__).with_name("de.anomic.mflux-server.plist.template")
    config = plistlib.loads(template.read_bytes())
    config["UserName"] = username
    config["ProgramArguments"] = ["/bin/bash", str(runscript), *server_args]
    return plistlib.dumps(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument("--runscript", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("server_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    server_args = args.server_args
    if server_args[:1] == ["--"]:
        server_args = server_args[1:]
    args.output.write_bytes(render_plist(args.username, args.runscript, server_args))
