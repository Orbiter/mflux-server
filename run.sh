#!/bin/bash

# Change to the directory where the script is located
cd "$(dirname "$0")"

# check the local environment for python3
python3 --version
pip3 --version

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    source .venv/bin/activate
    pip3 install --upgrade pip
    pip3 install -r requirements.txt
else
    source .venv/bin/activate
    # Check if requirements are already installed
    if ! pip3 check > /dev/null 2>&1; then
        pip3 install --upgrade pip
        pip3 install -r requirements.txt
    fi
fi

python3 server.py --quantize 8 --host 0.0.0.0
