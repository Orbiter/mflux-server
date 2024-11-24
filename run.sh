#!/bin/bash

# Change to the directory where the script is located
cd "$(dirname "$0")"

# check the local environment for python3
python3.12 --version
pip3.12 --version

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    python3.12 -m venv .venv
fi

# always look for upgrades
source .venv/bin/activate
pip3.12 install --upgrade pip
pip3.12 install --upgrade -r requirements.txt

python3.12 server.py --quantize 8 --host 0.0.0.0
