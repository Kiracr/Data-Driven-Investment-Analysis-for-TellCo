#!/bin/bash

# Install required system dependencies
apt-get update && apt-get install -y \
    python3 \
    python3-pip

# Install Python dependencies
pip3 install --no-cache-dir -r requirements.txt
