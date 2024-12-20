#!/bin/bash

# Download and install uv
wget -qO- https://astral.sh/uv/install.sh | sh

# Create virtual environment using uv
uv venv derma_env

# Activate virtual environment
source derma_env/bin/activate

# Install requirements using uv pip
uv pip install -r requirements.txt

# Deactivate virtual environment
deactivate
