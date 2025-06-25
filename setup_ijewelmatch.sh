#!/bin/bash

# Define variables
ENV_NAME="iJewelMatch"

# Check if conda is installed
if ! command -v conda &>/dev/null; then
    echo "conda not found. Please install Anaconda or Miniconda and try again."
    exit 1
fi

# Check if the environment already exists
if conda env list | grep -q "$ENV_NAME"; then
    echo "Environment $ENV_NAME already exists. Activating..."
else
    # Create a new conda environment with Python 3.11.6
    echo "Creating conda environment $ENV_NAME with Python 3.11.6..."
    conda create -n "$ENV_NAME" python=3.11.6 -y
fi

# Activate the conda environment
echo "Activating conda environment $ENV_NAME..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Install required packages
echo "Installing required packages..."
conda install -y flask pillow numpy tqdm werkzeug
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install faiss-cpu

# Run the Flask app
echo "Sparkling up your iJewelMatch..."
python ijewelmatch.py

# Deactivate the conda environment
conda deactivate

echo "iJewelMatch is up and running!"
