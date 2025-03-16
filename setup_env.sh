#!/bin/bash
# Rock Paper Scissors Detection - Conda Environment Setup
# This script creates a new conda environment with all required dependencies

# Set environment name
ENV_NAME="rps_detection"

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up conda environment for Rock Paper Scissors Detection...${NC}"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Conda is not installed or not in PATH. Please install Miniconda or Anaconda first.${NC}"
    exit 1
fi

# Check if environment already exists
ENV_EXISTS=$(conda env list | grep -w "$ENV_NAME" || true)
if [ ! -z "$ENV_EXISTS" ]; then
    echo -e "${YELLOW}Environment '$ENV_NAME' already exists.${NC}"
    read -p "Do you want to remove and recreate it? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Removing existing environment...${NC}"
        conda env remove -n $ENV_NAME
    else
        echo -e "${YELLOW}Exiting without changes.${NC}"
        exit 0
    fi
fi

# Create new environment with Python 3.10
echo -e "${GREEN}Creating new conda environment '$ENV_NAME' with Python 3.10...${NC}"
conda create -n $ENV_NAME python=3.10 -y

# Activate the environment and install packages
echo -e "${GREEN}Installing required packages...${NC}"
conda activate $ENV_NAME || source $(conda info --base)/etc/profile.d/conda.sh && conda activate $ENV_NAME

# Install PyTorch (CPU or GPU version based on availability)
if [ "$(uname)" == "Darwin" ] && [ "$(uname -m)" == "arm64" ]; then
    # MacOS with M1/M2 chip
    echo -e "${GREEN}Detected Apple Silicon. Installing PyTorch with MPS support...${NC}"
    conda install pytorch torchvision -c pytorch -y
elif command -v nvidia-smi &> /dev/null; then
    # NVIDIA GPU available
    echo -e "${GREEN}NVIDIA GPU detected. Installing PyTorch with CUDA support...${NC}"
    conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
else
    # CPU only
    echo -e "${YELLOW}No GPU detected. Installing CPU-only PyTorch...${NC}"
    conda install pytorch torchvision cpuonly -c pytorch -y
fi

# Install other required packages
echo -e "${GREEN}Installing additional dependencies...${NC}"
conda install -y matplotlib numpy scikit-learn seaborn
pip install opencv-python

# Create an activation script to set environment variables
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh << EOF
#!/bin/bash
# Environment variables for RPS Detection
export RPS_PROJECT_HOME=\$CONDA_PREFIX/share/rps_detection
EOF

# Create a directory for the project
mkdir -p $CONDA_PREFIX/share/rps_detection

# Create a simple test script to verify the environment
cat > $CONDA_PREFIX/share/rps_detection/test_env.py << EOF
import torch
import torchvision
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import seaborn as sns

print("Environment test:")
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"Seaborn version: {sns.__version__}")

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Create and run a simple model to check PyTorch functionality
x = torch.randn(1, 3, 128, 128).to(device)
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 16, 3, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2)
).to(device)
y = model(x)
print(f"Test tensor shape: {y.shape}")
print("All dependencies are working correctly!")
EOF

# Create a README with instructions
cat > $CONDA_PREFIX/share/rps_detection/README.md << EOF
# Rock Paper Scissors Detection Environment

This conda environment contains all dependencies needed for the Rock Paper Scissors Detection project.

## Testing the Environment

Run the test script to verify all dependencies are correctly installed:
\`\`\`
python $CONDA_PREFIX/share/rps_detection/test_env.py
\`\`\`

## Running the Project

1. Place your training script, test script, and webcam detection script in your project directory.
2. Run the scripts as follows:
   - Training: \`python train_model.py\`
   - Testing: \`python test_model.py\`
   - Webcam: \`python webcam_detection.py\`

## Environment Details

- Python 3.10
- PyTorch with GPU support (if available)
- OpenCV for webcam capture and image processing
- Matplotlib for visualization
- Scikit-learn for evaluation metrics
- Seaborn for enhanced visualizations
EOF

# Make the test script executable
chmod +x $CONDA_PREFIX/share/rps_detection/test_env.py

echo -e "${GREEN}Environment setup complete!${NC}"
echo -e "${GREEN}To activate the environment, run: conda activate $ENV_NAME${NC}"
echo -e "${GREEN}To test the environment, run: python $CONDA_PREFIX/share/rps_detection/test_env.py${NC}"
