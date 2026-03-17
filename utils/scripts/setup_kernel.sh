    #!/bin/bash
# setup_kernel.sh

ENV_NAME="snn_env"

echo "Creating Conda environment: $ENV_NAME"

# check if mamba is available, else use conda
if command -v mamba &> /dev/null; then
    CMD="mamba"
else
    CMD="conda"
fi

# Create environment with python 3.10
$CMD create -n $ENV_NAME python=3.10 -y

# Activate execution for this script context
source $($CMD info --base)/etc/profile.d/conda.sh
$CMD activate $ENV_NAME

echo "Installing core dependencies..."
# Core scientific stack
$CMD install -y numpy pandas matplotlib scikit-learn tqdm jupyterlab seaborn

# PyTorch (Mac friendly - usually cpu build from defaults or pytorch channel, but pip is often easier for latest)
# We will use pip for pytorch to get latest stable for MPS support
pip install torch torchvision torchaudio

echo "Installing SNN and specialized libraries..."
pip install spikingjelly tonic thop einops

echo "Registering Jupyter Kernel..."
pip install ipykernel
python -m ipykernel install --user --name=$ENV_NAME --display-name "Python ($ENV_NAME)"

echo "Done! Environment $ENV_NAME is ready."
echo "To use it, restart your Jupyter server or select 'Python ($ENV_NAME)' as the kernel in the notebook."
