#!/bin/bash

# --- NEW: Sourcing Check ---
# If the script is executed instead of sourced, it can't activate the venv for the user's shell.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo -e "\033[0;31mSTOP! To ensure the venv stays active in this terminal, run the script like this:\033[0m"
    echo -e "\033[1;33msource setup.sh\033[0m"
    # We continue anyway to do the installation, but warn them at the end.
    IS_SOURCED=false
else
    IS_SOURCED=true
fi

# Exit on error
set -e

# Define colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}Starting OpenCLIP environment setup...${NC}"

# 1. Get LiU ID automatically
LIUID=$USER

# 2. Define Paths
NOBACKUP_DIR="/nobackup/$LIUID"
VENV_DIR="$NOBACKUP_DIR/clip_venv"
CACHE_DIR="$NOBACKUP_DIR/cache"

# Check if nobackup exists
if [ ! -d "$NOBACKUP_DIR" ]; then
    echo -e "${RED}Error: Directory $NOBACKUP_DIR does not exist!${NC}"
    exit 1
fi

# 3. Create Cache Directories
mkdir -p "$CACHE_DIR/pip" "$CACHE_DIR/huggingface" "$CACHE_DIR/torch" "$CACHE_DIR/clip"

# 4. Redirect default CLIP cache via Symlink
if [ ! -L "$HOME/.cache/clip" ]; then
    echo -e "${CYAN}Rerouting ~/.cache/clip to nobackup...${NC}"
    rm -rf "$HOME/.cache/clip" 2>/dev/null
    mkdir -p "$HOME/.cache"
    ln -s "$CACHE_DIR/clip" "$HOME/.cache/clip"
fi

# 5. Export variables (for this current session)
export PIP_CACHE_DIR="$CACHE_DIR/pip"
export HF_HOME="$CACHE_DIR/huggingface"
export TORCH_HOME="$CACHE_DIR/torch"

# 6. Create Virtual Environment
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${CYAN}Creating virtual environment in $VENV_DIR...${NC}"
    python3 -m venv "$VENV_DIR"
    
    # Inject variables into the activate script for persistence
    cat <<EOT >> "$VENV_DIR/bin/activate"
export PIP_CACHE_DIR="$CACHE_DIR/pip"
export HF_HOME="$CACHE_DIR/huggingface"
export TORCH_HOME="$CACHE_DIR/torch"
EOT
fi

# 7. Activate the venv
# This works for the script, and if sourced, works for the user.
source "$VENV_DIR/bin/activate"

# 8. Install/Update Requirements
echo -e "${CYAN}Installing/Updating packages...${NC}"
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    pip install torch torchvision open_clip_torch ipykernel
fi

# 9. Register Kernel for VSCode
echo -e "${CYAN}Registering Jupyter kernel...${NC}"
python3 -m ipykernel install --user --name clip_venv --display-name "OpenCLIP ($LIUID)"

echo -e "${GREEN}====================================================${NC}"
echo -e "${GREEN}Setup Finished Successfully!${NC}"

if [ "$IS_SOURCED" = false ]; then
    echo -e "${YELLOW}NOTE: The venv is installed but NOT active in this terminal.${NC}"
    echo -e "${YELLOW}To activate it now, run:${NC} source $VENV_DIR/bin/activate"
else
    echo -e "${GREEN}VENV IS NOW ACTIVE IN THIS TERMINAL.${NC}"
fi

echo -e "${CYAN}VSCode Kernel:${NC} OpenCLIP ($LIUID)"
echo -e "${GREEN}====================================================${NC}"