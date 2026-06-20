#!/usr/bin/env bash
# ============================================================================
# quickstart.sh -- Night Road Vision Training Quick-Start
# ============================================================================
# Sets up the conda environment, installs dependencies, verifies the
# installation, and prints next steps for the training pipeline.
#
# Usage:
#   bash scripts/quickstart.sh            # from the training/ directory
#   bash training/scripts/quickstart.sh   # from the night-road-vision root
#
# Prerequisites:
#   - Conda (Miniconda or Anaconda) installed and on PATH
#   - NVIDIA GPU with CUDA 12.x drivers installed
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ENV_NAME="night-road"
PYTHON_VERSION="3.11"
TORCH_CUDA="cu121"

# Colors for terminal output (disabled if not a TTY)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    RED='' GREEN='' YELLOW='' BLUE='' BOLD='' NC=''
fi

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; }
header(){ echo -e "\n${BOLD}=== $* ===${NC}"; }

# ---------------------------------------------------------------------------
# Locate project root
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# If running from training/scripts/, go up twice to reach night-road-vision/
if [[ -d "${SCRIPT_DIR}/../../training" ]]; then
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
    PROJECT_ROOT="$(pwd)"
fi

TRAINING_DIR="${PROJECT_ROOT}/training"

info "Project root : ${PROJECT_ROOT}"
info "Training dir : ${TRAINING_DIR}"

# ---------------------------------------------------------------------------
# Step 1: Check prerequisites
# ---------------------------------------------------------------------------
header "Step 1: Checking prerequisites"

# Check conda
if ! command -v conda &>/dev/null; then
    fail "Conda is not installed or not on PATH."
    fail "Install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
ok "Conda found: $(conda --version)"

# Check nvidia-smi
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    ok "GPU detected: ${GPU_NAME} (${GPU_MEM} MiB)"
else
    warn "nvidia-smi not found. GPU training will not be available."
    warn "Install NVIDIA drivers: https://www.nvidia.com/drivers"
fi

# ---------------------------------------------------------------------------
# Step 2: Create conda environment
# ---------------------------------------------------------------------------
header "Step 2: Creating conda environment '${ENV_NAME}'"

if conda env list | grep -q "^${ENV_NAME} "; then
    warn "Environment '${ENV_NAME}' already exists."
    read -rp "  Recreate it? (y/N): " RECREATE
    if [[ "${RECREATE}" =~ ^[Yy]$ ]]; then
        info "Removing existing environment..."
        conda env remove -n "${ENV_NAME}" -y
        info "Creating fresh environment..."
        conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
    else
        info "Keeping existing environment."
    fi
else
    info "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi
ok "Environment '${ENV_NAME}' is ready."

# ---------------------------------------------------------------------------
# Step 3: Install PyTorch with CUDA
# ---------------------------------------------------------------------------
header "Step 3: Installing PyTorch with CUDA"

info "Installing PyTorch + torchvision + torchaudio (CUDA ${TORCH_CUDA})..."
conda run -n "${ENV_NAME}" pip install \
    torch torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

# Verify PyTorch CUDA
PYTORCH_CUDA=$(conda run -n "${ENV_NAME}" python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
if [[ "${PYTORCH_CUDA}" == "True" ]]; then
    ok "PyTorch CUDA is available."
else
    warn "PyTorch CUDA not available. Training will fall back to CPU (very slow)."
    warn "Check that your NVIDIA drivers support CUDA 12.x."
fi

# ---------------------------------------------------------------------------
# Step 4: Install project dependencies
# ---------------------------------------------------------------------------
header "Step 4: Installing project dependencies"

info "Installing core packages..."
conda run -n "${ENV_NAME}" pip install \
    "ultralytics>=8.3.0" \
    "albumentations>=1.4.0" \
    "opencv-python>=4.8.0" \
    "pandas>=2.0.0" \
    "pyyaml>=6.0" \
    "tqdm>=4.65.0" \
    "numpy>=1.24.0" \
    "pillow>=10.0.0" \
    "matplotlib>=3.7.0" \
    "seaborn>=0.12.0"

info "Installing optional logging packages..."
conda run -n "${ENV_NAME}" pip install \
    tensorboard \
    wandb 2>/dev/null || warn "wandb install failed (optional -- tensorboard is the default)."

ok "All dependencies installed."

# ---------------------------------------------------------------------------
# Step 5: Run verification checks
# ---------------------------------------------------------------------------
header "Step 5: Running verification checks"

CHECKS_PASSED=0
CHECKS_FAILED=0

# 5a. Python version
PY_VER=$(conda run -n "${ENV_NAME}" python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>/dev/null)
info "Python version: ${PY_VER}"

# 5b. PyTorch
TORCH_VER=$(conda run -n "${ENV_NAME}" python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "NOT INSTALLED")
if [[ "${TORCH_VER}" != "NOT INSTALLED" ]]; then
    ok "PyTorch: ${TORCH_VER}"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
else
    fail "PyTorch not installed."
    CHECKS_FAILED=$((CHECKS_FAILED + 1))
fi

# 5c. CUDA
CUDA_AVAIL=$(conda run -n "${ENV_NAME}" python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
if [[ "${CUDA_AVAIL}" == "True" ]]; then
    CUDA_DEV=$(conda run -n "${ENV_NAME}" python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    CUDA_VER=$(conda run -n "${ENV_NAME}" python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
    ok "CUDA: available (${CUDA_DEV}, CUDA ${CUDA_VER})"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
else
    warn "CUDA: not available (CPU-only mode)"
    CHECKS_FAILED=$((CHECKS_FAILED + 1))
fi

# 5d. Ultralytics + YOLO check
info "Running Ultralytics YOLO checks..."
YOLO_OK=$(conda run -n "${ENV_NAME}" python -c "
from ultralytics import YOLO
import ultralytics
print(f'version: {ultralytics.__version__}')
# Quick sanity: load a nano model (will download if not cached)
model = YOLO('yolov8n.pt')
print('model loaded OK')
" 2>/dev/null)

if [[ $? -eq 0 ]]; then
    ok "Ultralytics: ${YOLO_OK}"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
else
    fail "Ultralytics YOLO check failed."
    CHECKS_FAILED=$((CHECKS_FAILED + 1))
fi

# 5e. Verify training scripts exist
SCRIPTS=("train.py" "evaluate.py" "export_models.py" "prepare_datasets.py" "download_model.py")
for script in "${SCRIPTS[@]}"; do
    if [[ -f "${TRAINING_DIR}/scripts/${script}" ]]; then
        ok "Found: training/scripts/${script}"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
    else
        fail "Missing: training/scripts/${script}"
        CHECKS_FAILED=$((CHECKS_FAILED + 1))
    fi
done

# 5f. Verify config files exist
CONFIGS=("augmentation_night.yaml" "device_rtx3090.yaml" "hyperparams_stage1.yaml" "hyperparams_stage2.yaml")
for cfg in "${CONFIGS[@]}"; do
    if [[ -f "${TRAINING_DIR}/configs/${cfg}" ]]; then
        ok "Found: training/configs/${cfg}"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
    else
        fail "Missing: training/configs/${cfg}"
        CHECKS_FAILED=$((CHECKS_FAILED + 1))
    fi
done

# 5g. nvidia-smi
if command -v nvidia-smi &>/dev/null; then
    ok "nvidia-smi: available"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
else
    warn "nvidia-smi: not found"
    CHECKS_FAILED=$((CHECKS_FAILED + 1))
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
header "Verification Summary"
echo ""
echo "  Checks passed : ${CHECKS_PASSED}"
echo "  Checks failed : ${CHECKS_FAILED}"
echo ""

if [[ ${CHECKS_FAILED} -gt 0 ]]; then
    warn "Some checks failed. Review the output above."
fi

# ---------------------------------------------------------------------------
# Next Steps
# ---------------------------------------------------------------------------
header "Next Steps"
cat <<'NEXTSTEPS'

  1. ACTIVATE THE ENVIRONMENT
     conda activate night-road

  2. PREPARE YOUR DATASETS
     # Process each public source (run from night-road-vision/ root):
     python training/scripts/prepare_datasets.py bdd100k \
         --input /path/to/bdd100k \
         --output training/datasets/night_road

     python training/scripts/prepare_datasets.py nightowls \
         --input /path/to/nightowls \
         --output training/datasets/night_road

     python training/scripts/prepare_datasets.py exdark \
         --input /path/to/exdark \
         --output training/datasets/night_road

     # Extract frames from phone videos:
     python training/scripts/prepare_datasets.py extract-frames \
         --input /path/to/phone_videos \
         --output training/datasets/night_road_phone \
         --fps 2

     # Split and generate configs:
     python training/scripts/prepare_datasets.py split --dataset training/datasets/night_road
     python training/scripts/prepare_datasets.py split --dataset training/datasets/night_road_phone
     python training/scripts/prepare_datasets.py create-yaml --dataset training/datasets/night_road
     python training/scripts/prepare_datasets.py create-yaml --dataset training/datasets/night_road_phone

  3. TRAIN THE MODEL (two stages)
     python training/scripts/train.py both \
         --data-stage1 training/datasets/night_road/night_road.yaml \
         --data-stage2 training/datasets/night_road_phone/night_road.yaml \
         --batch -1 \
         --workers 8

     Or run stages individually:
     python training/scripts/train.py stage1 --data training/datasets/night_road/night_road.yaml --batch -1
     python training/scripts/train.py stage2 --data training/datasets/night_road_phone/night_road.yaml \
         --stage1-weights runs/night-road/stage1_public/weights/best.pt --batch -1

  4. EVALUATE
     python training/scripts/evaluate.py standard \
         --model runs/night-road/stage2_finetune/weights/best.pt \
         --data training/datasets/night_road_phone/night_road.yaml

  5. EXPORT FOR MOBILE
     python training/scripts/export_models.py all \
         --model runs/night-road/stage2_finetune/weights/best.pt \
         --calibration-data training/datasets/night_road_phone/night_road.yaml \
         --android-assets android-app/app/src/main/assets

  For full details, see: training/TRAINING_GUIDE.md

NEXTSTEPS

echo ""
info "Quick-start setup complete."
