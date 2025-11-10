#!/bin/bash
# Quick setup script for OpenVLA LIBERO evaluation

set -e  # Exit on error

echo "=========================================="
echo "OpenVLA LIBERO Setup Script"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo "Please install conda/miniconda first"
    exit 1
fi

# Check if we're in the openvla directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Please run this script from the openvla root directory"
    exit 1
fi

echo "Step 1: Creating conda environment 'openvla'..."
if conda env list | grep -q "^openvla "; then
    echo "⚠️  Environment 'openvla' already exists. Skipping creation."
    echo "   To recreate, run: conda env remove -n openvla"
else
    conda create -n openvla python=3.10 -y
    echo "✓ Created conda environment"
fi

echo ""
echo "Step 2: Activating environment and installing PyTorch..."
echo "⚠️  Note: This script will install PyTorch with CUDA 12.4"
echo "   If you need a different CUDA version, modify this script"
echo ""
echo "Please run the following commands manually:"
echo ""
echo "  conda activate openvla"
echo "  conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y"
echo "  cd $(pwd)"
echo "  pip install -e ."
echo ""
echo "Then continue with LIBERO installation:"
echo "  cd ~"
echo "  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git"
echo "  cd LIBERO"
echo "  pip install -e ."
echo "  cd $(pwd)"
echo "  pip install -r experiments/robot/libero/libero_requirements.txt"
echo ""
echo "After installation, test with:"
echo "  python3 experiments/robot/libero/run_libero_eval.py \\"
echo "    --model_family openvla \\"
echo "    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \\"
echo "    --task_suite_name libero_spatial \\"
echo "    --center_crop True \\"
echo "    --num_trials_per_task 2"
echo ""

