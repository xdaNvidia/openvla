# Getting Started with OpenVLA - LIBERO Simulation

This guide will help you set up and run OpenVLA evaluation in the LIBERO simulation environment.

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for model inference)
- ~30GB disk space for model checkpoints
- ~10GB disk space for LIBERO datasets (optional)

---

## ⚡ Quick Start (Fast Track)

If you want to get started quickly, follow these copy-paste commands:

### 1. Setup Environment

```bash
# Create conda environment
conda create -n openvla python=3.10 -y
conda activate openvla

# Install PyTorch (adjust CUDA version if needed)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Install OpenVLA
cd /home/gear/Projects/openvla
pip install -e .
```

### 2. Install LIBERO

```bash
# Clone and install LIBERO to Projects directory
cd /home/gear/Projects
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
cd /home/gear/Projects/openvla

# Install LIBERO dependencies
pip install -r experiments/robot/libero/libero_requirements.txt

# Fix NumPy version compatibility (TensorFlow requires numpy<2.0)
pip install "numpy<2.0.0,>=1.23.5"
```

### 3. Verify Setup

```bash
python3 check_setup.py
```

### 4. Run Evaluation (Quick Test)

```bash
# This will download the model checkpoint (~30GB) on first run
python3 experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 2
```

**With real-time visualization (GUI window):**
```bash
# Add --use_renderer True to see the simulation in real-time
python3 experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 2 \
  --use_renderer True
```

**What this does:**
- Downloads the pre-trained OpenVLA model (~30GB, first time only)
- Launches LIBERO simulation environment
- Runs 2 episodes per task (10 tasks = 20 total episodes)
- Saves rollout videos to `./rollouts/` directory
- Logs results to `./experiments/logs/`
- **With `--use_renderer True`**: Opens a GUI window showing real-time simulation

---

## 📖 Detailed Setup Guide

### Step 1: Create and Activate Conda Environment

```bash
# Create conda environment
conda create -n openvla python=3.10 -y
conda activate openvla
```

### Step 2: Install PyTorch

Install PyTorch with CUDA support (adjust CUDA version based on your system):

```bash
# For CUDA 12.4 (check your CUDA version with: nvidia-smi)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# OR for CUDA 11.8
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# OR CPU-only (slower, but works)
# conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

### Step 3: Install OpenVLA Package

```bash
cd /home/gear/Projects/openvla
pip install -e .
```

### Step 4: Install LIBERO

```bash
# Clone LIBERO repository to Projects directory
cd /home/gear/Projects
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
cd /home/gear/Projects/openvla  # return to openvla directory
```

### Step 5: Install LIBERO Dependencies

```bash
pip install -r experiments/robot/libero/libero_requirements.txt

# Fix NumPy version compatibility (TensorFlow requires numpy<2.0)
pip install "numpy<2.0.0,>=1.23.5"
```

### Step 6: Verify Installation

Run a quick test to verify everything is installed:

```bash
python3 check_setup.py
```

Or manually test:

```bash
python3 -c "import torch; import libero; import prismatic; print('✓ All modules imported successfully')"
```

---

## 🚀 Running Evaluation

### Quick Start: Run LIBERO-Spatial Evaluation

This will download the pre-trained model checkpoint automatically (~30GB) and run evaluation:

```bash
python3 experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 5  # Reduced for quick testing (default is 50)
```

### Available Task Suites

You can evaluate on different LIBERO task suites:

- `libero_spatial` - Spatial reasoning tasks (recommended to start)
- `libero_object` - Object manipulation tasks  
- `libero_goal` - Goal-conditioned tasks
- `libero_10` - Long-horizon tasks

### Evaluation Options

- `--num_trials_per_task`: Number of episodes per task (default: 50, use 2-5 for quick testing)
- `--seed`: Random seed for reproducibility (default: 7)
- `--use_renderer True`: Enable real-time GUI visualization (opens a window showing the simulation)
- `--use_wandb True`: Enable Weights & Biases logging (optional)
- `--wandb_project <PROJECT>`: W&B project name
- `--wandb_entity <ENTITY>`: W&B entity/username
- `--load_in_8bit True`: Use 8-bit quantization (for low memory GPUs)
- `--load_in_4bit True`: Use 4-bit quantization (for very low memory GPUs)

### Example: Quick Test Run

Run a minimal evaluation to test your setup:

```bash
python3 experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 2 \
  --seed 7
```

**With real-time visualization:**
```bash
python3 experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 2 \
  --seed 7 \
  --use_renderer True
```

This will:
- Download the model checkpoint (first time only, ~30GB)
- Run 2 episodes per task (10 tasks = 20 total episodes)
- Save rollout videos to `./rollouts/` directory
- Log results to `./experiments/logs/`
- **With `--use_renderer True`**: Opens a GUI window showing the simulation in real-time

### Understanding the Output

- **Rollout videos**: Saved in `./rollouts/<DATE>/` directory
- **Evaluation logs**: Saved in `./experiments/logs/` directory
- **Success rate**: Printed to console and logged to file
- **Real-time visualization**: When `--use_renderer True` is used, a GUI window opens showing the simulation in real-time

---

## 🐛 Troubleshooting

### Issue: CUDA out of memory
- **Solution**: Use quantization: `--load_in_8bit True` or `--load_in_4bit True`

### Issue: Model download fails
- **Solution**: Ensure you have HuggingFace access. You may need to login:
  ```bash
  pip install huggingface_hub
  huggingface-cli login
  ```

### Issue: LIBERO import errors
- **Solution**: Make sure LIBERO is installed in editable mode: `pip install -e .` from LIBERO directory

### Issue: NumPy version conflict
- **Solution**: TensorFlow requires numpy<2.0. Fix with:
  ```bash
  pip install "numpy<2.0.0,>=1.23.5"
  ```

### Issue: Missing dependencies
- **Solution**: Reinstall openvla: `pip install -e .` from openvla directory

### Issue: Check installation
- **Solution**: Run `python3 check_setup.py` to verify all dependencies

### Issue: Visualization window doesn't appear
- **Solution**: 
  - Ensure you have a display/GUI environment (not headless)
  - If running over SSH, use X11 forwarding: `ssh -X user@host`
  - Check that `DISPLAY` environment variable is set: `echo $DISPLAY`
  - Try running without `--use_renderer True` to use headless mode (videos still saved)

---

## 💡 Tips

- **First run**: Use `--num_trials_per_task 2` for quick testing
- **Full evaluation**: Use `--num_trials_per_task 50` (default, takes longer)
- **Real-time visualization**: Add `--use_renderer True` to see the simulation in a GUI window
- **Low memory**: Add `--load_in_8bit True` for 8-bit quantization
- **Watch videos**: Check `./rollouts/` directory after evaluation
- **Note**: Real-time visualization requires a display/GUI environment. If running over SSH, use X11 forwarding: `ssh -X user@host`

---

## 📚 Next Steps

1. **Explore the code**: Check `experiments/robot/libero/run_libero_eval.py` to understand how evaluation works
2. **Try different task suites**: Test on `libero_object`, `libero_goal`, or `libero_10`
3. **Fine-tune the model**: See README.md for fine-tuning instructions
4. **Read the paper**: [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)

## 🔗 Resources

- OpenVLA Paper: https://arxiv.org/abs/2406.09246
- LIBERO Project: https://libero-project.github.io/main.html
- HuggingFace Models: https://huggingface.co/openvla
- Full README: `README.md`
