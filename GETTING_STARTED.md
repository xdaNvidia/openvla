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

# IMPORTANT: Fix for PyTorch import errors (iJIT_NotifyEvent)
# Downgrade MKL/Intel OpenMP to compatible versions
conda install intel-openmp=2023.1.0 mkl=2023.1.0 -y

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
# Note: If you encounter LIBERO import errors, use the PYTHONPATH workaround below
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

**If you encounter "ModuleNotFoundError: No module named 'libero'" errors:**
```bash
# Use PYTHONPATH workaround
PYTHONPATH=/home/gear/Projects/LIBERO:$PYTHONPATH python3 experiments/robot/libero/run_libero_eval.py \
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

# IMPORTANT: Fix for PyTorch import errors (iJIT_NotifyEvent symbol not found)
# If you encounter "undefined symbol: iJIT_NotifyEvent" errors, run:
conda install intel-openmp=2023.1.0 mkl=2023.1.0 -y
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

### Issue: PyTorch import error - "undefined symbol: iJIT_NotifyEvent"
- **Solution**: This is caused by incompatible Intel OpenMP/MKL versions. Fix with:
  ```bash
  conda activate openvla
  conda install intel-openmp=2023.1.0 mkl=2023.1.0 -y
  ```
- **Root cause**: Newer versions of Intel OpenMP (2025.x) have compatibility issues with certain PyTorch builds

### Issue: LIBERO import errors ("ModuleNotFoundError: No module named 'libero'")
- **Solution 1**: Make sure LIBERO is installed in editable mode: `pip install -e .` from LIBERO directory
- **Solution 2**: If editable install doesn't work, use PYTHONPATH workaround:
  ```bash
  PYTHONPATH=/home/gear/Projects/LIBERO:$PYTHONPATH python3 experiments/robot/libero/run_libero_eval.py [args]
  ```

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

- **First run**: Use `--num_trials_per_task 2` for quick testing (~10 minutes)
- **Full evaluation**: Use `--num_trials_per_task 50` (default, takes ~4-5 hours)
- **Real-time visualization**: Add `--use_renderer True` to see the simulation in a GUI window
- **Low memory**: Add `--load_in_8bit True` for 8-bit quantization
- **Watch videos**: Check `./rollouts/` directory after evaluation
- **Expected performance**: The model achieves ~80-90% success rate on libero_spatial tasks
- **Note**: Real-time visualization requires a display/GUI environment. If running over SSH, use X11 forwarding: `ssh -X user@host`

## 🎯 Evaluation Results

After running the quick test (2 trials per task on libero_spatial):
- **Total episodes**: 20 (2 trials × 10 tasks)
- **Expected success rate**: ~80-90%
- **Rollout videos**: Saved to `./rollouts/<DATE>/`
- **Logs**: Saved to `./experiments/logs/`

---

## 🎓 Fine-tuning on Custom Datasets

You can fine-tune OpenVLA on your own datasets using LoRA (Low-Rank Adaptation) for parameter-efficient training.

### Prerequisites for Fine-tuning

1. **Dataset Format**: Your dataset should be in RLDS (Reinforcement Learning Datasets) format following the Open X-Embodiment standard
2. **GPU Memory**: At least 48GB GPU memory recommended (or use smaller batch sizes)
3. **Disk Space**: Ensure enough space for model checkpoints (~30GB per checkpoint)

### Quick Start: Fine-tune on LIBERO Dataset

Here's an example of fine-tuning on a LIBERO dataset:

```bash
# Fine-tune OpenVLA on LIBERO Spatial dataset
python -u -m torch.distributed.run \
  --standalone \
  --nnodes 1 \
  --nproc-per-node 1 \
  vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir ./modified_libero_rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir ./runs \
  --adapter_tmp_dir ./adapter-tmp \
  --lora_rank 32 \
  --batch_size 2 \
  --grad_accumulation_steps 4 \
  --learning_rate 5e-4 \
  --max_steps 200000 \
  --image_aug True \
  --save_steps 5000
```

**Note**: By default, Weights & Biases logging is disabled (`use_wandb=False`). To enable W&B logging, add `--use_wandb True --wandb_project <PROJECT> --wandb_entity <ENTITY>`.

### Fine-tuning Parameters Explained

- `--vla_path`: Pre-trained model to start from (default: "openvla/openvla-7b")
- `--data_root_dir`: Path to your RLDS dataset directory
- `--dataset_name`: Name of the dataset to use for fine-tuning
- `--run_root_dir`: Directory to save checkpoints and logs
- `--adapter_tmp_dir`: Temporary directory for LoRA adapter weights
- `--lora_rank`: Rank of LoRA weight matrix (default: 32, higher = more capacity but slower)
- `--batch_size`: Training batch size per GPU (adjust based on GPU memory)
- `--grad_accumulation_steps`: Gradient accumulation steps (effective batch size = batch_size × grad_accumulation_steps)
- `--learning_rate`: Learning rate for fine-tuning (default: 5e-4)
- `--max_steps`: Maximum training steps (default: 200,000)
- `--image_aug`: Whether to use image augmentations (default: True)
- `--save_steps`: Save checkpoint every N steps (default: 5000)
- `--use_wandb`: Enable Weights & Biases logging (default: False)

### Memory Requirements

- **48GB GPU**: `batch_size=12` (without gradient accumulation)
- **80GB GPU**: `batch_size=24` (without gradient accumulation)
- **24GB GPU**: `batch_size=2` with `grad_accumulation_steps=4` (effective batch size = 8)
- **Low Memory**: Add `--use_quantization True` for 4-bit quantization (reduces performance)

### Multi-GPU Training

To use multiple GPUs, adjust `--nproc-per-node`:

```bash
# Example: Use 4 GPUs
python -u -m torch.distributed.run \
  --standalone \
  --nnodes 1 \
  --nproc-per-node 4 \
  vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir ./modified_libero_rlds \
  --dataset_name libero_spatial_no_noops \
  --batch_size 6 \
  --grad_accumulation_steps 4 \
  --learning_rate 5e-4 \
  --max_steps 200000 \
  --image_aug True \
  --save_steps 5000
```

### Monitoring Training

Training progress will be logged to:
- **Console**: Real-time loss, accuracy, and L1 metrics
- **Checkpoints**: Saved to `./runs/<experiment_id>/`
- **Weights & Biases** (optional): Add `--use_wandb True` to enable

### Evaluating Your Fine-tuned Model

After fine-tuning, your model checkpoint will be saved in `./runs/<experiment_id>/`.

#### Finding Your Model Path

The experiment ID is automatically generated based on your training parameters:
```
<model>+<dataset>+b<batch_size>+lr-<learning_rate>+lora-r<rank>+dropout-<dropout>--image_aug
```

For the example above, the path would be:
```
./runs/openvla-7b+libero_spatial_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug/
```

**Quick way to find your model:**
```bash
# List all trained models (most recent first)
ls -lt ./runs/

# Get the most recent model directory
ls -t ./runs/ | head -1
```

#### Evaluation Commands

**Quick test (2 trials per task):**
```bash
python3 experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint ./runs/openvla-7b+libero_spatial_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 2
```

**Full evaluation (50 trials per task):**
```bash
python3 experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint ./runs/openvla-7b+libero_spatial_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 50
```

**With real-time visualization:**
```bash
python3 experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint ./runs/openvla-7b+libero_spatial_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 50 \
  --use_renderer True
```

**Using the most recent checkpoint dynamically:**
```bash
# Automatically use the most recent trained model
CHECKPOINT=$(ls -t ./runs/ | head -1)
python3 experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint ./runs/$CHECKPOINT \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 50
```

#### What to Expect

- **Rollout videos**: Saved to `./rollouts/<DATE>/`
- **Evaluation logs**: Saved to `./experiments/logs/`
- **Success rate**: Printed to console and logged
- **Expected performance**: Fine-tuned models should achieve 80-90%+ success rate on libero_spatial

### Tips for Fine-tuning

- **Start small**: Test with `--max_steps 1000` first to verify everything works
- **Monitor loss**: Loss should decrease steadily; if it plateaus, try adjusting learning rate
- **Save frequently**: Use `--save_steps 1000` for initial testing, then increase to 5000 for full runs
- **Dataset quality**: Better quality demonstration data = better performance
- **Image augmentation**: Keep `--image_aug True` to improve generalization

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
