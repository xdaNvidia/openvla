# OpenVLA Learning Resources

Documentation and tutorials for understanding and using OpenVLA with Weights & Biases.

## 📚 Documentation

### Core Concepts
- **[finetune.md](finetune.md)** - Complete guide to understanding the fine-tuning script
  - Explains VLAs, neural networks, and fine-tuning from first principles
  - Step-by-step code walkthrough of `vla-scripts/finetune.py`
  - Configuration options and hyperparameter explanations
  - Assumes no prior ML knowledge

### Weights & Biases (wandb)
- **[wandb.md](wandb.md)** - Comprehensive guide to Weights & Biases
  - What wandb is and why use it
  - Setup and authentication
  - Understanding the dashboard
  - Key features and best practices
  - Troubleshooting common issues

- **[wandb_sweep/wandb_sweeps.md](wandb_sweep/wandb_sweeps.md)** - Complete guide to hyperparameter sweeps
  - What sweeps are and how they work
  - Search strategies (random, Bayesian, grid)
  - Creating and running sweeps
  - Analyzing results
  - Advanced features (early stopping, multi-objective)

- **[wandb_sweep/sweep_quickstart.md](wandb_sweep/sweep_quickstart.md)** - Quick reference card
  - 3-step quick start
  - Common commands
  - Parameter types cheat sheet
  - Example configurations
  - Troubleshooting tips

## 🚀 Tutorials & Examples

### Wandb Tutorials
- **[wandb_tutorial.py](wandb_tutorial.py)** - Interactive tutorial with 7 examples
  - Basic logging
  - Different data types (images, tables, histograms)
  - Comparing multiple runs
  - Robot training simulation
  - Config usage
  - Offline mode
  - Watching model gradients

- **[verify_wandb.py](verify_wandb.py)** - Verification script
  - Checks wandb installation
  - Tests authentication
  - Validates logging works
  - Quick health check

### Sweep Examples
- **[wandb_sweep/sweep_example.py](wandb_sweep/sweep_example.py)** - Complete sweep example
  - Demonstrates creating sweeps programmatically
  - Includes simple training function
  - Shows how to run agents
  - Multiple usage modes

### Sweep Configurations
- **[wandb_sweep/sweep_simple.yaml](wandb_sweep/sweep_simple.yaml)** - Simple OpenVLA sweep
  - Optimizes learning rate, LoRA rank, batch size
  - Uses Bayesian optimization
  - Good starting point

- **[wandb_sweep/sweep_comprehensive.yaml](wandb_sweep/sweep_comprehensive.yaml)** - Full hyperparameter sweep
  - Optimizes all major parameters
  - Includes early stopping
  - For serious tuning

- **[wandb_sweep/sweep_ablation.yaml](wandb_sweep/sweep_ablation.yaml)** - Ablation study
  - Uses grid search
  - Tests effect of each component
  - For understanding what matters

## 🎯 Quick Start Guide

### 1. Understand the Basics
```bash
# Read the core concepts
cat learning/finetune.md
cat learning/wandb.md
```

### 2. Setup Wandb
```bash
# Already done! Your WANDB_API_KEY is in ~/.bashrc
# Open new terminal or run:
source ~/.bashrc

# Verify setup
python learning/verify_wandb.py
```

### 3. Try the Tutorial
```bash
# Run interactive wandb tutorial (offline mode)
WANDB_MODE=offline python learning/wandb_tutorial.py

# Or run online (uploads to wandb)
python learning/wandb_tutorial.py
```

### 4. Try a Sweep
```bash
# Option A: Quick demo sweep
python learning/wandb_sweep/sweep_example.py

# Option B: OpenVLA sweep (requires dataset)
wandb sweep learning/wandb_sweep/sweep_simple.yaml
wandb agent your-username/openvla/sweep-id
```

### 5. Fine-tune with Wandb
```bash
# Run fine-tuning with wandb tracking
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
    --use_wandb True \
    --wandb_entity "xda_nvidia" \
    --dataset_name "droid_wipe" \
    --batch_size 16 \
    --learning_rate 5e-4
```

## 📖 Learning Path

### For Complete Beginners
1. Read `finetune.md` - understand what's happening
2. Read `wandb.md` - understand tracking and visualization
3. Run `verify_wandb.py` - ensure setup works
4. Run `wandb_tutorial.py` - hands-on practice
5. Try fine-tuning with wandb enabled

### For Hyperparameter Tuning
1. Read `wandb_sweep/wandb_sweeps.md` - understand sweeps
2. Check `wandb_sweep/sweep_quickstart.md` - quick reference
3. Run `wandb_sweep/sweep_example.py` - see how it works
4. Create your own sweep with `wandb_sweep/sweep_simple.yaml`
5. Analyze results in dashboard

### For Advanced Users
1. Review `wandb_sweep/sweep_comprehensive.yaml` - see all options
2. Implement custom sweep logic
3. Use early stopping and multi-objective optimization
4. Set up distributed sweeps across multiple machines

## 🔑 Key Files Created

### Documentation
- ✅ `finetune.md` - Fine-tuning guide (beginner-friendly)
- ✅ `wandb.md` - Wandb complete guide
- ✅ `README.md` - This file

### Scripts
- ✅ `wandb_tutorial.py` - Interactive tutorial (8 examples including video logging)
- ✅ `verify_wandb.py` - Setup verification

### Wandb Sweep (wandb_sweep/)
- ✅ `wandb_sweeps.md` - Sweeps detailed guide
- ✅ `sweep_quickstart.md` - Quick reference card
- ✅ `sweep_example.py` - Sweep demonstration
- ✅ `sweep_simple.yaml` - Simple OpenVLA sweep
- ✅ `sweep_comprehensive.yaml` - Full hyperparameter sweep
- ✅ `sweep_ablation.yaml` - Ablation study

### Environment
- ✅ `~/.bashrc` - Added WANDB_API_KEY
- ✅ `~/.netrc` - Wandb credentials (auto-created)

## 🎓 Common Tasks

### Run Fine-tuning with Tracking
```bash
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
    --use_wandb True \
    --wandb_entity "xda_nvidia" \
    --dataset_name "droid_wipe"
```

### Create and Run a Sweep
```bash
# 1. Create sweep
wandb sweep learning/wandb_sweep/sweep_simple.yaml

# 2. Run agents (multiple GPUs)
CUDA_VISIBLE_DEVICES=0 wandb agent username/project/sweep-id &
CUDA_VISIBLE_DEVICES=1 wandb agent username/project/sweep-id &
```

### Compare Experiments
```bash
# All runs are automatically logged to:
# https://wandb.ai/xda_nvidia/openvla

# Select multiple runs in UI to compare
```

### Export Results
```python
import wandb

api = wandb.Api()
runs = api.runs("xda_nvidia/openvla")

for run in runs:
    print(f"{run.name}: {run.summary.get('action_accuracy')}")
```

## 💡 Tips & Best Practices

1. **Always use wandb for experiments**
   - Add `--use_wandb True` to your commands
   - Compare runs easily
   - Never lose results

2. **Start with small sweeps**
   - Use `max_steps: 10000` for sweeps
   - Run full training with best config

3. **Use descriptive names**
   - Add `--run_id_note "description"` to your runs
   - Tag experiments in wandb

4. **Monitor in real-time**
   - Open wandb dashboard while training
   - Catch issues early
   - Stop bad runs early

5. **Document your experiments**
   - Add notes in wandb UI
   - Explain what you tried and why
   - Future you will thank you!

## 🐛 Troubleshooting

**Wandb not working?**
```bash
python learning/verify_wandb.py
```

**Can't remember sweep commands?**
```bash
cat learning/wandb_sweep/sweep_quickstart.md
```

**Want to understand the code better?**
```bash
cat learning/finetune.md | less
```

**Need help with wandb features?**
```bash
cat learning/wandb.md | less
```

## 📊 Your Dashboard

After running experiments, view them at:
- **Main project**: https://wandb.ai/xda_nvidia/openvla
- **Tutorial**: https://wandb.ai/nv-gear/wandb-tutorial
- **Profile**: https://wandb.ai/xda_nvidia

## 🎉 Success Metrics

You're ready when you can:
- ✅ Explain what fine-tuning does (see `finetune.md`)
- ✅ Run fine-tuning with wandb tracking
- ✅ Create and run a hyperparameter sweep
- ✅ Compare multiple experiments in dashboard
- ✅ Identify best hyperparameters from sweep
- ✅ Use best config for final training

## 📚 Additional Resources

- **OpenVLA**: https://github.com/openvla/openvla
- **Wandb Docs**: https://docs.wandb.ai
- **Sweeps Guide**: https://docs.wandb.ai/guides/sweeps
- **Examples**: https://github.com/wandb/examples

---

**Created**: 2025-11-12
**Author**: Claude (with your help!)
**Purpose**: Make OpenVLA fine-tuning accessible and efficient

Happy training! 🚀
