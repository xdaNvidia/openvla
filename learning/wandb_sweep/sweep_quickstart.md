# Wandb Sweeps Quick Reference

## 🚀 Quick Start (3 Steps)

### 1. Create Sweep Config
```yaml
# sweep.yaml
method: bayes
metric:
  name: accuracy
  goal: maximize
parameters:
  learning_rate:
    min: 0.0001
    max: 0.001
    distribution: log_uniform
  batch_size:
    values: [8, 16, 32]
```

### 2. Initialize Sweep
```bash
wandb sweep sweep.yaml
# Output: wandb agent username/project/abc123xyz
```

### 3. Run Agent(s)
```bash
# Single agent
wandb agent username/project/abc123xyz

# Multiple agents (parallel)
CUDA_VISIBLE_DEVICES=0 wandb agent username/project/abc123xyz &
CUDA_VISIBLE_DEVICES=1 wandb agent username/project/abc123xyz &
```

---

## 📋 Common Commands

```bash
# Create sweep from YAML
wandb sweep sweep.yaml

# Create sweep from Python
python create_sweep.py

# Run agent (unlimited)
wandb agent username/project/sweep-id

# Run agent (5 experiments then stop)
wandb agent --count 5 username/project/sweep-id

# Stop sweep
wandb sweep --stop username/project/sweep-id

# Check sweep status
wandb sweep username/project/sweep-id
```

---

## 🎯 Search Methods Cheat Sheet

| Method | When to Use | Speed | Quality |
|--------|-------------|-------|---------|
| `random` | Initial exploration | ⚡⚡⚡ | ⭐⭐ |
| `bayes` | Main optimization | ⚡⚡ | ⭐⭐⭐⭐ |
| `grid` | Ablation studies | ⚡ | ⭐⭐⭐ |

---

## 📊 Parameter Types

```yaml
# Discrete values
batch_size:
  values: [8, 16, 32, 64]

# Continuous uniform
dropout:
  min: 0.0
  max: 0.5
  distribution: uniform

# Continuous log-uniform (for learning rates!)
learning_rate:
  min: 0.00001
  max: 0.1
  distribution: log_uniform

# Integer range
num_layers:
  min: 2
  max: 10
  distribution: int_uniform

# Constant (not searched)
model_name:
  value: "openvla-7b"

# Categorical
optimizer:
  values: ["adam", "sgd", "adamw"]
```

---

## 🎛️ Example Configs

### Simple (3 parameters)
```yaml
method: bayes
metric: {name: accuracy, goal: maximize}
parameters:
  learning_rate: {min: 0.0001, max: 0.001}
  batch_size: {values: [16, 32]}
  dropout: {min: 0.0, max: 0.3}
```

### OpenVLA Simple
```yaml
program: vla-scripts/finetune.py
method: bayes
metric: {name: action_accuracy, goal: maximize}
parameters:
  learning_rate: {min: 0.0001, max: 0.001, distribution: log_uniform}
  lora_rank: {values: [8, 16, 32, 64]}
  batch_size: {values: [8, 16, 32]}
  dataset_name: {value: "droid_wipe"}
  use_wandb: {value: true}
```

### With Early Stopping
```yaml
method: bayes
metric: {name: accuracy, goal: maximize}
parameters:
  learning_rate: {min: 0.0001, max: 0.001}

early_terminate:
  type: hyperband
  min_iter: 3
  max_iter: 27
  s: 2
```

---

## 🔧 Training Script Requirements

Your script must:

```python
import wandb

def train():
    # 1. Initialize (no hardcoded config)
    wandb.init()

    # 2. Read config from sweep
    lr = wandb.config.learning_rate
    batch_size = wandb.config.batch_size

    # 3. Train and log metrics
    for epoch in range(100):
        loss, acc = train_epoch(lr, batch_size)
        wandb.log({"loss": loss, "accuracy": acc})

if __name__ == "__main__":
    train()
```

---

## 🎨 Dashboard Features

After running sweep, view at: `https://wandb.ai/username/project/sweeps/sweep-id`

**Parallel Coordinates**
- Shows relationship between hyperparameters and metrics
- Color = performance
- Identify patterns visually

**Parameter Importance**
- Which hyperparameters matter most
- Focus tuning on important ones

**Table View**
- All runs sorted by performance
- Export to CSV

**Comparison**
- Select multiple runs
- Compare side-by-side

---

## 💡 Pro Tips

### 1. Start Small
```yaml
# Phase 1: Quick exploration
max_steps: {value: 5000}  # Short runs
count: 10  # Few experiments

# Phase 2: Focused optimization
max_steps: {value: 20000}  # Longer runs
count: 20  # More experiments
```

### 2. Use Log Scale for Learning Rates
```yaml
# ✅ Good - log scale
learning_rate:
  min: 0.00001
  max: 0.1
  distribution: log_uniform

# ❌ Bad - linear scale
learning_rate:
  min: 0.00001
  max: 0.1
  distribution: uniform  # Oversample large values
```

### 3. Parallelize Wisely
```bash
# Random: Full parallel (4+ agents)
method: random
# Run 4 agents simultaneously

# Bayesian: Limited parallel (2-3 agents)
method: bayes
# Run 2-3 agents (needs sequential learning)

# Grid: Full parallel
method: grid
# Run as many agents as you have GPUs
```

### 4. Early Stopping Saves Compute
```yaml
early_terminate:
  type: hyperband
  min_iter: 3  # Check after 3 saves
  max_iter: 27
  s: 2
```
Can save 50-80% of compute time!

---

## 🐛 Troubleshooting

**Problem: "No runs starting"**
```bash
# Check sweep exists
wandb sweep username/project/sweep-id

# Check agent command is correct
wandb agent username/project/sweep-id  # Not sweep.yaml!
```

**Problem: "All runs failing"**
```bash
# Test your script manually first
python train.py --learning_rate 0.001 --batch_size 16

# Check logs in wandb UI
```

**Problem: "Sweep too slow"**
```bash
# Run multiple agents
wandb agent sweep-id &  # Background
wandb agent sweep-id &
wandb agent sweep-id &

# Or reduce training time
max_steps: {value: 5000}  # In sweep config
```

**Problem: "Agent stopped unexpectedly"**
```bash
# Just restart - picks up where it left off
wandb agent sweep-id
```

---

## 📚 Files in /learning/

- `sweep_simple.yaml` - Simple OpenVLA sweep (lr, lora_rank, batch_size)
- `sweep_comprehensive.yaml` - Full hyperparameter sweep
- `sweep_ablation.yaml` - Ablation study (grid search)
- `sweep_example.py` - Python example with demo training function
- `wandb_sweeps.md` - Detailed guide (you are here!)

---

## 🎓 Learning Path

1. **Read**: `wandb_sweeps.md` (full guide)
2. **Try**: `python learning/sweep_example.py` (demo)
3. **Create**: Your own sweep with `sweep_simple.yaml`
4. **Optimize**: Use `sweep_comprehensive.yaml` for real training
5. **Analyze**: Check dashboard, iterate

---

## 📖 Resources

- Official Docs: https://docs.wandb.ai/guides/sweeps
- Examples: https://github.com/wandb/examples/tree/master/examples/sweeps
- Community: https://wandb.ai/community

---

Quick reference created! Save this for easy lookup. 🚀
