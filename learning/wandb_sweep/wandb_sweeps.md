# Weights & Biases Sweeps: Complete Guide

## Table of Contents
1. [What are Sweeps?](#what-are-sweeps)
2. [Why Use Sweeps?](#why-use-sweeps)
3. [How Sweeps Work](#how-sweeps-work)
4. [Search Strategies](#search-strategies)
5. [Creating Sweep Configurations](#creating-sweep-configurations)
6. [Running Sweeps](#running-sweeps)
7. [Practical Examples](#practical-examples)
8. [Analyzing Results](#analyzing-results)
9. [Advanced Features](#advanced-features)

---

## What are Sweeps?

**Sweeps** = Automated hyperparameter optimization

Instead of manually running experiments with different hyperparameters one by one, sweeps:
- 🤖 **Automatically** try different combinations
- 🧠 **Intelligently** choose which combinations to try next
- 📊 **Track** all results in one place
- 🎯 **Find** the best configuration for you

### The Problem Without Sweeps

```bash
# Manual approach - tedious and inefficient!
python train.py --lr 0.001 --batch_size 16  # Run 1
python train.py --lr 0.001 --batch_size 32  # Run 2
python train.py --lr 0.01 --batch_size 16   # Run 3
python train.py --lr 0.01 --batch_size 32   # Run 4
# ... 20 more combinations ...
# Then manually compare results 😫
```

### The Solution With Sweeps

```bash
# Create sweep config once
wandb sweep sweep.yaml

# Start agent(s) - they run all experiments automatically!
wandb agent your-username/project/sweep-id
```

---

## Why Use Sweeps?

### 1. **Save Time**
- No manual experiment tracking
- Run multiple agents in parallel (multi-GPU, multi-machine)
- Early stopping of bad runs

### 2. **Better Results**
- Bayesian optimization explores intelligently
- Finds better hyperparameters than manual search
- Can discover unexpected good combinations

### 3. **Organization**
- All experiments grouped together
- Easy to compare and visualize
- Reproducible configurations

### 4. **Resource Efficiency**
- Early stopping saves compute
- Smart search strategies avoid unnecessary runs
- Parallel execution maximizes hardware usage

---

## How Sweeps Work

### Architecture

```
┌─────────────────┐
│  Sweep Server   │  ← Lives on wandb cloud
│  (Orchestrator) │     Decides which config to try next
└────────┬────────┘
         │
    ┌────┴────┬────────┬────────┐
    │         │        │        │
┌───▼───┐ ┌──▼───┐ ┌──▼───┐ ┌──▼───┐
│Agent 1│ │Agent 2│ │Agent 3│ │Agent 4│  ← Run on your machines
│GPU 0  │ │GPU 1  │ │GPU 2  │ │GPU 3  │    Request configs and train
└───────┘ └──────┘ └──────┘ └──────┘
```

### The Process

1. **Create sweep**: Define search space and strategy
2. **Start agent(s)**: Request configurations from sweep server
3. **Train**: Agent trains model with given config
4. **Report**: Agent logs metrics back to sweep
5. **Optimize**: Server analyzes results and suggests next config
6. **Repeat**: Until budget exhausted or goal reached

---

## Search Strategies

### 1. Grid Search

**What it does**: Try every possible combination

```yaml
method: grid
parameters:
  learning_rate:
    values: [0.001, 0.01, 0.1]
  batch_size:
    values: [16, 32]
```

**Result**: 3 × 2 = 6 runs total

**Pros**:
- Exhaustive - won't miss the best combination
- Simple to understand

**Cons**:
- Expensive with many parameters
- Wastes time on clearly bad regions

**When to use**: Few parameters (<3), small search space

---

### 2. Random Search

**What it does**: Randomly sample from search space

```yaml
method: random
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
    distribution: log_uniform
  batch_size:
    values: [16, 32, 64]
```

**Pros**:
- More efficient than grid search
- Can explore continuous ranges
- Good for many parameters

**Cons**:
- Might miss optimal region
- No learning from previous runs

**When to use**: Initial exploration, many parameters, continuous ranges

---

### 3. Bayesian Optimization (Best!)

**What it does**: Build a model of hyperparameter → metric relationship, intelligently choose next configs

```yaml
method: bayes
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
    distribution: log_uniform
  lora_rank:
    values: [8, 16, 32, 64]
metric:
  name: action_accuracy
  goal: maximize
```

**How it works**:
1. Start with random exploration
2. Build probabilistic model (Gaussian Process)
3. Choose configs that are likely to improve OR explore uncertain regions
4. Update model with new results
5. Repeat

**Pros**:
- Most efficient - finds good configs faster
- Learns from previous runs
- Balances exploration and exploitation

**Cons**:
- Requires sequential runs (can't fully parallelize)
- More complex to understand

**When to use**: Expensive training runs, want best results with fewest experiments

---

## Creating Sweep Configurations

### Basic Structure

```yaml
# sweep.yaml
program: train.py              # Your training script
method: bayes                  # Search strategy
metric:                        # What to optimize
  name: validation_accuracy
  goal: maximize

parameters:                    # Hyperparameters to search
  learning_rate:
    min: 0.0001
    max: 0.1
  batch_size:
    values: [16, 32, 64]
```

### Parameter Types

#### 1. **Discrete Values**
```yaml
batch_size:
  values: [8, 16, 32, 64]
```

#### 2. **Continuous Range (Uniform)**
```yaml
dropout:
  min: 0.0
  max: 0.5
  distribution: uniform  # Equal probability across range
```

#### 3. **Continuous Range (Log Uniform)**
```yaml
learning_rate:
  min: 0.00001   # 1e-5
  max: 0.1       # 1e-1
  distribution: log_uniform  # Good for learning rates!
```

**Why log_uniform?**
- Learning rates span orders of magnitude (0.0001, 0.001, 0.01, 0.1)
- Log scale gives equal attention to each order of magnitude
- Better than uniform which would oversample large values

#### 4. **Integer Range**
```yaml
num_layers:
  min: 2
  max: 10
  distribution: int_uniform
```

#### 5. **Categorical**
```yaml
optimizer:
  values: ["adam", "sgd", "adamw"]

activation:
  values: ["relu", "gelu", "swish"]
```

#### 6. **Constant (Fixed)**
```yaml
model_name:
  value: "openvla-7b"  # Not searched, just logged
```

---

## Running Sweeps

### Step 1: Prepare Your Training Script

Your script must:
1. Use `wandb.init()` to initialize (without hardcoded config)
2. Read hyperparameters from `wandb.config`
3. Log metrics with `wandb.log()`

**Example training script**:
```python
# train.py
import wandb

def train():
    # Initialize wandb - sweep will inject config
    wandb.init()

    # Get hyperparameters from sweep
    lr = wandb.config.learning_rate
    batch_size = wandb.config.batch_size

    # Your training code
    for epoch in range(100):
        loss, acc = train_one_epoch(lr, batch_size)

        # Log metrics - sweep tracks these
        wandb.log({
            "loss": loss,
            "accuracy": acc,
            "epoch": epoch
        })

if __name__ == "__main__":
    train()
```

### Step 2: Create Sweep Config

**sweep.yaml**:
```yaml
program: train.py
method: bayes
metric:
  name: accuracy
  goal: maximize

parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
    distribution: log_uniform
  batch_size:
    values: [16, 32, 64]

early_terminate:
  type: hyperband
  min_iter: 5
```

### Step 3: Initialize Sweep

```bash
wandb sweep sweep.yaml
```

Output:
```
wandb: Creating sweep from: sweep.yaml
wandb: Created sweep with ID: abc123xyz
wandb: View sweep at: https://wandb.ai/your-username/project/sweeps/abc123xyz
wandb: Run sweep agent with: wandb agent your-username/project/abc123xyz
```

### Step 4: Start Agent(s)

**Single agent**:
```bash
wandb agent your-username/project/abc123xyz
```

**Multiple agents in parallel** (different terminals/machines):
```bash
# Terminal 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 wandb agent your-username/project/abc123xyz

# Terminal 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 wandb agent your-username/project/abc123xyz

# Terminal 3 (GPU 2)
CUDA_VISIBLE_DEVICES=2 wandb agent your-username/project/abc123xyz
```

**Limited runs per agent**:
```bash
wandb agent --count 5 your-username/project/abc123xyz
```
Runs 5 experiments then stops.

### Step 5: Monitor Progress

Open the sweep URL in your browser:
```
https://wandb.ai/your-username/project/sweeps/abc123xyz
```

You'll see:
- 📊 **Parallel coordinates plot**: Shows relationship between hyperparameters and metrics
- 📈 **Importance plot**: Which hyperparameters matter most
- 📋 **Table**: All runs sorted by performance
- 🎯 **Best run**: Highlighted automatically

---

## Practical Examples

### Example 1: Simple Sweep for OpenVLA

**Goal**: Find best learning rate and LoRA rank

**sweep_openvla_simple.yaml**:
```yaml
program: vla-scripts/finetune.py
method: bayes
metric:
  name: action_accuracy
  goal: maximize

parameters:
  # Fixed parameters
  vla_path:
    value: "openvla/openvla-7b"
  dataset_name:
    value: "droid_wipe"
  use_wandb:
    value: true
  max_steps:
    value: 10000  # Shorter for sweep

  # Parameters to search
  learning_rate:
    min: 0.0001
    max: 0.001
    distribution: log_uniform

  lora_rank:
    values: [8, 16, 32, 64]

  batch_size:
    values: [8, 16, 32]

# Early stopping - kill bad runs early
early_terminate:
  type: hyperband
  min_iter: 3
  max_iter: 27
  s: 2
```

**Run it**:
```bash
# Create sweep
wandb sweep sweep_openvla_simple.yaml

# Start agent
wandb agent your-username/openvla/sweep-id
```

---

### Example 2: Comprehensive Sweep

**Goal**: Optimize learning rate, LoRA config, batch size, and grad accumulation

**sweep_openvla_comprehensive.yaml**:
```yaml
program: vla-scripts/finetune.py
method: bayes
metric:
  name: action_accuracy
  goal: maximize

parameters:
  # Core training
  learning_rate:
    min: 0.00005
    max: 0.001
    distribution: log_uniform

  batch_size:
    values: [8, 12, 16, 24]

  grad_accumulation_steps:
    values: [1, 2, 4]

  # LoRA config
  lora_rank:
    values: [8, 16, 32, 64]

  lora_dropout:
    min: 0.0
    max: 0.3
    distribution: uniform

  # Data augmentation
  image_aug:
    values: [true, false]

  # Fixed
  vla_path:
    value: "openvla/openvla-7b"
  dataset_name:
    value: "droid_wipe"
  use_wandb:
    value: true
  max_steps:
    value: 20000
  save_steps:
    value: 5000

early_terminate:
  type: hyperband
  min_iter: 5
  max_iter: 20
```

**Advanced: Multi-GPU sweep**:
```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node 2 \
  $(wandb agent --count 3 your-username/openvla/sweep-id)

# Terminal 2
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc-per-node 2 \
  $(wandb agent --count 3 your-username/openvla/sweep-id)
```

---

### Example 3: Grid Search for Ablation Study

**Goal**: Systematically test effect of each component

**sweep_ablation.yaml**:
```yaml
program: vla-scripts/finetune.py
method: grid  # Try all combinations

parameters:
  learning_rate:
    value: 0.0005  # Fix optimal from previous sweep

  lora_rank:
    value: 32  # Fix optimal

  batch_size:
    value: 16  # Fix optimal

  # Ablation parameters
  image_aug:
    values: [true, false]

  use_lora:
    values: [true, false]

  lora_dropout:
    values: [0.0, 0.1]

metric:
  name: action_accuracy
  goal: maximize
```

This creates 2 × 2 × 2 = 8 runs to test all combinations.

---

### Example 4: Custom Sweep in Python

Instead of YAML, create sweeps programmatically:

```python
# create_sweep.py
import wandb

sweep_config = {
    'program': 'vla-scripts/finetune.py',
    'method': 'bayes',
    'metric': {
        'name': 'action_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'min': 0.0001,
            'max': 0.001,
            'distribution': 'log_uniform'
        },
        'lora_rank': {
            'values': [8, 16, 32, 64]
        },
        'batch_size': {
            'values': [8, 16, 32]
        },
        # Fixed params
        'vla_path': {'value': 'openvla/openvla-7b'},
        'dataset_name': {'value': 'droid_wipe'},
        'use_wandb': {'value': True},
        'max_steps': {'value': 10000},
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 3,
        'max_iter': 27,
    }
}

# Create sweep
sweep_id = wandb.sweep(
    sweep_config,
    project="openvla-sweeps",
    entity="your-username"
)

print(f"Sweep ID: {sweep_id}")
print(f"Run with: wandb agent your-username/openvla-sweeps/{sweep_id}")
```

Run:
```bash
python create_sweep.py
```

---

## Analyzing Results

### 1. Parallel Coordinates Plot

**What it shows**: Relationship between hyperparameters and final metric

```
Learning Rate ──────┐
                    │
Batch Size ─────────┤
                    ├──→ Action Accuracy
LoRA Rank ──────────┤
                    │
Dropout ────────────┘
```

**How to use**:
- Lines represent runs
- Color = performance (green = good, red = bad)
- See which hyperparameter values lead to good results

**Example insights**:
- "All best runs have learning_rate between 0.0003 and 0.0007"
- "LoRA rank doesn't matter much - all values work"
- "Batch size 16 or 32 both good, but 8 is bad"

### 2. Parameter Importance Plot

Shows which hyperparameters impact performance most:

```
┌────────────────────────────────────┐
│ learning_rate     ████████████ 0.82│  ← Most important
│ batch_size        ████████ 0.51    │
│ lora_rank         ███ 0.23         │
│ lora_dropout      █ 0.08           │  ← Least important
└────────────────────────────────────┘
```

**Use this to**:
- Focus tuning efforts on important parameters
- Fix unimportant parameters to reduce search space
- Understand your model better

### 3. Comparing Best Runs

Select top runs and compare:
- Training curves side-by-side
- Final metrics in table
- Configuration differences highlighted

### 4. Export Results

```python
import wandb

api = wandb.Api()
sweep = api.sweep("your-username/project/sweep-id")

print(f"Best run: {sweep.best_run().name}")
print(f"Best accuracy: {sweep.best_run().summary['action_accuracy']}")

# Get all runs
for run in sweep.runs:
    print(f"{run.name}: {run.summary.get('action_accuracy', 'N/A')}")
    print(f"  Config: {run.config}")
```

---

## Advanced Features

### 1. Early Stopping - Hyperband

**Problem**: Bad hyperparameters waste GPU time

**Solution**: Stop underperforming runs early

```yaml
early_terminate:
  type: hyperband
  min_iter: 3      # Minimum iterations before stopping
  max_iter: 27     # Maximum iterations
  s: 2             # Aggressiveness (higher = more aggressive)
```

**How it works**:
1. All runs start training
2. After `min_iter` steps, compare performance
3. Stop bottom 50% of runs
4. Continue with top 50%
5. Repeat until only best runs reach `max_iter`

**Savings**: Can reduce compute by 3-5x!

### 2. Multi-Objective Optimization

Optimize multiple metrics simultaneously:

```yaml
metric:
  name: combined_score
  goal: maximize

# In your training code:
wandb.log({
    "accuracy": acc,
    "inference_speed": speed,
    "combined_score": acc * speed  # Combine metrics
})
```

Or use Pareto optimization (coming soon in wandb).

### 3. Nested Sweeps

Sweep over high-level choices, then fine-tune:

**Coarse sweep** (sweep1.yaml):
```yaml
parameters:
  learning_rate:
    values: [0.0001, 0.0003, 0.001, 0.003, 0.01]
  # Wide range, fewer samples
```

After finding lr=0.0003 works best:

**Fine sweep** (sweep2.yaml):
```yaml
parameters:
  learning_rate:
    min: 0.0002
    max: 0.0005
    distribution: uniform
  # Narrow range, more samples
```

### 4. Resuming Failed Runs

If agent crashes:
```bash
# Same command - picks up where it left off
wandb agent your-username/project/sweep-id
```

Sweep server tracks which configs have been tried.

### 5. Stopping a Sweep

```bash
# In UI: Click "Stop Sweep" button
# Or programmatically:
wandb sweep --stop your-username/project/sweep-id
```

### 6. Adding Runs to Existing Sweep

```bash
# Just start another agent anytime
wandb agent your-username/project/sweep-id
```

Can add agents even after some runs complete!

---

## Complete Workflow Example

Let's walk through a real hyperparameter tuning project:

### Phase 1: Initial Exploration (Random Search)

```yaml
# sweep_phase1_explore.yaml
method: random
metric:
  name: action_accuracy
  goal: maximize

parameters:
  learning_rate:
    min: 0.00001
    max: 0.01
    distribution: log_uniform
  lora_rank:
    values: [8, 16, 32, 64, 128]
  batch_size:
    values: [4, 8, 16, 32, 64]
  lora_dropout:
    min: 0.0
    max: 0.5

  # Short runs for exploration
  max_steps:
    value: 5000
```

```bash
wandb sweep sweep_phase1_explore.yaml
wandb agent --count 30 sweep-id  # Try 30 random configs
```

**Results**:
- Best: lr=0.0004, lora_rank=32, batch=16, dropout=0.1
- Accuracy ~85%

### Phase 2: Focused Bayesian Search

```yaml
# sweep_phase2_optimize.yaml
method: bayes
metric:
  name: action_accuracy
  goal: maximize

parameters:
  # Narrow ranges based on Phase 1
  learning_rate:
    min: 0.0002
    max: 0.0008
    distribution: log_uniform

  lora_rank:
    values: [16, 32, 64]  # Remove 8 and 128

  batch_size:
    values: [12, 16, 24]  # Around optimal

  lora_dropout:
    min: 0.05
    max: 0.2

  # Longer runs
  max_steps:
    value: 20000
```

```bash
wandb sweep sweep_phase2_optimize.yaml
wandb agent --count 20 sweep-id
```

**Results**:
- Best: lr=0.00042, lora_rank=32, batch=16, dropout=0.12
- Accuracy ~89%

### Phase 3: Final Training

Use best config from sweep:

```bash
torchrun --nproc-per-node 4 vla-scripts/finetune.py \
    --learning_rate 0.00042 \
    --lora_rank 32 \
    --batch_size 16 \
    --lora_dropout 0.12 \
    --max_steps 200000  # Full training
```

**Final accuracy**: ~92%

---

## Best Practices

### 1. Start Small
- Begin with short training runs (fewer steps)
- Use small search space
- Expand once you understand the landscape

### 2. Use Appropriate Search Strategy
- **Random**: Initial exploration, many parameters
- **Bayes**: Main optimization phase
- **Grid**: Ablation studies, final verification

### 3. Log the Right Metrics
```python
wandb.log({
    # Primary metric for optimization
    "action_accuracy": acc,

    # Secondary metrics for analysis
    "train_loss": loss,
    "val_loss": val_loss,
    "inference_time": time,

    # Diagnostics
    "grad_norm": grad_norm,
    "learning_rate": current_lr,
})
```

### 4. Use Early Stopping
Save 50-80% of compute time with Hyperband.

### 5. Parallelize Wisely
- Random search: Fully parallel (many agents)
- Bayesian: 2-4 agents (needs some sequential learning)

### 6. Document Your Sweeps
Add notes in wandb UI about:
- Goal of sweep
- Insights discovered
- Which config was used for final model

---

## Common Issues & Solutions

### Issue 1: "Sweep not finding good configs"

**Causes**:
- Search space too large
- Not enough runs
- Wrong metric

**Solutions**:
- Narrow search space based on literature
- Run more iterations (Bayesian needs ~20+ to learn)
- Ensure metric is logged correctly

### Issue 2: "All runs failing"

**Causes**:
- Bug in training code
- Invalid hyperparameter combinations
- Resource issues

**Solutions**:
- Test training script manually first
- Add validation in script
- Check GPU memory with different batch sizes

### Issue 3: "Sweep is slow"

**Causes**:
- Running sequentially
- Long training times
- Bayesian waiting for results

**Solutions**:
- Start multiple agents in parallel
- Reduce max_steps for sweep (do full training later)
- Use early stopping

---

## Summary

### Key Takeaways

1. **Sweeps automate hyperparameter search** - save time and find better configs
2. **Choose right search strategy**: Random → Bayesian → Grid
3. **Use early stopping** to save compute
4. **Parallelize** with multiple agents
5. **Analyze results** with parallel coordinates and importance plots
6. **Iterate**: Explore → Optimize → Validate

### Quick Start Recipe

```bash
# 1. Create sweep config (sweep.yaml)
# 2. Initialize sweep
wandb sweep sweep.yaml

# 3. Start agents (multiple GPUs)
CUDA_VISIBLE_DEVICES=0 wandb agent sweep-id &
CUDA_VISIBLE_DEVICES=1 wandb agent sweep-id &
CUDA_VISIBLE_DEVICES=2 wandb agent sweep-id &

# 4. Monitor at wandb.ai
# 5. Use best config for final training
```

### Next Steps

1. Try the example sweep scripts in `/learning/`
2. Create a simple sweep for your current task
3. Analyze results and iterate
4. Read: https://docs.wandb.ai/guides/sweeps

Happy sweeping! 🧹✨
