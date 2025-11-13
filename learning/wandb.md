# Understanding Weights & Biases (wandb): Complete Guide

## Table of Contents
1. [What is Weights & Biases?](#what-is-weights--biases)
2. [Why Use wandb?](#why-use-wandb)
3. [Setup & Authentication](#setup--authentication)
4. [How wandb Works in OpenVLA](#how-wandb-works-in-openvla)
5. [Understanding the Dashboard](#understanding-the-dashboard)
6. [Key Features](#key-features)
7. [Practical Examples](#practical-examples)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## What is Weights & Biases?

**Weights & Biases (wandb)** is a platform for tracking, visualizing, and comparing machine learning experiments.

### The Problem It Solves

Imagine you're training a model and you need to answer questions like:
- "Which learning rate worked best?"
- "Did the loss decrease smoothly or have spikes?"
- "How does today's run compare to last week's?"
- "What hyperparameters did I use for my best model?"

**Without wandb**: You'd need to:
- Save logs to text files
- Write custom plotting scripts
- Manually track configurations in spreadsheets
- Compare runs by opening multiple terminal windows

**With wandb**: Everything is automatically tracked, visualized, and searchable in one dashboard.

---

## Why Use wandb?

### Core Benefits

1. **Automatic Tracking**
   - Metrics (loss, accuracy, etc.) logged in real-time
   - Hyperparameters saved automatically
   - System metrics (GPU usage, memory) monitored

2. **Visualization**
   - Beautiful, interactive charts
   - Compare multiple runs side-by-side
   - Zoom, pan, and explore your data

3. **Collaboration**
   - Share results with teammates
   - Comment on runs
   - Create reports and presentations

4. **Reproducibility**
   - Every experiment's configuration is saved
   - Git commit hash tracked
   - Environment details recorded

5. **Analysis**
   - Search and filter runs
   - Group experiments by tags
   - Export data for custom analysis

### Analogy

Think of wandb as **Google Analytics for machine learning**:
- Google Analytics tracks website visitors → wandb tracks training metrics
- Google Analytics shows graphs of traffic → wandb shows graphs of loss/accuracy
- Google Analytics helps optimize websites → wandb helps optimize models

---

## Setup & Authentication

### Step 1: Install wandb

```bash
pip install wandb
```

### Step 2: Login with Your API Key

You have an API key: `666ad01b7f3badc1ca7ba42e125a682c6696219d`

**Option A: Command Line Login**
```bash
wandb login
# When prompted, paste your API key
```

**Option B: Set Environment Variable**
```bash
export WANDB_API_KEY="666ad01b7f3badc1ca7ba42e125a682c6696219d"
```

**Option C: Login in Python**
```python
import wandb
wandb.login(key="666ad01b7f3badc1ca7ba42e125a682c6696219d")
```

### Step 3: Verify Authentication

```bash
wandb status
```

You should see:
```
Logged in as: <your-username>
```

### 🔒 Security Note

**IMPORTANT**: Your API key is like a password!
- ❌ Don't commit it to Git
- ❌ Don't share it publicly
- ✅ Use environment variables
- ✅ Add to `.gitignore` if in config files

**Best practice**: Add to `.bashrc` or `.zshrc`:
```bash
export WANDB_API_KEY="your-key-here"
```

---

## How wandb Works in OpenVLA

### Enabling wandb in Fine-tuning

In `vla-scripts/finetune.py`, wandb is controlled by these config options:

```python
use_wandb: bool = False          # Set to True to enable
wandb_project: str = "openvla"   # Project name on wandb
wandb_entity: str = "stanford-voltron"  # Your username or team name
```

### Running Fine-tuning with wandb

**Basic command**:
```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
    --use_wandb True \
    --wandb_entity "your-username"  # Replace with your wandb username
```

**Example with full config**:
```bash
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
    --data_root_dir datasets/open-x-embodiment \
    --dataset_name droid_wipe \
    --use_wandb True \
    --wandb_project "openvla-experiments" \
    --wandb_entity "your-username" \
    --batch_size 16 \
    --learning_rate 5e-4 \
    --lora_rank 32
```

### What Gets Logged?

Looking at `finetune.py:305-313`, wandb logs these metrics every 10 gradient steps:

```python
wandb.log(
    {
        "train_loss": smoothened_loss,        # Cross-entropy loss
        "action_accuracy": smoothened_action_accuracy,  # Token prediction accuracy
        "l1_loss": smoothened_l1_loss,        # Continuous action error
    },
    step=gradient_step_idx,
)
```

**Additionally tracked automatically**:
- All hyperparameters from `FinetuneConfig`
- System metrics (GPU utilization, memory, CPU)
- Training duration
- Git commit hash (if in a git repo)

---

## Understanding the Dashboard

### Accessing Your Dashboard

After starting a run, you'll see a URL like:
```
wandb: 🚀 View run at https://wandb.ai/your-username/openvla/runs/abc123xyz
```

Click it to open the dashboard!

### Main Dashboard Sections

#### 1. **Overview Tab**

Shows a summary of your run:
- **Run name**: Auto-generated or custom (e.g., `ft+openvla-7b+droid_wipe+b16+lr-0.0005`)
- **Status**: Running, finished, crashed, etc.
- **Duration**: How long it's been running
- **Config**: All hyperparameters used

#### 2. **Charts Tab**

Real-time graphs of your metrics:

**Train Loss Chart**
```
10.0 |
     |  •
 8.0 |    •
     |      •
 6.0 |        •
     |          ••
 4.0 |            •••
     |               ••••
 2.0 |                   •••••
     |                        •••••
 0.0 +----------------------------->
     0    50k   100k  150k  200k
         Training Steps
```

**What to look for**:
- ✅ Smooth downward trend = healthy training
- ⚠️ Plateaus early = learning rate too low or model converged
- ❌ Spikes or divergence = learning rate too high

**Action Accuracy Chart**
```
100%|                        •••••••
    |                   •••••
 80%|              •••••
    |          ••••
 60%|       •••
    |     ••
 40%|   ••
    |  •
 20%| •
    |•
  0%+----------------------------->
    0    50k   100k  150k  200k
        Training Steps
```

**What to look for**:
- ✅ Steady increase to 85-95% = good training
- ⚠️ Stuck at 100% = possible overfitting or data leakage
- ❌ Stuck at low % = model not learning

#### 3. **System Tab**

Hardware metrics:
- **GPU Utilization**: Should be 95-100% during training
- **GPU Memory**: Monitor to prevent OOM (out of memory)
- **CPU Usage**: Usually low unless data loading is slow
- **Disk I/O**: High if loading data from disk

#### 4. **Logs Tab**

Raw console output from your training script:
```
Fine-tuning OpenVLA Model `openvla/openvla-7b` on `droid_wipe`
trainable params: 167,772,160 || all params: 7,134,234,624 || trainable%: 2.35
Saving Model Checkpoint for Step 5000
...
```

#### 5. **Files Tab**

- Model checkpoints (if you save them to wandb)
- Dataset statistics
- Config files
- Any custom files you log

---

## Key Features

### 1. Comparing Multiple Runs

**Scenario**: You want to compare different learning rates.

**Steps**:
1. Run experiments with different configs:
   ```bash
   # Run 1
   torchrun ... --learning_rate 1e-4 --run_id_note "lr1e4"

   # Run 2
   torchrun ... --learning_rate 5e-4 --run_id_note "lr5e4"

   # Run 3
   torchrun ... --learning_rate 1e-3 --run_id_note "lr1e3"
   ```

2. In wandb dashboard:
   - Go to your project page
   - Select multiple runs (checkboxes)
   - All their charts overlay automatically

3. **Result**: See which learning rate converges fastest!

**Example comparison**:
```
Loss
10.0|
    | • lr=1e-4 (slow)
    |
 5.0|   ··· lr=5e-4 (optimal)
    |
    |       ••• lr=1e-3 (unstable)
 0.0+--------------------------------
    0        100k       200k
```

### 2. Parallel Coordinates Plot

**What it does**: Shows relationship between hyperparameters and metrics.

**Example**: Find which combination of `lora_rank`, `batch_size`, and `learning_rate` gives best accuracy.

**How to use**:
1. Run many experiments with different configs
2. In wandb: Click "Parallel Coordinates" visualization
3. See patterns:
   - High accuracy runs cluster at `lr=5e-4`
   - `lora_rank` between 16-32 works best
   - `batch_size` doesn't matter much

### 3. Grouping Experiments

**Use case**: You're testing two different datasets.

```python
wandb.init(
    entity="your-username",
    project="openvla",
    group="droid_wipe_experiments",  # Group related runs
    tags=["lora", "baseline"],        # Add searchable tags
)
```

Now you can filter:
- "Show me all `droid_wipe_experiments`"
- "Show me runs tagged with `baseline`"

### 4. Early Stopping Alerts

**Setup**: Get notified if training crashes or metrics look bad.

In wandb UI:
1. Go to "Alerts" tab
2. Create alert: "If `train_loss` > 5.0 after step 10000, send email"
3. Leave training overnight, get alerted if something goes wrong

### 5. Hyperparameter Sweeps

**Goal**: Automatically try many hyperparameter combinations.

**Create sweep config** (`sweep.yaml`):
```yaml
program: vla-scripts/finetune.py
method: bayes  # Bayesian optimization
metric:
  name: action_accuracy
  goal: maximize
parameters:
  learning_rate:
    min: 1e-5
    max: 1e-3
  lora_rank:
    values: [8, 16, 32, 64]
  batch_size:
    values: [8, 16, 24]
```

**Run sweep**:
```bash
# Initialize sweep
wandb sweep sweep.yaml

# Start agents (can run on multiple machines)
wandb agent your-username/openvla/sweep-id
```

wandb will intelligently try combinations and find the best!

---

## Practical Examples

### Example 1: Basic Logging

```python
import wandb

# Initialize run
wandb.init(
    project="my-robot-project",
    entity="your-username",
    config={
        "learning_rate": 5e-4,
        "batch_size": 16,
        "lora_rank": 32,
    }
)

# Training loop
for step in range(1000):
    loss = train_step()  # Your training code

    # Log metrics
    wandb.log({
        "loss": loss,
        "step": step,
    })

# Finish
wandb.finish()
```

### Example 2: Logging Images

```python
import wandb
import numpy as np
from PIL import Image

wandb.init(project="vision-debugging")

# Log image with bounding boxes
image = Image.open("robot_camera.jpg")
wandb.log({
    "robot_view": wandb.Image(
        image,
        caption="Robot's camera view at step 1000"
    )
})
```

**Use case**: Debug what the robot is seeing during failures.

### Example 3: Logging Tables

```python
import wandb

# Create table of model predictions vs ground truth
columns = ["step", "predicted_action", "true_action", "error"]
data = [
    [100, "[0.5, 0.2, 0.1]", "[0.5, 0.25, 0.1]", 0.05],
    [200, "[0.3, 0.4, 0.0]", "[0.3, 0.4, 0.05]", 0.05],
]

table = wandb.Table(columns=columns, data=data)
wandb.log({"predictions": table})
```

### Example 4: Logging Model Checkpoints

```python
import wandb
import torch

# Save model checkpoint to wandb
checkpoint_path = "checkpoints/model_step_5000.pt"
torch.save(model.state_dict(), checkpoint_path)

# Upload to wandb
wandb.save(checkpoint_path)
```

**Benefit**: Access checkpoints from anywhere, share with team.

### Example 5: Custom Metrics

In OpenVLA context, you might want to log additional metrics:

```python
# In your training loop (finetune.py)
if distributed_state.is_main_process and cfg.use_wandb:

    # Existing metrics
    wandb.log({
        "train_loss": smoothened_loss,
        "action_accuracy": smoothened_action_accuracy,
        "l1_loss": smoothened_l1_loss,
    }, step=gradient_step_idx)

    # Add custom metrics
    if gradient_step_idx % 100 == 0:
        wandb.log({
            # Learning rate (if using scheduler)
            "learning_rate": optimizer.param_groups[0]['lr'],

            # Gradient norm (for debugging)
            "grad_norm": torch.nn.utils.clip_grad_norm_(
                trainable_params, max_norm=float('inf')
            ),

            # Per-action dimension accuracy
            "accuracy_pos_x": compute_accuracy_for_dim(0),
            "accuracy_pos_y": compute_accuracy_for_dim(1),
            "accuracy_pos_z": compute_accuracy_for_dim(2),

        }, step=gradient_step_idx)
```

---

## Best Practices

### 1. Naming Conventions

**Good run names** (descriptive):
```
openvla-7b+droid_wipe+b16+lr5e-4+lora-r32
baseline_experiment_v1
ablation_no_image_aug
final_model_2024-01-15
```

**Bad run names** (not helpful):
```
test
run1
new_thing
asdf
```

**Tip**: The script auto-generates good names at `finetune.py:125-137`.

### 2. Use Tags Liberally

```python
wandb.init(
    tags=[
        "baseline",           # Type of experiment
        "droid_wipe",         # Dataset
        "lora_rank_32",       # Key hyperparameter
        "bug_fix_attempt",    # Purpose
        "high_priority",      # Importance
    ]
)
```

### 3. Log Config Explicitly

```python
config = {
    "learning_rate": 5e-4,
    "batch_size": 16,
    "dataset": "droid_wipe",
    "git_commit": "abc123",  # Track code version
    "notes": "Testing new data augmentation",
}

wandb.init(config=config)
```

### 4. Use Groups for Related Runs

```python
# Experiment 1: Baseline
wandb.init(group="ablation_study", name="baseline")

# Experiment 2: No image aug
wandb.init(group="ablation_study", name="no_img_aug")

# Experiment 3: No LoRA
wandb.init(group="ablation_study", name="full_finetune")
```

Now easily compare all ablation experiments!

### 5. Add Notes and Comments

In wandb UI:
- Click "Overview" → "Notes"
- Document:
  - Why you ran this experiment
  - Unexpected observations
  - Next steps
  - Known issues

**Example note**:
```
Trying LoRA rank 64 instead of 32 to see if we can improve
accuracy on the "pick up small objects" task. Previous runs
plateaued at 87% accuracy.

UPDATE (Step 50k): Looking promising! Accuracy at 89% already.

UPDATE (Step 100k): Converged at 92%. This is our new baseline!
```

### 6. Offline Mode for Debugging

When testing code changes (not real experiments):

```bash
# Don't upload to wandb server
export WANDB_MODE=offline

# Run your training
python vla-scripts/finetune.py ...
```

Or in Python:
```python
wandb.init(mode="offline")
```

Logs saved locally in `wandb/` directory. Can upload later if needed:
```bash
wandb sync wandb/offline-run-xxx
```

---

## Troubleshooting

### Problem 1: "wandb: ERROR Error while calling W&B API"

**Cause**: Not logged in or API key invalid.

**Solution**:
```bash
wandb login
# Paste your API key when prompted
```

### Problem 2: Too Much Data, Slow Logging

**Symptom**: Training slows down significantly when wandb is enabled.

**Cause**: Logging too frequently or logging large objects.

**Solution**: Reduce logging frequency:
```python
# Instead of logging every step
if step % 10 == 0:  # Log every 10 steps
    wandb.log({"loss": loss})
```

### Problem 3: Run Not Showing Up

**Cause**: Code crashed before `wandb.finish()` called.

**Solution**:
```python
try:
    # Training code
    train()
finally:
    wandb.finish()  # Ensure run is marked complete
```

### Problem 4: "You are trying to use wandb with multiple processes"

**Cause**: Multiple GPUs/processes trying to log to same run.

**Solution**: Only log from main process (already handled in `finetune.py:305`):
```python
if distributed_state.is_main_process and cfg.use_wandb:
    wandb.log(...)  # Only main process logs
```

### Problem 5: Quotas Exceeded

**Free tier limits**:
- 100 GB storage
- 1 user
- Unlimited runs

**Solution**:
- Delete old runs you don't need
- Avoid logging large files (images, videos)
- Upgrade to paid tier if needed

---

## Advanced Features

### 1. Artifacts for Dataset Versioning

Track dataset versions:

```python
# Log dataset
artifact = wandb.Artifact('droid_wipe_dataset', type='dataset')
artifact.add_dir('datasets/droid_wipe')
wandb.log_artifact(artifact)

# Later, in another run, use that exact dataset version
artifact = wandb.use_artifact('droid_wipe_dataset:v0')
artifact_dir = artifact.download()
```

**Benefit**: Reproducibility - know exactly which data was used.

### 2. Reports for Sharing Results

Create beautiful reports:
1. Go to your project page
2. Click "Create Report"
3. Drag and drop charts, add text, embed images
4. Share URL with colleagues or in papers

**Use case**:
- Weekly progress updates to advisor
- Paper submission figures
- Team presentations

### 3. Integration with Jupyter Notebooks

```python
# In Jupyter
import wandb

# Display runs in notebook
wandb.login()
api = wandb.Api()

runs = api.runs("your-username/openvla")
for run in runs:
    print(f"{run.name}: {run.summary['action_accuracy']}")
```

### 4. Callbacks for Frameworks

If using PyTorch Lightning or HuggingFace Trainer:

```python
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    report_to="wandb",  # Auto-integration!
    run_name="my-run",
)

trainer = Trainer(
    model=model,
    args=args,
    ...
)
```

---

## Quick Reference

### Essential Commands

```bash
# Login
wandb login

# Check status
wandb status

# Pull data from a run
wandb pull your-username/project/run-id

# Sync offline runs
wandb sync wandb/offline-run-xxx

# Disable wandb
export WANDB_MODE=disabled
```

### Essential Python API

```python
import wandb

# Start run
run = wandb.init(
    project="my-project",
    name="experiment-1",
    config={"lr": 0.001},
    tags=["baseline"],
    group="exp-group-1",
)

# Log metrics
wandb.log({"loss": 0.5, "acc": 0.9})

# Log images
wandb.log({"image": wandb.Image("path/to/img.jpg")})

# Save files
wandb.save("checkpoint.pt")

# Finish
wandb.finish()
```

### Key Configuration (for OpenVLA)

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
    --use_wandb True \
    --wandb_project "openvla" \
    --wandb_entity "your-username" \
    --dataset_name "droid_wipe" \
    --batch_size 16 \
    --learning_rate 5e-4 \
    --lora_rank 32 \
    --run_id_note "baseline_experiment"
```

---

## Real-World Example: Complete Fine-Tuning Session

Let's walk through a complete example of using wandb for OpenVLA fine-tuning.

### Scenario
You want to fine-tune OpenVLA on a new dataset and compare different learning rates.

### Step 1: Setup
```bash
# Login to wandb
wandb login
# Paste API key: 666ad01b7f3badc1ca7ba42e125a682c6696219d

# Verify
wandb status
# Output: Logged in as: your-username
```

### Step 2: Run Experiments

**Experiment 1: lr=1e-4**
```bash
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
    --use_wandb True \
    --wandb_project "my-robot-experiments" \
    --wandb_entity "your-username" \
    --dataset_name "droid_wipe" \
    --learning_rate 1e-4 \
    --run_id_note "lr1e4"
```

**Experiment 2: lr=5e-4**
```bash
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
    --use_wandb True \
    --wandb_project "my-robot-experiments" \
    --wandb_entity "your-username" \
    --dataset_name "droid_wipe" \
    --learning_rate 5e-4 \
    --run_id_note "lr5e4"
```

**Experiment 3: lr=1e-3**
```bash
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
    --use_wandb True \
    --wandb_project "my-robot-experiments" \
    --wandb_entity "your-username" \
    --dataset_name "droid_wipe" \
    --learning_rate 1e-3 \
    --run_id_note "lr1e3"
```

### Step 3: Monitor in Real-Time

1. Open browser: `https://wandb.ai/your-username/my-robot-experiments`
2. See all 3 runs updating live
3. Compare charts side-by-side
4. Notice lr=5e-4 converges fastest

### Step 4: Analyze Results

After runs complete:
1. Click "Table" view
2. Sort by "action_accuracy" descending
3. See lr=5e-4 achieved 92% accuracy
4. Export table to CSV for paper/report

### Step 5: Share Results

1. Create report: "Learning Rate Comparison Study"
2. Add charts showing convergence
3. Add table of final metrics
4. Share with team: `https://wandb.ai/your-username/my-robot-experiments/reports/...`

---

## Summary

**Weights & Biases is your experiment tracking companion**:
- 📊 Automatically logs metrics and creates visualizations
- 🔍 Makes comparing experiments effortless
- 🤝 Facilitates collaboration and sharing
- 📝 Ensures reproducibility

**Getting started**:
1. `pip install wandb`
2. `wandb login` (use your API key)
3. Add `--use_wandb True` to training commands
4. Watch your experiments on the dashboard

**Key takeaway**: wandb turns chaotic experiment tracking into organized, insightful analysis. It's like having a lab notebook that updates itself!

---

## Additional Resources

- **Official Docs**: https://docs.wandb.ai/
- **Quickstart**: https://docs.wandb.ai/quickstart
- **Examples**: https://github.com/wandb/examples
- **Community**: https://wandb.ai/community

**Happy experimenting! 🚀**
