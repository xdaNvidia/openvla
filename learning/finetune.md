# Understanding OpenVLA Fine-Tuning: A Beginner's Guide

## Table of Contents
1. [Core Concepts](#core-concepts)
2. [What is Fine-Tuning?](#what-is-fine-tuning)
3. [Key Technologies](#key-technologies)
4. [Code Walkthrough](#code-walkthrough)
5. [Training Process](#training-process)
6. [Configuration Options](#configuration-options)

---

## Core Concepts

### What is a VLA (Vision-Language-Action Model)?

Think of a VLA as a robot's "brain" that:
- **Sees** images from a camera (Vision)
- **Understands** instructions in natural language (Language)
- **Decides** what movements to make (Action)

For example:
- Input: An image of a table + instruction "pick up the cup"
- Output: Robot arm movements (angles, positions, gripper state)

### What is a Neural Network?

A neural network is like a complex mathematical function with millions of "knobs" (called **parameters** or **weights**). These knobs are adjusted during training so the network learns to map inputs to correct outputs.

---

## What is Fine-Tuning?

### The Big Picture

1. **Pre-training**: A model is first trained on massive amounts of data (expensive, takes weeks/months)
2. **Fine-tuning**: You take this pre-trained model and adapt it to your specific task (cheaper, takes hours/days)

**Analogy**:
- Pre-training = Medical school (learning general medicine)
- Fine-tuning = Specialization (becoming a cardiologist)

### Why Fine-Tune?

- **Customization**: Adapt the model to your specific robot, environment, or tasks
- **Efficiency**: Much faster and cheaper than training from scratch
- **Performance**: Often works better than the general model for specific tasks

---

## Key Technologies

### 1. LoRA (Low-Rank Adaptation)

**The Problem**: Fine-tuning all parameters of a large model requires:
- Huge GPU memory
- Long training time
- Risk of "forgetting" what was learned during pre-training

**The Solution**: LoRA
- Instead of changing all the original weights, add small "adapter" layers
- Only train these small adapters (maybe 1% of total parameters)
- Saves memory and speeds up training

**Analogy**: Instead of rebuilding a house (full fine-tuning), you just add new furniture and decorations (LoRA adapters).

**Technical Details**:
```python
# Instead of updating this:
W_original [4096 x 4096] = 16 million parameters

# LoRA adds these small matrices:
A [4096 x 32] = 131k parameters
B [32 x 4096] = 131k parameters
# Total: ~262k parameters (1.6% of original)
```

### 2. Quantization

**What it is**: Reducing the precision of numbers used in the model
- Normal: 32-bit floating point numbers
- Quantized: 4-bit or 8-bit integers

**Trade-off**:
- ✅ Uses 4-8x less memory
- ❌ Slight loss in accuracy

**When to use**: When you have limited GPU memory

### 3. Distributed Training

**What it is**: Using multiple GPUs simultaneously
- Each GPU processes a different batch of data
- Gradients are synchronized across GPUs
- Speeds up training proportionally to number of GPUs

**In the code**: Uses PyTorch's `DistributedDataParallel` (DDP)

---

## Code Walkthrough

### Configuration (`FinetuneConfig` class, lines 75-111)

```python
@dataclass
class FinetuneConfig:
    vla_path: str = "openvla/openvla-7b"
    # Which pre-trained model to start from

    data_root_dir: Path = Path("datasets/open-x-embodiment")
    # Where your training data is stored

    dataset_name: str = "droid_wipe"
    # Which dataset to use (e.g., robot wiping a table)

    batch_size: int = 16
    # How many examples to process at once
    # Larger = faster but needs more memory

    max_steps: int = 200_000
    # How long to train (200,000 gradient updates)

    learning_rate: float = 5e-4
    # How big each parameter update step is
    # Too large = unstable, too small = slow learning

    lora_rank: int = 32
    # Size of LoRA adapters (larger = more capacity but more memory)
```

### Main Training Function: Step-by-Step

#### Step 1: Setup (lines 118-122)
```python
distributed_state = PartialState()
torch.cuda.set_device(device_id := distributed_state.local_process_index)
```
**What's happening**:
- Detects how many GPUs are available
- Assigns each process to a specific GPU
- Sets up for multi-GPU training

#### Step 2: Create Experiment ID (lines 124-138)
```python
exp_id = (
    f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
    f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
    f"+lr-{cfg.learning_rate}"
)
```
**What's happening**:
- Creates a unique name for this training run
- Example: `openvla-7b+droid_wipe+b16+lr-0.0005+lora-r32`
- Helps you track different experiments

#### Step 3: Load the Model (lines 157-165)
```python
processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    cfg.vla_path,
    torch_dtype=torch.bfloat16,  # Use 16-bit floats (saves memory)
    quantization_config=quantization_config,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
```
**What's happening**:
- **Processor**: Handles image and text preprocessing
  - Resizes images to correct size
  - Converts text to numbers (tokenization)
- **Model**: The actual neural network
  - `torch.bfloat16`: Uses 16-bit numbers instead of 32-bit (saves memory)
  - `quantization_config`: Optional 4-bit quantization

#### Step 4: Apply LoRA (lines 174-183)
```python
if cfg.use_lora:
    lora_config = LoraConfig(
        r=cfg.lora_rank,                    # Rank of adapter matrices
        lora_alpha=min(cfg.lora_rank, 16),  # Scaling factor
        lora_dropout=cfg.lora_dropout,       # Regularization
        target_modules="all-linear",         # Apply to all linear layers
        init_lora_weights="gaussian",        # How to initialize
    )
    vla = get_peft_model(vla, lora_config)
```
**What's happening**:
- Wraps the model with LoRA adapters
- Only these adapters will be trained
- `target_modules="all-linear"`: Add adapters to all linear transformation layers

#### Step 5: Wrap for Multi-GPU (line 186)
```python
vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True)
```
**What's happening**:
- `DDP` = DistributedDataParallel
- Synchronizes gradients across multiple GPUs
- Each GPU has a copy of the model

#### Step 6: Create Optimizer (lines 189-190)
```python
trainable_params = [param for param in vla.parameters() if param.requires_grad]
optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
```
**What's happening**:
- **Optimizer**: Algorithm that updates the parameters
- **AdamW**: A popular, efficient optimizer (Adam with weight decay)
- **trainable_params**: Only LoRA adapters if using LoRA, all params otherwise

#### Step 7: Load Dataset (lines 210-223)
```python
batch_transform = RLDSBatchTransform(
    action_tokenizer,
    processor.tokenizer,
    image_transform=processor.image_processor.apply_transform,
    prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
)
vla_dataset = RLDSDataset(
    cfg.data_root_dir,
    cfg.dataset_name,
    batch_transform,
    resize_resolution=tuple(vla.module.config.image_sizes),
    shuffle_buffer_size=cfg.shuffle_buffer_size,
    image_aug=cfg.image_aug,
)
```
**What's happening**:
- **RLDSDataset**: Loads robotics data in RLDS format
- **batch_transform**: Preprocesses each example:
  - Resizes images
  - Converts actions to tokens (discrete numbers)
  - Formats prompts
- **shuffle_buffer_size**: Randomizes data order (helps learning)
- **image_aug**: Random image modifications (brightness, contrast) to prevent overfitting

#### Step 8: Training Loop (lines 251-370)

##### Core Training Step (lines 255-268)
```python
for batch_idx, batch in enumerate(dataloader):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"],
        )
        loss = output.loss

    normalized_loss = loss / cfg.grad_accumulation_steps
    normalized_loss.backward()
```

**What's happening (in simple terms)**:

1. **Get a batch**: Load a batch of examples (images + instructions + actions)

2. **Forward pass**: Feed data through the model
   - `input_ids`: Text instruction as numbers
   - `pixel_values`: Image data
   - `labels`: Ground truth actions (what the robot should do)
   - `loss`: How wrong the predictions are (lower is better)

3. **Backward pass**: Calculate gradients
   - Gradients = directions to adjust parameters to reduce loss
   - `backward()`: Automatic differentiation computes these

4. **Gradient accumulation**:
   - Divide loss by accumulation steps
   - Allows simulating larger batch sizes without more memory

##### Computing Metrics (lines 270-292)
```python
# Get predicted actions
action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
action_preds = action_logits.argmax(dim=2)
action_gt = batch["labels"][:, 1:].to(action_preds.device)

# Compute accuracy
mask = action_gt > action_tokenizer.action_token_begin_idx
correct_preds = (action_preds == action_gt) & mask
action_accuracy = correct_preds.sum().float() / mask.sum().float()

# Compute L1 loss (average absolute error)
continuous_actions_pred = action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
continuous_actions_gt = action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
```

**What's happening**:
- **Action accuracy**: Percentage of correctly predicted action tokens
- **L1 loss**: Average error in continuous action space (e.g., joint angles)
  - Example: Predicted angle 45°, actual 47° → error = 2°
- **Mask**: Only evaluate on actual action tokens (ignore text tokens)

##### Optimizer Step (lines 316-319)
```python
if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
    optimizer.step()    # Update parameters
    optimizer.zero_grad()  # Clear gradients
    progress.update()
```

**What's happening**:
- Every N micro-batches (gradient accumulation):
  1. Update parameters using accumulated gradients
  2. Reset gradients to zero
  3. Update progress bar

##### Saving Checkpoints (lines 322-365)
```python
if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
    # Save processor and model
    processor.save_pretrained(run_dir)
    vla.module.save_pretrained(save_dir)

    # Merge LoRA adapters into full model
    if cfg.use_lora:
        base_vla = AutoModelForVision2Seq.from_pretrained(cfg.vla_path, ...)
        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        merged_vla = merged_vla.merge_and_unload()
        merged_vla.save_pretrained(run_dir)
```

**What's happening**:
- Every `save_steps` (default 5000):
  1. Save current model state to disk
  2. If using LoRA:
     - Save adapter weights separately
     - Load original model
     - Merge adapters into original model
     - Save merged model (for easier deployment)

**Why merge?**
- LoRA adapters are separate from base model during training
- Merging combines them into a single model
- Makes inference simpler and slightly faster

---

## Training Process

### The Learning Cycle

```
1. Load batch of data (images + instructions + actions)
   ↓
2. Forward pass: Model predicts actions
   ↓
3. Compute loss: Compare predictions to ground truth
   ↓
4. Backward pass: Compute gradients
   ↓
5. Optimizer step: Update parameters to reduce loss
   ↓
6. Repeat for 200,000 steps
```

### What Happens During Training?

**Initially**: Model makes random predictions (high loss)

**Over time**:
- Loss decreases
- Action accuracy increases
- Model learns to:
  - Recognize objects in images
  - Understand instructions
  - Map them to correct robot actions

**Example progression**:
- Step 0: Accuracy 5%, Loss 10.5
- Step 10,000: Accuracy 45%, Loss 2.3
- Step 100,000: Accuracy 85%, Loss 0.8
- Step 200,000: Accuracy 92%, Loss 0.5

---

## Configuration Options

### Memory vs Performance Trade-offs

| Configuration | Memory Usage | Training Speed | Model Quality |
|---------------|--------------|----------------|---------------|
| Full fine-tuning | Very High | Slow | Best |
| LoRA (rank 32) | Medium | Fast | Good |
| LoRA + Quantization | Low | Medium | Okay |

### Recommended Settings

**High-memory GPU (80GB A100)**:
```python
batch_size: 24
use_lora: True
lora_rank: 32
use_quantization: False
```

**Medium-memory GPU (48GB A6000)**:
```python
batch_size: 12
use_lora: True
lora_rank: 32
use_quantization: False
```

**Low-memory GPU (24GB RTX 3090)**:
```python
batch_size: 4
use_lora: True
lora_rank: 16
use_quantization: True  # Enable 4-bit quantization
grad_accumulation_steps: 4  # Simulate larger batch
```

### Hyperparameters Explained

**Learning Rate** (`learning_rate: 5e-4`)
- Controls how much parameters change each step
- Too high: Training unstable, loss explodes
- Too low: Training too slow
- 5e-4 (0.0005) is a good starting point for LoRA

**Batch Size** (`batch_size: 16`)
- Number of examples processed together
- Larger: More stable gradients, faster training, more memory
- Smaller: Less memory, noisier gradients

**LoRA Rank** (`lora_rank: 32`)
- Size of adapter matrices
- Larger: More expressive, more memory
- Smaller: Less memory, might underfit
- 32 is a good balance

**Gradient Accumulation** (`grad_accumulation_steps: 1`)
- Simulate larger batch size without more memory
- If you want effective batch size 64 but only fit 16:
  - Set `batch_size: 16`
  - Set `grad_accumulation_steps: 4`
  - Accumulates gradients over 4 batches before updating

---

## Common Questions

### Q: How long does training take?
**A**: Depends on:
- Number of GPUs: 1 GPU → ~2-3 days, 4 GPUs → ~12-18 hours
- Dataset size: More data = more epochs needed
- Hardware: A100 is faster than V100

### Q: How do I know if training is working?
**A**: Monitor these metrics:
- **Loss should decrease**: If it's stuck or increasing, something's wrong
- **Accuracy should increase**: Eventually plateaus at 85-95%
- **L1 loss should decrease**: Continuous action error gets smaller

### Q: When should I stop training?
**A**: When:
- Metrics plateau (no improvement for many steps)
- You reach `max_steps`
- You test on real robot and it works well enough

### Q: What if I run out of memory?
**A**: Try in this order:
1. Reduce `batch_size`
2. Enable `use_quantization: True`
3. Reduce `lora_rank` (e.g., 32 → 16)
4. Increase `grad_accumulation_steps` (to compensate for smaller batch)
5. Reduce `shuffle_buffer_size`

### Q: Can I resume training if it stops?
**A**: Yes! The script saves checkpoints every `save_steps`. You can:
1. Load the checkpoint model as your starting point
2. Continue training from there

---

## Key Takeaways

1. **Fine-tuning adapts a pre-trained model to your specific task**
2. **LoRA makes fine-tuning efficient by only training small adapters**
3. **The training loop repeatedly adjusts parameters to minimize prediction error**
4. **Proper configuration balances memory usage, speed, and model quality**
5. **Monitor loss and accuracy to ensure training is progressing**

---

## Next Steps

1. **Try running fine-tuning**: Start with default settings
2. **Understand your data**: Look at examples from your dataset
3. **Monitor training**: Use Weights & Biases to track metrics
4. **Evaluate on robot**: Test the fine-tuned model in real environment
5. **Iterate**: Adjust hyperparameters based on results
