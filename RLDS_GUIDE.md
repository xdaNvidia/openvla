# RLDS Dataset Guide

This guide explains RLDS (Robotic Learning Datasets) format and provides tools for visualization and inspection.

## What is RLDS?

RLDS is a standardized format for storing robot learning data using TensorFlow Datasets (TFDS). It's designed to make robot learning datasets easy to share and use across different projects.

### Dataset Structure

```
RLDS Dataset
├── Trajectories (Episodes)
│   ├── Steps (Timesteps)
│   │   ├── observation
│   │   │   ├── image_primary       # RGB camera view
│   │   │   ├── image_wrist        # Wrist camera (optional)
│   │   │   ├── proprio            # Joint angles, gripper state
│   │   │   └── timestep           # Current timestep
│   │   ├── action                 # Robot action (7-DOF usually)
│   │   ├── task
│   │   │   └── language_instruction  # Task description
│   │   └── reward (optional)
```

### Key Components

1. **Trajectories**: A complete episode of robot interaction (from start to end of a task)
2. **Steps**: Individual timesteps within a trajectory
3. **Observations**: What the robot senses
   - **Images**: Camera observations (usually 256x256 RGB)
   - **Proprio**: Proprioceptive state (joint positions, gripper state, etc.)
4. **Actions**: What the robot does
   - Usually 7D: [x, y, z, roll, pitch, yaw, gripper]
   - Can be absolute positions or relative deltas
5. **Task**: Language instruction or goal specification

## Dataset Statistics and Normalization

From `prismatic/vla/datasets/rlds/utils/data_utils.py:185-271`, the system computes:
- **Mean/Std**: For standard normalization
- **Min/Max**: For bounds normalization [-1, 1]
- **Quantiles (q01, q99)**: For robust normalization
- **Trajectory counts**: Number of episodes and transitions

These statistics are cached in `dataset_statistics_{hash}.json` files.

## Using the Visualization Tools

### 0. Web Browser Viewer (Recommended) ⭐

**Interactive web-based viewer** - Browse your dataset in a browser with sliders and real-time visualization!

```bash
# Install dependencies
pip install -r scripts/rlds/requirements_web.txt

# Run locally
python scripts/rlds/gradio_viewer.py \
    --data_dir ./modified_libero_rlds \
    --dataset_name libero_spatial_no_noops \
    --max_trajectories 50

# Open http://localhost:7860 in your browser
```

**Features:**
- 🎮 Interactive sliders to scrub through trajectories and timesteps
- 🖼️ View all camera angles simultaneously
- 📊 Real-time action trajectory plots
- 🔗 `--share` flag creates public URL for remote viewing
- 🌐 Can be deployed to Hugging Face Spaces

### 1. Quick Inspection (Text-based)

```bash
python scripts/rlds/inspect_rlds.py \
    --data_dir ./modified_libero_rlds \
    --dataset_name libero_spatial_no_noops \
    --num_trajectories 3
```

This prints:
- Dataset metadata
- Available splits
- Trajectory structure
- Action statistics
- Observation details

### 2. Visual Inspection (Images & Plots)

```bash
python scripts/rlds/visualize_rlds.py \
    --data_dir ./modified_libero_rlds \
    --dataset_name libero_spatial_no_noops \
    --num_trajectories 5 \
    --save_dir ./visualizations
```

This creates:
- `traj_N_images.png`: Key frames from the trajectory
- `traj_N_actions.png`: Action profile over time

Options:
- `--num_trajectories N`: Number of trajectories to visualize
- `--split train[:10]`: Which split to use
- `--save_dir PATH`: Save figures instead of displaying

### 3. Video Creation

```bash
# Install ffmpeg first
sudo apt-get install ffmpeg

# Create videos
python scripts/rlds/visualize_rlds.py \
    --data_dir ./modified_libero_rlds \
    --dataset_name libero_spatial_no_noops \
    --num_trajectories 1 \
    --create_video \
    --save_dir ./videos
```

This creates `traj_N.mp4` videos showing:
- All camera views side-by-side
- Action trajectory plot (live updating)
- Frame counter

## Understanding Your Dataset

### LIBERO Dataset Structure

Your `modified_libero_rlds` directory contains LIBERO datasets with:
- **No-op filtering**: Transitions with zero actions removed (see `regenerate_libero_dataset.py:46-68`)
- **High-res images**: 256x256 instead of 128x128
- **Two camera views**:
  - `agentview_rgb`: Third-person view
  - `eye_in_hand_rgb`: Wrist-mounted camera

### Action Space

LIBERO uses 7-DOF actions:
```python
action = [
    delta_x,      # End-effector x displacement
    delta_y,      # End-effector y displacement
    delta_z,      # End-effector z displacement
    delta_roll,   # Roll rotation
    delta_pitch,  # Pitch rotation
    delta_yaw,    # Yaw rotation
    gripper       # Gripper command (open/close)
]
```

## Loading Data in Python

### Basic Loading

```python
import tensorflow_datasets as tfds

# Load dataset
builder = tfds.builder('libero_spatial_no_noops',
                       data_dir='./modified_libero_rlds')
dataset = builder.as_dataset(split='train')

# Iterate through trajectories
for trajectory in dataset:
    for step in trajectory['steps']:
        obs = step['observation']
        action = step['action'].numpy()

        # Access images
        agentview = obs['agentview_rgb']
        wrist = obs['eye_in_hand_rgb']

        # Decode if needed
        if agentview.dtype == tf.string:
            agentview = tf.io.decode_image(agentview).numpy()
```

### Using OpenVLA's Utilities

```python
from prismatic.vla.datasets.rlds.dataset import make_dataset_from_rlds

# Load with OpenVLA's pipeline
dataset, stats = make_dataset_from_rlds(
    name='libero_spatial_no_noops',
    data_dir='./modified_libero_rlds',
    train=True,
    image_obs_keys={'primary': 'agentview_rgb', 'wrist': 'eye_in_hand_rgb'},
    state_obs_keys=['joint_states', 'gripper_states'],
    language_key='language_instruction',
    action_proprio_normalization_type='bounds'
)

# Dataset is now normalized and in standard format
for traj in dataset:
    # traj['observation']['image_primary']  # 256x256x3
    # traj['observation']['proprio']         # Concatenated state
    # traj['action']                         # Normalized to [-1, 1]
    # traj['task']['language_instruction']
    pass
```

## Common Patterns

### 1. Filter by Task

```python
def filter_by_task(trajectory, task_text):
    for step in trajectory['steps']:
        if 'language_instruction' in step:
            instruction = step['language_instruction'].numpy().decode('utf-8')
            return task_text.lower() in instruction.lower()
    return False

dataset = dataset.filter(lambda x: filter_by_task(x, 'put'))
```

### 2. Extract Action Sequences

```python
def get_action_sequence(trajectory):
    actions = []
    for step in trajectory['steps']:
        actions.append(step['action'].numpy())
    return np.array(actions)
```

### 3. Create Image Sequences

```python
def get_image_sequence(trajectory, camera='agentview_rgb'):
    images = []
    for step in trajectory['steps']:
        img = step['observation'][camera]
        if img.dtype == tf.string:
            img = tf.io.decode_image(img)
        images.append(img.numpy())
    return np.array(images)
```

## Data Pipeline in OpenVLA

The training pipeline (from `prismatic/vla/datasets/rlds/dataset.py`) applies:

1. **Standardization**: Convert to common format
2. **Normalization**: Scale actions/proprio to [-1, 1] or mean=0, std=1
3. **Goal Relabeling**: Optionally relabel trajectory goals
4. **Chunking**: Create observation-action windows
5. **Augmentation**: Apply image augmentation (random crops, color jitter)
6. **Batching**: Batch multiple frames together

## Troubleshooting

### "Dataset not found"
Make sure the dataset directory structure is:
```
modified_libero_rlds/
└── libero_spatial_no_noops/
    ├── 1.0.0/
    │   ├── dataset_info.json
    │   └── libero_spatial_no_noops-train.tfrecord-*
```

### "Cannot decode image"
Some images are stored as encoded strings (JPEG/PNG). Decode them:
```python
if img.dtype == tf.string:
    img = tf.io.decode_image(img)
```

### Memory Issues
RLDS datasets can be large. Use:
- `.take(N)` to limit number of trajectories
- `.cache()` carefully (only for small datasets)
- Streaming instead of loading all at once

## References

- **RLDS Format**: https://github.com/google-research/rlds
- **TensorFlow Datasets**: https://www.tensorflow.org/datasets
- **LIBERO**: https://lifelong-robot-learning.github.io/LIBERO/
- **OpenVLA Paper**: https://arxiv.org/abs/2406.09246

## File Locations in Codebase

- Dataset loading: `prismatic/vla/datasets/rlds/dataset.py`
- Data utilities: `prismatic/vla/datasets/rlds/utils/data_utils.py`
- Transforms: `prismatic/vla/datasets/rlds/oxe/transforms.py`
- LIBERO utils: `experiments/robot/libero/libero_utils.py`
