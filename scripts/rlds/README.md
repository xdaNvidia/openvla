# RLDS Visualization Scripts

Tools for inspecting and visualizing RLDS (Robotic Learning Datasets) format datasets.

## Scripts

### 1. gradio_viewer.py ⭐ **NEW - Web Browser Viewer**
Interactive web-based viewer powered by Gradio. Browse trajectories in your browser!

**Features:**
- 🖼️ View all camera angles side-by-side
- 📊 Interactive action trajectory plots
- 🎮 Scrub through timesteps with sliders
- 🌐 Can be deployed to Hugging Face Spaces
- 🔗 Share link option for remote viewing

**Local Usage:**
```bash
# Install dependencies
pip install -r scripts/rlds/requirements_web.txt

# Run locally
python scripts/rlds/gradio_viewer.py \
    --data_dir ./modified_libero_rlds \
    --dataset_name libero_spatial_no_noops \
    --max_trajectories 50

# With public share link (for remote access)
python scripts/rlds/gradio_viewer.py \
    --data_dir ./modified_libero_rlds \
    --dataset_name libero_spatial_no_noops \
    --share
```

Then open **http://localhost:7860** in your browser!

**Deploy to Hugging Face Spaces:**
1. Create new Space at https://huggingface.co/new-space
2. Choose "Gradio" as SDK
3. Upload `gradio_viewer.py` as `app.py`
4. Upload your dataset to the Space
5. Modify paths in the script and push

### 2. inspect_rlds.py
Text-based inspection tool that prints dataset structure and statistics.

**Usage:**
```bash
python scripts/rlds/inspect_rlds.py \
    --data_dir ./modified_libero_rlds \
    --dataset_name libero_spatial_no_noops \
    --num_trajectories 3
```

**Output:**
- Dataset metadata and info
- Available splits (train/val)
- Trajectory structure
- Observation details (shapes, dtypes)
- Action statistics (mean, std, min, max)
- Language instructions

### 3. visualize_rlds.py
Visual inspection tool that creates plots and videos.

**Basic Usage:**
```bash
python scripts/rlds/visualize_rlds.py \
    --data_dir ./modified_libero_rlds \
    --dataset_name libero_spatial_no_noops \
    --num_trajectories 5 \
    --save_dir ./visualizations
```

**Create Videos:**
```bash
python scripts/rlds/visualize_rlds.py \
    --data_dir ./modified_libero_rlds \
    --dataset_name libero_spatial_no_noops \
    --num_trajectories 1 \
    --create_video \
    --save_dir ./videos
```

**Output:**
- `traj_N_images.png`: Grid of key frames
- `traj_N_actions.png`: Action trajectory plots
- `traj_N.mp4`: Video with images and live action plot (if --create_video)

## Arguments

### Common Arguments
- `--data_dir`: Path to RLDS dataset directory (required)
- `--dataset_name`: Name of the dataset (required)
- `--num_trajectories`: Number of trajectories to process
- `--split`: Dataset split to use (default: 'train[:10]')

### Web Viewer Specific
- `--max_trajectories`: Max trajectories to load (default: 50)
- `--port`: Server port (default: 7860)
- `--share`: Create public Gradio share link

### Visualization-Specific
- `--save_dir`: Directory to save outputs
- `--create_video`: Create MP4 videos (requires ffmpeg)

## Dependencies

**Basic (inspect_rlds.py):**
```bash
pip install tensorflow tensorflow-datasets numpy
```

**Visualization (visualize_rlds.py):**
```bash
pip install tensorflow tensorflow-datasets numpy matplotlib
# For video: sudo apt-get install ffmpeg
```

**Web Viewer (gradio_viewer.py):**
```bash
pip install -r scripts/rlds/requirements_web.txt
```

## Quick Start Examples

**1. Browse in browser (easiest):**
```bash
pip install -r scripts/rlds/requirements_web.txt
python scripts/rlds/gradio_viewer.py \
    --data_dir ./modified_libero_rlds \
    --dataset_name libero_spatial_no_noops
# Open http://localhost:7860
```

**2. Quick text inspection:**
```bash
python scripts/rlds/inspect_rlds.py \
    --data_dir ./modified_libero_rlds \
    --dataset_name libero_spatial_no_noops
```

**3. Create visualization files:**
```bash
python scripts/rlds/visualize_rlds.py \
    --data_dir ./modified_libero_rlds \
    --dataset_name libero_spatial_no_noops \
    --num_trajectories 10 \
    --save_dir ./viz_output
```

**4. Share dataset online:**
```bash
python scripts/rlds/gradio_viewer.py \
    --data_dir ./modified_libero_rlds \
    --dataset_name libero_spatial_no_noops \
    --share
# Returns a public URL valid for 72 hours
```

## Comparison: Which Tool to Use?

| Task | Tool | Why |
|------|------|-----|
| **Browse interactively** | `gradio_viewer.py` | Best UX, works in browser |
| **Quick data check** | `inspect_rlds.py` | Fastest, no GUI needed |
| **Generate figures for paper** | `visualize_rlds.py` | High-quality matplotlib plots |
| **Create demo video** | `visualize_rlds.py --create_video` | MP4 output |
| **Share with collaborators** | `gradio_viewer.py --share` | Generates public link |
| **Deploy online** | `gradio_viewer.py` → HF Space | Permanent web viewer |

## Advanced Usage

**Custom split range:**
```bash
python scripts/rlds/gradio_viewer.py \
    --data_dir ./modified_libero_rlds \
    --dataset_name libero_spatial_no_noops \
    --max_trajectories 100  # Load more data
```

**Different port:**
```bash
python scripts/rlds/gradio_viewer.py \
    --data_dir ./modified_libero_rlds \
    --dataset_name libero_spatial_no_noops \
    --port 8080
```

## See Also

See the main [RLDS_GUIDE.md](../../RLDS_GUIDE.md) for comprehensive documentation on:
- RLDS dataset format
- Data structure and components
- Loading data in Python
- Common patterns and examples
