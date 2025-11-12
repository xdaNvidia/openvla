"""
Gradio-based web viewer for RLDS datasets.
Can be deployed to Hugging Face Spaces or run locally.

Usage:
    # Local
    python scripts/rlds/gradio_viewer.py --data_dir ./modified_libero_rlds --dataset_name libero_spatial_no_noops

    # Deploy to HF Spaces: Just upload this file as app.py
"""

import argparse
import io
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image

# Configure Tensorflow with no GPU devices
tf.config.set_visible_devices([], "GPU")

# Global variables to store dataset
DATASET = None
TRAJECTORIES = []
DATA_DIR = None
DATASET_NAME = None


def load_dataset(data_dir, dataset_name, max_trajectories=50):
    """Load dataset and cache trajectories."""
    global DATASET, TRAJECTORIES, DATA_DIR, DATASET_NAME

    DATA_DIR = data_dir
    DATASET_NAME = dataset_name

    builder = tfds.builder(dataset_name, data_dir=data_dir)
    dataset = builder.as_dataset(split=f'train[:{max_trajectories}]')

    # Cache trajectories
    TRAJECTORIES = list(dataset)

    return f"Loaded {len(TRAJECTORIES)} trajectories from {dataset_name}"


def get_trajectory_info(traj_idx):
    """Get basic info about a trajectory."""
    if traj_idx >= len(TRAJECTORIES):
        return "Invalid trajectory index"

    trajectory = TRAJECTORIES[traj_idx]

    # Count steps
    num_steps = sum(1 for _ in trajectory['steps'])

    # Get language instruction if available
    first_step = next(iter(trajectory['steps']))
    instruction = ""
    if 'language_instruction' in first_step:
        instruction = first_step['language_instruction'].numpy().decode('utf-8')

    # Get observation keys
    obs_keys = list(first_step['observation'].keys())

    info = f"""
**Trajectory {traj_idx}**
- Steps: {num_steps}
- Task: {instruction or 'N/A'}
- Observations: {', '.join(obs_keys)}
- Action shape: {first_step['action'].shape}
"""
    return info


def get_frame_images(traj_idx, step_idx):
    """Get all camera views for a specific frame."""
    if traj_idx >= len(TRAJECTORIES):
        return None, None

    trajectory = TRAJECTORIES[traj_idx]

    # Get specific step
    for i, step in enumerate(trajectory['steps']):
        if i == step_idx:
            obs = step['observation']
            images = []

            # Find all image keys
            for key in sorted(obs.keys()):
                if 'image' in key or 'rgb' in key.lower():
                    img = obs[key].numpy()

                    # Decode if needed
                    if img.dtype == object or img.shape == ():
                        img = tf.io.decode_image(img).numpy()

                    images.append((key, img))

            return images

    return None


def visualize_frame(traj_idx, step_idx):
    """Visualize a specific frame with all camera views."""
    images = get_frame_images(traj_idx, step_idx)

    if not images:
        return None, None

    # Create figure with subplots for each camera
    num_cams = len(images)
    fig, axes = plt.subplots(1, num_cams, figsize=(5*num_cams, 5))
    if num_cams == 1:
        axes = [axes]

    for idx, (name, img) in enumerate(images):
        axes[idx].imshow(img)
        axes[idx].set_title(name)
        axes[idx].axis('off')

    plt.tight_layout()

    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    # Return first camera view and combined view
    primary_img = Image.fromarray(images[0][1])
    combined_img = Image.open(buf)

    return primary_img, combined_img


def plot_trajectory_actions(traj_idx):
    """Plot action trajectory over time."""
    if traj_idx >= len(TRAJECTORIES):
        return None

    trajectory = TRAJECTORIES[traj_idx]

    # Collect actions
    actions = []
    for step in trajectory['steps']:
        actions.append(step['action'].numpy())

    actions = np.array(actions)

    # Plot
    num_dims = min(7, actions.shape[1])  # Plot max 7 dimensions
    fig, axes = plt.subplots(num_dims, 1, figsize=(12, 2*num_dims))
    if num_dims == 1:
        axes = [axes]

    dim_names = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Gripper']

    for dim in range(num_dims):
        axes[dim].plot(actions[:, dim], linewidth=2, color='#2E86AB')
        axes[dim].set_ylabel(f'{dim_names[dim] if dim < len(dim_names) else f"Dim {dim}"}')
        axes[dim].grid(True, alpha=0.3)
        axes[dim].axhline(y=0, color='red', linestyle='--', alpha=0.5)

    axes[-1].set_xlabel('Timestep')
    plt.suptitle(f'Trajectory {traj_idx} - Action Profile')
    plt.tight_layout()

    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return Image.open(buf)


def create_interface():
    """Create Gradio interface."""

    with gr.Blocks(title="RLDS Dataset Viewer") as demo:
        gr.Markdown("# 🤖 RLDS Dataset Viewer")
        gr.Markdown("Browse robot learning trajectories interactively")

        with gr.Row():
            with gr.Column(scale=1):
                traj_slider = gr.Slider(
                    minimum=0,
                    maximum=len(TRAJECTORIES)-1 if TRAJECTORIES else 0,
                    step=1,
                    value=0,
                    label="Trajectory Index"
                )

                step_slider = gr.Slider(
                    minimum=0,
                    maximum=100,  # Will be updated dynamically
                    step=1,
                    value=0,
                    label="Step Index"
                )

                info_box = gr.Textbox(
                    label="Trajectory Info",
                    lines=6,
                    interactive=False
                )

                refresh_btn = gr.Button("🔄 Refresh Info")

            with gr.Column(scale=2):
                gr.Markdown("### 📸 Current Frame")
                with gr.Row():
                    primary_view = gr.Image(label="Primary View", type="pil")
                    all_views = gr.Image(label="All Camera Views", type="pil")

                gr.Markdown("### 📊 Action Trajectory")
                action_plot = gr.Image(label="Actions over Time", type="pil")

        # Update info when trajectory changes
        def update_info_and_max_steps(traj_idx):
            info = get_trajectory_info(traj_idx)

            # Get number of steps for slider
            if traj_idx < len(TRAJECTORIES):
                trajectory = TRAJECTORIES[traj_idx]
                num_steps = sum(1 for _ in trajectory['steps'])
                max_steps = num_steps - 1
            else:
                max_steps = 0

            # Generate action plot
            action_img = plot_trajectory_actions(traj_idx)

            return info, gr.Slider(maximum=max_steps, value=0), action_img

        # Update visualizations when step changes
        def update_frame(traj_idx, step_idx):
            return visualize_frame(traj_idx, step_idx)

        # Event handlers
        traj_slider.change(
            update_info_and_max_steps,
            inputs=[traj_slider],
            outputs=[info_box, step_slider, action_plot]
        )

        step_slider.change(
            update_frame,
            inputs=[traj_slider, step_slider],
            outputs=[primary_view, all_views]
        )

        refresh_btn.click(
            update_info_and_max_steps,
            inputs=[traj_slider],
            outputs=[info_box, step_slider, action_plot]
        )

        # Initialize
        demo.load(
            update_info_and_max_steps,
            inputs=[traj_slider],
            outputs=[info_box, step_slider, action_plot]
        )
        demo.load(
            update_frame,
            inputs=[traj_slider, step_slider],
            outputs=[primary_view, all_views]
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description='Web-based RLDS dataset viewer')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to RLDS dataset directory')
    parser.add_argument('--dataset_name', type=str, required=True,
                       help='Name of the dataset')
    parser.add_argument('--max_trajectories', type=int, default=50,
                       help='Maximum trajectories to load (default: 50)')
    parser.add_argument('--port', type=int, default=7860,
                       help='Port to run server on (default: 7860)')
    parser.add_argument('--share', action='store_true',
                       help='Create public share link')

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset: {args.dataset_name}")
    status = load_dataset(args.data_dir, args.dataset_name, args.max_trajectories)
    print(status)

    # Create and launch interface
    demo = create_interface()
    demo.launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0"  # Allow external connections
    )


if __name__ == "__main__":
    main()
