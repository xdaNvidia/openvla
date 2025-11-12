"""
Visualize RLDS datasets - display images, actions, and trajectories.

Usage:
    python visualize_rlds.py --data_dir ./modified_libero_rlds --dataset_name libero_spatial_no_noops
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path

# Configure Tensorflow with no GPU devices
tf.config.set_visible_devices([], "GPU")


def visualize_trajectory(trajectory, traj_idx=0, save_dir=None):
    """
    Visualize a single trajectory from RLDS dataset.

    Args:
        trajectory: Dictionary containing trajectory data
        traj_idx: Trajectory index for labeling
        save_dir: Optional directory to save visualizations
    """
    print(f"\n{'='*60}")
    print(f"Trajectory {traj_idx}")
    print(f"{'='*60}")

    # Extract data
    steps = trajectory['steps']

    # Get trajectory length
    traj_len = 0
    for step in steps:
        traj_len += 1

    print(f"Trajectory length: {traj_len} steps")

    # Collect all steps
    observations = []
    actions = []
    language_instructions = []

    for step in steps:
        observations.append(step['observation'])
        actions.append(step['action'].numpy())
        if 'language_instruction' in step:
            language_instructions.append(step['language_instruction'].numpy().decode('utf-8'))

    # Print language instruction if available
    if language_instructions and language_instructions[0]:
        print(f"Task: {language_instructions[0]}")

    # Print action statistics
    actions_array = np.array(actions)
    print(f"\nAction shape: {actions_array.shape}")
    print(f"Action mean: {actions_array.mean(axis=0)}")
    print(f"Action std: {actions_array.std(axis=0)}")
    print(f"Action min: {actions_array.min(axis=0)}")
    print(f"Action max: {actions_array.max(axis=0)}")

    # Visualize images if available
    image_keys = [k for k in observations[0].keys() if 'image' in k or 'rgb' in k.lower()]

    if image_keys:
        print(f"\nAvailable image observations: {image_keys}")

        # Create figure for first, middle, and last frames
        num_frames = min(5, traj_len)
        frame_indices = np.linspace(0, traj_len-1, num_frames, dtype=int)

        fig, axes = plt.subplots(len(image_keys), num_frames, figsize=(15, 3*len(image_keys)))
        if len(image_keys) == 1:
            axes = axes[np.newaxis, :]

        for img_idx, img_key in enumerate(image_keys):
            for frame_idx, step_idx in enumerate(frame_indices):
                ax = axes[img_idx, frame_idx]

                # Get image
                img = observations[step_idx][img_key].numpy()

                # Decode if needed
                if img.dtype == object or img.shape == ():
                    img = tf.io.decode_image(img).numpy()

                # Display
                ax.imshow(img)
                ax.set_title(f"{img_key}\nStep {step_idx}")
                ax.axis('off')

        plt.suptitle(f"Trajectory {traj_idx} - Image Observations")
        plt.tight_layout()

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(f"{save_dir}/traj_{traj_idx}_images.png", dpi=150, bbox_inches='tight')
            print(f"Saved image visualization to {save_dir}/traj_{traj_idx}_images.png")
        else:
            plt.show()

    # Plot action trajectories
    if len(actions) > 0:
        fig, axes = plt.subplots(actions_array.shape[1], 1, figsize=(12, 2*actions_array.shape[1]))
        if actions_array.shape[1] == 1:
            axes = [axes]

        for dim in range(actions_array.shape[1]):
            axes[dim].plot(actions_array[:, dim], linewidth=2)
            axes[dim].set_ylabel(f'Action dim {dim}')
            axes[dim].grid(True, alpha=0.3)
            axes[dim].axhline(y=0, color='r', linestyle='--', alpha=0.5)

        axes[-1].set_xlabel('Timestep')
        plt.suptitle(f"Trajectory {traj_idx} - Action Profile")
        plt.tight_layout()

        if save_dir:
            plt.savefig(f"{save_dir}/traj_{traj_idx}_actions.png", dpi=150, bbox_inches='tight')
            print(f"Saved action visualization to {save_dir}/traj_{traj_idx}_actions.png")
        else:
            plt.show()


def visualize_video(trajectory, traj_idx=0, save_path=None):
    """
    Create a video visualization of the trajectory.

    Args:
        trajectory: Dictionary containing trajectory data
        traj_idx: Trajectory index
        save_path: Path to save video (e.g., 'output.mp4')
    """
    import matplotlib.animation as animation

    steps = trajectory['steps']
    observations = []
    actions = []

    for step in steps:
        observations.append(step['observation'])
        actions.append(step['action'].numpy())

    # Get image keys
    image_keys = [k for k in observations[0].keys() if 'image' in k or 'rgb' in k.lower()]

    if not image_keys:
        print("No image observations found for video creation")
        return

    actions_array = np.array(actions)

    # Setup figure
    fig = plt.figure(figsize=(15, 5))

    # Create subplots
    num_img_plots = len(image_keys)
    gs = fig.add_gridspec(2, num_img_plots + 1, height_ratios=[2, 1])

    # Image axes
    img_axes = [fig.add_subplot(gs[0, i]) for i in range(num_img_plots)]

    # Action plot axis
    action_ax = fig.add_subplot(gs[:, -1])

    # Text axis
    text_ax = fig.add_subplot(gs[1, :num_img_plots])
    text_ax.axis('off')

    def update(frame):
        # Clear
        for ax in img_axes:
            ax.clear()
        action_ax.clear()

        # Display images
        for img_idx, img_key in enumerate(image_keys):
            img = observations[frame][img_key].numpy()
            if img.dtype == object or img.shape == ():
                img = tf.io.decode_image(img).numpy()

            img_axes[img_idx].imshow(img)
            img_axes[img_idx].set_title(f"{img_key}")
            img_axes[img_idx].axis('off')

        # Plot actions up to current frame
        for dim in range(min(7, actions_array.shape[1])):  # Show max 7 dimensions
            action_ax.plot(actions_array[:frame+1, dim],
                          label=f'dim {dim}', alpha=0.7, linewidth=2)

        action_ax.axvline(x=frame, color='r', linestyle='--', alpha=0.5)
        action_ax.set_xlabel('Timestep')
        action_ax.set_ylabel('Action')
        action_ax.set_title('Action Trajectory')
        action_ax.legend(loc='upper right', fontsize=8)
        action_ax.grid(True, alpha=0.3)
        action_ax.set_xlim(0, len(actions))

        # Display frame info
        text_ax.clear()
        text_ax.axis('off')
        text_ax.text(0.5, 0.5, f'Frame: {frame}/{len(observations)-1}',
                    ha='center', va='center', fontsize=16, transform=text_ax.transAxes)

        fig.suptitle(f'Trajectory {traj_idx}')

    anim = animation.FuncAnimation(fig, update, frames=len(observations),
                                   interval=100, repeat=True)

    if save_path:
        print(f"Saving video to {save_path}... (this may take a while)")
        writer = animation.FFMpegWriter(fps=10, bitrate=1800)
        anim.save(save_path, writer=writer)
        print(f"Video saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize RLDS dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to RLDS dataset directory')
    parser.add_argument('--dataset_name', type=str, required=True,
                       help='Name of the dataset')
    parser.add_argument('--num_trajectories', type=int, default=3,
                       help='Number of trajectories to visualize')
    parser.add_argument('--split', type=str, default='train[:10]',
                       help='Dataset split (e.g., train[:10], val)')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save visualizations')
    parser.add_argument('--create_video', action='store_true',
                       help='Create video visualization (requires ffmpeg)')

    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset_name}")
    print(f"Data directory: {args.data_dir}")

    # Load dataset
    builder = tfds.builder(args.dataset_name, data_dir=args.data_dir)

    # Print dataset info
    print("\n" + "="*60)
    print("Dataset Information")
    print("="*60)
    print(builder.info)

    # Load data
    dataset = builder.as_dataset(split=args.split)

    # Visualize trajectories
    for traj_idx, trajectory in enumerate(dataset.take(args.num_trajectories)):
        visualize_trajectory(trajectory, traj_idx=traj_idx, save_dir=args.save_dir)

        if args.create_video:
            video_path = f"{args.save_dir}/traj_{traj_idx}.mp4" if args.save_dir else None
            try:
                visualize_video(trajectory, traj_idx=traj_idx, save_path=video_path)
            except Exception as e:
                print(f"Could not create video: {e}")
                print("Make sure ffmpeg is installed: sudo apt-get install ffmpeg")


if __name__ == "__main__":
    main()
