"""
Inspect RLDS dataset structure and print detailed information.

Usage:
    python inspect_rlds.py --data_dir ./modified_libero_rlds --dataset_name libero_spatial_no_noops
"""

import argparse
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# Configure Tensorflow with no GPU devices
tf.config.set_visible_devices([], "GPU")


def print_structure(data, prefix="", max_depth=3, current_depth=0):
    """Recursively print data structure."""
    if current_depth >= max_depth:
        return

    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{prefix}{key}:")
            print_structure(value, prefix + "  ", max_depth, current_depth + 1)
    elif isinstance(data, (tf.Tensor, np.ndarray)):
        shape = data.shape if hasattr(data, 'shape') else 'unknown'
        dtype = data.dtype if hasattr(data, 'dtype') else 'unknown'
        print(f"{prefix}  Shape: {shape}, Dtype: {dtype}")
    else:
        print(f"{prefix}  Type: {type(data)}")


def inspect_dataset(data_dir, dataset_name, num_trajectories=1):
    """
    Inspect RLDS dataset and print detailed information.

    Args:
        data_dir: Path to RLDS dataset directory
        dataset_name: Name of the dataset
        num_trajectories: Number of trajectories to inspect
    """
    print("\n" + "="*80)
    print(f"RLDS Dataset Inspection: {dataset_name}")
    print("="*80)

    # Load dataset builder
    builder = tfds.builder(dataset_name, data_dir=data_dir)

    # Print dataset info
    print("\n" + "-"*80)
    print("DATASET INFO")
    print("-"*80)
    print(builder.info)

    # Print available splits
    print("\n" + "-"*80)
    print("AVAILABLE SPLITS")
    print("-"*80)
    for split_name, split_info in builder.info.splits.items():
        print(f"  {split_name}: {split_info.num_examples} trajectories")

    # Load a few trajectories
    dataset = builder.as_dataset(split='train[:10]')

    print("\n" + "-"*80)
    print("TRAJECTORY DETAILS")
    print("-"*80)

    for traj_idx, trajectory in enumerate(dataset.take(num_trajectories)):
        print(f"\n{'='*80}")
        print(f"Trajectory {traj_idx}")
        print(f"{'='*80}")

        # Count steps
        num_steps = sum(1 for _ in trajectory['steps'])
        print(f"Number of steps: {num_steps}")

        # Examine first step
        first_step = next(iter(trajectory['steps']))

        print("\n--- Step Structure ---")
        print_structure(first_step)

        # Print observation details
        print("\n--- Observation Details ---")
        obs = first_step['observation']
        for key, value in obs.items():
            if isinstance(value, tf.Tensor):
                print(f"\n{key}:")
                print(f"  Shape: {value.shape}")
                print(f"  Dtype: {value.dtype}")

                # Decode images if needed
                if 'image' in key or 'rgb' in key.lower():
                    try:
                        if value.dtype == tf.string:
                            decoded = tf.io.decode_image(value)
                            print(f"  Decoded shape: {decoded.shape}")
                    except:
                        pass
                elif value.dtype in [tf.float32, tf.float64, tf.int32, tf.int64]:
                    print(f"  Sample values: {value.numpy()}")

        # Print action details
        print("\n--- Action Details ---")
        action = first_step['action']
        print(f"Shape: {action.shape}")
        print(f"Dtype: {action.dtype}")

        # Collect all actions in trajectory for statistics
        actions = []
        for step in trajectory['steps']:
            actions.append(step['action'].numpy())
        actions = np.array(actions)

        print(f"\nAction statistics across trajectory:")
        print(f"  Mean: {actions.mean(axis=0)}")
        print(f"  Std:  {actions.std(axis=0)}")
        print(f"  Min:  {actions.min(axis=0)}")
        print(f"  Max:  {actions.max(axis=0)}")

        # Print language instruction if available
        if 'language_instruction' in first_step:
            instruction = first_step['language_instruction'].numpy()
            if isinstance(instruction, bytes):
                instruction = instruction.decode('utf-8')
            print(f"\n--- Language Instruction ---")
            print(f"  {instruction}")

    print("\n" + "="*80)
    print("Inspection complete!")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Inspect RLDS dataset structure')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to RLDS dataset directory')
    parser.add_argument('--dataset_name', type=str, required=True,
                       help='Name of the dataset')
    parser.add_argument('--num_trajectories', type=int, default=3,
                       help='Number of trajectories to inspect in detail')

    args = parser.parse_args()

    inspect_dataset(args.data_dir, args.dataset_name, args.num_trajectories)


if __name__ == "__main__":
    main()
