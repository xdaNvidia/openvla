"""
wandb_tutorial.py

A hands-on tutorial demonstrating Weights & Biases (wandb) features.
This script simulates a simple machine learning training loop to show how to use wandb.

Prerequisites:
    pip install wandb numpy matplotlib pillow

Usage:
    # Run basic example
    python learning/wandb_tutorial.py

    # Run without uploading to wandb (for testing)
    WANDB_MODE=offline python learning/wandb_tutorial.py
"""

import random
import time
import numpy as np
import wandb
from PIL import Image


# ============================================================================
# Example 1: Basic Logging
# ============================================================================
def example_1_basic_logging():
    """
    Demonstrates the most basic wandb workflow:
    1. Initialize a run
    2. Log metrics in a loop
    3. Finish the run
    """
    print("\n" + "="*70)
    print("Example 1: Basic Logging")
    print("="*70)

    # Initialize wandb run
    # This creates a new experiment and starts tracking
    run = wandb.init(
        project="wandb-tutorial",           # Project name (groups related runs)
        name="basic-example",                # Name for this specific run
        config={                             # Hyperparameters to track
            "learning_rate": 0.01,
            "epochs": 10,
            "batch_size": 32,
        },
        tags=["tutorial", "basic"],          # Tags for filtering/searching
    )

    print(f"✓ Run initialized: {run.name}")
    print(f"✓ View at: {run.url}")

    # Simulate a training loop
    print("\nSimulating training...")
    for epoch in range(10):
        # Simulate some metrics that improve over time
        loss = 2.0 * np.exp(-0.3 * epoch) + random.uniform(0, 0.1)
        accuracy = (1 - np.exp(-0.4 * epoch)) * 0.95 + random.uniform(0, 0.05)

        # Log metrics to wandb
        # These will appear as charts in the dashboard
        wandb.log({
            "epoch": epoch,
            "train/loss": loss,              # Using "/" creates grouped charts
            "train/accuracy": accuracy,
        })

        print(f"  Epoch {epoch}: loss={loss:.4f}, accuracy={accuracy:.4f}")
        time.sleep(0.1)  # Small delay to see progress

    print("\n✓ Training complete!")

    # Finish the run
    # This marks the run as complete and finalizes all uploads
    wandb.finish()
    print("✓ Run finished and synced to wandb\n")


# ============================================================================
# Example 2: Logging Different Data Types
# ============================================================================
def example_2_data_types():
    """
    Demonstrates logging various data types:
    - Scalars (numbers)
    - Images
    - Tables
    - Histograms
    """
    print("\n" + "="*70)
    print("Example 2: Logging Different Data Types")
    print("="*70)

    run = wandb.init(
        project="wandb-tutorial",
        name="data-types-example",
        tags=["tutorial", "data-types"],
    )

    print(f"✓ Run initialized: {run.url}\n")

    # 1. Scalars (basic numbers)
    print("Logging scalars...")
    wandb.log({
        "learning_rate": 0.001,
        "batch_size": 32,
        "temperature": 0.7,
    })

    # 2. Images
    print("Logging images...")
    # Create a simple synthetic image (simulating robot camera view)
    image_array = np.random.rand(224, 224, 3) * 255
    image = Image.fromarray(image_array.astype('uint8'))

    wandb.log({
        "robot_camera_view": wandb.Image(
            image,
            caption="Simulated robot camera view"
        )
    })

    # 3. Tables (great for predictions vs ground truth)
    print("Logging table...")
    columns = ["step", "predicted_action", "true_action", "error"]
    data = []
    for i in range(10):
        pred = random.uniform(0, 1)
        true = random.uniform(0, 1)
        error = abs(pred - true)
        data.append([i, f"{pred:.3f}", f"{true:.3f}", f"{error:.3f}"])

    table = wandb.Table(columns=columns, data=data)
    wandb.log({"predictions_table": table})

    # 4. Histograms (for weight distributions, action distributions, etc.)
    print("Logging histogram...")
    action_distribution = np.random.normal(0, 1, 1000)  # Simulated action values
    wandb.log({"action_distribution": wandb.Histogram(action_distribution)})

    print("\n✓ All data types logged!")
    wandb.finish()
    print("✓ Run finished\n")


# ============================================================================
# Example 3: Comparing Multiple Runs
# ============================================================================
def example_3_comparing_runs():
    """
    Demonstrates how to run multiple experiments with different configs.
    This is useful for hyperparameter tuning.
    """
    print("\n" + "="*70)
    print("Example 3: Comparing Multiple Runs")
    print("="*70)
    print("Running 3 experiments with different learning rates...\n")

    learning_rates = [0.001, 0.01, 0.1]

    for lr in learning_rates:
        print(f"Running experiment with lr={lr}")

        # Each run gets its own wandb.init()
        run = wandb.init(
            project="wandb-tutorial",
            name=f"lr-comparison-{lr}",
            config={
                "learning_rate": lr,
                "epochs": 20,
                "model": "simple-nn",
            },
            tags=["tutorial", "comparison", "lr-sweep"],
            group="lr-comparison",  # Group related experiments together
            reinit=True,  # Allow multiple wandb.init() in same script
        )

        # Simulate training with different learning rates
        for epoch in range(20):
            # Loss behaves differently based on learning rate
            if lr == 0.001:
                # Too small - slow convergence
                loss = 2.0 * np.exp(-0.1 * epoch) + random.uniform(0, 0.1)
            elif lr == 0.01:
                # Just right - good convergence
                loss = 2.0 * np.exp(-0.4 * epoch) + random.uniform(0, 0.05)
            else:  # lr == 0.1
                # Too large - unstable
                loss = 2.0 * np.exp(-0.2 * epoch) + random.uniform(0, 0.3)

            wandb.log({
                "epoch": epoch,
                "loss": loss,
                "learning_rate": lr,
            })

        wandb.finish()
        print(f"  ✓ Completed lr={lr}\n")

    print("="*70)
    print("✓ All comparison runs complete!")
    print("Go to your wandb dashboard and select all 3 runs to compare them!")
    print("="*70 + "\n")


# ============================================================================
# Example 4: Simulating Robot Training (Realistic Example)
# ============================================================================
def example_4_robot_training():
    """
    A more realistic example simulating robot VLA training.
    Shows metrics similar to what you'd see in OpenVLA fine-tuning.
    """
    print("\n" + "="*70)
    print("Example 4: Simulating Robot VLA Training")
    print("="*70)

    # Configuration similar to OpenVLA
    config = {
        "model": "openvla-7b",
        "dataset": "droid_wipe",
        "learning_rate": 5e-4,
        "batch_size": 16,
        "lora_rank": 32,
        "lora_dropout": 0.0,
        "max_steps": 100,
        "grad_accumulation_steps": 1,
    }

    run = wandb.init(
        project="wandb-tutorial",
        name="robot-training-simulation",
        config=config,
        tags=["tutorial", "robot", "vla"],
    )

    print(f"✓ Run initialized: {run.url}")
    print(f"\nConfig:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("\nSimulating training...\n")

    # Simulate training loop
    for step in range(config["max_steps"]):
        # Simulate metrics that improve over time (similar to real training)
        train_loss = 5.0 * np.exp(-0.02 * step) + random.uniform(0, 0.2)
        action_accuracy = (1 - np.exp(-0.03 * step)) * 0.95 + random.uniform(0, 0.05)
        l1_loss = 0.5 * np.exp(-0.015 * step) + random.uniform(0, 0.05)

        # Additional metrics
        grad_norm = 1.0 + random.uniform(-0.3, 0.3)
        gpu_memory = 35000 + random.uniform(-1000, 1000)  # MB

        # Log metrics (similar to finetune.py)
        wandb.log({
            "step": step,
            "train_loss": train_loss,
            "action_accuracy": action_accuracy,
            "l1_loss": l1_loss,
            "grad_norm": grad_norm,
            "gpu_memory_mb": gpu_memory,
        })

        # Print progress every 20 steps
        if step % 20 == 0:
            print(f"  Step {step:3d}: "
                  f"loss={train_loss:.4f}, "
                  f"acc={action_accuracy:.4f}, "
                  f"l1={l1_loss:.4f}")

        time.sleep(0.01)  # Small delay

    print("\n✓ Training simulation complete!")

    # Log final summary
    wandb.summary.update({
        "final_accuracy": action_accuracy,
        "final_loss": train_loss,
        "total_steps": config["max_steps"],
    })

    wandb.finish()
    print("✓ Run finished\n")


# ============================================================================
# Example 5: Using wandb.config for Hyperparameter Access
# ============================================================================
def example_5_config_usage():
    """
    Demonstrates how to use wandb.config throughout your code.
    This makes your code cleaner and ensures all hyperparameters are tracked.
    """
    print("\n" + "="*70)
    print("Example 5: Using wandb.config")
    print("="*70)

    # Initialize with config
    wandb.init(
        project="wandb-tutorial",
        name="config-usage-example",
        config={
            "learning_rate": 0.001,
            "epochs": 5,
            "batch_size": 32,
            "model_name": "resnet50",
            "optimizer": "adam",
        }
    )

    # Access config values throughout your code
    print("\nAccessing configuration:")
    print(f"  Learning Rate: {wandb.config.learning_rate}")
    print(f"  Epochs: {wandb.config.epochs}")
    print(f"  Batch Size: {wandb.config.batch_size}")

    # Use config in training loop
    print("\nTraining with config values...")
    for epoch in range(wandb.config.epochs):
        # Use config values
        lr = wandb.config.learning_rate
        loss = 2.0 * np.exp(-lr * epoch * 10) + random.uniform(0, 0.1)

        wandb.log({
            "epoch": epoch,
            "loss": loss,
        })
        print(f"  Epoch {epoch}: loss={loss:.4f}")

    # Update config during run (if needed)
    wandb.config.update({"final_learning_rate": wandb.config.learning_rate * 0.1})

    print("\n✓ Config usage demonstrated!")
    wandb.finish()
    print("✓ Run finished\n")


# ============================================================================
# Example 6: Offline Mode (for testing)
# ============================================================================
def example_6_offline_mode():
    """
    Demonstrates offline mode - useful for debugging without uploading to wandb.
    """
    print("\n" + "="*70)
    print("Example 6: Offline Mode")
    print("="*70)
    print("\nTip: Set WANDB_MODE=offline to test without uploading")
    print("Example: WANDB_MODE=offline python learning/wandb_tutorial.py\n")

    # Initialize in offline mode
    wandb.init(
        project="wandb-tutorial",
        name="offline-example",
        mode="offline",  # This run won't upload to wandb servers
    )

    print("Running in OFFLINE mode...")
    print("Logs are saved locally in ./wandb/ directory\n")

    # Normal logging works the same
    for i in range(10):
        wandb.log({"metric": random.uniform(0, 1)})

    print("✓ Run completed offline")
    print("✓ To sync later: wandb sync wandb/offline-run-xxx")
    wandb.finish()
    print()


# ============================================================================
# Example 7: Watch Model Gradients (Advanced)
# ============================================================================
def example_7_watch_model():
    """
    Demonstrates wandb.watch() for tracking model gradients and parameters.
    Note: Requires PyTorch
    """
    print("\n" + "="*70)
    print("Example 7: Watching Model (Requires PyTorch)")
    print("="*70)

    try:
        import torch
        import torch.nn as nn

        # Simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 50)
                self.fc2 = nn.Linear(50, 1)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                return self.fc2(x)

        wandb.init(
            project="wandb-tutorial",
            name="model-watching-example",
            config={"model": "simple-nn"}
        )

        model = SimpleModel()

        # Watch the model - logs gradients and parameters
        wandb.watch(model, log="all", log_freq=10)
        print("✓ Model is being watched by wandb")
        print("  Gradients and parameters will be logged\n")

        # Simulate training
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        print("Training with gradient tracking...")
        for step in range(50):
            # Forward pass
            x = torch.randn(32, 10)
            target = torch.randn(32, 1)
            output = model(x)
            loss = criterion(output, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log loss
            wandb.log({"loss": loss.item()})

            if step % 10 == 0:
                print(f"  Step {step}: loss={loss.item():.4f}")

        print("\n✓ Training complete with gradient tracking!")
        print("✓ Check wandb dashboard for gradient histograms")
        wandb.finish()

    except ImportError:
        print("\n⚠️  PyTorch not installed - skipping this example")
        print("Install with: pip install torch")

    print()


# ============================================================================
# Example 8: Logging Videos
# ============================================================================
def example_8_video_logging():
    """
    Demonstrates how to log videos to wandb.
    Very useful for VLA/robotics to visualize:
    - Policy rollouts
    - Robot trajectories
    - Camera feeds
    - Attention visualizations
    """
    print("\n" + "="*70)
    print("Example 8: Logging Videos")
    print("="*70)

    wandb.init(
        project="wandb-tutorial",
        name="video-logging-example",
        tags=["tutorial", "video", "robotics"],
    )

    print(f"\nDemonstrating video logging for robotics/VLA applications...\n")

    # 1. Basic video from numpy array
    print("1. Creating simple animated video...")
    # Shape: (num_frames, channels, height, width)
    # Create a simple animation: moving square
    num_frames = 60
    height, width = 128, 128
    video_frames = []

    for i in range(num_frames):
        # Create blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw moving square
        x_pos = int((i / num_frames) * (width - 20))
        frame[40:60, x_pos:x_pos+20] = [255, 100, 100]  # Red square
        video_frames.append(frame)

    # Convert to (T, C, H, W) format
    video_array = np.stack(video_frames).transpose(0, 3, 1, 2)

    wandb.log({
        "animation/moving_square": wandb.Video(video_array, fps=30, format="mp4")
    })
    print("   ✓ Logged moving square animation")

    # 2. Simulate robot camera feed
    print("\n2. Simulating robot wrist camera feed...")
    robot_frames = []
    for i in range(40):
        # Simulate a robot camera view with changing scene
        frame = np.random.rand(224, 224, 3) * 100  # Dark background

        # Add "robot gripper" (simulated with shapes)
        # Gripper opening/closing animation
        gripper_open = int(10 + 15 * abs(np.sin(i * 0.2)))
        frame[180:220, 100:110] = [200, 200, 200]  # Left finger
        frame[180:220, 110+gripper_open:120+gripper_open] = [200, 200, 200]  # Right finger

        # Add some "objects" in scene
        object_y = int(100 - i * 2)  # Object moving towards gripper
        if object_y > 0:
            frame[max(0, object_y):min(224, object_y+30), 100:130] = [100, 200, 100]  # Green object

        robot_frames.append(frame.astype(np.uint8))

    robot_video = np.stack(robot_frames).transpose(0, 3, 1, 2)
    wandb.log({
        "robot/wrist_camera": wandb.Video(robot_video, fps=20, format="mp4",
                                          caption="Simulated robot gripper picking up object")
    })
    print("   ✓ Logged robot wrist camera feed")

    # 3. Multiple synchronized videos (common in robotics - multiple camera angles)
    print("\n3. Logging multiple synchronized camera views...")

    # Third-person camera
    third_person_frames = []
    for i in range(40):
        frame = np.random.rand(224, 224, 3) * 80
        # Draw "robot arm" from side view
        arm_extend = int(50 + i * 2)
        frame[100:120, 50:50+arm_extend] = [150, 150, 200]  # Robot arm
        third_person_frames.append(frame.astype(np.uint8))

    third_person_video = np.stack(third_person_frames).transpose(0, 3, 1, 2)

    wandb.log({
        "robot/third_person_camera": wandb.Video(third_person_video, fps=20, format="mp4",
                                                  caption="Third-person view of robot"),
    })
    print("   ✓ Logged third-person camera view")

    # 4. Visualization: Policy rollout with action overlays
    print("\n4. Creating policy rollout visualization...")
    rollout_frames = []
    for i in range(50):
        # Base observation
        frame = np.random.rand(256, 256, 3) * 120

        # Add trajectory visualization
        for j in range(i):
            # Draw past trajectory
            x = int(128 + 60 * np.cos(j * 0.1))
            y = int(128 + 60 * np.sin(j * 0.1))
            frame[max(0, y-2):min(256, y+2), max(0, x-2):min(256, x+2)] = [255, 200, 0]  # Yellow trail

        # Current position
        x_curr = int(128 + 60 * np.cos(i * 0.1))
        y_curr = int(128 + 60 * np.sin(i * 0.1))
        frame[max(0, y_curr-5):min(256, y_curr+5), max(0, x_curr-5):min(256, x_curr+5)] = [255, 0, 0]  # Red dot

        # Predicted next action (arrow)
        x_next = int(128 + 60 * np.cos((i+1) * 0.1))
        y_next = int(128 + 60 * np.sin((i+1) * 0.1))
        # Simple line for arrow
        steps = 10
        for s in range(steps):
            x_arrow = int(x_curr + (x_next - x_curr) * s / steps)
            y_arrow = int(y_curr + (y_next - y_curr) * s / steps)
            frame[max(0, y_arrow-1):min(256, y_arrow+1), max(0, x_arrow-1):min(256, x_arrow+1)] = [0, 255, 0]

        rollout_frames.append(frame.astype(np.uint8))

    rollout_video = np.stack(rollout_frames).transpose(0, 3, 1, 2)
    wandb.log({
        "eval/policy_rollout": wandb.Video(rollout_video, fps=25, format="mp4",
                                            caption="Policy rollout with predicted actions")
    })
    print("   ✓ Logged policy rollout visualization")

    # 5. GIF format (smaller file size, good for quick previews)
    print("\n5. Creating GIF animation...")
    gif_frames = []
    for i in range(20):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        # Pulsing circle
        radius = int(10 + 15 * abs(np.sin(i * 0.3)))
        y, x = np.ogrid[:64, :64]
        mask = (x - 32)**2 + (y - 32)**2 <= radius**2
        frame[mask] = [100, 150, 255]
        gif_frames.append(frame)

    gif_array = np.stack(gif_frames).transpose(0, 3, 1, 2)
    wandb.log({
        "animation/pulsing_circle": wandb.Video(gif_array, fps=10, format="gif")
    })
    print("   ✓ Logged GIF animation")

    print("\n" + "="*70)
    print("✓ All videos logged successfully!")
    print("="*70)
    print("\nTips for using videos in wandb:")
    print("  • Videos appear in the 'Media' panel in the dashboard")
    print("  • You can compare videos across different runs")
    print("  • Supported formats: MP4, AVI, MOV, GIF")
    print("  • Shape formats: (T,C,H,W) or (T,H,W,C)")
    print("  • Use fps parameter to control playback speed")
    print("  • Add captions to provide context")
    print("\nFor VLA/Robotics:")
    print("  • Log rollouts to visualize policy behavior")
    print("  • Log attention maps to understand model focus")
    print("  • Log multiple camera angles for complete context")
    print("  • Use lower fps (10-20) to reduce file size")
    print("="*70)

    wandb.finish()
    print("\n✓ Run finished\n")


# ============================================================================
# Main Function
# ============================================================================
def main():
    """
    Run all examples sequentially.
    """
    print("\n" + "="*70)
    print("🚀 Weights & Biases (wandb) Tutorial")
    print("="*70)
    print("\nThis script will walk through various wandb features.")
    print("Each example will create a run on your wandb dashboard.\n")
    print("Press Ctrl+C at any time to stop.\n")

    # Check if wandb is logged in
    try:
        wandb.login()
    except Exception as e:
        print(f"⚠️  Warning: {e}")
        print("Run 'wandb login' first, or set WANDB_MODE=offline")
        return

    try:
        # Run all examples
        example_1_basic_logging()
        example_2_data_types()
        example_3_comparing_runs()
        example_4_robot_training()
        example_5_config_usage()
        example_6_offline_mode()
        example_7_watch_model()
        example_8_video_logging()

        print("="*70)
        print("🎉 Tutorial Complete!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Go to https://wandb.ai and explore your runs")
        print("  2. Try selecting multiple runs to compare them")
        print("  3. Create a report with your favorite charts")
        print("  4. Modify this script to experiment with your own data")
        print("\n✓ Happy experimenting with wandb!\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  Tutorial interrupted by user")
        wandb.finish()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        wandb.finish()


if __name__ == "__main__":
    main()
