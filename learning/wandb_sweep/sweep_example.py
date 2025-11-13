"""
sweep_example.py

Demonstrates how to create and run wandb sweeps programmatically.
Shows a complete example with a simple training function.

Usage:
    # Create sweep and run locally
    python learning/sweep_example.py

    # Or create sweep and run agents separately
    python learning/sweep_example.py --create-only
    # Then in other terminals:
    wandb agent your-username/wandb-tutorial/sweep-id
"""

import argparse
import random
import numpy as np
import wandb


# ============================================================================
# Training Function
# ============================================================================
def train():
    """
    Simple training function that wandb sweep will call with different configs.

    The sweep will automatically inject hyperparameters via wandb.config.
    """
    # Initialize wandb - sweep injects config automatically
    run = wandb.init()

    # Get hyperparameters from sweep
    lr = wandb.config.learning_rate
    batch_size = wandb.config.batch_size
    momentum = wandb.config.get('momentum', 0.9)  # Default if not in sweep
    optimizer_type = wandb.config.get('optimizer', 'adam')

    print(f"\nTraining with config:")
    print(f"  Learning Rate: {lr}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Momentum: {momentum}")
    print(f"  Optimizer: {optimizer_type}")

    # Simulate training loop
    best_accuracy = 0.0

    for epoch in range(50):
        # Simulate training with these hyperparameters
        # In reality, hyperparameters affect convergence differently

        # Optimal lr is around 0.01, optimal batch_size is 32
        lr_score = 1.0 - abs(np.log10(lr) - np.log10(0.01))  # Closer to 0.01 is better
        batch_score = 1.0 - abs(batch_size - 32) / 32  # Closer to 32 is better

        # Combine scores (simplified model of how hyperparams affect training)
        base_accuracy = 0.5 + 0.3 * lr_score + 0.2 * batch_score

        # Add learning dynamics (improve over time)
        progress = 1 - np.exp(-0.1 * epoch)

        # Final accuracy with some noise
        accuracy = base_accuracy * progress + random.uniform(-0.02, 0.02)
        accuracy = max(0.0, min(1.0, accuracy))  # Clip to [0, 1]

        # Simulate loss
        loss = 2.0 * (1 - accuracy) + random.uniform(0, 0.1)

        # Log metrics
        wandb.log({
            'epoch': epoch,
            'accuracy': accuracy,
            'loss': loss,
            'learning_rate': lr,
        })

        best_accuracy = max(best_accuracy, accuracy)

        # Simulate early stopping for bad configs
        if epoch == 10 and accuracy < 0.3:
            print(f"  Early stopping - accuracy too low at epoch 10")
            break

    # Log final metrics
    wandb.summary['best_accuracy'] = best_accuracy
    print(f"\n  Final best accuracy: {best_accuracy:.4f}\n")


# ============================================================================
# Sweep Configuration
# ============================================================================
def create_sweep_config():
    """Create sweep configuration."""

    # Method 1: Simple config (good for getting started)
    simple_config = {
        'method': 'random',  # or 'grid', 'bayes'
        'metric': {
            'name': 'accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'min': 0.001,
                'max': 0.1,
                'distribution': 'log_uniform'
            },
            'batch_size': {
                'values': [16, 32, 64]
            }
        }
    }

    # Method 2: Comprehensive config (for serious tuning)
    comprehensive_config = {
        'method': 'bayes',
        'metric': {
            'name': 'accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'min': 0.001,
                'max': 0.1,
                'distribution': 'log_uniform'
            },
            'batch_size': {
                'values': [8, 16, 32, 64]
            },
            'optimizer': {
                'values': ['adam', 'sgd', 'rmsprop']
            },
            'momentum': {
                'min': 0.8,
                'max': 0.99,
                'distribution': 'uniform'
            }
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 3,
            'max_iter': 27,
            's': 2
        }
    }

    # Choose which config to use
    return simple_config


# ============================================================================
# Main Functions
# ============================================================================
def create_and_run_sweep():
    """Create sweep and run agent locally."""

    print("="*70)
    print("Creating wandb sweep...")
    print("="*70)

    # Create sweep
    sweep_config = create_sweep_config()
    sweep_id = wandb.sweep(
        sweep_config,
        project="wandb-tutorial",
        entity="xda_nvidia"  # Change to your username
    )

    print(f"\n✓ Sweep created!")
    print(f"  Sweep ID: {sweep_id}")
    print(f"  View at: https://wandb.ai/xda_nvidia/wandb-tutorial/sweeps/{sweep_id}")
    print(f"\n  Starting agent to run experiments...")
    print("="*70 + "\n")

    # Run sweep agent (runs 5 experiments)
    wandb.agent(
        sweep_id,
        function=train,
        count=5,  # Run 5 experiments
        project="wandb-tutorial",
        entity="xda_nvidia"
    )

    print("\n" + "="*70)
    print("✓ Sweep complete!")
    print("="*70)
    print(f"\nView results at:")
    print(f"https://wandb.ai/xda_nvidia/wandb-tutorial/sweeps/{sweep_id}")
    print("\nThe sweep dashboard shows:")
    print("  • Parallel coordinates plot (hyperparameter relationships)")
    print("  • Parameter importance (which params matter most)")
    print("  • Best run (highlighted automatically)")
    print()


def create_sweep_only():
    """Just create the sweep, don't run agents."""

    print("Creating sweep (no agents)...")

    sweep_config = create_sweep_config()
    sweep_id = wandb.sweep(
        sweep_config,
        project="wandb-tutorial",
        entity="xda_nvidia"
    )

    print(f"\n✓ Sweep created: {sweep_id}")
    print(f"\nTo run agents:")
    print(f"  wandb agent xda_nvidia/wandb-tutorial/{sweep_id}")
    print(f"\nOr run multiple agents in parallel:")
    print(f"  # Terminal 1:")
    print(f"  wandb agent xda_nvidia/wandb-tutorial/{sweep_id}")
    print(f"  # Terminal 2:")
    print(f"  wandb agent xda_nvidia/wandb-tutorial/{sweep_id}")
    print()


# ============================================================================
# Alternative: Simple Sweep Example (No Function)
# ============================================================================
def simple_inline_sweep():
    """
    Simplest possible sweep - define everything inline.
    Good for quick experiments.
    """

    # Define sweep inline
    sweep_id = wandb.sweep({
        'method': 'random',
        'parameters': {
            'learning_rate': {'min': 0.001, 'max': 0.1},
            'batch_size': {'values': [16, 32, 64]}
        }
    }, project='wandb-tutorial')

    # Define training inline
    def quick_train():
        run = wandb.init()
        lr = wandb.config.learning_rate
        bs = wandb.config.batch_size

        for i in range(10):
            loss = np.exp(-i * lr) + random.random() * 0.1
            wandb.log({'loss': loss})

    # Run it
    wandb.agent(sweep_id, function=quick_train, count=3)


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Wandb sweep example')
    parser.add_argument(
        '--create-only',
        action='store_true',
        help='Only create sweep, do not run agents'
    )
    parser.add_argument(
        '--simple',
        action='store_true',
        help='Run simple inline sweep example'
    )

    args = parser.parse_args()

    if args.simple:
        simple_inline_sweep()
    elif args.create_only:
        create_sweep_only()
    else:
        create_and_run_sweep()
