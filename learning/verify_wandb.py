"""
verify_wandb.py

Quick script to verify wandb is set up correctly.
"""

import os
import sys

def verify_wandb_setup():
    print("="*70)
    print("Verifying Weights & Biases Setup")
    print("="*70)

    # Check 1: API Key in environment
    print("\n1. Checking for WANDB_API_KEY in environment...")
    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        # Mask the key for security
        masked_key = api_key[:8] + "*" * (len(api_key) - 12) + api_key[-4:]
        print(f"   ✓ WANDB_API_KEY found: {masked_key}")
    else:
        print("   ✗ WANDB_API_KEY not found in environment")
        print("   Run: export WANDB_API_KEY='your-key-here'")
        print("   Or open a new terminal (if you just added it to ~/.bashrc)")
        return False

    # Check 2: wandb package installed
    print("\n2. Checking if wandb is installed...")
    try:
        import wandb
        print(f"   ✓ wandb version {wandb.__version__} installed")
    except ImportError:
        print("   ✗ wandb not installed")
        print("   Run: pip install wandb")
        return False

    # Check 3: Login test
    print("\n3. Testing wandb login...")
    try:
        wandb.login(key=api_key, relogin=True)
        print("   ✓ Successfully logged in to wandb")

        # Get user info
        try:
            viewer = wandb.Api().viewer
            print(f"   ✓ Logged in as: {viewer['username']}")
            if viewer.get('entity'):
                print(f"   ✓ Default entity: {viewer['entity']}")
        except Exception as e:
            print(f"   ⚠ Could not fetch user info: {e}")

    except Exception as e:
        print(f"   ✗ Login failed: {e}")
        return False

    # Check 4: Quick logging test
    print("\n4. Testing basic logging...")
    try:
        run = wandb.init(
            project="wandb-setup-test",
            name="verification-test",
            mode="offline",  # Don't actually upload
        )
        wandb.log({"test_metric": 42})
        wandb.finish()
        print("   ✓ Logging test successful")
    except Exception as e:
        print(f"   ✗ Logging test failed: {e}")
        return False

    # All checks passed
    print("\n" + "="*70)
    print("✓ All checks passed! Your wandb setup is ready to use.")
    print("="*70)
    print("\nNext steps:")
    print("  • Run the tutorial: python learning/wandb_tutorial.py")
    print("  • Use in training: --use_wandb True --wandb_entity 'your-username'")
    print("\n")
    return True

if __name__ == "__main__":
    success = verify_wandb_setup()
    sys.exit(0 if success else 1)
