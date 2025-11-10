#!/usr/bin/env python3
"""
Quick setup verification script for OpenVLA LIBERO evaluation.
Run this to check if your environment is properly configured.
"""

import sys
import importlib

def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    if package_name is None:
        package_name = module_name
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {package_name} (version: {version})")
        return True
    except ImportError as e:
        print(f"✗ {package_name} - NOT INSTALLED")
        print(f"  Error: {e}")
        return False

def main():
    print("=" * 60)
    print("OpenVLA LIBERO Setup Verification")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # Check Python version
    print("Python version:", sys.version)
    if sys.version_info < (3, 10):
        print("⚠️  Warning: Python 3.10+ recommended")
    print()
    
    # Core dependencies
    print("Checking core dependencies...")
    print("-" * 60)
    all_ok &= check_import("torch")
    all_ok &= check_import("torchvision")
    all_ok &= check_import("transformers")
    all_ok &= check_import("timm")
    print()
    
    # OpenVLA specific
    print("Checking OpenVLA dependencies...")
    print("-" * 60)
    all_ok &= check_import("prismatic")
    all_ok &= check_import("draccus")
    print()
    
    # LIBERO dependencies
    print("Checking LIBERO dependencies...")
    print("-" * 60)
    libero_ok = check_import("libero.libero", "libero")
    all_ok &= libero_ok
    all_ok &= check_import("robosuite")
    all_ok &= check_import("imageio")
    print()
    
    # Check CUDA availability
    print("Checking GPU/CUDA availability...")
    print("-" * 60)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print("⚠️  CUDA not available - will use CPU (slower)")
    except:
        print("⚠️  Could not check CUDA status")
    print()
    
    # Summary
    print("=" * 60)
    if all_ok:
        print("✓ All dependencies are installed!")
        print()
        print("You can now run evaluation with:")
        print("  python3 experiments/robot/libero/run_libero_eval.py \\")
        print("    --model_family openvla \\")
        print("    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \\")
        print("    --task_suite_name libero_spatial \\")
        print("    --center_crop True \\")
        print("    --num_trials_per_task 2")
    else:
        print("✗ Some dependencies are missing.")
        print()
        print("Please install missing packages:")
        print("  1. Install OpenVLA: pip install -e .")
        print("  2. Install LIBERO: git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git")
        print("     Then: cd LIBERO && pip install -e .")
        print("  3. Install LIBERO requirements: pip install -r experiments/robot/libero/libero_requirements.txt")
        print()
        print("See GETTING_STARTED.md for detailed instructions.")
    print("=" * 60)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())

