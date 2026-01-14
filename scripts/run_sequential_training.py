"""
Run multiple training experiments sequentially with a cooldown period between runs.

This script takes multiple config files as input and runs training for each one,
with a configurable gap between experiments to allow GPU memory cleanup and prevent crashes.

Usage:
    python scripts/run_sequential_training.py config1.yaml config2.yaml config3.yaml
    python scripts/run_sequential_training.py configs/danish/train/*.yaml --gap 10
    python scripts/run_sequential_training.py -c configs/danish/train/whisper-s_baseline.yaml configs/danish/train/whisper-s_ablation_6L.yaml
"""
import os
import sys
import argparse
import subprocess
import time
import gc
from pathlib import Path
from datetime import datetime, timedelta
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def clear_gpu_memory():
    """Attempt to clear GPU memory between runs."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            print("  ✓ GPU cache cleared")
    except ImportError:
        print("  ⚠ PyTorch not available, skipping GPU cleanup")
    except Exception as e:
        print(f"  ⚠ GPU cleanup failed: {e}")
    
    # Force garbage collection
    gc.collect()
    print("  ✓ Garbage collection completed")


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def countdown_timer(seconds: int, message: str = "Waiting"):
    """Display a countdown timer."""
    print(f"\n{message}: ", end="", flush=True)
    for remaining in range(seconds, 0, -1):
        mins, secs = divmod(remaining, 60)
        print(f"\r{message}: {mins:02d}:{secs:02d} remaining", end="", flush=True)
        time.sleep(1)
    print(f"\r{message}: Done!                    ")


def run_training(config_path: str, run_number: int, total_runs: int) -> tuple:
    """
    Run training with the specified config file.
    
    Returns:
        tuple: (success: bool, duration_seconds: float)
    """
    config_name = Path(config_path).stem
    
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT [{run_number}/{total_runs}]: {config_name}")
    print(f"{'='*70}")
    print(f"  Config: {config_path}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'─'*70}\n")
    
    start_time = time.time()
    
    try:
        # Run training
        result = subprocess.run(
            [sys.executable, "train.py", "--config", config_path],
            check=True,
            cwd=Path(__file__).parent.parent  # Run from project root
        )
        success = result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n  ✗ Training failed with return code {e.returncode}")
        success = False
    except KeyboardInterrupt:
        print("\n\n  ⚠ Training interrupted by user")
        raise
    except Exception as e:
        print(f"\n  ✗ Training failed with error: {e}")
        success = False
    
    duration = time.time() - start_time
    
    status = "✓ COMPLETED" if success else "✗ FAILED"
    print(f"\n{'─'*70}")
    print(f"  {status} in {format_time(duration)}")
    print(f"  Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return success, duration


def validate_configs(config_paths: List[str]) -> List[str]:
    """Validate that all config files exist and return absolute paths."""
    valid_configs = []
    
    for config in config_paths:
        config_path = Path(config)
        
        # If relative path, try from project root
        if not config_path.is_absolute():
            config_path = Path(__file__).parent.parent / config
        
        if not config_path.exists():
            print(f"  ⚠ Warning: Config not found, skipping: {config}")
            continue
        
        if not config_path.suffix in ['.yaml', '.yml']:
            print(f"  ⚠ Warning: Not a YAML file, skipping: {config}")
            continue
            
        valid_configs.append(str(config_path))
    
    return valid_configs


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple training experiments sequentially with cooldown periods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run specific configs
  python scripts/run_sequential_training.py configs/danish/train/whisper-s_baseline.yaml configs/danish/train/whisper-s_ablation_6L.yaml
  
  # Run all configs in a folder (bash/powershell glob)
  python scripts/run_sequential_training.py configs/danish/train/*.yaml
  
  # Custom gap between experiments (10 minutes)
  python scripts/run_sequential_training.py -c config1.yaml config2.yaml --gap 10
  
  # Dry run to see what would be executed
  python scripts/run_sequential_training.py configs/danish/train/*.yaml --dry-run
        """
    )
    parser.add_argument(
        "configs",
        nargs="*",
        help="Config files to run (can also use -c/--config)"
    )
    parser.add_argument(
        "-c", "--config",
        nargs="+",
        default=[],
        help="Config files to run"
    )
    parser.add_argument(
        "-g", "--gap",
        type=int,
        default=1,
        help="Gap between experiments in minutes (default: 1)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without executing"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with next experiment even if one fails"
    )
    parser.add_argument(
        "--no-countdown",
        action="store_true",
        help="Disable countdown timer display (just sleep)"
    )
    
    args = parser.parse_args()
    
    # Combine config sources
    all_configs = args.configs + args.config
    
    if not all_configs:
        parser.print_help()
        print("\n  Error: No config files provided")
        sys.exit(1)
    
    # Validate configs
    print("\n📋 Validating config files...")
    valid_configs = validate_configs(all_configs)
    
    if not valid_configs:
        print("  ✗ No valid config files found")
        sys.exit(1)
    
    # Print experiment plan
    gap_seconds = args.gap * 60
    total_runs = len(valid_configs)
    
    print(f"\n{'='*70}")
    print(f"  SEQUENTIAL TRAINING PLAN")
    print(f"{'='*70}")
    print(f"  Total experiments: {total_runs}")
    print(f"  Gap between runs:  {args.gap} minutes")
    print(f"  Continue on error: {'Yes' if args.continue_on_error else 'No'}")
    print(f"\n  Experiments in order:")
    for i, config in enumerate(valid_configs, 1):
        print(f"    {i}. {Path(config).stem}")
    print(f"{'='*70}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - Commands that would be executed:\n")
        for i, config in enumerate(valid_configs, 1):
            print(f"  [{i}/{total_runs}] python train.py --config {config}")
            if i < total_runs:
                print(f"          └─ Wait {args.gap} minutes for GPU cooldown")
        print("\n  No actual training was performed.")
        return
    
    # Confirm before starting
    print(f"\n⏳ Starting in 5 seconds... (Ctrl+C to cancel)")
    time.sleep(5)
    
    # Run experiments
    results = []
    total_start_time = time.time()
    
    for i, config in enumerate(valid_configs, 1):
        try:
            success, duration = run_training(config, i, total_runs)
            results.append({
                "config": Path(config).stem,
                "success": success,
                "duration": duration
            })
            
            if not success and not args.continue_on_error:
                print(f"\n  ⚠ Stopping due to failed experiment (use --continue-on-error to override)")
                break
            
            # Cooldown between experiments (skip after last one)
            if i < total_runs:
                print(f"\n🧊 Cooling down GPU before next experiment...")
                clear_gpu_memory()
                
                if args.no_countdown:
                    print(f"  Sleeping for {args.gap} minutes...")
                    time.sleep(gap_seconds)
                else:
                    countdown_timer(gap_seconds, "  GPU cooldown")
                    
        except KeyboardInterrupt:
            print(f"\n\n{'='*70}")
            print("  ⚠ INTERRUPTED BY USER")
            print(f"{'='*70}")
            break
    
    # Print summary
    total_duration = time.time() - total_start_time
    successful = sum(1 for r in results if r["success"])
    failed = sum(1 for r in results if not r["success"])
    
    print(f"\n{'='*70}")
    print(f"  TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"  Total time:    {format_time(total_duration)}")
    print(f"  Completed:     {len(results)}/{total_runs}")
    print(f"  Successful:    {successful}")
    print(f"  Failed:        {failed}")
    print(f"\n  Results:")
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"    {status} {r['config']:<40} ({format_time(r['duration'])})")
    print(f"{'='*70}\n")
    
    # Exit with error code if any failed
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
