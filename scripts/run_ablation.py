"""
Run layer ablation experiments for SPEAKABLE 2026 paper.

This script runs training with different numbers of Whisper encoder layers
to study the efficiency-accuracy tradeoff.

Usage:
    python scripts/run_ablation.py [--layers 6 4 2] [--config configs/train_config.yaml]
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf


def run_experiment(base_config: str, num_layers: int, dry_run: bool = False):
    """Run a single ablation experiment with specified number of layers."""
    
    # Load config
    cfg = OmegaConf.load(base_config)
    
    # Modify for this experiment
    cfg.model.encoder_num_layers = num_layers if num_layers < 6 else None
    cfg.log.wandb_exp_name = f"ablation_{num_layers}L_danish"
    cfg.train.output_dir = f"outputs/ablation_{num_layers}L/"
    
    # Create output directory
    Path(cfg.train.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save temporary config
    temp_config = f"configs/_temp_ablation_{num_layers}L.yaml"
    OmegaConf.save(cfg, temp_config)
    
    print(f"\n{'='*60}")
    print(f"ABLATION EXPERIMENT: {num_layers} Encoder Layers")
    print(f"{'='*60}")
    print(f"Config: {temp_config}")
    print(f"Output: {cfg.train.output_dir}")
    print(f"W&B Run: {cfg.log.wandb_exp_name}")
    
    if dry_run:
        print(f"\n[DRY RUN] Would execute: python train.py --config {temp_config}")
        os.remove(temp_config)
        return True
    
    print(f"\nStarting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'─'*60}\n")
    
    # Run training
    try:
        result = subprocess.run(
            ["python", "train.py", "--config", temp_config],
            check=True
        )
        success = result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\nError: Training failed with return code {e.returncode}")
        success = False
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        success = False
    finally:
        # Cleanup temp config
        if os.path.exists(temp_config):
            os.remove(temp_config)
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Run layer ablation experiments for SPEAKABLE 2026"
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/train_config.yaml",
        help="Base config file (default: configs/train_config.yaml)"
    )
    parser.add_argument(
        "--layers", "-l",
        nargs="+",
        type=int,
        default=[6, 5, 4, 3, 2, 1],
        help="Layer counts to test (default: 6 5 4 3 2 1)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without executing"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip experiments that already have outputs"
    )
    
    args = parser.parse_args()
    
    # Validate config exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Print experiment plan
    print("=" * 60)
    print("SLAM-ASR Layer Ablation Study")
    print("SPEAKABLE 2026")
    print("=" * 60)
    print(f"\nBase config: {args.config}")
    print(f"Layer configurations: {args.layers}")
    print(f"Total experiments: {len(args.layers)}")
    
    # Expected parameters per layer configuration (Whisper-base)
    layer_params = {
        6: 20.59, 5: 17.44, 4: 14.29,
        3: 11.13, 2: 7.98, 1: 4.83
    }
    
    print(f"\n{'Layers':<10} {'Params (M)':<15} {'% of Full':<10}")
    print("─" * 35)
    for layers in sorted(args.layers, reverse=True):
        params = layer_params.get(layers, "?")
        pct = (params / 20.59 * 100) if isinstance(params, float) else "?"
        print(f"{layers:<10} {params:<15} {pct:.1f}%")
    
    if args.dry_run:
        print("\n[DRY RUN MODE - No training will be executed]\n")
    
    # Run experiments
    results = {}
    for i, num_layers in enumerate(args.layers, 1):
        print(f"\n[{i}/{len(args.layers)}] ", end="")
        
        # Check if output already exists
        output_dir = f"outputs/ablation_{num_layers}L/"
        if args.skip_existing and os.path.exists(output_dir):
            checkpoint_files = list(Path(output_dir).glob("*.pt"))
            if checkpoint_files:
                print(f"Skipping {num_layers}L - output already exists")
                results[num_layers] = "skipped"
                continue
        
        success = run_experiment(args.config, num_layers, args.dry_run)
        results[num_layers] = "success" if success else "failed"
        
        if not success and not args.dry_run:
            print(f"\nExperiment with {num_layers} layers failed!")
            user_input = input("Continue with remaining experiments? [y/N]: ")
            if user_input.lower() != 'y':
                break
    
    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"\n{'Layers':<10} {'Status':<15}")
    print("─" * 25)
    for layers in args.layers:
        status = results.get(layers, "not run")
        print(f"{layers:<10} {status:<15}")
    
    print(f"\nCompleted at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNext steps:")
    print("1. Check W&B dashboard for training curves")
    print("2. Run evaluation: python eval.py --config configs/eval_config.yaml")
    print("3. Collect results for paper table")


if __name__ == "__main__":
    main()
