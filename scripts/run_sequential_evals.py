"""
Run multiple evaluation experiments sequentially.

This script takes multiple eval config files as input and runs evaluation for each one,
collecting and summarizing results across all experiments.

Usage:
    python scripts/run_sequential_eval.py config1.yaml config2.yaml config3.yaml
    python scripts/run_sequential_eval.py configs/danish/eval/*.yaml
    python scripts/run_sequential_eval.py -c configs/danish/eval/eval_baseline.yaml configs/danish/eval/eval_6L.yaml
    
    # Run on specific GPU
    python scripts/run_sequential_eval.py --gpu 0 config1.yaml config2.yaml
    
    # Override beam size for all experiments
    python scripts/run_sequential_eval.py configs/danish/eval/*.yaml --num_beams 4
    
    # Custom batch size
    python scripts/run_sequential_eval.py configs/danish/eval/*.yaml --batch_size 4
"""
import os
import sys
import argparse
import subprocess
import time
import gc
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix CUDA memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


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


def get_eval_results(output_dir: str) -> Optional[Dict]:
    """Read evaluation results from output directory."""
    summary_path = Path(output_dir) / "eval_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"  ⚠ Failed to read results: {e}")
    return None


def extract_output_dir_from_config(config_path: str) -> Optional[str]:
    """Extract output directory from config file."""
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(config_path)
        if hasattr(cfg, 'eval') and hasattr(cfg.eval, 'output_dir'):
            return cfg.eval.output_dir
    except Exception:
        pass
    return None


def run_evaluation(
    config_path: str, 
    run_number: int, 
    total_runs: int, 
    gpu_id: int = None,
    num_beams: int = None,
    batch_size: int = None,
    extra_args: List[str] = None
) -> Dict:
    """
    Run evaluation with the specified config file.
    
    Args:
        config_path: Path to the config file
        run_number: Current experiment number
        total_runs: Total number of experiments
        gpu_id: GPU device ID to use (None for default)
        num_beams: Override beam size (None to use config)
        batch_size: Override batch size (None to use config)
        extra_args: Additional CLI arguments to pass
    
    Returns:
        dict: Results including success, duration, WER, etc.
    """
    config_name = Path(config_path).stem
    output_dir = extract_output_dir_from_config(config_path)
    
    gpu_info = f" [GPU {gpu_id}]" if gpu_id is not None else ""
    print(f"\n{'='*70}")
    print(f"  EVALUATION [{run_number}/{total_runs}]{gpu_info}: {config_name}")
    print(f"{'='*70}")
    print(f"  Config: {config_path}")
    if gpu_id is not None:
        print(f"  GPU: CUDA:{gpu_id}")
    if num_beams is not None:
        print(f"  Beam size: {num_beams} (override)")
    if batch_size is not None:
        print(f"  Batch size: {batch_size} (override)")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'─'*70}\n")
    
    start_time = time.time()
    
    # Set up environment
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Build command
    cmd = [sys.executable, "eval.py", "--config", config_path]
    
    if num_beams is not None:
        cmd.extend(["--num_beams", str(num_beams)])
    if batch_size is not None:
        cmd.extend(["--batch_size", str(batch_size)])
    if extra_args:
        cmd.extend(extra_args)
    
    result_data = {
        "config": config_name,
        "config_path": config_path,
        "success": False,
        "duration": 0,
        "wer": None,
        "word_accuracy": None,
        "num_samples": None,
        "num_beams": num_beams,
    }
    
    try:
        # Run evaluation
        result = subprocess.run(
            cmd,
            check=True,
            cwd=Path(__file__).parent.parent,  # Run from project root
            env=env
        )
        result_data["success"] = result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n  ✗ Evaluation failed with return code {e.returncode}")
        result_data["success"] = False
    except KeyboardInterrupt:
        print("\n\n  ⚠ Evaluation interrupted by user")
        raise
    except Exception as e:
        print(f"\n  ✗ Evaluation failed with error: {e}")
        result_data["success"] = False
    
    duration = time.time() - start_time
    result_data["duration"] = duration
    
    # Try to read results from output
    if output_dir and result_data["success"]:
        eval_results = get_eval_results(output_dir)
        if eval_results:
            result_data["wer"] = eval_results.get("wer")
            result_data["word_accuracy"] = eval_results.get("word_accuracy")
            result_data["num_samples"] = eval_results.get("num_samples")
            result_data["num_beams"] = eval_results.get("num_beams", num_beams)
    
    status = "✓ COMPLETED" if result_data["success"] else "✗ FAILED"
    print(f"\n{'─'*70}")
    print(f"  {status} in {format_time(duration)}")
    if result_data["wer"] is not None:
        print(f"  WER: {result_data['wer']:.4f} | Word Accuracy: {result_data['word_accuracy']:.4f}")
    print(f"  Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return result_data


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


def print_results_table(results: List[Dict]):
    """Print a formatted results table."""
    if not results:
        return
    
    # Header
    print(f"\n{'─'*90}")
    print(f"  {'Experiment':<35} {'Status':<10} {'WER':<10} {'Acc':<10} {'Beams':<8} {'Time':<10}")
    print(f"{'─'*90}")
    
    # Rows
    for r in results:
        status = "✓ Pass" if r["success"] else "✗ Fail"
        wer = f"{r['wer']:.4f}" if r["wer"] is not None else "N/A"
        acc = f"{r['word_accuracy']:.4f}" if r["word_accuracy"] is not None else "N/A"
        beams = str(r["num_beams"]) if r["num_beams"] is not None else "N/A"
        time_str = format_time(r["duration"])
        
        print(f"  {r['config']:<35} {status:<10} {wer:<10} {acc:<10} {beams:<8} {time_str:<10}")
    
    print(f"{'─'*90}")
    
    # Summary statistics
    successful = [r for r in results if r["success"] and r["wer"] is not None]
    if successful:
        wers = [r["wer"] for r in successful]
        best_result = min(successful, key=lambda x: x["wer"])
        worst_result = max(successful, key=lambda x: x["wer"])
        avg_wer = sum(wers) / len(wers)
        
        print(f"\n  📊 Summary Statistics:")
        print(f"     Best WER:  {best_result['wer']:.4f} ({best_result['config']})")
        print(f"     Worst WER: {worst_result['wer']:.4f} ({worst_result['config']})")
        print(f"     Avg WER:   {avg_wer:.4f}")


def save_results_summary(results: List[Dict], output_path: str):
    """Save all results to a JSON file."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "num_experiments": len(results),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "results": results,
    }
    
    # Add aggregated statistics
    successful = [r for r in results if r["success"] and r["wer"] is not None]
    if successful:
        wers = [r["wer"] for r in successful]
        summary["statistics"] = {
            "best_wer": min(wers),
            "worst_wer": max(wers),
            "avg_wer": sum(wers) / len(wers),
            "best_model": min(successful, key=lambda x: x["wer"])["config"],
        }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n  📁 Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple evaluation experiments sequentially",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run specific configs
  python scripts/run_sequential_eval.py configs/danish/eval/eval_baseline.yaml configs/danish/eval/eval_6L.yaml
  
  # Run all configs in a folder (bash/powershell glob)
  python scripts/run_sequential_eval.py configs/danish/eval/*.yaml
  
  # Override beam size for all experiments
  python scripts/run_sequential_eval.py configs/danish/eval/*.yaml --num_beams 4
  
  # Run on specific GPU with custom batch size
  python scripts/run_sequential_eval.py --gpu 1 --batch_size 4 configs/danish/eval/*.yaml
  
  # Dry run to see what would be executed
  python scripts/run_sequential_eval.py configs/danish/eval/*.yaml --dry-run
  
  # Save combined results to file
  python scripts/run_sequential_eval.py configs/danish/eval/*.yaml --save-results results/eval_summary.json
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
        "--gpu",
        type=int,
        default=None,
        help="GPU device ID to use (e.g., 0 or 1). Sets CUDA_VISIBLE_DEVICES."
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="Override beam size for all experiments"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size for all experiments"
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
        "--save-results",
        type=str,
        default=None,
        help="Path to save combined results JSON"
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=30,
        help="Gap between experiments in seconds (default: 30)"
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
    total_runs = len(valid_configs)
    
    print(f"\n{'='*70}")
    print(f"  SEQUENTIAL EVALUATION PLAN")
    print(f"{'='*70}")
    print(f"  Total experiments: {total_runs}")
    if args.gpu is not None:
        print(f"  GPU Device:        CUDA:{args.gpu}")
    if args.num_beams is not None:
        print(f"  Beam size:         {args.num_beams} (override)")
    if args.batch_size is not None:
        print(f"  Batch size:        {args.batch_size} (override)")
    print(f"  Gap between runs:  {args.gap} seconds")
    print(f"  Continue on error: {'Yes' if args.continue_on_error else 'No'}")
    print(f"\n  Experiments in order:")
    for i, config in enumerate(valid_configs, 1):
        print(f"    {i}. {Path(config).stem}")
    print(f"{'='*70}")
    
    if args.dry_run:
        gpu_prefix = f"CUDA_VISIBLE_DEVICES={args.gpu} " if args.gpu is not None else ""
        print(f"\n🔍 DRY RUN - Commands that would be executed:\n")
        for i, config in enumerate(valid_configs, 1):
            cmd_parts = [f"{gpu_prefix}python eval.py --config {config}"]
            if args.num_beams is not None:
                cmd_parts.append(f"--num_beams {args.num_beams}")
            if args.batch_size is not None:
                cmd_parts.append(f"--batch_size {args.batch_size}")
            print(f"  [{i}/{total_runs}] {' '.join(cmd_parts)}")
        print("\n  No actual evaluation was performed.")
        return
    
    # Confirm before starting
    print(f"\n⏳ Starting in 3 seconds... (Ctrl+C to cancel)")
    time.sleep(3)
    
    # Run experiments
    results = []
    total_start_time = time.time()
    
    for i, config in enumerate(valid_configs, 1):
        try:
            result = run_evaluation(
                config_path=config,
                run_number=i,
                total_runs=total_runs,
                gpu_id=args.gpu,
                num_beams=args.num_beams,
                batch_size=args.batch_size,
            )
            results.append(result)
            
            if not result["success"] and not args.continue_on_error:
                print(f"\n  ⚠ Stopping due to failed experiment (use --continue-on-error to override)")
                break
            
            # Brief cooldown between experiments (skip after last one)
            if i < total_runs:
                print(f"\n🧊 Cleaning up before next experiment...")
                clear_gpu_memory()
                print(f"  Waiting {args.gap} seconds...")
                time.sleep(args.gap)
                    
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
    print(f"  EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Total time:    {format_time(total_duration)}")
    print(f"  Completed:     {len(results)}/{total_runs}")
    print(f"  Successful:    {successful}")
    print(f"  Failed:        {failed}")
    
    # Print results table
    print_results_table(results)
    
    # Save results if requested
    if args.save_results:
        save_results_summary(results, args.save_results)
    
    print(f"{'='*70}\n")
    
    # Exit with error code if any failed
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()