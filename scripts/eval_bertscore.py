"""
Batch BERTScore Evaluation with Directory Scanning

Scans a root directory for experiment outputs and computes BERTScore for each.
Creates a summary CSV with all results.

Directory structure expected:
    rootdir/
    ├── baseline/eval/test_examples.jsonl
    ├── ablation_11L/eval/test_examples.jsonl
    ├── ablation_10L/eval/test_examples.jsonl
    └── ...

Output structure created:
    rootdir/
    ├── bert_scores/
    │   ├── model_name_lang/
    │   │   ├── baseline_bert_results.json
    │   │   ├── ablation_11L_bert_results.json
    │   │   └── ...
    │   └── summary_model_name_lang.csv
    └── ...

Usage:
    python eval_bertscore_batch.py --root_dir outputs/ --model_name whisper_large --lang nl
    python eval_bertscore_batch.py --root_dir outputs/ --model_name whisper_large --lang da --batch_size 16
    python eval_bertscore_batch.py --root_dir outputs/ --model_name whisper_large --lang nl --parallel
"""

import argparse
import json
import os
import glob
import csv
from typing import List, Dict, Tuple
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

try:
    from bert_score import score as bert_score_fn
except ImportError:
    print("Please install bert-score: pip install bert-score")
    exit(1)


# XLM-RoBERTa-large for consistent multilingual evaluation
MODEL_TYPE = "xlm-roberta-large"


def load_jsonl(filepath: str) -> Tuple[List[str], List[str]]:
    """Load hypotheses and references from JSONL file."""
    hypotheses = []
    references = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                hypotheses.append(entry["hypothesis"])
                references.append(entry["reference"])
            except json.JSONDecodeError as e:
                print(f"  Warning: Skipping malformed line {line_num}: {e}")
            except KeyError as e:
                print(f"  Warning: Line {line_num} missing key {e}")
    
    return hypotheses, references


def calculate_bertscore(
    hypotheses: List[str],
    references: List[str],
    lang: str,
    batch_size: int = 32,
    verbose: bool = False,
) -> Dict:
    """Calculate BERTScore for hypothesis-reference pairs."""
    
    P, R, F1 = bert_score_fn(
        cands=hypotheses,
        refs=references,
        model_type=MODEL_TYPE,
        lang=lang,
        batch_size=batch_size,
        verbose=verbose,
        rescale_with_baseline=True,
    )
    
    results = {
        "precision": {
            "mean": P.mean().item(),
            "std": P.std().item(),
        },
        "recall": {
            "mean": R.mean().item(),
            "std": R.std().item(),
        },
        "f1": {
            "mean": F1.mean().item(),
            "std": F1.std().item(),
        },
        "num_samples": len(hypotheses),
        "model": MODEL_TYPE,
        "lang": lang,
    }
    
    return results


def process_single_experiment(
    jsonl_path: str,
    exp_name: str,
    lang: str,
    batch_size: int,
    output_dir: str,
) -> Dict:
    """Process a single experiment and save results."""
    
    print(f"\n{'='*60}")
    print(f"Processing: {exp_name}")
    print(f"{'='*60}")
    print(f"  Input: {jsonl_path}")
    
    # Load data
    hypotheses, references = load_jsonl(jsonl_path)
    print(f"  Samples: {len(hypotheses)}")
    
    if not hypotheses:
        print(f"  ❌ No samples found, skipping...")
        return None
    
    # Calculate BERTScore
    results = calculate_bertscore(
        hypotheses,
        references,
        lang=lang,
        batch_size=batch_size,
        verbose=True,
    )
    
    # Add experiment metadata
    results["experiment"] = exp_name
    results["input_path"] = jsonl_path
    results["timestamp"] = datetime.now().isoformat()
    
    # Save individual result
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"{exp_name}_bert_results.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {result_path}")
    
    # Print summary
    print(f"  Precision: {results['precision']['mean']:.4f} (±{results['precision']['std']:.4f})")
    print(f"  Recall:    {results['recall']['mean']:.4f} (±{results['recall']['std']:.4f})")
    print(f"  F1:        {results['f1']['mean']:.4f} (±{results['f1']['std']:.4f})")
    
    return results


def discover_experiments(root_dir: str, pattern: str = "*/eval/test_examples.jsonl") -> List[Tuple[str, str]]:
    """
    Discover all experiment directories matching the pattern.
    
    Returns:
        List of (experiment_name, jsonl_path) tuples
    """
    search_pattern = os.path.join(root_dir, pattern)
    jsonl_files = glob.glob(search_pattern)
    
    experiments = []
    for jsonl_path in sorted(jsonl_files):
        # Extract experiment name from path: rootdir/EXP_NAME/eval/test_examples.jsonl
        rel_path = os.path.relpath(jsonl_path, root_dir)
        exp_name = rel_path.split(os.sep)[0]
        experiments.append((exp_name, jsonl_path))
    
    return experiments


def create_summary_csv(
    results: List[Dict],
    output_path: str,
    model_name: str,
    lang: str,
):
    """Create summary CSV with all experiment results."""
    
    # Sort by experiment name
    results = sorted(results, key=lambda x: x["experiment"])
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            "exp_name",
            "avg_precision",
            "avg_recall", 
            "avg_f1",
            "std_precision",
            "std_recall",
            "std_f1",
            "num_samples",
        ])
        
        # Data rows
        for r in results:
            writer.writerow([
                r["experiment"],
                f"{r['precision']['mean']:.4f}",
                f"{r['recall']['mean']:.4f}",
                f"{r['f1']['mean']:.4f}",
                f"{r['precision']['std']:.4f}",
                f"{r['recall']['std']:.4f}",
                f"{r['f1']['std']:.4f}",
                r["num_samples"],
            ])
    
    print(f"\n✅ Summary CSV saved: {output_path}")


def print_summary_table(results: List[Dict], model_name: str, lang: str):
    """Print a formatted summary table to console."""
    
    results = sorted(results, key=lambda x: x["experiment"])
    
    print("\n")
    print("=" * 80)
    print(f"BERTSCORE SUMMARY: {model_name} ({lang})")
    print("=" * 80)
    print(f"{'Experiment':<25} {'Precision':>12} {'Recall':>12} {'F1':>12} {'Samples':>10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['experiment']:<25} "
              f"{r['precision']['mean']:>12.4f} "
              f"{r['recall']['mean']:>12.4f} "
              f"{r['f1']['mean']:>12.4f} "
              f"{r['num_samples']:>10}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Batch BERTScore evaluation with directory scanning"
    )
    parser.add_argument(
        "--root_dir", "-r",
        type=str,
        required=True,
        help="Root directory containing experiment folders"
    )
    parser.add_argument(
        "--model_name", "-m",
        type=str,
        required=True,
        help="Model name for output organization (e.g., whisper_large)"
    )
    parser.add_argument(
        "--lang", "-l",
        type=str,
        required=True,
        choices=["da", "nl", "en"],
        help="Language code: da (Danish), nl (Dutch), en (English)"
    )
    parser.add_argument(
        "--pattern", "-p",
        type=str,
        default="*/eval/test_examples.jsonl",
        help="Glob pattern to find JSONL files (default: */eval/test_examples.jsonl)"
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=32,
        help="Batch size for BERTScore computation (default: 32)"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process experiments in parallel (use with caution - high GPU memory)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2)"
    )
    
    args = parser.parse_args()
    
    # Validate root directory
    if not os.path.isdir(args.root_dir):
        print(f"Error: Root directory not found: {args.root_dir}")
        exit(1)
    
    # Discover experiments
    print(f"\nScanning: {args.root_dir}")
    print(f"Pattern:  {args.pattern}")
    experiments = discover_experiments(args.root_dir, args.pattern)
    
    if not experiments:
        print(f"❌ No experiments found matching pattern: {args.pattern}")
        exit(1)
    
    print(f"\nFound {len(experiments)} experiments:")
    for exp_name, jsonl_path in experiments:
        print(f"  • {exp_name}")
    
    # Create output directory
    output_folder = f"{args.model_name}_{args.lang}"
    output_dir = os.path.join(args.root_dir, "bert_scores", output_folder)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Process experiments
    all_results = []
    
    if args.parallel and len(experiments) > 1:
        print(f"\n⚡ Running in parallel mode with {args.num_workers} workers")
        print("   (Note: May require high GPU memory)")
        
        # Parallel processing - be careful with GPU memory
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(
                    process_single_experiment,
                    jsonl_path,
                    exp_name,
                    args.lang,
                    args.batch_size,
                    output_dir,
                ): exp_name
                for exp_name, jsonl_path in experiments
            }
            
            for future in as_completed(futures):
                exp_name = futures[future]
                try:
                    result = future.result()
                    if result:
                        all_results.append(result)
                except Exception as e:
                    print(f"❌ Error processing {exp_name}: {e}")
    else:
        # Sequential processing (default - safer for GPU)
        for exp_name, jsonl_path in experiments:
            try:
                result = process_single_experiment(
                    jsonl_path,
                    exp_name,
                    args.lang,
                    args.batch_size,
                    output_dir,
                )
                if result:
                    all_results.append(result)
            except Exception as e:
                print(f"❌ Error processing {exp_name}: {e}")
    
    if not all_results:
        print("\n❌ No results collected!")
        exit(1)
    
    # Create summary CSV
    summary_path = os.path.join(args.root_dir, "bert_scores", f"summary_{output_folder}.csv")
    create_summary_csv(all_results, summary_path, args.model_name, args.lang)
    
    # Print summary table
    print_summary_table(all_results, args.model_name, args.lang)
    
    print(f"\n✅ Completed! Processed {len(all_results)}/{len(experiments)} experiments.")
    print(f"   Results: {output_dir}")
    print(f"   Summary: {summary_path}")


if __name__ == "__main__":
    main()