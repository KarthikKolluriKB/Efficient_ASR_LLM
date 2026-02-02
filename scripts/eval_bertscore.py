"""
Standalone BERTScore Evaluation for ASR Outputs

Reads hypothesis/reference pairs from saved JSONL files and computes BERTScore.
No model loading required - works directly on saved transcripts.

Usage:
    python eval_bertscore.py --input eval_results/test_examples.jsonl --lang nl
    python eval_bertscore.py --input eval_results/test_examples.jsonl --lang da --output bert_results.json
    python eval_bertscore.py --input path1.jsonl path2.jsonl --lang en  # Multiple files
"""

import argparse
import json
import os
from typing import List, Dict, Tuple
from tqdm import tqdm

try:
    from bert_score import score as bert_score_fn
except ImportError:
    print("Please install bert-score: pip install bert-score")
    exit(1)


# XLM-RoBERTa-large for consistent multilingual evaluation
MODEL_TYPE = "xlm-roberta-large"


def load_jsonl(filepath: str) -> Tuple[List[str], List[str]]:
    """
    Load hypotheses and references from JSONL file.
    
    Expected format (from eval.py):
        {"id": 0, "reference": "...", "hypothesis": "..."}
    
    Returns:
        Tuple of (hypotheses, references)
    """
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
                print(f"Warning: Skipping malformed line {line_num}: {e}")
            except KeyError as e:
                print(f"Warning: Line {line_num} missing key {e}")
    
    return hypotheses, references


def calculate_bertscore(
    hypotheses: List[str],
    references: List[str],
    lang: str,
    batch_size: int = 32,
    verbose: bool = True,
) -> Dict:
    """
    Calculate BERTScore for hypothesis-reference pairs.
    
    Args:
        hypotheses: Model predictions
        references: Ground truth transcriptions
        lang: Language code (da, nl, en)
        batch_size: Batch size for scoring
        verbose: Show progress bar
        
    Returns:
        Dictionary with precision, recall, F1 (mean and std)
    """
    print(f"\nCalculating BERTScore...")
    print(f"  Model: {MODEL_TYPE}")
    print(f"  Language: {lang}")
    print(f"  Samples: {len(hypotheses)}")
    print(f"  Batch size: {batch_size}")
    
    P, R, F1 = bert_score_fn(
        cands=hypotheses,
        refs=references,
        model_type=MODEL_TYPE,
        lang=lang,
        batch_size=batch_size,
        verbose=verbose,
        rescale_with_baseline=True,  # Normalize scores for interpretability
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
    
    return results, (P.tolist(), R.tolist(), F1.tolist())


def print_results(results: Dict):
    """Pretty print BERTScore results."""
    print("\n" + "=" * 60)
    print("BERTSCORE RESULTS")
    print("=" * 60)
    print(f"Model:       {results['model']}")
    print(f"Language:    {results['lang']}")
    print(f"Samples:     {results['num_samples']}")
    print("-" * 60)
    print(f"Precision:   {results['precision']['mean']:.4f} (±{results['precision']['std']:.4f})")
    print(f"Recall:      {results['recall']['mean']:.4f} (±{results['recall']['std']:.4f})")
    print(f"F1:          {results['f1']['mean']:.4f} (±{results['f1']['std']:.4f})")
    print("=" * 60)


def save_results(
    results: Dict,
    output_path: str,
    per_sample_scores: Tuple[List[float], List[float], List[float]] = None,
    hypotheses: List[str] = None,
    references: List[str] = None,
):
    """Save results to JSON file."""
    output = {
        "summary": results,
    }
    
    # Optionally include per-sample scores
    if per_sample_scores and hypotheses and references:
        P, R, F1 = per_sample_scores
        output["per_sample"] = [
            {
                "id": i,
                "reference": ref,
                "hypothesis": hyp,
                "precision": p,
                "recall": r,
                "f1": f1,
            }
            for i, (ref, hyp, p, r, f1) in enumerate(zip(references, hypotheses, P, R, F1))
        ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate BERTScore from saved ASR evaluation outputs"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to JSONL file(s) with hypothesis/reference pairs"
    )
    parser.add_argument(
        "--lang", "-l",
        type=str,
        required=True,
        choices=["da", "nl", "en"],
        help="Language code: da (Danish), nl (Dutch), en (English)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON path for results (optional)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size for BERTScore computation (default: 32)"
    )
    parser.add_argument(
        "--save-per-sample",
        action="store_true",
        help="Include per-sample scores in output JSON"
    )
    
    args = parser.parse_args()
    
    # Load all JSONL files
    all_hypotheses = []
    all_references = []
    
    for filepath in args.input:
        if not os.path.exists(filepath):
            print(f"Error: File not found: {filepath}")
            exit(1)
        
        print(f"Loading: {filepath}")
        hyps, refs = load_jsonl(filepath)
        print(f"  Loaded {len(hyps)} samples")
        all_hypotheses.extend(hyps)
        all_references.extend(refs)
    
    if not all_hypotheses:
        print("Error: No samples loaded!")
        exit(1)
    
    print(f"\nTotal samples: {len(all_hypotheses)}")
    
    # Calculate BERTScore
    results, per_sample = calculate_bertscore(
        all_hypotheses,
        all_references,
        lang=args.lang,
        batch_size=args.batch_size,
    )
    
    # Print results
    print_results(results)
    
    # Save if output path specified
    if args.output:
        save_results(
            results,
            args.output,
            per_sample_scores=per_sample if args.save_per_sample else None,
            hypotheses=all_hypotheses if args.save_per_sample else None,
            references=all_references if args.save_per_sample else None,
        )
    
    # Return F1 for easy scripting
    return results["f1"]["mean"]


if __name__ == "__main__":
    main()