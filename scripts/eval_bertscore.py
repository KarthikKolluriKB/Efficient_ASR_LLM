"""
BERTScore Evaluation for Multilingual ASR
Supports: Danish (da), Dutch (nl), English (en)
"""

import argparse
from bert_score import score as bert_score
from typing import List, Dict, Tuple
import json


# Recommended models for multilingual BERTScore
# xlm-roberta-large generally performs best for multilingual tasks
LANG_TO_MODEL = {
    "da": "xlm-roberta-large",  # Danish
    "nl": "xlm-roberta-large",  # Dutch
    "en": "xlm-roberta-large",  # English (could also use roberta-large for monolingual)
    "multilingual": "xlm-roberta-large",  # Default for mixed/unknown
}


def calculate_bert_score(
    predictions: List[str],
    references: List[str],
    lang: str = "multilingual",
    model_type: str = None,
    batch_size: int = 32,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Calculate BERTScore for ASR predictions.
    
    Args:
        predictions: List of ASR hypotheses
        references: List of ground truth transcriptions
        lang: Language code ('da', 'nl', 'en', or 'multilingual')
        model_type: Override model (default: xlm-roberta-large)
        batch_size: Batch size for scoring
        verbose: Print progress
        
    Returns:
        Dictionary with P, R, F1 scores (mean and std)
    """
    if model_type is None:
        model_type = LANG_TO_MODEL.get(lang, LANG_TO_MODEL["multilingual"])
    
    P, R, F1 = bert_score(
        cands=predictions,
        refs=references,
        model_type=model_type,
        lang=lang if lang != "multilingual" else None,
        batch_size=batch_size,
        verbose=verbose,
        rescale_with_baseline=True,  # Recommended for interpretable scores
    )
    
    return {
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
        "model": model_type,
        "num_samples": len(predictions),
    }


def calculate_bert_score_per_sample(
    predictions: List[str],
    references: List[str],
    lang: str = "multilingual",
    model_type: str = None,
    batch_size: int = 32,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Get per-sample BERTScores for detailed analysis.
    
    Returns:
        Tuple of (precision_list, recall_list, f1_list)
    """
    if model_type is None:
        model_type = LANG_TO_MODEL.get(lang, LANG_TO_MODEL["multilingual"])
    
    P, R, F1 = bert_score(
        cands=predictions,
        refs=references,
        model_type=model_type,
        lang=lang if lang != "multilingual" else None,
        batch_size=batch_size,
        rescale_with_baseline=True,
    )
    
    return P.tolist(), R.tolist(), F1.tolist()


def evaluate_asr_with_bertscore(
    predictions: List[str],
    references: List[str],
    lang: str,
    wer_scores: List[float] = None,
) -> Dict:
    """
    Full evaluation combining BERTScore with optional WER correlation analysis.
    """
    results = calculate_bert_score(predictions, references, lang, verbose=True)
    
    if wer_scores is not None:
        import numpy as np
        _, _, f1_scores = calculate_bert_score_per_sample(predictions, references, lang)
        
        # Correlation between WER and BERTScore (expect negative correlation)
        correlation = np.corrcoef(wer_scores, f1_scores)[0, 1]
        results["wer_bertscore_correlation"] = correlation
    
    return results


# Example integration with your evaluation pipeline
def integrate_with_existing_eval(
    eval_results: Dict,  # Your existing eval dict with 'predictions' and 'references'
    lang: str,
) -> Dict:
    """
    Add BERTScore to your existing evaluation results.
    
    Example usage with your pipeline:
        eval_results = {
            'predictions': [...],
            'references': [...],
            'wer': 0.15,
            'cer': 0.08,
        }
        eval_results = integrate_with_existing_eval(eval_results, lang='nl')
    """
    bert_results = calculate_bert_score(
        predictions=eval_results["predictions"],
        references=eval_results["references"],
        lang=lang,
    )
    
    eval_results["bert_score"] = {
        "precision": bert_results["precision"]["mean"],
        "recall": bert_results["recall"]["mean"],
        "f1": bert_results["f1"]["mean"],
    }
    
    return eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate BERTScore for ASR")
    parser.add_argument("--predictions", type=str, help="Path to predictions file (one per line)")
    parser.add_argument("--references", type=str, help="Path to references file (one per line)")
    parser.add_argument("--lang", type=str, default="multilingual", 
                        choices=["da", "nl", "en", "multilingual"])
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--batch-size", type=int, default=32)
    
    args = parser.parse_args()
    
    if args.predictions and args.references:
        with open(args.predictions, "r") as f:
            predictions = [line.strip() for line in f]
        with open(args.references, "r") as f:
            references = [line.strip() for line in f]
        
        results = calculate_bert_score(
            predictions, references, 
            lang=args.lang, 
            batch_size=args.batch_size,
            verbose=True
        )
        
        print(f"\n{'='*50}")
        print(f"BERTScore Results ({args.lang})")
        print(f"{'='*50}")
        print(f"Model: {results['model']}")
        print(f"Samples: {results['num_samples']}")
        print(f"Precision: {results['precision']['mean']:.4f} (±{results['precision']['std']:.4f})")
        print(f"Recall:    {results['recall']['mean']:.4f} (±{results['recall']['std']:.4f})")
        print(f"F1:        {results['f1']['mean']:.4f} (±{results['f1']['std']:.4f})")
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")
    else:
        # Demo with example data
        print("Running demo with example data...")
        
        # Example: comparing exact match vs semantic equivalent
        demo_refs = [
            "I went to the store yesterday",
            "The meeting is at three o'clock",
            "She did not want to go",
        ]
        demo_preds = [
            "I went to the store yesterday",  # Exact match
            "The meeting is at 3 o'clock",     # Number format difference
            "She didn't want to go",           # Contraction difference
        ]
        
        results = calculate_bert_score(demo_preds, demo_refs, lang="en", verbose=True)
        
        print(f"\nDemo Results:")
        print(f"F1: {results['f1']['mean']:.4f}")
        
        # Show per-sample scores
        P, R, F1 = calculate_bert_score_per_sample(demo_preds, demo_refs, lang="en")
        print(f"\nPer-sample F1 scores:")
        for ref, pred, f1 in zip(demo_refs, demo_preds, F1):
            print(f"  {f1:.4f}: '{pred}' vs '{ref}'")