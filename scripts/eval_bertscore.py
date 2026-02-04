"""
Batch BERTScore Evaluation for Multilingual ASR

Computes BERTScore for ASR evaluation across multiple experiments.
Designed for Danish, Dutch, and English ASR evaluation with fair
cross-lingual comparison using consistent configuration.

Configuration:
    - Model: xlm-roberta-large (best multilingual model)
    - Layer: 17 (optimal for xlm-roberta-large per BERTScore paper)
    - Rescaling: Disabled by default (for cross-lingual fairness)

Input format (JSONL, one entry per line):
    {"id": 0, "reference": "ground truth text", "hypothesis": "asr output"}

Usage:
    python eval_bertscore.py --input_file paths.txt --model_name whisper_small --lang da
    python eval_bertscore.py --input_file paths.txt --model_name whisper_small --lang nl
    python eval_bertscore.py --input_file paths.txt --model_name whisper_small --lang en

Reference:
    Zhang et al. (2020). BERTScore: Evaluating Text Generation with BERT. ICLR.
"""

import argparse
import json
import os
import csv
import warnings
from typing import List, Dict, Tuple, Optional
from datetime import datetime

try:
    import numpy as np
except ImportError:
    print("Error: numpy is required. Install with: pip install numpy")
    exit(1)

try:
    from bert_score import score as bert_score_fn
    from bert_score import __version__ as BERT_SCORE_VERSION
except ImportError:
    print("Error: bert-score is required. Install with: pip install bert-score")
    exit(1)

try:
    import torch
except ImportError:
    print("Error: torch is required. Install with: pip install torch")
    exit(1)


# =============================================================================
# Configuration
# =============================================================================

MODEL_TYPE = "xlm-roberta-large"
NUM_LAYERS = 17
RESCALE_WITH_BASELINE = False

SUPPORTED_LANGUAGES = {
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "sv": "Swedish",
    "no": "Norwegian",
}


# =============================================================================
# Utility Functions
# =============================================================================

def print_configuration(lang: str, rescale: bool) -> None:
    """Print evaluation configuration."""
    print("")
    print("=" * 70)
    print("BERTScore Configuration")
    print("=" * 70)
    print("  bert_score version : {}".format(BERT_SCORE_VERSION))
    print("  PyTorch version    : {}".format(torch.__version__))
    print("  CUDA available     : {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("  GPU                : {}".format(torch.cuda.get_device_name(0)))
    print("  Model              : {}".format(MODEL_TYPE))
    print("  Layer              : {}".format(NUM_LAYERS))
    print("  Language           : {} ({})".format(lang, SUPPORTED_LANGUAGES.get(lang, "Unknown")))
    print("  Rescale baseline   : {}".format(rescale))
    print("=" * 70)


def load_jsonl(filepath: str) -> Tuple[List[str], List[str], List[dict]]:
    """
    Load hypotheses and references from JSONL file.

    Args:
        filepath: Path to JSONL file.

    Returns:
        Tuple of (hypotheses, references, metadata).
    """
    hypotheses = []
    references = []
    metadata = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)

                hyp = entry.get("hypothesis") or entry.get("hyp") or entry.get("pred")
                ref = entry.get("reference") or entry.get("ref") or entry.get("target")

                if hyp is None or ref is None:
                    print("  Warning: Line {} missing hypothesis or reference".format(line_num))
                    continue

                hypotheses.append(str(hyp))
                references.append(str(ref))
                metadata.append(entry)

            except json.JSONDecodeError as e:
                print("  Warning: Skipping malformed line {}: {}".format(line_num, e))

    return hypotheses, references, metadata


def load_paths_from_file(filepath: str) -> List[str]:
    """
    Load experiment paths from text file.

    Args:
        filepath: Path to text file with one path per line.

    Returns:
        List of paths.
    """
    paths = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                paths.append(line)
    return paths


def extract_experiment_name(jsonl_path: str) -> str:
    """
    Extract experiment name from file path.

    Args:
        jsonl_path: Path to JSONL file.

    Returns:
        Experiment name string.
    """
    parts = jsonl_path.rstrip("/").split(os.sep)

    for i, part in enumerate(parts):
        if part == "eval" and i > 0:
            return parts[i - 1]

    if len(parts) >= 3:
        return parts[-3]

    return os.path.splitext(os.path.basename(jsonl_path))[0]


# =============================================================================
# BERTScore Calculation
# =============================================================================

def calculate_bertscore(
    hypotheses: List[str],
    references: List[str],
    lang: str,
    batch_size: int = 32,
    verbose: bool = False,
    rescale: bool = RESCALE_WITH_BASELINE,
) -> Dict:
    """
    Calculate BERTScore for hypothesis-reference pairs.

    Args:
        hypotheses: List of hypothesis strings.
        references: List of reference strings.
        lang: Language code (e.g., 'da', 'nl', 'en').
        batch_size: Batch size for processing.
        verbose: Whether to print progress.
        rescale: Whether to apply baseline rescaling.

    Returns:
        Dictionary containing precision, recall, F1 statistics.
    """
    if not rescale:
        warnings.filterwarnings("ignore", message=".*baseline.*")

    P, R, F1 = bert_score_fn(
        cands=hypotheses,
        refs=references,
        model_type=MODEL_TYPE,
        num_layers=NUM_LAYERS,
        lang=lang,
        batch_size=batch_size,
        verbose=verbose,
        rescale_with_baseline=rescale,
        use_fast_tokenizer=True,
    )

    P_np = P.numpy()
    R_np = R.numpy()
    F1_np = F1.numpy()

    results = {
        "precision": {
            "mean": float(P_np.mean()),
            "std": float(P_np.std()),
            "min": float(P_np.min()),
            "max": float(P_np.max()),
            "median": float(np.median(P_np)),
        },
        "recall": {
            "mean": float(R_np.mean()),
            "std": float(R_np.std()),
            "min": float(R_np.min()),
            "max": float(R_np.max()),
            "median": float(np.median(R_np)),
        },
        "f1": {
            "mean": float(F1_np.mean()),
            "std": float(F1_np.std()),
            "min": float(F1_np.min()),
            "max": float(F1_np.max()),
            "median": float(np.median(F1_np)),
        },
        "num_samples": len(hypotheses),
        "config": {
            "model": MODEL_TYPE,
            "num_layers": NUM_LAYERS,
            "lang": lang,
            "rescaled": rescale,
            "bert_score_version": BERT_SCORE_VERSION,
        },
        "per_example": {
            "precision": P_np.tolist(),
            "recall": R_np.tolist(),
            "f1": F1_np.tolist(),
        },
    }

    return results


# =============================================================================
# Experiment Processing
# =============================================================================

def process_experiment(
    jsonl_path: str,
    exp_name: str,
    lang: str,
    batch_size: int,
    output_dir: str,
    rescale: bool = RESCALE_WITH_BASELINE,
) -> Optional[Dict]:
    """
    Process a single experiment.

    Args:
        jsonl_path: Path to JSONL file.
        exp_name: Experiment name.
        lang: Language code.
        batch_size: Batch size for processing.
        output_dir: Output directory.
        rescale: Whether to apply baseline rescaling.

    Returns:
        Results dictionary or None if processing failed.
    """
    print("")
    print("-" * 70)
    print("Processing: {}".format(exp_name))
    print("-" * 70)
    print("  Input: {}".format(jsonl_path))

    hypotheses, references, metadata = load_jsonl(jsonl_path)
    print("  Samples: {}".format(len(hypotheses)))

    if not hypotheses:
        print("  No samples found, skipping.")
        return None

    results = calculate_bertscore(
        hypotheses,
        references,
        lang=lang,
        batch_size=batch_size,
        verbose=True,
        rescale=rescale,
    )

    results["experiment"] = exp_name
    results["input_path"] = jsonl_path
    results["timestamp"] = datetime.now().isoformat()

    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, "{}_bertscore.json".format(exp_name))
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("  Saved: {}".format(result_path))

    print("")
    print("  Results:")
    print("    Precision : {:.4f} (+/-{:.4f})".format(
        results["precision"]["mean"], results["precision"]["std"]))
    print("    Recall    : {:.4f} (+/-{:.4f})".format(
        results["recall"]["mean"], results["recall"]["std"]))
    print("    F1        : {:.4f} (+/-{:.4f})".format(
        results["f1"]["mean"], results["f1"]["std"]))
    print("    F1 range  : [{:.4f}, {:.4f}]".format(
        results["f1"]["min"], results["f1"]["max"]))

    return results


# =============================================================================
# Output Generation
# =============================================================================

def create_summary_csv(
    results: List[Dict],
    output_path: str,
    model_name: str,
    lang: str,
) -> None:
    """
    Create summary CSV file.

    Args:
        results: List of result dictionaries.
        output_path: Output CSV path.
        model_name: Model name for reference.
        lang: Language code.
    """
    results = sorted(results, key=lambda x: x["experiment"])

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "experiment",
            "language",
            "num_samples",
            "precision_mean",
            "precision_std",
            "recall_mean",
            "recall_std",
            "f1_mean",
            "f1_std",
            "f1_min",
            "f1_max",
            "f1_median",
            "model",
            "num_layers",
            "rescaled",
        ])

        for r in results:
            writer.writerow([
                r["experiment"],
                r["config"]["lang"],
                r["num_samples"],
                "{:.4f}".format(r["precision"]["mean"]),
                "{:.4f}".format(r["precision"]["std"]),
                "{:.4f}".format(r["recall"]["mean"]),
                "{:.4f}".format(r["recall"]["std"]),
                "{:.4f}".format(r["f1"]["mean"]),
                "{:.4f}".format(r["f1"]["std"]),
                "{:.4f}".format(r["f1"]["min"]),
                "{:.4f}".format(r["f1"]["max"]),
                "{:.4f}".format(r["f1"]["median"]),
                r["config"]["model"],
                r["config"]["num_layers"],
                r["config"]["rescaled"],
            ])

    print("")
    print("Summary CSV saved: {}".format(output_path))


def print_summary_table(results: List[Dict], model_name: str, lang: str) -> None:
    """
    Print formatted summary table.

    Args:
        results: List of result dictionaries.
        model_name: Model name.
        lang: Language code.
    """
    results = sorted(results, key=lambda x: x["experiment"])

    print("")
    print("=" * 90)
    print("BERTScore Summary: {} | Language: {} ({})".format(
        model_name, lang, SUPPORTED_LANGUAGES.get(lang, "")))
    print("Model: {} | Layer: {} | Rescaled: {}".format(
        MODEL_TYPE, NUM_LAYERS, RESCALE_WITH_BASELINE))
    print("=" * 90)
    print("{:<30} {:>10} {:>10} {:>10} {:>10} {:>8}".format(
        "Experiment", "P", "R", "F1", "F1_std", "N"))
    print("-" * 90)

    for r in results:
        print("{:<30} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>8}".format(
            r["experiment"],
            r["precision"]["mean"],
            r["recall"]["mean"],
            r["f1"]["mean"],
            r["f1"]["std"],
            r["num_samples"]))

    print("=" * 90)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch BERTScore evaluation for multilingual ASR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python eval_bertscore.py -i paths_da.txt -m whisper_small -l da
    python eval_bertscore.py -i paths_nl.txt -m whisper_small -l nl
    python eval_bertscore.py -i paths_en.txt -m whisper_small -l en

Configuration:
    Model: xlm-roberta-large
    Layer: 17 (optimal per BERTScore paper)
    Rescaling: Disabled (for cross-lingual fairness)
        """
    )

    parser.add_argument(
        "--input_file", "-i",
        type=str,
        required=True,
        help="Text file with paths to JSONL files (one per line)"
    )
    parser.add_argument(
        "--model_name", "-m",
        type=str,
        required=True,
        help="ASR model name for output organization"
    )
    parser.add_argument(
        "--lang", "-l",
        type=str,
        required=True,
        choices=list(SUPPORTED_LANGUAGES.keys()),
        help="Language code"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=".",
        help="Output directory (default: current directory)"
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--rescale",
        action="store_true",
        default=False,
        help="Apply baseline rescaling (default: False)"
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print("Error: Input file not found: {}".format(args.input_file))
        exit(1)

    print_configuration(args.lang, args.rescale)

    print("")
    print("Loading paths from: {}".format(args.input_file))
    paths = load_paths_from_file(args.input_file)

    if not paths:
        print("Error: No paths found in input file.")
        exit(1)

    experiments = []
    print("")
    print("Found {} paths:".format(len(paths)))
    for jsonl_path in paths:
        if not os.path.isfile(jsonl_path):
            print("  [NOT FOUND] {}".format(jsonl_path))
            continue
        exp_name = extract_experiment_name(jsonl_path)
        experiments.append((exp_name, jsonl_path))
        print("  [OK] {}".format(exp_name))

    if not experiments:
        print("Error: No valid paths found.")
        exit(1)

    output_folder = "{}_{}".format(args.model_name, args.lang)
    output_dir = os.path.join(args.output_dir, "bertscore_results", output_folder)
    os.makedirs(output_dir, exist_ok=True)
    print("")
    print("Output directory: {}".format(output_dir))

    all_results = []
    for exp_name, jsonl_path in experiments:
        try:
            result = process_experiment(
                jsonl_path,
                exp_name,
                args.lang,
                args.batch_size,
                output_dir,
                rescale=args.rescale,
            )
            if result:
                all_results.append(result)
        except Exception as e:
            print("Error processing {}: {}".format(exp_name, e))

    if not all_results:
        print("Error: No results collected.")
        exit(1)

    summary_path = os.path.join(
        args.output_dir, "bertscore_results", "summary_{}.csv".format(output_folder))
    create_summary_csv(all_results, summary_path, args.model_name, args.lang)

    print_summary_table(all_results, args.model_name, args.lang)

    print("")
    print("Completed: {}/{} experiments processed.".format(
        len(all_results), len(experiments)))
    print("Results directory: {}".format(output_dir))
    print("Summary file: {}".format(summary_path))


if __name__ == "__main__":
    main()