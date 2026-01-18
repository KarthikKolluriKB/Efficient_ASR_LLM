#!/usr/bin/env python3
"""
Download and preprocess Common Voice dataset for training.

This script downloads the Common Voice dataset from HuggingFace and saves it
in HuggingFace Dataset format that's compatible with the training pipeline.

Usage:
    python scripts/download_dataset.py --language da
    python scripts/download_dataset.py --language da --output-dir data/cv22_hf
    python scripts/download_dataset.py --language en --max-hours 100

The dataset will be saved to: {output_dir}/{language}/
    - train/
    - validation/
    - test/
    - dataset_dict.json

To load the dataset:
    from datasets import load_from_disk
    dataset = load_from_disk("data/cv22_hf/da")
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datamodule.hf_data import prepare_and_save


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download Common Voice dataset in HuggingFace format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download Danish dataset (default)
    python scripts/download_dataset.py --language da
    
    # Download English dataset with custom output directory
    python scripts/download_dataset.py --language en --output-dir data/cv22_english
    
    # Download English with 100 hours of training data
    python scripts/download_dataset.py --language en --max-hours 100
    
    # Download with custom duration filters
    python scripts/download_dataset.py --language da --min-duration 1.0 --max-duration 15.0
    
    # Download only train and validation splits
    python scripts/download_dataset.py --language da --splits train dev
        """
    )
    
    parser.add_argument(
        "--language", "-l",
        type=str,
        default="da",
        help="Language code: da (Danish), en (English), nl (Dutch), etc. (default: da)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="data/cv22_hf",
        help="Output directory for HuggingFace dataset (default: data/cv22_hf)"
    )
    parser.add_argument(
        "--splits", "-s",
        type=str,
        nargs="+",
        default=["train", "dev", "test"],
        choices=["train", "dev", "test"],
        help="Splits to download (default: train dev test)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target audio sample rate in Hz (default: 16000)"
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        help="Minimum audio duration in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum audio duration in seconds (default: 30.0)"
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="Maximum hours of training data (default: None = use all data). "
             "Only applies to training split; dev and test are kept complete."
    )
    parser.add_argument(
        "--limit-seed",
        type=int,
        default=42,
        help="Random seed for reproducible sample selection when using --max-hours (default: 42)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4000,
        help="Batch size for dataset creation to avoid OOM (default: 4000)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Common Voice Dataset Downloader")
    print("=" * 70)
    print(f"Language:     {args.language}")
    print(f"Output:       {args.output_dir}/{args.language}/")
    print(f"Splits:       {', '.join(args.splits)}")
    print(f"Sample rate:  {args.sample_rate} Hz")
    print(f"Duration:     {args.min_duration}s - {args.max_duration}s")
    if args.max_hours is not None:
        print(f"Max hours:    {args.max_hours} hours (training only)")
        print(f"Limit seed:   {args.limit_seed}")
    else:
        print(f"Max hours:    No limit (download all)")
    print("=" * 70)
    
    # Download and prepare dataset
    save_path = prepare_and_save(
        language=args.language,
        output_dir=args.output_dir,
        splits=args.splits,
        sample_rate=args.sample_rate,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        batch_size=args.batch_size,
        max_train_hours=args.max_hours,
        limit_seed=args.limit_seed,
    )
    
    print(f"\n{'=' * 70}")
    print("Download Complete!")
    print("=" * 70)
    print(f"Dataset saved to: {save_path}")
    print(f"\nTo verify, run:")
    print(f"  python -c \"from datasets import load_from_disk; print(load_from_disk('{save_path}'))\"")
    print(f"\nTo start training:")
    print(f"  python train.py --config configs/{args.language}/train/baseline.yaml")


if __name__ == "__main__":
    main()