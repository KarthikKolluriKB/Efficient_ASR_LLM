"""
Process Common Voice English dataset - 500 HOURS for data scaling experiment.

This creates a 500h training subset to test if more data improves WER.
100h achieved 26% WER, expecting 12-18% WER with 500h.

STEP 1: Make sure you have Common Voice English downloaded and extracted
        to: data/cv-corpus-XX/en/

STEP 2: Run this script:
        python scripts/download_cv_english_500h.py

STEP 3: Train:
        python train.py --config configs/train_config_english_500h.yaml
"""

import os
import csv
import json
from pathlib import Path
from tqdm import tqdm

# Config - UPDATE THIS PATH to match your downloaded corpus version
CV_CORPUS_DIR = Path("data/cv-corpus-24.0-2025-12-05/en")  # Adjust version if needed
OUTPUT_DIR = Path("data/common_voice_en_500h")

# =============================================================================
# 500 HOURS for data scaling experiment
# =============================================================================
MAX_TRAIN_HOURS = 500  # Target ~500 hours of training data
AVG_CLIP_SECONDS = 5   # Average clip length in Common Voice
MAX_TRAIN_SAMPLES = int(MAX_TRAIN_HOURS * 3600 / AVG_CLIP_SECONDS)  # ~360,000 samples

# Use same val/test as 100h for fair comparison
MAX_VAL_SAMPLES = 3000   # ~4 hours for validation
MAX_TEST_SAMPLES = 3000  # ~4 hours for testing

print(f"Will process max {MAX_TRAIN_SAMPLES} train samples (~{MAX_TRAIN_HOURS} hours)")


def process_tsv(tsv_path, output_jsonl, audio_dir, max_samples=None):
    """Process a Common Voice TSV file to JSONL format."""
    samples = []
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        
        for i, row in enumerate(tqdm(reader, desc=f"Processing {tsv_path.name}")):
            if max_samples and i >= max_samples:
                break
            
            # Get audio path and text
            audio_filename = row['path']
            text = row['sentence'].strip()
            
            if not text:
                continue
            
            # Build full audio path
            audio_path = audio_dir / audio_filename
            
            # Check if audio file exists
            if not audio_path.exists():
                continue
            
            # Create sample entry matching dataset.py expectations
            samples.append({
                "source": str(audio_path),
                "target": text,
                "prompt": "ASR",
            })
    
    # Write JSONL
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / output_jsonl
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"  Saved {len(samples)} samples to {output_path}")
    return len(samples)


def main():
    print("=" * 60)
    print("Processing Common Voice English - 500 HOURS")
    print("Data scaling experiment: 100h->500h")
    print("=" * 60)
    
    # Check if corpus exists
    if not CV_CORPUS_DIR.exists():
        print(f"\nERROR: Corpus not found at {CV_CORPUS_DIR}")
        print("\nPlease download Common Voice English from:")
        print("  https://commonvoice.mozilla.org/en/datasets")
        print(f"\nExtract to: {CV_CORPUS_DIR.parent}/")
        print("\nThen run this script again.")
        return
    
    # Find clips directory
    clips_dir = CV_CORPUS_DIR / "clips"
    if not clips_dir.exists():
        print(f"\nERROR: Clips directory not found at {clips_dir}")
        return
    
    print(f"\nFound corpus at: {CV_CORPUS_DIR}")
    print(f"Clips directory: {clips_dir}")
    
    # Process each split
    total_samples = 0
    
    # Train - 500 hours
    train_tsv = CV_CORPUS_DIR / "train.tsv"
    if train_tsv.exists():
        n = process_tsv(train_tsv, "train.jsonl", clips_dir, MAX_TRAIN_SAMPLES)
        total_samples += n
    else:
        print(f"WARNING: {train_tsv} not found")
    
    # Validation (dev.tsv in Common Voice)
    dev_tsv = CV_CORPUS_DIR / "dev.tsv"
    if dev_tsv.exists():
        n = process_tsv(dev_tsv, "validation.jsonl", clips_dir, MAX_VAL_SAMPLES)
        total_samples += n
    else:
        print(f"WARNING: {dev_tsv} not found")
    
    # Test
    test_tsv = CV_CORPUS_DIR / "test.tsv"
    if test_tsv.exists():
        n = process_tsv(test_tsv, "test.jsonl", clips_dir, MAX_TEST_SAMPLES)
        total_samples += n
    else:
        print(f"WARNING: {test_tsv} not found")
    
    # Create dataset_info.json
    info = {
        "name": "common_voice_en_500h",
        "language": "en",
        "description": "Common Voice English 500h subset for data scaling experiment",
        "train_hours_target": MAX_TRAIN_HOURS,
        "source": str(CV_CORPUS_DIR)
    }
    with open(OUTPUT_DIR / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Done! Processed {total_samples} total samples")
    print(f"Dataset saved to: {OUTPUT_DIR}")
    print("=" * 60)
    print("\nNext step: Run training")
    print("  python train.py --config configs/train_config_english_500h.yaml")


if __name__ == "__main__":
    main()
