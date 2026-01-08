"""
Download Common Voice English subset for architecture validation test.
This is temporary - delete after confirming architecture works.

Usage: python scripts/download_cv_english.py
"""

import os
import json
from pathlib import Path
from datasets import load_dataset
import soundfile as sf
from tqdm import tqdm

# Config
OUTPUT_DIR = Path("data/common_voice_en")
AUDIO_DIR = OUTPUT_DIR / "audio"
MAX_TRAIN_SAMPLES = 30000  # ~50-60 hours (avg 6-7 sec per clip)
MAX_VAL_SAMPLES = 3000
MAX_TEST_SAMPLES = 3000

def main():
    print("=" * 60)
    print("Downloading Common Voice English (subset for testing)")
    print("=" * 60)
    
    # Create directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load dataset from Hugging Face (streaming to avoid full download)
    print("\nLoading dataset from Hugging Face...")
    print("This may take a while for the first download...\n")
    
    # Load English Common Voice
    dataset = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        "en",
        trust_remote_code=True
    )
    
    # Process each split
    for split_name, max_samples, output_file in [
        ("train", MAX_TRAIN_SAMPLES, "train.jsonl"),
        ("validation", MAX_VAL_SAMPLES, "validation.jsonl"),
        ("test", MAX_TEST_SAMPLES, "test.jsonl"),
    ]:
        print(f"\nProcessing {split_name} split (max {max_samples} samples)...")
        
        split_audio_dir = AUDIO_DIR / split_name
        split_audio_dir.mkdir(parents=True, exist_ok=True)
        
        split_data = dataset[split_name]
        samples = []
        
        for i, item in enumerate(tqdm(split_data, total=min(max_samples, len(split_data)))):
            if i >= max_samples:
                break
            
            # Get audio and text
            audio = item["audio"]
            text = item["sentence"].strip()
            
            if not text:  # Skip empty transcripts
                continue
            
            # Save audio file
            audio_filename = f"en_{split_name}_{i:06d}.wav"
            audio_path = split_audio_dir / audio_filename
            
            # Save as WAV (16kHz)
            sf.write(
                str(audio_path),
                audio["array"],
                audio["sampling_rate"]
            )
            
            # Create sample entry
            samples.append({
                "audio": str(audio_path.relative_to(OUTPUT_DIR.parent.parent)),
                "text": text,
                "duration": len(audio["array"]) / audio["sampling_rate"]
            })
        
        # Write JSONL
        output_path = OUTPUT_DIR / output_file
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        # Calculate hours
        total_hours = sum(s["duration"] for s in samples) / 3600
        print(f"  Saved {len(samples)} samples ({total_hours:.1f} hours) to {output_path}")
    
    # Create dataset_info.json
    info = {
        "name": "common_voice_en",
        "language": "en",
        "description": "Common Voice English subset for architecture validation",
        "temporary": True
    }
    with open(OUTPUT_DIR / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Done! Dataset saved to:", OUTPUT_DIR)
    print("=" * 60)
    print("\nNext step: Run training with English config")
    print("  python train.py --config configs/train_config_english_test.yaml")


if __name__ == "__main__":
    main()
