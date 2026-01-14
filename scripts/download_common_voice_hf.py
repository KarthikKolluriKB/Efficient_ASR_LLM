"""
Download and preprocess Common Voice 22.0 dataset from HuggingFace.
Dataset: fsicoli/common_voice_22_0

This script:
1. Downloads the dataset files directly from HuggingFace Hub
2. Preprocesses transcriptions (lowercase, remove punctuation)
3. Filters by audio duration (min: 0.5s, max: 30s)
4. Saves processed data with both raw and preprocessed transcriptions

Usage:
    python scripts/download_common_voice_hf.py --language da --output_dir data/cv22_danish
    python scripts/download_common_voice_hf.py --language en --output_dir data/cv22_english
    python scripts/download_common_voice_hf.py --language nl --output_dir data/cv22_dutch
"""

import os
import re
import json
import string
import argparse
import tarfile
import tempfile
from pathlib import Path
from typing import Optional, Dict, List, Any
from tqdm import tqdm
import csv

try:
    from huggingface_hub import hf_hub_download, list_repo_files
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("[ERROR] huggingface_hub not installed. Run: pip install huggingface_hub")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("[WARNING] soundfile not installed. Run: pip install soundfile")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


# Dataset configuration
DATASET_REPO = "fsicoli/common_voice_22_0"
SAMPLE_RATE = 16000


def preprocess_transcription(text: str) -> str:
    """
    Preprocess transcription:
    - Convert to lowercase
    - Remove all punctuation
    - Collapse multiple spaces
    
    Args:
        text: Raw transcription text
    
    Returns:
        Preprocessed transcription
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation (keep apostrophes for contractions like "don't")
    # Remove all standard punctuation and common special characters
    punct_pattern = r"[^\w\s']"
    text = re.sub(punct_pattern, " ", text)
    
    # Remove standalone apostrophes
    text = re.sub(r"(?<!\w)'|'(?!\w)", " ", text)
    
    # Collapse multiple spaces and strip
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds."""
    try:
        if SOUNDFILE_AVAILABLE:
            info = sf.info(audio_path)
            return info.duration
        elif LIBROSA_AVAILABLE:
            duration = librosa.get_duration(path=audio_path)
            return duration
        else:
            # Fallback: load and calculate
            import wave
            with wave.open(audio_path, 'r') as f:
                frames = f.getnframes()
                rate = f.getframerate()
                return frames / float(rate)
    except Exception as e:
        print(f"[WARN] Could not get duration for {audio_path}: {e}")
        return 0.0


def download_and_extract_audio(
    language: str,
    split: str,
    output_dir: Path,
    cache_dir: Optional[Path] = None,
) -> Path:
    """
    Download and extract audio files for a specific language and split.
    
    Args:
        language: Language code (da, en, nl, etc.)
        split: Data split (train, dev, test)
        output_dir: Output directory for extracted audio
        cache_dir: Cache directory for downloaded files
    
    Returns:
        Path to extracted audio directory
    """
    if not HF_HUB_AVAILABLE:
        raise ImportError("huggingface_hub is required")
    
    # Audio files are stored as: audio/{lang}/{split}/{lang}_{split}_0.tar
    # e.g., audio/da/train/da_train_0.tar
    audio_filename = f"audio/{language}/{split}/{language}_{split}_0.tar"
    
    print(f"[INFO] Downloading audio: {audio_filename}")
    
    try:
        tar_path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=audio_filename,
            repo_type="dataset",
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    except Exception as e:
        print(f"[ERROR] Failed to download {audio_filename}: {e}")
        raise
    
    # Extract tar file
    audio_dir = output_dir / "audio" / split
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Extracting audio to {audio_dir}")
    
    with tarfile.open(tar_path, "r") as tar:
        # Extract all files
        for member in tqdm(tar.getmembers(), desc=f"Extracting {split}"):
            if member.isfile():
                # Extract to flat directory structure
                member.name = os.path.basename(member.name)
                tar.extract(member, path=audio_dir)
    
    return audio_dir


def download_transcript(
    language: str,
    split: str,
    cache_dir: Optional[Path] = None,
) -> Path:
    """
    Download transcript TSV file for a specific language and split.
    
    Args:
        language: Language code
        split: Data split
        cache_dir: Cache directory
    
    Returns:
        Path to downloaded TSV file
    """
    if not HF_HUB_AVAILABLE:
        raise ImportError("huggingface_hub is required")
    
    # Transcript files are stored as: transcript/{lang}/{split}.tsv
    transcript_filename = f"transcript/{language}/{split}.tsv"
    
    print(f"[INFO] Downloading transcript: {transcript_filename}")
    
    try:
        tsv_path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=transcript_filename,
            repo_type="dataset",
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        return Path(tsv_path)
    except Exception as e:
        print(f"[ERROR] Failed to download {transcript_filename}: {e}")
        raise


def load_transcript_tsv(tsv_path: Path) -> List[Dict[str, Any]]:
    """
    Load transcript TSV file.
    
    Args:
        tsv_path: Path to TSV file
    
    Returns:
        List of sample dictionaries
    """
    samples = []
    
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            samples.append(dict(row))
    
    return samples


def process_dataset(
    language: str,
    split: str,
    output_dir: Path,
    min_duration: float = 0.5,
    max_duration: float = 30.0,
    cache_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Process a single split of the dataset.
    
    Args:
        language: Language code
        split: Data split (train, dev, test)
        output_dir: Output directory
        min_duration: Minimum audio duration (seconds)
        max_duration: Maximum audio duration (seconds)
        cache_dir: Cache directory
    
    Returns:
        List of processed samples
    """
    # Download transcript
    tsv_path = download_transcript(language, split, cache_dir)
    samples = load_transcript_tsv(tsv_path)
    print(f"[INFO] Loaded {len(samples)} samples from transcript")
    
    # Download and extract audio
    audio_dir = download_and_extract_audio(language, split, output_dir, cache_dir)
    
    # Process samples
    processed = []
    stats = {
        "total": len(samples),
        "valid": 0,
        "too_short": 0,
        "too_long": 0,
        "missing_audio": 0,
        "empty_text": 0,
    }
    
    print(f"[INFO] Processing {len(samples)} samples...")
    
    for sample in tqdm(samples, desc=f"Processing {split}"):
        # Get audio path (path field contains filename like "common_voice_da_12345.mp3")
        audio_filename = sample.get("path", "")
        if not audio_filename:
            stats["missing_audio"] += 1
            continue
        
        # Audio files are extracted as .mp3
        audio_path = audio_dir / audio_filename
        
        if not audio_path.exists():
            stats["missing_audio"] += 1
            continue
        
        # Get raw transcription
        raw_transcription = sample.get("sentence", "").strip()
        
        if not raw_transcription:
            stats["empty_text"] += 1
            continue
        
        # Get audio duration
        duration = get_audio_duration(str(audio_path))
        
        # Filter by duration
        if duration < min_duration:
            stats["too_short"] += 1
            continue
        
        if duration > max_duration:
            stats["too_long"] += 1
            continue
        
        # Preprocess transcription
        preprocessed_transcription = preprocess_transcription(raw_transcription)
        
        # Create processed sample
        processed_sample = {
            "audio_path": str(audio_path.resolve()),
            "raw_transcription": raw_transcription,
            "transcription": preprocessed_transcription,  # Preprocessed version
            "duration": round(duration, 3),
            "split": split,
            "language": language,
            "client_id": sample.get("client_id", ""),
            "age": sample.get("age", ""),
            "gender": sample.get("gender", ""),
            "accent": sample.get("accent", ""),
        }
        
        processed.append(processed_sample)
        stats["valid"] += 1
    
    # Print stats
    print(f"\n[STATS] {split}:")
    print(f"  Total samples: {stats['total']}")
    print(f"  Valid samples: {stats['valid']}")
    print(f"  Too short (<{min_duration}s): {stats['too_short']}")
    print(f"  Too long (>{max_duration}s): {stats['too_long']}")
    print(f"  Missing audio: {stats['missing_audio']}")
    print(f"  Empty text: {stats['empty_text']}")
    
    return processed


def save_jsonl(samples: List[Dict[str, Any]], output_path: Path):
    """Save samples to JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"[INFO] Saved {len(samples)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and preprocess Common Voice 22.0 from HuggingFace"
    )
    parser.add_argument(
        "--language", 
        type=str, 
        default="da",
        help="Language code (da=Danish, en=English, nl=Dutch, etc.)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cv22",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "dev", "test"],
        help="Splits to download (train, dev, test)"
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=0.5,
        help="Minimum audio duration in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=30.0,
        help="Maximum audio duration in seconds (default: 30.0)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for HuggingFace downloads"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    output_dir = Path(args.output_dir) / args.language
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    
    print("=" * 70)
    print("Common Voice 22.0 Dataset Download & Preprocessing")
    print("=" * 70)
    print(f"Repository: {DATASET_REPO}")
    print(f"Language: {args.language}")
    print(f"Splits: {args.splits}")
    print(f"Output: {output_dir}")
    print(f"Duration filter: [{args.min_duration}s, {args.max_duration}s]")
    print("=" * 70)
    
    # Process each split
    all_stats = {}
    
    for split in args.splits:
        print(f"\n{'=' * 70}")
        print(f"Processing split: {split}")
        print("=" * 70)
        
        try:
            samples = process_dataset(
                language=args.language,
                split=split,
                output_dir=output_dir,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
                cache_dir=cache_dir,
            )
            
            # Map dev -> validation for consistency
            output_split = "validation" if split == "dev" else split
            
            # Save to JSONL
            jsonl_path = output_dir / f"{output_split}.jsonl"
            save_jsonl(samples, jsonl_path)
            
            # Calculate total duration
            total_duration = sum(s["duration"] for s in samples)
            all_stats[split] = {
                "samples": len(samples),
                "hours": total_duration / 3600,
            }
            
        except Exception as e:
            print(f"[ERROR] Failed to process {split}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save dataset info
    info = {
        "dataset": DATASET_REPO,
        "language": args.language,
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
        "sample_rate": SAMPLE_RATE,
        "preprocessing": {
            "lowercase": True,
            "remove_punctuation": True,
        },
        "splits": all_stats,
    }
    
    info_path = output_dir / "dataset_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for split, stats in all_stats.items():
        print(f"  {split}: {stats['samples']} samples, {stats['hours']:.2f} hours")
    print(f"\nOutput directory: {output_dir}")
    print("=" * 70)
    
    # Print example of preprocessing
    print("\n[EXAMPLE] Preprocessing demonstration:")
    examples = [
        "Hej, hvordan har du det?",
        "Det er en god dag!",
        "Don't stop the music.",
        "Prijs: €50,00 (incl. BTW)",
    ]
    for ex in examples:
        print(f"  Raw:         '{ex}'")
        print(f"  Preprocessed: '{preprocess_transcription(ex)}'")
        print()


if __name__ == "__main__":
    main()
