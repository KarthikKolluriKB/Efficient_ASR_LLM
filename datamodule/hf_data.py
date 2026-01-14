"""
Main script to download, preprocess, and save Common Voice dataset as HuggingFace Dataset.

This script orchestrates:
1. Downloading transcript and audio files from HuggingFace
2. Preprocessing audio (loading, resampling) and text (lowercase, no punctuation)
3. Filtering by duration
4. Saving as HuggingFace Dataset with pre-computed audio arrays

Usage:
    python -m datamodule.hf_data --language da --output-dir data/cv22_hf
    python -m datamodule.hf_data --language en --output-dir data/cv22_hf
    python -m datamodule.hf_data --language nl --output-dir data/cv22_hf
"""

from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict

from datamodule.download_data import download_transcript, download_and_extract_audio
from datamodule.preprocess_data import load_transcript, process_split


# Default configuration
DEFAULT_SPLITS = ["train", "dev", "test"]
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MIN_DURATION = 0.5   # seconds
DEFAULT_MAX_DURATION = 30.0  # seconds


def prepare_dataset(
    language: str,
    output_dir: Path,
    splits: Optional[list[str]] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
    repo_id: str = "fsicoli/common_voice_22_0",
) -> DatasetDict:
    """
    Download, preprocess, and create HuggingFace DatasetDict for a language.
    
    Args:
        language: Language code (e.g., 'da', 'en', 'nl')
        output_dir: Base output directory
        splits: List of splits to process (default: ['train', 'dev', 'test'])
        sample_rate: Target audio sample rate
        min_duration: Minimum audio duration in seconds
        max_duration: Maximum audio duration in seconds
        repo_id: HuggingFace dataset repository ID
        
    Returns:
        HuggingFace DatasetDict with all splits
    """
    if splits is None:
        splits = DEFAULT_SPLITS
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Temporary directory for extracted audio
    temp_dir = output_dir / "_temp" / language
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    datasets_dict = {}
    total_stats = {"total": 0, "valid": 0}
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing: {language} - {split}")
        print('='*60)
        
        # Download transcript
        transcript_path = download_transcript(language, split, repo_id)
        transcript = load_transcript(transcript_path)
        print(f"[Transcript] Loaded {len(transcript)} entries")
        
        # Download and extract audio
        audio_dir = download_and_extract_audio(language, split, temp_dir, repo_id)
        
        # Process samples
        print(f"\n[Processing] Processing {len(transcript)} samples...")
        processed_samples, stats = process_split(
            transcript,
            audio_dir,
            target_sr=sample_rate,
            min_duration=min_duration,
            max_duration=max_duration,
        )
        
        # Print stats
        print(f"\n[Stats] {split}:")
        print(f"  Total: {stats['total']}")
        print(f"  Valid: {stats['valid']}")
        print(f"  Too short (<{min_duration}s): {stats['too_short']}")
        print(f"  Too long (>{max_duration}s): {stats['too_long']}")
        print(f"  Missing/Error: {stats['missing']}")
        print(f"  Empty text: {stats['empty']}")
        
        total_hours = sum(s["duration"] for s in processed_samples) / 3600
        print(f"  Total duration: {total_hours:.2f} hours")
        
        # Create HuggingFace Dataset
        dataset = Dataset.from_list(processed_samples)
        
        # Rename 'dev' to 'validation' for consistency
        split_name = "validation" if split == "dev" else split
        datasets_dict[split_name] = dataset
        
        total_stats["total"] += stats["total"]
        total_stats["valid"] += stats["valid"]
    
    print(f"\n{'='*60}")
    print(f"All splits processed for {language}!")
    print('='*60)
    for name, ds in datasets_dict.items():
        print(f"  {name}: {len(ds)} samples")
    
    return DatasetDict(datasets_dict)


def save_dataset(
    dataset_dict: DatasetDict,
    output_dir: Path,
    language: str,
) -> Path:
    """
    Save DatasetDict to disk.
    
    Args:
        dataset_dict: HuggingFace DatasetDict to save
        output_dir: Base output directory
        language: Language code for subdirectory
        
    Returns:
        Path where dataset was saved
    """
    save_path = Path(output_dir) / language
    print(f"\n[Save] Saving dataset to: {save_path}")
    
    dataset_dict.save_to_disk(str(save_path))
    
    print(f"[Save] Dataset saved successfully!")
    return save_path


def prepare_and_save(
    language: str,
    output_dir: str = "data/cv22_hf",
    splits: Optional[list[str]] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
) -> Path:
    """
    Convenience function to prepare and save dataset in one call.
    
    Args:
        language: Language code (e.g., 'da', 'en', 'nl')
        output_dir: Output directory path
        splits: List of splits to process
        sample_rate: Target audio sample rate
        min_duration: Minimum audio duration
        max_duration: Maximum audio duration
        
    Returns:
        Path where dataset was saved
    """
    output_path = Path(output_dir)
    
    # Prepare dataset
    dataset_dict = prepare_dataset(
        language=language,
        output_dir=output_path,
        splits=splits,
        sample_rate=sample_rate,
        min_duration=min_duration,
        max_duration=max_duration,
    )
    
    # Save dataset
    save_path = save_dataset(dataset_dict, output_path, language)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Language: {language}")
    print(f"Output: {save_path.absolute()}")
    print(f"\nDataset features:")
    print(f"  - audio_array: Pre-computed audio as float32 array")
    print(f"  - sampling_rate: {sample_rate} Hz")
    print(f"  - raw_transcription: Original text (with punctuation)")
    print(f"  - transcription: Preprocessed text (lowercase, no punctuation)")
    print(f"  - duration: Audio duration in seconds")
    print(f"  - speaker_id: Anonymized speaker ID")
    print(f"\nTo load:")
    print(f"  from datasets import load_from_disk")
    print(f"  dataset = load_from_disk('{save_path}')")
    
    return save_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download and preprocess Common Voice dataset"
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default="da",
        help="Language code (da=Danish, en=English, nl=Dutch, etc.)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="data/cv22_hf",
        help="Output directory for HuggingFace dataset"
    )
    parser.add_argument(
        "--splits", "-s",
        type=str,
        nargs="+",
        default=["train", "dev", "test"],
        help="Splits to process (default: train dev test)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target audio sample rate (default: 16000)"
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
    
    args = parser.parse_args()
    
    prepare_and_save(
        language=args.language,
        output_dir=args.output_dir,
        splits=args.splits,
        sample_rate=args.sample_rate,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )
