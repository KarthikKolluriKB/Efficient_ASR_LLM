"""
Download Common Voice dataset from HuggingFace.

Downloads transcript TSV files and audio tar files from the 
fsicoli/common_voice_22_0 dataset repository.
"""

import os
import tarfile
from pathlib import Path
from tqdm.auto import tqdm

from huggingface_hub import hf_hub_download


# Default dataset repository
DEFAULT_REPO = "fsicoli/common_voice_22_0"


def download_transcript(
    language: str,
    split: str,
    repo_id: str = DEFAULT_REPO,
) -> Path:
    """
    Download transcript TSV file for a language and split.
    
    Args:
        language: Language code (e.g., 'da', 'en', 'nl')
        split: Dataset split ('train', 'dev', 'test')
        repo_id: HuggingFace dataset repository ID
        
    Returns:
        Path to downloaded TSV file
    """
    filename = f"transcript/{language}/{split}.tsv"
    print(f"[Download] Downloading transcript: {filename}")
    
    tsv_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
    )
    
    return Path(tsv_path)


def download_and_extract_audio(
    language: str,
    split: str,
    output_dir: Path,
    repo_id: str = DEFAULT_REPO,
) -> Path:
    """
    Download and extract audio tar file for a language and split.
    
    Args:
        language: Language code (e.g., 'da', 'en', 'nl')
        split: Dataset split ('train', 'dev', 'test')
        output_dir: Directory to extract audio files to
        repo_id: HuggingFace dataset repository ID
        
    Returns:
        Path to directory containing extracted audio files
    """
    # Audio format: audio/{lang}/{split}/{lang}_{split}_0.tar
    filename = f"audio/{language}/{split}/{language}_{split}_0.tar"
    print(f"[Download] Downloading audio: {filename}")
    
    tar_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
    )
    
    # Extract to output directory
    audio_dir = output_dir / "audio" / split
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Download] Extracting to: {audio_dir}")
    extracted_count = 0
    with tarfile.open(tar_path, "r") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc=f"Extracting {split}"):
            if member.isfile() and member.name.endswith('.mp3'):
                # Extract just the filename, not full path
                basename = os.path.basename(member.name)
                # Extract file content and write to destination
                dest_path = audio_dir / basename
                try:
                    with tar.extractfile(member) as src:
                        if src is not None:
                            with open(dest_path, 'wb') as dst:
                                dst.write(src.read())
                            extracted_count += 1
                except Exception as e:
                    print(f"[Warning] Failed to extract {member.name}: {e}")
    
    print(f"[Download] Extracted {extracted_count} audio files to {audio_dir}")
    
    # Verify extraction
    mp3_files = list(audio_dir.glob("*.mp3"))
    print(f"[Download] Verified {len(mp3_files)} .mp3 files in {audio_dir}")
    
    return audio_dir


def download_split(
    language: str,
    split: str,
    output_dir: Path,
    repo_id: str = DEFAULT_REPO,
) -> tuple[Path, Path]:
    """
    Download both transcript and audio for a split.
    
    Args:
        language: Language code (e.g., 'da', 'en', 'nl')
        split: Dataset split ('train', 'dev', 'test')
        output_dir: Directory to extract audio files to
        repo_id: HuggingFace dataset repository ID
        
    Returns:
        Tuple of (transcript_path, audio_dir)
    """
    transcript_path = download_transcript(language, split, repo_id)
    audio_dir = download_and_extract_audio(language, split, output_dir, repo_id)
    return transcript_path, audio_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Common Voice dataset")
    parser.add_argument("--language", "-l", type=str, default="da",
                        help="Language code (da, en, nl, etc.)")
    parser.add_argument("--split", "-s", type=str, default="train",
                        choices=["train", "dev", "test"],
                        help="Dataset split")
    parser.add_argument("--output-dir", "-o", type=str, default="data/cv22_raw",
                        help="Output directory for extracted files")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    transcript_path, audio_dir = download_split(
        args.language, args.split, output_dir
    )
    
    print(f"\nDownload complete!")
    print(f"  Transcript: {transcript_path}")
    print(f"  Audio dir: {audio_dir}")
