"""
Dataset preparation script for Common Voice (any language).
Downloads from Mozilla Data Collective API, extracts, converts to WAV, creates JSONL files.

Usage:
    # First time setup:
    # 1. Create account at https://datacollective.mozillafoundation.org
    # 2. Get API key from https://datacollective.mozillafoundation.org/profile/credentials
    # 3. Set environment variable: set MDC_API_KEY=your_api_key
    
    python datamodule/get_dataset.py --language da --max_hours 5
    python datamodule/get_dataset.py --language da  # Use all available data
"""

import os
import csv
import json
import tarfile
import argparse
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import subprocess
import requests

# Dataset IDs for Common Voice languages on Mozilla Data Collective
# Find more at: https://datacollective.mozillafoundation.org/datasets
DATASET_IDS = {
    "da": "cmj8u3ozj005hnxxbpiiwy7ph",  # Danish
    # Add more language codes -> dataset IDs as needed
}

# Available splits in Common Voice
SPLITS = ["train", "validated", "test", "dev"]
SPLIT_MAPPING = {"validation": "dev", "dev": "dev", "validated": "validated", "train": "train", "test": "test"}


def parse_args():
    p = argparse.ArgumentParser(description="Download and prepare Common Voice dataset")
    p.add_argument("--language", type=str, default="da", help="Language code (e.g., 'da' for Danish)")
    p.add_argument("--root", type=str, default="data", help="Output directory for processed data")
    p.add_argument("--archive", type=str, default=None, 
                   help="Path to already downloaded .tar.gz file (skip download)")
    p.add_argument("--dataset_id", type=str, default=None,
                   help="Mozilla Data Collective dataset ID (overrides --language lookup)")
    p.add_argument("--max_hours", type=float, default=None, help="Limit total hours (None = use all)")
    p.add_argument("--sample_rate", type=int, default=16000, help="Target sample rate")
    return p.parse_args()


def download_from_mdc(dataset_id: str, output_dir: Path) -> Path:
    """Download dataset using Mozilla Data Collective REST API."""
    import requests
    
    api_key = os.environ.get("MDC_API_KEY")
    if not api_key:
        print("\n" + "="*70)
        print("API KEY REQUIRED")
        print("="*70)
        print("\nTo download programmatically, set up Mozilla Data Collective API:")
        print("1. Create account: https://datacollective.mozillafoundation.org")
        print("2. Accept dataset terms on the dataset page first!")
        print("3. Get API key: https://datacollective.mozillafoundation.org/profile/credentials")
        print("4. Set environment variable:")
        print("   Windows: $env:MDC_API_KEY = 'your_api_key'")
        print("   Linux/Mac: export MDC_API_KEY=your_api_key")
        print("="*70)
        raise RuntimeError("MDC_API_KEY environment variable not set")
    
    base_url = "https://datacollective.mozillafoundation.org/api"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    # Step 1: Get dataset info
    print(f"[INFO] Fetching dataset info for {dataset_id}...")
    resp = requests.get(f"{base_url}/datasets/{dataset_id}", headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to get dataset info: {resp.status_code} - {resp.text}")
    
    dataset_info = resp.json()
    print(f"[INFO] Dataset: {dataset_info.get('name', dataset_id)}")
    print(f"[INFO] Size: {int(dataset_info.get('sizeBytes', 0)) / 1024 / 1024:.1f} MB")
    
    # Step 2: Create download session
    print("[INFO] Creating download session...")
    resp = requests.post(f"{base_url}/datasets/{dataset_id}/download", headers=headers)
    if resp.status_code == 403:
        print("\n" + "="*70)
        print("TERMS AGREEMENT REQUIRED")
        print("="*70)
        print("\nYou must accept the dataset terms on the website first:")
        print(f"  https://datacollective.mozillafoundation.org/datasets/{dataset_id}")
        print("\nClick 'Download' and agree to the terms, then try again.")
        print("="*70)
        raise RuntimeError("Must agree to dataset terms on website first")
    elif resp.status_code != 200:
        raise RuntimeError(f"Failed to create download session: {resp.status_code} - {resp.text}")
    
    download_info = resp.json()
    download_url = download_info["downloadUrl"]
    filename = download_info.get("filename", f"{dataset_id}.tar.gz")
    
    # Step 3: Download the file
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    
    print(f"[INFO] Downloading to {output_path}...")
    resp = requests.get(download_url, headers=headers, stream=True)
    if resp.status_code != 200:
        raise RuntimeError(f"Download failed: {resp.status_code}")
    
    total_size = int(resp.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f:
        downloaded = 0
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = downloaded * 100 // total_size
                print(f"\r[INFO] Progress: {pct}% ({downloaded // 1024 // 1024} MB)", end="", flush=True)
    
    print(f"\n[INFO] Downloaded: {output_path}")
    return output_path


def find_archive(root: str, language: str) -> Path:
    """Find Common Voice archive in the data directory."""
    root_path = Path(root)
    
    # Look for common patterns
    patterns = [
        f"cv-corpus-*-{language}.tar.gz",
        f"*common*voice*{language}*.tar.gz",
        f"cv-*-{language}.tar.gz",
        f"mcv-*-{language}*.tar.gz",
    ]
    
    for pattern in patterns:
        matches = list(root_path.glob(pattern))
        if matches:
            return matches[0]
    
    # Also check current directory
    for pattern in patterns:
        matches = list(Path(".").glob(pattern))
        if matches:
            return matches[0]
    
    return None


def extract_archive(archive_path: Path, output_dir: Path) -> Path:
    """Extract tar.gz archive and return the extracted directory."""
    print(f"[INFO] Extracting {archive_path}...")
    
    with tarfile.open(archive_path, "r:gz") as tar:
        # Get the top-level directory name
        members = tar.getmembers()
        if members:
            top_dir = members[0].name.split('/')[0]
        
        # Extract all
        tar.extractall(path=output_dir)
    
    extracted_path = output_dir / top_dir
    print(f"[INFO] Extracted to {extracted_path}")
    return extracted_path


def load_tsv_split(tsv_path: Path) -> list:
    """Load a TSV file and return list of samples."""
    samples = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            samples.append(row)
    return samples


def convert_mp3_to_wav(mp3_path: Path, wav_path: Path, sample_rate: int = 16000):
    """Convert MP3 to WAV using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", str(mp3_path),
        "-ar", str(sample_rate),
        "-ac", "1",
        "-loglevel", "error",
        str(wav_path)
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def process_split(samples: list, split_name: str, clips_dir: Path, output_dir: Path,
                  sample_rate: int, max_samples: int = None):
    """Process a single split: convert audio and create JSONL."""
    
    audio_dir = output_dir / "audio" / split_name
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Map split names for output
    output_split = "validation" if split_name == "dev" else split_name
    jsonl_path = output_dir / f"{output_split}.jsonl"
    
    n_total = len(samples)
    n_process = n_total if max_samples is None else min(max_samples, n_total)
    
    print(f"[INFO] Processing {split_name}: {n_process}/{n_total} samples")
    
    entries = []
    total_duration = 0.0
    skipped = 0
    
    for idx in tqdm(range(n_process), desc=f"Processing {split_name}"):
        sample = samples[idx]
        
        mp3_filename = sample.get("path", "")
        sentence = sample.get("sentence", "")
        client_id = sample.get("client_id", f"speaker_{idx}")
        
        if not mp3_filename or not sentence:
            skipped += 1
            continue
        
        mp3_path = clips_dir / mp3_filename
        if not mp3_path.exists():
            skipped += 1
            continue
        
        # Convert MP3 to WAV
        wav_filename = f"{split_name}_{idx:06d}.wav"
        wav_path = audio_dir / wav_filename
        
        try:
            convert_mp3_to_wav(mp3_path, wav_path, sample_rate)
        except Exception as e:
            skipped += 1
            continue
        
        # Get duration from the saved WAV
        try:
            audio_data, sr = sf.read(str(wav_path))
            duration = len(audio_data) / sr
        except:
            duration = 0.0
        
        total_duration += duration
        
        # Create entry
        entry = {
            "source": str(wav_path.resolve()),
            "target": sentence.strip(),
            "duration": round(duration, 2),
            "key": f"{split_name}_{idx:06d}",
            "speaker_id": client_id[:8] if client_id else "unknown"
        }
        entries.append(entry)
    
    # Write JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    hours = total_duration / 3600
    print(f"[DONE] {split_name}: {len(entries)} samples ({skipped} skipped), {hours:.2f} hours -> {jsonl_path}")
    
    return len(entries), total_duration


def main():
    args = parse_args()
    
    output_dir = Path(args.root) / f"common_voice_{args.language}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Common Voice Dataset Preparation")
    print("="*70)
    print(f"[INFO] Language: {args.language}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Max hours: {args.max_hours if args.max_hours else 'all available'}")
    
    # Determine archive path
    archive_path = None
    
    if args.archive:
        # User provided archive path
        archive_path = Path(args.archive)
        if not archive_path.exists():
            print(f"[ERROR] Archive not found: {archive_path}")
            return
    else:
        # Try to find existing archive
        archive_path = find_archive(args.root, args.language)
        
        if archive_path is None:
            # Try to download via API
            dataset_id = args.dataset_id or DATASET_IDS.get(args.language)
            
            if dataset_id:
                print(f"[INFO] No local archive found. Attempting API download...")
                try:
                    archive_path = download_from_mdc(dataset_id, Path(args.root))
                except Exception as e:
                    print(f"[ERROR] Download failed: {e}")
                    return
            else:
                print("\n" + "="*70)
                print("DATASET NOT FOUND")
                print("="*70)
                print(f"\nNo dataset ID configured for language '{args.language}'")
                print("Options:")
                print("  1. Find dataset ID at: https://datacollective.mozillafoundation.org/datasets")
                print(f"     Then run: python datamodule/get_dataset.py --dataset_id YOUR_ID")
                print("  2. Download manually and use --archive flag")
                print("="*70)
                return
    
    print(f"[INFO] Found archive: {archive_path}")
    
    # Extract archive
    extracted_dir = extract_archive(archive_path, Path(args.root))
    
    # Find the language folder inside extracted dir
    lang_dir = extracted_dir / args.language
    if not lang_dir.exists():
        # Try without language subfolder
        lang_dir = extracted_dir
    
    clips_dir = lang_dir / "clips"
    if not clips_dir.exists():
        print(f"[ERROR] Clips directory not found: {clips_dir}")
        print("[INFO] Looking for clips in extracted directory...")
        # Search for clips folder
        for p in extracted_dir.rglob("clips"):
            if p.is_dir():
                clips_dir = p
                lang_dir = p.parent
                print(f"[INFO] Found clips at: {clips_dir}")
                break
    
    # Process each split
    splits_to_process = ["train", "dev", "test"]  # CV uses 'dev' not 'validation'
    
    total_samples = 0
    total_duration = 0.0
    
    for split_name in splits_to_process:
        tsv_path = lang_dir / f"{split_name}.tsv"
        if not tsv_path.exists():
            print(f"[WARN] {tsv_path} not found, skipping {split_name}")
            continue
        
        # Load TSV
        samples = load_tsv_split(tsv_path)
        print(f"[INFO] Loaded {len(samples)} samples from {tsv_path.name}")
        
        # Calculate max samples for train split
        max_samples = None
        if args.max_hours and split_name == "train":
            # Estimate: ~5 seconds average per clip
            estimated_samples = int(args.max_hours * 3600 / 5)
            max_samples = min(estimated_samples, len(samples))
            print(f"[INFO] Limiting train to ~{max_samples} samples for {args.max_hours}h")
        
        n_samples, duration = process_split(
            samples,
            split_name,
            clips_dir,
            output_dir,
            args.sample_rate,
            max_samples
        )
        total_samples += n_samples
        total_duration += duration
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Language: {args.language}")
    print(f"  Total samples: {total_samples}")
    print(f"  Total duration: {total_duration/3600:.2f} hours")
    print(f"  Output: {output_dir}")
    print("="*70)
    
    # Create dataset info
    info_path = output_dir / "dataset_info.json"
    with open(info_path, "w") as f:
        json.dump({
            "language": args.language,
            "total_samples": total_samples,
            "total_hours": round(total_duration/3600, 2),
            "sample_rate": args.sample_rate,
        }, f, indent=2)
    
    print(f"\n[NEXT STEPS]")
    print(f"  1. Update config paths:")
    print(f"     train_data_path: {output_dir}/train.jsonl")
    print(f"     val_data_path: {output_dir}/validation.jsonl") 
    print(f"     test_data_path: {output_dir}/test.jsonl")
    print(f"  2. Run training: python train.py --config configs/config_danish.yaml")


if __name__ == "__main__":
    main()
