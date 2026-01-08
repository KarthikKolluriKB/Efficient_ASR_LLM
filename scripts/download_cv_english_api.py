#!/usr/bin/env python3
"""
Download Common Voice Scripted Speech English from Mozilla Data Collective
===========================================================================

Usage:
    python scripts/download_cv_english_api.py YOUR_API_KEY

To get your API key:
    1. Go to: https://datacollective.mozillafoundation.org
    2. Create an account / Sign in
    3. Go to Account Settings -> API Keys -> Create new key
"""

import os
import sys
import json
import urllib.request
import urllib.error
import tarfile
from pathlib import Path

# Configuration
DATASET_ID = "cmj8u3p1w0075nxxbe8bedl00"
OUTPUT_DIR = Path("data")
ARCHIVE_NAME = "cv-scripted-en.tar.gz"

def main():
    if len(sys.argv) < 2:
        print("\nUsage: python scripts/download_cv_english_api.py API_KEY\n")
        print("To get your API key:")
        print("  1. Go to: https://datacollective.mozillafoundation.org")
        print("  2. Create an account / Sign in")
        print("  3. Go to Account Settings -> API Keys -> Create new key")
        sys.exit(1)
    
    api_key = sys.argv[1]
    
    print("=" * 60)
    print("Download Common Voice Scripted Speech - English")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = OUTPUT_DIR / ARCHIVE_NAME
    
    # Step 1: Get download token
    print("\n[Step 1/4] Getting download token...")
    
    token_url = f"https://datacollective.mozillafoundation.org/api/datasets/{DATASET_ID}/download"
    
    req = urllib.request.Request(
        token_url,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        data=b""
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            response_data = json.loads(response.read().decode())
            print(f"Response: {response_data}")
    except urllib.error.HTTPError as e:
        print(f"ERROR: HTTP {e.code} - {e.reason}")
        print(f"Response: {e.read().decode()}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Extract download token
    download_token = response_data.get("downloadToken") or response_data.get("token")
    
    if not download_token:
        print(f"\nERROR: Could not find download token in response")
        print(f"Response keys: {list(response_data.keys())}")
        print(f"\nFull response: {json.dumps(response_data, indent=2)}")
        sys.exit(1)
    
    print(f"Got download token: {download_token[:20]}...")
    
    # Step 2: Download the file
    print("\n[Step 2/4] Downloading Common Voice Scripted Speech English...")
    print("This may take a while...")
    
    download_url = f"https://datacollective.mozillafoundation.org/api/datasets/{DATASET_ID}/download/{download_token}"
    
    req = urllib.request.Request(
        download_url,
        method="GET",
        headers={"Authorization": f"Bearer {api_key}"}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            block_size = 8192
            
            with open(archive_path, 'wb') as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    f.write(buffer)
                    
                    # Progress
                    if total_size > 0:
                        percent = downloaded / total_size * 100
                        mb_down = downloaded / 1024 / 1024
                        mb_total = total_size / 1024 / 1024
                        print(f"\r  {mb_down:.1f}/{mb_total:.1f} MB ({percent:.1f}%)", end="", flush=True)
                    else:
                        mb_down = downloaded / 1024 / 1024
                        print(f"\r  {mb_down:.1f} MB downloaded", end="", flush=True)
            
            print()  # Newline after progress
            
    except urllib.error.HTTPError as e:
        print(f"\nERROR: HTTP {e.code} - {e.reason}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    
    file_size = archive_path.stat().st_size / 1024 / 1024
    print(f"Download complete: {file_size:.1f} MB")
    
    # Step 3: Extract
    print("\n[Step 3/4] Extracting archive...")
    
    try:
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(path=OUTPUT_DIR)
        print("Extraction complete!")
    except Exception as e:
        print(f"ERROR extracting: {e}")
        print("You may need to extract manually:")
        print(f"  cd {OUTPUT_DIR} && tar -xzf {ARCHIVE_NAME}")
        sys.exit(1)
    
    # Step 4: Find extracted folder
    print("\n[Step 4/4] Finding extracted folder...")
    
    # List contents
    print(f"\nContents of {OUTPUT_DIR}:")
    for item in sorted(OUTPUT_DIR.iterdir()):
        if item.is_dir():
            print(f"  📁 {item.name}/")
        else:
            size_mb = item.stat().st_size / 1024 / 1024
            print(f"  📄 {item.name} ({size_mb:.1f} MB)")
    
    # Look for English data
    en_dirs = list(OUTPUT_DIR.glob("**/en")) + list(OUTPUT_DIR.glob("**/*english*"))
    if en_dirs:
        print(f"\nFound English data at: {en_dirs[0]}")
        
        # Check for TSV files
        train_tsv = en_dirs[0] / "train.tsv"
        if train_tsv.exists():
            with open(train_tsv, 'r') as f:
                train_count = sum(1 for _ in f) - 1  # Subtract header
            print(f"Train samples: {train_count}")
    
    # Cleanup prompt
    print(f"\nArchive file: {archive_path}")
    response = input("Delete archive to save space? (y/n): ").strip().lower()
    if response == 'y':
        archive_path.unlink()
        print("Archive deleted.")
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Check extracted folder structure above")
    print("  2. Update CV_CORPUS_DIR in scripts/download_cv_english.py")
    print("  3. Run: python scripts/download_cv_english.py")
    print("  4. Train: python train.py --config configs/train_config_english_test.yaml")
    print()

if __name__ == "__main__":
    main()
