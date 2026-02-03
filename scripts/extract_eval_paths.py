"""
Extract evaluation output paths from config files.

Scans a folder for YAML config files, extracts eval.output_dir field,
and saves paths to a text file for use with eval_bertscore_batch.py.

Usage:
    python extract_eval_paths.py --input_folder configs/whisper/danish --output_file whisper_small_danish
    python extract_eval_paths.py -i configs/whisper/dutch -o whisper_small_dutch
"""

import argparse
import os
import glob
from typing import List, Tuple

try:
    import yaml
except ImportError:
    print("Please install PyYAML: pip install pyyaml")
    exit(1)


def find_config_files(folder_path: str) -> List[str]:
    """Find all YAML config files in a folder."""
    patterns = ["*.yaml", "*.yml"]
    config_files = []
    
    for pattern in patterns:
        search_path = os.path.join(folder_path, pattern)
        config_files.extend(glob.glob(search_path))
    
    return sorted(config_files)


def extract_output_dir(config_path: str) -> Tuple[str, str]:
    """
    Extract eval.output_dir from a config file.
    
    Returns:
        Tuple of (config_filename, output_dir) or (config_filename, None) if not found
    """
    config_name = os.path.basename(config_path)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        
        # Try to get eval.output_dir
        if cfg and 'eval' in cfg and 'output_dir' in cfg['eval']:
            output_dir = cfg['eval']['output_dir']
            return config_name, output_dir
        else:
            return config_name, None
            
    except Exception as e:
        print(f"  ⚠️  Error loading {config_name}: {e}")
        return config_name, None


def save_paths_to_file(paths: List[str], output_file: str):
    """Save paths to text file (one per line)."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for path in paths:
            f.write(path + '\n')


def main():
    parser = argparse.ArgumentParser(
        description="Extract eval output paths from config files"
    )
    parser.add_argument(
        "--input_folder", "-i",
        type=str,
        required=True,
        help="Folder containing YAML config files"
    )
    parser.add_argument(
        "--output_file", "-o",
        type=str,
        required=True,
        help="Output text file for paths (will add .txt if not present)"
    )
    parser.add_argument(
        "--filename", "-f",
        type=str,
        default="test_examples.jsonl",
        help="Filename to append to output_dir (default: test_examples.jsonl)"
    )
    
    args = parser.parse_args()
    
    # Ensure output file has .txt extension
    if not args.output_file.endswith('.txt'):
        args.output_file = args.output_file + '.txt'
    
    # Validate input folder
    if not os.path.isdir(args.input_folder):
        print(f"Error: Folder not found: {args.input_folder}")
        exit(1)
    
    # Find config files
    print(f"\nScanning: {args.input_folder}")
    config_files = find_config_files(args.input_folder)
    
    if not config_files:
        print("❌ No YAML config files found!")
        exit(1)
    
    print(f"Found {len(config_files)} config files\n")
    
    # Extract output_dir from each config
    paths = []
    print(f"{'Config File':<40} {'Status':<10} {'Output Path'}")
    print("-" * 100)
    
    for config_path in config_files:
        config_name, output_dir = extract_output_dir(config_path)
        
        if output_dir:
            # Append filename to output_dir
            full_path = os.path.join(output_dir, args.filename)
            paths.append(full_path)
            print(f"{config_name:<40} {'✅':<10} {full_path}")
        else:
            print(f"{config_name:<40} {'❌':<10} eval.output_dir not found")
    
    if not paths:
        print("\n❌ No valid paths extracted!")
        exit(1)
    
    # Save to output file
    save_paths_to_file(paths, args.output_file)
    
    print("\n" + "=" * 60)
    print(f"✅ Saved {len(paths)} paths to: {args.output_file}")
    print("=" * 60)
    
    # Preview output file
    print(f"\nContents of {args.output_file}:")
    print("-" * 60)
    with open(args.output_file, 'r') as f:
        for line in f:
            print(f"  {line.strip()}")


if __name__ == "__main__":
    main()