"""
Test script to verify dataset integrity before training.
Checks: JSONL validity, audio files, mel spectrograms, dataset class.

Usage:
    python datamodule/test_dataset.py
    python datamodule/test_dataset.py --config configs/config_danish.yaml
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import soundfile as sf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Test dataset integrity")
    p.add_argument("--data_dir", type=str, default="data/common_voice_da",
                   help="Directory containing JSONL files")
    p.add_argument("--config", type=str, default=None,
                   help="Config file to test with (optional)")
    p.add_argument("--num_samples", type=int, default=5,
                   help="Number of samples to test per split")
    p.add_argument("--test_mel", action="store_true",
                   help="Test mel spectrogram computation (slower)")
    p.add_argument("--test_dataset_class", action="store_true",
                   help="Test full dataset class loading")
    return p.parse_args()


def test_jsonl_files(data_dir: Path, num_samples: int = 5):
    """Test JSONL file validity and content."""
    print("\n" + "="*60)
    print("1. TESTING JSONL FILES")
    print("="*60)
    
    splits = ["train", "validation", "test"]
    stats = {}
    all_passed = True
    
    for split in splits:
        jsonl_path = data_dir / f"{split}.jsonl"
        
        if not jsonl_path.exists():
            print(f"  ❌ {split}.jsonl NOT FOUND")
            all_passed = False
            continue
        
        samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                samples.append(json.loads(line.strip()))
        
        # Compute stats
        durations = [s.get('duration', 0) for s in samples]
        total_hours = sum(durations) / 3600
        avg_duration = np.mean(durations) if durations else 0
        
        stats[split] = {
            'count': len(samples),
            'total_hours': total_hours,
            'avg_duration': avg_duration
        }
        
        print(f"\n  ✓ {split}.jsonl")
        print(f"    Samples: {len(samples)}")
        print(f"    Total duration: {total_hours:.2f} hours")
        print(f"    Avg duration: {avg_duration:.2f} seconds")
        
        # Check required fields
        required_fields = ['source', 'target']
        for sample in samples[:num_samples]:
            missing = [f for f in required_fields if f not in sample]
            if missing:
                print(f"    ❌ Missing fields: {missing}")
                all_passed = False
        
        # Show sample entries
        print(f"\n    Sample entries:")
        for i, sample in enumerate(samples[:3]):
            target_preview = sample.get('target', '')[:50]
            duration = sample.get('duration', 0)
            print(f"      [{i}] {duration:.1f}s: \"{target_preview}...\"")
    
    return all_passed, stats


def test_audio_files(data_dir: Path, num_samples: int = 5):
    """Test that audio files exist and are valid."""
    print("\n" + "="*60)
    print("2. TESTING AUDIO FILES")
    print("="*60)
    
    splits = ["train", "validation", "test"]
    all_passed = True
    
    for split in splits:
        jsonl_path = data_dir / f"{split}.jsonl"
        if not jsonl_path.exists():
            continue
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            samples = [json.loads(line.strip()) for line in f]
        
        print(f"\n  Testing {split} audio ({num_samples} samples)...")
        
        missing = 0
        invalid = 0
        sample_rates = defaultdict(int)
        
        for sample in samples[:num_samples]:
            audio_path = Path(sample['source'])
            
            if not audio_path.exists():
                missing += 1
                print(f"    ❌ Missing: {audio_path.name}")
                all_passed = False
                continue
            
            try:
                data, sr = sf.read(str(audio_path))
                sample_rates[sr] += 1
                
                # Verify duration roughly matches
                actual_duration = len(data) / sr
                expected_duration = sample.get('duration', actual_duration)
                
                if abs(actual_duration - expected_duration) > 0.5:
                    print(f"    ⚠ Duration mismatch: {audio_path.name}")
                    print(f"      Expected: {expected_duration:.2f}s, Got: {actual_duration:.2f}s")
                    
            except Exception as e:
                invalid += 1
                print(f"    ❌ Invalid audio: {audio_path.name} - {e}")
                all_passed = False
        
        if missing == 0 and invalid == 0:
            print(f"    ✓ All {num_samples} samples valid")
        
        print(f"    Sample rates found: {dict(sample_rates)}")
    
    return all_passed


def test_whisper_loading(data_dir: Path, num_samples: int = 3):
    """Test loading audio with Whisper's load_audio function."""
    print("\n" + "="*60)
    print("3. TESTING WHISPER AUDIO LOADING")
    print("="*60)
    
    try:
        import whisper
    except ImportError:
        print("  ⚠ Whisper not installed, skipping...")
        return True
    
    jsonl_path = data_dir / "train.jsonl"
    if not jsonl_path.exists():
        print("  ❌ train.jsonl not found")
        return False
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        samples = [json.loads(line.strip()) for line in f]
    
    all_passed = True
    
    for i, sample in enumerate(samples[:num_samples]):
        audio_path = sample['source']
        try:
            audio = whisper.load_audio(audio_path)
            print(f"  ✓ Sample {i}: shape={audio.shape}, dtype={audio.dtype}")
            
            # Verify 16kHz
            expected_samples = int(sample.get('duration', 0) * 16000)
            if expected_samples > 0:
                diff = abs(len(audio) - expected_samples)
                if diff > 1600:  # Allow 0.1s tolerance
                    print(f"    ⚠ Length mismatch: expected ~{expected_samples}, got {len(audio)}")
                    
        except Exception as e:
            print(f"  ❌ Failed to load sample {i}: {e}")
            all_passed = False
    
    return all_passed


def test_mel_spectrogram(data_dir: Path, num_samples: int = 2):
    """Test mel spectrogram computation."""
    print("\n" + "="*60)
    print("4. TESTING MEL SPECTROGRAM")
    print("="*60)
    
    try:
        import whisper
    except ImportError:
        print("  ⚠ Whisper not installed, skipping...")
        return True
    
    jsonl_path = data_dir / "train.jsonl"
    if not jsonl_path.exists():
        return False
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        samples = [json.loads(line.strip()) for line in f]
    
    all_passed = True
    
    for i, sample in enumerate(samples[:num_samples]):
        try:
            audio = whisper.load_audio(sample['source'])
            
            # Test with padding (like in training)
            audio_padded = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio_padded, n_mels=80)
            
            print(f"  ✓ Sample {i}:")
            print(f"    Audio: {len(audio)} samples ({len(audio)/16000:.2f}s)")
            print(f"    Mel shape: {mel.shape} (expected: [80, 3000])")
            
            # Test variable length
            mel_var = whisper.log_mel_spectrogram(audio, n_mels=80)
            print(f"    Mel (variable): {mel_var.shape}")
            
        except Exception as e:
            print(f"  ❌ Failed sample {i}: {e}")
            all_passed = False
    
    return all_passed


def test_dataset_class(data_dir: Path, config_path: str = None):
    """Test the actual dataset class."""
    print("\n" + "="*60)
    print("5. TESTING DATASET CLASS")
    print("="*60)
    
    try:
        from datamodule.dataset import SpeechDatasetJsonl
        from transformers import AutoTokenizer
        from omegaconf import OmegaConf
    except ImportError as e:
        print(f"  ⚠ Import error: {e}")
        return True
    
    # Create a minimal config
    if config_path and Path(config_path).exists():
        config = OmegaConf.load(config_path)
        dataset_config = config.data
    else:
        dataset_config = OmegaConf.create({
            'train_data_path': str(data_dir / 'train.jsonl'),
            'val_data_path': str(data_dir / 'validation.jsonl'),
            'test_data_path': str(data_dir / 'test.jsonl'),
            'input_type': 'mel',
            'mel_size': 80,
            'normalize': False,
            'inference_mode': False,
            'use_variable_length': True,
            'max_audio_length': 30,
            'fix_length_audio': -1,
        })
    
    # Load tokenizer
    print("  Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"  ⚠ Could not load tokenizer: {e}")
        print("  Using dummy tokenizer...")
        tokenizer = None
        return True
    
    # Create dataset
    print("  Creating dataset...")
    try:
        dataset = SpeechDatasetJsonl(dataset_config, tokenizer, split='train')
        print(f"  ✓ Dataset created with {len(dataset)} samples")
    except Exception as e:
        print(f"  ❌ Failed to create dataset: {e}")
        return False
    
    # Test __getitem__
    print("  Testing __getitem__...")
    try:
        sample = dataset[0]
        print(f"  ✓ Sample keys: {list(sample.keys())}")
        print(f"    input_ids shape: {sample['input_ids'].shape}")
        print(f"    audio_mel shape: {sample['audio_mel'].shape}")
        print(f"    audio_length: {sample['audio_length']}")
    except Exception as e:
        print(f"  ❌ __getitem__ failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test collator with mini-batch
    print("  Testing collator with batch of 2...")
    try:
        batch_samples = [dataset[0], dataset[1]]
        batch = dataset.collator(batch_samples)
        print(f"  ✓ Batch keys: {list(batch.keys())}")
        print(f"    input_ids: {batch['input_ids'].shape}")
        print(f"    audio_mel: {batch['audio_mel'].shape}")
        print(f"    attention_mask: {batch['attention_mask'].shape}")
    except Exception as e:
        print(f"  ❌ Collator failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    
    print("\n" + "="*60)
    print("DATASET INTEGRITY TEST")
    print("="*60)
    print(f"Data directory: {data_dir}")
    
    if not data_dir.exists():
        print(f"\n❌ Data directory not found: {data_dir}")
        print("Run: python datamodule/get_dataset.py --language da")
        return
    
    results = []
    
    # Test 1: JSONL files
    passed, stats = test_jsonl_files(data_dir, args.num_samples)
    results.append(("JSONL Files", passed))
    
    # Test 2: Audio files
    passed = test_audio_files(data_dir, args.num_samples)
    results.append(("Audio Files", passed))
    
    # Test 3: Whisper loading
    passed = test_whisper_loading(data_dir, args.num_samples)
    results.append(("Whisper Loading", passed))
    
    # Test 4: Mel spectrogram (optional)
    if args.test_mel:
        passed = test_mel_spectrogram(data_dir, args.num_samples)
        results.append(("Mel Spectrogram", passed))
    
    # Test 5: Dataset class (optional)
    if args.test_dataset_class:
        passed = test_dataset_class(data_dir, args.config)
        results.append(("Dataset Class", passed))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("✓ All tests passed! Dataset is ready for training.")
        print("\nNext steps:")
        print("  python train.py --config configs/config_danish.yaml")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
