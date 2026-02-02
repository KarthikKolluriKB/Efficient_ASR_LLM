"""
Verify that saved evaluation outputs contain all test samples.

Checks:
1. Number of entries in JSONL matches expected test set size
2. No duplicate IDs
3. No empty hypotheses or references
4. All entries have required fields

Usage:
    python verify_eval_output.py --jsonl eval_results/test_examples.jsonl --config configs/eval.yaml
    python verify_eval_output.py --jsonl eval_results/test_examples.jsonl --expected 1000
"""

import argparse
import json
import os
from collections import Counter


def load_jsonl(filepath: str) -> list:
    """Load all entries from JSONL file."""
    entries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entry['_line_num'] = line_num
                entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"  ❌ Line {line_num}: JSON parse error - {e}")
    return entries


def get_expected_count_from_config(config_path: str) -> int:
    """Get expected test set size from config + dataset."""
    try:
        from omegaconf import OmegaConf
        from datasets import load_dataset
        
        cfg = OmegaConf.load(config_path)
        data_cfg = cfg.data
        
        # Load dataset to get actual count
        dataset = load_dataset(
            data_cfg.dataset_name,
            data_cfg.dataset_config,
            split="test",
            trust_remote_code=True,
        )
        return len(dataset)
    except Exception as e:
        print(f"  ⚠️  Could not load dataset from config: {e}")
        return None


def verify_output(jsonl_path: str, expected_count: int = None, config_path: str = None):
    """
    Verify the evaluation output file.
    
    Args:
        jsonl_path: Path to test_examples.jsonl
        expected_count: Expected number of test samples (optional)
        config_path: Path to config to auto-detect expected count (optional)
    """
    print("=" * 60)
    print("VERIFICATION REPORT")
    print("=" * 60)
    print(f"File: {jsonl_path}")
    print("-" * 60)
    
    # Check file exists
    if not os.path.exists(jsonl_path):
        print(f"❌ File not found: {jsonl_path}")
        return False
    
    # Load entries
    entries = load_jsonl(jsonl_path)
    num_entries = len(entries)
    print(f"Total entries loaded: {num_entries}")
    
    # Get expected count
    if expected_count is None and config_path:
        expected_count = get_expected_count_from_config(config_path)
    
    # Check 1: Count match
    print("\n[1] COUNT CHECK")
    if expected_count:
        if num_entries == expected_count:
            print(f"  ✅ Entry count matches expected: {num_entries} == {expected_count}")
        else:
            print(f"  ❌ Entry count mismatch: {num_entries} != {expected_count} (expected)")
            print(f"     Missing: {expected_count - num_entries} samples")
    else:
        print(f"  ⚠️  No expected count provided, found {num_entries} entries")
    
    # Check 2: Duplicate IDs
    print("\n[2] DUPLICATE ID CHECK")
    ids = [e.get("id") for e in entries]
    id_counts = Counter(ids)
    duplicates = {k: v for k, v in id_counts.items() if v > 1}
    if duplicates:
        print(f"  ❌ Found {len(duplicates)} duplicate IDs:")
        for id_, count in list(duplicates.items())[:5]:
            print(f"     ID {id_}: appears {count} times")
    else:
        print(f"  ✅ No duplicate IDs")
    
    # Check 3: ID sequence (are IDs sequential from 0?)
    print("\n[3] ID SEQUENCE CHECK")
    if all(isinstance(id_, int) for id_ in ids):
        expected_ids = set(range(num_entries))
        actual_ids = set(ids)
        missing_ids = expected_ids - actual_ids
        extra_ids = actual_ids - expected_ids
        
        if not missing_ids and not extra_ids:
            print(f"  ✅ IDs are sequential: 0 to {num_entries - 1}")
        else:
            if missing_ids:
                print(f"  ⚠️  Missing IDs: {sorted(missing_ids)[:10]}{'...' if len(missing_ids) > 10 else ''}")
            if extra_ids:
                print(f"  ⚠️  Unexpected IDs: {sorted(extra_ids)[:10]}{'...' if len(extra_ids) > 10 else ''}")
    else:
        print(f"  ⚠️  Non-integer IDs found")
    
    # Check 4: Required fields
    print("\n[4] REQUIRED FIELDS CHECK")
    required_fields = ["id", "reference", "hypothesis"]
    missing_fields = []
    for entry in entries:
        for field in required_fields:
            if field not in entry:
                missing_fields.append((entry.get('_line_num', '?'), field))
    
    if missing_fields:
        print(f"  ❌ {len(missing_fields)} entries missing required fields:")
        for line_num, field in missing_fields[:5]:
            print(f"     Line {line_num}: missing '{field}'")
    else:
        print(f"  ✅ All entries have required fields: {required_fields}")
    
    # Check 5: Empty values
    print("\n[5] EMPTY VALUE CHECK")
    empty_hyp = [e for e in entries if not e.get("hypothesis", "").strip()]
    empty_ref = [e for e in entries if not e.get("reference", "").strip()]
    
    if empty_hyp:
        print(f"  ⚠️  {len(empty_hyp)} entries with empty hypothesis:")
        for e in empty_hyp[:3]:
            print(f"     ID {e.get('id')}: ref='{e.get('reference', '')[:50]}...'")
    else:
        print(f"  ✅ No empty hypotheses")
    
    if empty_ref:
        print(f"  ⚠️  {len(empty_ref)} entries with empty reference:")
        for e in empty_ref[:3]:
            print(f"     ID {e.get('id')}: hyp='{e.get('hypothesis', '')[:50]}...'")
    else:
        print(f"  ✅ No empty references")
    
    # Check 6: Sample preview
    print("\n[6] SAMPLE PREVIEW")
    print("-" * 60)
    for i in [0, num_entries // 2, num_entries - 1]:
        if i < len(entries):
            e = entries[i]
            print(f"  [{i}] ID: {e.get('id')}")
            print(f"      REF: {e.get('reference', '')[:60]}...")
            print(f"      HYP: {e.get('hypothesis', '')[:60]}...")
            print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    issues = []
    if expected_count and num_entries != expected_count:
        issues.append(f"Count mismatch ({num_entries}/{expected_count})")
    if duplicates:
        issues.append(f"{len(duplicates)} duplicate IDs")
    if missing_fields:
        issues.append(f"{len(missing_fields)} missing fields")
    if empty_hyp:
        issues.append(f"{len(empty_hyp)} empty hypotheses")
    if empty_ref:
        issues.append(f"{len(empty_ref)} empty references")
    
    if issues:
        print(f"⚠️  Issues found: {', '.join(issues)}")
        return False
    else:
        print(f"✅ All checks passed! {num_entries} samples verified.")
        return True


def main():
    parser = argparse.ArgumentParser(description="Verify evaluation output completeness")
    parser.add_argument(
        "--jsonl", "-j",
        type=str,
        required=True,
        help="Path to test_examples.jsonl"
    )
    parser.add_argument(
        "--expected", "-e",
        type=int,
        default=None,
        help="Expected number of test samples"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Config file to auto-detect expected count from dataset"
    )
    
    args = parser.parse_args()
    
    success = verify_output(
        jsonl_path=args.jsonl,
        expected_count=args.expected,
        config_path=args.config,
    )
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()