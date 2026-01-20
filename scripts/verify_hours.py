import sys
from datasets import load_from_disk

def print_split_hours(dataset, split_name):
    if split_name not in dataset:
        print(f"[WARNING] Split '{split_name}' not found.")
        return 0.0
    split = dataset[split_name]
    total_seconds = sum(split['duration'])
    total_hours = total_seconds / 3600
    print(f"{split_name.capitalize()} set: {len(split)} samples (~{total_hours:.2f} hours)")
    return total_hours

def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_hours.py <path_to_hf_dataset>")
        sys.exit(1)
    dataset_path = sys.argv[1]
    dataset = load_from_disk(dataset_path)
    print(f"Loaded dataset from: {dataset_path}")
    total_hours = 0.0
    for split in ['train', 'validation', 'test']:
        total_hours += print_split_hours(dataset, split)
    print(f"Total hours (all splits): ~{total_hours:.2f} hours")

if __name__ == "__main__":
    main()
