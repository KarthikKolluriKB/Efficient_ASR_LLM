from datasets import load_dataset

def get_split_info(ds, split_name):
    num_samples = len(ds)
    total_hours = sum(ds['duration']) / 3600 if 'duration' in ds.features else 0
    return num_samples, total_hours

def print_dataset_report(language_code, language_name):
    ds = load_dataset("fsicoli/common_voice_22_0", language_code)
    train_samples, train_hours = get_split_info(ds['train'], 'train')
    dev_samples, dev_hours = get_split_info(ds['validation'], 'validation')
    test_samples, test_hours = get_split_info(ds['test'], 'test')
    total_samples = train_samples + dev_samples + test_samples
    total_hours = train_hours + dev_hours + test_hours

    print(f"Dataset: {language_name}")
    print(f"Train set: {train_samples} (~{train_hours:.2f} hours)")
    print(f"Dev set: {dev_samples} (~{dev_hours:.2f} hours)")
    print(f"Test set: {test_samples} (~{test_hours:.2f} hours)")
    print(f"Total size: {total_samples} samples (~{total_hours:.2f} hours)")

# Example usage:
# print_dataset_report("nl", "Dutch")
# print_dataset_report("da", "Danish")