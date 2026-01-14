# SLAM-ASR Danish: Setup and Training Guide

This guide covers setting up the environment, downloading the Common Voice Danish dataset, and training the SLAM-ASR model.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Dataset Download](#dataset-download)
4. [Training](#training)
5. [Layer Ablation Experiments](#layer-ablation-experiments)

---

## Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- ~20GB disk space for dataset
- ~16GB GPU VRAM for Qwen2.5-3B

---

## Environment Setup

### 1. Clone the repository
```bash
git clone https://github.com/KarthikKolluriKB/Efficient_ASR_LLM.git
cd Efficient_ASR_LLM
```

### 2. Create virtual environment
```bash
python -m venv .venv

# Linux/Mac
source .venv/bin/activate

# Windows
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

**Option A: Using uv (recommended, faster)**
```bash
pip install uv
uv sync
```

**Option B: Using pip**
```bash
pip install -e .
```

---

## Dataset Download

### 1. Login to HuggingFace

```bash
huggingface-cli login
# Enter your token from: https://huggingface.co/settings/tokens
```

### 2. Download Dataset

```bash
python scripts/download_dataset.py --language da
```

This will:
- Download Common Voice 22.0 Danish from HuggingFace
- Preprocess transcriptions (lowercase, remove punctuation)
- Filter by duration (0.5s - 30s)
- Save as HuggingFace Dataset format

**Options:**
```bash
# Custom output directory
python scripts/download_dataset.py --language da --output-dir data/my_dataset

# Custom duration filters
python scripts/download_dataset.py --language da --min-duration 1.0 --max-duration 20.0

# Download specific splits only
python scripts/download_dataset.py --language da --splits train dev
```

### 3. Verify Dataset

```bash
python -c "from datasets import load_from_disk; ds = load_from_disk('data/cv22_hf/da'); print(ds)"
```

Expected output:
```
DatasetDict({
    train: Dataset({features: [...], num_rows: ~3600})
    validation: Dataset({features: [...], num_rows: ~2500})
    test: Dataset({features: [...], num_rows: ~2700})
})
```

---

## Training

### 1. Setup Weights & Biases (Optional but Recommended)

```bash
wandb login
# Enter your API key from: https://wandb.ai/settings
```

Or set in `.env` file:
```
WANDB_API_KEY=your_key_here
```

### 2. Start Training

**Baseline (12 layers):**
```bash
python train.py --config configs/danish/train/baseline.yaml
```

**LR Scheduler Experiments:**
```bash
# With LR scheduler (cosine warmup)
python train.py --config configs/danish/train/baseline_with_scheduler.yaml

# Fixed LR (constant)
python train.py --config configs/danish/train/baseline_fixed_lr.yaml
```

### 3. Monitor Training

- **Console:** Watch for WER/CER metrics
- **W&B Dashboard:** View loss curves at https://wandb.ai
- **Checkpoints:** Saved to `outputs/` directory

---

## Layer Ablation Experiments

For testing encoder efficiency with different layer counts:

### Available Configs

| Experiment | Layers | Config |
|------------|--------|--------|
| Baseline | 12 | `configs/danish/train/baseline.yaml` |
| Ablation 11L | 11 | `configs/danish/train/ablation_11L.yaml` |
| Ablation 10L | 10 | `configs/danish/train/ablation_10L.yaml` |
| Ablation 9L | 9 | `configs/danish/train/ablation_9L.yaml` |
| Ablation 8L | 8 | `configs/danish/train/ablation_8L.yaml` |

### Run All Ablations

```bash
python scripts/run_ablation.py
```

Or run individually:
```bash
python train.py --config configs/danish/train/ablation_11L.yaml
python train.py --config configs/danish/train/ablation_10L.yaml
# etc.
```

---

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config (try 4 → 2)
- Enable `mixed_precision: true`

### Dataset Download Issues
- Ensure HuggingFace login: `huggingface-cli login`
- Check internet connection

### W&B Issues
- Disable with `use_wandb: false` in config
- Or run `wandb offline` for offline logging

### Audio Loading Errors
- Ensure `ffmpeg` is installed: `apt install ffmpeg` (Linux)

---

## Quick Start Summary

```bash
# 1. Setup
git clone https://github.com/KarthikKolluriKB/Efficient_ASR_LLM.git
cd Efficient_ASR_LLM
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Download dataset
huggingface-cli login
python scripts/download_dataset.py --language da

# 3. Verify
python -c "from datasets import load_from_disk; print(load_from_disk('data/cv22_hf/da'))"

# 4. Train
wandb login
python train.py --config configs/danish/train/baseline.yaml
```

---

## File Structure

```
Efficient_ASR_LLM/
├── configs/
│   └── danish/
│       ├── train/
│       │   ├── baseline.yaml
│       │   ├── baseline_with_scheduler.yaml
│       │   ├── baseline_fixed_lr.yaml
│       │   └── ablation_*.yaml
│       └── eval/
├── data/
│   └── cv22_hf/
│       └── da/                  # HuggingFace Dataset
│           ├── train/
│           ├── validation/
│           ├── test/
│           └── dataset_dict.json
├── datamodule/
│   ├── dataset.py              # Dataset class
│   ├── download_data.py        # Download utilities
│   ├── preprocess_data.py      # Preprocessing utilities
│   └── hf_data.py              # Main data preparation
├── models/
│   ├── encoder.py              # Whisper encoder
│   ├── model.py                # SLAM-ASR model
│   └── projector.py            # Projector layers
├── scripts/
│   ├── download_dataset.py     # Dataset download script
│   └── run_ablation.py         # Run all ablations
├── train.py                    # Training script
└── eval.py                     # Evaluation script
```
