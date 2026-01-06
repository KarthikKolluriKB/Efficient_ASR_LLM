# SLAM-ASR Danish: Setup and Training Guide

This guide covers setting up the environment, downloading the Common Voice Danish dataset, and training the SLAM-ASR model for the SPEAKABLE 2026 paper.

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
- ~16GB GPU VRAM for Qwen2.5-3B (or use 0.5B variant for 6GB)

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
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate omegaconf wandb jiwer
pip install openai-whisper
pip install datasets  # For downloading Common Voice
```

---

## Dataset Download

### Setup Mozilla Data Collective API

1. Create account at https://datacollective.mozillafoundation.org
2. Go to the Danish dataset page and **accept the terms**
3. Get API key from https://datacollective.mozillafoundation.org/profile/credentials
4. Set environment variable:
   ```bash
   # Linux/Mac
   export MDC_API_KEY=your_api_key
   
   # Windows PowerShell
   $env:MDC_API_KEY = "your_api_key"
   ```

### Download Dataset

```bash
python datamodule/get_dataset.py --language da
```

Options:
- `--max_hours 10` - Limit total hours (useful for testing)
- `--archive path/to/file.tar.gz` - Use pre-downloaded archive
- `--root data` - Output directory (default: data)

### Verify Dataset

After download, verify the structure:
```bash
python -c "
import json
from pathlib import Path

data_dir = Path('data/common_voice_da')
for split in ['train', 'validation', 'test']:
    jsonl = data_dir / f'{split}.jsonl'
    if jsonl.exists():
        with open(jsonl) as f:
            count = sum(1 for _ in f)
        print(f'{split}: {count} samples')
    else:
        print(f'{split}: NOT FOUND')
"
```

Expected output:
```
train: ~3500 samples
validation: ~2500 samples  
test: ~2600 samples
```

---

## Training

### 1. Configure Weights & Biases (Optional but Recommended)

```bash
wandb login
# Enter your API key from https://wandb.ai/settings
```

### 2. Verify Configuration

Check `configs/train_config.yaml`:

```yaml
model:
  encoder_num_layers: null  # null = all 6 layers, or set 1-6

data:
  train_data_path: data/common_voice_da/train.jsonl
  val_data_path: data/common_voice_da/validation.jsonl
  test_data_path: data/common_voice_da/test.jsonl

log:
  use_wandb: true
  wandb_project_name: SLAM_ASR_Danish
  wandb_exp_name: whisper_base_qwen_danish
```

### 3. Start Training

```bash
# Baseline training (all 6 encoder layers)
python train.py --config configs/train_config.yaml
```

Training will:
- Load Whisper-base encoder (frozen)
- Load Qwen2.5-3B LLM (frozen)
- Train only the linear projector
- Log to W&B with efficiency metrics
- Save best checkpoint based on validation WER

### 4. Monitor Training

- **Console:** Watch for efficiency metrics at startup
- **W&B Dashboard:** View loss curves, WER, and efficiency table
- **Checkpoints:** Saved to `outputs/slam_asr_danish/`

---

## Layer Ablation Experiments

For the SPEAKABLE 2026 paper, run experiments with different encoder layer counts:

### Quick Reference

| Layers | Params Used | % of Total | Config Setting |
|--------|-------------|------------|----------------|
| 6 | 20.59M | 100% | `encoder_num_layers: null` |
| 5 | 17.44M | 84.7% | `encoder_num_layers: 5` |
| 4 | 14.29M | 69.4% | `encoder_num_layers: 4` |
| 3 | 11.13M | 54.1% | `encoder_num_layers: 3` |
| 2 | 7.98M | 38.8% | `encoder_num_layers: 2` |
| 1 | 4.83M | 23.5% | `encoder_num_layers: 1` |

### Run Ablation Experiments

**Option 1: Modify config directly**
```bash
# Edit configs/train_config.yaml
# Change: encoder_num_layers: 4

python train.py --config configs/train_config.yaml
```

**Option 2: Use a script to run all experiments**

Create `scripts/run_ablation.py`:

```python
"""Run layer ablation experiments for SPEAKABLE 2026 paper."""
import os
import subprocess
from omegaconf import OmegaConf

BASE_CONFIG = "configs/train_config.yaml"
LAYER_CONFIGS = [6, 5, 4, 3, 2, 1]  # null=6 means all layers

def run_experiment(num_layers):
    # Load config
    cfg = OmegaConf.load(BASE_CONFIG)
    
    # Modify for this experiment
    cfg.model.encoder_num_layers = num_layers if num_layers < 6 else None
    cfg.log.wandb_exp_name = f"whisper_base_{num_layers}L_danish"
    cfg.train.output_dir = f"outputs/ablation_{num_layers}L/"
    
    # Save temporary config
    temp_config = f"configs/temp_ablation_{num_layers}L.yaml"
    OmegaConf.save(cfg, temp_config)
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {num_layers} encoder layers")
    print(f"{'='*60}\n")
    
    # Run training
    subprocess.run(["python", "train.py", "--config", temp_config])
    
    # Cleanup
    os.remove(temp_config)

if __name__ == "__main__":
    for layers in LAYER_CONFIGS:
        run_experiment(layers)
```

Run all experiments:
```bash
python scripts/run_ablation.py
```

### Expected Results Format (for Paper)

After training, collect results from W&B or logs:

| Layers | Params (M) | WER (%) | CER (%) | RTF |
|--------|------------|---------|---------|-----|
| 6 | 20.59 | X.XX | X.XX | X.XX |
| 4 | 14.29 | X.XX | X.XX | X.XX |
| 2 | 7.98 | X.XX | X.XX | X.XX |

---

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config (try 4 or 2)
- Use `test_config.yaml` with Qwen2.5-0.5B for debugging

### Dataset Download Issues
- Ensure you have accepted HuggingFace terms for Common Voice
- Login with `huggingface-cli login`

### W&B Issues
- Disable with `use_wandb: false` in config
- Or run `wandb offline` for offline logging

### Audio Loading Errors
- Ensure `ffmpeg` is installed: `apt install ffmpeg` (Linux) or `choco install ffmpeg` (Windows)
- Verify audio paths in JSONL are absolute paths

---

## File Structure After Setup

```
Efficient_ASR_LLM/
├── configs/
│   ├── train_config.yaml    # Main training config
│   ├── eval_config.yaml     # Evaluation config
│   └── test_config.yaml     # Debug config (6GB VRAM)
├── data/
│   └── common_voice_da/
│       ├── audio/
│       │   ├── train/       # Training audio files
│       │   ├── validation/  # Validation audio files
│       │   └── test/        # Test audio files
│       ├── train.jsonl
│       ├── validation.jsonl
│       ├── test.jsonl
│       └── dataset_info.json
├── models/
│   ├── encoder.py           # Whisper encoder with layer pruning
│   ├── model.py             # SLAM-ASR model
│   └── projector.py         # Linear/MLP projector
├── utils/
│   └── metrics.py           # WER, CER, efficiency metrics
├── outputs/
│   └── slam_asr_danish/     # Training outputs & checkpoints
├── train.py
├── eval.py
└── scripts/
    ├── download_common_voice.py
    └── run_ablation.py
```

---

## Quick Start Summary

```bash
# 1. Setup environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Set API key and download dataset
export MDC_API_KEY=your_api_key
python datamodule/get_dataset.py --language da

# 3. Login to W&B (optional)
wandb login

# 4. Train baseline
python train.py --config configs/train_config.yaml

# 5. Run ablation experiments
python scripts/run_ablation.py
```
