# SLAM-ASR: A Beginner's Guide

## What Are We Building?

We're building a **Speech-to-Text system** that can listen to Danish audio and write out what was said. 

Think of it like this:
```
🎤 Audio: "Hej, hvordan har du det?"
    ↓
🤖 Our System
    ↓
📝 Text: "Hej, hvordan har du det?"
```

---

## The Three Main Components

Our system has **3 parts** that work together:

```
┌─────────────────────────────────────────────────────────────┐
│                      SLAM-ASR System                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   ENCODER    │───▶│  PROJECTOR   │───▶│     LLM      │  │
│  │  (Whisper)   │    │   (Linear)   │    │  (Qwen2.5)   │  │
│  │              │    │              │    │              │  │
│  │ "Listens to  │    │ "Translates  │    │ "Writes the  │  │
│  │  the audio"  │    │  between     │    │   text"      │  │
│  │              │    │   them"      │    │              │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│      FROZEN             TRAINABLE           FROZEN          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1. Encoder (Whisper) - "The Ear" 👂

**What it does:** Listens to audio and extracts meaningful features.

**Analogy:** Imagine someone who can hear sounds but doesn't know any language. They can tell:
- This sound is high-pitched
- This sound is long
- These sounds are similar

**Technical details:**
- We use **Whisper-base** (trained by OpenAI)
- Input: Raw audio waveform (16kHz)
- Output: Audio features (numbers that represent the sound)
- **FROZEN** = We don't change it during training (it's already good at listening)

```
Audio waveform    →    Whisper Encoder    →    Audio Features
[samples at 16kHz]                             [512-dimensional vectors]
```

### 2. Projector - "The Translator" 🔄

**What it does:** Converts audio features into a format the LLM understands.

**Analogy:** Imagine a translator between two people who speak different languages. The Encoder speaks "Audio Language" and the LLM speaks "Text Language". The Projector translates between them.

**Technical details:**
- Simple neural network (2 linear layers)
- Input: 512-dimensional audio features
- Output: 2048-dimensional vectors (matching LLM)
- **TRAINABLE** = This is what we're teaching!

```
Audio Features (512-dim)  →  Projector  →  LLM-compatible vectors (2048-dim)
```

**Why we train ONLY this part:**
- Whisper already knows how to hear (pre-trained)
- Qwen already knows Danish (pre-trained)
- We just need to teach them how to "talk to each other"

### 3. LLM (Qwen2.5-3B) - "The Writer" ✍️

**What it does:** Takes the translated audio features and generates text.

**Analogy:** A person who is very good at writing, but needs someone to tell them what to write. Once they receive the information (from the Projector), they write it out word by word.

**Technical details:**
- We use **Qwen2.5-3B** (3 billion parameters)
- Already knows Danish language
- Generates text one token at a time
- **FROZEN** = We don't change it (it's already good at writing)

---

## How Does It All Work Together?

### Step-by-Step Process:

```
STEP 1: Load Audio
────────────────────────────────────────────────────
📁 Audio file (MP3/WAV)
    ↓
🔊 Raw waveform at 16,000 samples per second
    Example: 5 seconds = 80,000 numbers


STEP 2: Convert to Mel Spectrogram
────────────────────────────────────────────────────
🔊 Raw waveform
    ↓
📊 Mel spectrogram (like a "picture" of sound)
    - 80 frequency bands
    - Time frames (1 frame per ~10ms)
    - 5 seconds → ~500 frames


STEP 3: Whisper Encoder
────────────────────────────────────────────────────
📊 Mel spectrogram [500 frames × 80 features]
    ↓
🎯 Whisper's 2 conv layers (2x downsample)
    ↓
🧠 6 Transformer layers
    ↓
📤 Audio features [250 frames × 512 dimensions]


STEP 4: Projector
────────────────────────────────────────────────────
📤 Audio features [250 frames × 512 dim]
    ↓
🔗 Concatenate 5 frames together (5x downsample)
    [50 frames × 2560 dim]
    ↓
🧮 Linear layer 1: 2560 → 512
    ↓
⚡ ReLU activation
    ↓
🧮 Linear layer 2: 512 → 2048
    ↓
📦 LLM-ready features [50 frames × 2048 dim]


STEP 5: Combine with Text
────────────────────────────────────────────────────
Create the input sequence:

[AUDIO EMBEDDINGS] + [PROMPT] + [ANSWER]
      50 tokens       6 tokens    ~10 tokens

Example:
┌────────────────────┬───────────────────────────┬─────────────────────┐
│  Audio features    │ "Transcribe speech to    │ "Hej verden"        │
│  (from projector)  │  text.\n"                │ (target text)       │
└────────────────────┴───────────────────────────┴─────────────────────┘


STEP 6: LLM Forward Pass
────────────────────────────────────────────────────
The LLM processes everything and predicts the next token:

Position:  [0...49]  [50-55]     [56]    [57]    [58]    [59]
Input:     [AUDIO]   [PROMPT]    "Hej"   " ver"  "den"   <EOS>
                          ↓        ↓       ↓       ↓       ↓
Predict:      ✗         "Hej"  " ver"   "den"   <EOS>    ✗

Only the ANSWER part is used for training loss!
```

---

## Training: Teaching the Projector

### What Does "Training" Mean?

Training = Adjusting the Projector's numbers so it gets better at its job.

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING LOOP                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Take an audio sample and its transcription              │
│     Audio: "common_voice_da_123.mp3"                        │
│     Text:  "Hej, hvordan har du det?"                       │
│                                                             │
│  2. Run through the model                                   │
│     Audio → Encoder → Projector → LLM → Predicted Text      │
│                                                             │
│  3. Compare prediction to actual text                       │
│     Predicted: "Hej, hvordan har du det?"  ✓                │
│     Actual:    "Hej, hvordan har du det?"                   │
│                                                             │
│  4. Calculate error (loss)                                  │
│     Loss = How wrong were the predictions?                  │
│                                                             │
│  5. Update Projector weights                                │
│     Adjust numbers to make predictions better next time     │
│                                                             │
│  6. Repeat thousands of times!                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Loss Function (Cross-Entropy)

The loss measures "how surprised the model is by the correct answer."

```
If model is confident about correct answer → Low loss (good!)
If model is confused or wrong            → High loss (bad!)

Goal: Minimize the loss over training
```

### Why Only Train the Projector?

| Component | Parameters | Why Frozen? |
|-----------|------------|-------------|
| Whisper   | 74 million | Already trained on 680,000 hours of audio |
| Projector | 2.6 million | **THIS IS WHAT WE TRAIN** |
| Qwen2.5   | 3 billion  | Already trained on trillions of tokens |

Training everything would:
1. Require much more data (we only have 3,592 samples)
2. Take much longer (days instead of hours)
3. Risk "forgetting" what they already know

---

## Label Masking: What Gets Trained?

Not every part of the input is used for training. We use **masking**:

```
Input sequence:
┌──────────────────────────────────────────────────────────┐
│ [AUDIO TOKENS]  [PROMPT TOKENS]  [ANSWER TOKENS] [EOS]  │
│     (50)            (6)              (10)         (1)   │
└──────────────────────────────────────────────────────────┘

Labels (what model should predict):
┌──────────────────────────────────────────────────────────┐
│ [-100 -100 ...]  [-100 -100 ...]  [answer ids]   [EOS]  │
│   IGNORED          IGNORED         TRAINED!             │
└──────────────────────────────────────────────────────────┘

-100 = "Ignore this, don't calculate loss here"
```

**Why mask audio and prompt?**
- Audio tokens: Model can't predict audio from text
- Prompt: Always the same, no learning needed
- Answer: This is what we want the model to learn!

---

## Key Metrics

### 1. Loss (Training Signal)
- **What:** How wrong the model's predictions are
- **Good values:** Should decrease over training
- **Starting:** ~8-10 (random predictions)
- **Target:** <2 (confident correct predictions)

### 2. Accuracy (Token-level)
- **What:** % of tokens predicted correctly
- **Good values:** Should increase over training
- **Starting:** ~0% (random)
- **Target:** >80%

### 3. WER (Word Error Rate) - Most Important for ASR!
- **What:** % of words that are wrong in the transcription
- **Formula:** (Insertions + Deletions + Substitutions) / Total Words
- **Good values:** Lower is better
- **Starting:** ~100% (gibberish output)
- **Target:** <30% (understandable), <10% (good), <5% (excellent)

```
Example WER calculation:

Reference: "Hej hvordan har du det"     (5 words)
Predicted: "Hej vordan har du det"      (1 substitution: hvordan→vordan)

WER = 1/5 = 20%
```

---

## Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                         COMPLETE DATA FLOW                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. AUDIO INPUT                                                     │
│     ├── File: "clip.mp3"                                            │
│     ├── Duration: 5 seconds                                         │
│     └── Format: 16kHz mono                                          │
│              ↓                                                      │
│  2. MEL SPECTROGRAM                                                 │
│     ├── Shape: [500, 80]                                            │
│     └── (500 time frames, 80 frequency bins)                        │
│              ↓                                                      │
│  3. WHISPER ENCODER (FROZEN)                                        │
│     ├── Conv layers: 2x downsample → [250, 80]                      │
│     ├── Transformer: → [250, 512]                                   │
│     └── Output: 250 frames of 512-dim features                      │
│              ↓                                                      │
│  4. PROJECTOR (TRAINABLE)                                           │
│     ├── Concat 5 frames: [250, 512] → [50, 2560]                    │
│     ├── Linear1: [50, 2560] → [50, 512]                             │
│     ├── ReLU activation                                             │
│     ├── Linear2: [50, 512] → [50, 2048]                             │
│     └── Output: 50 "audio tokens" of 2048-dim                       │
│              ↓                                                      │
│  5. COMBINE WITH TEXT                                               │
│     ├── Audio embeddings: [50, 2048]                                │
│     ├── Prompt tokens: [6, 2048] ("Transcribe speech to text.\n")   │
│     ├── Answer tokens: [10, 2048] (target transcription)            │
│     └── Combined: [66, 2048]                                        │
│              ↓                                                      │
│  6. LLM FORWARD (FROZEN)                                            │
│     ├── Input: [66, 2048] embeddings                                │
│     ├── 24 Transformer layers                                       │
│     └── Output logits: [66, 151936] (vocab size)                    │
│              ↓                                                      │
│  7. LOSS CALCULATION                                                │
│     ├── Only on answer positions [56:66]                            │
│     ├── Cross-entropy between predicted and actual tokens           │
│     └── Backpropagate through Projector only                        │
│              ↓                                                      │
│  8. UPDATE PROJECTOR                                                │
│     ├── Optimizer: AdamW                                            │
│     ├── Learning rate: 1e-4                                         │
│     └── Update ~2.6M parameters                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Training Configuration Explained

```yaml
# WHAT WE'RE TRAINING
freeze_llm: true        # Don't change LLM weights
freeze_encoder: true    # Don't change Whisper weights
# Only Projector is trained!

# HOW MUCH DATA PER UPDATE
batch_size: 8                    # 8 samples at once
gradient_accumulation_steps: 4   # Accumulate 4 batches
# Effective batch size = 8 × 4 = 32 samples per update

# HOW LONG TO TRAIN
num_epochs: 50         # Go through all data 50 times
warmup_steps: 200      # Start with small learning rate, increase gradually

# LEARNING RATE
lr: 1.0e-4             # How big each update is (0.0001)
weight_decay: 0.01     # Prevent overfitting

# WHEN TO CHECK PROGRESS
validation_interval: 112   # Validate every 112 steps (= 1 epoch)

# WHEN TO STOP EARLY
patience: 15           # Stop if no improvement for 15 epochs
min_delta: 0.005       # Improvement must be at least 0.5% WER
```

---

## Expected Training Progress

```
Epoch  1: Loss=8.5, WER=100% (random noise output)
Epoch  5: Loss=5.2, WER=95%  (some patterns emerging)
Epoch 10: Loss=3.8, WER=80%  (starting to form words)
Epoch 15: Loss=2.5, WER=60%  (recognizable words)
Epoch 20: Loss=1.8, WER=45%  (sentences forming)
Epoch 30: Loss=1.2, WER=30%  (mostly correct)
Epoch 40: Loss=0.9, WER=20%  (good transcriptions)
Epoch 50: Loss=0.7, WER=15%  (minor errors only)
```

---

## Common Problems and Solutions

### Problem: WER stuck at ~90%
**Cause:** Projector not learning properly
**Solutions:**
- ✅ Make sure projector is in FP32 (not BF16)
- ✅ Check learning rate (try 1e-4 to 1e-3)
- ✅ Train for more epochs

### Problem: Loss is NaN
**Cause:** Numerical instability
**Solutions:**
- ✅ Use bfloat16 instead of fp16
- ✅ Lower learning rate
- ✅ Add gradient clipping

### Problem: Out of Memory (OOM)
**Cause:** GPU doesn't have enough memory
**Solutions:**
- ✅ Reduce batch size
- ✅ Reduce max_audio_length
- ✅ Use gradient checkpointing

---

## Quick Reference

| Term | Meaning |
|------|---------|
| **Encoder** | Converts audio to features (Whisper) |
| **Projector** | Bridges encoder and LLM |
| **LLM** | Generates text (Qwen2.5) |
| **Frozen** | Parameters not updated during training |
| **Epoch** | One pass through all training data |
| **Batch** | Group of samples processed together |
| **Loss** | How wrong the predictions are |
| **WER** | Word Error Rate (lower = better) |
| **Mel spectrogram** | Visual representation of audio frequencies |
| **Token** | Smallest unit of text (word piece) |
| **Embedding** | Vector representation of a token |

---

## Running Training

```bash
# Start training
python train.py --config configs/train_config.yaml

# Monitor with WandB (opens in browser)
# Or check logs in ./logs/train.log
```

Good luck with your training! 🚀
