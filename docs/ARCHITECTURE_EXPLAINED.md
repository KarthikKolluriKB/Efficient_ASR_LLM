# SLAM-ASR Architecture: A Complete Technical Guide

## Overview

SLAM-ASR (Speech-Language Model for ASR) connects a **speech encoder** to a **language model** via a **projector**. The goal: make an LLM "understand" audio and transcribe it to text.

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Audio     │ ──── │  Projector  │ ──── │    LLM      │ ──── Text
│  (Whisper)  │      │  (Linear)   │      │  (Qwen)     │
│   FROZEN    │      │  TRAINABLE  │      │   FROZEN    │
└─────────────┘      └─────────────┘      └─────────────┘
     512-dim    ────>    896-dim     ────>   generates
```

**Key Insight**: Only the projector is trained. It learns to "translate" Whisper's audio representations into a format Qwen can understand.

---

## Part 1: The Data Pipeline

### Step 1: Raw Audio → Mel Spectrogram

```python
# In dataset.py: _compute_mel_spectrogram()
audio_raw = whisper.load_audio("clip.mp3")  # 16kHz waveform
audio_mel = whisper.log_mel_spectrogram(audio_raw, n_mels=80)
# Shape: [T_frames, 80] where T_frames = audio_seconds * 100
```

**Example**: 3-second audio → 300 mel frames × 80 frequency bins

### Step 2: Calculate Audio Length (accounting for downsampling)

```python
# In dataset.py: __getitem__()
audio_length = (audio_mel.shape[0] + 1) // 2  # Whisper 2x downsample
audio_length = audio_length // 5              # Projector 5x downsample
# 300 frames → 150 → 30 tokens
```

**Why?** We need to know how many "positions" the audio will occupy in the final sequence.

### Step 3: Build the Input Sequence

```python
# Create placeholder tokens for audio positions
audio_pseudo = torch.full((audio_length,), -1)  # [-1, -1, -1, ...] (30 tokens)

# Tokenize prompt and answer
prompt_ids = tokenizer("Transcribe speech to text.\n")  # [tok1, tok2, ..., tok6]
answer_ids = tokenizer("Hello world")                    # [tok7, tok8, tok9]

# Concatenate: [AUDIO_PLACEHOLDERS] + [PROMPT] + [ANSWER] + [EOS]
input_ids = [-1]*30 + [tok1..tok6] + [tok7, tok8, tok9] + [eos]
#            ↑ audio   ↑ prompt     ↑ answer
```

### Step 4: Create Labels (what to predict)

```python
# Labels = input_ids shifted, with audio+prompt masked
labels = [-100]*30 + [-100]*6 + [tok7, tok8, tok9] + [eos]
#         ↑ ignore    ↑ ignore   ↑ predict these!
```

**-100** is PyTorch's ignore index - loss is NOT computed for these positions.

### Step 5: Collator - Batch Multiple Samples

```python
# Pad sequences to same length (left-pad for causal LM)
# Sample 1: 30 audio + 6 prompt + 7 answer = 43 tokens
# Sample 2: 60 audio + 6 prompt + 22 answer = 88 tokens
# Batch: pad Sample 1 to 88 tokens

input_ids = [
    [PAD]*45 + [-1]*30 + [prompt] + [answer1],  # Sample 1 (left-padded)
    [-1]*60 + [prompt] + [answer2],              # Sample 2
]

# Create modality_mask: True where audio embeddings should go
modality_mask = [
    [False]*45 + [True]*30 + [False]*13,  # Audio at positions 45-74
    [True]*60 + [False]*28,                # Audio at positions 0-59
]
```

---

## Part 2: The Model Forward Pass

### Step 1: Whisper Encodes Audio

```python
# In model.py: forward()
# Input: audio_mel [B, T_mel, 80] = [4, 700, 80] (padded to max)
encoder_outputs = self.encoder.extract_variable_length_features(audio_mel.permute(0, 2, 1))
# Output: [B, T_mel/2, 512] = [4, 350, 512]
```

**Whisper's 2x downsampling**: 700 mel frames → 350 encoder frames

### Step 2: Projector Maps to LLM Space

```python
# In projector.py: EncoderProjectorConcat.forward()
# Input: [B, 350, 512]

# Concatenate every 5 frames: [B, 350, 512] → [B, 70, 512*5]
x = x.view(batch_size, 350 // 5, 512 * 5)  # [B, 70, 2560]

# Project to LLM dimension
x = self.linear1(x)  # [B, 70, 2560] → [B, 70, hidden_dim]
x = self.relu(x)
x = self.dropout(x)  # Regularization
x = self.linear2(x)  # [B, 70, hidden_dim] → [B, 70, 896]

# Output: [B, 70, 896] - same dimension as Qwen embeddings!
```

**5x downsampling**: 350 frames → 70 audio tokens

### Step 3: Get Text Embeddings from LLM

```python
# Replace -1 placeholders with 0 for valid lookup
input_ids[input_ids == -1] = 0

# Get token embeddings from Qwen
inputs_embeds = self.llm.model.embed_tokens(input_ids)
# Shape: [B, seq_len, 896] = [4, 105, 896]
```

### Step 4: Replace Audio Positions with Encoder Outputs (THE KEY STEP!)

```python
# Create empty tensor same shape as inputs_embeds
encoder_outs_pad = torch.zeros_like(inputs_embeds)  # [B, 105, 896]

# For each sample, copy encoder outputs to correct positions
for i in range(batch_size):
    start = modality_mask_start_indices[i]  # e.g., 45 for sample 1
    length = modality_lengths[i]            # e.g., 30 for sample 1
    
    # Copy first 30 encoder frames to positions 45-74
    encoder_outs_pad[i, start:start+length] = encoder_outputs[i][:length]

# Combine: audio positions get encoder output, text positions keep text embeddings
# modality_mask: True for audio, False for text
# ~modality_mask: False for audio, True for text
inputs_embeds = encoder_outs_pad + inputs_embeds * (~modality_mask[:, :, None])
```

**Visual representation**:
```
Position:    0...44  | 45...74  | 75...104
             ↓       |    ↓     |    ↓
inputs_embeds = [PAD] + [AUDIO] + [PROMPT+ANSWER]
                zeros   encoder    text embeds
                        outputs
```

### Step 5: LLM Forward Pass

```python
# Feed combined embeddings to Qwen
model_outputs = self.llm(
    inputs_embeds=inputs_embeds,  # [B, 105, 896] - audio + text embeddings
    attention_mask=attention_mask,
    labels=labels,                 # For loss computation
    use_cache=False
)

loss = model_outputs.loss          # Cross-entropy loss on predicted tokens
logits = model_outputs.logits      # [B, 105, vocab_size] = [B, 105, 151936]
```

---

## Part 3: Training - How the Projector Learns

### The Loss Function

```python
# LLM computes cross-entropy loss ONLY where labels != -100
# labels = [-100, -100, ..., tok7, tok8, tok9, eos]
#           ↑ ignored         ↑ loss computed here

# For each answer position, compare:
#   - Predicted: argmax(logits[position])
#   - Target: labels[position]
```

### Gradient Flow

```
Loss ──────────────────────────────────────────────────────┐
  ↓                                                        │
Logits ← LLM (frozen, no gradients) ← inputs_embeds        │
                                            ↑              │
              ┌─────────────────────────────┘              │
              ↓                                            │
        [AUDIO EMBEDDINGS] ← Projector (TRAINABLE) ← Encoder outputs
                                   ↑                       │
                              GRADIENTS FLOW HERE! ────────┘
```

**Key**: Even though LLM is frozen, gradients flow THROUGH it to the projector.

### What the Projector Learns

Initially, projector outputs random vectors → LLM generates gibberish
After training, projector learns to map:
```
Whisper("Hello") → embedding that makes Qwen predict "Hello"
Whisper("World") → embedding that makes Qwen predict "World"
```

---

## Part 4: Generation (Inference)

### The Problem We Fixed

**Wrong approach** (caused empty outputs):
```python
# Full sequence: [AUDIO] + [PROMPT] + [ANSWER] + [EOS]
generated = model.generate(full_input_ids)
# Model has nothing to generate - answer already in input!
```

**Fixed approach**:
```python
# Truncated: [AUDIO] + [PROMPT] only
answer_start = (labels != -100).nonzero()[0]  # Find where answer begins
truncated_input = input_ids[:, :answer_start]
generated = model.generate(truncated_input)
# Now model generates the answer!
```

### Generation Process

```python
# 1. Forward pass with truncated input to get embeddings
inputs_embeds, attention_mask = model.forward(
    input_ids=truncated_input,
    audio_mel=audio_mel,
    modality_mask=truncated_modality_mask,
    inference_mode=True  # Returns embeddings, not logits
)

# 2. LLM generates tokens autoregressively
generated_ids = self.llm.generate(
    inputs_embeds=inputs_embeds,
    max_new_tokens=128,
    do_sample=False,  # Greedy decoding
)

# 3. Decode to text
transcription = tokenizer.decode(generated_ids)
```

---

## Part 5: Key Numbers to Remember

| Stage | Input Shape | Output Shape | Downsampling |
|-------|-------------|--------------|--------------|
| Audio (3 sec) | 48000 samples | - | - |
| Mel Spectrogram | 48000 | [300, 80] | 160x |
| Whisper Encoder | [300, 80] | [150, 512] | 2x |
| Projector | [150, 512] | [30, 896] | 5x |
| **Total** | 3 sec audio | **30 tokens** | **~1600x** |

**Rule of thumb**: 1 second of audio ≈ 10 tokens after full processing

---

## Part 6: Why Danish Failed, English Should Work

### The Math

| Dataset | Hours | Samples | Unique Audio Tokens | Projector Params |
|---------|-------|---------|---------------------|------------------|
| Danish | ~4h | 3,500 | ~144,000 | 7M |
| English | ~50h | 36,000 | ~1,800,000 | 7M |

**Danish**: 144K tokens to learn 7M parameters → severe overfitting  
**English**: 1.8M tokens to learn 7M parameters → should generalize

### The Visualization

```
Danish Training:
Epoch 1:  Loss ████████████████████ 
Epoch 10: Loss ████                  (Train overfits)
Val WER:  ████████████████████████████ (Stays high - no generalization)

English Training (expected):
Epoch 1:  Loss ████████████████████ 
Epoch 10: Loss ████████              (Train learns)
Val WER:  ████████████               (Decreases - generalization!)
```

---

## Summary: The Complete Flow

```
1. AUDIO FILE
   ↓ load_audio()
2. WAVEFORM [48000 samples @ 16kHz]
   ↓ log_mel_spectrogram()
3. MEL SPECTROGRAM [300, 80]
   ↓ Whisper encoder (2x downsample)
4. ENCODER FEATURES [150, 512]
   ↓ Projector (5x downsample + dimension change)
5. AUDIO EMBEDDINGS [30, 896]
   ↓ Replace placeholder positions
6. COMBINED EMBEDDINGS [seq_len, 896]
   = [audio_embeds] + [text_embeds]
   ↓ Qwen LLM
7. LOGITS [seq_len, 151936]
   ↓ Cross-entropy loss on answer positions
8. LOSS → Backprop → Update Projector
   ↓ After training
9. GENERATION: Audio + Prompt → Transcription
```

**The magic**: Projector learns to make Qwen "hear" by converting Whisper's audio representations into Qwen's "language".
