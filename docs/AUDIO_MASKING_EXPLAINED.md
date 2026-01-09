# Audio Dataset & Masking: Deep Dive with Examples

## The Core Challenge

LLMs understand **text tokens**, not audio. We need to:
1. Convert audio → embeddings that "fit" where text tokens would go
2. Tell the model: "These positions contain audio, not text"
3. Train the model to predict text based on audio embeddings

---

## Part 1: A Concrete Example

### Input Data
```
Audio file: "hello_world.mp3" (3 seconds of someone saying "Hello world")
Transcription: "Hello world"
Task prompt: "Transcribe speech to text.\n"
```

### Step-by-Step Processing

#### Step 1: Audio to Mel Spectrogram
```python
# 3 seconds @ 16kHz = 48,000 audio samples
audio_raw = [0.01, -0.02, 0.05, ...]  # 48,000 values

# Convert to mel spectrogram (100 frames per second, 80 mel bins)
audio_mel = whisper.log_mel_spectrogram(audio_raw)
# Shape: [300, 80]  (3 sec × 100 = 300 frames)
```

**Visualization**:
```
Time →
     0.0s   0.5s   1.0s   1.5s   2.0s   2.5s   3.0s
     |      |      |      |      |      |      |
Mel: ████████████████████████████████████████████
     Frame 0    Frame 150              Frame 300
     
Each frame = 10ms of audio
Each frame = 80 frequency values (mel bins)
```

#### Step 2: Calculate Final Audio Length (after all downsampling)
```python
# Whisper encoder: 2x downsample
# Projector: 5x downsample  
# Total: 10x downsample

audio_length = (300 + 1) // 2  # Whisper: 300 → 150
audio_length = audio_length // 5  # Projector: 150 → 30

# Result: 3 seconds of audio → 30 "audio tokens"
print(audio_length)  # 30
```

#### Step 3: Create Audio Placeholder Tokens
```python
# We don't have actual audio tokens - we use -1 as placeholder
audio_pseudo = torch.full((30,), -1)
# [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  # 10
#  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  # 20
#  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  # 30
```

**Why -1?** It's an invalid token ID that we'll replace with actual embeddings later.

#### Step 4: Tokenize Text Parts
```python
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# Tokenize prompt
prompt = "Transcribe speech to text.\n"
prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
# [Transcribe, speech, to, text, ., \n] → [transcript_id, speech_id, to_id, text_id, dot_id, newline_id]
# Let's say: [46865, 8948, 311, 1495, 13, 198]  (6 tokens)

# Tokenize answer
answer = "Hello world"
answer_ids = tokenizer.encode(answer, add_special_tokens=False)
# [Hello, world] → [9707, 1917]  (2 tokens)

# Add EOS token
eos_id = tokenizer.eos_token_id  # e.g., 151643
```

#### Step 5: Concatenate into Full Sequence
```python
input_ids = audio_pseudo.tolist() + prompt_ids + answer_ids + [eos_id]
# = [-1]*30 + [46865, 8948, 311, 1495, 13, 198] + [9707, 1917] + [151643]

# Total length: 30 + 6 + 2 + 1 = 39 tokens
```

**Visual representation**:
```
Position:  0  1  2  ... 29 | 30    31    32  33   34  35 | 36   37   | 38
           ↓  ↓  ↓      ↓  |  ↓     ↓     ↓   ↓    ↓   ↓  |  ↓    ↓   |  ↓
input_ids: -1 -1 -1 ... -1 | Trans speech to text .  \n | Hello world| EOS
           └──────────────┘ └────────────────────────────┘└───────────┘└───┘
              AUDIO (30)           PROMPT (6)              ANSWER (2)   EOS
```

#### Step 6: Create Labels (What to Predict)
```python
IGNORE_INDEX = -100  # PyTorch's "don't compute loss here"

labels = [-100]*30 + [-100]*6 + answer_ids + [eos_id]
# = [-100]*36 + [9707, 1917, 151643]
```

**Visual representation**:
```
Position:  0   1   2  ... 29 | 30   31   32   33   34   35 | 36    37   | 38
           ↓   ↓   ↓      ↓  |  ↓    ↓    ↓    ↓    ↓    ↓  |  ↓     ↓   |  ↓
labels:   -100-100-100..-100 |-100 -100 -100 -100 -100 -100 | 9707  1917 | EOS
          └─────────────────────────────────────────────────┘└───────────────┘
                     DON'T COMPUTE LOSS HERE                  COMPUTE LOSS HERE
```

---

## Part 2: Batching Multiple Samples

### Two Samples with Different Lengths

**Sample A**: 3 sec audio, "Hello world" (39 tokens total)
**Sample B**: 6 sec audio, "This is a longer transcription" (88 tokens total)

```python
# Sample A (shorter)
audio_length_A = 30
input_ids_A = [-1]*30 + [prompt] + [Hello, world, EOS]  # 39 tokens

# Sample B (longer)  
audio_length_B = 60
input_ids_B = [-1]*60 + [prompt] + [This, is, a, longer, transcription, EOS]  # 88 tokens
```

### Left-Padding to Same Length

We pad Sample A to match Sample B's length (88 tokens):

```python
# Pad Sample A on the LEFT with pad_token_id
pad_length = 88 - 39  # = 49
input_ids_A_padded = [PAD]*49 + input_ids_A

# Now both have length 88
batch_input_ids = [
    [PAD]*49 + [-1]*30 + [prompt] + [Hello, world, EOS],      # Sample A
    [-1]*60 + [prompt] + [This, is, a, longer, trans, EOS],   # Sample B
]
```

**Visual comparison**:
```
Sample A: |PAD PAD PAD ... PAD| -1 -1 -1 ... -1 | prompt | answer |
          └───────49──────────┘└──────30───────┘└───6───┘└───3────┘

Sample B: | -1 -1 -1 -1 -1 ... -1 -1 -1 -1 -1 -1| prompt | answer     |
          └────────────────60────────────────────┘└───6───┘└────22─────┘
```

---

## Part 3: The Modality Mask (Crucial!)

### What is the Modality Mask?

A boolean tensor that tells us: **"Which positions contain audio embeddings?"**

```python
modality_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

# For Sample A: audio is at positions 49-78 (after 49 padding tokens)
modality_mask[0, 49:79] = True  # 30 audio positions

# For Sample B: audio is at positions 0-59
modality_mask[1, 0:60] = True   # 60 audio positions
```

**Visual**:
```
Sample A modality_mask:
Position: 0...48 | 49...78 | 79...87
          False  |  True   | False
          (pad)  | (audio) | (text)

Sample B modality_mask:
Position: 0...59 | 60...87
          True   | False
         (audio) | (text)
```

### Why Do We Need It?

When we feed embeddings to the LLM, we need to:
1. Replace positions where `modality_mask=True` with **encoder outputs**
2. Keep positions where `modality_mask=False` as **text embeddings**

---

## Part 4: The Embedding Replacement (In Model)

### Step 1: Get Text Embeddings for Everything

```python
# First, replace -1 with 0 (valid token ID for lookup)
input_ids_safe = input_ids.clone()
input_ids_safe[input_ids_safe == -1] = 0

# Get embeddings from LLM's embedding layer
text_embeds = llm.model.embed_tokens(input_ids_safe)
# Shape: [2, 88, 896]  (batch=2, seq=88, dim=896)
```

At this point, `text_embeds` has:
- Meaningless embeddings at audio positions (we looked up token 0)
- Correct embeddings at text positions

### Step 2: Process Audio Through Encoder + Projector

```python
# Whisper encodes the audio
encoder_out = whisper.encode(audio_mel)  # [2, 350, 512] for 700 mel frames

# Projector maps to LLM dimension
audio_embeds = projector(encoder_out)  # [2, 70, 896]
```

**Note**: After projection, we have 70 frames for the max audio length (70 = 700 / 10).

### Step 3: Create Padded Audio Embeddings

```python
# Create zero tensor same shape as text_embeds
encoder_outs_pad = torch.zeros_like(text_embeds)  # [2, 88, 896]

# For each sample, copy audio embeddings to correct positions
for i in range(batch_size):
    start = modality_mask_start_indices[i]  # Where audio starts
    length = modality_lengths[i]            # How many audio tokens
    
    # Copy encoder outputs to the right positions
    encoder_outs_pad[i, start:start+length] = audio_embeds[i, :length]
```

**Sample A** (start=49, length=30):
```
encoder_outs_pad[0]:
Position: 0...48 | 49...78 | 79...87
          zeros  | audio   | zeros
                  embeds
```

**Sample B** (start=0, length=60):
```
encoder_outs_pad[1]:
Position: 0...59 | 60...87
          audio  | zeros
          embeds
```

### Step 4: Combine Audio and Text Embeddings

```python
# The magic formula:
final_embeds = encoder_outs_pad + text_embeds * (~modality_mask[:, :, None])
```

**Breaking this down**:

```python
~modality_mask  # Invert: True where TEXT, False where AUDIO

text_embeds * (~modality_mask[:, :, None])
# Multiplies text_embeds by 0 where AUDIO, by 1 where TEXT
# Result: zeros at audio positions, text embeds at text positions

encoder_outs_pad + (above)
# Add audio embeds (non-zero only at audio positions)
# Add text embeds (non-zero only at text positions)
```

**Final result**:
```
Sample A final_embeds:
Position: 0...48   | 49...78    | 79...87
          PAD_emb  | AUDIO_emb  | TEXT_emb
          (text)   | (encoder)  | (text)

Sample B final_embeds:  
Position: 0...59   | 60...87
          AUDIO_emb| TEXT_emb
          (encoder)| (text)
```

---

## Part 5: Complete Example with Real Numbers

### Input
```
Audio: 3-second clip of "Hello world"
Prompt: "Transcribe speech to text.\n"
```

### Processing Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ AUDIO PROCESSING                                                    │
├─────────────────────────────────────────────────────────────────────┤
│ Raw audio: 48,000 samples (3 sec @ 16kHz)                          │
│     ↓ log_mel_spectrogram()                                        │
│ Mel spectrogram: [300, 80]                                         │
│     ↓ Whisper encoder (2x downsample)                              │
│ Encoder output: [150, 512]                                         │
│     ↓ Projector (5x downsample)                                    │
│ Audio embeddings: [30, 896]  ← These replace placeholders!         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ SEQUENCE CONSTRUCTION                                               │
├─────────────────────────────────────────────────────────────────────┤
│ Audio placeholders: [-1, -1, -1, ... -1]  (30 tokens)              │
│ Prompt tokens:      [46865, 8948, 311, 1495, 13, 198]  (6 tokens)  │
│ Answer tokens:      [9707, 1917]  (2 tokens)                       │
│ EOS token:          [151643]  (1 token)                            │
│                                                                     │
│ input_ids = [-1]*30 + [46865,8948,311,1495,13,198,9707,1917,151643]│
│             └──audio──┘└────────prompt────────┘└─answer─┘└─eos─┘   │
│                                                                     │
│ Total: 39 tokens                                                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ LABELS (for loss computation)                                       │
├─────────────────────────────────────────────────────────────────────┤
│ labels = [-100]*30 + [-100]*6 + [9707, 1917, 151643]               │
│          └─ignore──┘ └ignore─┘ └──compute loss here──┘             │
│                                                                     │
│ Loss is ONLY computed for positions 36, 37, 38 (the answer + EOS)  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ MODALITY MASK                                                       │
├─────────────────────────────────────────────────────────────────────┤
│ modality_mask = [True]*30 + [False]*9                              │
│                 └─audio──┘  └─text──┘                              │
│                                                                     │
│ Position 0-29:  True  → use encoder embeddings                     │
│ Position 30-38: False → use text embeddings                        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ FINAL EMBEDDINGS TO LLM                                             │
├─────────────────────────────────────────────────────────────────────┤
│ Position 0-29:   Audio embeddings from encoder (30 × 896)          │
│ Position 30-35:  Text embeddings for prompt (6 × 896)              │
│ Position 36-38:  Text embeddings for answer+EOS (3 × 896)          │
│                                                                     │
│ final_embeds shape: [1, 39, 896]                                   │
│                                                                     │
│ This goes into Qwen → predicts next token at each position        │
│ Loss computed only at positions 36-38 (where labels != -100)       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 6: What the LLM "Sees"

From the LLM's perspective:

```
Position 0-29:   [Some embeddings that encode audio information]
Position 30-35:  "Transcribe speech to text.\n"
Position 36:     → Predict next token (should be "Hello")
Position 37:     → Predict next token (should be "world")  
Position 38:     → Predict next token (should be EOS)
```

The LLM learns:
- "When I see these audio embeddings followed by 'Transcribe speech to text', I should output the transcription"
- The projector learns to create embeddings that make the LLM predict correctly

---

## Part 7: Attention Mask

### What is the Attention Mask?

Tells the model which positions can attend to which:

```python
attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

# Set False for padding positions
attention_mask[0, :49] = False  # Sample A has 49 padding tokens
```

**Effect**:
```
Sample A:
Position: 0...48 | 49...87
Attend?:  NO     | YES
          (pad)  | (real tokens)
```

This prevents the model from "looking at" meaningless padding tokens.

---

## Summary: The Complete Picture

```
INPUT:
┌────────────────────────────────────────────────────────────────┐
│ Audio file + Transcription + Prompt                            │
└────────────────────────────────────────────────────────────────┘
                              ↓
TOKENIZATION:
┌────────────────────────────────────────────────────────────────┐
│ input_ids:  [-1,-1,...,-1] + [prompt_tokens] + [answer_tokens] │
│ labels:     [-100,...,-100] + [-100,...] + [answer_tokens]     │
│ mod_mask:   [True,...,True] + [False,...,False]                │
└────────────────────────────────────────────────────────────────┘
                              ↓
EMBEDDING:
┌────────────────────────────────────────────────────────────────┐
│ Where mod_mask=True:  Use encoder(audio) → projector output    │
│ Where mod_mask=False: Use LLM.embed_tokens(input_ids)          │
└────────────────────────────────────────────────────────────────┘
                              ↓
LLM FORWARD:
┌────────────────────────────────────────────────────────────────┐
│ Combined embeddings → LLM → Logits → Loss (only on answer)    │
└────────────────────────────────────────────────────────────────┘
                              ↓
TRAINING:
┌────────────────────────────────────────────────────────────────┐
│ Gradients flow back through LLM → to projector → update it    │
│ Projector learns: audio features → text-like embeddings        │
└────────────────────────────────────────────────────────────────┘
```

The **modality mask** is the key that enables this hybrid audio-text processing!
