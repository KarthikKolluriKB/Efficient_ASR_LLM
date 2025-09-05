import torch
import types
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, WhisperModel

from models.encoder import WhisperWrappedEncoder
from models.projector import EncoderProjectorConcat
from models.model import ASRLLM
from datamodule.dataset import get_speech_dataset

# ---- Paths ----
TRAIN_JSONL = "data/train.jsonl"
TEST_JSONL = "data/test.jsonl"
LLM_NAME = "sshleifer/tiny-gpt2"           # fast, tiny model
WHISPER_NAME = "openai/whisper-base"       # HF checkpoint (Windows-compatible!)
BATCH_SIZE = 2
MEL_SIZE = 80                             # Set to 128 for large-v3

# --- Tokenizer & LLM ---
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
llm = AutoModelForCausalLM.from_pretrained(LLM_NAME)

# --- Minimal config for projector ---
class DummyModelConfig:
    encoder_model = "openai/whisper-base"   # HF transformer model
    whisper_decode = False
    encoder_projector_ds_rate = 5
    encoder_dim = 1280
    llm_dim = llm.config.hidden_size


class DummyDataConfig:
    mel_size = MEL_SIZE
    max_source_length = 128
    max_target_length = 32
    pad_audio_length = None
    pad_input_length = 128
    pad_target_length = 32
    val_data_path = TEST_JSONL
    train_data_path = TRAIN_JSONL


projector = EncoderProjectorConcat(DummyModelConfig())

# --- Encoder (HuggingFace only for Windows support) ---
encoder = WhisperWrappedEncoder.load(DummyModelConfig())


# --- ASRLLM Model ---
model = ASRLLM(
    encoder=encoder,
    llm=llm,
    encoder_projector=projector,
    tokenizer=tokenizer,
    train_config={"freeze_encoder": True, "freeze_llm": True},
    model_config={"encoder_projector": "linear"},
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# --- Freeze everything except projector ---
for p in encoder.parameters(): p.requires_grad = False
for p in llm.parameters(): p.requires_grad = False
model.encoder.eval()
model.llm.eval()

optimizer = torch.optim.AdamW(model.encoder_projector.parameters(), lr=1e-3)

cfg = {
    "train_data_path": "data/train.jsonl",
    "val_data_path": "data/test.jsonl",
    "inference_mode": False,
    "input_type": "mel",
    "mel_size": 80,
}

dataset_config = types.SimpleNamespace(**cfg)

# Train: load small train set
train_ds = get_speech_dataset(dataset_config, tokenizer, split="test")
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=train_ds.collator)

# ---- TRAIN LOOP ----
print("Training projector on small batch...")

model.train()
for batch in train_loader:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    audio_mel = batch["audio_mel"].to(device)

    optimizer.zero_grad()
    outputs, _ = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        audio_mel=audio_mel,
    )
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    print(f"[TRAIN] input_ids: {input_ids.shape} audio_mel: {audio_mel.shape} loss: {loss.item():.5f}")

print("TRAIN LOOP DONE.\n")

# ---- TEST LOOP ----
print("Testing generation on small test set...")

# model.eval()
# with torch.no_grad():
#     for batch in test_loader:
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         audio_mel = batch["audio_mel"].to(device)

#         # Get combined embeddings
#         inputs_embeds, attn = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             audio_mel=audio_mel,
#             inference_mode=True,
#         )

#         # Generate
#         gen_ids = model.llm.generate(
#             inputs_embeds=inputs_embeds,
#             attention_mask=attn,
#             max_new_tokens=24,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.pad_token_id,
#         )
#         hyp = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
#         print(f"[TEST] hyp: {hyp}")

print("TEST LOOP DONE.")
