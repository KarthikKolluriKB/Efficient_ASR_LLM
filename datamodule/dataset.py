"""
Speech Dataset for SLAM-ASR training.
Supports variable-length audio (Common Voice, LibriSpeech, etc.)
Reads data from JSONL format with fields: source, target, key, duration (optional)
"""

import os.path as osp
import random 
import json
import copy 

import numpy as np
import soundfile as sf

import torch
from torch.utils.data import Dataset
import whisper
from utils.compute_utils import calculate_output_length_1d


class SpeechDatasetJsonl(torch.utils.data.Dataset):
    """
    Dataset for Speech-to-Text with LLM.
    
    Supports:
    - Variable-length audio (no padding to 30s)
    - Multiple input types: raw waveform or mel spectrogram
    - Training and inference modes
    """
    
    def __init__(self,
                 dataset_config,
                 tokenizer=None,
                 split='train',
                 ):
        super().__init__()
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        data_parallel_size = 1
        
        self.IGNORE_INDEX = -100  # CrossEntropyLoss ignore index
        self.prompt = None
        self.mel_size = getattr(dataset_config, 'mel_size', 80)  # 80 for whisper base/small/medium
        
        # Simple prompt without chat format - works better for ASR
        # Avoid USER/ASSISTANT format that can conflict with LLM's chat template
        self.prompt_template = "{}\n"
        self.answer_template = "{}"
        
        self.fix_length_audio = getattr(dataset_config, 'fix_length_audio', -1)
        self.inference_mode = getattr(dataset_config, 'inference_mode', False)
        self.normalize = getattr(dataset_config, 'normalize', False)
        self.input_type = getattr(dataset_config, 'input_type', 'mel')
        
        # Projector downsampling rate - MUST match model config!
        self.projector_ds_rate = getattr(dataset_config, 'projector_ds_rate', 5)
        
        # Variable length support (important for Common Voice)
        self.use_variable_length = getattr(dataset_config, 'use_variable_length', True)
        self.max_audio_length = getattr(dataset_config, 'max_audio_length', 30)  # seconds
        
        # Max target text length - filter out corrupted samples with TSV data in target
        # For 20s audio at ~150 words/min = ~50 words = ~300 chars max reasonable
        self.max_target_chars = getattr(dataset_config, 'max_target_chars', 500)
        
        assert self.input_type in ["raw", "mel"], "input_type must be one of [raw, mel]" 

        # Load data from JSONL
        self.data_list = []
        skipped = 0
        if split == "train":
            data_path = dataset_config.train_data_path
        elif split == "test":
            data_path = dataset_config.test_data_path
        else:  # validation
            data_path = dataset_config.val_data_path
            
        print(f"[Dataset] Loading {split} data from: {data_path}")
        with open(data_path, encoding='utf-8') as fin:
            for line in fin:
                data_dict = json.loads(line.strip())
                # Filter out corrupted samples (TSV data in target field)
                target_len = len(data_dict.get('target', ''))
                if target_len > self.max_target_chars:
                    skipped += 1
                    continue
                self.data_list.append(data_dict)
        
        if skipped > 0:
            print(f"[Dataset] WARNING: Skipped {skipped} samples with target > {self.max_target_chars} chars (corrupted data)")
        print(f"[Dataset] Loaded {len(self.data_list)} samples for {split}")

    def get_source_len(self, data_dict):
        return data_dict.get("source_len", 0)

    def get_target_len(self, data_dict):
        return data_dict.get("target_len", 0)
    
    def __len__(self):
        return len(self.data_list)
    
    def _load_audio(self, audio_path):
        """Load audio file and return numpy array at 16kHz."""
        audio_raw = whisper.load_audio(audio_path)  # Always returns 16kHz
        return audio_raw
    
    def _compute_mel_spectrogram(self, audio_raw):
        """Compute mel spectrogram, handling variable-length audio."""
        if self.use_variable_length:
            # Pad or trim to max length, but track actual length
            max_samples = int(self.max_audio_length * 16000)
            if len(audio_raw) > max_samples:
                audio_raw = audio_raw[:max_samples]
            # Pad to nearest second for cleaner processing
            target_len = min(max_samples, ((len(audio_raw) // 16000) + 1) * 16000)
            if len(audio_raw) < target_len:
                audio_raw = np.pad(audio_raw, (0, target_len - len(audio_raw)))
        else:
            # Original behavior: pad/trim to exactly 30 seconds
            audio_raw = whisper.pad_or_trim(audio_raw)
        
        audio_mel = whisper.log_mel_spectrogram(audio_raw, n_mels=self.mel_size).permute(1, 0)
        return audio_mel, audio_raw
    
    def __getitem__(self, index):
        data_dict = self.data_list[index]
        audio_path = data_dict.get("source")
        target = data_dict.get("target", None)
        task = data_dict.get("prompt", "ASR")
        key = data_dict.get("key", None)

        # Load audio
        audio_raw = self._load_audio(audio_path)
        
        if self.input_type == "raw":
            audio_raw = torch.from_numpy(audio_raw)
            if self.normalize:
                audio_raw = torch.nn.functional.layer_norm(audio_raw, audio_raw.shape)
            audio_length = len(audio_raw) // 320  # for fairseq 320x downsample
            audio_length = audio_length // self.projector_ds_rate  # projector downsample
            audio_mel = None
        elif self.input_type == "mel":
            audio_mel, audio_raw = self._compute_mel_spectrogram(audio_raw)
            audio_length = (audio_mel.shape[0] + 1) // 2  # whisper 2x downsample
            audio_length = audio_length // self.projector_ds_rate  # projector downsample
            
        if self.fix_length_audio > 0:
            audio_length = self.fix_length_audio
            
        audio_pseudo = torch.full((audio_length,), -1)  # placeholder

        # Prompt handling
        prompt = self.prompt
        if prompt is None:
            prompt = "Transcribe speech to text."
        prompt = self.prompt_template.format(prompt)
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_length = len(prompt_ids)

        if self.inference_mode:
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64)
            example_ids = torch.cat((audio_pseudo, prompt_ids))
            example_mask = example_ids.ge(-1)

            return {
                "input_ids": example_ids,
                "attention_mask": example_mask,
                "audio": audio_raw if self.input_type == "raw" else None,
                "audio_mel": audio_mel if self.input_type == "mel" else None,
                "audio_length": audio_length,
                "key": key,
                "target": target,
                "prompt_length": prompt_length,
            }

        # Training mode
        answer = self.answer_template.format(target)
        example = prompt + answer
        example_ids = self.tokenizer.encode(example)
        example_ids.append(self.tokenizer.eos_token_id)
        example_ids = torch.tensor(example_ids, dtype=torch.int64)
        example_ids = torch.cat((audio_pseudo, example_ids))

        labels_ids = copy.deepcopy(example_ids)
        labels_ids[:audio_length + prompt_length] = -1
        example_mask = example_ids.ge(-1)

        label_mask = labels_ids.ge(0)
        example_ids[~example_mask] = 0
        labels_ids[~label_mask] = self.IGNORE_INDEX

        return {
            "input_ids": example_ids,
            "labels": labels_ids,
            "attention_mask": example_mask,
            "audio": audio_raw if self.input_type == "raw" else None,
            "audio_mel": audio_mel if self.input_type == "mel" else None,
            "audio_length": audio_length,
            "prompt_length": prompt_length,
        }

    def pad(self, sequence, max_length, padding_idx=0):
        """Pad sequence to max_length."""
        if isinstance(sequence, (int, list, tuple)):
            if len(sequence) < max_length:
                sequence = sequence + [padding_idx] * (max_length - len(sequence))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, torch.Tensor):
            if len(sequence) < max_length:
                sequence = torch.cat(
                    (sequence, torch.full(([max_length - len(sequence)] + list(sequence.size())[1:]), padding_idx)))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, np.ndarray):
            if len(sequence) < max_length:
                sequence = np.concatenate(
                    (sequence, np.full((max_length - len(sequence),) + sequence.shape[1:], padding_idx)))
            else:
                sequence = sequence[:max_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence
        
    @classmethod
    def padding(cls, sequence, padding_length, padding_idx=0, padding_side="right"):
        """Add padding to sequence."""
        if isinstance(sequence, (int, list, tuple)):
            if padding_length >= 0:
                sequence = sequence + [padding_idx] * padding_length
            else:
                sequence = sequence[:padding_length]
        elif isinstance(sequence, torch.Tensor):
            if sequence.ndimension() == 2:
                if padding_length >= 0:
                    sequence = torch.nn.functional.pad(sequence, (0, padding_length))
                else:
                    sequence = sequence[:, :padding_length]
            else:
                if padding_length >= 0:
                    if padding_side == "left":
                        sequence = torch.cat((torch.full(([padding_length] + list(sequence.size())[1:]), padding_idx), sequence))
                    else:
                        sequence = torch.cat((sequence, torch.full(([padding_length] + list(sequence.size())[1:]), padding_idx)))
                else:
                    sequence = sequence[:padding_length]
        elif isinstance(sequence, np.ndarray):
            if padding_length >= 0:
                sequence = np.concatenate(
                    (sequence, np.full((padding_length,) + sequence.shape[1:], padding_idx)))
            else:
                sequence = sequence[:padding_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence

    def collator(self, samples):
        """Collate samples into a batch."""
        assert samples is not None 
        
        # Maximum sequence length to prevent OOM (vocab=152k needs ~0.3GB per 1000 tokens per batch)
        MAX_SEQ_LENGTH = 512  # Safety limit
        
        input_prompt_lengths = [s["audio_length"] + s['prompt_length'] for s in samples]
        input_answer_lengths = [len(s["input_ids"]) - s["audio_length"] - s['prompt_length'] for s in samples]

        input_prompt_max_length = max(input_prompt_lengths)
        input_answer_max_length = max(input_answer_lengths)
        
        # DEBUG: Check for unexpectedly large sequences
        total_max_len = input_prompt_max_length + input_answer_max_length
        if total_max_len > 500:
            print(f"\n[DEBUG COLLATOR] WARNING: Large sequence detected!")
            print(f"[DEBUG COLLATOR] input_prompt_max_length: {input_prompt_max_length}")
            print(f"[DEBUG COLLATOR] input_answer_max_length: {input_answer_max_length}")
            print(f"[DEBUG COLLATOR] audio_lengths: {[s['audio_length'] for s in samples]}")
            print(f"[DEBUG COLLATOR] prompt_lengths: {[s['prompt_length'] for s in samples]}")
            print(f"[DEBUG COLLATOR] input_ids_lengths: {[len(s['input_ids']) for s in samples]}")
        
        # Truncate to prevent OOM
        if total_max_len > MAX_SEQ_LENGTH:
            print(f"[DEBUG COLLATOR] Truncating from {total_max_len} to {MAX_SEQ_LENGTH}")
            # Proportionally reduce both parts
            scale = MAX_SEQ_LENGTH / total_max_len
            input_prompt_max_length = int(input_prompt_max_length * scale)
            input_answer_max_length = MAX_SEQ_LENGTH - input_prompt_max_length
        
        input_ids = torch.stack([
            self.padding(
                self.padding(samples[index]["input_ids"], input_prompt_max_length - input_prompt_lengths[index], self.tokenizer.pad_token_id, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], self.tokenizer.pad_token_id
            ) for index in range(len(samples))
        ])

        attention_mask = torch.stack([
            self.padding(
                self.padding(samples[index]["attention_mask"], input_prompt_max_length - input_prompt_lengths[index], False, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], False
            ) for index in range(len(samples))
        ])

        if self.input_type == "raw":
            audio_raw_max_length = max([s['audio'].shape[0] for s in samples])
            audio_raw = torch.stack([self.pad(s['audio'], audio_raw_max_length, 0)
                                     for s in samples])
            audio_mask = torch.zeros(len(samples), audio_raw_max_length)
            for line, sample in enumerate(samples):
                audio_mask[line, :sample['audio'].shape[0]] = 1
            audio_mel = None
            audio_mel_post_mask = None
        elif self.input_type == "mel":
            audio_mel_max_length = max([s['audio_mel'].shape[0] for s in samples])
            
            # DEBUG: Check audio_mel sizes
            if audio_mel_max_length > 2000:
                print(f"\n[DEBUG COLLATOR] WARNING: Large audio_mel detected!")
                print(f"[DEBUG COLLATOR] audio_mel_max_length: {audio_mel_max_length}")
                print(f"[DEBUG COLLATOR] audio_mel shapes: {[s['audio_mel'].shape for s in samples]}")
            
            audio_mel = torch.stack([self.pad(s['audio_mel'], audio_mel_max_length, 0)
                                  for s in samples])
            audio_mel_post_mask = torch.zeros(len(samples), (audio_mel_max_length + 1) // 2)
            for line, sample in enumerate(samples):
                audio_mel_post_mask[line, :(sample['audio_mel'].shape[0] + 1) // 2] = 1
            audio_raw = None
            audio_mask = None
    
        modality_mask = torch.zeros_like(attention_mask)
        for index in range(len(samples)):
            padding_left = input_prompt_max_length - input_prompt_lengths[index]
            modality_mask[index, padding_left:padding_left+samples[index]["audio_length"]] = True

        if self.inference_mode:
            keys = [s['key'] for s in samples]
            targets = [s['target'] for s in samples]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "audio": audio_raw,
                "audio_mask": audio_mask,
                "audio_mel": audio_mel,
                "audio_mel_post_mask": audio_mel_post_mask,
                "modality_mask": modality_mask,
                "keys": keys,
                "targets": targets
            }

        labels = torch.stack([
            self.padding(
                self.padding(samples[index]['labels'], input_prompt_max_length - input_prompt_lengths[index], self.IGNORE_INDEX, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], self.IGNORE_INDEX)
            for index in range(len(samples))
        ])
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "audio": audio_raw,
            "audio_mask": audio_mask,
            "audio_mel": audio_mel,
            "audio_mel_post_mask": audio_mel_post_mask,
            "modality_mask": modality_mask
        }


def get_speech_dataset(dataset_config, tokenizer, split):
    """Factory function to create speech dataset."""
    dataset = SpeechDatasetJsonl(dataset_config, tokenizer, split)
    return dataset