import os 
import types 
import torch 
import soundfile as sf
import torch.nn as nn
import logging 
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch.distributed as dist
from typing import List, Optional, Tuple, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from peft import PeftModel, PeftConfig


from models.encoder import WhisperWrappedEncoder
from utils.metrics import compute_accuracy, compute_wer, decode_texts_from_outputs
from utils.train_utils import print_model_size, print_module_size

logger = logging.getLogger(__name__)


def model_builder(train_config, model_config, **kwargs):
    """"""
    # 1. tokenizer
    tokenizer = setup_tokenizer(train_config, model_config, **kwargs)

    # 2. encoder 
    encoder = setup_encoder(train_config, model_config, **kwargs)

    # 3. llm 
    llm = setup_llm(train_config, model_config, **kwargs)

    # 4. projector - Keep in FP32 for stable training
    # The autocast in train.py will handle mixed precision during forward/backward
    # Optimizer will maintain FP32 weights for precise gradient accumulation
    projector = setup_projector(train_config, model_config, **kwargs)  # FP32

    # 5. model
    model = ASRLLM(
        encoder,
        llm,
        projector,
        tokenizer,
        train_config,
        model_config,
        **kwargs
    )

    # load ckpt 
    ckpt_path = kwargs.get("ckpt_path", None) 
    # TODO: check models is loading correctly
    if ckpt_path is not None:
        logger.info(f"Load checkpoint from {ckpt_path}")
        ckpt_dir = torch.load(ckpt_path, map_location="cpu")
        model.projector.load_state_dict(ckpt_dir['projector'], strict=True)
    
    print_model_size(model, train_config)

    return model, tokenizer


def setup_tokenizer(train_config, model_config, **kwargs):
    # Load the tokenizer and special tokens
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.llm_model
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def setup_encoder(train_config, model_config, **kwargs):
    
    # whisper encoder
    encoder_name = model_config.encoder_model_name
    
    encoder = WhisperWrappedEncoder.load(model_config)

    print_module_size(encoder, encoder_name)

    if train_config.freeze_encoder:
        for name, params in encoder.named_parameters():
            params.requires_grad = False
        encoder.eval()
    print_module_size(encoder, encoder_name)

    return encoder


def setup_llm(train_config, model_config, **kwargs):
    
    model = AutoModelForCausalLM.from_pretrained(
            model_config.llm_model,
            torch_dtype=torch.bfloat16 if train_config.mixed_precision else torch.float32,
            attn_implementation="sdpa",
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
    )

    print_module_size(model, model_config.llm_model_name)

    if train_config.quantization:
        model = prepare_model_for_kbit_training(model)

    if train_config.freeze_llm: 
        for name, param in model.named_parameters(): 
            param.requires_grad = False
        model.eval()

     # TODO: No PEFT

    return model



def setup_projector(train_config, model_config, **kwargs):
    if model_config.projector == "linear":
        from models.projector import EncoderProjectorConcat
        projector = EncoderProjectorConcat(model_config)
    elif model_config.projector == "cov1d-linear":
        from models.projector import EncoderProjectorCov1d
        projector = EncoderProjectorCov1d(model_config)
    elif model_config.projector == "q-former":
        from models.projector import EncoderProjectorQFormer
        projector = EncoderProjectorQFormer(model_config)
    else:
        return None
    print_module_size(projector, model_config.projector)
    return projector


class ASRLLM(nn.Module):
    
    def __init__(self,
                 encoder: nn.Module,
                 llm: nn.Module,
                 projector: Optional[nn.Module],
                 tokenizer,
                 train_config,
                 model_config,
                 **kwargs
    ):
        super().__init__()

        # encoder
        self.encoder = encoder

        # llm 
        self.llm = llm

        # projector
        self.projector = projector

        # tokenizer
        self.tokenizer = tokenizer

        self.train_config = train_config
        self.model_config = model_config
        self.dataset_config = kwargs.get("data_config", None)
        
        # Initialize metric flag for accuracy computation
        self.metric = kwargs.get("metric", True)  # Default to True if not specified
        
        # Label smoothing for regularization (reduces overconfidence)
        self.label_smoothing = getattr(train_config, 'label_smoothing', 0.0)
        if self.label_smoothing > 0:
            logger.info(f"Label smoothing enabled: {self.label_smoothing}")


    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ):
        
        audio_mel = kwargs.get("audio_mel", None)
        audio_mel_mask = kwargs.get("audio_mel_mask", None)
        audio_mel_post_mask = kwargs.get("audio_mel_post_mask", None) # optional, downsampled mask for whisper
        audio = kwargs.get("audio", None)
        audio_mask = kwargs.get("audio_mask", None)

        modality_mask = kwargs.get("modality_mask", None)

        # Freeze encoder
        if getattr(self.train_config, "freeze_encoder", False):
            self.encoder.eval()
            context = torch.no_grad()
        else:
            context = torch.enable_grad()

        # 1. Whisper encode audio -> [B, n_mels, T] -> permute -> [B, T, n_mels] for var-length 
        with context:
            encoder_outputs = self.encoder.extract_variable_length_features(audio_mel.permute(0, 2, 1)) # bs*seq*dim
        
        # DEBUG: Print shapes to diagnose OOM (always print first batch of each mode)
        debug_key = f"_debug_{labels is not None}"  # Different key for train vs eval
        if not hasattr(self, debug_key):
            mode = "TRAIN" if labels is not None else "EVAL"
            print(f"\n[DEBUG MODEL {mode}] audio_mel input shape: {audio_mel.shape}")
            print(f"[DEBUG MODEL {mode}] encoder_outputs shape: {encoder_outputs.shape}")
            setattr(self, debug_key, True)

        if audio_mel_post_mask is None:
            audio_mel_post_mask = torch.ones(encoder_outputs.size()[:-1], dtype=torch.long, device=encoder_outputs.device) # [B, T_enc]

        # Note: With autocast enabled in train.py, PyTorch will automatically handle
        # precision for matmul operations. Projector weights are FP32 for stable
        # optimizer updates, but autocast may run matmuls in lower precision.
        # This is the correct mixed precision pattern.

        # 2. Projector (Project to LLM embedding space)
        if self.model_config.projector == "q-former":
            # Q-former
            encoder_outputs = self.projector(encoder_outputs, audio_mel_post_mask) # [B, T_enc_proj, D_llm]

        elif self.model_config.projector in ["linear", "cov1d-linear"]:
            # linear or conv1d + linear
            encoder_outputs = self.projector(encoder_outputs)  # [B, T_enc_proj, D_llm]
        
        # DEBUG: Print projected shape
        debug_key2 = f"_debug_proj_{labels is not None}"
        if not hasattr(self, debug_key2):
            mode = "TRAIN" if labels is not None else "EVAL"
            print(f"[DEBUG MODEL {mode}] encoder_outputs after projector: {encoder_outputs.shape}")
            setattr(self, debug_key2, True)


        # 3. Token embedding 
        if input_ids is not None: 
            # Santize any placeholder ids for embedding lookup
            input_ids[input_ids == -1] = 0
    
            # Resolve embedding layer across model architectures
            if hasattr(self.llm, 'model') and hasattr(self.llm.model, "embed_tokens"):
                inputs_embeds = self.llm.model.embed_tokens(input_ids)
            elif hasattr(self.llm, "model") and hasattr(self.llm.model, "model") and hasattr(self.llm.model.model, "embed_tokens"):
                inputs_embeds = self.llm.model.model.embed_tokens(input_ids)
            else:
                inputs_embeds = self.llm.model.model.model.embed_tokens(input_ids)

        
        if modality_mask is not None:
            modality_mask_start_indices = (modality_mask == True).float().argmax(dim=1)
            modality_lengths = torch.clamp(modality_mask.sum(dim=1), max=encoder_outputs.shape[1]).tolist()
            
            # DEBUG: Check modality mask alignment
            debug_key_mod = f"_debug_modality_{labels is not None}"
            if not hasattr(self, debug_key_mod):
                mode = "TRAIN" if labels is not None else "EVAL"
                print(f"[DEBUG MODEL {mode}] modality_mask shape: {modality_mask.shape}")
                print(f"[DEBUG MODEL {mode}] modality_mask_start_indices: {modality_mask_start_indices.tolist()}")
                print(f"[DEBUG MODEL {mode}] modality_lengths (clamped to encoder output): {modality_lengths}")
                print(f"[DEBUG MODEL {mode}] encoder_outputs shape for replacement: {encoder_outputs.shape}")
                print(f"[DEBUG MODEL {mode}] Original modality_mask True counts: {modality_mask.sum(dim=1).tolist()}")
                setattr(self, debug_key_mod, True)

            encoder_outs_pad = torch.zeros_like(inputs_embeds)
            for i in range(encoder_outputs.shape[0]):
                encoder_outs_pad[
                    i, modality_mask_start_indices[i]:modality_mask_start_indices[i]+modality_lengths[i]
                ] = encoder_outputs[i][:modality_lengths[i]]
            
            # DEBUG: Verify embedding replacement
            debug_key_emb = f"_debug_embedding_{labels is not None}"
            if not hasattr(self, debug_key_emb):
                mode = "TRAIN" if labels is not None else "EVAL"
                # Check if encoder embeddings are actually being used
                sample_idx = 0
                start = int(modality_mask_start_indices[sample_idx])
                length = int(modality_lengths[sample_idx])
                text_embed_norm = inputs_embeds[sample_idx, start:start+5].norm().item()
                enc_embed_norm = encoder_outs_pad[sample_idx, start:start+5].norm().item()
                final_audio_norm = (encoder_outs_pad + inputs_embeds * (~modality_mask[:, :, None]))[sample_idx, start:start+5].norm().item()
                final_text_norm = (encoder_outs_pad + inputs_embeds * (~modality_mask[:, :, None]))[sample_idx, start+length:start+length+5].norm().item()
                print(f"[DEBUG MODEL {mode}] Sample 0 - Audio region norms:")
                print(f"  Text embedding at audio pos (should be zeroed): {text_embed_norm:.4f}")
                print(f"  Encoder embedding at audio pos: {enc_embed_norm:.4f}")
                print(f"  Final embedding at audio pos (should be encoder): {final_audio_norm:.4f}")
                print(f"  Final embedding at text pos (prompt): {final_text_norm:.4f}")
                # Check if modality_mask is correct
                print(f"  modality_mask at audio positions: {modality_mask[sample_idx, start:start+5].tolist()}")
                print(f"  modality_mask at text positions: {modality_mask[sample_idx, start+length:start+length+5].tolist()}")
                setattr(self, debug_key_emb, True)
            
            inputs_embeds = encoder_outs_pad + inputs_embeds * (~modality_mask[:, :, None])
        
        # DEBUG: Print final inputs_embeds shape going into LLM
        debug_key3 = f"_debug_embed_{labels is not None}"
        if not hasattr(self, debug_key3):
            mode = "TRAIN" if labels is not None else "EVAL"
            print(f"[DEBUG MODEL {mode}] input_ids shape: {input_ids.shape if input_ids is not None else 'None'}")
            print(f"[DEBUG MODEL {mode}] inputs_embeds shape to LLM: {inputs_embeds.shape}")
            expected_mem = inputs_embeds.shape[0] * inputs_embeds.shape[1] * 151936 * 2 / 1e9
            print(f"[DEBUG MODEL {mode}] Expected logits memory: {expected_mem:.2f} GB (bfloat16)")
            setattr(self, debug_key3, True)

        # Fast path for generation setup
        if kwargs.get("inference_mode", False): 
            return inputs_embeds, attention_mask
        
        # Default path for training / evaluation
        # Explicitly disable KV cache to prevent memory accumulation during training
        
        # If label smoothing is enabled, compute loss ourselves
        if self.label_smoothing > 0 and labels is not None:
            # Get logits without loss computation
            model_outputs = self.llm(
                inputs_embeds=inputs_embeds, 
                attention_mask=attention_mask, 
                labels=None,  # Don't compute loss in LLM
                use_cache=False
            )
            
            # Compute loss with label smoothing
            logits = model_outputs.logits
            # Shift for causal LM: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for cross entropy
            loss_fct = CrossEntropyLoss(
                ignore_index=-100, 
                label_smoothing=self.label_smoothing
            )
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
            
            # Replace the loss in model_outputs
            model_outputs.loss = loss
        else:
            model_outputs = self.llm(
                inputs_embeds=inputs_embeds, 
                attention_mask=attention_mask, 
                labels=labels,  # Can be None during eval to skip loss computation
                use_cache=False  # Critical: prevents KV cache memory accumulation
            )

        # Metrics (computed on CPU to prevent GPU memory accumulation)
        # Skip metrics when labels=None (eval mode skips loss computation to save memory)
        metrics = {}
        if self.metric and labels is not None:
            with torch.no_grad():
                # Move logits to CPU ONCE and delete GPU copy immediately
                logits_cpu = model_outputs.logits.detach().cpu()
                labels_cpu = labels.detach().cpu()
                
                # Compute token accuracy
                preds = torch.argmax(logits_cpu, dim=-1)
                acc = compute_accuracy(preds[:, :-1], labels_cpu[:, 1:], ignore_label=-100)
                metrics["acc"] = float(acc.item())

                # Compute WER (batch-level average) - use CPU tensors
                hyp_texts, ref_texts = decode_texts_from_outputs(
                    logits=logits_cpu,
                    labels=labels_cpu,
                    tokenizer=self.tokenizer,
                    ignore_label=-100
                )
                
                # Then compute WER using the decoded texts
                wer_score = compute_wer(
                    hyp_texts=hyp_texts,
                    ref_texts=ref_texts
                )
                metrics["wer"] = float(wer_score)
                
                # Explicitly delete CPU tensors to free memory
                del preds, labels_cpu, logits_cpu

        return model_outputs, metrics
    
    @torch.no_grad()
    def generate(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs,
                ):
        kwargs["inference_mode"] = True

        if inputs_embeds is None:
            inputs_embeds, attention_mask = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        model_outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            # max_length=kwargs.get("max_length", 200),
            max_new_tokens=kwargs.get("max_new_tokens", 200),
            num_beams=kwargs.get("num_beams", 4),
            do_sample=kwargs.get("do_sample", False),
            min_length=kwargs.get("min_length", 1),
            top_p=kwargs.get("top_p", 1.0),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
            length_penalty=kwargs.get("length_penalty", 1.0),
            temperature=kwargs.get("temperature", 1.0),
            attention_mask=attention_mask,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )

        return model_outputs
    
    @torch.no_grad()
    def inference(
        self,
        audio_path=None,
        prompt=None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=None,
        assistant_model=None,
        streamer=None,
        negative_prompt_ids=None,
        negative_prompt_attention_mask=None,
        **kwargs,
    ):
        # inference for asr model

        device = kwargs.get("device", "cuda")
        if os.path.exists(audio_path):  # Audio-Text QA
            import whisper

            audio_raw = whisper.load_audio(audio_path)
            audio_raw = whisper.pad_or_trim(audio_raw)

            mel_size = getattr(
                self.dataset_config, "mel_size", 80
            )  # 80 for large v1 and v2, 128 for large v3
            audio_mel = (
                whisper.log_mel_spectrogram(audio_raw, n_mels=mel_size)
                .permute(1, 0)[None, :, :]
                .to(device)
            )

            encoder_outs = self.encoder.extract_variable_length_features(
                audio_mel.permute(0, 2, 1)
            )

            
            projector_dtype = next(self.projector.parameters()).dtype
            encoder_outs = encoder_outs.to(projector_dtype)

            if self.model_config.projector == "q-former":
                audio_mel_post_mask = torch.ones(
                    encoder_outs.size()[:-1], dtype=torch.long
                ).to(encoder_outs.device)
                encoder_outs = self.projector(encoder_outs, audio_mel_post_mask)
            if self.model_config.projector == "linear":
                encoder_outs = self.projector(encoder_outs)
        else:  # Text QA
            encoder_outs = torch.empty(
                1, 0, self.llm.model.embed_tokens.embedding_dim
            ).to(device)

        prompt = "USER: {}\n ASSISTANT:".format(prompt)
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_length = len(prompt_ids)
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64).to(device)

        if hasattr(self.llm.model, "embed_tokens"):
            inputs_embeds = self.llm.model.embed_tokens(prompt_ids)
        elif hasattr(self.llm.model.model, "embed_tokens"):
            inputs_embeds = self.llm.model.model.embed_tokens(prompt_ids)
        else:
            inputs_embeds = self.llm.model.model.model.embed_tokens(prompt_ids)

        inputs_embeds = torch.cat(
            (encoder_outs, inputs_embeds[None, :, :]), dim=1
        )  # [audio,prompt]

        attention_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long).to(
            inputs_embeds.device
        )

        # generate
        model_outputs = self.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
        )

        # Decode output ids to text
        output_text = self.tokenizer.decode(
            model_outputs[0], skip_special_tokens=True
        )

        return output_text