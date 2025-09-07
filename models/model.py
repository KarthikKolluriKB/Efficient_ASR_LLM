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
from utils.metrics import compute_accuracy
from utils.train_utils import print_model_size, print_module_size

logger = logging.getLogger(__name__)


def model_factory(train_config, model_config, **kwargs):
    """"""
    # 1. tokenizer
    tokenizer = setup_tokenizer(train_config, model_config, **kwargs)

    # 2. encoder 
    encoder = setup_encoder(train_config, model_config, **kwargs)

    # 3. llm 
    llm = setup_llm(train_config, model_config, **kwargs)

    # 4. projector 
    encoder_projector = setup_encoder_projector(train_config, model_config, **kwargs).to(torch.bfloat16)

    # 5. model
    model = ASRLLM(
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        train_config,
        model_config,
        **kwargs
    )

    # load ckpt 
    ckpt_path = kwargs.get("ckpt_path", None) 

    if ckpt_path is not None:
        logger.info(f"Load checkpoint from {ckpt_path}")
        ckpt_dir = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt_dir, strict=False)
    
    print_model_size(model, train_config)

    return model, tokenizer


def setup_tokenizer(train_config, model_config, **kwargs):
    # Load the tokenizer and special tokens
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.llm_path
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def setup_encoder(train_config, model_config, **kwargs):
    
    # whisper encoder
    encoder_name = model_config.encoder_name
    
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
            model_config.llm_path,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
    )

    print_module_size(model, model_config.llm_path)

    if train_config.quantization:
        model = prepare_model_for_kbit_training(model)

    if train_config.freeze_llm: # TODO:to test offical `freeze_layers` and `num_freeze_layers`
        for name, param in model.named_parameters(): 
            param.requires_grad = False
        model.eval()

     # TODO: No PEFT

    return model



def setup_encoder_projector(train_config, model_config, **kwargs):
    if model_config.encoder_projector == "linear":
        from models.projector import EncoderProjectorConcat
        encoder_projector = EncoderProjectorConcat(model_config)
    elif model_config.encoder_projector == "cov1d-linear":
        from models.projector import EncoderProjectorCov1d
        encoder_projector = EncoderProjectorCov1d(model_config)
    elif model_config.encoder_projector == "q-former":
        from models.projector import EncoderProjectorQFormer
        encoder_projector = EncoderProjectorQFormer(model_config)
    else:
        return None
    print_module_size(encoder_projector, model_config.encoder_projector)
    return encoder_projector


class ASRLLM(nn.Module):
    
    def __init__(self,
                 encoder: nn.Module,
                 llm: nn.Module,
                 encoder_projector: Optional[nn.Module],
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
        self.encoder_projector = encoder_projector

        # tokenizer
        self.tokenizer = tokenizer
        self.metric = kwargs.get("metric", None)

        self.train_config = train_config
        self.model_config = model_config

    

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

        encoder_outputs = None 

        # Freeze encoder 
        if getattr(self.train_config, "freeze_encoder", False):
            self.encoder.eval()
            context = torch.no_grad()

        # 1. Whisper encode audio -> [B, n_mels, T] -> permute -> [B, T, n_mels] for var-length 
        with context:
            encoder_outputs = self.encoder(audio_mel.permute(0, 2, 1)).last_hidden_state # [B, T_enc, D]

        if audio_mel_post_mask is None:
            audio_mel_post_mask = torch.ones(encoder_outputs.size()[:-1], dtype=torch.long, device=encoder_outputs.device) # [B, T_enc]

        # 2. Projector (Project to LLM embedding space)
        if self.model_config.encoder_projector == "q-former":
            # Q-former
            encoder_outputs = self.encoder_projector(encoder_outputs, audio_mel_post_mask) # [B, T_enc_proj, D_llm]

        elif self.model_config.encoder_projector in ["linear", "cov1d-linear"]:
            # linear or conv1d + linear
            encoder_outputs = self.encoder_projector(encoder_outputs)  # [B, T_enc_proj, D_llm]


        # 3. Token embedding 
        token_embeds = None 
        if input_ids is not None: 
            # Santize any placeholder ids for embedding lookup
            input_ids = input_ids.clone()
            input_ids[input_ids == -1] = 0
    
            # Resolve embedding layer across model architectures
            if hasattr(self.llm, 'model') and hasattr(self.llm.model, "embed_tokens"):
                token_embeds = self.llm.model.embed_tokens(input_ids)
            elif hasattr(self.llm, "model") and hasattr(self.llm.model, "model") and hasattr(self.llm.model.model, "embed_tokens"):
                token_embeds = self.llm.model.model.embed_tokens(input_ids)
            else:
                token_embeds = self.llm.get_input_embeddings()(input_ids)


        # 4. Concat encoder feature and token embeddings (audio prefix + text token embeddings)
        if token_embeds is not None:
            inputs_embeds = torch.cat([encoder_outputs, token_embeds], dim=1) # [B, T_audio_enc + T_text_tok, D_llm]
        else: 
            inputs_embeds = encoder_outputs

        # 5. Build attention mask aligned with inputs_embeds
        B, T_total, _  = inputs_embeds.size()
        T_audio = encoder_outputs.size(1)
        if attention_mask is not None and input_ids is not None:
            audio_attn = torch.ones((B, T_audio), dtype=attention_mask.dtype, device=inputs_embeds.device)
            attention_mask = torch.cat([audio_attn, attention_mask], dim=1) # [B, T_audio + T_text_tok]
        else: 
            attention_mask = torch.ones((B, T_total), dtype=torch.long, device=inputs_embeds.device)

        # 6. Labels: ignore loss on audio prefix (mask with ignore_index e.g: -100)
        if labels is not None: 
            if labels.dim() != 2: 
                raise ValueError("Labels should be of shape 2D (B, T_text) for cross-entropy loss.")
            ignore_pad = torch.full((B, T_audio), -100, dtype=labels.dtype, device=labels.device)
            labels = torch.cat([ignore_pad, labels], dim=1) # [B, T_audio + T_text]

        # 7. Forward LLM

        # Fast path for generation setup
        if kwargs.get("inference_mode", False): 
            return inputs_embeds, attention_mask
        
        # Forward through LLM using inputs_embeds only 
        llm_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "labels": labels,
            "use_cache": use_cache,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }

        llm_kwargs = {k: v for k, v in llm_kwargs.items() if v is not None}

        model_outputs = self.llm(**llm_kwargs)

        # 8. Compute additional metrics
        acc = -1 
        if self.metric and labels is not None and hasattr(model_outputs, "logits"):
            with torch.no_grad():
                preds = torch.argmax(model_outputs.logits, dim=-1) # [B, T] 
                acc = compute_accuracy(preds.detach()[:, :-1], labels.detach()[:, : -1], ignore_index=-100)

        return model_outputs, acc
