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

    # FIXME: temporarily force dtype to bfloat16 for projector to match Whisper encoder output dtype
    # 4. projector 
    projector = setup_projector(train_config, model_config, **kwargs).to(torch.bfloat16)
    #encoder_projector = setup_encoder_projector(train_config, model_config, **kwargs) # fp32

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
        
        # Initialize metric flag for accuracy computation
        self.metric = kwargs.get("metric", True)  # Default to True if not specified





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

        encoder_outputs = None 

        # Freeze encoder 
        if getattr(self.train_config, "freeze_encoder", False):
            self.encoder.eval()
            context = torch.no_grad()
        else:
            context = torch.enable_grad()

        # 1. Whisper encode audio -> [B, n_mels, T] -> permute -> [B, T, n_mels] for var-length 
        with context:
            encoder_outputs = self.encoder(audio_mel.permute(0, 2, 1)).last_hidden_state # [B, T_enc, D]

        if audio_mel_post_mask is None:
            audio_mel_post_mask = torch.ones(encoder_outputs.size()[:-1], dtype=torch.long, device=encoder_outputs.device) # [B, T_enc]

        # determine the dtypes 
        proj_dtype = next(self.projector.parameters()).dtype
        llm_dtype  = next(self.llm.parameters()).dtype

        # 2. Projector (Project to LLM embedding space)
        if self.model_config.projector == "q-former":
            # Q-former
            encoder_outputs = self.projector(encoder_outputs, audio_mel_post_mask) # [B, T_enc_proj, D_llm]

        elif self.model_config.projector in ["linear", "cov1d-linear"]:
            # linear or conv1d + linear
            encoder_outputs = self.projector(encoder_outputs)  # [B, T_enc_proj, D_llm]


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
        
        # Cast to LLM dtype (e.g., bfloat16)
        llm_dtype = next(self.llm.parameters()).dtype
        # Cast encoder outputs to LLM dtype if different (e.g., bfloat16)
        if encoder_outputs.dtype != llm_dtype:
            encoder_outputs = encoder_outputs.to(llm_dtype)
        # Cast token embeddings to LLM dtype if different (e.g., bfloat16)
        if token_embeds is not None and token_embeds.dtype != llm_dtype:
            token_embeds = token_embeds.to(llm_dtype)

        # 4. Concat encoder feature and token embeddings (audio prefix + text token embeddings)
        if token_embeds is not None:
             # Ensure encoder outputs and token embeddings have compatible shapes
            B, T_enc, D = encoder_outputs.size()
            B, T_tok, D = token_embeds.size()
            
            inputs_embeds = torch.cat([encoder_outputs, token_embeds], dim=1) # [B, T_audio_enc + T_text_tok, D_llm]

             # Ensure labels are properly aligned with the concatenated sequence
            if labels is not None:
                # Pad labels to match the total sequence length
                total_length = T_enc + T_tok
                if labels.size(1) < total_length:
                    labels = F.pad(labels, (0, total_length - labels.size(1)), value=-100)
                elif labels.size(1) > total_length:
                    labels = labels[:, :total_length]
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

        # # 6. Align encoder outputs using modality mask
        # if modality_mask is not None:
        #     modality_mask_start_indices = (modality_mask == True).float().argmax(dim=1)
        #     modality_lengths = torch.clamp(modality_mask.sum(dim=1), max=encoder_outputs.shape[1]).tolist()

        #     encoder_outputs_pad = torch.zeros_like(inputs_embeds)
        #     for i in range(encoder_outputs.shape[0]):
        #         encoder_outputs_pad[
        #             i, modality_mask_start_indices[i]:modality_mask_start_indices[i]+modality_lengths[i]
        #         ] = encoder_outputs[i][:modality_lengths[i]]
            
        #     inputs_embeds = encoder_outputs_pad + inputs_embeds * (~modality_mask[:, :, None])

        # 7. Forward LLM

        # Fast path for generation setup
        if kwargs.get("inference_mode", False): 
            return inputs_embeds, attention_mask
        
        # Default path for training / evaluation
        model_outputs =  self.llm(inputs_embeds=inputs_embeds, 
                                  attention_mask=attention_mask, 
                                  labels=labels)
        # Metrics
        metrics = {}
        if self.metric:
            with torch.no_grad():
                # Compute token accuracy
                preds = torch.argmax(model_outputs.logits, -1)
                acc = compute_accuracy(preds.detach()[:, :-1], labels.detach()[:, 1:], ignore_label=-100)
                metrics["acc"] = float(acc.item())

                # Compute WER
                if hasattr(model_outputs, "logits"):
                    # First decode the texts using decode_texts_from_outputs
                    hyp_texts, ref_texts = decode_texts_from_outputs(
                        logits=model_outputs.logits,
                        labels=labels,
                        tokenizer=self.tokenizer,
                        ignore_label=-100
                    )
                    
                    # Then compute WER using the decoded texts
                    wer_score = compute_wer(
                        hyp_texts=hyp_texts,
                        ref_texts=ref_texts
                    )
                    metrics["wer"] = float(wer_score)

        return model_outputs, metrics
    
    @torch.no_grad()
    def inference(
        self,
        audio_mel: torch.Tensor, # [B, n_mels, T] or [n_mels, T]
        prompt: str = "",
        max_new_tokens: int = 64, 
        temperature: float = 0.7, 
        top_p: float = 0.9,
        num_beams: int = 1, 
        do_sample: bool = None,
        device: str = None, 
        **gen_kwargs
    ):
        """
        Run Batched Inference. Assumes audio_mel is already preprocessed.
        """
        # 1. Ensure dtype/device consistency 
        if device is None:
            device = next(self.parameters()).device

        llm_dtype = next(self.llm.parameters()).dtype

        if audio_mel.dim() == 2: 
            audio_mel = audio_mel.unsqueeze(0)

        audio_mel = audio_mel.to(device, dtype=llm_dtype)

        # 2. Tokenize prompt 
        input = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = input["input_ids"].to(device)
        attention_mask = input["attention_mask"].to(device)
        prompt_len = input_ids.shape[1]

        # 3. Forward to get input_embeds for generation 
        inputs_embeds, attn_mask = self.forward(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            audio_mel=audio_mel,
            inference_mode = True
        )

        # 4. Decide on sampling vs beam search
        if do_sample is None: 
            do_sample = num_beams == 1 and temperature > 0.0 

        # 5. Generate tokens from LLM 
        gen_ids = self.llm.generate(
            inputs_embeds=inputs_embeds, 
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams if not do_sample else 1, 
            eos_token_id = self.tokenizer.eos_token_id,
            pad_token_id = self.tokenizer.pad_token_id,
            use_cache=True,
            **gen_kwargs
        )

        # Strip prompt (as LLM sees concatenated audio prefix + prompt) 
        full_token_ids = gen_ids[:, prompt_len:]

        # Decode to text 
        texts = self.tokenizer.batch_decode(full_token_ids, skip_special_tokens=True)

        if audio_mel.size(0) == 1: 
            texts = texts[0]

        return texts