import os 
import time 
import argparse 
import gc

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file (including WANDB_API_KEY)

# Fix CUDA memory fragmentation (prevents OOM on long training runs)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch 
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from omegaconf import OmegaConf

# internal imports
from utils.utils import set_seed, get_device, resolve_pad_token, ensure_dir, save_projector
from utils.log_config import get_logger
from utils.wand_config import init_wandb
from models.model import model_builder
from datamodule.dataset import get_speech_dataset
from utils.metrics import decode_texts_from_outputs, compute_wer, count_encoder_parameters
from utils.train_utils import print_model_size, print_module_size, save_and_print_examples
import sys


def log_gpu_memory(logger, step, prefix=""):
    """Log GPU memory usage for debugging OOM issues."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        logger.debug(f"{prefix}Step {step} | GPU Mem: {allocated:.2f}GB alloc, {reserved:.2f}GB reserved, {max_allocated:.2f}GB max")
        return allocated, reserved, max_allocated
    return 0, 0, 0


class EarlyStopChecker:
    """Simple early stopping checker."""
    def __init__(self, mode="min", patience=5, min_delta=0.001):
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        
    def check(self, value):
        """Returns True if training should stop."""
        if self.mode == "min":
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)
            
        if improved:
            self.best_value = value
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

def evaluate(cfg, model, dataloader, device, enc_dtype, tokenizer):
    """Evaluate the model on the given dataloader using actual generation.
    
    NOTE: The dataloader returns full sequences (audio + prompt + answer) for training.
    For generation, we must truncate to only (audio + prompt) so the model can generate.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader containing validation/test data
        device: Device to run evaluation on
        enc_dtype: Data type for encoder inputs
        
    Returns:
        tuple: Contains (validation loss, accuracy, WER score, word accuracy, 
               hypothesis texts, reference texts)
    """
    model.eval()
    
    # Temporarily disable gradient checkpointing during eval (saves memory)
    grad_ckpt_was_enabled = False
    if hasattr(model.llm, 'is_gradient_checkpointing') and model.llm.is_gradient_checkpointing:
        grad_ckpt_was_enabled = True
        model.llm.gradient_checkpointing_disable()
    
    total_loss, n_batches = 0.0, 0
    all_accuracies = []
    all_wer_scores = []
    all_hyp_texts, all_ref_texts = [], []
    
    use_autocast = bool(cfg.train.mixed_precision and getattr(device, "type", str(device)) == "cuda")
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    
    # Limit validation to first N batches for faster evaluation (optional)
    max_eval_batches = cfg.train.get("max_eval_batches", None)  # None = use all
    total_batches = len(dataloader)
    
    # Generation config for ASR
    max_new_tokens = cfg.train.get("max_new_tokens", 128)  # Max output tokens

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Optional: limit number of eval batches for faster validation
            if max_eval_batches is not None and batch_idx >= max_eval_batches:
                break
                
            # Progress indicator every 50 batches
            if batch_idx % 50 == 0:
                print(f"  Eval batch {batch_idx}/{total_batches}...", end="\r")
            
            # Move batch data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            audio_mel = batch["audio_mel"].to(device).to(enc_dtype)
            modality_mask = batch['modality_mask'].to(device)
            
            # Get reference texts from labels
            labels_cpu = labels.detach().cpu()
            ref_texts = []
            for label in labels_cpu:
                valid_tokens = label[label != -100]
                if len(valid_tokens) > 0:
                    ref_text = tokenizer.decode(valid_tokens, skip_special_tokens=True).strip()
                    ref_texts.append(ref_text)
            
            # === CRITICAL FIX: Truncate inputs for generation ===
            # The input_ids contain: [audio_placeholder] + [prompt] + [answer] + [eos]
            # For generation, we need ONLY: [audio_placeholder] + [prompt]
            # Find where answer starts: first position where labels != -100
            # Then truncate input_ids, attention_mask, and modality_mask to that position
            
            gen_input_ids_list = []
            gen_attention_mask_list = []
            gen_modality_mask_list = []
            
            for i in range(labels.shape[0]):
                # Find where the answer starts (first non-ignored label)
                label_row = labels[i]
                answer_start_positions = (label_row != -100).nonzero(as_tuple=True)[0]
                
                if len(answer_start_positions) > 0:
                    answer_start = answer_start_positions[0].item()
                else:
                    # Fallback: use full sequence (shouldn't happen)
                    answer_start = label_row.shape[0]
                
                # Truncate to only audio + prompt (exclude answer)
                gen_input_ids_list.append(input_ids[i, :answer_start])
                gen_attention_mask_list.append(attention_mask[i, :answer_start])
                gen_modality_mask_list.append(modality_mask[i, :answer_start])
            
            # Pad truncated sequences to same length
            max_gen_len = max(len(seq) for seq in gen_input_ids_list)
            gen_input_ids = torch.zeros(len(gen_input_ids_list), max_gen_len, dtype=input_ids.dtype, device=device)
            gen_attention_mask = torch.zeros(len(gen_attention_mask_list), max_gen_len, dtype=attention_mask.dtype, device=device)
            gen_modality_mask = torch.zeros(len(gen_modality_mask_list), max_gen_len, dtype=modality_mask.dtype, device=device)
            
            for i, (ids, mask, mod_mask) in enumerate(zip(gen_input_ids_list, gen_attention_mask_list, gen_modality_mask_list)):
                seq_len = len(ids)
                # Left-pad to align with original padding strategy
                gen_input_ids[i, max_gen_len - seq_len:] = ids
                gen_attention_mask[i, max_gen_len - seq_len:] = mask
                gen_modality_mask[i, max_gen_len - seq_len:] = mod_mask
            
            # Debug: print truncation info on first batch
            if batch_idx == 0:
                print(f"\n[DEBUG EVAL] Original input_ids shape: {input_ids.shape}")
                print(f"[DEBUG EVAL] Truncated gen_input_ids shape: {gen_input_ids.shape}")
                print(f"[DEBUG EVAL] First sample - answer starts at position: {(labels[0] != -100).nonzero(as_tuple=True)[0][0].item() if (labels[0] != -100).any() else 'N/A'}")
            
            # === STEP 1: Compute loss and accuracy with forward pass ===
            if use_autocast:
                with torch.autocast(device_type=device, dtype=amp_dtype):
                    model_outputs, metrics = model.forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        audio_mel=audio_mel,
                        modality_mask=modality_mask,
                        inference_mode=False
                    )
            else:
                model_outputs, metrics = model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    audio_mel=audio_mel,
                    modality_mask=modality_mask,
                    inference_mode=False
                )
            
            # Accumulate loss and accuracy
            if model_outputs.loss is not None:
                total_loss += model_outputs.loss.item()
            if "acc" in metrics:
                all_accuracies.append(metrics["acc"])
            
            # Free the model outputs before generation
            del model_outputs
            
            # === STEP 2: Generate for WER computation using model.generate() ===
            # Use truncated inputs (audio + prompt only, no answer)
            repetition_penalty = cfg.train.get("repetition_penalty", 1.2)
            
            if use_autocast:
                with torch.autocast(device_type=device, dtype=amp_dtype):
                    generated_ids = model.generate(
                        input_ids=gen_input_ids,
                        attention_mask=gen_attention_mask,
                        audio_mel=audio_mel,
                        modality_mask=gen_modality_mask,
                        max_new_tokens=max_new_tokens,
                        num_beams=1,
                        do_sample=False,
                        repetition_penalty=repetition_penalty,
                    )
            else:
                generated_ids = model.generate(
                    input_ids=gen_input_ids,
                    attention_mask=gen_attention_mask,
                    audio_mel=audio_mel,
                    modality_mask=gen_modality_mask,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                    repetition_penalty=repetition_penalty,
                )
            
            # Decode generated texts
            hyp_texts = []
            for gen_ids in generated_ids:
                hyp_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                hyp_texts.append(hyp_text)
            
            n_batches += 1
            
            # Compute WER for this batch
            if hyp_texts and ref_texts:
                # Match lengths (in case of batch size mismatch)
                min_len = min(len(hyp_texts), len(ref_texts))
                batch_hyp = hyp_texts[:min_len]
                batch_ref = ref_texts[:min_len]
                
                if batch_hyp and batch_ref:
                    batch_wer = compute_wer(batch_hyp, batch_ref)
                    all_wer_scores.append(batch_wer)
                    all_hyp_texts.extend(batch_hyp)
                    all_ref_texts.extend(batch_ref)
            
            # CRITICAL: Clean up GPU memory after each eval batch
            del input_ids, attention_mask, labels, audio_mel, modality_mask
            del gen_input_ids, gen_attention_mask, gen_modality_mask
            del generated_ids, labels_cpu
            
            # Clean GPU cache after EVERY batch during evaluation
            gc.collect()
            torch.cuda.empty_cache()

    # Final cleanup after evaluation
    gc.collect()
    torch.cuda.empty_cache()
    
    # Re-enable gradient checkpointing if it was enabled before
    if grad_ckpt_was_enabled:
        model.llm.gradient_checkpointing_enable()
    
    # Calculate final metrics
    val_loss = total_loss / max(n_batches, 1)
    val_acc = sum(all_accuracies) / max(len(all_accuracies), 1) if all_accuracies else 0.0
    val_wer_score = sum(all_wer_scores) / max(len(all_wer_scores), 1) if all_wer_scores else 1.0
    val_word_acc = 1.0 - val_wer_score if val_wer_score <= 1.0 else 0.0
    
    # Reset model to training mode
    model.train()
    
    return val_loss, val_acc, val_wer_score, val_word_acc, all_hyp_texts, all_ref_texts


def main(): 
    """
    Main training loop for training the projector module.
    """
    # For better matmul performance on GPUs with Tensor Cores 
    torch.set_float32_matmul_precision("high")   

    parser = argparse.ArgumentParser() 
    parser.add_argument("--config", type=str, default="configs/config.yaml", required=True, help="Path to the config file.")
    args = parser.parse_args() 
    #args = argparse.Namespace( config='configs/config.yaml')

    # load YAML into an OmegaConf dict
    cfg = OmegaConf.load(args.config)

    # Prepare logging/output directories
    ensure_dir(cfg.train.output_dir)
    log_dir = os.path.join(cfg.log.log_dir) if cfg.log.log_dir else "."
    ensure_dir(log_dir)
    logger = get_logger(log_dir=log_dir, filename=os.path.basename(cfg.log.log_filename))
    logger.info(f"Loaded configuration file from {args.config}")

    # Seed and device
    set_seed(cfg.train.seed)
    logger.info(f"Set random seed to {cfg.train.seed}")
    device = get_device() 
    logger.info(f"Device: {device}")

    # Model (ASR + LLM)
    model, tokenizer = model_builder(cfg.train, cfg.model)
    pad_id = resolve_pad_token(tokenizer, model.llm)
    logger.info(f"Resolved pad_token_id: {pad_id}")

    # Move model to device
    model.to(device)

    # Enable gradient checkpointing for LLM to reduce memory usage
    # This trades compute for memory by recomputing activations during backward pass
    if hasattr(model.llm, 'gradient_checkpointing_enable'):
        model.llm.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing for LLM (reduces memory usage)")

    # Freeze modules per config
    if cfg.train.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
    if cfg.train.freeze_llm:
        for p in model.llm.parameters():
            p.requires_grad = False

    # Optimizer only for projector
    projector_params = [p for p in model.projector.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(projector_params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")

    # Log encoder efficiency metrics
    num_layers = getattr(cfg.model, 'encoder_num_layers', None)
    encoder_params = count_encoder_parameters(model.encoder, num_layers=num_layers)
    
    # Build efficiency summary string for logging (ASCII only for Windows compatibility)
    efficiency_summary = f"""
{'='*60}
ENCODER EFFICIENCY METRICS
{'='*60}
Model:              Whisper-{cfg.model.encoder_model}
Layers Used:        {encoder_params['num_layers_used']} / {encoder_params['num_layers_total']}
Layer Reduction:    {100 - (encoder_params['num_layers_used']/encoder_params['num_layers_total']*100):.1f}%
{'-'*60}
Conv Params:        {encoder_params['conv_params']:>12,} ({encoder_params['conv_params']/1e6:.2f}M)
Pos Embedding:      {encoder_params['pos_embedding_params']:>12,} ({encoder_params['pos_embedding_params']/1e6:.2f}M)
LayerNorm:          {encoder_params['ln_post_params']:>12,} ({encoder_params['ln_post_params']/1e6:.2f}M)
Per Block:          {encoder_params['block_params_per_layer']:>12,} ({encoder_params['block_params_per_layer']/1e6:.2f}M)
{'-'*60}
Total Params:       {encoder_params['total_params']:>12,} ({encoder_params['total_params']/1e6:.2f}M)
Used Params:        {encoder_params['used_params']:>12,} ({encoder_params['used_params']/1e6:.2f}M)
Pruned Params:      {encoder_params['pruned_params']:>12,} ({encoder_params['pruned_params']/1e6:.2f}M)
Param Reduction:    {(encoder_params['pruned_params']/encoder_params['total_params']*100):.1f}%
{'='*60}
"""
    logger.info(efficiency_summary)

    # Dataset and DataLoader
    # FIXME: currently only support train dataset )
    split = cfg.data.get("train_split", cfg.data.get("test_split", "train"))

    train_ds = get_speech_dataset(cfg.data, tokenizer, split=split)
    train_dataloader = DataLoader(
        train_ds, 
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=train_ds.collator,
        num_workers=cfg.train.num_workers,
        # FIXME: check if device is cuda or cuda:0 
        pin_memory=(device == "cuda"),
        persistent_workers=True if cfg.train.num_workers > 0 else False
    )

    logger.info(f"Train dataset: {len(train_ds)} samples, {len(train_dataloader)} batches (batch_size={cfg.train.batch_size})")

    # Validation dataloader 
    val_split = cfg.data.get("val_split", "dev")
    vald_ds = get_speech_dataset(cfg.data, tokenizer, split=val_split)
    val_dataloader = DataLoader(
        vald_ds, 
        batch_size=cfg.train.val_batch_size,
        shuffle=False,
        collate_fn=vald_ds.collator,
        num_workers=cfg.train.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=True if cfg.train.num_workers > 0 else False
    )

    logger.info(f"Validation dataset: {len(vald_ds)} samples, {len(val_dataloader)} batches")
    logger.info(f"Log interval: {cfg.log.log_interval} steps")  # Debug: verify log interval

    # W&B init 
    run = None 
    if cfg.log.use_wandb:
        import wandb
        
        logger.info(f"Attempting to initialize W&B: project={cfg.log.wandb_project_name}, entity={cfg.log.wandb_entity_name}")
        
        run = init_wandb(
            use_wand=True, 
            project=cfg.log.wandb_project_name,
            run_name=cfg.log.wandb_exp_name,
            tags=[cfg.model.llm_model_name, cfg.model.encoder_model_name, cfg.model.projector, "projector-only"],
            entity=cfg.log.wandb_entity_name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        
        if run is not None:
            logger.info(f"Initialized W&B run: {run.url}")
        else:
            logger.warning("W&B initialization failed - run is None")
        
        # Log efficiency metrics to W&B summary (for comparing runs in table view)
        if run is not None:
            run.summary["efficiency/encoder_layers_used"] = encoder_params['num_layers_used']
            run.summary["efficiency/encoder_layers_total"] = encoder_params['num_layers_total']
            run.summary["efficiency/encoder_params_used_M"] = round(encoder_params['used_params'] / 1e6, 2)
            run.summary["efficiency/encoder_params_pruned_M"] = round(encoder_params['pruned_params'] / 1e6, 2)
            run.summary["efficiency/param_reduction_pct"] = round(encoder_params['pruned_params'] / encoder_params['total_params'] * 100, 1)
            
            # Log efficiency metrics as W&B Table (textual display in dashboard)
            efficiency_table = wandb.Table(
                columns=["Metric", "Value"],
                data=[
                    ["Whisper Model", f"whisper-{cfg.model.encoder_model}"],
                    ["Encoder Layers", f"{encoder_params['num_layers_used']} / {encoder_params['num_layers_total']}"],
                    ["Layer Reduction", f"{100 - (encoder_params['num_layers_used']/encoder_params['num_layers_total']*100):.1f}%"],
                    ["─────────────", "─────────────"],
                    ["Total Encoder Params", f"{encoder_params['total_params']/1e6:.2f}M"],
                    ["Used Encoder Params", f"{encoder_params['used_params']/1e6:.2f}M"],
                    ["Pruned Encoder Params", f"{encoder_params['pruned_params']/1e6:.2f}M"],
                    ["Param Reduction", f"{encoder_params['pruned_params']/encoder_params['total_params']*100:.1f}%"],
                    ["─────────────", "─────────────"],
                    ["Conv Params", f"{encoder_params['conv_params']/1e6:.2f}M"],
                    ["Positional Embedding", f"{encoder_params['pos_embedding_params']/1e6:.2f}M"],
                    ["Params per Block", f"{encoder_params['block_params_per_layer']/1e6:.2f}M"],
                ]
            )
            run.log({"efficiency/model_config": efficiency_table})
            
            # Also log as text artifact for easy reference
            run.log({"efficiency/summary_text": wandb.Html(f"<pre>{efficiency_summary}</pre>")})

    # Mixed precision and scaler
    use_autocast = bool(cfg.train.mixed_precision and device == "cuda")
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    # Grad Scaler for mixed precision (only needed for float16, not bfloat16)
    use_scaler = bool(cfg.train.use_fp16) and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(enabled=use_scaler)

    # Match Whisper encoder parameters dtype (prevent type mismatch)
    enc_dtype = next(model.encoder.parameters()).dtype

    # Early stopping from config
    early_stop = None
    if hasattr(cfg, 'early_stopping') and cfg.early_stopping is not None:
        early_stop = EarlyStopChecker(
            mode=cfg.early_stopping.get('mode', 'min'),
            patience=cfg.early_stopping.get('patience', 5),
            min_delta=cfg.early_stopping.get('min_delta', 0.001)
        )
        logger.info(f"Early stopping enabled: patience={early_stop.patience}, min_delta={early_stop.min_delta}")

    # Training loop
    global_step = 0
    best_val_loss = float("inf")
    best_val_wer = float("inf")
    best_val_path = None
    training_start_time = time.time()  # Total training time
    log_interval_start = time.time()   # Per-interval timing
    model.train()

    logger.info("Starting training...")

    for epoch in range(cfg.train.num_epochs):
        for step, batch in enumerate(train_dataloader):
            # Incrementing global step 
            global_step += 1

            # DEBUG: Print shapes on first step
            if global_step == 1:
                print(f"\n[DEBUG TRAIN] input_ids shape: {batch['input_ids'].shape}")
                print(f"[DEBUG TRAIN] audio_mel shape: {batch['audio_mel'].shape}")
                print(f"[DEBUG TRAIN] attention_mask shape: {batch['attention_mask'].shape}")

            # Move batch to device
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            audio_mel = batch["audio_mel"].to(device, non_blocking=True).to(enc_dtype)
            modality_mask = batch['modality_mask'].to(device, non_blocking=True)

            # Optimizer zero grad
            optimizer.zero_grad(set_to_none=True)

            if use_autocast:
                with torch.autocast(device_type=device, dtype=amp_dtype): 
                    # Forward pass
                    outputs, metrics = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        audio_mel=audio_mel,
                        modality_mask=modality_mask
                    )
                    loss = outputs.loss
                    
            else: 
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    audio_mel=audio_mel,
                    modality_mask=modality_mask
                )
                loss = outputs.loss

            # Accuracy for logging
            acc = 0.0
            batch_wer = 0.0
            if metrics is not None:
                if "acc" in metrics:
                    acc = metrics["acc"]
                if "wer" in metrics:
                    batch_wer = metrics["wer"]

            # Word-level accuracy
            word_acc = 1.0 - batch_wer if batch_wer >= 0.0 else 0.0

            # Backward pass and optimization step
            if scaler.is_enabled(): 
                scaler.scale(loss).backward()
                if cfg.train.grad_clip is not None: 
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(projector_params, cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.train.grad_clip is not None: 
                    clip_grad_norm_(projector_params, cfg.train.grad_clip)
                optimizer.step()

            # Store loss value BEFORE deleting tensors (for logging)
            loss_value = loss.item()

            # Aggressive GPU memory cleanup to prevent OOM
            # Delete tensors no longer needed
            del input_ids, attention_mask, labels, audio_mel, modality_mask
            del outputs, loss
            
            # Force garbage collection every 10 steps
            if global_step % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            if global_step % cfg.log.log_interval == 0: 
                elapsed = time.time() - log_interval_start
                lr = optimizer.param_groups[0]["lr"]
                
                # Log memory usage for debugging
                allocated, reserved, max_alloc = log_gpu_memory(logger, global_step, prefix="")
                
                logger.info(f"Epoch={epoch} | Step={global_step} | WER={batch_wer:.4f} | W_ACC={word_acc:.4f} | Loss={loss_value:.4f} | Acc={float(acc):.4f} | LR={lr:.6e} | GPU={allocated:.1f}GB | Time={elapsed:.2f}s")
                if run is not None: 
                    run.log({
                        "train/wer": batch_wer,
                        "train/word_acc": word_acc,
                        "train/loss": loss_value,
                        "train/acc": acc,
                        "train/lr": lr,
                        "train/epoch": epoch,
                        "train/step": global_step,
                        "train/time_elapsed": elapsed,
                        "train/gpu_allocated_gb": allocated,
                        "train/gpu_reserved_gb": reserved,
                    }, step=global_step)
                # Reset interval timer (not total training time)
                log_interval_start = time.time()


        # Validation at the end of each epoch
        val_loss, val_acc, val_wer_score, val_word_acc, all_hyp_texts, all_ref_texts = evaluate(cfg, model, val_dataloader, device, enc_dtype, tokenizer=tokenizer)
        logger.info(f"Epoch {epoch} | Val WER: {val_wer_score:.4f} | Val Word Acc: {val_word_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        if run is not None: 
            run.log({
                "val/wer": val_wer_score,
                "val/word_acc": val_word_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "val/epoch": epoch
            }, step=global_step)


        # Save hyp and ref texts for a few examples
        save_and_print_examples(
            hyp_texts=all_hyp_texts,
            ref_texts=all_ref_texts,
            output_path=cfg.train.output_dir,
            epoch=epoch,
            n_save=10,
            n_print=5,
            run=run,
            seed=cfg.train.seed
        )

        # Save best model
        if val_wer_score < best_val_wer:
            best_val_wer = val_wer_score
            best_val_path = os.path.join(cfg.train.output_dir, "projector_best_wer.pt")
            save_projector(model, best_val_path, global_step)
            logger.info(f"New best model saved at step {global_step} to {best_val_path} with val_wer {best_val_wer:.4f}")

        # Early stopping check
        if early_stop is not None:
            # Use val_loss or val_wer based on config monitor
            monitor_value = val_loss if cfg.early_stopping.get('monitor', 'val/loss') == 'val/loss' else val_wer_score
            if early_stop.check(monitor_value):
                logger.info(f"Early stopping triggered at epoch {epoch} (patience={early_stop.patience})")
                break

    # Final model checkpoint (for reference)
    final_path = os.path.join(cfg.train.output_dir, "projector_final.pt")
    save_projector(model, final_path, global_step)
    logger.info(f"Final model checkpointed at: {final_path}")

    # End of training     
    logger.info("Training completed.....")
    total_training_time = (time.time() - training_start_time) / 60
    logger.info("Training Time: {:.2f} minutes ({:.2f} hours)".format(total_training_time, total_training_time / 60))
    logger.info(f"Best validation wer: {best_val_wer:.4f}")
    if best_val_path:
        logger.info(f"Best projector model saved to: {best_val_path}")

    if run is not None: 
        run.finish()


if __name__ == "__main__":
    # =========================================================================
    # DEBUG MODE TOGGLE
    # Set to True for local debugging with VS Code (uses test_config.yaml)
    # Set to False for production training (uses command line --config)
    # =========================================================================
    DEBUG = False
    # =========================================================================
    
    if DEBUG:
        print("=" * 60)
        print("🔧 RUNNING IN DEBUG MODE")
        print("   Config: configs/test_config.yaml")
        print("   Set DEBUG = False for production training")
        print("=" * 60)
        sys.argv = ['train.py', '--config', 'configs/test_config.yaml']
    
    main()

