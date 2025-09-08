import os 
import time 
import argparse 

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
import sys

def main(): 
    # parser = argparse.ArgumentParser() 
    # parser.add_argument("--config", type=str, default="configs/config.yaml", required=True, help="Path to the config file.")
    # args = parser.parse_args() 
    args = argparse.Namespace( config='configs/config.yaml')

    # load YAML into an OmegaConf dict
    cfg = OmegaConf.load(args.config)

    # Prepare I/O 
    ensure_dir(cfg.train.output_dir)
    log_dir = os.path.join(cfg.log.log_dir) or "."
    ensure_dir(log_dir)
    logger = get_logger(log_dir=log_dir, filename=os.path.basename(cfg.log.log_filename))
    logger.info(f"Loaded configuration file from {args.config}")

    # Seed and device
    set_seed(cfg.train.seed)
    device = get_device() 
    logger.info(f"Device: {device}")

    # Model (ASR + LLM)
    model, tokenizer = model_builder(cfg.train, cfg.model)

    # Ensure a valid pad_token_id for Llama-style tokenizers
    pad_id = resolve_pad_token(tokenizer, model.llm)
    logger.info(f"Resolved pad_token_id: {pad_id}")

    # Move model to device
    model.to(device)

    # Freeze modules per config
    if cfg.train.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
    if cfg.train.freeze_llm:
        for p in model.llm.parameters():
            p.requires_grad = False

    # Optimizer only for projector
    projector_params = [p for p in model.encoder_projector.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(projector_params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

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
        pin_memory=(device == "cuda")
    )

    # W&B init 
    run = None 
    if cfg.log.use_wandb:
        run = init_wandb(
            use_wand=True, 
            project=cfg.log.wandb_project,
            run_name=cfg.log.wandb_run_name,
            tags=[cfg.model.llm_model, cfg.model.encoder_name, cfg.model.encoder_projector, "projector-only"],
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    # Mixed precision and scaler
    use_autocast = bool(cfg.train.mixed_precision and device == "cuda")
    scaler = torch.amp.GradScaler(enabled=bool(cfg.train.use_fp16))

    # Match Whisper encoder parameters dtype (prevent type mismatch)
    enc_dtype = next(model.encoder.parameters()).dtype

    # Training loop
    global_step = 0
    best_loss = float("inf")
    start_time = time.time()
    model.train()
    logger.info("Starting training...")

    for epoch in range(cfg.train.num_epochs):
        for step, batch in enumerate(train_dataloader):
            # Incrementing global step 
            global_step += 1

            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            audio_mel = batch["audio_mel"].to(device).to(enc_dtype)

            # Optimizer zero grad
            optimizer.zero_grad(set_to_none=True)

            if use_autocast:
                with torch.autocast(device_type=device, dtype=torch.bfloat16 if enc_dtype == torch.bfloat16 else torch.float16): 
                    # Forward pass
                    # FIXME: Metric is not used currently
                    outputs, _ = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        audio_mel=audio_mel
                    )
                    loss = outputs.loss
            else: 
                # Forward pass
                outputs, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    audio_mel=audio_mel
                )
                loss = outputs.loss

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

            if global_step % cfg.log.log_interval == 0: 
                elapsed = time.time() - start_time
                lr = optimizer.param_groups["lr"]
                logger.info(f"Epoch={epoch} Step={global_step} Loss={loss.item():.4f} LR={lr:.6e} Time={elapsed:.2f}s")
                if run is not None: 
                    run.log({
                        "train/loss": loss.item(),
                        "train/lr": lr,
                        "train/epoch": epoch,
                        "train/step": global_step,
                        "train/time_elapsed": elapsed
                    }, step=global_step)
                start_time = time.time()

            if cfg.train.save_model and global_step % cfg.train.validation_interval == 0: 
                ckpt_path = os.path.join(cfg.train.output_dir, f"projector_step{global_step}.pt")
                save_projector(model, ckpt_path, global_step)
                logger.info(f"Saved checkpoint: {ckpt_path}")

            if loss.item() < best_loss: 
                best_loss = loss.item() 
                best_path = os.path.join(cfg.train.output_dir, f"projector_best.pt")
                save_projector(model, best_path, global_step)

    logger.info("Training completed.....")
    if run is not None: 
        run.finish()

if __name__ == "__main__":
    main()
                
            



