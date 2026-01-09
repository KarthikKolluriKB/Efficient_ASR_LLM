"""
Evaluation script for SLAM-ASR model.
Tests trained model on test set using actual generation (not teacher forcing).

Usage:
    python eval.py --cfg_path configs/eval_config.yaml --ckpt_path outputs/slam_asr_eng_test/projector_best_wer.pt

Output:
    - WER, Word Accuracy metrics
    - Saved examples (JSONL)
    - Wandb logging (if enabled)
"""

import argparse
import json 
import os 
import gc
import torch 

from tqdm import tqdm
from omegaconf import OmegaConf
from types import SimpleNamespace 
from torch.utils.data import DataLoader

# Internal imports
from models.model import model_builder
from datamodule.dataset import get_speech_dataset
from utils.metrics import compute_wer, count_encoder_parameters
from utils.wand_config import init_wandb
from utils.log_config import get_logger
from utils.utils import ensure_dir

logger = get_logger(log_dir="logs", filename="eval.log")


def truncate_for_generation(input_ids, attention_mask, labels, modality_mask, device):
    """
    Truncate inputs to only include audio + prompt (remove answer).
    This is required for proper generation - we don't want the answer in the input.
    
    Returns:
        Truncated tensors ready for generation
    """
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
            answer_start = label_row.shape[0]
        
        # Truncate to only audio + prompt (exclude answer)
        gen_input_ids_list.append(input_ids[i, :answer_start])
        gen_attention_mask_list.append(attention_mask[i, :answer_start])
        gen_modality_mask_list.append(modality_mask[i, :answer_start])
    
    # Pad truncated sequences to same length (left-pad)
    max_gen_len = max(len(seq) for seq in gen_input_ids_list)
    gen_input_ids = torch.zeros(len(gen_input_ids_list), max_gen_len, dtype=input_ids.dtype, device=device)
    gen_attention_mask = torch.zeros(len(gen_attention_mask_list), max_gen_len, dtype=attention_mask.dtype, device=device)
    gen_modality_mask = torch.zeros(len(gen_modality_mask_list), max_gen_len, dtype=modality_mask.dtype, device=device)
    
    for i, (ids, mask, mod_mask) in enumerate(zip(gen_input_ids_list, gen_attention_mask_list, gen_modality_mask_list)):
        seq_len = len(ids)
        # Left-pad
        gen_input_ids[i, max_gen_len - seq_len:] = ids
        gen_attention_mask[i, max_gen_len - seq_len:] = mask
        gen_modality_mask[i, max_gen_len - seq_len:] = mod_mask
    
    return gen_input_ids, gen_attention_mask, gen_modality_mask


def save_examples_jsonl(hyp_texts, ref_texts, output_path, filename="test_examples.jsonl"):
    """Save all examples to a JSONL file."""
    ensure_dir(output_path)
    filepath = os.path.join(output_path, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for i, (hyp, ref) in enumerate(zip(hyp_texts, ref_texts)):
            example = {
                "id": i,
                "reference": ref,
                "hypothesis": hyp,
            }
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(hyp_texts)} examples to {filepath}")
    return filepath


def log_examples_to_wandb(hyp_texts, ref_texts, run, num_examples=50):
    """Log example predictions to wandb as a table."""
    import wandb
    
    # Create a table with examples
    table = wandb.Table(columns=["ID", "Reference", "Hypothesis", "Match"])
    
    for i in range(min(num_examples, len(hyp_texts))):
        match = "✓" if hyp_texts[i].strip() == ref_texts[i].strip() else ""
        table.add_data(i, ref_texts[i], hyp_texts[i], match)
    
    run.log({"test/examples": table})
    logger.info(f"Logged {min(num_examples, len(hyp_texts))} examples to wandb")

@torch.no_grad()
def run_eval(args):
    """
    Run evaluation on the test set using actual generation.
    """
    run = None
    model = None
    device = None
    
    try:
        # 1. Load config and check files exist
        if not os.path.exists(args.cfg_path):
            raise FileNotFoundError(f"Config file not found: {args.cfg_path}")
        if not os.path.exists(args.ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {args.ckpt_path}")
        
        cfg = OmegaConf.load(args.cfg_path)
        train_cfg = cfg.train
        model_cfg = cfg.model
        data_cfg = cfg.data
        wandb_cfg = cfg.log if hasattr(cfg, 'log') else None
        eval_cfg = cfg.eval if hasattr(cfg, 'eval') else None
        
        # Get generation settings
        max_new_tokens = args.max_new_tokens or (eval_cfg.max_new_tokens if eval_cfg else 128)
        repetition_penalty = args.repetition_penalty or 1.3
        
        # 2. Initialize wandb if enabled
        if wandb_cfg and wandb_cfg.use_wandb:
            run = init_wandb(
                use_wand=True,
                project=wandb_cfg.wandb_project_name,
                run_name=args.wandb_exp_name or f"eval_{wandb_cfg.wandb_exp_name}",
                tags=["eval", "asr-llm", "test"],
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            logger.info(f"Initialized wandb run: {run.url}")
        
        # 3. Build Model and load weights
        logger.info(f"Loading model with checkpoint: {args.ckpt_path}")
        model, tokenizer = model_builder(train_cfg, model_cfg, data_config=data_cfg)
        
        # Load projector weights
        projector_state = torch.load(args.ckpt_path, map_location='cpu', weights_only=True)
        model.projector.load_state_dict(projector_state)
        logger.info("Loaded projector weights successfully")
        
        # 4. Move model to device
        device = torch.device(args.device)
        model = model.to(device)
        model.eval()
        logger.info(f"Model moved to device: {device}")
        
        # Log encoder efficiency metrics
        num_layers = getattr(model_cfg, 'encoder_num_layers', None)
        encoder_params = count_encoder_parameters(model.encoder, num_layers=num_layers)
        
        efficiency_summary = f"""
{'='*60}
ENCODER EFFICIENCY METRICS
{'='*60}
Model:              Whisper-{model_cfg.encoder_model}
Layers Used:        {encoder_params['num_layers_used']} / {encoder_params['num_layers_total']}
Used Params:        {encoder_params['used_params']:,} ({encoder_params['used_params']/1e6:.2f}M)
{'='*60}
"""
        logger.info(efficiency_summary)
        
        if run is not None:
            run.summary["efficiency/encoder_layers_used"] = encoder_params['num_layers_used']
            run.summary["efficiency/encoder_params_used_M"] = round(encoder_params['used_params'] / 1e6, 2)
        
        # 5. Load Test Dataset
        test_dataset = get_speech_dataset(data_cfg, tokenizer, split=args.split)
        logger.info(f"Loaded {len(test_dataset)} samples from {args.split} split")
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=test_dataset.collator,
            num_workers=4,
            pin_memory=True if device.type == 'cuda' else False,
        )
        
        # 6. Determine precision
        use_autocast = train_cfg.mixed_precision and device.type == 'cuda'
        amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        enc_dtype = amp_dtype if use_autocast else torch.float32
        
        # 7. Evaluation Loop with Generation
        all_hyp_texts = []
        all_ref_texts = []
        all_wer_scores = []
        
        logger.info("Starting evaluation with generation...")
        
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Evaluating")):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            audio_mel = batch["audio_mel"].to(device).to(enc_dtype)
            modality_mask = batch["modality_mask"].to(device)
            
            # Get reference texts from labels
            ref_texts = []
            for label in labels:
                valid_tokens = label[label != -100]
                if len(valid_tokens) > 0:
                    ref_text = tokenizer.decode(valid_tokens, skip_special_tokens=True).strip()
                    ref_texts.append(ref_text)
            
            # Truncate inputs for generation (remove answer)
            gen_input_ids, gen_attention_mask, gen_modality_mask = truncate_for_generation(
                input_ids, attention_mask, labels, modality_mask, device
            )
            
            # Generate transcriptions
            if use_autocast:
                with torch.autocast(device_type='cuda', dtype=amp_dtype):
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
            
            # Compute WER for this batch
            if hyp_texts and ref_texts:
                min_len = min(len(hyp_texts), len(ref_texts))
                batch_hyp = hyp_texts[:min_len]
                batch_ref = ref_texts[:min_len]
                
                batch_wer = compute_wer(batch_hyp, batch_ref)
                all_wer_scores.append(batch_wer)
                all_hyp_texts.extend(batch_hyp)
                all_ref_texts.extend(batch_ref)
            
            # Print some examples from first batch
            if batch_idx == 0:
                logger.info("\n" + "="*60)
                logger.info("SAMPLE PREDICTIONS (First Batch):")
                logger.info("="*60)
                for i in range(min(5, len(hyp_texts))):
                    logger.info(f"\n[{i+1}]")
                    logger.info(f"  REF: {ref_texts[i]}")
                    logger.info(f"  HYP: {hyp_texts[i]}")
            
            # Memory cleanup
            if batch_idx % 20 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # 8. Calculate final metrics
        if all_wer_scores:
            avg_wer = sum(all_wer_scores) / len(all_wer_scores)
            word_acc = 1.0 - avg_wer if avg_wer <= 1.0 else 0.0
            
            # Final results
            logger.info("\n" + "="*60)
            logger.info("FINAL EVALUATION RESULTS:")
            logger.info("="*60)
            logger.info(f"Total samples: {len(all_hyp_texts)}")
            logger.info(f"Average WER: {avg_wer:.4f}")
            logger.info(f"Word Accuracy: {word_acc:.4f}")
            logger.info("="*60)
            
            # Log to wandb
            if run is not None:
                run.log({
                    "test/wer": avg_wer,
                    "test/word_accuracy": word_acc,
                    "test/num_samples": len(all_hyp_texts),
                })
                run.summary["test/final_wer"] = avg_wer
                run.summary["test/final_word_accuracy"] = word_acc
                
                # Log examples table
                log_examples_to_wandb(all_hyp_texts, all_ref_texts, run, num_examples=50)
            
            # Save predictions to JSONL
            output_dir = args.output_dir or (eval_cfg.output_dir if eval_cfg else "eval_results")
            save_examples_jsonl(all_hyp_texts, all_ref_texts, output_dir)
            
            # Also save a summary file
            summary_path = os.path.join(output_dir, "eval_summary.json")
            summary = {
                "checkpoint": args.ckpt_path,
                "config": args.cfg_path,
                "split": args.split,
                "num_samples": len(all_hyp_texts),
                "wer": avg_wer,
                "word_accuracy": word_acc,
            }
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Saved summary to {summary_path}")
            
        else:
            logger.error("No samples were processed!")
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA out of memory: {e}")
        logger.error("Try reducing batch size")
        if device and device.type == 'cuda':
            torch.cuda.empty_cache()
        raise
        
    except Exception as e:
        logger.exception(f"Error during evaluation: {e}")
        raise
        
    finally:
        # Cleanup
        if device and device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        if run is not None:
            try:
                run.finish()
            except Exception as e:
                logger.warning(f"Failed to finish wandb run: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ASR-LLM model on test set")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the config file (alternative to --cfg_path).",
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        default=None,
        help="Path to the config file.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to the model checkpoint (projector weights). If not provided, uses eval.projector_path from config.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation (smaller for generation).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Data split to evaluate on.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run the evaluation on.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Directory to save evaluation results.",
    )
    parser.add_argument(
        "--wandb_exp_name",
        type=str,
        default=None,
        help="Wandb experiment name.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.3,
        help="Repetition penalty for generation.",
    )
    args = parser.parse_args()
    
    # Handle --config as alias for --cfg_path
    if args.config and not args.cfg_path:
        args.cfg_path = args.config
    
    # Validate cfg_path is provided
    if not args.cfg_path:
        parser.error("--config or --cfg_path is required")
    
    # If ckpt_path not provided, load from config
    if not args.ckpt_path:
        cfg = OmegaConf.load(args.cfg_path)
        if hasattr(cfg, 'eval') and hasattr(cfg.eval, 'projector_path'):
            args.ckpt_path = cfg.eval.projector_path
            print(f"Using projector_path from config: {args.ckpt_path}")
        else:
            parser.error("--ckpt_path is required (or set eval.projector_path in config)")
    
    # Load output_dir from config if not specified
    if args.output_dir == "eval_results":
        cfg = OmegaConf.load(args.cfg_path)
        if hasattr(cfg, 'eval') and hasattr(cfg.eval, 'output_dir'):
            args.output_dir = cfg.eval.output_dir
    
    return args


if __name__ == "__main__":
    # For debugging: create temporary args
    DEBUG = False
    if DEBUG:
        args = SimpleNamespace(
            cfg_path="configs/eval_config.yaml",
            ckpt_path="outputs/ASRLLM_enc_lin_20h/projector_best_wer.pt",
            split="test",
            batch_size=8,
            device="cuda" if torch.cuda.is_available() else "cpu",
            output_dir="eval_results",
            wandb_exp_name="eval_debug",
            max_new_tokens=128,
            repetition_penalty=1.3,
        )
        logger.info("Using debug arguments")
    else:
        args = parse_args()
    
    run_eval(args)
