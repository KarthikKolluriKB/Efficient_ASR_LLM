import os 
import random 
import torch 
from typing import Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def set_seed(seed: int, deterministic: bool = False):
    """
    Set the random seed for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: If True, enables full determinism (slower but reproducible).
                      If False (default), allows some non-determinism for speed.
    
    Note:
        Full determinism requires deterministic=True, but this can slow down training
        significantly (up to 10-20% slower). For ablation studies comparing different
        configurations, it's often better to run multiple seeds and average results.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    
    if HAS_NUMPY:
        np.random.seed(seed)
    
    if deterministic:
        # Enable full determinism (slower)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For CUDA >= 10.2, this helps with some operations
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # PyTorch 1.8+ deterministic algorithms
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass  # Some operations don't have deterministic implementations
    else:
        # Allow cuDNN to find optimal algorithms (faster but less reproducible)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# List of good seeds found through experimentation
# These tend to work well for ASR tasks
RECOMMENDED_SEEDS = [42, 123, 456, 789, 1337, 2024, 3407, 7777, 9999, 12345]


def get_seed_from_config(config_seed: int, seed_index: int = 0) -> int:
    """
    Get a seed, optionally from the recommended list.
    
    Args:
        config_seed: Seed from config (use if >= 0)
        seed_index: Index into RECOMMENDED_SEEDS (use if config_seed < 0)
    
    Returns:
        The seed to use
    """
    if config_seed >= 0:
        return config_seed
    else:
        return RECOMMENDED_SEEDS[seed_index % len(RECOMMENDED_SEEDS)]


# TODO: Need to check if this works correctly (single GPU case) (cuda vs cuda:0)
def get_device() -> str:
    """Get the available device (GPU0 or CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"

# TODO: need to understand this function
def resolve_pad_token(tokenizer, model) -> int:
    """
    Ensure the tokenizer.pad_token_id is set and syncs it to model configs. 
    If tokenizer.pad_token_id is None, set it to tokenizer.eos_token_id (int). 
    
    Args:
        tokenizer: The tokenizer to check and set the pad_token_id.
        model: The model whose config needs to be updated with the pad_token_id.

    Returns:
        pad_id (int): The resolved pad token ID.
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None: 
        # Use tokenizer.eos_token_id as pad_token_id if pad_token_id is not set
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
            pad_id = tokenizer.pad_token_id
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            pad_id = tokenizer.pad_token_id
            if hasattr(model, 'resize_token_embeddings'):
                model.resize_token_embeddings(len(tokenizer))
        
        # propagate the pad_token_id to model config
        if hasattr(model, 'config'):
            model.config.pad_token_id = pad_id
        if hasattr(model , 'generation_config'):
            model.generation_config.pad_token_id = pad_id
    return pad_id


def ensure_dir(dir_path: str):
    """Ensure that a directory exists; if not, create it."""
    os.makedirs(dir_path, exist_ok=True)

def save_projector(model, path: str, step: int):
    """
    Save the projector weights to `path` under the key "projector".

    This function prefers `model.projector` but falls back to
    `model.encoder_projector` for backward compatibility with older code.
    """
    proj = None
    if hasattr(model, "projector") and model.projector is not None:
        proj = model.projector
    elif hasattr(model, "encoder_projector") and model.encoder_projector is not None:
        proj = model.encoder_projector
    else:
        raise AttributeError("Model has no attribute 'projector' or 'encoder_projector' to save.")

    torch.save({"step": step, "projector": proj.state_dict()}, path)
    print(f"Projector saved at step {step} to {path}")


def save_checkpoint(model, path: str, step: int, save_lora: bool = False):
    """
    Save checkpoint with projector weights and optionally LoRA adapter weights.
    
    Args:
        model: The ASRLLM model
        path: Path to save checkpoint
        step: Current training step
        save_lora: Whether to save LoRA adapter weights
    """
    checkpoint = {"step": step}
    
    # Save projector
    proj = None
    if hasattr(model, "projector") and model.projector is not None:
        proj = model.projector
    elif hasattr(model, "encoder_projector") and model.encoder_projector is not None:
        proj = model.encoder_projector
    
    if proj is not None:
        checkpoint["projector"] = proj.state_dict()
    
    # Save LoRA adapter weights if enabled
    if save_lora and hasattr(model, "llm"):
        try:
            # Check if LLM has LoRA adapters (PEFT model)
            from peft import PeftModel
            if isinstance(model.llm, PeftModel):
                # Get only the LoRA adapter parameters
                lora_state_dict = {}
                for name, param in model.llm.named_parameters():
                    if "lora_" in name or "modules_to_save" in name:
                        lora_state_dict[name] = param.cpu().clone()
                if lora_state_dict:
                    checkpoint["lora"] = lora_state_dict
                    print(f"LoRA adapter weights included in checkpoint ({len(lora_state_dict)} tensors)")
        except ImportError:
            pass
        except Exception as e:
            print(f"Warning: Could not save LoRA weights: {e}")
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at step {step} to {path}")


def save_lora_adapter(model, output_dir: str, adapter_name: str = "lora_adapter"):
    """
    Save LoRA adapter weights separately using PEFT's save method.
    This creates a more portable adapter that can be loaded with PeftModel.from_pretrained()
    
    Args:
        model: The ASRLLM model with LoRA-enabled LLM
        output_dir: Directory to save the adapter
        adapter_name: Name for the adapter subdirectory
    """
    import os
    from peft import PeftModel
    
    if not hasattr(model, "llm"):
        raise ValueError("Model does not have 'llm' attribute")
    
    if not isinstance(model.llm, PeftModel):
        raise ValueError("LLM is not a PeftModel (LoRA not applied)")
    
    adapter_path = os.path.join(output_dir, adapter_name)
    os.makedirs(adapter_path, exist_ok=True)
    
    # Save adapter using PEFT's method
    model.llm.save_pretrained(adapter_path)
    print(f"LoRA adapter saved to {adapter_path}")
