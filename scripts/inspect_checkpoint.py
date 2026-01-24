#!/usr/bin/env python3
"""
Checkpoint Inspector Script
===========================
Verifies checkpoint contents, including LoRA parameters.

Usage:
    python inspect_checkpoint.py <checkpoint_path>
    
Examples:
    python inspect_checkpoint.py outputs/dutch/whisper-s_dutch_baseline_5h/checkpoint_best_wer.pt
    python inspect_checkpoint.py outputs/danish/whisper-s_danish_lora/checkpoint_best_wer.pt
"""

import argparse
import torch
from collections import defaultdict


def get_param_info(param):
    """Safely get parameter shape/info."""
    if isinstance(param, torch.Tensor):
        return f"Tensor{tuple(param.shape)}"
    elif isinstance(param, dict):
        return f"Dict with {len(param)} keys"
    else:
        return f"{type(param).__name__}: {param}"


def inspect_state_dict(state_dict, name="state_dict"):
    """Inspect a state dict and report parameter details."""
    print(f"\n    [{name}] Contains {len(state_dict)} parameters:")
    
    lora_params = []
    total_params = 0
    
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            shape = tuple(value.shape)
            num_params = value.numel()
            total_params += num_params
            print(f"      - {key}: {shape} ({num_params:,} params)")
            
            if 'lora' in key.lower():
                lora_params.append(key)
        else:
            print(f"      - {key}: {get_param_info(value)}")
    
    print(f"    Total parameters in {name}: {total_params:,}")
    return lora_params, total_params


def inspect_checkpoint(ckpt_path: str, verbose: bool = False):
    """Inspect a checkpoint file and report its contents."""
    
    print(f"\n{'='*60}")
    print(f"Inspecting: {ckpt_path}")
    print(f"{'='*60}\n")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint: {e}")
        return
    
    # Check checkpoint type
    print(f"[1] Checkpoint Type: {type(checkpoint)}")
    
    if not isinstance(checkpoint, dict):
        print(f"    Unexpected checkpoint type!")
        return
    
    print(f"[2] Top-level keys: {list(checkpoint.keys())}")
    
    # Print metadata
    if 'step' in checkpoint:
        print(f"    → Step: {checkpoint['step']}")
    if 'epoch' in checkpoint:
        print(f"    → Epoch: {checkpoint['epoch']}")
    if 'best_wer' in checkpoint:
        print(f"    → Best WER: {checkpoint['best_wer']}")
    if 'best_val_wer' in checkpoint:
        print(f"    → Best Val WER: {checkpoint['best_val_wer']}")
    
    # Check for nested structure (projector, lora as separate dicts)
    has_nested_structure = 'projector' in checkpoint or 'lora' in checkpoint
    
    if has_nested_structure:
        print(f"\n[3] Nested Checkpoint Structure Detected")
        print(f"    This checkpoint stores 'projector' and 'lora' separately.\n")
        
        all_lora_params = []
        total_all_params = 0
        
        # Inspect projector
        if 'projector' in checkpoint:
            print(f"{'='*60}")
            print(f"PROJECTOR WEIGHTS")
            print(f"{'='*60}")
            
            projector_data = checkpoint['projector']
            if isinstance(projector_data, dict):
                lora_p, total_p = inspect_state_dict(projector_data, "projector")
                all_lora_params.extend(lora_p)
                total_all_params += total_p
            else:
                print(f"    Projector is not a dict: {type(projector_data)}")
        else:
            print(f"\n[!] No 'projector' key found in checkpoint")
        
        # Inspect LoRA
        if 'lora' in checkpoint:
            print(f"\n{'='*60}")
            print(f"LoRA WEIGHTS")
            print(f"{'='*60}")
            
            lora_data = checkpoint['lora']
            if lora_data is None:
                print(f"\n    ✗ 'lora' key exists but is None (no LoRA weights saved)")
            elif isinstance(lora_data, dict):
                if len(lora_data) == 0:
                    print(f"\n    ✗ 'lora' key exists but is empty dict (no LoRA weights)")
                else:
                    lora_p, total_p = inspect_state_dict(lora_data, "lora")
                    all_lora_params.extend(lora_p)
                    total_all_params += total_p
            else:
                print(f"    LoRA is not a dict: {type(lora_data)}")
        else:
            print(f"\n[!] No 'lora' key found in checkpoint")
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        
        if 'projector' in checkpoint and isinstance(checkpoint['projector'], dict):
            proj_params = sum(v.numel() for v in checkpoint['projector'].values() if isinstance(v, torch.Tensor))
            print(f"✓ Projector weights: {len(checkpoint['projector'])} tensors, {proj_params:,} parameters")
        else:
            print(f"✗ No projector weights found")
        
        if 'lora' in checkpoint:
            lora_data = checkpoint['lora']
            if lora_data is None:
                print(f"✗ LoRA: None (this is a non-LoRA checkpoint)")
            elif isinstance(lora_data, dict) and len(lora_data) == 0:
                print(f"✗ LoRA: Empty dict (this is a non-LoRA checkpoint)")
            elif isinstance(lora_data, dict):
                lora_params = sum(v.numel() for v in lora_data.values() if isinstance(v, torch.Tensor))
                print(f"✓ LoRA weights: {len(lora_data)} tensors, {lora_params:,} parameters")
                
                # Count LoRA A and B matrices
                lora_a = [k for k in lora_data.keys() if 'lora_A' in k or 'lora_a' in k]
                lora_b = [k for k in lora_data.keys() if 'lora_B' in k or 'lora_b' in k]
                print(f"  → LoRA A matrices: {len(lora_a)}")
                print(f"  → LoRA B matrices: {len(lora_b)}")
        else:
            print(f"✗ No 'lora' key in checkpoint")
        
        print(f"{'='*60}\n")
        
    else:
        # Flat state_dict structure
        print(f"\n[3] Flat State Dict Structure")
        
        # Try to find the state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        lora_params, total_params = inspect_state_dict(state_dict, "model")
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Total parameters: {total_params:,}")
        if lora_params:
            print(f"✓ LoRA parameters found: {len(lora_params)}")
        else:
            print(f"✗ No LoRA parameters found")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Inspect checkpoint for LoRA parameters')
    parser.add_argument('checkpoint_path', type=str, help='Path to checkpoint file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show all parameter names')
    
    args = parser.parse_args()
    inspect_checkpoint(args.checkpoint_path, verbose=args.verbose)


if __name__ == '__main__':
    main()