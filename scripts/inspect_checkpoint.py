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
    
    if isinstance(checkpoint, dict):
        print(f"[2] Top-level keys: {list(checkpoint.keys())}\n")
        
        # Get the state dict (could be nested or direct)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("    → Using 'state_dict' key")
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("    → Using 'model_state_dict' key")
        else:
            # Assume the checkpoint itself is the state dict
            state_dict = checkpoint
            print("    → Checkpoint appears to be a direct state_dict")
        
        # Print metadata if available
        if 'epoch' in checkpoint:
            print(f"    → Epoch: {checkpoint['epoch']}")
        if 'step' in checkpoint:
            print(f"    → Step: {checkpoint['step']}")
        if 'best_wer' in checkpoint:
            print(f"    → Best WER: {checkpoint['best_wer']}")
        if 'best_val_wer' in checkpoint:
            print(f"    → Best Val WER: {checkpoint['best_val_wer']}")
            
    else:
        state_dict = checkpoint
        print("    → Checkpoint is a direct state_dict (not a dict with metadata)")
    
    print(f"\n[3] Total parameters in state_dict: {len(state_dict)}")
    
    # Categorize parameters
    categories = defaultdict(list)
    lora_params = []
    projector_params = []
    encoder_params = []
    llm_params = []
    other_params = []
    
    for key in state_dict.keys():
        key_lower = key.lower()
        
        # Check for LoRA parameters
        if 'lora' in key_lower:
            lora_params.append(key)
        elif 'projector' in key_lower:
            projector_params.append(key)
        elif 'encoder' in key_lower or 'whisper' in key_lower:
            encoder_params.append(key)
        elif 'llm' in key_lower or 'qwen' in key_lower or 'model.model' in key_lower:
            llm_params.append(key)
        else:
            other_params.append(key)
    
    # Report findings
    print(f"\n[4] Parameter Categories:")
    print(f"    → LoRA parameters:      {len(lora_params)}")
    print(f"    → Projector parameters: {len(projector_params)}")
    print(f"    → Encoder parameters:   {len(encoder_params)}")
    print(f"    → LLM parameters:       {len(llm_params)}")
    print(f"    → Other parameters:     {len(other_params)}")
    
    # LoRA specific analysis
    print(f"\n[5] LoRA Analysis:")
    if lora_params:
        print(f"    ✓ LoRA parameters FOUND ({len(lora_params)} total)")
        
        # Analyze LoRA structure
        lora_a_params = [p for p in lora_params if 'lora_a' in p.lower() or 'lora_A' in p]
        lora_b_params = [p for p in lora_params if 'lora_b' in p.lower() or 'lora_B' in p]
        
        print(f"    → LoRA A matrices: {len(lora_a_params)}")
        print(f"    → LoRA B matrices: {len(lora_b_params)}")
        
        if verbose or len(lora_params) <= 20:
            print(f"\n    LoRA parameter names:")
            for param in sorted(lora_params)[:20]:
                shape = tuple(state_dict[param].shape)
                print(f"      - {param}: {shape}")
            if len(lora_params) > 20:
                print(f"      ... and {len(lora_params) - 20} more")
    else:
        print(f"    ✗ NO LoRA parameters found")
        print(f"    → This checkpoint does NOT contain LoRA weights")
    
    # Projector analysis
    print(f"\n[6] Projector Analysis:")
    if projector_params:
        print(f"    ✓ Projector parameters FOUND ({len(projector_params)} total)")
        if verbose or len(projector_params) <= 10:
            print(f"\n    Projector parameter names:")
            for param in sorted(projector_params):
                shape = tuple(state_dict[param].shape)
                print(f"      - {param}: {shape}")
    else:
        print(f"    ✗ NO Projector parameters found")
    
    # Show sample of all keys if verbose
    if verbose:
        print(f"\n[7] All parameter keys (first 50):")
        for i, key in enumerate(sorted(state_dict.keys())[:50]):
            shape = tuple(state_dict[key].shape)
            print(f"    {i+1}. {key}: {shape}")
        if len(state_dict) > 50:
            print(f"    ... and {len(state_dict) - 50} more")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    if lora_params:
        print(f"✓ This is a LoRA checkpoint with {len(lora_params)} LoRA parameters")
    else:
        print(f"✗ This is NOT a LoRA checkpoint (no LoRA parameters found)")
    
    if projector_params:
        print(f"✓ Contains projector weights ({len(projector_params)} parameters)")
    else:
        print(f"? No projector weights found (might be named differently)")
    
    print(f"{'='*60}\n")
    
    return {
        'has_lora': len(lora_params) > 0,
        'lora_count': len(lora_params),
        'projector_count': len(projector_params),
        'total_params': len(state_dict)
    }


def main():
    parser = argparse.ArgumentParser(description='Inspect checkpoint for LoRA parameters')
    parser.add_argument('checkpoint_path', type=str, help='Path to checkpoint file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show all parameter names')
    
    args = parser.parse_args()
    inspect_checkpoint(args.checkpoint_path, verbose=args.verbose)


if __name__ == '__main__':
    main()