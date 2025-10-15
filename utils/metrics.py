from typing import List, Tuple, Union, Dict
import torch
import numpy as np
from transformers import AutoTokenizer
import evaluate

def compute_accuracy(pad_outputs: torch.LongTensor,
                     pad_targets: torch.LongTensor,
                     ignore_label: int) -> torch.Tensor:
    """Calculate accuracy.

    Args:
        pad_outputs (LongTensor): Prediction tensors (B, Lmax).
        pad_targets (LongTensor): Target label tensors (B, Lmax).
        ignore_label (int): Ignore label id.

    Returns:
        torch.Tensor: Accuracy value (0.0 - 1.0).
    """

    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_outputs.masked_select(mask) == pad_targets.masked_select(mask)
    )
    denominator = torch.sum(mask)
    return numerator.float() / denominator.float()

def decode_texts_from_outputs(logits: torch.Tensor,
                labels: torch.Tensor,
                tokenizer: AutoTokenizer,
                ignore_label: int = -100) -> Tuple[List[str], List[str]]:
    """Decode logits and labels into text using the tokenizer.
    
    Args:
        logits (torch.Tensor): Model output logits of shape (B, L, V)
        labels (torch.Tensor): Label indices of shape (B, L)
        tokenizer (AutoTokenizer): Tokenizer for decoding indices to text
        ignore_label (int): Label to ignore in computation
        
    Returns:
        Tuple[List[str], List[str]]: Tuple of (hypothesis texts, reference texts)
    """
    # Get predictions by taking argmax of logits
    pred_ids = torch.argmax(logits, dim=-1)  # (B, L)
    
    hyp_texts = []
    ref_texts = []
    
    # Process each sequence in the batch
    for pred, label in zip(pred_ids, labels):
        # Remove padding and ignored labels
        valid_mask = (label != ignore_label) & (label != tokenizer.pad_token_id)
        valid_label = label[valid_mask]
        
        # For predictions, remove padding and positions that were ignored in labels
        valid_pred = pred[valid_mask]
        
        # Skip empty sequences
        if len(valid_label) == 0 or len(valid_pred) == 0:
            continue
        
        # Decode to text and strip any extra whitespace
        pred_text = tokenizer.decode(valid_pred, skip_special_tokens=True).strip()
        label_text = tokenizer.decode(valid_label, skip_special_tokens=True).strip()
        
        # Only add non-empty sequences
        if pred_text and label_text:
            hyp_texts.append(pred_text)
            ref_texts.append(label_text)
            
    return hyp_texts, ref_texts

def compute_wer(hyp_texts: List[str], ref_texts: List[str]) -> float:
    """Calculate Word Error Rate (WER) from hypothesis and reference texts.
    
    Args:
        hyp_texts (List[str]): List of hypothesis (predicted) texts
        ref_texts (List[str]): List of reference (ground truth) texts
        
    Returns:
        float: Word Error Rate score (lower is better)
    """
    if not hyp_texts or not ref_texts:
        return 0.0
        
    wer_metric = evaluate.load("wer")
    return wer_metric.compute(predictions=hyp_texts, references=ref_texts)