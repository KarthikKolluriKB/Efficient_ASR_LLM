from typing import List, Tuple, Union, Dict
import torch
import numpy as np
import re
import unicodedata
from transformers import AutoTokenizer
import jiwer  # Direct jiwer - same as HuggingFace evaluate uses internally


class DanishTextNormalizer:
    """
    Text normalizer for Danish ASR evaluation.
    
    Based on OpenAI Whisper's BasicTextNormalizer, adapted for Danish.
    Key difference: Preserves Danish special characters (æ, ø, å) which are
    distinct letters in the Danish alphabet, not diacritics.
    
    Reference: https://github.com/openai/whisper/blob/main/whisper/normalizers/basic.py
    """
    
    def __call__(self, text: str) -> str:
        # Lowercase
        text = text.lower()
        
        # Remove content in brackets [like this] or <like this>
        text = re.sub(r"[<\[][^>\]]*[>\]]", "", text)
        
        # Remove content in parentheses (like this)
        text = re.sub(r"\(([^)]+?)\)", "", text)
        
        # Normalize unicode (NFKC keeps æ, ø, å intact)
        text = unicodedata.normalize("NFKC", text)
        
        # Remove punctuation and symbols, but keep letters (including æøå) and numbers
        # Category M = Mark, S = Symbol, P = Punctuation
        text = "".join(
            " " if unicodedata.category(c)[0] in "MSP" else c
            for c in text
        )
        
        # Collapse multiple spaces into single space
        text = re.sub(r"\s+", " ", text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text


# Global normalizer instance
_danish_normalizer = DanishTextNormalizer()


def normalize_danish(text: str) -> str:
    """Apply Danish ASR text normalization."""
    return _danish_normalizer(text)


# jiwer transformation using our Danish normalizer
class JiwerDanishTransform:
    """jiwer-compatible transformation for Danish text."""
    def __call__(self, sentences):
        if isinstance(sentences, str):
            return [[w for w in normalize_danish(sentences).split() if w]]
        return [[w for w in normalize_danish(s).split() if w] for s in sentences]


DANISH_NORMALIZER = JiwerDanishTransform()

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
    """Decode model outputs and labels into texts.

    Args:
        logits (torch.Tensor): Prediction logits (B, L, V).
        labels (torch.Tensor): Target label tensors (B, L).
        tokenizer (AutoTokenizer): Tokenizer for decoding indices to text.
        ignore_label (int): Label to ignore in decoding.

    Returns:
        Tuple[List[str], List[str]]: Tuple of (hypothesis texts, reference texts).
    """
    # Get predictions by taking argmax of logits
    pred_ids = torch.argmax(logits, dim=-1)  # (B, L)
    
    # Align sequence lengths
    seq_diff = pred_ids.size(1) - labels.size(1)
    if seq_diff: 
        raise ValueError(f"Prediction and label sequence lengths do not match: {pred_ids.size(1)} vs {labels.size(1)}")
        
    # Create mask for valid tokens
    valid_mask = (labels != ignore_label)
    
    hyp_texts, ref_texts = [], []
    for pred, label, mask in zip(pred_ids, labels, valid_mask):
        # Select only valid tokens
        valid_pred = pred[mask]
        valid_label = label[mask]
        
        if len(valid_pred) == 0 or len(valid_label) == 0:
            continue
            
        try:
            pred_text = tokenizer.decode(valid_pred, skip_special_tokens=True).strip()
            label_text = tokenizer.decode(valid_label, skip_special_tokens=True).strip()
            
            if pred_text and label_text:
                hyp_texts.append(pred_text)
                ref_texts.append(label_text)
        except:
            raise ValueError("Decoding failed for predictions or labels.")
            
    return hyp_texts, ref_texts


def compute_wer(hyp_texts: List[str], ref_texts: List[str], normalize: bool = True) -> float:
    """Calculate Word Error Rate (WER) from hypothesis and reference texts.
    
    Uses jiwer library - the standard WER implementation used by most ASR papers.
    Applies Danish-specific text normalization (preserves æ, ø, å).
    
    Based on OpenAI Whisper's BasicTextNormalizer for non-English languages.
    
    Args:
        hyp_texts (List[str]): List of hypothesis (predicted) texts
        ref_texts (List[str]): List of reference (ground truth) texts
        normalize (bool): Whether to apply Danish ASR normalization 
                         (lowercase, remove punctuation, preserve æøå).
                         Default True for fair comparison with published results.
        
    Returns:
        float: Word Error Rate score (0.0 = perfect, 1.0 = 100% error)
    """
    if not hyp_texts or not ref_texts:
        return 0.0
    
    # Compute WER with Danish normalization
    if normalize:
        return jiwer.wer(
            ref_texts, 
            hyp_texts,
            truth_transform=DANISH_NORMALIZER,
            hypothesis_transform=DANISH_NORMALIZER
        )
    else:
        # Raw WER without normalization
        return jiwer.wer(ref_texts, hyp_texts)