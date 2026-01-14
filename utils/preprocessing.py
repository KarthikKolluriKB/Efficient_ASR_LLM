"""
Text preprocessing utilities for ASR evaluation.

Best Practice: Apply text normalization (lowercase, punctuation removal) ONLY during 
evaluation for WER calculation, NOT during training. This preserves the model's 
ability to learn proper casing and punctuation.
"""

import re
import string
from typing import Optional
import unicodedata


# Common punctuation patterns
PUNCTUATION = string.punctuation + "¿¡«»„""''‚‹›"


def normalize_for_wer(text: str) -> str:
    """
    Normalize text for WER (Word Error Rate) calculation.
    
    Applies:
    - Lowercase
    - Remove all punctuation
    - Collapse multiple spaces
    - Strip whitespace
    
    Args:
        text: Original transcription
    
    Returns:
        Normalized text suitable for WER comparison
    """
    if not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation (but keep apostrophes for contractions)
    # Pattern: remove all punctuation except apostrophe
    text = re.sub(r"[^\w\s']", " ", text)
    
    # Remove standalone apostrophes (not in contractions)
    text = re.sub(r"(?<!\w)'|'(?!\w)", " ", text)
    
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def normalize_danish(text: str) -> str:
    """
    Normalize Danish text with language-specific rules.
    
    Handles:
    - Danish special characters (æ, ø, å)
    - Common abbreviations
    - Number formatting
    """
    if not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Keep Danish special characters
    # Remove other punctuation
    text = re.sub(r"[^\w\s'æøåÆØÅ]", " ", text)
    
    # Collapse spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def normalize_english(text: str) -> str:
    """
    Normalize English text with language-specific rules.
    """
    if not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Common replacements
    replacements = {
        "&": " and ",
        "%": " percent ",
        "@": " at ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove punctuation
    text = re.sub(r"[^\w\s']", " ", text)
    
    # Collapse spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def normalize_dutch(text: str) -> str:
    """
    Normalize Dutch text with language-specific rules.
    
    Handles Dutch-specific characters and patterns.
    """
    if not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Keep Dutch special characters (ë, ï, é, etc.)
    text = re.sub(r"[^\w\s'ëïéèêàâùûôî]", " ", text)
    
    # Collapse spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def get_normalizer(language: str):
    """
    Get the appropriate normalizer for a language.
    
    Args:
        language: Language code (da, en, nl, etc.)
    
    Returns:
        Normalization function
    """
    normalizers = {
        "da": normalize_danish,
        "danish": normalize_danish,
        "en": normalize_english,
        "english": normalize_english,
        "nl": normalize_dutch,
        "dutch": normalize_dutch,
    }
    
    return normalizers.get(language.lower(), normalize_for_wer)


def remove_special_tokens(text: str, tokens: list = None) -> str:
    """
    Remove special tokens from model output.
    
    Args:
        text: Model output text
        tokens: List of special tokens to remove (default: common LLM tokens)
    
    Returns:
        Cleaned text
    """
    if tokens is None:
        tokens = [
            "<|endoftext|>", "<|im_end|>", "<|im_start|>",
            "</s>", "<s>", "<pad>", "<eos>", "<bos>",
            "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]",
        ]
    
    for token in tokens:
        text = text.replace(token, "")
    
    return text.strip()


def preprocess_for_training(text: str, keep_case: bool = True, keep_punct: bool = True) -> str:
    """
    Preprocess text for training (minimal processing).
    
    By default, keeps casing and punctuation to preserve the model's
    ability to learn these features.
    
    Args:
        text: Original transcription
        keep_case: Keep original case (recommended: True)
        keep_punct: Keep punctuation (recommended: True)
    
    Returns:
        Preprocessed text
    """
    if not text:
        return ""
    
    # Strip whitespace
    text = text.strip()
    
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    
    if not keep_case:
        text = text.lower()
    
    if not keep_punct:
        text = re.sub(r"[^\w\s]", "", text)
    
    return text


def validate_audio_duration(
    duration: float,
    min_duration: float = 0.5,
    max_duration: float = 30.0,
) -> bool:
    """
    Validate if audio duration is within acceptable range.
    
    Args:
        duration: Audio duration in seconds
        min_duration: Minimum acceptable duration (default: 0.5s)
        max_duration: Maximum acceptable duration (default: 30s for Whisper)
    
    Returns:
        True if duration is valid, False otherwise
    """
    return min_duration <= duration <= max_duration


class ASRPreprocessor:
    """
    Unified preprocessor for ASR tasks.
    
    Usage:
        preprocessor = ASRPreprocessor(language="da")
        
        # During training (minimal processing):
        train_text = preprocessor.process_for_training(raw_text)
        
        # During evaluation (normalize for WER):
        pred_normalized = preprocessor.normalize_for_eval(prediction)
        ref_normalized = preprocessor.normalize_for_eval(reference)
        wer = compute_wer(pred_normalized, ref_normalized)
    """
    
    def __init__(
        self,
        language: str = "en",
        min_duration: float = 0.5,
        max_duration: float = 30.0,
    ):
        self.language = language.lower()
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.normalizer = get_normalizer(language)
    
    def process_for_training(self, text: str) -> str:
        """Process text for training (minimal normalization)."""
        return preprocess_for_training(text, keep_case=True, keep_punct=True)
    
    def normalize_for_eval(self, text: str) -> str:
        """Normalize text for evaluation (WER calculation)."""
        # Remove special tokens first
        text = remove_special_tokens(text)
        # Apply language-specific normalization
        return self.normalizer(text)
    
    def is_valid_sample(self, duration: float, text: str) -> bool:
        """Check if a sample is valid for training."""
        # Check duration
        if not validate_audio_duration(duration, self.min_duration, self.max_duration):
            return False
        
        # Check text
        if not text or len(text.strip()) == 0:
            return False
        
        return True
    
    def filter_samples(self, samples: list) -> list:
        """
        Filter a list of samples based on validity criteria.
        
        Args:
            samples: List of dicts with 'duration' and 'target' fields
        
        Returns:
            Filtered list of valid samples
        """
        valid = []
        skipped = {"short": 0, "long": 0, "empty_text": 0}
        
        for sample in samples:
            duration = sample.get("duration", 0)
            text = sample.get("target", "")
            
            if duration < self.min_duration:
                skipped["short"] += 1
            elif duration > self.max_duration:
                skipped["long"] += 1
            elif not text or len(text.strip()) == 0:
                skipped["empty_text"] += 1
            else:
                valid.append(sample)
        
        total_skipped = sum(skipped.values())
        if total_skipped > 0:
            print(f"[Preprocessor] Filtered {total_skipped} samples: "
                  f"{skipped['short']} too short, {skipped['long']} too long, "
                  f"{skipped['empty_text']} empty text")
        
        return valid


# Quick test
if __name__ == "__main__":
    # Test normalization
    test_cases = [
        ("Hello, World!", "hello world"),
        ("Don't stop!", "don't stop"),
        ("Det er en god dag.", "det er en god dag"),  # Danish
        ("Hoe gaat het?", "hoe gaat het"),  # Dutch
    ]
    
    print("Testing normalize_for_wer:")
    for original, expected in test_cases:
        result = normalize_for_wer(original)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{original}' -> '{result}' (expected: '{expected}')")
    
    # Test preprocessor
    print("\nTesting ASRPreprocessor:")
    prep = ASRPreprocessor(language="da", min_duration=0.5, max_duration=30.0)
    
    samples = [
        {"duration": 0.02, "target": "too short"},  # Should be filtered
        {"duration": 5.0, "target": "Valid sample"},  # Valid
        {"duration": 35.0, "target": "too long"},  # Should be filtered
        {"duration": 3.0, "target": ""},  # Should be filtered
        {"duration": 10.0, "target": "Another valid sample!"},  # Valid
    ]
    
    valid_samples = prep.filter_samples(samples)
    print(f"  Kept {len(valid_samples)}/{len(samples)} samples")
